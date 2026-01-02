import os
import pdb
import time
import copy
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#from resnet import *
from radam import *
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig


import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Sampler
from torchvision import datasets, transforms
# from imagenet import ImageNet

import collections
from utils.cutout import Cutout

class ResNet_features(nn.Module):
    def __init__(self, original_model):
        super(ResNet_features, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
class Learner():
    def __init__(self,model,args,trainloader,testloader, use_cuda):
        self.model=model
        self.best_model=model
        self.args=args
        self.title='incremental-learning' + self.args.checkpoint.split("/")[-1]
        self.trainloader=trainloader 
        self.use_cuda=use_cuda
        self.state= {key:value for key, value in self.args.__dict__.items() if not key.startswith('__') and not callable(key)} 
        self.best_acc = 0 
        self.testloader=testloader
        self.test_loss=0.0
        self.train_acc=0.0
        self.test_acc=0.0
        
        self.test_mIoU=0.0
        self.train_loss, self.train_mIoU=0.0,0.0
        self.best_mIoU = 0.0  # Initialize best mIoU for segmentation tracking
        
        meta_parameters = []
        normal_parameters = []
        for n,p in self.model.named_parameters():
            meta_parameters.append(p)
            p.requires_grad = True
            if("fc" in n):
                normal_parameters.append(p)
      
        if(self.args.optimizer=="radam"):
            self.optimizer = RAdam(meta_parameters, lr=self.args.lr, betas=(0.9, 0.999), weight_decay=0)
        elif(self.args.optimizer=="adam"):
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
        elif(self.args.optimizer=="sgd"):
            self.optimizer = optim.SGD(meta_parameters, lr=self.args.lr, momentum=0.9, weight_decay=0.001)
 

    def learn(self):
        logger = Logger(os.path.join(self.args.checkpoint, 'session_'+str(self.args.sess)+'_log.txt'), title=self.title)
        # Use mIoU headers for segmentation mode, accuracy headers for classification
        if getattr(self.args, 'segmentation', False):
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train mIoU', 'Valid mIoU', 'Best mIoU'])
        else:
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.', 'Best Acc'])
            
        for epoch in range(0, self.args.epochs):
            self.adjust_learning_rate(epoch)
            print('\nEpoch: [%d | %d] LR: %f Sess: %d' % (epoch + 1, self.args.epochs, self.state['lr'],self.args.sess))

            self.train(self.model, epoch)
            
            self.test(self.model)
        
            # append logger file
            logger.append([self.state['lr'], self.train_loss, self.test_loss, self.train_mIoU, self.test_mIoU, self.best_mIoU])

            # save model
            is_best = self.test_mIoU > self.best_mIoU
            if(is_best and epoch>self.args.epochs-10):
                self.best_model = copy.deepcopy(self.model)

            self.best_mIoU = max(self.test_mIoU, self.best_mIoU)
            if(epoch==self.args.epochs-1):
                self.save_checkpoint(self.best_model.state_dict(), True, checkpoint=self.args.savepoint, filename='session_'+str(self.args.sess)+'_model_best.pth.tar')
        self.model = copy.deepcopy(self.best_model)
        
        logger.close()
        logger.plot()
        savefig(os.path.join(self.args.checkpoint, 'log.eps'))

        print('Best mIoU:')
        print(self.best_mIoU)
    
    def train(self, model, epoch):
        model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(self.trainloader))

        # segmentation training branch with REPTILE META-UPDATE
        if getattr(self.args, 'segmentation', False):
            losses = AverageMeter()
            iou_meter = AverageMeter()

            for batch_idx, batch in enumerate(self.trainloader):
                # measure data loading time
                data_time.update(time.time() - end)

                inputs, seg_maps, targets = batch
                # targets: class labels (1-indexed for foreground classes in segmentation)
                # seg_maps: segmentation masks with 0=background, 1,2,...=foreground classes

                if self.use_cuda:
                    inputs = inputs.cuda()
                    seg_maps = seg_maps.cuda()
                    targets = targets.cuda() if isinstance(targets, torch.Tensor) else torch.tensor(targets).cuda()

                # =====================================================
                # REPTILE META-LEARNING IMPLEMENTATION
                # =====================================================
                
                # Convert targets to 0-indexed for task grouping (subtract 1 since dataloader adds 1)
                np_targets = (targets - 1).detach().cpu().numpy()  # Now 0-indexed: 0, 1, 2, ...
                
                # Storage for task-adapted parameters
                reptile_grads = {}
                num_updates = 0
                
                # Initial forward pass for metrics (before meta-updates)
                outputs_initial, _ = model(inputs)
                
                # 1️⃣ Create base model snapshot (θ⁰) - the reference weights
                model_base = copy.deepcopy(model)
                
                # 2️⃣ Loop through all seen tasks (sessions 0 to current session)
                for task_idx in range(1 + self.args.sess):
                    # Task classes in 0-indexed targets: task_idx*class_per_task to (task_idx+1)*class_per_task
                    task_class_start = task_idx * self.args.class_per_task
                    task_class_end = (task_idx + 1) * self.args.class_per_task
                    
                    # Find samples belonging to this task
                    idx = np.where(
                        (np_targets >= task_class_start) & 
                        (np_targets < task_class_end)
                    )[0]
                    
                    if len(idx) > 0:
                        # 3️⃣ Reset model to base weights (θ⁰) before each task adaptation
                        for p, q in zip(model.parameters(), model_base.parameters()):
                            p.data.copy_(q.data)
                        
                        # Get task-specific samples
                        task_inputs = inputs[idx]
                        task_seg_maps = seg_maps[idx]
                        
                        # 4️⃣ Inner-loop task adaptation (produces θᵗ)
                        # Number of adaptation steps (controlled by args.r)
                        num_inner_steps = getattr(self.args, 'r', 1)
                        
                        for kr in range(num_inner_steps):
                            task_logits, _ = model(task_inputs)
                            
                            # Compute loss for this task's segmentation
                            loss = F.cross_entropy(task_logits, task_seg_maps)
                            
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()
                        
                        # 5️⃣ Store task-adapted parameters (θᵗ)
                        for i, p in enumerate(model.parameters()):
                            if num_updates == 0:
                                reptile_grads[i] = [p.data.clone()]
                            else:
                                reptile_grads[i].append(p.data.clone())
                        num_updates += 1
                
                # 6️⃣ CORE META-UPDATE (Reptile averaging)
                # θ ← (1-α)·θ⁰ + α·mean(θᵗ)
                if num_updates > 0:
                    # Compute meta-learning rate α (decays with session for stability)
                    alpha = np.exp(-self.args.beta * ((1.0 * self.args.sess) / self.args.num_task))
                    
                    for i, (p, q) in enumerate(zip(model.parameters(), model_base.parameters())):
                        # Stack all task-adapted weights and compute mean
                        ll = torch.stack(reptile_grads[i])
                        # Reptile update: interpolate between base and mean of adapted weights
                        p.data.copy_(torch.mean(ll, 0) * alpha + (1 - alpha) * q.data)
                else:
                    # No meta-update needed (shouldn't happen, but safety fallback)
                    alpha = 1.0
                
                # =====================================================
                # END REPTILE META-UPDATE
                # =====================================================

                # Compute metrics using initial outputs (before meta-update) for logging
                preds = torch.argmax(outputs_initial, dim=1)
                final_loss = F.cross_entropy(outputs_initial, seg_maps)

                # compute per-batch mean IoU
                batch_iou = self._compute_batch_miou(
                    preds.detach().cpu().numpy(), 
                    seg_maps.detach().cpu().numpy(), 
                    num_classes=self.args.num_class + 1, 
                    ignore_index=getattr(self.args, 'ignore_index', 255)
                )

                losses.update(final_loss.item(), inputs.size(0))
                iou_meter.update(batch_iou, inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # plot progress
                bar.suffix = '({batch}/{size}) | Total: {total:} | Loss: {loss:.4f} | mIoU: {miou: .4f} | α: {alpha:.3f}'.format(
                    batch=batch_idx + 1,
                    size=len(self.trainloader),
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    miou=iou_meter.avg,
                    alpha=alpha if num_updates > 0 else 1.0
                )
                bar.next()
            bar.finish()

            self.train_loss, self.train_mIoU = losses.avg, iou_meter.avg
            
        
        # Non-segmentation training path would go here if needed
        # Currently only segmentation mode is implemented

    def test(self, model):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        bar = Bar('Processing', max=len(self.testloader))

        # switch to evaluate mode
        model.eval()

        # segmentation evaluation branch
        if getattr(self.args, 'segmentation', False):
            losses = AverageMeter()
            iou_meter = AverageMeter()

            for batch_idx, batch in enumerate(self.testloader):
                data_time.update(time.time() - end)

                inputs, seg_maps, targets = batch

                if self.use_cuda:
                    inputs = inputs.cuda()
                    seg_maps = seg_maps.cuda()
                
                with torch.no_grad():
                    logits, _ = model(inputs)
                    # outputs: logits (B, C, H, W)

                    
                    loss = F.cross_entropy(logits, seg_maps)
                    preds = torch.argmax(logits, dim=1)

                batch_iou = self._compute_batch_miou(preds.detach().cpu().numpy(), seg_maps.detach().cpu().numpy(), num_classes=self.args.num_class + 1, ignore_index=getattr(self.args, 'ignore_index', 255))

                losses.update(loss.item(), inputs.size(0))
                iou_meter.update(batch_iou, inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                bar.suffix  = '({batch}/{size})  Total: {total:} | Loss: {loss:.4f} | mIoU: {miou: .4f}'.format(
                            batch=batch_idx + 1,
                            size=len(self.testloader),
                            total=bar.elapsed_td,
                            loss=losses.avg,
                            miou=iou_meter.avg
                            )
                bar.next()
            bar.finish()

            self.test_loss, self.test_mIoU = losses.avg, iou_meter.avg

            # Save dummy acc_task structure for compatibility
            acc_task = {i: float(self.test_mIoU) for i in range(self.args.sess+1)}
            with open(self.args.savepoint + "/acc_task_test_"+str(self.args.sess)+".pickle", 'wb') as handle:
                pickle.dump(acc_task, handle, protocol=pickle.HIGHEST_PROTOCOL)

            return
    
    def _compute_batch_miou( self,preds_np, target_np, num_classes, ignore_index=255):
        """Compute mean IoU for a batch given numpy preds and targets."""
        ious = []
        for cls in range(num_classes):
            if cls == ignore_index:
                continue
            inter = 0
            union = 0
            for p, t in zip(preds_np, target_np):
                p_mask = (p == cls)
                t_mask = (t == cls)
                inter += int((p_mask & t_mask).sum())
                union += int((p_mask | t_mask).sum())
            if union == 0:
                continue
            iou = inter / union
            ious.append(inter / union)
        if len(ious) == 0:
            return 0.0
        
        return float(np.nanmean(ious))

    def _compute_per_class_iou(self, preds_np, target_np, class_ids, ignore_index=255):
        """Compute IoU for specific class IDs only (for per-task evaluation)."""
        ious = {}
        for cls in class_ids:
            if cls == ignore_index:
                continue
            inter = 0
            union = 0
            for p, t in zip(preds_np, target_np):
                p_mask = (p == cls)
                t_mask = (t == cls)
                inter += int((p_mask & t_mask).sum())
                union += int((p_mask | t_mask).sum())
            if union == 0:
                ious[cls] = None  # Class not present in this batch
            else:
                ious[cls] = inter / union
        return ious

    def _meta_test_segmentation(self, model, memory, inc_dataset):
        """
        Meta-test for segmentation with per-task fine-tuning (like classification ITAML).
        
        For each task:
        1. Create a copy of the model (meta_model)
        2. Fine-tune on that task's memory samples (META TRAINING)
        3. Evaluate on test data for that task's classes
        """
        print("\n" + "="*60)
        print("SEGMENTATION META-TEST: Per-Task Fine-tuning & Evaluation")
        print("="*60)
        
        model.eval()

        base_model = copy.deepcopy(model)
        acc_task = {}
        per_class_iou_all = {}
        
        # Get memory data
        if memory is not None:
            memory_data, memory_target = memory
            memory_data = np.array(memory_data, dtype="int32")
            memory_target = np.array(memory_target, dtype="int32")
        else:
            memory_data, memory_target = np.array([]), np.array([])
        
        for task_idx in range(self.args.sess + 1):
            print(f"\n--- Task {task_idx} ---")
            
            # Get memory indices for this task
            # Memory targets are class indices (0-indexed for classes, but in segmentation masks they're 1-indexed)
            # So task 0 has classes 0 to class_per_task-1 in memory_target
            mem_idx = np.where(
                (memory_target >= task_idx * self.args.class_per_task) & 
                (memory_target < (task_idx + 1) * self.args.class_per_task)
            )[0]
            
            # Create meta model for this task
            meta_model = copy.deepcopy(base_model)
            # Use the same learning rate as the main training or a specific meta-learning rate
            meta_lr = self.args.lr 
            meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
            
            # META TRAINING: Fine-tune on task's memory samples
            if self.args.sess != 0 and len(mem_idx) > 0:
                meta_memory_data = memory_data[mem_idx]
                # Use args.train_batch or a smaller batch size for meta-learning
                meta_batch_size = getattr(self.args, 'train_batch', 16)
                meta_loader = inc_dataset.get_custom_loader_idx(meta_memory_data, mode="train", batch_size=meta_batch_size)
                
                meta_model.train()
                print(f"  Meta-training on {len(meta_memory_data)} samples...")
                
                # Use the same optimizer type as main training if possible, or default to Adam
                if self.args.optimizer == "radam":
                    meta_optimizer = RAdam(meta_model.parameters(), lr=meta_lr, betas=(0.9, 0.999), weight_decay=0)
                elif self.args.optimizer == "sgd":
                    meta_optimizer = optim.SGD(meta_model.parameters(), lr=meta_lr, momentum=0.9, weight_decay=0.001)
                else:
                    meta_optimizer = optim.Adam(meta_model.parameters(), lr=meta_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)

                for ep in range(1):  # 1 epoch of fine-tuning
                    bar = Bar('  Meta-train', max=len(meta_loader))
                    for batch_idx, batch in enumerate(meta_loader):
                        # Segmentation data: (inputs, seg_maps, targets)
                        inputs, seg_maps, targets = batch
                        
                        if self.use_cuda:
                            inputs = inputs.cuda()
                            seg_maps = seg_maps.cuda()
                        
                        # Zero gradients BEFORE forward pass
                        meta_optimizer.zero_grad()
                        
                        # Forward pass
                        logits, _ = meta_model(inputs)
                        
                        # Compute loss only for this task's classes + background
                        # Task classes in mask: 0 (bg), task_idx*class_per_task+1 to (task_idx+1)*class_per_task
                        loss = F.cross_entropy(logits, seg_maps)
                        
                        loss.backward()
                        meta_optimizer.step()
                        
                        bar.suffix = f'Meta-Train ({batch_idx+1}/{len(meta_loader)}) Loss: {loss.item():.4f}'
                        bar.next()
                    bar.finish()
            else:
                print(f"  No meta-training (sess=0 or no memory for task)")
            
            # META TESTING: Evaluate on test data
            meta_model.eval()
            
            # Task classes in segmentation masks (1-indexed, 0 is background)
            task_class_start = task_idx * self.args.class_per_task + 1
            task_class_end = (task_idx + 1) * self.args.class_per_task + 1
            task_classes = list(range(task_class_start, task_class_end))
            
            # Collect predictions for evaluation
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                bar = Bar('  Evaluating_meta', max=len(self.testloader))
                for batch_idx, batch in enumerate(self.testloader):
                    inputs, seg_maps, targets = batch
                    
                    if self.use_cuda:
                        inputs = inputs.cuda()
                        seg_maps = seg_maps.cuda()
                    
                    logits, _ = meta_model(inputs)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.append(preds.detach().cpu().numpy())
                    all_targets.append(seg_maps.detach().cpu().numpy())
                    
                    bar.next()
                bar.finish()
            
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Compute IoU for this task's classes
            eval_classes = [0] + task_classes  # Include background
            task_ious = self._compute_per_class_iou(
                all_preds, all_targets, 
                eval_classes, 
                ignore_index=getattr(self.args, 'ignore_index', 255)
            )
            
            # Store per-class IoU
            for cls, iou in task_ious.items():
                per_class_iou_all[cls] = iou
            
            # Compute mean IoU for this task (excluding background)
            task_class_ious = [task_ious[c] for c in task_classes if task_ious.get(c) is not None]
            if len(task_class_ious) > 0:
                acc_task[task_idx] = float(np.mean(task_class_ious)) * 100
            else:
                acc_task[task_idx] = 0.0
            
            print(f"  Task {task_idx} classes {task_classes}:")
            for cls in task_classes:
                iou = task_ious.get(cls)
                if iou is not None:
                    print(f"    Class {cls}: IoU = {iou:.4f}")
                else:
                    print(f"    Class {cls}: IoU = N/A")
            print(f"  Task mIoU: {acc_task[task_idx]:.2f}%")
            
            del meta_model  # Free memory
        
        # Overall summary
        all_class_ious = [iou for cls, iou in per_class_iou_all.items() if cls != 0 and iou is not None]
        overall_miou = float(np.mean(all_class_ious)) * 100 if all_class_ious else 0.0
        
        print(f"\n{'='*60}")
        print(f"Overall mIoU (all tasks): {overall_miou:.2f}%")
        print(f"Per-task mIoU: {[f'{acc_task[k]:.2f}%' for k in sorted(acc_task.keys())]}")
        print(f"{'='*60}\n")
        
        # Save per-class IoU details
        with open(self.args.savepoint + "/per_class_iou_"+str(self.args.sess)+".pickle", 'wb') as handle:
            pickle.dump(per_class_iou_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        return acc_task

    def meta_test(self, model, memory, inc_dataset):
        
        if getattr(self.args, 'segmentation', False):
            return self._meta_test_segmentation(model, memory, inc_dataset)
        else:
            raise NotImplementedError("Meta-test for non-segmentation not implemented in this snippet.")
        
    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.sess+1)*self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1
        
        #update old memory
        if(memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task*(self.args.sess)):
                idx = np.where(targets_memory==class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))   ])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))    ])
                
                
        #add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task*(self.args.sess),self.args.class_per_task*(1+self.args.sess)):
            idx = np.where(new_targets==class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx],(mu,))   ])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx],(mu,))    ])
            
        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))

    def save_checkpoint(self, state, is_best, checkpoint, filename):
        if is_best:
            torch.save(state, os.path.join(checkpoint, filename))

    def adjust_learning_rate(self, epoch):
        if epoch in self.args.schedule:
            self.state['lr'] *= self.args.gamma
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.state['lr']

if __name__ == '__main__':
    def test_miou_computation():    
        preds = np.array([     [ [0, 1],[2, 0]   ]  , [  [1, 0],[2, 0]  ]   ])

        targets = np.array([
        [[0, 1],
        [2, 0]],

        [[1, 0],
        [2, 0]]
    ])

        num_classes = 3
        print("Predictions:\n", preds)
        print("Targets:\n", targets)
        print()
        miou = Learner._compute_batch_miou( preds, targets, num_classes)

        print("\nFinal mIoU:", miou)
