import os
import torch
import random
import collections
import numpy as np

from PIL import Image
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import SegformerImageProcessor
from sklearn.model_selection import train_test_split
from torch.utils.data.sampler import SubsetRandomSampler

GLOBAL_PROCESSOR = SegformerImageProcessor.from_pretrained("nvidia/mit-b2")

class SegformerCustomSegDataset(Dataset):

    def __init__(self, args, root, train=True, test_size=0.2, random_state=42, transform=None, download=False, processor=None,background_label=0, ignore_index=255):
        
        super().__init__()
        self.args = args
        self.root = root
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.processor = processor if processor else GLOBAL_PROCESSOR
        self.background_label = background_label
        self.ignore_index = ignore_index
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []


        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        
        classes = ["Gun", "Knife","Hammer"]  # Manual override as Sir requested for 2 classes.
        if len(classes)==0:
            raise RuntimeError(f"No Class folder is found in {self.root}")
        
        if len(classes)!= args.num_class:
            raise RuntimeError(f"Number of classes found in {self.root} is {len(classes)}, which is not equal to num_class={args.num_class} provided in args.")
        label2id = {c: i for i, c in enumerate(classes)}

        # Scan and populate images + masks
        for class_name, label in label2id.items():
            class_dir = os.path.join(root, class_name)

            img_dir = os.path.join(class_dir, "images")
            mask_dir = os.path.join(class_dir, "masks")


            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(img_dir, img_name)
                    baseimg_name, _ = os.path.splitext(img_name)
                    mask_path=os.path.join(mask_dir,baseimg_name+ ".png")
                    
                    if not os.path.exists(mask_path):
                        raise RuntimeError(f"Mask missing for {img_path}")

                    self.image_paths.append(img_path)
                    self.mask_paths.append(mask_path)
                    self.labels.append(label)

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image/mask pairs found under {root}")

        self.targets = []
        
                # Split train/test
        indices = list(range(len(self.image_paths)))
        train_idx, test_idx = train_test_split(indices, test_size=self.test_size, random_state=self.random_state)
        
        selected_idx = train_idx if train else test_idx

        self.image_path = [self.image_paths[i] for i in selected_idx]
        self.mask_path = [self.mask_paths[i] for i in selected_idx]
        self.label      = [self.labels[i] for i in selected_idx]
        
        
        # following line will work if--> only background + class_A pixels, not class_B, class_C, etc. in masks img 
        ## for my case it satisfy condition
        # populating targets for compatibility with IncrementalDataset as in original code
        
        self.targets = list(self.label)
        
        
        assert len(self.image_path) == len(self.targets) == len(self.mask_path)



    def __len__(self):
        return len(self.image_path)
    


    def __getitem__(self, idx):

        image = Image.open(self.image_path[idx]).convert("RGB")
        # load mask and map to class ids
        mask_pil = Image.open(self.mask_path[idx]).convert("L")
        target = int(self.label[idx])

        # convert mask to numpy and map foreground (>0) to (target+1) while background stays 0
        mask_np = np.array(mask_pil)
        mapped_mask = np.where(mask_np > 0, (target + 1), 0).astype(np.uint8)

        # use processor to prepare pixel values and resize segmentation map to model size
        encoded = self.processor(images=image, segmentation_maps=mapped_mask, return_tensors="pt")
        
        # remove batch dim
        
        pixel_values = encoded["pixel_values"].squeeze(0)
        seg_map = encoded["labels"].squeeze(0).long()
        target=target+1  # because 0 is reserved for background class 
        
        # return  (inputs, seg_map, target_class)
        
        return pixel_values, seg_map, target



class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices, shuffle):
        self.indices = indices
        self.shuffle = shuffle

    def __iter__(self):
        if(self.shuffle):
            return (self.indices[i] for i in torch.randperm(len(self.indices)))
        else:
            return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
    

class IncrementalDataset:

    def __init__(
        self,
        dataset_name,
        args,
        random_order=False,
        shuffle=True,
        workers=10,
        batch_size=128,
        seed=1,
        increment=10,
        validation_split=0.
    ):
        self.dataset_name = dataset_name.lower().strip()
        datasets = _get_datasets(dataset_name)
        self.train_transforms = datasets[0].train_transforms 
        self.common_transforms = datasets[0].common_transforms
        try:
            self.meta_transforms = datasets[0].meta_transforms
        except:
            self.meta_transforms = datasets[0].train_transforms
        self.args = args
        
        self._setup_data(
            datasets,
            args.data_path,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )
        

        self._current_task = 0

        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}
    @property
    def n_tasks(self):
        return len(self.increments)
    
    def get_same_index(self, target, label, mode="train", memory=None):
        label_indices = []
        label_targets = []

        for i in range(len(target)):
            if int(target[i]) in label:
                label_indices.append(i)
                label_targets.append(target[i])
        for_memory = (label_indices.copy(),label_targets.copy())
        
#         if(self.args.overflow and not(mode=="test")):
#             memory_indices, memory_targets = memory
#             return memory_indices, memory
            
        if memory is not None:
            memory_indices, memory_targets = memory
            memory_indices2 = np.tile(memory_indices, (self.args.mu,))
            all_indices = np.concatenate([memory_indices2,label_indices])
        else:
            all_indices = label_indices
            
        return all_indices, for_memory
    
    def get_same_index_test_chunk(self, target, label, mode="test", memory=None):
        label_indices = []
        label_targets = []
        
        np_target = np.array(target, dtype="uint32")
        np_indices = np.array(list(range(len(target))), dtype="uint32")

        for t in range(len(label)//self.args.class_per_task):
            task_idx = []
            for class_id in label[t*self.args.class_per_task: (t+1)*self.args.class_per_task]:
                idx = np.where(np_target==class_id)[0]
                task_idx.extend(list(idx.ravel()))
            task_idx = np.array(task_idx, dtype="uint32")
            task_idx.ravel()
            random.shuffle(task_idx)

            label_indices.extend(list(np_indices[task_idx]))
            label_targets.extend(list(np_target[task_idx]))
            if(t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_idx)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets
    

    def new_task(self, memory=None):
        
        print(self._current_task)
        print(self.increments)
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
#         if(self.args.overflow):
#             min_class = 0
#             max_class = sum(self.increments)
        
        train_indices, for_memory = self.get_same_index(self.train_dataset.targets, list(range(min_class, max_class)), mode="train", memory=memory)
        test_indices, _ = self.get_same_index_test_chunk(self.test_dataset.targets, list(range(max_class)), mode="test")

        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(train_indices, True))
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,shuffle=False,num_workers=16, sampler=SubsetRandomSampler(test_indices, False))

        
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indices),
            "n_test_data": len(test_indices)
        }

        self._current_task += 1

        return task_info, self.train_data_loader, self.test_data_loader, self.test_data_loader, for_memory
    
     
        
    # for verification   
    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not(t in seen_classes) and (t< (task+1)*self.args.class_per_task and (t>= (task)*self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i
                
        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items(): 
            indexes.append(v)
            
        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):
     
        if(mode=="train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else: 
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(indexes, False))
    
        return data_loader
    
    
    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):
        
        if(mode=="train"):
            train_indices, for_memory = self.get_same_index(self.train_dataset.targets, class_id, mode="train", memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else: 
            test_indices, _ = self.get_same_index(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=SubsetRandomSampler(test_indices, False))
            
        return data_loader

    def _setup_data(self, datasets, path, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        
        trsf_train = transforms.Compose(self.train_transforms)
        try:
            trsf_mata = transforms.Compose(self.meta_transforms)
        except:
            trsf_mata = transforms.Compose(self.train_transforms)
            
        trsf_test = transforms.Compose(self.common_transforms)
        
        current_class_idx = 0  # When using multiple datasets
        for dataset in datasets:
            if(self.dataset_name=="imagenet"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=False, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='val', download=False, transform=trsf_test)
                
            elif(self.dataset_name=="cub200" or self.dataset_name=="cifar100" or self.dataset_name=="mnist"  or self.dataset_name=="caltech101"  or self.dataset_name=="omniglot"  or self.dataset_name=="celeb"):
                train_dataset = dataset.base_dataset(root=path, train=True, download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, train=False, download=True, transform=trsf_test)

            elif(self.dataset_name=="svhn"):
                train_dataset = dataset.base_dataset(root=path, split='train', download=True, transform=trsf_train)
                test_dataset = dataset.base_dataset(root=path, split='test', download=True, transform=trsf_test)
                train_dataset.targets = train_dataset.labels
                test_dataset.targets = test_dataset.labels
            elif (self.dataset_name=="custom"):
                
                train_dataset = dataset.base_dataset(root=path, args=self.args, train=True, transform=trsf_train, processor=GLOBAL_PROCESSOR)
                test_dataset = dataset.base_dataset(root=path, args=self.args, train=False, transform=trsf_test,  processor=GLOBAL_PROCESSOR)    

            self.train_dataset = train_dataset
            self.test_dataset = test_dataset






            order = [i for i in range(self.args.num_class)]
            if random_order:
                random.seed(seed)  
                random.shuffle(order)
            elif dataset.class_order is not None:
                order = dataset.class_order
                
            for i,t in enumerate(train_dataset.targets):
                train_dataset.targets[i] = order[t]
            for i,t in enumerate(test_dataset.targets):
                test_dataset.targets[i] = order[t]
            self.class_order.append(order)

            self.increments = [increment for _ in range(len(order) // increment)] 

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))
    
    
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
    
def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar10":
        return iCIFAR10
    elif dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "custom":
        return iCUSTOM
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iCIFAR10(DataHandler):
    base_dataset = datasets.cifar.CIFAR10
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=63 / 255),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]


class iCIFAR100(DataHandler):
    base_dataset = datasets.cifar.CIFAR100
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    
    common_transforms = [
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ]
    

class iCUSTOM(DataHandler):
    base_dataset = SegformerCustomSegDataset
    train_transforms = []
    test_transforms = []
    class_order = None


if __name__ == "__main__":
    # simple test
    from types import SimpleNamespace
    data_root = r"D:/Datasets/PIDRAY"  # point to parent with class subfolders

    # build args minimally required by IncrementalDataset
    args = SimpleNamespace()
    args.num_class = len([d for d in os.listdir(os.path.join(data_root)) if os.path.isdir(os.path.join(data_root, d))])
    args.class_per_task = 2
    args.memory = 200
    args.sess = 0
    args.test_batch = 64
    args.mu = 1
    args.dataset_name = "custom"
    args.data_path = data_root  # IncrementalDataset will call base_dataset(root=args.data_path, train=...)
    print(args.num_class)

    # instantiate
    inc = IncrementalDataset(dataset_name="custom", args=args, random_order=False, shuffle=True, workers=1, batch_size=32, seed=123, increment=args.class_per_task)

    # create first task loaders
    task_info, train_loader, test_loader, _, for_memory = inc.new_task(memory=None)
    print("task_info:", task_info)
    # iterate one batch to validate shapes
    for x, y in train_loader:
        print("batch x:", type(x), getattr(x, "shape", None), "batch y:", type(y), len(y) if hasattr(y, "__len__") else None)
        break

