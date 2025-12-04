"""
Author Waqas Iqrar
This file will take PIDRAY dataset and will train Segformer model for x number of classes segmentation
Working Fine as of DEC 2025
"""


import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image 
from tqdm import tqdm
GLOBAL_PROCESSOR = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
class SegformerCustomSegDataset(Dataset):

    def __init__(self, root, train=True, test_size=0.2, random_state=42, processor=None):
        
        super().__init__()
        self.root = root
        self.train = train
        self.test_size = test_size
        self.random_state = random_state
        self.processor = processor if processor else GLOBAL_PROCESSOR
        
        self.image_paths = []
        self.mask_paths = []
        self.labels = []


        classes = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
        if len(classes)==0:
            raise RuntimeError(f"No Class folder is found in {self.root} ")
        classes = classes[:2]  # consider only first 2 classes for binary segmentation


        label2id = {c: i for i, c in enumerate(classes)}
        self.seeClassesvar=label2id
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
        self.targets = self.label
        self.targets = list(self.targets)
        
        
        assert len(self.image_path) == len(self.targets) == len(self.mask_path)

    def seeClasses(self):
        return self.seeClassesvar

    def __len__(self):
        return len(self.image_path)
    


    def __getitem__(self, idx):

        image = Image.open(self.image_path[idx]).convert("RGB")
        mask_pil = Image.open(self.mask_path[idx]).convert("L")
        label = int(self.label[idx ])
        
        # convert mask to numpy and map foreground (>0) to (label+1) while background stays 0
        mask_np = np.array(mask_pil)
        mapped_mask = np.where(mask_np > 0, (label + 1), 0).astype(np.uint8)

        # use processor to prepare pixel values and resize segmentation map to model size
        encoded = self.processor(images=image, segmentation_maps=mapped_mask, return_tensors="pt")
        # remove batch dim
        pixel_values = encoded["pixel_values"].squeeze(0)
        seg_map = encoded["labels"].squeeze(0).long()
 
        return {"pixel_values": pixel_values, "seg_map": seg_map, "label": label}

def mIoU(pred,label,num_classes):
    pred = pred.flatten()
    label = label.flatten()
    ious = []
    for cls in range(num_classes):
        inter = ((pred == cls) & (label == cls)).sum()
        union = ((pred == cls) | (label == cls)).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0
def main():
    #########################
    
    root_dir = r"D:/Datasets/PIDRAY"
    num_epochs = 20
    batch_size = 8
    learning_rate = 0.0001
    
    loss_fn = CrossEntropyLoss
    save_path = r"output/mysegformer/best_segformer_model.pt"

    ###########################
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    model=SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=3)  # 2 classes + background
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_dataset = SegformerCustomSegDataset(root=root_dir, train=True)
    test_dataset = SegformerCustomSegDataset(root=root_dir, train=False)
    print("-"*30)
    print(f"Loaded Train Examples: {len(train_dataset)}")
    print(f"Loaded Test Examples: {len(test_dataset)}")
    print(f"Training for Classes: {train_dataset.seeClasses()  }")
    print("-"*30)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    best_miou = 0.0
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss=0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Training] "):
            pixel_values=batch["pixel_values"].to(device)
            seg_map=batch["seg_map"].to(device)

            optimizer.zero_grad()
            outputs=model(pixel_values=pixel_values, labels=seg_map)
            loss=outputs.loss
            logits=outputs.logits
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch} - Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        miou_scores = []
        with torch.no_grad():
            for batch in tqdm(test_loader,desc=f"Epoch {epoch}/{num_epochs} [Testing]"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["seg_map"].to(device)

                logits = model(pixel_values).logits
                logits = torch.nn.functional.interpolate(
                    logits, size=labels.shape[1:], mode="bilinear", align_corners=False
                )

                pred = logits.argmax(1).cpu().numpy()
                gt_mask = labels.cpu().numpy()

                # Calculate mIoU
                score = mIoU(pred, gt_mask, num_classes=3)
                miou_scores.append(score)
        
        epoch_miou = np.mean(miou_scores)
        print(f"Test mIoU: {epoch_miou:.4f}")

        if epoch_miou > best_miou:
            best_miou = epoch_miou
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model with mIoU: {best_miou:.4f}")

if __name__ == "__main__":
    main()