"""
This code is running and will train o football_Player datasets

 """

import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

TRAIN_IMG = r"Dataset/Football_Player/train/images"
TRAIN_MASK = r"Dataset/Football_Player/train/masks"
TEST_IMG = r"Dataset/Football_Player/test/images"
TEST_MASK = r"Dataset/Football_Player/test/masks"
SAVE_PATH = r"outputs_codex/best_segformer.pt"

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)])
        self.masks = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.processor = processor

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        mask = (np.array(mask) > 127).astype(np.uint8)

        encoded = self.processor(images=img, segmentation_maps=mask, return_tensors="pt")

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
        }

def mIoU(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    ious = []
    for cls in [0, 1]:
        inter = ((pred == cls) & (label == cls)).sum()
        union = ((pred == cls) | (label == cls)).sum()
        if union == 0:
            continue
        ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0

processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

train_ds = SegDataset(TRAIN_IMG, TRAIN_MASK, processor)
test_ds = SegDataset(TEST_IMG, TEST_MASK, processor)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

loss_fn = CrossEntropyLoss()
opt = AdamW(model.parameters(), lr=5e-5)

best_miou = 0

for epoch in range(1, 11):
    model.train()
    total_loss = 0

    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        opt.zero_grad()
        logits = model(pixel_values).logits
        logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[1:], mode="bilinear", align_corners=False
        )

        loss = loss_fn(logits, labels)
        loss.backward()
        opt.step()

        total_loss += loss.item()

    print(f"Epoch {epoch} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    miou_scores = []

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            logits = model(pixel_values).logits
            logits = torch.nn.functional.interpolate(
                logits, size=labels.shape[1:], mode="bilinear", align_corners=False
            )

            preds = logits.argmax(1).cpu().numpy()
            labs = labels.cpu().numpy()

            for p, l in zip(preds, labs):
                miou_scores.append(mIoU(p, l))

    epoch_miou = np.mean(miou_scores)
    print(f"Test mIoU: {epoch_miou:.4f}")

    if epoch_miou > best_miou:
        best_miou = epoch_miou
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"Saved best model → {SAVE_PATH}")

print("Training complete.")
print(f"Best mIoU: {best_miou:.4f}")
