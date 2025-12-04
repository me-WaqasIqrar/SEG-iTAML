import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt
import cv2

TEST_IMG = r"Dataset/Football_Player/test/images"
TEST_MASK = r"Dataset/Football_Player/test/masks"
MODEL_PATH = r"outputs_codex//best_segformer.pt" 
OUTPUT_DIR = r"outputs_codex/inference_results"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
            "original_image": np.array(img),
            "original_mask": mask,
            "filename": os.path.basename(self.imgs[idx])
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

# Load processor and model
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b0", num_labels=3)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load trained weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Create test dataset
test_ds = SegDataset(TEST_IMG, TEST_MASK, processor)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print(f"Running inference on {min(10, len(test_ds))} samples...")
print(f"Device: {device}")
print(f"Model loaded from: {MODEL_PATH}")
print(f"Saving results to: {OUTPUT_DIR}\n")

# Run inference on 10 samples
miou_scores = []

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        if idx >= 10:
            break
        
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        original_img = batch["original_image"][0].cpu().numpy()
        original_mask = batch["original_mask"][0].cpu().numpy()
        filename = batch["filename"][0]
        
        # Get prediction
        logits = model(pixel_values).logits
        logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[1:], mode="bilinear", align_corners=False
        )
        
        pred = logits.argmax(1).cpu().numpy()[0]
        gt_mask = labels.cpu().numpy()[0]
        
        # Calculate mIoU
        score = mIoU(pred, gt_mask)
        miou_scores.append(score)
        
        # Resize prediction to match original image size
        pred_resized = np.array(Image.fromarray(pred.astype(np.uint8)).resize(
            (original_img.shape[1], original_img.shape[0]), Image.NEAREST
        ))
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image\n{filename}")
        axes[0].axis('off')
        
        # Ground truth mask
        axes[1].imshow(original_mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        # Predicted mask
        axes[2].imshow(pred_resized, cmap='gray')
        axes[2].set_title(f"Predicted Mask\nmIoU: {score:.4f}")
        axes[2].axis('off')
        
        # Overlay
        overlay = original_img.copy().astype(np.float32)
        mask_colored = np.zeros_like(original_img, dtype=np.float32)
        mask_colored[pred_resized == 1] = [0, 255, 0]  # Green for predicted class
        overlay = (overlay * 0.7 + mask_colored * 0.3).astype(np.uint8)
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        axes[3].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(OUTPUT_DIR, f"inference_{idx+1:02d}_{os.path.splitext(filename)[0]}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sample {idx+1}/10 - {filename} - mIoU: {score:.4f} - Saved to {save_path}")
        
        # Also save individual masks
        pred_mask_path = os.path.join(OUTPUT_DIR, f"pred_mask_{idx+1:02d}_{os.path.splitext(filename)[0]}.png")
        Image.fromarray((pred_resized * 255).astype(np.uint8)).save(pred_mask_path)

print(f"\n{'='*60}")
print(f"Inference complete!")
print(f"Average mIoU over 10 samples: {np.mean(miou_scores):.4f}")
print(f"Results saved to: {OUTPUT_DIR}")
print(f"{'='*60}")
