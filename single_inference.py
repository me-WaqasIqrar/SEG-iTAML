
# Radmdomm single-image inference script from test dataset

import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from basic_net import BasicNet1
from incremental_dataloader import SegformerCustomSegDataset
from PIL import Image

# Configuration - Model args (must match training config)
class Args:
    segmentation = True
    num_class = 2
    dataset = "custom"

args = Args()

# Configuration  
MODEL_PATH = r"D:/SYNC-SEG_ITAML/models/PIDRAY/EXP3/session_1_model_best.pth.tar"
DATA_PATH = r"D:/Datasets/PIDRAY"  # Root path to dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create model architecture and load state_dict
model = BasicNet1(args, device=DEVICE)
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

# Load test dataset using SegformerCustomSegDataset
test_dataset = SegformerCustomSegDataset(
    args=args,
    root=DATA_PATH,
    train=False,  # Use test set
    test_size=0.2,
    random_state=42
)

# Get a random sample from test dataset
random_idx = random.randint(0, len(test_dataset) - 1)
pixel_values, seg_map, target = test_dataset[random_idx]

# Prepare input
image_tensor = pixel_values.unsqueeze(0).to(DEVICE)

# Run inference
with torch.no_grad():
    output, _ = model(image_tensor)  # model returns (logits, features)
    predicted_label = torch.argmax(output, dim=1).squeeze().cpu().numpy()

# Get original image for visualization
original_image_path = test_dataset.image_path[random_idx]
image_pil = Image.open(original_image_path).convert("RGB")
image_rgb = np.array(image_pil)

# Get ground truth segmentation mask and resize to match original image
gt_seg_map = seg_map.cpu().numpy()
gt_seg_map_tensor = torch.from_numpy(gt_seg_map).unsqueeze(0).unsqueeze(0).float()
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
gt_seg_map_resized = resize(gt_seg_map_tensor, image_rgb.shape[:2], InterpolationMode.NEAREST)
gt_seg_map = gt_seg_map_resized.squeeze().numpy().astype(np.uint8)

# Resize predicted label to match original image size
predicted_label_tensor = torch.from_numpy(predicted_label).unsqueeze(0).unsqueeze(0).float()
predicted_label_resized = resize(predicted_label_tensor, image_rgb.shape[:2], InterpolationMode.NEAREST)
predicted_label = predicted_label_resized.squeeze().numpy().astype(np.uint8)

# Create overlay
overlay = image_rgb.copy()
mask = predicted_label > 0
green_color = np.array([0, 255, 0], dtype=np.uint8)
overlay[mask] = (overlay[mask] * 0.7 + green_color * 0.3).astype(np.uint8)

# Display results
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(image_rgb)
axes[0].set_title(f"Original Image (Test Sample {random_idx})\nClass: {target}")
axes[0].axis("off")

axes[1].imshow(gt_seg_map, cmap="gray")
axes[1].set_title("Ground Truth Label")
axes[1].axis("off")

axes[2].imshow(predicted_label, cmap="gray")
axes[2].set_title("Predicted Label")
axes[2].axis("off")

axes[3].imshow(overlay)
axes[3].set_title("Overlay (Prediction)")
axes[3].axis("off")

plt.tight_layout()
plt.show()

print(f"\nInference completed on test sample {random_idx}")
print(f"Image path: {original_image_path}")
print(f"Ground truth class: {target}")
print(f"Image shape: {image_rgb.shape}")
print(f"Ground truth mask shape: {gt_seg_map.shape}")
print(f"Predicted mask shape: {predicted_label.shape}")
print(f"Unique ground truth values: {np.unique(gt_seg_map)}")
print(f"Unique predicted values: {np.unique(predicted_label)}")