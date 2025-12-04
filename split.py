import os
import shutil
import random
from tqdm import tqdm  #  Progress bar

'''
takes Path for images and masks
splits them into train and test folders 80/20
'''

# Set random seed for reproducibility
random.seed(42)

# Define paths
root_dir = "d:/Segformer/Dataset/Football_Player_Data"
image_dir = os.path.join(root_dir, "images")
mask_dir = os.path.join(root_dir, "masks")

train_image_dir = os.path.join(root_dir, "train/images")
train_mask_dir = os.path.join(root_dir, "train/masks")
test_image_dir = os.path.join(root_dir, "test/images")
test_mask_dir = os.path.join(root_dir, "test/masks")

# Create output directories if they don't exist
for path in [train_image_dir, train_mask_dir, test_image_dir, test_mask_dir]:
    os.makedirs(path, exist_ok=True)

# Get all image filenames (assuming corresponding masks share the same filenames)
images = sorted(os.listdir(image_dir))
masks = sorted(os.listdir(mask_dir))

# Ensure both lists match
assert len(images) == len(masks), "Mismatch between image and mask counts."
assert all(os.path.splitext(img)[0] == os.path.splitext(msk)[0] for img, msk in zip(images, masks)), \
    "Image and mask filenames do not match."

# Split 80/20
split_index = int(0.8 * len(images))
combined = list(zip(images, masks))
random.shuffle(combined)
train_pairs = combined[:split_index]
test_pairs = combined[split_index:]

print(f"Total pairs: {len(combined)}, Train: {len(train_pairs)}, Test: {len(test_pairs)}")
print("Starting to copy files...")

# Move files with progress bars
for img, msk in tqdm(train_pairs, desc="Copying training data", unit="pair"):
    shutil.copy(os.path.join(image_dir, img), os.path.join(train_image_dir, img))
    shutil.copy(os.path.join(mask_dir, msk), os.path.join(train_mask_dir, msk))

for img, msk in tqdm(test_pairs, desc="Copying test data", unit="pair"):
    shutil.copy(os.path.join(image_dir, img), os.path.join(test_image_dir, img))
    shutil.copy(os.path.join(mask_dir, msk), os.path.join(test_mask_dir, msk))

print(f"Split complete: {len(train_pairs)} training pairs, {len(test_pairs)} testing pairs.")
