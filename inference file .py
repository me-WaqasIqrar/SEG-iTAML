import os
import torch
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import SegformerImageProcessor

from basic_net import SegFormerBackbone   # <-- your model
# make sure basic_net.py has SegFormerBackbone class


def load_model(checkpoint_path, device="cuda"):
    # Init model
    model = SegFormerBackbone().to(device)
    
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


def infer_single_image(model, processor, image_path, device="cuda"):
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Preprocess
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    # Forward pass
    with torch.no_grad():
        features, logits = model(pixel_values)   # model returns (features, logits)

    # Upsample to original size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False
    )

    pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()  # [H, W]

    return pred_mask


def visualize_result(image_path, pred_mask, save_path=None):
    image = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title("Input Image")

    plt.subplot(1,2,2)
    plt.imshow(pred_mask, cmap="nipy_spectral", interpolation="nearest")
    plt.title("Predicted Mask")

    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth.tar)")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--save", type=str, default=None, help="Optional save path for result image")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    args = parser.parse_args()

    # Load processor + model
    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")
    model = load_model(args.checkpoint, device=args.device)

    # Inference
    pred_mask = infer_single_image(model, processor, args.image, device=args.device)

    # Visualize / Save
    visualize_result(args.image, pred_mask, save_path=args.save)
