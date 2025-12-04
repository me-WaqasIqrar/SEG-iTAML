# segformer_infer.py
import os
import argparse
from typing import Tuple, List

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from tqdm import tqdm
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

# -------------------------
# DEFAULTS (match your train config)
# -------------------------
DATA_ROOT     = r"D:/Segformer/Dataset/PIDRAY"
Experiment_Name = "Exp_01"
TEST_IMG_DIR  = os.path.join(DATA_ROOT, "test", "images")
NUM_CLASSES   = 2
CLASS_NAMES   = {0: "background", 1: "weapon"}
IMG_SIZE      = (512, 512)  # (H, W)
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where you saved checkpoints in training:
DEFAULT_CHECKPOINT = os.path.join(DATA_ROOT, Experiment_Name, "checkpoints", "segformer-weapon-best")

# -------------------------
# DATASET
# -------------------------
class InferenceImageDataset(Dataset):
    def __init__(self, image_dir: str, processor: SegformerImageProcessor, img_size: Tuple[int, int], exts=(".png", ".jpg", ".jpeg")):
        self.paths = [
            os.path.join(image_dir, f)
            for f in sorted(os.listdir(image_dir))
            if f.lower().endswith(exts)
        ]
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found in: {image_dir}")
        self.processor = processor
        self.img_size  = img_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")
        # store (H, W) exactly two ints
        orig_size_hw = (image.size[1], image.size[0])
        encoded = self.processor(image, return_tensors="pt")
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        return {
            "pixel_values": encoded["pixel_values"],
            "path": path,
            "orig_hw": orig_size_hw,
        }

# -------------------------
# UTILS
# -------------------------
def collate_infer(batch):
    # batch is a list of dicts from __getitem__
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)  # (B,3,H,W)
    paths = [b["path"] for b in batch]
    orig_hw = torch.tensor([b["orig_hw"] for b in batch], dtype=torch.long)  # (B,2)
    return {"pixel_values": pixel_values, "path": paths, "orig_hw": orig_hw}

def colorize_mask(mask_np: np.ndarray, palette: List[Tuple[int, int, int]]):
    """mask_np: (H,W) int; returns PIL Image with simple palette applied."""
    img = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    pal = []
    for c in range(256):
        if c < len(palette):
            pal.extend(list(palette[c]))
        else:
            pal.extend([0, 0, 0])  # pad
    img.putpalette(pal)
    return img

def overlay_on_image(image: Image.Image, mask_np: np.ndarray, color=(255, 0, 0), alpha=0.45):
    """Return RGBA overlay visualization."""
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_arr = np.array(overlay)
    mh, mw = mask_np.shape
    if (mh, mw) != (image.size[1], image.size[0]):
        mask_np = np.array(Image.fromarray(mask_np).resize(image.size, resample=Image.NEAREST))
    r, g, b = color
    overlay_arr[mask_np > 0] = (r, g, b, int(255 * alpha))
    return Image.alpha_composite(image.convert("RGBA"), Image.fromarray(overlay_arr, mode="RGBA"))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# -------------------------
# INFERENCE
# -------------------------
@torch.no_grad()
def run_inference(
    checkpoint_dir: str,
    image_dir: str,
    out_dir: str,
    batch_size: int = 8,
    num_workers: int = 0,
    save_overlays: bool = True,
    save_probs: bool = False,
    keep_original_size: bool = True,
    palette: List[Tuple[int,int,int]] = None
):
    """
    - checkpoint_dir: folder with model + processor saved via .save_pretrained(...)
    - image_dir: folder of images to run inference on
    - out_dir: base output directory (masks/, overlays/, probs/)
    """

    # Load processor & model
    processor = SegformerImageProcessor.from_pretrained(checkpoint_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(
        checkpoint_dir,
        num_labels=NUM_CLASSES,
        id2label=CLASS_NAMES,
        label2id={v: k for k, v in CLASS_NAMES.items()},
        ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.eval()

    # Dataset + loader
    dataset = InferenceImageDataset(image_dir=image_dir, processor=processor, img_size=IMG_SIZE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,collate_fn=collate_infer)

    # Outputs
    out_masks    = os.path.join(out_dir, "masks")
    out_overlays = os.path.join(out_dir, "overlays")
    out_probs    = os.path.join(out_dir, "probs")  # per-class probability maps (npz)
    ensure_dir(out_masks)
    if save_overlays:
        ensure_dir(out_overlays)
    if save_probs:
        ensure_dir(out_probs)

    # Simple default palette: class 0 black, class 1 red
    if palette is None:
        palette = [
            (0, 0, 0),        # background
            (255, 0, 0),      # weapon
        ]

    pbar = tqdm(loader, desc="[inference]")
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
        paths = batch["path"]

        # --- FIX: normalize orig_hw to list of exact (H, W) tuples ---
        raw_hw = batch["orig_hw"]                 # (B,2) LongTensor from collate_infer
        hw_list = [tuple(map(int, hw.tolist())) for hw in raw_hw]  # list of (H, W)
        if isinstance(raw_hw, torch.Tensor):
            # shape (B, 2) or (B, >=2)
            hw_list = [tuple(int(x) for x in row[:2]) for row in raw_hw.tolist()]
        elif isinstance(raw_hw, (list, tuple)):
            for x in raw_hw:
                if isinstance(x, torch.Tensor):
                    x = x.tolist()
                x = list(x)[:2]  # keep first two
                if len(x) != 2:
                    raise ValueError(f"Unexpected orig_hw element length: {len(x)}; value={x}")
                hw_list.append((int(x[0]), int(x[1])))
        else:
            raise TypeError(f"Unexpected type for orig_hw: {type(raw_hw)}")
        # --- /FIX ---

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, C, h, w)

        if keep_original_size:
            probs_list = []
            preds_list = []
            for i in range(logits.shape[0]):
                h, w = hw_list[i]  # (H, W)
                logit_i = logits[i:i+1]
                logit_i = F.interpolate(logit_i, size=(h, w), mode="bilinear", align_corners=False)
                prob_i = logit_i.softmax(dim=1).squeeze(0)  # (C,H,W)
                pred_i = prob_i.argmax(dim=0).cpu().numpy().astype(np.uint8)
                probs_list.append(prob_i.cpu().numpy())
                preds_list.append(pred_i)
        else:
            target_hw = (pixel_values.shape[-2], pixel_values.shape[-1])  # (H, W)
            logits_rs = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
            probs = logits_rs.softmax(dim=1).cpu().numpy()                # (B,C,H,W)
            preds = probs.argmax(axis=1).astype(np.uint8)                 # (B,H,W)
            probs_list = [probs[i] for i in range(probs.shape[0])]
            preds_list = [preds[i] for i in range(preds.shape[0])]

        # Save outputs
        for i, path in enumerate(paths):
            base = os.path.splitext(os.path.basename(path))[0]

            # Save mask (indexed PNG with palette)
            mask_np = preds_list[i]
            mask_col = colorize_mask(mask_np, palette)
            mask_col.save(os.path.join(out_masks, f"{base}.png"))

            # Optional overlay
            if save_overlays:
                img = Image.open(path).convert("RGB")
                over = overlay_on_image(img, mask_np, color=palette[1], alpha=0.45)
                over.save(os.path.join(out_overlays, f"{base}.png"))

            # Optional per-class probabilities
            if save_probs:
                np.savez_compressed(os.path.join(out_probs, f"{base}.npz"), probs=probs_list[i])

    print(f"\nDone. Masks -> {out_masks}")
    if save_overlays:
        print(f"Overlays -> {out_overlays}")
    if save_probs:
        print(f"Per-class probabilities -> {out_probs}")

# -------------------------
# CLI
# -------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="SegFormer inference for weapon segmentation")
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT,
                    help="Path to folder with model/processor saved via save_pretrained (default: best)")
    ap.add_argument("--images", type=str, default=TEST_IMG_DIR,
                    help="Folder of images to segment")
    ap.add_argument("--out", type=str, default=os.path.join(DATA_ROOT, "predictions", Experiment_Name),
                    help="Output folder (creates masks/, overlays/, probs/)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)

    # Keep compatibility with your previous flag name but make it actually work:
    # Default behavior: keep_original_size=True
    ap.set_defaults(keep_original_size=True)
    ap.add_argument("--resize_to_original", dest="keep_original_size", action="store_true",
                    help="Resize predictions back to each image's original size (default).")
    ap.add_argument("--fixed_512", dest="keep_original_size", action="store_false",
                    help="Export predictions at the processor size (e.g., 512x512).")

    ap.add_argument("--no_overlays", action="store_true", help="Disable RGBA overlay exports")
    ap.add_argument("--save_probs", action="store_true", help="Also save per-class probability maps as .npz")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    run_inference(
        checkpoint_dir=args.checkpoint,
        image_dir=args.images,
        out_dir=args.out,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_overlays=(not args.no_overlays),
        save_probs=args.save_probs,
        keep_original_size=args.keep_original_size,
    )

if __name__ == "__main__":
    main()
