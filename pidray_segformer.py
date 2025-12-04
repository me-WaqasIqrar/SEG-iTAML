"""
SegFormer Training Script - Multi-class (12 classes + background)
Dataset structure:
data_root/
  <class_name_1>/
    Images/
    masks/
  <class_name_2>/
    Images/
    masks/
  ...
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.logger import Logger  # keep your existing Logger
import warnings
warnings.filterwarnings("ignore")


class MultiClassDataset(Dataset):
    
    def __init__(self, samples, processor=None):
        """
        samples: list of tuples (image_path, mask_path, class_id)
        processor: SegformerImageProcessor (optional)
        """
        self.samples = samples
        self.processor = processor
        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path, class_id = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale

        mask_np = np.array(mask)
        # foreground pixels (>127) become this sample's class id, background stays 0
        label_mask = np.zeros_like(mask_np, dtype=np.uint8)
        label_mask[mask_np > 127] = int(class_id)  # class_id should be 1..N

        if self.processor is not None:
            encoded = self.processor(images=image, segmentation_maps=label_mask, return_tensors="pt")
            pixel_values = encoded["pixel_values"].squeeze(0)
            labels = encoded["labels"].squeeze(0).long()
            return {"pixel_values": pixel_values, "labels": labels}
        else:
            image_t = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            labels = torch.from_numpy(label_mask).long()
            return {"pixel_values": image_t, "labels": labels}


def build_samples_from_root(data_root, expected_num_classes=None):
    """
    Scan data_root for class folders and build a list of samples.
    Returns:
      class_names: ordered list of class folder names
      samples: list of (image_path, mask_path, class_id)
    class_id mapping: background=0, first class in class_names -> 1, ...
    """
    data_root = Path(data_root)
    class_dirs = sorted([p for p in data_root.iterdir() if p.is_dir() and not p.name.startswith('.')])
    if expected_num_classes is not None and len(class_dirs) != expected_num_classes:
        print(f"Warning: expected {expected_num_classes} class folders but found {len(class_dirs)}")

    class_names = [p.name for p in class_dirs]
    #select first 2 classes
    class_names=class_names[:2]
    class_dirs=class_dirs[:2]
    samples = []
    for idx, class_dir in enumerate(class_dirs, start=1):  # class IDs start at 1
        images_dir = class_dir / "Images"
        masks_dir = class_dir / "masks"
        if not images_dir.exists() or not masks_dir.exists():
            raise FileNotFoundError(f"Missing Images/ or masks/ in {class_dir}")

        images = sorted([p for p in images_dir.iterdir() if p.is_file() and not p.name.startswith('.')])
        masks = sorted([p for p in masks_dir.iterdir() if p.is_file() and not p.name.startswith('.')])
        # Try to pair by filename (best-effort)
        map_masks = {p.name: p for p in masks}
        for img in images:
            mask = map_masks.get(img.name)
            if mask is None:
                # fallback: try same stem + common extensions
                candidates = [m for m in masks if m.stem == img.stem]
                if len(candidates) > 0:
                    mask = candidates[0]
                else:
                    raise FileNotFoundError(f"No mask found for image {img} in class {class_dir}")
            samples.append((str(img), str(mask), idx))

    return class_names, samples


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(conf_matrix):
    """
    conf_matrix shape: (num_labels, num_labels) where rows=true, cols=pred
    returns: miou (float), ious_list (per class)
    """
    num_labels = conf_matrix.shape[0]
    ious = []
    for cls in range(num_labels):
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp) / float(denom) if denom > 0 else 0.0
        ious.append(iou)
    miou = float(np.mean(ious)) if len(ious) > 0 else 0.0
    return miou, ious


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_labels):
    model.eval()
    conf_matrix = np.zeros((num_labels, num_labels), dtype=np.int64)

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, num_labels, H, W)

        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )

        preds = torch.argmax(logits, dim=1)

        preds_np = preds.cpu().numpy().reshape(-1)
        labels_np = labels.cpu().numpy().reshape(-1)

        # If there are any ignore indexes in labels (e.g. -100), filter them
        valid_mask = labels_np != -100
        preds_np = preds_np[valid_mask]
        labels_np = labels_np[valid_mask]

        for t, p in zip(labels_np, preds_np):
            if 0 <= t < num_labels and 0 <= p < num_labels:
                conf_matrix[int(t), int(p)] += 1

    miou, ious = compute_metrics(conf_matrix)
    return miou, ious, conf_matrix


def save_checkpoint(model, processor, output_dir, checkpoint_name, metrics=None):
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    if metrics:
        with open(os.path.join(checkpoint_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
    print(f"✓ Saved checkpoint to {checkpoint_dir}")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build class list and all samples from data_root
    class_names, samples = build_samples_from_root(args.data_root, expected_num_classes=args.expected_num_classes)
    num_classes = len(class_names)  # number of object classes (12 expected)
    print(f"Detected classes ({num_classes}): {class_names}")

    # train/test split (stratify by class id)
    img_paths = [s[0] for s in samples]
    mask_paths = [s[1] for s in samples]
    class_ids = [s[2] for s in samples]

    train_idx, test_idx = train_test_split(
        range(len(samples)),
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=class_ids
    )

    train_samples = [samples[i] for i in train_idx]
    test_samples = [samples[i] for i in test_idx]

    # Load processor
    print(f"Loading processor: {args.model_name}")
    # reduce_labels must be False because we have many labels
    processor = SegformerImageProcessor.from_pretrained(args.model_name, reduce_labels=False)

    # Datasets
    train_dataset = MultiClassDataset(train_samples, processor=processor)
    test_dataset = MultiClassDataset(test_samples, processor=processor)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # num_labels for segmentation model includes background
    num_labels = num_classes + 1  # background + object classes
    id2label = {i: ("background" if i == 0 else class_names[i - 1]) for i in range(num_labels)}
    label2id = {v: k for k, v in id2label.items()}

    print(f"Initializing model with {num_labels} labels (background + {num_classes} classes)")
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup training bookkeeping
    best_miou = -1.0
    training_history = {"train_loss": [], "val_miou": [], "val_iou_per_class": [], "best_miou": 0.0, "best_epoch": 0}

    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare logger names: dynamically make columns for each class IoU (including background)
    iou_col_names = [f"IoU_{id2label[i]}" for i in range(num_labels)]
    logger_names = ["Epoch", "Train_Loss", "Val_mIoU"] + iou_col_names
    log_path = os.path.join(args.output_dir, "training_log.txt")
    logger = Logger(log_path, title="Segformer Multi-class Segmentation")
    logger.set_names(logger_names)

    # Save training config
    config = vars(args).copy()
    config.update({
        "device": str(device),
        "num_classes": num_classes,
        "num_labels": num_labels,
        "class_names": class_names,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    with open(os.path.join(args.output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"✓ Saved training config")

    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for batch_idx, batch in enumerate(train_loop):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            train_loop.set_postfix(loss=f"{avg_loss:.4f}")

        epoch_loss = running_loss / len(train_loader)
        training_history["train_loss"].append(epoch_loss)

        print(f"\nEpoch {epoch}/{args.epochs} - Evaluating on test set...")
        miou, ious, conf_matrix = evaluate_model(model, test_loader, device, num_labels=num_labels)

        training_history["val_miou"].append(float(miou))
        training_history["val_iou_per_class"].append([float(x) for x in ious])

        # Prepare logger row
        log_row = [epoch, epoch_loss, miou] + ious
        # logger expects length matching set_names
        logger.append(log_row)

        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} Results:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val mIoU: {miou:.4f}")
        for i, cname in id2label.items():
            print(f"  IoU {cname}: {ious[i]:.4f}")
        print(f"{'='*60}\n")

        if miou > best_miou:
            best_miou = miou
            training_history["best_miou"] = float(best_miou)
            training_history["best_epoch"] = epoch

            metrics = {
                "epoch": epoch,
                "miou": float(miou),
                "iou_per_class": {id2label[i]: float(ious[i]) for i in range(num_labels)},
                "train_loss": float(epoch_loss),
                "confusion_matrix": conf_matrix.tolist()
            }
            save_checkpoint(model, processor, args.output_dir, "best_model", metrics)
            print(f"🏆 New best model! mIoU: {best_miou:.4f}\n")

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, processor, args.output_dir, f"checkpoint_epoch_{epoch}")

    # final save
    final_metrics = {"epoch": args.epochs, "final_miou": float(miou), "best_miou": float(best_miou), "best_epoch": training_history["best_epoch"]}
    save_checkpoint(model, processor, args.output_dir, "last_model", final_metrics)

    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")

    logger.close()
    print(f"✓ Saved training log to {log_path}")

    # Optional: plotting (kept simple)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, args.epochs + 1)

        ax1.plot(epochs_range, training_history['train_loss'], linewidth=2)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs_range, training_history['val_miou'], linewidth=2)
        ax2.set_title('Validation mIoU')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best mIoU: {best_miou:.4f} (Epoch {training_history['best_epoch']})")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer on multi-class dataset")

    parser.add_argument("--data_root", type=str, default="D:/Datasets/PIDRAY",
                        help="Root folder containing class subfolders (each with Images/ and masks/)")
    parser.add_argument("--expected_num_classes", type=int, default=12,
                        help="Expected number of object classes (for sanity check)")

    parser.add_argument("--model_name", type=str, default="nvidia/mit-b0",
                        help="SegFormer model variant (nvidia/mit-b0 to mit-b5)")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--output_dir", type=str, default="./output/Pidray")
    parser.add_argument("--save_every", type=int, default=0)

    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()
    train(args)
