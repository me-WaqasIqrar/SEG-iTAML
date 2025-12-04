##################################
# Discard this file if not training #
##################################



"""
SegFormer Training Script for Football Player Segmentation
Trains on Football_Player dataset with train/test splits
Calculates mIoU and saves the best model based on validation mIoU
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
from utils.logger import Logger


class FootballPlayerDataset(Dataset):
    """Dataset loader for Football Player images and masks"""
    
    def __init__(self, images_dir, masks_dir, processor=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        # Get all image files
        self.images = sorted([p for p in self.images_dir.iterdir() if p.is_file() and not p.name.startswith('.')])
        self.masks = sorted([p for p in self.masks_dir.iterdir() if p.is_file() and not p.name.startswith('.')])
        
        assert len(self.images) == len(self.masks), f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks"
        assert len(self.images) > 0, "No images found in dataset"
        
        self.processor = processor
        print(f"Loaded {len(self.images)} image-mask pairs from {images_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale
        
        # Convert mask to binary: background (0) and player (1)
        mask_np = np.array(mask)
        # Assuming white pixels (>127) represent the player/object
        mask_bin = (mask_np > 127).astype(np.uint8)
        
        if self.processor is not None:
            # Use processor for resizing and normalization
            encoded = self.processor(images=image, segmentation_maps=mask_bin, return_tensors="pt")
            # Remove batch dimension
            pixel_values = encoded["pixel_values"].squeeze(0)
            labels = encoded["labels"].squeeze(0).long()
            return {"pixel_values": pixel_values, "labels": labels}
        else:
            # Fallback without processor
            image_t = torch.from_numpy(np.array(image).transpose(2, 0, 1)).float() / 255.0
            labels = torch.from_numpy(mask_bin).long()
            return {"pixel_values": image_t, "labels": labels}


def collate_fn(batch):
    """Collate function for dataloader"""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}


def compute_metrics(conf_matrix, num_classes=2):
    """Compute IoU metrics from confusion matrix"""
    ious = []
    for cls in range(num_classes):
        tp = conf_matrix[cls, cls]
        fp = conf_matrix[:, cls].sum() - tp
        fn = conf_matrix[cls, :].sum() - tp
        denom = tp + fp + fn
        iou = tp / denom if denom > 0 else 0.0
        ious.append(iou)
    
    miou = np.mean(ious)
    return miou, ious


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes=2):
    """Evaluate model and compute mIoU"""
    model.eval()
    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits  # (B, num_labels, H, W)
        
        # Upsample logits to match labels size
        if logits.shape[-2:] != labels.shape[-2:]:
            logits = torch.nn.functional.interpolate(
                logits,
                size=labels.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Get predictions
        preds = torch.argmax(logits, dim=1)
        
        # Flatten for confusion matrix
        preds_np = preds.cpu().numpy().reshape(-1)
        labels_np = labels.cpu().numpy().reshape(-1)
        
        # Filter out ignore index (-100)
        valid_mask = labels_np != -100
        preds_np = preds_np[valid_mask]
        labels_np = labels_np[valid_mask]
        
        # Update confusion matrix
        for true_label, pred_label in zip(labels_np, preds_np):
            if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
                conf_matrix[true_label, pred_label] += 1
    
    # Compute metrics
    miou, ious = compute_metrics(conf_matrix, num_classes)
    
    return miou, ious, conf_matrix


def save_checkpoint(model, processor, output_dir, checkpoint_name, metrics=None):
    """Save model checkpoint"""
    checkpoint_dir = os.path.join(output_dir, checkpoint_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save model and processor
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)
    
    # Save metrics
    if metrics:
        metrics_path = os.path.join(checkpoint_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    print(f"✓ Saved checkpoint to {checkpoint_dir}")


def train(args):
    """Main training function"""
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load processor
    print(f"Loading SegFormer model: {args.model_name}")
    processor = SegformerImageProcessor.from_pretrained(args.model_name, reduce_labels=True)
    
    # Setup dataset paths
    train_images = os.path.join(args.data_root, "train", "images")
    train_masks = os.path.join(args.data_root, "train", "masks")
    test_images = os.path.join(args.data_root, "test", "images")
    test_masks = os.path.join(args.data_root, "test", "masks")
    
    # Verify paths exist
    for path in [train_images, train_masks, test_images, test_masks]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path not found: {path}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = FootballPlayerDataset(train_images, train_masks, processor=processor)
    test_dataset = FootballPlayerDataset(test_images, test_masks, processor=processor)
    
    # Create dataloaders
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
    
    # Initialize model
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "background", 1: "player"},
        label2id={"background": 0, "player": 1},
    )
    model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training tracking
    best_miou = -1.0
    training_history = {
        "train_loss": [],
        "val_miou": [],
        "val_iou_per_class": [],
        "best_miou": 0.0,
        "best_epoch": 0
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logger
    log_path = os.path.join(args.output_dir, 'training_log.txt')
    logger = Logger(log_path, title='Football Player Segmentation')
    logger.set_names(['Epoch', 'Train_Loss', 'Val_mIoU', 'Val_IoU_BG', 'Val_IoU_Player'])
    
    # Save training config
    config = vars(args)
    config['device'] = str(device)
    config['train_samples'] = len(train_dataset)
    config['test_samples'] = len(test_dataset)
    config['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Saved training config to {config_path}")
    
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
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        
        for batch_idx, batch in enumerate(train_loop):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Forward pass
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            train_loop.set_postfix(loss=f"{avg_loss:.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        training_history["train_loss"].append(epoch_loss)
        
        # Validation phase
        print(f"\nEpoch {epoch}/{args.epochs} - Evaluating on test set...")
        miou, ious, conf_matrix = evaluate_model(model, test_loader, device, num_classes=2)
        
        training_history["val_miou"].append(float(miou))
        training_history["val_iou_per_class"].append([float(iou) for iou in ious])
        
        # Log metrics
        logger.append([epoch, epoch_loss, miou, ious[0], ious[1]])
        
        # Print results
        print(f"{'='*60}")
        print(f"Epoch {epoch}/{args.epochs} Results:")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val mIoU: {miou:.4f}")
        print(f"  Background IoU: {ious[0]:.4f}")
        print(f"  Player IoU: {ious[1]:.4f}")
        print(f"{'='*60}\n")
        
        # Save best model
        if miou > best_miou:
            best_miou = miou
            training_history["best_miou"] = float(best_miou)
            training_history["best_epoch"] = epoch
            
            metrics = {
                "epoch": epoch,
                "miou": float(miou),
                "iou_background": float(ious[0]),
                "iou_player": float(ious[1]),
                "train_loss": float(epoch_loss),
                "confusion_matrix": conf_matrix.tolist()
            }
            
            save_checkpoint(model, processor, args.output_dir, "best_model", metrics)
            print(f"🏆 New best model! mIoU: {best_miou:.4f}\n")
        
        # Save checkpoint every N epochs
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(model, processor, args.output_dir, f"checkpoint_epoch_{epoch}")
    
    # Save final model
    final_metrics = {
        "epoch": args.epochs,
        "final_miou": float(miou),
        "best_miou": float(best_miou),
        "best_epoch": training_history["best_epoch"]
    }
    save_checkpoint(model, processor, args.output_dir, "last_model", final_metrics)
    
    # Save training history
    history_path = os.path.join(args.output_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"✓ Saved training history to {history_path}")
    
    # Close logger and save plots
    logger.close()
    print(f"✓ Saved training log to {log_path}")
    
    try:
        # Generate training plots
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        epochs_range = range(1, args.epochs + 1)
        
        # Plot training loss
        ax1.plot(epochs_range, training_history['train_loss'], 'b-', linewidth=2)
        ax1.set_title('Training Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # Plot validation mIoU
        ax2.plot(epochs_range, training_history['val_miou'], 'g-', linewidth=2)
        ax2.axhline(y=best_miou, color='r', linestyle='--', label=f'Best: {best_miou:.4f}')
        ax2.set_title('Validation mIoU', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot IoU per class
        iou_bg = [x[0] for x in training_history['val_iou_per_class']]
        iou_player = [x[1] for x in training_history['val_iou_per_class']]
        ax3.plot(epochs_range, iou_bg, 'c-', linewidth=2, label='Background')
        ax3.plot(epochs_range, iou_player, 'm-', linewidth=2, label='Player')
        ax3.set_title('IoU per Class', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('IoU')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot loss vs mIoU
        ax4.plot(training_history['train_loss'], training_history['val_miou'], 'ro-', linewidth=2, markersize=4)
        ax4.set_title('Training Loss vs Validation mIoU', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Training Loss')
        ax4.set_ylabel('Validation mIoU')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved training curves to {plot_path}")
    except Exception as e:
        print(f"⚠ Could not generate plots: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Best mIoU: {best_miou:.4f} (Epoch {training_history['best_epoch']})")
    print(f"Final mIoU: {miou:.4f}")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SegFormer on Football Player Dataset")
    
    # Dataset parameters
    parser.add_argument("--data_root", type=str, 
                        default=r"D:/Segformer/Dataset/Football_Player",
                        help="Root folder containing train/ and test/ subdirectories")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, 
                        default="nvidia/mit-b0",
                        help="SegFormer model variant (nvidia/mit-b0 to mit-b5)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default=r"D:/Segformer/output",
                        help="Directory to save models and logs")
    parser.add_argument("--save_every", type=int, default=0,
                        help="Save checkpoint every N epochs (0 to disable)")
    
    # System parameters
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for data loading")
    
    args = parser.parse_args()
    
    # Run training
    train(args)
