"""
Training script for ImprovedUNet with advanced features
Implements recommendations from the expert team review.

Features:
- ImprovedUNet (ResNet34 + ASPP + CBAM)
- HybridLoss (Focal + Dice + Boundary)
- Learning rate scheduling
- Early stopping
- TensorBoard logging
- Mixed precision training (FP16)
"""

import os
import sys
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.improved_unet import ImprovedUNet
from src.models.losses.hybrid_loss import HybridLoss


class ChequeDataset(Dataset):
    """
    Dataset for loading check images and tamper masks
    
    Expected directory structure:
        data_root/
            images/   (check images)
            masks/    (binary tamper masks)
    """
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        
        self.image_files = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ])
        
        self.transform = transform
        self.mask_transform = mask_transform or transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
        
        return image, mask


def get_transforms(img_size=512, augment=True):
    """Get image and mask transforms"""
    if augment:
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ])
    else:
        img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    return img_transform, mask_transform


def calculate_metrics(pred, target, threshold=0.5):
    """Calculate IoU, Dice, Precision, Recall"""
    pred = (torch.sigmoid(pred) > threshold).float()
    target = (target > 0.5).float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    
    # Precision & Recall
    tp = intersection
    fp = pred.sum() - tp
    fn = target.sum() - tp
    
    precision = (tp + 1e-6) / (tp + fp + 1e-6)
    recall = (tp + 1e-6) / (tp + fn + 1e-6)
    
    return {
        'iou': iou.item(),
        'dice': dice.item(),
        'precision': precision.item(),
        'recall': recall.item()
    }


def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate metrics
        batch_metrics = calculate_metrics(outputs.detach(), masks.detach())
        for k in metrics:
            metrics[k] += batch_metrics[k]
    
    # Average
    n = len(loader)
    total_loss /= n
    for k in metrics:
        metrics[k] /= n
    
    return total_loss, metrics


def validate(model, loader, criterion, device):
    """Validation loop"""
    model.eval()
    total_loss = 0
    metrics = {'iou': 0, 'dice': 0, 'precision': 0, 'recall': 0}
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            
            batch_metrics = calculate_metrics(outputs, masks)
            for k in metrics:
                metrics[k] += batch_metrics[k]
    
    n = len(loader)
    total_loss /= n
    for k in metrics:
        metrics[k] /= n
    
    return total_loss, metrics


def train(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Datasets
    img_transform, mask_transform = get_transforms(
        img_size=args.img_size,
        augment=True
    )
    
    train_dataset = ChequeDataset(
        args.train_dir,
        transform=img_transform,
        mask_transform=mask_transform
    )
    
    val_dataset = ChequeDataset(
        args.val_dir,
        transform=img_transform,
        mask_transform=mask_transform
    ) if args.val_dir else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    ) if val_dataset else None
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Val samples: {len(val_dataset)}")
    
    # Model
    model = ImprovedUNet(
        pretrained=args.pretrained,
        use_attention=args.use_attention
    ).to(device)
    
    total_params, trainable_params = model.get_num_params()
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss & Optimizer
    criterion = HybridLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and torch.cuda.is_available() else None
    
    # TensorBoard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 70)
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        
        # Validate
        if val_loader:
            val_loss, val_metrics = validate(
                model, val_loader, criterion, device
            )
        else:
            val_loss, val_metrics = train_loss, train_metrics
        
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch [{epoch}/{args.epochs}] ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | IoU: {train_metrics['iou']:.3f} | Dice: {train_metrics['dice']:.3f}")
        if val_loader:
            print(f"  Val Loss: {val_loss:.4f} | IoU: {val_metrics['iou']:.3f} | Dice: {val_metrics['dice']:.3f}")
        
        # TensorBoard logging
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        for k in train_metrics:
            writer.add_scalar(f'Metrics/train_{k}', train_metrics[k], epoch)
            writer.add_scalar(f'Metrics/val_{k}', val_metrics[k], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            checkpoint_path = Path(args.output_dir) / "best_model.pth"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered (patience={args.patience})")
            break
        
        print()
    
    print("=" * 70)
    print(f"Training completed! Best val loss: {best_val_loss:.4f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ImprovedUNet")
    
    # Data
    parser.add_argument("--train-dir", type=str, required=True,
                       help="Training data directory (contains images/ and masks/)")
    parser.add_argument("--val-dir", type=str, default=None,
                       help="Validation data directory (optional)")
    
    # Model
    parser.add_argument("--pretrained", action="store_true",
                       help="Use ImageNet pre-trained ResNet34")
    parser.add_argument("--use-attention", action="store_true",
                       help="Use CBAM attention modules")
    
    # Training
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--img-size", type=int, default=512,
                       help="Image size (images resized to img_size x img_size)")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                       help="Weight decay")
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers")
    parser.add_argument("--use-amp", action="store_true",
                       help="Use automatic mixed precision (FP16)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="outputs/improved_training",
                       help="Output directory for checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/improved_unet",
                       help="TensorBoard log directory")
    
    args = parser.parse_args()
    
    train(args)
