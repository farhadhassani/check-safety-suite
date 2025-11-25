"""
Prepare training data for ImprovedUNet

This script organizes the existing check dataset into the required format:
    data/train/
        images/  (check images)
        masks/   (binary tamper masks)
    data/val/
        images/
        masks/

For demonstration purposes, we'll create synthetic masks from the existing data.
In production, you would have real ground truth masks.
"""

import os
import json
import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def create_synthetic_mask(image_path, is_tampered):
    """
    Create a synthetic tamper mask for demonstration.
    In production, replace this with real ground truth.
    
    For tampered checks: Create random regions
    For authentic checks: All zeros
    """
    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if is_tampered:
        # Create 1-3 random tamper regions
        num_regions = np.random.randint(1, 4)
        for _ in range(num_regions):
            # Random position and size
            cx = np.random.randint(w//4, 3*w//4)
            cy = np.random.randint(h//4, 3*h//4)
            radius_x = np.random.randint(w//10, w//5)
            radius_y = np.random.randint(h//10, h//5)
            
            # Draw ellipse as tamper region
            cv2.ellipse(
                mask,
                (cx, cy),
                (radius_x, radius_y),
                angle=np.random.randint(0, 180),
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1
            )
    
    return mask


def prepare_dataset(
    source_dir="data/idrbt_tamper",
    output_dir="data/prepared",
    val_split=0.2,
    seed=42
):
    """Prepare dataset for training"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Load labels
    labels_path = source_path / "labels.json"
    
    if not labels_path.exists():
        print(f"⚠️  Labels file not found: {labels_path}")
        print("Creating minimal dataset from available images...")
        
        # Get all images
        image_files = list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg"))
        
        # Create synthetic labels (assume first half tampered, second half authentic)
        labels = []
        for i, img_path in enumerate(image_files):
            labels.append({
                "path": str(img_path),
                "label": 1 if i < len(image_files) // 2 else 0
            })
    else:
        with open(labels_path, 'r') as f:
            labels = json.load(f)
    
    print(f"Found {len(labels)} samples")
    
    # Split into train/val
    train_labels, val_labels = train_test_split(
        labels,
        test_size=val_split,
        random_state=seed,
        stratify=[l['label'] for l in labels]
    )
    
    print(f"Train: {len(train_labels)}, Val: {len(val_labels)}")
    
    # Create directories
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Process train set
    print("\nProcessing training set...")
    for i, item in enumerate(train_labels):
        src_path = Path(item['path'])
        
        if not src_path.exists():
            print(f"⚠️  Skipping missing file: {src_path}")
            continue
        
        # Copy image
        dst_img = output_path / 'train' / 'images' / f"img_{i:04d}.jpg"
        shutil.copy2(src_path, dst_img)
        
        # Create mask
        mask = create_synthetic_mask(src_path, item['label'] == 1)
        dst_mask = output_path / 'train' / 'masks' / f"img_{i:04d}.jpg"
        cv2.imwrite(str(dst_mask), mask)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(train_labels)}")
    
    # Process val set
    print("\nProcessing validation set...")
    for i, item in enumerate(val_labels):
        src_path = Path(item['path'])
        
        if not src_path.exists():
            print(f"⚠️  Skipping missing file: {src_path}")
            continue
        
        # Copy image
        dst_img = output_path / 'val' / 'images' / f"img_{i:04d}.jpg"
        shutil.copy2(src_path, dst_img)
        
        # Create mask
        mask = create_synthetic_mask(src_path, item['label'] == 1)
        dst_mask = output_path / 'val' / 'masks' / f"img_{i:04d}.jpg"
        cv2.imwrite(str(dst_mask), mask)
    
    print(f"\n✅ Dataset prepared in: {output_path}")
    print(f"   Train: {len(train_labels)} samples")
    print(f"   Val: {len(val_labels)} samples")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--source", type=str, default="data/idrbt_tamper",
                       help="Source directory with images and labels.json")
    parser.add_argument("--output", type=str, default="data/prepared",
                       help="Output directory for organized data")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    prepare_dataset(
        source_dir=args.source,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print("\n🚀 Ready to train!")
    print("Run: python scripts/train_improved_unet.py --train-dir data/prepared/train --val-dir data/prepared/val --pretrained --use-attention")
