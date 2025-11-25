"""
Prepare BCSD dataset with REAL ground truth masks for training.

The BCSD (Bank Check Security Dataset) contains:
- Images in X/ directories (check images)
- Masks in y/ directories (ground truth tamper masks)

This script organizes them into the required format for training.
"""

import shutil
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def prepare_bcsd_dataset(
    source_dir="data/BCSD",
    output_dir="data/bcsd_prepared",
    val_split=0.2,
    seed=42
):
    """
    Prepare BCSD dataset with real ground truth masks
    
    Args:
        source_dir: Path to BCSD dataset root
        output_dir: Output directory for organized data
        val_split: Validation split ratio
        seed: Random seed for reproducibility
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("PREPARING BCSD DATASET WITH REAL GROUND TRUTH MASKS")
    print("=" * 70)
    
    # Collect all image-mask pairs
    image_mask_pairs = []
    
    # Process TrainSet
    train_img_dir = source_path / "TrainSet" / "X"
    train_mask_dir = source_path / "TrainSet" / "y"
    
    if train_img_dir.exists() and train_mask_dir.exists():
        for img_file in sorted(train_img_dir.glob("*.jpeg")):
            # Corresponding mask file (X_000.jpeg -> y_000.jpeg)
            mask_file = train_mask_dir / img_file.name.replace("X_", "y_")
            
            if mask_file.exists():
                image_mask_pairs.append({
                    'image': img_file,
                    'mask': mask_file,
                    'split': 'train_orig'
                })
    
    # Process TestSet
    test_img_dir = source_path / "TestSet" / "X"
    test_mask_dir = source_path / "TestSet" / "y"
    
    if test_img_dir.exists() and test_mask_dir.exists():
        for img_file in sorted(test_img_dir.glob("*.jpeg")):
            mask_file = test_mask_dir / img_file.name.replace("X_", "y_")
            
            if mask_file.exists():
                image_mask_pairs.append({
                    'image': img_file,
                    'mask': mask_file,
                    'split': 'test_orig'
                })
    
    print(f"\nFound {len(image_mask_pairs)} image-mask pairs")
    print(f"  - From TrainSet: {sum(1 for p in image_mask_pairs if p['split'] == 'train_orig')}")
    print(f"  - From TestSet: {sum(1 for p in image_mask_pairs if p['split'] == 'test_orig')}")
    
    if len(image_mask_pairs) == 0:
        print("\n❌ No image-mask pairs found!")
        return
    
    # Split into train/val
    train_pairs, val_pairs = train_test_split(
        image_mask_pairs,
        test_size=val_split,
        random_state=seed
    )
    
    print(f"\nSplit: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # Create directories
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'masks').mkdir(parents=True, exist_ok=True)
    
    # Copy train set
    print("\nProcessing training set...")
    for i, pair in enumerate(train_pairs):
        # Copy image
        dst_img = output_path / 'train' / 'images' / f"img_{i:04d}.jpg"
        shutil.copy2(pair['image'], dst_img)
        
        # Copy mask
        dst_mask = output_path / 'train' / 'masks' / f"img_{i:04d}.jpg"
        shutil.copy2(pair['mask'], dst_mask)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(train_pairs)}")
    
    # Copy val set
    print("\nProcessing validation set...")
    for i, pair in enumerate(val_pairs):
        dst_img = output_path / 'val' / 'images' / f"img_{i:04d}.jpg"
        shutil.copy2(pair['image'], dst_img)
        
        dst_mask = output_path / 'val' / 'masks' / f"img_{i:04d}.jpg"
        shutil.copy2(pair['mask'], dst_mask)
    
    print(f"\n{'=' * 70}")
    print("✅ BCSD Dataset prepared with REAL ground truth masks!")
    print(f"{'=' * 70}")
    print(f"\nOutput directory: {output_path}")
    print(f"  Train: {len(train_pairs)} samples")
    print(f"  Val: {len(val_pairs)} samples")
    print(f"\nMasks are REAL annotated tamper regions, not synthetic!")
    print(f"{'=' * 70}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare BCSD dataset with real masks")
    parser.add_argument("--source", type=str, default="data/BCSD",
                       help="Source BCSD directory")
    parser.add_argument("--output", type=str, default="data/bcsd_prepared",
                       help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.2,
                       help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    prepare_bcsd_dataset(
        source_dir=args.source,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed
    )
    
    print("\n🚀 Ready to train with REAL data!")
    print("Run: python scripts/train_demo_quick.py  # (update data paths)")
