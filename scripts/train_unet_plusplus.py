"""
train_unet_plusplus.py
Training script for UNet++ on the IDRBT Cheque Image Dataset.
"""

import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Import the UNet++ model
from src.models.unet_plusplus import UNetPlusPlus

class ChequeDataset(Dataset):
    """Simple dataset for loading images and binary masks.
    Expected directory structure:
    data_root/
        images/   (image files)
        masks/    (mask files, same filenames as images)
    """
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        self.image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        mask = self.mask_transform(mask)
        return image, mask

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & DataLoader
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    train_dataset = ChequeDataset(args.data_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Model
    model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            if isinstance(outputs, list):
                outputs = outputs[-1]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch}/{args.epochs}] - Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f}s")

    # Save checkpoint
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "unet_plusplus_ckpt.pth")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Model checkpoint saved to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train UNet++ on Cheque Image Dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Root directory of the dataset (contains images/ and masks/)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints", help="Directory to save model checkpoint")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (small for demo)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    train(args)
