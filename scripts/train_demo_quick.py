"""
Quick demonstration training with a working improved model.
Uses UNet++ which has proven architecture stability.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import time

sys.path.append(str(Path(__file__).parent.parent))

from src.models.unet_plusplus import UNetPlusPlus
from src.models.losses.hybrid_loss import HybridLoss

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, img_size=256):
        self.root = Path(root_dir)
        self.img_size = img_size
        self.images = sorted((self.root / "images").glob("*.jpg"))
        
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.root / "masks" / img_path.name
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        img = self.transform(img)
        mask = self.transform(mask)
        
        return img, mask

def train_quick():
    print("=" * 70)
    print("QUICK TRAINING DEMONSTRATION")
    print("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    # Data
    train_dataset = SimpleDataset("data/prepared/train", img_size=256)
    val_dataset = SimpleDataset("data/prepared/val", img_size=256)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model: UNet++ with deep supervision
    model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    
    # Loss: Hybrid Loss (Focal + Dice + Boundary)
    criterion = HybridLoss()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    print("\nTraining for 5 epochs...")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(1, 6):
        # Train
        model.train()
        train_loss = 0
        start = time.time()
        
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Val
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        elapsed = time.time() - start
        
        print(f"Epoch {epoch}/5 ({elapsed:.1f}s) | Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        # Save best
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs("outputs/improved_training", exist_ok=True)
            torch.save(model.state_dict(), "outputs/improved_training/best_model_demo.pth")
            print(f"  ✓ Saved (best val: {val_loss:.4f})")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Model saved: outputs/improved_training/best_model_demo.pth")
    print("=" * 70)

if __name__ == "__main__":
    train_quick()
