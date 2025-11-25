"""
calibrate_mc_dropout.py
Run calibration of MC Dropout model on validation set to compute ECE.
"""
import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.models.unet import UNet
from src.models.uncertainty.mc_dropout import MCDropoutWrapper, calibrate_uncertainty
from PIL import Image
import numpy as np

class ChequeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root = Path(root_dir)
        # Handle both structured (images/) and flat directories
        if (self.root / "images").exists():
            self.images_dir = self.root / "images"
            self.masks_dir = self.root / "masks"
        else:
            self.images_dir = self.root
            self.masks_dir = None
            
        self.image_files = sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
        self.transform = transform
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask if exists, else create dummy
        if self.masks_dir and (self.masks_dir / img_path.name).exists():
            mask = Image.open(self.masks_dir / img_path.name).convert("L")
        else:
            # Create dummy mask (all zeros/authentic)
            w, h = image.size
            mask = Image.fromarray(np.zeros((h, w), dtype=np.uint8))
            
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        mask = self.mask_transform(mask)
        return image, mask

def main():
    parser = argparse.ArgumentParser(description="Calibrate MC Dropout on validation set")
    parser.add_argument("--data-dir", type=str, default="data/idrbt_tamper", help="Dataset root")
    parser.add_argument("--ckpt", type=str, default="outputs/training/best_model.pth", help="Path to UNet checkpoint")
    parser.add_argument("--val-split", type=float, default=0.2, help="Fraction of data for validation")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for calibration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    
    try:
        full_dataset = ChequeDataset(args.data_dir, transform=transform)
        if len(full_dataset) == 0:
            raise ValueError("No images found")
            
        val_len = int(len(full_dataset) * args.val_split)
        train_len = len(full_dataset) - val_len
        
        # Handle small datasets
        if val_len == 0 and len(full_dataset) > 0:
            val_dataset = full_dataset
        else:
            _, val_dataset = random_split(full_dataset, [train_len, val_len])
            
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Load base model
        base_model = UNet(n_channels=3, n_classes=1).to(device)
        if Path(args.ckpt).exists():
            base_model.load_state_dict(torch.load(args.ckpt, map_location=device))
            print(f"Loaded checkpoint from {args.ckpt}")
        else:
            print("Warning: Checkpoint not found, using random weights")
            
        # Wrap with MC Dropout
        mc_model = MCDropoutWrapper(base_model, dropout_rate=0.1, n_samples=10)
        mc_model.to(device)

        # Calibrate
        print("Running calibration...")
        metrics = calibrate_uncertainty(mc_model, val_loader, device=device)
        print("\nCalibration metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
