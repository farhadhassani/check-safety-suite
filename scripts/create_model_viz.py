"""
Create side-by-side visualization: Model Prediction vs Ground Truth

Generates a comparison image showing:
- Original check image
- Ground truth mask
- Model prediction
- Overlay (prediction on original)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from src.models.unet_plusplus import UNetPlusPlus


def load_trained_model(checkpoint_path, device='cpu'):
    """Load trained UNet++ model"""
    model = UNetPlusPlus(n_channels=3, n_classes=1, deep_supervision=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, target_size=256):
    """Load and preprocess image for model input"""
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img_resized = cv2.resize(img, (target_size, target_size))
    
    # To tensor and normalize
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    return img_tensor, img_resized


def load_mask(mask_path, target_size=256):
    """Load ground truth mask"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (target_size, target_size))
    mask = (mask > 127).astype(np.uint8)  # Binarize
    return mask


def predict(model, image_tensor, device='cpu'):
    """Run model inference"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        pred_mask = (probs > 0.5).float()
    
    pred_mask = pred_mask.squeeze().cpu().numpy()
    return pred_mask


def create_comparison_viz(image, gt_mask, pred_mask, save_path):
    """Create 4-panel comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Model Performance: UNet++ with Hybrid Loss', fontsize=16, fontweight='bold')
    
    # Panel 1: Original Image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Check Image', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Panel 2: Ground Truth
    axes[0, 1].imshow(gt_mask, cmap='Reds', alpha=0.8)
    axes[0, 1].set_title('Ground Truth Tamper Mask', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Panel 3: Model Prediction
    axes[1, 0].imshow(pred_mask, cmap='Blues', alpha=0.8)
    axes[1, 0].set_title('Model Prediction', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Panel 4: Overlay on Original
    overlay = image.copy()
    pred_colored = np.zeros_like(image)
    pred_colored[:, :, 0] = pred_mask * 255  # Red channel for prediction
    overlay = cv2.addWeighted(overlay, 0.7, pred_colored, 0.3, 0)
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Prediction Overlay', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization: {save_path}")
    plt.close()


def main():
    print("=" * 70)
    print("CREATING MODEL VISUALIZATION: Prediction vs Ground Truth")
    print("=" * 70)
    
    # Paths
    checkpoint_path = "outputs/real_mask_training/best_model.pth"
    data_dir = Path("data/bcsd_prepared/val")
    output_path = "outputs/demo/model_comparison.png"
    
    # Check if model exists
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Model not found: {checkpoint_path}")
        print("Please train the model first using: python scripts/train_real_data.py")
        return
    
    # Load model
    print(f"\nLoading model from: {checkpoint_path}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(checkpoint_path, device)
    print(f"✓ Model loaded on {device}")
    
    # Get a sample from validation set
    image_files = sorted((data_dir / "images").glob("*.jpg"))
    
    if len(image_files) == 0:
        print(f"\n❌ No validation images found in: {data_dir}/images")
        return
    
    # Use first image (or you can select a specific one)
    sample_idx = 0
    image_path = image_files[sample_idx]
    mask_path = data_dir / "masks" / image_path.name
    
    print(f"\nProcessing sample: {image_path.name}")
    
    # Load and preprocess
    image_tensor, image_display = preprocess_image(image_path)
    gt_mask = load_mask(mask_path)
    
    # Predict
    print("Running inference...")
    pred_mask = predict(model, image_tensor, device)
    
    # Calculate metrics
    intersection = (pred_mask * gt_mask).sum()
    union = ((pred_mask + gt_mask) > 0).sum()
    iou = intersection / (union + 1e-7)
    
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-7)
    
    print(f"\nMetrics:")
    print(f"  IoU: {iou:.4f}")
    print(f"  Dice: {dice:.4f}")
    
    # Create visualization
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    create_comparison_viz(image_display, gt_mask, pred_mask, output_path)
    
    print("\n" + "=" * 70)
    print("✅ Visualization created successfully!")
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
