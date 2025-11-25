"""
Monte Carlo Dropout for Uncertainty Estimation
Enables uncertainty quantification without retraining the model
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict
import cv2
from tqdm import tqdm


class MCDropoutWrapper(nn.Module):
    """
    Wrapper for any PyTorch model to enable Monte Carlo Dropout
    
    Usage:
        model = UNet(3, 1)
        mc_model = MCDropoutWrapper(model, dropout_rate=0.2)
        mean, uncertainty = mc_model.predict_with_uncertainty(image, n_samples=20)
    """
    
    def __init__(self, model: nn.Module, dropout_rate: float = 0.2):
        """
        Args:
            model: Base model (e.g., UNet)
            dropout_rate: Dropout probability for MC sampling
        """
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        
        # Add dropout layers after each conv block
        self._add_dropout_layers()
    
    def _add_dropout_layers(self):
        """Recursively add dropout after ReLU activations"""
        def add_dropout_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    # Add dropout after ReLU
                    setattr(module, name, nn.Sequential(
                        child,
                        nn.Dropout2d(p=self.dropout_rate)
                    ))
                else:
                    add_dropout_recursive(child)
        
        add_dropout_recursive(self.model)
    
    def enable_dropout(self):
        """Enable dropout during inference"""
        def enable_dropout_recursive(module):
            for m in module.modules():
                if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                    m.train()
        
        enable_dropout_recursive(self.model)
    
    def forward(self, x):
        """Standard forward pass"""
        return self.model(x)
    
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        n_samples: int = 20,
        return_all: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Predict with uncertainty using Monte Carlo Dropout
        
        Args:
            x: Input tensor (B, C, H, W)
            n_samples: Number of MC samples
            return_all: If True, return all predictions
        
        Returns:
            mean_pred: Mean prediction (B, 1, H, W)
            uncertainty: Uncertainty map (B, 1, H, W)
            metrics: Dictionary with uncertainty metrics
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(x)
                pred = torch.sigmoid(pred)
                predictions.append(pred.cpu())
        
        # Stack predictions: (n_samples, B, 1, H, W)
        predictions = torch.stack(predictions)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Uncertainty metrics
        # 1. Predictive variance
        variance = predictions.var(dim=0)
        
        # 2. Predictive entropy
        epsilon = 1e-8
        entropy = -(mean_pred * torch.log(mean_pred + epsilon) + 
                   (1 - mean_pred) * torch.log(1 - mean_pred + epsilon))
        
        # 3. Mutual information (epistemic uncertainty)
        # MI = H[E[y|x]] - E[H[y|x,θ]]
        expected_entropy = -(predictions * torch.log(predictions + epsilon) +
                           (1 - predictions) * torch.log(1 - predictions + epsilon)).mean(dim=0)
        mutual_info = entropy - expected_entropy
        
        metrics = {
            'mean': mean_pred,
            'std': std_pred,
            'variance': variance,
            'entropy': entropy,
            'mutual_information': mutual_info,
            'coefficient_of_variation': std_pred / (mean_pred + epsilon)
        }
        
        if return_all:
            metrics['all_predictions'] = predictions
        
        return mean_pred, variance, metrics


class MCDropoutPredictor:
    """
    High-level interface for MC Dropout predictions on images
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        dropout_rate: float = 0.2,
        n_samples: int = 20
    ):
        """
        Args:
            model: Trained model
            device: Device to run on
            dropout_rate: Dropout probability
            n_samples: Number of MC samples
        """
        self.mc_model = MCDropoutWrapper(model, dropout_rate).to(device)
        self.device = device
        self.n_samples = n_samples
        self.img_size = 512
    
    def predict_image(
        self,
        image: np.ndarray,
        return_metrics: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict on a single image with uncertainty
        
        Args:
            image: Input image (H, W, 3) in BGR
            return_metrics: If True, return detailed metrics
        
        Returns:
            mean_pred: Mean prediction map (H, W)
            uncertainty: Uncertainty map (H, W)
            metrics: Dictionary with uncertainty metrics
        """
        h, w = image.shape[:2]
        
        # Preprocess
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        # MC Dropout prediction
        mean_pred, variance, metrics = self.mc_model.predict_with_uncertainty(
            img_tensor,
            n_samples=self.n_samples
        )
        
        # Convert to numpy and resize back
        mean_pred_np = mean_pred[0, 0].cpu().numpy()
        variance_np = variance[0, 0].cpu().numpy()
        
        mean_pred_resized = cv2.resize(mean_pred_np, (w, h))
        variance_resized = cv2.resize(variance_np, (w, h))
        
        if return_metrics:
            # Resize all metrics
            metrics_resized = {}
            for key, value in metrics.items():
                if isinstance(value, torch.Tensor) and value.dim() == 4:
                    value_np = value[0, 0].cpu().numpy()
                    metrics_resized[key] = cv2.resize(value_np, (w, h))
                else:
                    metrics_resized[key] = value
            
            return mean_pred_resized, variance_resized, metrics_resized
        
        return mean_pred_resized, variance_resized, {}
    
    def predict_batch(
        self,
        images: list,
        show_progress: bool = True
    ) -> Tuple[list, list, list]:
        """
        Predict on a batch of images
        
        Args:
            images: List of images (H, W, 3) in BGR
            show_progress: Show progress bar
        
        Returns:
            mean_preds: List of mean predictions
            uncertainties: List of uncertainty maps
            metrics_list: List of metrics dictionaries
        """
        mean_preds = []
        uncertainties = []
        metrics_list = []
        
        iterator = tqdm(images, desc="MC Dropout") if show_progress else images
        
        for img in iterator:
            mean, unc, metrics = self.predict_image(img)
            mean_preds.append(mean)
            uncertainties.append(unc)
            metrics_list.append(metrics)
        
        return mean_preds, uncertainties, metrics_list
    
    def get_uncertainty_threshold(
        self,
        uncertainty_map: np.ndarray,
        percentile: float = 95
    ) -> float:
        """
        Get uncertainty threshold at given percentile
        
        Args:
            uncertainty_map: Uncertainty map
            percentile: Percentile for threshold
        
        Returns:
            threshold: Uncertainty threshold
        """
        return np.percentile(uncertainty_map, percentile)
    
    def flag_uncertain_regions(
        self,
        mean_pred: np.ndarray,
        uncertainty: np.ndarray,
        uncertainty_threshold: float = None,
        percentile: float = 90
    ) -> Tuple[np.ndarray, Dict]:
        """
        Flag regions with high uncertainty for manual review
        
        Args:
            mean_pred: Mean prediction
            uncertainty: Uncertainty map
            uncertainty_threshold: Manual threshold (optional)
            percentile: Percentile for auto threshold
        
        Returns:
            flags: Binary mask of uncertain regions
            info: Dictionary with flagging information
        """
        if uncertainty_threshold is None:
            uncertainty_threshold = self.get_uncertainty_threshold(
                uncertainty,
                percentile=percentile
            )
        
        # Flag high uncertainty regions
        uncertain_mask = uncertainty > uncertainty_threshold
        
        # Also flag regions with medium confidence (0.3-0.7)
        ambiguous_mask = (mean_pred > 0.3) & (mean_pred < 0.7)
        
        # Combine flags
        flags = uncertain_mask | ambiguous_mask
        
        info = {
            'uncertainty_threshold': uncertainty_threshold,
            'uncertain_pixels': uncertain_mask.sum(),
            'ambiguous_pixels': ambiguous_mask.sum(),
            'total_flagged': flags.sum(),
            'flagged_percentage': flags.mean() * 100,
            'max_uncertainty': uncertainty.max(),
            'mean_uncertainty': uncertainty.mean()
        }
        
        return flags, info


def visualize_uncertainty(
    image: np.ndarray,
    mean_pred: np.ndarray,
    uncertainty: np.ndarray,
    save_path: str = None
) -> np.ndarray:
    """
    Create visualization of prediction with uncertainty
    
    Args:
        image: Original image (H, W, 3)
        mean_pred: Mean prediction (H, W)
        uncertainty: Uncertainty map (H, W)
        save_path: Optional path to save visualization
    
    Returns:
        vis: Visualization image
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Mean prediction
    axes[1].imshow(mean_pred, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('Mean Prediction')
    axes[1].axis('off')
    
    # Uncertainty
    im = axes[2].imshow(uncertainty, cmap='hot')
    axes[2].set_title('Uncertainty (Variance)')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    # Overlay
    overlay = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
    uncertainty_colored = cm.hot(uncertainty / uncertainty.max())[:, :, :3]
    uncertainty_colored = (uncertainty_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(overlay, 0.5, uncertainty_colored, 0.5, 0)
    axes[3].imshow(overlay)
    axes[3].set_title('Uncertainty Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to image
    fig.canvas.draw()
    vis = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis = vis.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    return vis


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    
    from src.models.unet import UNet
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = UNet(3, 1).to(device)
    
    checkpoint_path = Path("outputs/training/best_model.pth")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("✅ Model loaded")
    
    # Create MC Dropout predictor
    predictor = MCDropoutPredictor(
        model,
        device=device,
        dropout_rate=0.2,
        n_samples=20
    )
    
    # Test on sample image
    sample_image_path = "data/idrbt_tamper/tamper_001.jpg"
    if Path(sample_image_path).exists():
        image = cv2.imread(sample_image_path)
        
        print(f"\n🔍 Running MC Dropout with {predictor.n_samples} samples...")
        mean_pred, uncertainty, metrics = predictor.predict_image(image)
        
        print(f"\n📊 Results:")
        print(f"  Mean prediction: {mean_pred.mean():.3f} ± {mean_pred.std():.3f}")
        print(f"  Uncertainty (variance): {uncertainty.mean():.4f} (max: {uncertainty.max():.4f})")
        print(f"  Entropy: {metrics['entropy'].mean():.4f}")
        print(f"  Mutual Information: {metrics['mutual_information'].mean():.4f}")
        
        # Flag uncertain regions
        flags, info = predictor.flag_uncertain_regions(mean_pred, uncertainty)
        print(f"\n🚩 Uncertain Regions:")
        print(f"  Flagged: {info['flagged_percentage']:.1f}% of pixels")
        print(f"  Threshold: {info['uncertainty_threshold']:.4f}")
        
        # Visualize
        vis = visualize_uncertainty(
            image,
            mean_pred,
            uncertainty,
            save_path="outputs/mc_dropout_demo.png"
        )
        print(f"\n✅ Visualization saved to outputs/mc_dropout_demo.png")
    else:
        print(f"❌ Sample image not found: {sample_image_path}")
