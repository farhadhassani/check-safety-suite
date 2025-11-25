"""
Monte Carlo Dropout for Uncertainty Quantification

This module implements MC Dropout to provide uncertainty estimates
for model predictions. It wraps existing models and enables dropout
during inference for multiple forward passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class MCDropoutWrapper(nn.Module):
    """
    Wrapper for any PyTorch model to enable Monte Carlo Dropout.
    
    Monte Carlo Dropout performs multiple forward passes with dropout enabled
    to estimate prediction uncertainty. The variance across predictions
    indicates model uncertainty.
    
    Args:
        base_model: The base model to wrap (e.g., UNet)
        dropout_rate: Dropout probability (default: 0.1)
        n_samples: Number of MC samples for uncertainty estimation (default: 10)
        apply_dropout_to: Which layers to apply dropout ('all', 'decoder', 'encoder')
    
    Example:
        >>> model = UNet(n_channels=3, n_classes=1)
        >>> mc_model = MCDropoutWrapper(model, dropout_rate=0.1, n_samples=10)
        >>> mean_pred, uncertainty = mc_model(image, return_uncertainty=True)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 10,
        apply_dropout_to: str = 'all'
    ):
        super().__init__()
        self.base_model = base_model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.apply_dropout_to = apply_dropout_to
        
        # Inject dropout layers
        self._inject_dropout()
    
    def _inject_dropout(self):
        """Inject dropout layers into the base model"""
        def add_dropout(module):
            for name, child in module.named_children():
                if isinstance(child, nn.ReLU):
                    # Add dropout after ReLU activations
                    setattr(
                        module,
                        name,
                        nn.Sequential(
                            child,
                            nn.Dropout2d(p=self.dropout_rate)
                        )
                    )
                else:
                    add_dropout(child)
        
        if self.apply_dropout_to == 'all':
            add_dropout(self.base_model)
        elif self.apply_dropout_to == 'decoder':
            if hasattr(self.base_model, 'decoder'):
                add_dropout(self.base_model.decoder)
        elif self.apply_dropout_to == 'encoder':
            if hasattr(self.base_model, 'encoder'):
                add_dropout(self.base_model.encoder)
    
    def enable_dropout(self):
        """Enable dropout layers during inference"""
        def _enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()
        
        self.base_model.apply(_enable_dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional uncertainty estimation.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_uncertainty: If True, perform MC sampling and return uncertainty
        
        Returns:
            If return_uncertainty=False:
                predictions: Model predictions [B, 1, H, W]
            If return_uncertainty=True:
                (mean_predictions, uncertainty): Tuple of mean and std [B, 1, H, W]
        """
        if not return_uncertainty:
            # Standard inference (no MC sampling)
            self.base_model.eval()
            return self.base_model(x)
        
        # Monte Carlo Dropout inference
        self.base_model.eval()  # Set base model to eval
        self.enable_dropout()   # But keep dropout enabled
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.base_model(x)
                predictions.append(pred)
        
        # Stack predictions: [n_samples, B, 1, H, W]
        predictions = torch.stack(predictions, dim=0)
        
        # Calculate statistics
        mean_pred = predictions.mean(dim=0)  # [B, 1, H, W]
        uncertainty = predictions.std(dim=0)  # [B, 1, H, W]
        
        return mean_pred, uncertainty
    
    def predict_with_metrics(
        self,
        x: torch.Tensor
    ) -> dict:
        """
        Predict with comprehensive uncertainty metrics.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Dictionary containing:
                - mean: Mean prediction
                - uncertainty: Pixel-wise uncertainty (std)
                - entropy: Predictive entropy
                - mutual_info: Mutual information
                - confidence: Mean confidence (1 - uncertainty)
        """
        mean_pred, uncertainty = self.forward(x, return_uncertainty=True)
        
        # Apply sigmoid to get probabilities
        mean_prob = torch.sigmoid(mean_pred)
        
        # Calculate entropy: -p*log(p) - (1-p)*log(1-p)
        eps = 1e-7
        entropy = -(
            mean_prob * torch.log(mean_prob + eps) +
            (1 - mean_prob) * torch.log(1 - mean_prob + eps)
        )
        
        # Confidence: inverse of uncertainty
        confidence = 1.0 - uncertainty
        
        return {
            'mean': mean_pred,
            'uncertainty': uncertainty,
            'entropy': entropy,
            'confidence': confidence,
            'uncertainty_mean': float(uncertainty.mean()),
            'uncertainty_max': float(uncertainty.max()),
            'confidence_mean': float(confidence.mean()),
        }


class MCDropoutUNet(MCDropoutWrapper):
    """
    Specialized MC Dropout wrapper for UNet models.
    
    This class provides UNet-specific functionality and optimizations.
    """
    
    def __init__(
        self,
        base_unet: nn.Module,
        dropout_rate: float = 0.1,
        n_samples: int = 10
    ):
        super().__init__(
            base_model=base_unet,
            dropout_rate=dropout_rate,
            n_samples=n_samples,
            apply_dropout_to='decoder'  # Apply only to decoder for efficiency
        )
    
    def get_uncertain_regions(
        self,
        x: torch.Tensor,
        uncertainty_threshold: float = 0.3
    ) -> Tuple[torch.Tensor, list]:
        """
        Identify regions with high uncertainty.
        
        Args:
            x: Input tensor [B, C, H, W]
            uncertainty_threshold: Threshold for high uncertainty
        
        Returns:
            uncertainty_mask: Binary mask of uncertain regions [B, 1, H, W]
            uncertain_regions: List of bounding boxes for uncertain regions
        """
        _, uncertainty = self.forward(x, return_uncertainty=True)
        
        # Create binary mask
        uncertainty_mask = (uncertainty > uncertainty_threshold).float()
        
        # Find connected components (bounding boxes)
        uncertain_regions = []
        
        for b in range(uncertainty.shape[0]):
            # Convert to numpy for connected components
            mask_np = uncertainty_mask[b, 0].cpu().numpy()
            
            # Find contours
            import cv2
            contours, _ = cv2.findContours(
                (mask_np * 255).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                uncertain_regions.append({
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'uncertainty_mean': float(uncertainty[b, 0, y:y+h, x:x+w].mean())
                })
        
        return uncertainty_mask, uncertain_regions


def calibrate_uncertainty(
    model: MCDropoutWrapper,
    val_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> dict:
    """
    Calibrate uncertainty estimates on validation set.
    
    Measures how well uncertainty correlates with prediction errors.
    
    Args:
        model: MC Dropout model
        val_loader: Validation data loader
        device: Device to run on
    
    Returns:
        Dictionary with calibration metrics:
            - uncertainty_error_correlation: Correlation between uncertainty and error
            - ece: Expected Calibration Error
            - bins: Calibration bins
    """
    model.eval()
    
    uncertainties = []
    errors = []
    predictions = []
    targets = []
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Get predictions with uncertainty
            mean_pred, uncertainty = model(images, return_uncertainty=True)
            
            # Calculate error
            error = torch.abs(torch.sigmoid(mean_pred) - masks)
            
            uncertainties.append(uncertainty.cpu())
            errors.append(error.cpu())
            predictions.append(torch.sigmoid(mean_pred).cpu())
            targets.append(masks.cpu())
    
    # Concatenate all batches
    uncertainties = torch.cat(uncertainties, dim=0).flatten()
    errors = torch.cat(errors, dim=0).flatten()
    predictions = torch.cat(predictions, dim=0).flatten()
    targets = torch.cat(targets, dim=0).flatten()
    
    # Calculate correlation
    correlation = np.corrcoef(
        uncertainties.numpy(),
        errors.numpy()
    )[0, 1]
    
    # Calculate Expected Calibration Error (ECE)
    n_bins = 10
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    bin_stats = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find predictions in this bin
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.float().mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_stats.append({
                'bin_lower': float(bin_lower),
                'bin_upper': float(bin_upper),
                'accuracy': float(accuracy_in_bin),
                'confidence': float(avg_confidence_in_bin),
                'proportion': float(prop_in_bin)
            })
    
    return {
        'uncertainty_error_correlation': float(correlation),
        'ece': float(ece),
        'bins': bin_stats,
        'mean_uncertainty': float(uncertainties.mean()),
        'mean_error': float(errors.mean())
    }


if __name__ == '__main__':
    # Example usage
    print("Monte Carlo Dropout - Example Usage")
    print("=" * 50)
    
    # Create dummy model
    from torch import nn
    
    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 1, 3, padding=1)
        
        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.conv2(x)
            return x
    
    # Wrap with MC Dropout
    base_model = DummyUNet()
    mc_model = MCDropoutWrapper(base_model, dropout_rate=0.1, n_samples=10)
    
    # Test inference
    dummy_input = torch.randn(1, 3, 512, 512)
    
    print("\n1. Standard inference (no uncertainty):")
    pred = mc_model(dummy_input, return_uncertainty=False)
    print(f"   Output shape: {pred.shape}")
    
    print("\n2. MC Dropout inference (with uncertainty):")
    mean_pred, uncertainty = mc_model(dummy_input, return_uncertainty=True)
    print(f"   Mean prediction shape: {mean_pred.shape}")
    print(f"   Uncertainty shape: {uncertainty.shape}")
    print(f"   Mean uncertainty: {uncertainty.mean():.4f}")
    print(f"   Max uncertainty: {uncertainty.max():.4f}")
    
    print("\n3. Comprehensive metrics:")
    metrics = mc_model.predict_with_metrics(dummy_input)
    print(f"   Mean uncertainty: {metrics['uncertainty_mean']:.4f}")
    print(f"   Max uncertainty: {metrics['uncertainty_max']:.4f}")
    print(f"   Mean confidence: {metrics['confidence_mean']:.4f}")
    
    print("\n✅ MC Dropout implementation complete!")
