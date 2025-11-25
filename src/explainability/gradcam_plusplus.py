"""
GradCAM++ for Enhanced Explainability

Improved Gradient-weighted Class Activation Mapping for better localization
of important regions in the model's decision-making process.

Reference:
    Chattopadhay et al. "Grad-CAM++: Generalized Gradient-Based Visual
    Explanations for Deep Convolutional Networks" (2018)
    https://arxiv.org/abs/1710.11063
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, Tuple, List


class GradCAMPlusPlus:
    """
    GradCAM++ for generating visual explanations.
    
    Improvements over GradCAM:
    - Better localization for multiple instances
    - Handles multiple occurrences of same class
    - More robust weight calculation
    
    Args:
        model: PyTorch model
        target_layer: Layer to generate CAM from (usually last conv layer)
    
    Example:
        >>> model = UNet(...)
        >>> gradcam = GradCAMPlusPlus(model, target_layer=model.decoder.final_conv)
        >>> cam = gradcam.generate_cam(image)
        >>> heatmap = gradcam.apply_colormap(cam)
    """
    
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map.
        
        Args:
            input_tensor: Input image tensor [B, C, H, W]
            target_class: Target class index (None for regression/segmentation)
        
        Returns:
            cam: Class activation map [H, W] normalized to [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # For segmentation, use mean of output
        if target_class is None:
            score = output.mean()
        else:
            score = output[:, target_class].mean()
        
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients  # [B, C, H', W']
        activations = self.activations  # [B, C, H', W']
        
        # GradCAM++ weights calculation
        # α_kc = (∂²Y^c / ∂A^k²) / (2 * ∂²Y^c / ∂A^k² + Σ(A^k * ∂³Y^c / ∂A^k³))
        
        # Second derivative (approximated)
        grad_2 = gradients.pow(2)
        
        # Third derivative (approximated)
        grad_3 = gradients.pow(3)
        
        # Alpha weights
        alpha_num = grad_2
        alpha_denom = 2 * grad_2 + (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha_denom = torch.where(
            alpha_denom != 0.0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )
        alpha = alpha_num / alpha_denom
        
        # Weights for each channel
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode='bilinear',
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def apply_colormap(
        self,
        cam: np.ndarray,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Apply colormap to CAM.
        
        Args:
            cam: Class activation map [H, W]
            colormap: OpenCV colormap (default: COLORMAP_JET)
        
        Returns:
            heatmap: Colored heatmap [H, W, 3]
        """
        # Convert to uint8
        cam_uint8 = (cam * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(cam_uint8, colormap)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    def overlay_on_image(
        self,
        cam: np.ndarray,
        image: np.ndarray,
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Overlay CAM on original image.
        
        Args:
            cam: Class activation map [H, W]
            image: Original image [H, W, 3] in range [0, 255]
            alpha: Transparency factor (default: 0.5)
        
        Returns:
            overlayed: Image with CAM overlay [H, W, 3]
        """
        # Get heatmap
        heatmap = self.apply_colormap(cam)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Resize heatmap if needed
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Overlay
        overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlayed


class MultiLayerGradCAM:
    """
    Generate GradCAM from multiple layers and combine them.
    
    Useful for understanding different levels of feature hierarchy.
    """
    
    def __init__(self, model: nn.Module, target_layers: List[nn.Module]):
        self.gradcams = [
            GradCAMPlusPlus(model, layer) for layer in target_layers
        ]
    
    def generate_multilayer_cam(
        self,
        input_tensor: torch.Tensor,
        weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Generate combined CAM from multiple layers.
        
        Args:
            input_tensor: Input image tensor
            weights: Weights for each layer (default: equal weights)
        
        Returns:
            combined_cam: Combined activation map
        """
        if weights is None:
            weights = [1.0 / len(self.gradcams)] * len(self.gradcams)
        
        cams = []
        for gradcam in self.gradcams:
            cam = gradcam.generate_cam(input_tensor)
            cams.append(cam)
        
        # Weighted combination
        combined_cam = np.zeros_like(cams[0])
        for cam, weight in zip(cams, weights):
            combined_cam += weight * cam
        
        # Normalize
        combined_cam = (combined_cam - combined_cam.min()) / \
                      (combined_cam.max() - combined_cam.min() + 1e-8)
        
        return combined_cam


class GuidedBackprop:
    """
    Guided Backpropagation for generating high-resolution saliency maps.
    
    Can be combined with GradCAM for Guided GradCAM.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.gradients = None
        
        # Replace ReLU with guided ReLU
        self._update_relus()
    
    def _update_relus(self):
        """Replace ReLU backward pass with guided version"""
        
        def guided_relu_hook(module, grad_input, grad_output):
            # Only backprop positive gradients
            return (F.relu(grad_input[0]),)
        
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_full_backward_hook(guided_relu_hook)
    
    def generate_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate guided backpropagation saliency map.
        
        Args:
            input_tensor: Input image tensor [B, C, H, W]
            target_class: Target class (None for regression)
        
        Returns:
            saliency: Saliency map [H, W, C]
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        if target_class is None:
            score = output.mean()
        else:
            score = output[:, target_class].mean()
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Get gradients
        saliency = input_tensor.grad.data.squeeze().cpu().numpy()
        saliency = np.transpose(saliency, (1, 2, 0))
        
        # Normalize
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency


def guided_gradcam(
    gradcam: GradCAMPlusPlus,
    guided_bp: GuidedBackprop,
    input_tensor: torch.Tensor
) -> np.ndarray:
    """
    Combine GradCAM and Guided Backpropagation.
    
    Args:
        gradcam: GradCAM++ instance
        guided_bp: Guided Backpropagation instance
        input_tensor: Input image tensor
    
    Returns:
        guided_gradcam: Combined visualization
    """
    # Get CAM
    cam = gradcam.generate_cam(input_tensor)
    
    # Get guided backprop
    saliency = guided_bp.generate_gradients(input_tensor)
    
    # Combine: element-wise multiplication
    cam_3d = np.expand_dims(cam, axis=2)
    guided_gradcam = cam_3d * saliency
    
    # Normalize
    guided_gradcam = (guided_gradcam - guided_gradcam.min()) / \
                    (guided_gradcam.max() - guided_gradcam.min() + 1e-8)
    
    return guided_gradcam


if __name__ == '__main__':
    # Example usage
    print("GradCAM++ - Example Usage")
    print("=" * 50)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.final_conv = nn.Conv2d(128, 1, 1)
        
        def forward(self, x):
            x = self.relu1(self.conv1(x))
            x = self.relu2(self.conv2(x))
            x = self.final_conv(x)
            return x
    
    # Create model and GradCAM
    model = DummyModel()
    gradcam = GradCAMPlusPlus(model, target_layer=model.conv2)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    
    print("\n1. Generating CAM...")
    cam = gradcam.generate_cam(dummy_input)
    print(f"   CAM shape: {cam.shape}")
    print(f"   CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
    
    print("\n2. Applying colormap...")
    heatmap = gradcam.apply_colormap(cam)
    print(f"   Heatmap shape: {heatmap.shape}")
    
    print("\n3. Creating overlay...")
    dummy_image = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
    overlay = gradcam.overlay_on_image(cam, dummy_image, alpha=0.5)
    print(f"   Overlay shape: {overlay.shape}")
    
    print("\n✅ GradCAM++ implementation complete!")
    print("\nUsage in production:")
    print("  1. Generate CAM for tampered regions")
    print("  2. Overlay on original check image")
    print("  3. Show to bank teller for verification")
