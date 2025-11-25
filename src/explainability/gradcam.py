"""
GradCAM (Gradient-weighted Class Activation Mapping) for UNet
Visualizes which regions the model focuses on for predictions
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class GradCAM:
    """
    GradCAM implementation for segmentation models
    
    Reference:
    Selvaraju et al. "Grad-CAM: Visual Explanations from Deep Networks 
    via Gradient-based Localization" (ICCV 2017)
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Args:
            model: PyTorch model
            target_layer: Name of the layer to visualize (default: last conv layer)
        """
        self.model = model
        self.model.eval()
        
        # Storage for gradients and activations
        self.gradients = None
        self.activations = None
        
        # Register hooks
        if target_layer is None:
            # Find last convolutional layer
            target_layer = self._find_last_conv_layer()
        
        self.target_layer = self._get_layer_by_name(target_layer)
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _find_last_conv_layer(self) -> str:
        """Find the last convolutional layer in the model"""
        last_conv = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = name
        return last_conv
    
    def _get_layer_by_name(self, layer_name: str):
        """Get layer by its name"""
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")
    
    def _forward_hook(self, module, input, output):
        """Hook to capture activations"""
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        """Hook to capture gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate GradCAM heatmap
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (for classification)
                         For segmentation, use None to get average activation
        
        Returns:
            cam: GradCAM heatmap (H, W) normalized to [0, 1]
        """
        # Forward pass
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # For segmentation, we want to maximize the overall prediction
        if target_class is None:
            # Average over spatial dimensions
            score = output.mean()
        else:
            score = output[0, target_class].mean()
        
        # Backward pass
        score.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (1, C, H', W')
        activations = self.activations  # (1, C, H', W')
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        
        # Weighted combination of activation maps
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        
        # ReLU to keep only positive influences
        cam = F.relu(cam)
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # Resize to input size
        h, w = input_tensor.shape[2:]
        cam = cv2.resize(cam, (w, h))
        
        return cam
    
    def visualize(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Create visualization by overlaying CAM on image
        
        Args:
            image: Original image (H, W, 3) in RGB, range [0, 255]
            cam: GradCAM heatmap (H, W) in range [0, 1]
            alpha: Blending factor
            colormap: OpenCV colormap
        
        Returns:
            overlay: Visualization (H, W, 3) in RGB
        """
        # Convert CAM to heatmap
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        # Blend
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        
        return overlay


class GuidedBackprop:
    """
    Guided Backpropagation for visualizing gradients
    
    Reference:
    Springenberg et al. "Striving for Simplicity: The All Convolutional Net" (2015)
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        
        # Store original forward functions
        self.relu_outputs = []
        
        # Register hooks for all ReLU layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks for ReLU layers"""
        def relu_backward_hook(module, grad_input, grad_output):
            # Only backprop positive gradients
            return (F.relu(grad_input[0]),)
        
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                module.register_full_backward_hook(relu_backward_hook)
    
    def generate_gradients(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate guided backpropagation gradients
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class (None for segmentation)
        
        Returns:
            gradients: Gradient visualization (H, W, C)
        """
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Backward pass
        self.model.zero_grad()
        if target_class is None:
            score = output.mean()
        else:
            score = output[0, target_class].mean()
        
        score.backward()
        
        # Get gradients
        gradients = input_tensor.grad.data.squeeze().cpu().numpy()
        gradients = gradients.transpose(1, 2, 0)  # (H, W, C)
        
        # Normalize
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        return gradients


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Overlay heatmap on image
    
    Args:
        image: RGB image (H, W, 3)
        heatmap: Heatmap (H, W) in [0, 1]
        alpha: Blending factor
    
    Returns:
        overlay: Blended image
    """
    # Convert heatmap to RGB
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Blend
    overlay = cv2.addWeighted(
        image.astype(np.uint8),
        1 - alpha,
        heatmap_colored,
        alpha,
        0
    )
    
    return overlay


if __name__ == "__main__":
    # Example usage
    from src.models.unet import UNet
    
    # Load model
    model = UNet(n_channels=3, n_classes=1)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Generate GradCAM
    gradcam = GradCAM(model)
    cam = gradcam.generate_cam(dummy_input)
    
    print(f"GradCAM shape: {cam.shape}")
    print(f"GradCAM range: [{cam.min():.3f}, {cam.max():.3f}]")
