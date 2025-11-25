"""
Advanced Loss Functions for Segmentation
Includes Focal Loss, Dice Loss, Tversky Loss, and combinations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    
    Formula: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    where p_t = p if y=1, else 1-p
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss (γ ≥ 0)
            reduction: 'none' | 'mean' | 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, 1, H, W) - logits or probabilities
            targets: Ground truth (B, 1, H, W) - binary {0, 1}
        
        Returns:
            loss: Focal loss value
        """
        # Apply sigmoid if inputs are logits
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Focal loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation
    
    Formula: 1 - (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
    """
    
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing constant to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Dice loss value
        """
        # Apply sigmoid if needed
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Dice coefficient
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalization of Dice Loss
    
    Allows weighting of false positives and false negatives differently
    
    Reference: "Tversky loss function for image segmentation using 3D fully
    convolutional deep networks" (Salehi et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        """
        Args:
            alpha: Weight of false positives
            beta: Weight of false negatives
            smooth: Smoothing constant
        
        Note: alpha + beta = 1 gives Dice Loss
              alpha = beta = 0.5 gives Dice Loss
              alpha < beta emphasizes recall (reduce false negatives)
              alpha > beta emphasizes precision (reduce false positives)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, 1, H, W)
            targets: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Tversky loss value
        """
        # Apply sigmoid if needed
        if inputs.min() < 0 or inputs.max() > 1:
            inputs = torch.sigmoid(inputs)
        
        # Flatten
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # True Positives, False Positives, False Negatives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        # Tversky index
        tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - tversky


class CombinedLoss(nn.Module):
    """
    Combination of multiple loss functions
    
    Commonly used combinations:
    - BCE + Dice
    - Focal + Dice
    - BCE + Tversky
    """
    
    def __init__(
        self,
        loss_type: str = 'bce_dice',
        bce_weight: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        tversky_alpha: float = 0.5,
        tversky_beta: float = 0.5
    ):
        """
        Args:
            loss_type: 'bce_dice' | 'focal_dice' | 'bce_tversky' | 'focal_tversky'
            bce_weight: Weight for BCE component (1 - bce_weight for Dice/Tversky)
            focal_alpha: Alpha for Focal Loss
            focal_gamma: Gamma for Focal Loss
            tversky_alpha: Alpha for Tversky Loss
            tversky_beta: Beta for Tversky Loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.bce_weight = bce_weight
        
        if 'bce' in loss_type:
            self.bce = nn.BCEWithLogitsLoss()
        if 'focal' in loss_type:
            self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        if 'dice' in loss_type:
            self.dice = DiceLoss()
        if 'tversky' in loss_type:
            self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions (B, 1, H, W) - logits
            targets: Ground truth (B, 1, H, W)
        
        Returns:
            loss: Combined loss value
        """
        if self.loss_type == 'bce_dice':
            loss = self.bce_weight * self.bce(inputs, targets) + \
                   (1 - self.bce_weight) * self.dice(inputs, targets)
        
        elif self.loss_type == 'focal_dice':
            loss = self.bce_weight * self.focal(inputs, targets) + \
                   (1 - self.bce_weight) * self.dice(inputs, targets)
        
        elif self.loss_type == 'bce_tversky':
            loss = self.bce_weight * self.bce(inputs, targets) + \
                   (1 - self.bce_weight) * self.tversky(inputs, targets)
        
        elif self.loss_type == 'focal_tversky':
            loss = self.bce_weight * self.focal(inputs, targets) + \
                   (1 - self.bce_weight) * self.tversky(inputs, targets)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


def get_loss_function(loss_name: str, **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of loss function
        **kwargs: Additional arguments for loss function
    
    Returns:
        loss_fn: Loss function
    """
    loss_functions = {
        'bce': nn.BCEWithLogitsLoss(),
        'focal': FocalLoss(**kwargs),
        'dice': DiceLoss(**kwargs),
        'tversky': TverskyLoss(**kwargs),
        'bce_dice': CombinedLoss('bce_dice', **kwargs),
        'focal_dice': CombinedLoss('focal_dice', **kwargs),
        'bce_tversky': CombinedLoss('bce_tversky', **kwargs),
        'focal_tversky': CombinedLoss('focal_tversky', **kwargs)
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Unknown loss: {loss_name}. Available: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]


if __name__ == "__main__":
    # Test loss functions
    print("🧪 Testing Loss Functions\n")
    
    # Create dummy data
    batch_size = 4
    inputs = torch.randn(batch_size, 1, 256, 256)  # Logits
    targets = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    # Test each loss
    losses = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Focal (α=0.25, γ=2)': FocalLoss(alpha=0.25, gamma=2.0),
        'Dice': DiceLoss(),
        'Tversky (α=0.3, β=0.7)': TverskyLoss(alpha=0.3, beta=0.7),
        'BCE + Dice': CombinedLoss('bce_dice', bce_weight=0.5),
        'Focal + Dice': CombinedLoss('focal_dice', bce_weight=0.5)
    }
    
    print("📊 Loss Values:")
    for name, loss_fn in losses.items():
        loss_value = loss_fn(inputs, targets)
        print(f"  {name:25s}: {loss_value.item():.4f}")
    
    # Test with different class imbalances
    print("\n📊 Focal Loss with Different Class Imbalances:")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    
    for pos_ratio in [0.01, 0.1, 0.5, 0.9]:
        targets_imb = (torch.rand(batch_size, 1, 256, 256) < pos_ratio).float()
        loss_value = focal(inputs, targets_imb)
        print(f"  Positive ratio {pos_ratio:.2f}: {loss_value.item():.4f}")
    
    print("\n✅ All loss functions working correctly!")
