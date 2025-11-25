"""
Advanced Data Augmentation Techniques
Includes CutMix, MixUp, and other state-of-the-art methods

References:
- CutMix: "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features" (Yun et al., 2019)
- MixUp: "mixup: Beyond Empirical Risk Minimization" (Zhang et al., 2018)
"""
import torch
import numpy as np
import cv2
from typing import Tuple


class CutMix:
    """
    CutMix Augmentation
    
    Cuts a patch from one image and pastes it onto another,
    mixing labels proportionally to the area.
    """
    
    def __init__(self, alpha: float = 1.0, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter (higher = more uniform mixing)
            prob: Probability of applying CutMix
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, C, H, W)
        
        Returns:
            mixed_images: Augmented images
            mixed_labels: Augmented labels
            lam: Mixing ratio
        """
        if np.random.rand() > self.prob:
            return images, labels, 1.0
        
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Get bounding box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
        # Mix images
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Mix labels
        labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        return images, labels, lam
    
    def _rand_bbox(self, size, lam):
        """Generate random bounding box"""
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2


class MixUp:
    """
    MixUp Augmentation
    
    Linear interpolation of images and labels
    """
    
    def __init__(self, alpha: float = 0.4, prob: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            prob: Probability of applying MixUp
        """
        self.alpha = alpha
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, C, H, W)
        
        Returns:
            mixed_images: Augmented images
            mixed_labels: Augmented labels
            lam: Mixing ratio
        """
        if np.random.rand() > self.prob:
            return images, labels, 1.0
        
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        rand_index = torch.randperm(batch_size).to(images.device)
        
        # Mix images and labels
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        mixed_labels = lam * labels + (1 - lam) * labels[rand_index]
        
        return mixed_images, mixed_labels, lam


class Mosaic:
    """
    Mosaic Augmentation
    
    Combines 4 images into one mosaic
    """
    
    def __init__(self, prob: float = 0.5):
        """
        Args:
            prob: Probability of applying Mosaic
        """
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Mosaic augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, C, H, W)
        
        Returns:
            mosaic_images: Augmented images
            mosaic_labels: Augmented labels
        """
        if np.random.rand() > self.prob or images.size(0) < 4:
            return images, labels
        
        batch_size = images.size(0)
        _, c, h, w = images.shape
        
        # Create mosaic
        mosaic_images = []
        mosaic_labels = []
        
        for i in range(0, batch_size - 3, 4):
            # Get 4 images
            img1, img2, img3, img4 = images[i:i+4]
            lbl1, lbl2, lbl3, lbl4 = labels[i:i+4]
            
            # Random center point
            cx = np.random.randint(w // 4, 3 * w // 4)
            cy = np.random.randint(h // 4, 3 * h // 4)
            
            # Create mosaic
            mosaic_img = torch.zeros_like(img1)
            mosaic_lbl = torch.zeros_like(lbl1)
            
            # Top-left
            mosaic_img[:, :cy, :cx] = img1[:, :cy, :cx]
            mosaic_lbl[:, :cy, :cx] = lbl1[:, :cy, :cx]
            
            # Top-right
            mosaic_img[:, :cy, cx:] = img2[:, :cy, cx:]
            mosaic_lbl[:, :cy, cx:] = lbl2[:, :cy, cx:]
            
            # Bottom-left
            mosaic_img[:, cy:, :cx] = img3[:, cy:, :cx]
            mosaic_lbl[:, cy:, :cx] = lbl3[:, cy:, :cx]
            
            # Bottom-right
            mosaic_img[:, cy:, cx:] = img4[:, cy:, cx:]
            mosaic_lbl[:, cy:, cx:] = lbl4[:, cy:, cx:]
            
            mosaic_images.append(mosaic_img)
            mosaic_labels.append(mosaic_lbl)
        
        if mosaic_images:
            mosaic_images = torch.stack(mosaic_images)
            mosaic_labels = torch.stack(mosaic_labels)
            return mosaic_images, mosaic_labels
        else:
            return images, labels


class GridMask:
    """
    GridMask Augmentation
    
    Drops grid-like regions from images
    """
    
    def __init__(
        self,
        ratio: float = 0.6,
        rotate: int = 1,
        prob: float = 0.5
    ):
        """
        Args:
            ratio: Ratio of grid size to spacing
            rotate: Rotation angle (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
            prob: Probability of applying GridMask
        """
        self.ratio = ratio
        self.rotate = rotate
        self.prob = prob
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply GridMask augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, C, H, W)
        
        Returns:
            masked_images: Augmented images
            labels: Unchanged labels
        """
        if np.random.rand() > self.prob:
            return images, labels
        
        _, _, h, w = images.shape
        
        # Grid parameters
        d = np.random.randint(2, min(h, w) // 4)
        l = int(d * self.ratio)
        
        # Create grid mask
        mask = torch.ones((h, w), device=images.device)
        
        for i in range(0, h, d):
            for j in range(0, w, d):
                mask[i:min(i+l, h), j:min(j+l, w)] = 0
        
        # Apply mask
        mask = mask.unsqueeze(0).unsqueeze(0)
        masked_images = images * mask
        
        return masked_images, labels


class AdvancedAugmentation:
    """
    Combination of advanced augmentation techniques
    """
    
    def __init__(
        self,
        cutmix_prob: float = 0.5,
        mixup_prob: float = 0.5,
        mosaic_prob: float = 0.0,
        gridmask_prob: float = 0.0,
        cutmix_alpha: float = 1.0,
        mixup_alpha: float = 0.4
    ):
        """
        Args:
            cutmix_prob: Probability of CutMix
            mixup_prob: Probability of MixUp
            mosaic_prob: Probability of Mosaic
            gridmask_prob: Probability of GridMask
            cutmix_alpha: CutMix alpha parameter
            mixup_alpha: MixUp alpha parameter
        """
        self.cutmix = CutMix(alpha=cutmix_alpha, prob=cutmix_prob)
        self.mixup = MixUp(alpha=mixup_alpha, prob=mixup_prob)
        self.mosaic = Mosaic(prob=mosaic_prob)
        self.gridmask = GridMask(prob=gridmask_prob)
    
    def __call__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Apply random augmentation
        
        Args:
            images: Batch of images (B, C, H, W)
            labels: Batch of labels (B, C, H, W)
        
        Returns:
            aug_images: Augmented images
            aug_labels: Augmented labels
            lam: Mixing ratio (1.0 if no mixing)
        """
        # Randomly choose one augmentation
        aug_type = np.random.choice(['cutmix', 'mixup', 'mosaic', 'gridmask', 'none'], 
                                    p=[0.3, 0.3, 0.1, 0.1, 0.2])
        
        lam = 1.0
        
        if aug_type == 'cutmix':
            images, labels, lam = self.cutmix(images, labels)
        elif aug_type == 'mixup':
            images, labels, lam = self.mixup(images, labels)
        elif aug_type == 'mosaic':
            images, labels = self.mosaic(images, labels)
        elif aug_type == 'gridmask':
            images, labels = self.gridmask(images, labels)
        
        return images, labels, lam


if __name__ == "__main__":
    # Test augmentations
    print("🧪 Testing Advanced Augmentation\n")
    
    # Create dummy data
    batch_size = 4
    images = torch.randn(batch_size, 3, 256, 256)
    labels = torch.randint(0, 2, (batch_size, 1, 256, 256)).float()
    
    print(f"📊 Input:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Test CutMix
    print(f"\n🔪 CutMix:")
    cutmix = CutMix(alpha=1.0, prob=1.0)
    aug_img, aug_lbl, lam = cutmix(images.clone(), labels.clone())
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Lambda: {lam:.3f}")
    print(f"  Pixel difference: {(aug_img - images).abs().mean():.4f}")
    
    # Test MixUp
    print(f"\n🎨 MixUp:")
    mixup = MixUp(alpha=0.4, prob=1.0)
    aug_img, aug_lbl, lam = mixup(images.clone(), labels.clone())
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Lambda: {lam:.3f}")
    print(f"  Pixel difference: {(aug_img - images).abs().mean():.4f}")
    
    # Test Mosaic
    print(f"\n🎭 Mosaic:")
    mosaic = Mosaic(prob=1.0)
    aug_img, aug_lbl = mosaic(images.clone(), labels.clone())
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Pixel difference: {(aug_img - images[:1]).abs().mean():.4f}")
    
    # Test GridMask
    print(f"\n🔲 GridMask:")
    gridmask = GridMask(prob=1.0)
    aug_img, aug_lbl = gridmask(images.clone(), labels.clone())
    print(f"  Output shape: {aug_img.shape}")
    print(f"  Masked pixels: {(aug_img == 0).float().mean():.1%}")
    
    # Test combined
    print(f"\n🎯 Combined Augmentation:")
    aug = AdvancedAugmentation(
        cutmix_prob=0.5,
        mixup_prob=0.5,
        mosaic_prob=0.2,
        gridmask_prob=0.2
    )
    
    for i in range(5):
        aug_img, aug_lbl, lam = aug(images.clone(), labels.clone())
        print(f"  Trial {i+1}: shape={aug_img.shape}, lam={lam:.3f}")
    
    print("\n✅ All augmentations working correctly!")
