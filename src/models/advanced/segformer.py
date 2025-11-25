"""
SegFormer Implementation
Transformer-based semantic segmentation architecture

Reference: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
Xie et al., 2021

Note: This is a simplified implementation of SegFormer-B0 (smallest variant)
For production, consider using pretrained weights from timm or transformers library
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class PatchEmbed(nn.Module):
    """Patch Embedding Layer"""
    
    def __init__(self, in_channels=3, embed_dim=32, patch_size=7, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        x = self.norm(x)
        return x, H, W


class EfficientSelfAttention(nn.Module):
    """Efficient Self-Attention with reduction"""
    
    def __init__(self, dim, num_heads=8, sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        
        # Spatial reduction
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # Query
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        # Key, Value with spatial reduction
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class MLP(nn.Module):
    """Feed-Forward Network"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientSelfAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    
    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class MixVisionTransformer(nn.Module):
    """Mix Vision Transformer (MiT) Encoder"""
    
    def __init__(
        self,
        in_channels=3,
        embed_dims=[32, 64, 160, 256],  # SegFormer-B0
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1]
    ):
        super().__init__()
        
        # Patch embeddings
        self.patch_embed1 = PatchEmbed(in_channels, embed_dims[0], patch_size=7, stride=4)
        self.patch_embed2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2)
        self.patch_embed3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2)
        self.patch_embed4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2)
        
        # Transformer blocks
        self.block1 = nn.ModuleList([
            TransformerBlock(embed_dims[0], num_heads[0], mlp_ratios[0], sr_ratios[0])
            for _ in range(depths[0])
        ])
        self.block2 = nn.ModuleList([
            TransformerBlock(embed_dims[1], num_heads[1], mlp_ratios[1], sr_ratios[1])
            for _ in range(depths[1])
        ])
        self.block3 = nn.ModuleList([
            TransformerBlock(embed_dims[2], num_heads[2], mlp_ratios[2], sr_ratios[2])
            for _ in range(depths[2])
        ])
        self.block4 = nn.ModuleList([
            TransformerBlock(embed_dims[3], num_heads[3], mlp_ratios[3], sr_ratios[3])
            for _ in range(depths[3])
        ])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dims[0])
        self.norm2 = nn.LayerNorm(embed_dims[1])
        self.norm3 = nn.LayerNorm(embed_dims[2])
        self.norm4 = nn.LayerNorm(embed_dims[3])
    
    def forward(self, x):
        B = x.shape[0]
        outs = []
        
        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        outs.append(x)
        
        return outs


class MLPDecoder(nn.Module):
    """Lightweight MLP Decoder"""
    
    def __init__(self, in_channels=[32, 64, 160, 256], embedding_dim=256, num_classes=1):
        super().__init__()
        
        # Linear layers to project all features to same dimension
        self.linear_c4 = nn.Linear(in_channels[3], embedding_dim)
        self.linear_c3 = nn.Linear(in_channels[2], embedding_dim)
        self.linear_c2 = nn.Linear(in_channels[1], embedding_dim)
        self.linear_c1 = nn.Linear(in_channels[0], embedding_dim)
        
        # Fusion layer
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
        # Final classifier
        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)
    
    def forward(self, features):
        c1, c2, c3, c4 = features
        
        # Get dimensions - use c1 as reference (largest spatial size)
        n, _, h, w = c1.shape
        
        # Project all features to embedding_dim and upsample to c1 size
        _c4 = self.linear_c4(c4.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c4 = F.interpolate(_c4, size=(h, w), mode='bilinear', align_corners=False)
        
        _c3 = self.linear_c3(c3.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c3 = F.interpolate(_c3, size=(h, w), mode='bilinear', align_corners=False)
        
        _c2 = self.linear_c2(c2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        _c2 = F.interpolate(_c2, size=(h, w), mode='bilinear', align_corners=False)
        
        _c1 = self.linear_c1(c1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Concatenate and fuse
        _c = torch.cat([_c4, _c3, _c2, _c1], dim=1)
        _c = self.linear_fuse(_c)
        
        # Classify
        x = self.classifier(_c)
        
        return x


class SegFormer(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation
    
    Key Features:
    - Hierarchical Transformer encoder (MiT)
    - Lightweight MLP decoder
    - No positional encoding needed
    - Excellent for fine details
    
    This is SegFormer-B0 (smallest variant):
    - embed_dims: [32, 64, 160, 256]
    - depths: [2, 2, 2, 2]
    - ~3.8M parameters
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Encoder
        self.encoder = MixVisionTransformer(
            in_channels=in_channels,
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1]
        )
        
        # Decoder
        self.decoder = MLPDecoder(
            in_channels=[32, 64, 160, 256],
            embedding_dim=256,
            num_classes=num_classes
        )
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encode
        features = self.encoder(x)
        
        # Decode
        x = self.decoder(features)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        
        return x


if __name__ == "__main__":
    import time
    
    # Test the model
    print("🧪 Testing SegFormer-B0\n")
    
    model = SegFormer(in_channels=3, num_classes=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = model.to(device)
    x = torch.randn(2, 3, 512, 512).to(device)
    
    print(f"\n🔄 Forward pass:")
    print(f"  Input shape: {x.shape}")
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    # Timed inference
    start = time.time()
    with torch.no_grad():
        y = model(x)
    elapsed = (time.time() - start) * 1000
    
    print(f"  Output shape: {y.shape}")
    print(f"  Inference time: {elapsed:.1f}ms ({elapsed/2:.1f}ms per image)")
    
    # Test with different batch sizes
    print(f"\n🔄 Testing different batch sizes:")
    for batch_size in [1, 2, 4]:
        x_test = torch.randn(batch_size, 3, 512, 512).to(device)
        start = time.time()
        with torch.no_grad():
            y_test = model(x_test)
        elapsed = (time.time() - start) * 1000
        print(f"  Batch {batch_size}: {elapsed:.1f}ms total, {elapsed/batch_size:.1f}ms per image")
    
    # Memory usage
    if device == 'cuda':
        print(f"\n💾 GPU Memory:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"  Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
    
    print("\n✅ SegFormer test passed!")
    print("\n⚠️ Note: SegFormer is slower on CPU (~2-3s per image)")
    print("   Recommended for GPU deployment or research comparison only")
