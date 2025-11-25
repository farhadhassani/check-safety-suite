"""
DeepLabV3+ Implementation
State-of-the-art semantic segmentation architecture

Reference: "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation"
Chen et al., 2018
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise Separable Convolution"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        if dilation == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.conv(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling Module"""
    
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        
        # Different dilation rates
        dilations = [1, 6, 12, 18]
        
        self.aspp1 = ASPPModule(in_channels, out_channels, dilation=dilations[0])
        self.aspp2 = ASPPModule(in_channels, out_channels, dilation=dilations[1])
        self.aspp3 = ASPPModule(in_channels, out_channels, dilation=dilations[2])
        self.aspp4 = ASPPModule(in_channels, out_channels, dilation=dilations[3])
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Apply ASPP modules
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=size, mode='bilinear', align_corners=True)
        
        # Concatenate and project
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        
        return x


class Encoder(nn.Module):
    """ResNet-based Encoder"""
    
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # ResNet blocks (simplified)
        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        
        # First block with stride
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ))
        
        # Remaining blocks
        for _ in range(1, blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder with skip connections
        x = self.conv1(x)
        x = self.maxpool(x)
        
        low_level_feat = self.layer1(x)  # 1/4 resolution
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)  # 1/16 resolution
        
        return x, low_level_feat


class Decoder(nn.Module):
    """DeepLabV3+ Decoder"""
    
    def __init__(self, low_level_channels=64, num_classes=1):
        super().__init__()
        
        # Low-level feature projection
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            SeparableConv2d(256 + 48, 256, 3, padding=1),
            SeparableConv2d(256, 256, 3, padding=1),
            nn.Dropout(0.5)
        )
        
        # Final classification
        self.classifier = nn.Conv2d(256, num_classes, 1)
    
    def forward(self, x, low_level_feat):
        # Upsample encoder output
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # Project low-level features
        low_level_feat = self.project(low_level_feat)
        
        # Concatenate
        x = torch.cat([x, low_level_feat], dim=1)
        
        # Decode
        x = self.decoder(x)
        
        # Classify
        x = self.classifier(x)
        
        return x


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ Architecture
    
    Key Features:
    - Atrous Spatial Pyramid Pooling (ASPP)
    - Encoder-decoder structure
    - Multi-scale feature extraction
    - State-of-the-art performance
    """
    
    def __init__(self, in_channels=3, num_classes=1):
        """
        Args:
            in_channels: Number of input channels
            num_classes: Number of output classes
        """
        super().__init__()
        
        # Encoder
        self.encoder = Encoder(in_channels)
        
        # ASPP
        self.aspp = ASPP(in_channels=512, out_channels=256)
        
        # Decoder
        self.decoder = Decoder(low_level_channels=64, num_classes=num_classes)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Encoder
        x, low_level_feat = self.encoder(x)
        
        # ASPP
        x = self.aspp(x)
        
        # Decoder
        x = self.decoder(x, low_level_feat)
        
        # Upsample to input size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing DeepLabV3+\n")
    
    model = DeepLabV3Plus(in_channels=3, num_classes=1)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Test forward pass
    x = torch.randn(2, 3, 512, 512)
    print(f"\n🔄 Forward pass:")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        y = model(x)
    
    print(f"  Output shape: {y.shape}")
    
    # Test with different input sizes
    print(f"\n🔄 Testing different input sizes:")
    for size in [256, 384, 512, 640]:
        x_test = torch.randn(1, 3, size, size)
        with torch.no_grad():
            y_test = model(x_test)
        print(f"  Input {size}×{size} → Output {y_test.shape[2]}×{y_test.shape[3]}")
    
    print("\n✅ DeepLabV3+ test passed!")
