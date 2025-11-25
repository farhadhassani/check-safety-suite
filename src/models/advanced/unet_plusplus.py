"""
UNet++ (Nested UNet) Implementation
Improved architecture with nested skip connections

Reference: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"
Zhou et al., 2018
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Double convolution block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNetPlusPlus(nn.Module):
    """
    UNet++ with nested skip connections
    
    Architecture:
        - Dense skip connections between encoder and decoder
        - Multiple supervision levels (deep supervision)
        - Better gradient flow
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        deep_supervision: bool = False
    ):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            deep_supervision: Use deep supervision (multiple outputs)
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        
        # Filters for each level
        filters = [64, 128, 256, 512, 1024]
        
        # Encoder (downsampling)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Level 0 (bottom)
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])
        self.conv4_0 = ConvBlock(filters[3], filters[4])
        
        # Nested connections - Level 1
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])
        self.conv3_1 = ConvBlock(filters[3] + filters[4], filters[3])
        
        # Nested connections - Level 2
        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])
        self.conv2_2 = ConvBlock(filters[2]*2 + filters[3], filters[2])
        
        # Nested connections - Level 3
        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])
        self.conv1_3 = ConvBlock(filters[1]*3 + filters[2], filters[1])
        
        # Nested connections - Level 4
        self.conv0_4 = ConvBlock(filters[0]*4 + filters[1], filters[0])
        
        # Upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final convolutions
        if self.deep_supervision:
            self.final1 = nn.Conv2d(filters[0], out_channels, 1)
            self.final2 = nn.Conv2d(filters[0], out_channels, 1)
            self.final3 = nn.Conv2d(filters[0], out_channels, 1)
            self.final4 = nn.Conv2d(filters[0], out_channels, 1)
        else:
            self.final = nn.Conv2d(filters[0], out_channels, 1)
    
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        # Output
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            return self.final(x0_4)


if __name__ == "__main__":
    # Test the model
    print("🧪 Testing UNet++\n")
    
    model = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=False)
    
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
    
    # Test with deep supervision
    print(f"\n🔄 Forward pass (deep supervision):")
    model_ds = UNetPlusPlus(in_channels=3, out_channels=1, deep_supervision=True)
    
    with torch.no_grad():
        outputs = model_ds(x)
    
    print(f"  Number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output {i+1} shape: {out.shape}")
    
    print("\n✅ UNet++ test passed!")
