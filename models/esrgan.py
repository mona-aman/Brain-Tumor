"""
Enhanced Super-Resolution GAN (ESRGAN) implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with dense connections"""
    
    def __init__(self, in_channels: int, growth_rate: int = 32, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.ModuleList()
        
        current_channels = in_channels
        for _ in range(num_blocks):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(current_channels, growth_rate, 3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.BatchNorm2d(growth_rate)
            ))
            current_channels += growth_rate
        
        self.final_conv = nn.Conv2d(current_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        for block in self.blocks:
            out = block(x)
            x = torch.cat([x, out], dim=1)
        return self.final_conv(x) + residual


class UpsampleBlock(nn.Module):
    """Upsampling block using pixel shuffle"""
    
    def __init__(self, in_channels: int, out_channels: int, scale_factor: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (scale_factor ** 2),
            3,
            padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    """
    ESRGAN Generator
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        out_channels: Number of output channels (3 for RGB)
        num_features: Number of feature maps
        num_blocks: Number of residual dense blocks
        scale_factor: Upsampling scale (2 for 2x, 4 for 4x)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_features: int = 64,
        num_blocks: int = 8,
        growth_rate: int = 32,
        scale_factor: int = 4
    ):
        super().__init__()
        
        self.scale_factor = scale_factor
        
        # Initial convolution
        self.initial_conv = nn.Conv2d(in_channels, num_features, 3, padding=1)
        
        # Residual dense blocks
        self.residual_blocks = nn.Sequential(
            *[ResidualDenseBlock(num_features, growth_rate) for _ in range(num_blocks)]
        )
        
        # Post-residual convolution
        self.post_residual_conv = nn.Conv2d(num_features, num_features, 3, padding=1)
        
        # Upsampling blocks
        if scale_factor == 4:
            self.upsample = nn.Sequential(
                UpsampleBlock(num_features, num_features, scale_factor=2),
                UpsampleBlock(num_features, num_features, scale_factor=2)
            )
        elif scale_factor == 2:
            self.upsample = UpsampleBlock(num_features, num_features, scale_factor=2)
        else:
            raise ValueError(f"Unsupported scale_factor: {scale_factor}")
        
        # Final convolution
        self.output_conv = nn.Conv2d(num_features, out_channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor, target_size: tuple = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, C, H, W]
            target_size: Optional target size (H, W) for output
        
        Returns:
            Super-resolved image [B, C, H*scale, W*scale]
        """
        # Initial feature extraction
        initial = self.initial_conv(x)
        
        # Residual dense blocks
        residual = self.residual_blocks(initial)
        residual = self.post_residual_conv(residual)
        
        # Add skip connection
        residual = residual + initial
        
        # Upsampling
        upsampled = self.upsample(residual)
        
        # Output
        output = self.output_conv(upsampled)
        
        # Resize to target size if specified
        if target_size is not None:
            output = F.interpolate(
                output,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
        
        return output