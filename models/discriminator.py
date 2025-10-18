"""
Discriminator network for ESRGAN
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    VGG-style discriminator for ESRGAN
    
    Args:
        in_channels: Number of input channels (3 for RGB)
        num_features: Base number of feature maps
    """
    
    def __init__(self, in_channels: int = 3, num_features: int = 64):
        super().__init__()
        
        self.features = nn.Sequential(
            # Initial layer
            nn.Conv2d(in_channels, num_features, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Downsampling blocks
            *self._conv_block(num_features, num_features * 2),
            *self._conv_block(num_features * 2, num_features * 4),
            *self._conv_block(num_features * 4, num_features * 8),
            *self._conv_block(num_features * 8, num_features * 8),
            
            # Final conv
            nn.Conv2d(num_features * 8, num_features * 8, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Global pooling and classification
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features * 8, 1),
            nn.Sigmoid()
        )
    
    @staticmethod
    def _conv_block(in_channels: int, out_channels: int):
        """Convolutional block with downsampling"""
        return [
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        ]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, C, H, W]
        
        Returns:
            Probability of being real [B, 1]
        """
        features = self.features(x)
        pooled = self.global_avg_pool(features)
        return self.classifier(pooled)


class RaGANLoss(nn.Module):
    """Relativistic Average GAN Loss"""
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
        for_discriminator: bool = True
    ) -> torch.Tensor:
        """
        Args:
            real_pred: Discriminator output for real images
            fake_pred: Discriminator output for fake images
            for_discriminator: If True, compute D loss, else G loss
        
        Returns:
            Loss value
        """
        if for_discriminator:
            # Discriminator loss
            real_loss = torch.mean(
                torch.nn.functional.softplus(-(real_pred - torch.mean(fake_pred)))
            )
            fake_loss = torch.mean(
                torch.nn.functional.softplus(fake_pred - torch.mean(real_pred))
            )
            return (real_loss + fake_loss) / 2
        else:
            # Generator loss
            loss = torch.mean(
                torch.nn.functional.softplus(-(fake_pred - torch.mean(real_pred)))
            )
            return loss