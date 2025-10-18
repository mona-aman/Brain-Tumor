"""
ResNet-based classifier for brain tumor classification
"""

import torch
import torch.nn as nn
import torchvision.models as models


class BrainTumorClassifier(nn.Module):
    """
    ResNet-50 based classifier for brain tumor classification
    
    Args:
        num_classes: Number of tumor classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Load pretrained ResNet50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Freeze early layers (optional)
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [B, 3, H, W]
        
        Returns:
            Logits [B, num_classes]
        """
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        """Freeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features
    """
    
    def __init__(self, layer_indices: list = [5, 10, 19, 28, 37]):
        super().__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract specific layers
        self.layers = nn.ModuleList()
        prev_idx = 0
        
        for idx in layer_indices:
            self.layers.append(vgg[prev_idx:idx])
            prev_idx = idx
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        self.eval()
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss between x and y
        
        Args:
            x: Generated images [B, 3, H, W]
            y: Target images [B, 3, H, W]
        
        Returns:
            Perceptual loss
        """
        loss = 0.0
        
        for layer in self.layers:
            x = layer(x)
            y = layer(y)
            loss += nn.functional.l1_loss(x, y)
        
        return loss