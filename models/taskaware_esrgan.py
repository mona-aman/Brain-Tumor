"""
Task-Aware ESRGAN - Optimizes for downstream classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .esrgan import Generator


class TaskAwareGenerator(Generator):
    """
    Generator that optimizes for classification performance
    
    Args:
        classifier: Pre-trained classifier model (frozen)
        All other args same as Generator
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        freeze_classifier: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.classifier = classifier
        self.classifier.eval()
        
        # Freeze classifier parameters
        if freeze_classifier:
            for param in self.classifier.parameters():
                param.requires_grad = False
    
    def forward(
        self,
        x: torch.Tensor,
        target_size: tuple = None,
        return_classification: bool = False
    ):
        """
        Args:
            x: Input tensor [B, C, H, W]
            target_size: Optional target size
            return_classification: If True, also return classification logits
        
        Returns:
            sr_img: Super-resolved image
            (optional) logits: Classification logits
        """
        sr_img = super().forward(x, target_size)
        
        if return_classification:
            with torch.no_grad() if self.classifier.training else torch.enable_grad():
                logits = self.classifier(sr_img)
            return sr_img, logits
        
        return sr_img
    
    def compute_classification_loss(
        self,
        sr_img: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss on generated images
        
        Args:
            sr_img: Super-resolved images [B, C, H, W]
            labels: Ground truth labels [B]
        
        Returns:
            Classification loss
        """
        logits = self.classifier(sr_img)
        return F.cross_entropy(logits, labels)