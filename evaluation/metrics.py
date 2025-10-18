"""
Evaluation metrics for image quality and classification
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Tuple


def calculate_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> float:
    """
    Calculate SSIM between two images
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        data_range: Data range (1.0 for normalized images)
    
    Returns:
        Average SSIM across batch
    """
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    ssim_values = []
    
    for i in range(img1.shape[0]):
        # Transpose to (H, W, C)
        img1_single = np.transpose(img1[i], (1, 2, 0))
        img2_single = np.transpose(img2[i], (1, 2, 0))
        
        # Clip to valid range
        img1_single = np.clip(img1_single, 0, 1)
        img2_single = np.clip(img2_single, 0, 1)
        
        # Calculate SSIM
        ssim_value = ssim(
            img1_single,
            img2_single,
            multichannel=True,
            data_range=data_range,
            win_size=3
        )
        ssim_values.append(ssim_value)
    
    return np.mean(ssim_values)


def calculate_psnr(
    img1: torch.Tensor,
    img2: torch.Tensor,
    data_range: float = 1.0
) -> float:
    """
    Calculate PSNR between two images
    
    Args:
        img1: First image [B, C, H, W]
        img2: Second image [B, C, H, W]
        data_range: Data range (1.0 for normalized images)
    
    Returns:
        Average PSNR across batch
    """
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    psnr_values = []
    
    for i in range(img1.shape[0]):
        # Transpose to (H, W, C)
        img1_single = np.transpose(img1[i], (1, 2, 0))
        img2_single = np.transpose(img2[i], (1, 2, 0))
        
        # Clip to valid range
        img1_single = np.clip(img1_single, 0, 1)
        img2_single = np.clip(img2_single, 0, 1)
        
        # Calculate PSNR
        psnr_value = psnr(img1_single, img2_single, data_range=data_range)
        psnr_values.append(psnr_value)
    
    return np.mean(psnr_values)


def calculate_mse(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Mean Squared Error"""
    mse = F.mse_loss(img1, img2)
    return mse.item()


def calculate_classification_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int = 4
) -> dict:
    """
    Calculate classification metrics
    
    Args:
        predictions: Predicted labels [N]
        labels: Ground truth labels [N]
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix,
        classification_report
    )
    
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = \
        precision_recall_fscore_support(labels, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'support': support
    }