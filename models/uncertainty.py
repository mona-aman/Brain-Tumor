"""
Uncertainty quantification using Monte Carlo Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import numpy as np


class UncertaintyWrapper(nn.Module):
    """
    Wrapper that adds Monte Carlo Dropout to any classifier
    
    Args:
        model: Base classification model
        dropout_rate: Dropout probability
        num_samples: Number of MC samples for uncertainty estimation
    """
    
    def __init__(
        self,
        model: nn.Module,
        dropout_rate: float = 0.2,
        num_samples: int = 20
    ):
        super().__init__()
        self.model = model
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
        # Add dropout layers if not present
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        x: torch.Tensor,
        return_uncertainty: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional uncertainty estimation
        
        Args:
            x: Input images [B, C, H, W]
            return_uncertainty: If True, perform MC Dropout
        
        Returns:
            Dictionary containing:
                - prediction: Predicted class [B]
                - confidence: Max probability [B]
                - logits: Raw logits [B, num_classes] (if not MC)
                - entropy: Predictive entropy [B] (if MC)
                - mutual_info: Epistemic uncertainty [B] (if MC)
                - all_probs: All MC samples [num_samples, B, num_classes] (if MC)
        """
        if not return_uncertainty:
            # Standard inference
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'logits': logits,
                'probs': probs
            }
        
        # Monte Carlo Dropout
        self.train()  # Enable dropout
        
        all_probs = []
        for _ in range(self.num_samples):
            with torch.no_grad():
                logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)
        
        self.eval()  # Restore eval mode
        
        all_probs = torch.stack(all_probs)  # [num_samples, B, num_classes]
        
        # Compute statistics
        mean_probs = all_probs.mean(dim=0)  # [B, num_classes]
        confidence, prediction = torch.max(mean_probs, dim=1)
        
        # Uncertainty metrics
        # 1. Predictive entropy (total uncertainty)
        entropy = self._compute_entropy(mean_probs)
        
        # 2. Mutual information (epistemic uncertainty)
        expected_entropy = torch.mean(
            self._compute_entropy(all_probs),
            dim=0
        )
        mutual_info = entropy - expected_entropy
        
        # 3. Variance
        variance = all_probs.var(dim=0).mean(dim=1)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probs': mean_probs,
            'entropy': entropy,
            'mutual_info': mutual_info,
            'variance': variance,
            'all_probs': all_probs
        }
    
    @staticmethod
    def _compute_entropy(probs: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy: H = -Σ p(y) log p(y)
        
        Args:
            probs: Probabilities [..., num_classes]
        
        Returns:
            Entropy values [...]
        """
        return -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    
    def calibrate_threshold(
        self,
        dataloader: torch.utils.data.DataLoader,
        target_accuracy: float = 0.95,
        device: str = 'cuda'
    ) -> float:
        """
        Find uncertainty threshold that achieves target accuracy on confident predictions
        
        Args:
            dataloader: Validation dataloader
            target_accuracy: Desired accuracy on non-flagged samples
            device: Device to run on
        
        Returns:
            Optimal uncertainty threshold
        """
        self.eval()
        self.to(device)
        
        all_uncertainties = []
        all_correct = []
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                
                output = self(images, return_uncertainty=True)
                correct = (output['prediction'] == labels).float()
                
                all_uncertainties.append(output['entropy'])
                all_correct.append(correct)
        
        uncertainties = torch.cat(all_uncertainties).cpu().numpy()
        correct = torch.cat(all_correct).cpu().numpy()
        
        # Find threshold
        thresholds = np.percentile(uncertainties, np.linspace(0, 100, 100))
        
        best_threshold = 0.0
        best_diff = float('inf')
        
        for threshold in thresholds:
            confident_mask = uncertainties < threshold
            
            if confident_mask.sum() == 0:
                continue
            
            confident_accuracy = correct[confident_mask].mean()
            diff = abs(confident_accuracy - target_accuracy)
            
            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
        
        return float(best_threshold)


def evaluate_uncertainty(
    model: UncertaintyWrapper,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5
) -> Dict:
    """
    Comprehensive uncertainty evaluation
    
    Args:
        model: UncertaintyWrapper model
        dataloader: Test dataloader
        device: Device
        threshold: Uncertainty threshold for flagging
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    all_confidences = []
    all_correct = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            output = model(images, return_uncertainty=True)
            correct = (output['prediction'] == labels)
            
            all_predictions.append(output['prediction'])
            all_labels.append(labels)
            all_uncertainties.append(output['entropy'])
            all_confidences.append(output['confidence'])
            all_correct.append(correct)
    
    # Concatenate
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    uncertainties = torch.cat(all_uncertainties)
    confidences = torch.cat(all_confidences)
    correct = torch.cat(all_correct)
    
    # Compute metrics
    accuracy = correct.float().mean().item()
    
    # Flagging analysis
    flagged = uncertainties > threshold
    flag_rate = flagged.float().mean().item()
    
    if (~flagged).sum() > 0:
        confident_accuracy = correct[~flagged].float().mean().item()
    else:
        confident_accuracy = 0.0
    
    if flagged.sum() > 0:
        flag_precision = (~correct[flagged]).float().mean().item()
    else:
        flag_precision = 0.0
    
    # Correlation between uncertainty and errors
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(
        uncertainties.cpu().numpy(),
        (~correct).cpu().numpy()
    )
    
    # AUC for uncertainty as error detector
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(
        (~correct).cpu().numpy(),
        uncertainties.cpu().numpy()
    )
    
    return {
        'accuracy': accuracy,
        'flag_rate': flag_rate,
        'confident_accuracy': confident_accuracy,
        'flag_precision': flag_precision,
        'uncertainty_error_correlation': correlation,
        'uncertainty_auc': auc,
        'mean_uncertainty': uncertainties.mean().item(),
        'mean_confidence': confidences.mean().item()
    }