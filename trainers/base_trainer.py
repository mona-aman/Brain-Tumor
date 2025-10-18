"""
Base trainer class with common functionality
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Base trainer with common training utilities
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        config: Configuration dict
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: str,
        config: Dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.early_stopping_counter = 0
        
        # Setup directories
        self.checkpoint_dir = Path(config['checkpointing']['save_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        raise NotImplementedError
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        raise NotImplementedError
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint
        
        Args:
            metrics: Metrics dict
            is_best: If True, save as best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'checkpoint_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['metrics']
    
    def check_early_stopping(self, metric: float) -> bool:
        """
        Check if training should stop early
        
        Args:
            metric: Metric to monitor
        
        Returns:
            True if should stop
        """
        patience = self.config['training']['early_stopping']['patience']
        min_delta = self.config['training']['early_stopping']['min_delta']
        
        if metric < self.best_metric - min_delta:
            self.best_metric = metric
            self.early_stopping_counter = 0
            return False
        else:
            self.early_stopping_counter += 1
            if self.early_stopping_counter >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement")
                return True
            return False