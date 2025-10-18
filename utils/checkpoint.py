"""
Checkpoint management utilities
"""

import torch
from pathlib import Path
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage model checkpoints
    
    Args:
        save_dir: Directory to save checkpoints
        max_checkpoints: Maximum number of checkpoints to keep
    """
    
    def __init__(self, save_dir: str, max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict,
        is_best: bool = False,
        filename: str = None
    ):
        """Save model checkpoint"""
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save with filename
        if filename is None:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        checkpoint_path = self.save_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Track checkpoints
        self.checkpoints.append(checkpoint_path)
        
        # Remove old checkpoints
        if len(self.checkpoints) > self.max_checkpoints:
            old_checkpoint = self.checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
                logger.debug(f"Removed old checkpoint: {old_checkpoint}")
        
        # Save best checkpoint
        if is_best:
            best_path = self.save_dir / 'checkpoint_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint: {best_path}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: str = None,
        load_best: bool = True
    ) -> Dict:
        """Load model checkpoint"""
        
        if checkpoint_path is None:
            if load_best:
                checkpoint_path = self.save_dir / 'checkpoint_best.pth'
            else:
                checkpoint_path = self.save_dir / 'checkpoint_latest.pth'
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Loaded checkpoint from: {checkpoint_path}")
        logger.info(f"Epoch: {checkpoint['epoch']}")
        logger.info(f"Metrics: {checkpoint['metrics']}")
        
        return checkpoint