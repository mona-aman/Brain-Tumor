"""
Trainer for Task-Aware ESRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import logging

from .esrgan_trainer import ESRGANTrainer
from models.taskaware_esrgan import TaskAwareGenerator
from models.discriminator import Discriminator

logger = logging.getLogger(__name__)


class TaskAwareTrainer(ESRGANTrainer):
    """
    Trainer for Task-Aware ESRGAN
    
    Extends ESRGANTrainer with classification loss
    """
    
    def __init__(
        self,
        generator: TaskAwareGenerator,
        discriminator: Discriminator,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: str,
        config: Dict
    ):
        super().__init__(
            generator, discriminator,
            g_optimizer, d_optimizer,
            device, config
        )
        
        # Additional loss weight for classification
        self.lambda_classification = config['loss_weights']['classification']
        
        # Warmup epochs (train without classification loss first)
        self.warmup_epochs = config['training'].get('warmup_epochs', 0)
        
        logger.info(f"Task-Aware Trainer initialized with λ_classification={self.lambda_classification}")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch with classification loss"""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        total_class_loss = 0
        num_batches = 0
        
        # Check if we're in warmup phase
        use_classification_loss = self.current_epoch >= self.warmup_epochs
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for hr_img, labels in pbar:
            hr_img = hr_img.to(self.device)
            labels = labels.to(self.device)
            
            # Skip invalid batches
            if (labels == -1).any():
                continue
            
            # Create low-resolution images
            lr_img = F.interpolate(
                hr_img,
                scale_factor=0.25,
                mode='bilinear',
                align_corners=False
            )
            
            # Generate super-resolved images
            sr_img = self.generator(lr_img, target_size=hr_img.shape[2:])
            
            # --- Train Discriminator (same as standard ESRGAN) ---
            self.d_optimizer.zero_grad()
            
            real_pred = self.discriminator(hr_img)
            fake_pred = self.discriminator(sr_img.detach())
            
            d_loss = self.ragan_loss(real_pred, fake_pred, for_discriminator=True)
            d_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.d_optimizer.step()
            
            # --- Train Generator (with classification loss) ---
            self.g_optimizer.zero_grad()
            
            # Adversarial loss
            real_pred = self.discriminator(hr_img)
            fake_pred = self.discriminator(sr_img)
            g_adv_loss = self.ragan_loss(real_pred, fake_pred, for_discriminator=False)
            
            # Perceptual loss
            g_perceptual_loss = self.perceptual_loss(sr_img, hr_img)
            
            # Content loss
            g_content_loss = self.content_loss(sr_img, hr_img)
            
            # Classification loss (NEW!)
            if use_classification_loss:
                g_class_loss = self.generator.compute_classification_loss(sr_img, labels)
            else:
                g_class_loss = torch.tensor(0.0).to(self.device)
            
            # Combined loss
            g_loss = (
                self.lambda_adv * g_adv_loss +
                self.lambda_perceptual * g_perceptual_loss +
                self.lambda_content * g_content_loss +
                self.lambda_classification * g_class_loss
            )
            
            g_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.g_optimizer.step()
            
            # Accumulate losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            total_class_loss += g_class_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item(),
                'Class_loss': g_class_loss.item()
            })
        
        return {
            'g_loss': total_g_loss / num_batches,
            'd_loss': total_d_loss / num_batches,
            'class_loss': total_class_loss / num_batches
        }