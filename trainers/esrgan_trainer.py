"""
Trainer for standard ESRGAN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
import logging

from .base_trainer import BaseTrainer
from models.esrgan import Generator
from models.discriminator import Discriminator, RaGANLoss
from models.classifier import PerceptualLoss

logger = logging.getLogger(__name__)


class ESRGANTrainer(BaseTrainer):
    """
    Trainer for standard ESRGAN
    
    Args:
        generator: Generator model
        discriminator: Discriminator model
        g_optimizer: Generator optimizer
        d_optimizer: Discriminator optimizer
        device: Device
        config: Configuration dict
    """
    
    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        g_optimizer: torch.optim.Optimizer,
        d_optimizer: torch.optim.Optimizer,
        device: str,
        config: Dict
    ):
        # Initialize base trainer with generator
        super().__init__(generator, g_optimizer, nn.L1Loss(), device, config)
        
        self.generator = generator
        self.discriminator = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        
        # Loss functions
        self.ragan_loss = RaGANLoss()
        self.perceptual_loss = PerceptualLoss().to(device)
        self.content_loss = nn.L1Loss()
        
        # Loss weights
        self.lambda_adv = config['loss_weights']['adversarial']
        self.lambda_perceptual = config['loss_weights']['perceptual']
        self.lambda_content = config['loss_weights']['content']
        
        # Metrics tracking
        self.metrics_history = {
            'g_loss': [],
            'd_loss': [],
            'ssim': [],
            'psnr': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0
        total_d_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}")
        
        for hr_img, labels in pbar:
            hr_img = hr_img.to(self.device)
            batch_size = hr_img.size(0)
            
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
            
            # --- Train Discriminator ---
            self.d_optimizer.zero_grad()
            
            real_pred = self.discriminator(hr_img)
            fake_pred = self.discriminator(sr_img.detach())
            
            d_loss = self.ragan_loss(real_pred, fake_pred, for_discriminator=True)
            d_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.d_optimizer.step()
            
            # --- Train Generator ---
            self.g_optimizer.zero_grad()
            
            # Adversarial loss
            real_pred = self.discriminator(hr_img)
            fake_pred = self.discriminator(sr_img)
            g_adv_loss = self.ragan_loss(real_pred, fake_pred, for_discriminator=False)
            
            # Perceptual loss
            g_perceptual_loss = self.perceptual_loss(sr_img, hr_img)
            
            # Content loss
            g_content_loss = self.content_loss(sr_img, hr_img)
            
            # Combined loss
            g_loss = (
                self.lambda_adv * g_adv_loss +
                self.lambda_perceptual * g_perceptual_loss +
                self.lambda_content * g_content_loss
            )
            
            g_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.generator.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.g_optimizer.step()
            
            # Accumulate losses
            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'G_loss': g_loss.item(),
                'D_loss': d_loss.item()
            })
        
        avg_g_loss = total_g_loss / num_batches
        avg_d_loss = total_d_loss / num_batches
        
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.generator.eval()
        
        from evaluation.metrics import calculate_ssim, calculate_psnr
        
        total_ssim = 0
        total_psnr = 0
        num_batches = 0
        
        with torch.no_grad():
            for hr_img, labels in dataloader:
                hr_img = hr_img.to(self.device)
                
                # Skip invalid batches
                if (labels == -1).any():
                    continue
                
                # Create LR images
                lr_img = F.interpolate(
                    hr_img,
                    scale_factor=0.25,
                    mode='bilinear',
                    align_corners=False
                )
                
                # Generate SR images
                sr_img = self.generator(lr_img, target_size=hr_img.shape[2:])
                
                # Calculate metrics
                ssim = calculate_ssim(sr_img, hr_img)
                psnr = calculate_psnr(sr_img, hr_img)
                
                total_ssim += ssim
                total_psnr += psnr
                num_batches += 1
        
        return {
            'ssim': total_ssim / num_batches,
            'psnr': total_psnr / num_batches
        }
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save checkpoint including discriminator"""
        checkpoint = {
            'epoch': self.current_epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        # Save latest
        checkpoint_path = self.checkpoint_dir / 'esrgan_latest.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / 'esrgan_best.pth'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best ESRGAN model")