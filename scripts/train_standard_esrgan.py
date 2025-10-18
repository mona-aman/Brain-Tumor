"""
Train standard ESRGAN (without task-aware component)
"""

import sys
sys.path.append('.')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
import logging

from data.dataset import MRIDataset, get_train_transforms, get_test_transforms
from models.esrgan import Generator
from models.discriminator import Discriminator
from trainers.esrgan_trainer import ESRGANTrainer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def main(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logger(
        log_file='results/logs/standard_esrgan_training.log',
        level=logging.INFO
    )
    
    logger.info("Starting Standard ESRGAN training")
    logger.info(f"Config: {config_path}")
    
    # Device
    # device = torch.device(config['experiment']['device'])
    device = torch.device(config.get('experiment', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    logger.info(f"Using device: {device}")
    
    # Datasets
    train_dataset = MRIDataset(
        root_dir=config['data']['primary_dataset']['root'],
        split=config['data']['primary_dataset']['train_split'],
        transform=get_train_transforms(config['data']['image_size'])
    )
    
    test_dataset = MRIDataset(
        root_dir=config['data']['primary_dataset']['root'],
        split=config['data']['primary_dataset']['test_split'],
        transform=get_test_transforms(config['data']['image_size'])
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        num_workers = config.get('experiment', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers = config.get('experiment', {}).get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    
    # Models
    generator = Generator(
        in_channels=3,
        out_channels=3,
        num_features=config['model']['generator']['num_features'],
        num_blocks=config['model']['generator']['num_blocks'],
        scale_factor=4
    ).to(device)
    
    discriminator = Discriminator(
        in_channels=3,
        num_features=config['model']['discriminator']['num_features']
    ).to(device)
    
    logger.info(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    logger.info(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    
    # Optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=config['training']['learning_rate'],
        betas=config['training']['betas'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Trainer
    trainer = ESRGANTrainer(
        generator=generator,
        discriminator=discriminator,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=device,
        config=config
    )
    
    # Training loop
    best_ssim = 0.0
    
    for epoch in range(config['training']['epochs']):
        trainer.current_epoch = epoch
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{config['training']['epochs']}")
        logger.info(f"{'='*50}")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        logger.info(f"Train - G Loss: {train_metrics['g_loss']:.4f} | D Loss: {train_metrics['d_loss']:.4f}")
        
        # Validate
        val_metrics = trainer.validate(test_loader)
        logger.info(f"Val - SSIM: {val_metrics['ssim']:.4f} | PSNR: {val_metrics['psnr']:.2f}")
        
        # Save checkpoint
        is_best = val_metrics['ssim'] > best_ssim
        if is_best:
            best_ssim = val_metrics['ssim']
            logger.info(f"New best SSIM: {best_ssim:.4f}")
        
        trainer.save_checkpoint(
            metrics={**train_metrics, **val_metrics},
            is_best=is_best
        )
        
        # Early stopping check
        if trainer.check_early_stopping(1 - val_metrics['ssim']):
            logger.info("Early stopping triggered")
            break
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best SSIM: {best_ssim:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Standard ESRGAN')
    parser.add_argument(
        '--config',
        type=str,
        default='config/standard_esrgan.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args.config)