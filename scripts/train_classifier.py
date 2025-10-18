"""
Train brain tumor classifier on low-resolution images
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from pathlib import Path
import logging

from data.dataset import MRIDataset, get_train_transforms, get_test_transforms
from models.classifier import BrainTumorClassifier
from utils.logger import setup_logger
from utils.checkpoint import CheckpointManager

logger = logging.getLogger(__name__)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Skip invalid samples
        if (labels == -1).any():
            continue
        
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            if (labels == -1).any():
                continue
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy


def main(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logger(
        log_file='results/logs/classifier_training.log',
        level=logging.INFO
    )
    
    logger.info("Starting classifier training")
    logger.info(f"Config: {config_path}")
    
    # Device
    device = torch.device(config['experiment']['device'])
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
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=config['experiment']['num_workers'],
        pin_memory=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Testing samples: {len(test_dataset)}")
    
    # Model
    model = BrainTumorClassifier(
        num_classes=config['data']['num_classes'],
        pretrained=config['model']['classifier']['pretrained'],
        dropout=config['model']['classifier']['dropout']
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=config['training']['scheduler']['patience'],
        factor=config['training']['scheduler']['factor'],
        min_lr=config['training']['scheduler']['min_lr']
    )
    
    # Checkpoint manager
    checkpoint_manager = CheckpointManager(
        save_dir='results/models',
        max_checkpoints=5
    )
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    max_patience = config['training']['early_stopping']['patience']
    
    for epoch in range(config['training']['epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        
        # Validate
        val_loss, val_acc = validate(
            model, test_loader, criterion, device
        )
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Scheduler step
        scheduler.step(val_loss)
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
            logger.info(f"New best validation accuracy: {best_val_acc:.2f}%")
        else:
            patience_counter += 1
        
        checkpoint_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={'val_loss': val_loss, 'val_acc': val_acc},
            is_best=is_best,
            filename=f'classifier_epoch_{epoch+1}.pth'
        )
        
        # Early stopping
        if patience_counter >= max_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    logger.info(f"\nTraining completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain tumor classifier')
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to config file'
    )
    
    args = parser.parse_args()
    main(args.config)