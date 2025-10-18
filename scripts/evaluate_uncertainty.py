"""
Evaluate models with uncertainty quantification - FULLY FIXED VERSION
"""

import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
import argparse
import numpy as np
import logging
from pathlib import Path

from data.dataset import MRIDataset, get_test_transforms
from models.esrgan import Generator
from models.taskaware_esrgan import TaskAwareGenerator
from models.classifier import BrainTumorClassifier
from evaluation.uncertainty_analysis import UncertaintyAnalyzer
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def enable_dropout(model):
    """
    Enable dropout layers while keeping BatchNorm in eval mode
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def evaluate_model_with_uncertainty(
    generator_path: str,
    classifier: BrainTumorClassifier,
    test_loader: DataLoader,
    device: str,
    is_taskaware: bool = False,
    num_mc_samples: int = 20
) -> dict:
    """
    Evaluate with proper MC Dropout - FULLY FIXED
    """
    
    # Load generator
    if is_taskaware:
        generator = TaskAwareGenerator(
            classifier=classifier,
            freeze_classifier=True
        ).to(device)
    else:
        generator = Generator().to(device)
    
    checkpoint = torch.load(generator_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    logger.info(f"Loaded generator from {generator_path}")
    
    # Storage for results
    all_predictions = []
    all_labels = []
    all_uncertainties = []
    all_confidences = []
    
    # Process each batch
    classifier.eval()
    
    for batch_idx, (hr_img, labels) in enumerate(test_loader):
        hr_img = hr_img.to(device)
        labels = labels.to(device)
        
        # Skip invalid batches
        if (labels == -1).any():
            continue
        
        # Generate SR images
        with torch.no_grad():
            lr_img = F.interpolate(
                hr_img,
                scale_factor=0.25,
                mode='bilinear',
                align_corners=False
            )
            sr_img = generator(lr_img, target_size=hr_img.shape[2:])
        
        # MC Dropout predictions for THIS batch
        enable_dropout(classifier)
        
        mc_predictions = []
        with torch.no_grad():
            for _ in range(num_mc_samples):
                logits = classifier(sr_img)
                probs = F.softmax(logits, dim=1)
                mc_predictions.append(probs)
        
        # Compute uncertainty for THIS batch
        mc_predictions = torch.stack(mc_predictions)  # [num_samples, batch, classes]
        mean_probs = mc_predictions.mean(dim=0)  # [batch, classes]
        
        # Predictions
        confidence, prediction = mean_probs.max(dim=1)
        
        # Uncertainty: predictive entropy
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=1)
        
        # Store results for THIS batch
        all_predictions.append(prediction.cpu())
        all_labels.append(labels.cpu())
        all_uncertainties.append(entropy.cpu())
        all_confidences.append(confidence.cpu())
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Restore full eval mode
    classifier.eval()
    
    # Concatenate ALL batches
    predictions = torch.cat(all_predictions).numpy()
    labels = torch.cat(all_labels).numpy()
    uncertainties = torch.cat(all_uncertainties).numpy()
    confidences = torch.cat(all_confidences).numpy()
    
    logger.info(f"Total samples processed: {len(predictions)}")
    
    return {
        'predictions': predictions,
        'labels': labels,
        'uncertainties': uncertainties,
        'confidences': confidences
    }


def main(config_path: str, model_type: str):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    setup_logger(
        log_file='results/logs/uncertainty_evaluation.log',
        level=logging.INFO
    )
    
    logger.info("Starting Uncertainty Evaluation - FULLY FIXED")
    logger.info(f"Model type: {model_type}")
    
    # Device
    device = torch.device(config['experiment']['device'])
    
    # Load classifier
    classifier = BrainTumorClassifier(
        num_classes=config['data']['num_classes'],
        dropout=config['model']['classifier']['dropout']
    ).to(device)
    
    classifier_checkpoint = torch.load(
        'results/models/checkpoint_best.pth',
        map_location=device
    )
    classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
    logger.info("Loaded classifier")
    
    # Test dataset
    test_dataset = MRIDataset(
        root_dir=config['data']['primary_dataset']['root'],
        split=config['data']['primary_dataset']['test_split'],
        transform=get_test_transforms(config['data']['image_size'])
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        num_workers=min(config['experiment']['num_workers'], 2),
        drop_last=False  # Don't drop last batch!
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} images")
    logger.info(f"Number of batches: {len(test_loader)}")
    
    # Initialize analyzer
    analyzer = UncertaintyAnalyzer(save_dir='results/plots')
    
    # Evaluate models
    if model_type in ['standard', 'both']:
        logger.info("\n" + "="*50)
        logger.info("Evaluating Standard ESRGAN")
        logger.info("="*50)
        
        standard_results = evaluate_model_with_uncertainty(
            generator_path='results/checkpoints/Standared_esrgan_best.pth',
            classifier=classifier,
            test_loader=test_loader,
            device=device,
            is_taskaware=False,
            num_mc_samples=20
        )
        
        analyzer.add_results(
            name='Standard ESRGAN',
            **standard_results
        )
        
        accuracy = np.mean(standard_results['predictions'] == standard_results['labels'])
        logger.info(f"✓ Standard ESRGAN Accuracy: {accuracy*100:.2f}%")
    
    if model_type in ['taskaware', 'both']:
        logger.info("\n" + "="*50)
        logger.info("Evaluating Task-Aware ESRGAN")
        logger.info("="*50)
        
        taskaware_results = evaluate_model_with_uncertainty(
            generator_path='results/checkpoints/esrgan_best.pth',
            classifier=classifier,
            test_loader=test_loader,
            device=device,
            is_taskaware=True,
            num_mc_samples=20
        )
        
        analyzer.add_results(
            name='Task-Aware ESRGAN',
            **taskaware_results
        )
        
        accuracy = np.mean(taskaware_results['predictions'] == taskaware_results['labels'])
        logger.info(f"✓ Task-Aware ESRGAN Accuracy: {accuracy*100:.2f}%")
    
    # Generate comparison
    logger.info("\n" + "="*50)
    logger.info("Generating Comparison Analysis")
    logger.info("="*50)
    
    comparison = analyzer.compare_models(threshold=0.5)
    
    # Print comparison
    for name, metrics in comparison.items():
        logger.info(f"\n{name}:")
        logger.info("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    # Generate plots
    logger.info("\nGenerating plots...")
    try:
        analyzer.generate_all_plots(
            class_names=config['data']['classes']
        )
        analyzer.save_comparison_table()
        logger.info("✓ All plots and tables generated successfully!")
    except Exception as e:
        logger.warning(f"Some plots failed (probably due to edge cases): {e}")
        logger.info("But main results are available above")
    
    logger.info("\nEvaluation completed!")
    logger.info("Results saved to results/plots/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate with Uncertainty')
    parser.add_argument(
        '--config',
        type=str,
        default='config/base_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['standard', 'taskaware', 'both'],
        default='both',
        help='Which model(s) to evaluate'
    )
    
    args = parser.parse_args()
    main(args.config, args.model_type)