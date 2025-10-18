"""
Run complete training and evaluation pipeline
"""

import sys
sys.path.append('.')

import subprocess
import logging
from pathlib import Path
import yaml

from utils.logger import setup_logger

logger = logging.getLogger(__name__)


def run_command(command: str, description: str):
    """Run a shell command with logging"""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {command}")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Command failed with error:\n{result.stderr}")
        raise RuntimeError(f"Failed at step: {description}")
    else:
        logger.info(f"✓ Completed: {description}")
    
    return result


def main():
    # Setup logging
    setup_logger(
        log_file='results/logs/full_pipeline.log',
        level=logging.INFO
    )
    
    logger.info("="*60)
    logger.info("TASK-AWARE MEDICAL DENOISING - FULL PIPELINE")
    logger.info("="*60)
    
    # Create directories
    Path('results/models').mkdir(parents=True, exist_ok=True)
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    Path('results/logs').mkdir(parents=True, exist_ok=True)
    Path('results/checkpoints').mkdir(parents=True, exist_ok=True)
    
    # Pipeline steps
    steps = [
        {
            'command': 'python scripts/train_classifier.py --config config/base_config.yaml',
            'description': 'Train Baseline Classifier'
        },
        {
            'command': 'python scripts/train_standard_esrgan.py --config config/standard_esrgan.yaml',
            'description': 'Train Standard ESRGAN'
        },
        {
            'command': 'python scripts/train_taskaware_esrgan.py --config config/taskaware_esrgan.yaml',
            'description': 'Train Task-Aware ESRGAN'
        },
        {
            'command': 'python scripts/evaluate_uncertainty.py --config config/base_config.yaml --model-type both',
            'description': 'Evaluate Both Models with Uncertainty'
        }
    ]
    
    # Execute pipeline
    for i, step in enumerate(steps, 1):
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"PIPELINE STEP {i}/{len(steps)}")
        logger.info(f"{'#'*60}")
        
        try:
            run_command(step['command'], step['description'])
        except RuntimeError as e:
            logger.error(f"Pipeline failed: {e}")
            return
    
    # Final summary
    logger.info(f"\n\n{'='*60}")
    logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info(f"{'='*60}")
    logger.info("\nResults saved to:")
    logger.info("  - Models: results/models/")
    logger.info("  - Plots: results/plots/")
    logger.info("  - Logs: results/logs/")
    logger.info("  - Checkpoints: results/checkpoints/")


if __name__ == '__main__':
    main()