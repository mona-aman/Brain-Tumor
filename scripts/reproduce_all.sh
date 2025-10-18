#!/bin/bash

# Script to reproduce all experiments

echo "================================================"
echo "TASK-AWARE MEDICAL DENOISING"
echo "Reproducing All Experiments"
echo "================================================"

# Set error handling
set -e

# Create necessary directories
mkdir -p results/models
mkdir -p results/plots
mkdir -p results/logs
mkdir -p results/checkpoints

echo ""
echo "Step 1/4: Training Baseline Classifier"
echo "----------------------------------------"
python scripts/train_classifier.py --config config/base_config.yaml

echo ""
echo "Step 2/4: Training Standard ESRGAN"
echo "----------------------------------------"
python scripts/train_standard_esrgan.py --config config/standard_esrgan.yaml

echo ""
echo "Step 3/4: Training Task-Aware ESRGAN"
echo "----------------------------------------"
python scripts/train_taskaware_esrgan.py --config config/taskaware_esrgan.yaml

echo ""
echo "Step 4/4: Evaluating with Uncertainty Quantification"
echo "----------------------------------------"
python scripts/evaluate_uncertainty.py --config config/base_config.yaml --model-type both

echo ""
echo "================================================"
echo "All experiments completed successfully!"
echo "================================================"
echo "Results saved to:"
echo "  - Models: results/models/"
echo "  - Plots: results/plots/"
echo "  - Logs: results/logs/"