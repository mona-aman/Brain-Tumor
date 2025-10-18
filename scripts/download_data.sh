#!/bin/bash

# Download datasets script

echo "================================================"
echo "Downloading Brain Tumor MRI Datasets"
echo "================================================"

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null
then
    echo "Error: Kaggle CLI not found. Please install with: pip install kaggle"
    echo "Also configure your Kaggle API credentials: ~/.kaggle/kaggle.json"
    exit 1
fi

# Create data directories
mkdir -p data/primary
mkdir -p data/crossval

echo ""
echo "Downloading primary dataset..."
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset
unzip -q brain-tumor-mri-dataset.zip -d data/primary/
rm brain-tumor-mri-dataset.zip
echo "✓ Primary dataset downloaded"

echo ""
echo "Downloading cross-validation dataset..."
kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection
unzip -q brain-mri-images-for-brain-tumor-detection.zip -d data/crossval/
rm brain-mri-images-for-brain-tumor-detection.zip
echo "✓ Cross-validation dataset downloaded"

echo ""
echo "================================================"
echo "Download completed!"
echo "================================================"
echo "Primary dataset: data/primary/"
echo "Cross-val dataset: data/crossval/"