#!/bin/bash
# MonoLayout Inference Script for Argoverse Dataset
# This script runs inference on test images using pretrained models

echo "=========================================="
echo "MonoLayout - Argoverse Inference"
echo "=========================================="
echo ""

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate monolayout

# Set paths
PROJECT_DIR="/home/detour/Documents/MobileRobotics/Project/monolayout_MR_M25_IIITH"
cd "$PROJECT_DIR"

echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"
echo ""

# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
echo ""

# Option 1: Static inference (Road layouts)
echo "=========================================="
echo "Option 1: Running STATIC inference (Roads)"
echo "=========================================="
python3 test.py \
    --type static \
    --model_path ./pretrained_models/argoverse_static/ \
    --image_path ./test_images/argoverse/ \
    --out_dir ./output/argoverse_static/ \
    --ext jpg

echo ""
echo "Static inference complete!"
echo "Results saved in: ./output/argoverse_static/static/"
echo ""

# Option 2: Dynamic inference (Vehicles)
echo "=========================================="
echo "Option 2: Running DYNAMIC inference (Vehicles)"
echo "=========================================="
python3 test.py \
    --type dynamic \
    --model_path ./pretrained_models/argoverse_dynamic/ \
    --image_path ./test_images/argoverse/ \
    --out_dir ./output/argoverse_dynamic/ \
    --ext jpg

echo ""
echo "Dynamic inference complete!"
echo "Results saved in: ./output/argoverse_dynamic/dynamic/"
echo ""

echo "=========================================="
echo "All inference tasks completed!"
echo "=========================================="
echo ""
echo "To view results:"
echo "  Static (roads):    ls ./output/argoverse_static/static/"
echo "  Dynamic (vehicles): ls ./output/argoverse_dynamic/dynamic/"
echo ""
