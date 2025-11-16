#!/bin/bash
# MonoLayout Inference Script - CPU Mode
# For systems with incompatible CUDA/GPU

echo "=========================================="
echo "MonoLayout - Argoverse Inference (CPU Mode)"
echo "=========================================="
echo ""
echo "NOTE: Running on CPU (slower but compatible)"
echo "Estimated time: ~2-5 seconds per image"
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

# Check PyTorch
python -c "import torch; print('PyTorch version:', torch.__version__); print('Using device: CPU')"
echo ""

# Count images
NUM_IMAGES=$(ls test_images/argoverse/*.jpg 2>/dev/null | wc -l)
echo "Found $NUM_IMAGES test images"
echo ""

# Option: Choose which inference to run
echo "Select inference type:"
echo "  1) Static only (Roads)"
echo "  2) Dynamic only (Vehicles)"  
echo "  3) Both (default)"
echo ""
read -p "Enter choice [1/2/3] (default=3): " CHOICE
CHOICE=${CHOICE:-3}

if [ "$CHOICE" = "1" ] || [ "$CHOICE" = "3" ]; then
    echo ""
    echo "=========================================="
    echo "Running STATIC inference (Roads)"
    echo "=========================================="
    python3 test.py \
        --type static \
        --model_path ./pretrained_models/argoverse_static/ \
        --image_path ./test_images/argoverse/ \
        --out_dir ./output/argoverse_static/ \
        --ext jpg

    echo ""
    echo "✓ Static inference complete!"
    echo "  Results: ./output/argoverse_static/static/"
    echo ""
fi

if [ "$CHOICE" = "2" ] || [ "$CHOICE" = "3" ]; then
    echo ""
    echo "=========================================="
    echo "Running DYNAMIC inference (Vehicles)"
    echo "=========================================="
    python3 test.py \
        --type dynamic \
        --model_path ./pretrained_models/argoverse_dynamic/ \
        --image_path ./test_images/argoverse/ \
        --out_dir ./output/argoverse_dynamic/ \
        --ext jpg

    echo ""
    echo "✓ Dynamic inference complete!"
    echo "  Results: ./output/argoverse_dynamic/dynamic/"
    echo ""
fi

echo "=========================================="
echo "All inference tasks completed!"
echo "=========================================="
echo ""
echo "View results:"
if [ "$CHOICE" = "1" ] || [ "$CHOICE" = "3" ]; then
    echo "  Static (roads):    ls ./output/argoverse_static/static/"
fi
if [ "$CHOICE" = "2" ] || [ "$CHOICE" = "3" ]; then
    echo "  Dynamic (vehicles): ls ./output/argoverse_dynamic/dynamic/"
fi
echo ""
