# MonoLayout Installation Guide

## Quick Installation (Tested on 13 Nov 2025)

### Prerequisites
- Python 3.7 (recommended: use conda environment)
- CUDA 9.2 or higher (for GPU support)
- Linux OS

### Step 1: Create Conda Environment

```bash
conda create -n monolayout python=3.7
conda activate monolayout
```

### Step 2: Install PyTorch with CUDA Support

```bash
pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 3: Install Other Dependencies

```bash
pip install opencv-python pandas scikit-image scikit-learn scipy seaborn tqdm pykitti matplotlib
```

**OR** use the fixed requirements file:

```bash
pip install -r requirements_fixed.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import torchvision; import cv2; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available())"

python -c "from monolayout import Encoder, Decoder, Discriminator; print('MonoLayout modules loaded successfully')"
```

Expected output:
```
PyTorch: 1.6.0+cu92
CUDA Available: True
MonoLayout modules loaded successfully
```

## Installed Package Versions

- **PyTorch**: 1.6.0+cu92
- **Torchvision**: 0.7.0+cu92
- **OpenCV**: 4.12.0.88
- **Pandas**: 1.3.5
- **NumPy**: 1.21.6
- **Scikit-image**: 0.19.3
- **Scikit-learn**: 1.0.2
- **Scipy**: 1.7.3
- **Matplotlib**: 3.5.3
- **Seaborn**: 0.12.2
- **Tqdm**: 4.67.1
- **PyKitti**: 0.3.1

## Troubleshooting

### Issue: `pytorch==1.6.0` not found
**Solution**: The package name is `torch`, not `pytorch`. Use:
```bash
pip install torch==1.6.0 torchvision==0.7.0
```

### Issue: Python version errors
**Solution**: Make sure you're using Python 3.7:
```bash
conda activate monolayout
python --version  # Should show Python 3.7.16
```

### Issue: CUDA not available
**Solution**: Install CUDA-enabled PyTorch:
```bash
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

### Issue: Module import errors
**Solution**: Ensure you're in the correct directory and environment:
```bash
conda activate monolayout
cd /path/to/monolayout_MR_M25_IIITH
python -c "import monolayout"
```

## Next Steps

After successful installation:

1. **Download pretrained models** for Argoverse dataset
2. **Prepare test images** 
3. **Run inference** using `test.py`
4. **Evaluate models** using `eval.py`

See [README.md](README.md) for detailed usage instructions.

## Environment Summary

```
Environment: monolayout
Python: 3.7.16
CUDA: Available ✓
All packages: Installed ✓
```

Installation complete! 🎉
