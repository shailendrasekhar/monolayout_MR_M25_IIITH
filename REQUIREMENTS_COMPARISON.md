# Requirements Comparison: Original vs Fixed

## Summary
The original `requirements.txt` had several issues that prevented installation on modern systems. This document explains the changes made in `requirements_fixed.txt`.

## Fixed Issues

### 1. **Critical Fix: pytorch → torch**
- **Original**: `pytorch==1.6.0` ❌
- **Fixed**: `torch==1.6.0` ✅
- **Reason**: The package is named `torch`, not `pytorch`. This was causing installation to fail completely.

## Package Version Updates

The following packages were updated to compatible versions that work with Python 3.7:

| Package | Original Version | Fixed Version | Reason for Update |
|---------|-----------------|---------------|-------------------|
| opencv-python | 4.1.1.26 | 4.12.0.88 | Latest compatible with Python 3.7 |
| pandas | 0.24.2 | 1.3.5 | More stable, compatible version |
| Pillow | 6.2.0 | 9.5.0 | Security updates, compatible |
| numpy | (not specified) | 1.21.6 | Required dependency |
| scikit-image | 0.14.2 | 0.19.3 | Bug fixes, compatible |
| scikit-learn | 0.20.3 | 1.0.2 | More stable version |
| scipy | 1.2.3 | 1.7.3 | Bug fixes, compatible |
| seaborn | 0.9.0 | 0.12.2 | Updated for compatibility |
| matplotlib | (not specified) | 3.5.3 | Required for visualization |
| six | 1.14.0 | 1.17.0 | Latest stable |
| sympy | 1.5.1 | 1.10.1 | Compatible update |
| tqdm | 4.36.1 | 4.67.1 | Latest compatible |

## Packages NOT Installed (from original requirements.txt)

These packages were in the original requirements but are **not needed** for MonoLayout to work:

### 1. **pptk==0.1.0**
- **Purpose**: Point cloud visualization toolkit
- **Why not installed**: Optional dependency, not used in core MonoLayout functionality
- **Impact**: None on model inference or training

### 2. **TensorFlow packages** (Optional)
- `tensorflow==1.15.4`
- `tensorflow-estimator==1.14.0`
- `tensorboard==1.14.0`
- `protobuf==3.10.0`
- **Why not installed**: MonoLayout uses PyTorch. TensorFlow was only used for comparisons in the original paper.
- **Impact**: None. Uncomment in requirements_fixed.txt if you need TensorFlow for research comparisons.

### 3. **Unnecessary/Deprecated packages**
| Package | Why Not Needed |
|---------|---------------|
| SecretStorage==2.3.1 | Linux keyring library, not used by MonoLayout |
| Send2Trash==1.5.0 | File deletion utility, not used |
| service-identity==16.0.0 | Twisted library dependency, not needed |
| simplegeneric==0.8.1 | Deprecated, use `functools.singledispatch` instead |
| simplejson==3.13.2 | Standard `json` module is sufficient |
| singledispatch==3.4.0.3 | Built into Python 3.7+ standard library |
| sklearn==0.0 | Dummy package, `scikit-learn` is the real package |
| subprocess32==3.5.4 | Python 2 backport, not needed in Python 3.7+ |

## Installation Instructions

### Quick Install (Recommended)
```bash
conda activate monolayout
pip install -r requirements_fixed.txt
```

### Manual Install (Step by Step)
```bash
conda activate monolayout

# Install PyTorch with CUDA
pip install torch==1.6.0 torchvision==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install other packages
pip install opencv-python pandas scikit-image scikit-learn scipy seaborn tqdm pykitti matplotlib sympy
```

## Verification

After installation, verify everything works:

```bash
conda activate monolayout

# Check PyTorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Check MonoLayout modules
python -c "from monolayout import Encoder, Decoder, Discriminator; print('MonoLayout: OK')"
```

Expected output:
```
PyTorch: 1.6.0+cu92
CUDA: True
MonoLayout: OK
```

## Python Version Compatibility

- ✅ **Python 3.7**: Fully compatible (tested)
- ❌ **Python 3.8+**: Some packages like older PyTorch versions may have issues
- ❌ **Python 3.6 or below**: Not supported

## Notes

1. All updated versions maintain backward compatibility with the original MonoLayout code
2. The fixed requirements are tested and working as of November 13, 2025
3. CUDA 9.2 support is included in the PyTorch installation
4. If you need CPU-only version, modify the torch installation command

## For Developers

If you encounter any issues:
1. Make sure you're using Python 3.7 in a conda environment
2. Verify CUDA is properly installed if using GPU
3. Check that you're using `torch` not `pytorch`
4. See `INSTALLATION.md` for detailed troubleshooting

---

**Last Updated**: November 13, 2025  
**Tested On**: Ubuntu with Python 3.7.16, CUDA 9.2
