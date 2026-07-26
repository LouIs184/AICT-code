# AICTVer2 — AI CT Image Reconstruction (Refactored)

## Overview

AICTVer2 is a **refactored** version of the [AICT](https://github.com/your-org/AICT) project,
focused on deep-learning-based CT image reconstruction. It uses **DenseUNet2d** and **UNet2d**
architectures to denoise/enhance low-dose CT (LDCT) images.

This refactor follows the recommendations in `docs/ARCHITECTURE_ANALYSIS.md` to eliminate
code duplication, improve modularity, and introduce a proper configuration system.

## Key Improvements

| Issue | Original | AICTVer2 |
|-------|----------|----------|
| Code duplication | 9 near-identical training scripts | 1 unified `Trainer` |
| Package structure | Flat root directory (20+ files) | `src/` package with submodules |
| Configuration | Hardcoded hyperparameters | YAML config files |
| Model I/O & metrics | Mixed in `utils.py` | Split: `io_utils.py`, `metrics.py` |
| Data augmentation | In `utils.py` | Separated `augments.py` |

## Project Structure

```
AICTVer2/
├── src/                          # Core Python package
│   ├── __init__.py
│   ├── config.py                 # Configuration dataclasses + YAML loader
│   ├── trainer.py                # Unified training engine
│   ├── models/                   # Neural network architectures
│   │   ├── __init__.py
│   │   ├── blocks.py             # ConvBnRelu2d, StackEncoder/Decoder, Dense blocks
│   │   ├── unet.py               # UNet2d, DenseUNet2d, UNet2dDeepSuperviser
│   │   └── discriminator.py      # Discriminator2D, compute_gradient_penalty
│   ├── losses/                   # Loss functions
│   │   ├── __init__.py
│   │   ├── regularizers.py       # CharbonnierLoss, TVLoss
│   │   ├── perceptual.py         # VGG perceptual loss
│   │   └── ssim.py               # SSIM metric
│   ├── data/                     # Data loaders
│   │   ├── __init__.py
│   │   ├── raw_reader.py         # PhaseCT raw volume reader
│   │   ├── dicom_reader.py       # DICOM file reader
│   │   └── proj_reader.py        # ProjCT projection reader
│   ├── utils/                    # Utility modules
│   │   ├── __init__.py
│   │   ├── io_utils.py           # Model save/load
│   │   ├── metrics.py            # AverageMeter, normalization, PSNR
│   │   ├── augments.py           # MixUp, CutMix augmentations
│   │   └── viz.py                # Gaussian smooth, patch extraction
│   └── optimizers/               # Optimizers
│       ├── __init__.py
│       └── sophia.py             # SophiaG optimizer
├── experiments/                  # YAML configuration files
│   ├── denseunet_phasect.yaml    # PhaseCT baseline
│   ├── denseunet_projct.yaml     # ProjCT baseline
│   ├── denseunet_mixup.yaml      # MixUp augmentation
│   ├── unet_cutmix.yaml          # CutMix augmentation
│   ├── unet_gan.yaml             # WGAN-GP adversarial training
│   ├── unet_vgg.yaml             # VGG perceptual loss
│   ├── unet_deepsup.yaml         # Deep supervision
│   └── denseunet_sophia.yaml     # SophiaG optimizer
├── docs/                         # Documentation
├── results/                      # Inference outputs
├── results_proj/                 # Projection-domain inference outputs
├── runs/                         # Training checkpoints + logs
├── train.py                      # Entry point: training
├── test.py                       # Entry point: inference
├── prepare_data.py               # Generate data index files (PhaseCT)
├── prepare_proj_data.py          # Generate data index files (ProjCT)
├── mk_data_list.py               # Generate DICOM file pairs (legacy)
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Usage

### Training

```bash
# PhaseCT (image domain) — DenseUNet2d
python train.py --config experiments/denseunet_phasect.yaml

# Projection domain
python train.py --config experiments/denseunet_projct.yaml

# With MixUp augmentation
python train.py --config experiments/denseunet_mixup.yaml

# GAN training
python train.py --config experiments/unet_gan.yaml

# Resume from latest checkpoint
python train.py --config experiments/denseunet_phasect.yaml --resume

# Override parameters via CLI
python train.py --config experiments/denseunet_phasect.yaml --epochs 300 --lr 0.0005
```

### Inference

```bash
# PhaseCT inference
python test.py --config experiments/denseunet_phasect.yaml \\
    --checkpoint ./runs/DenseUNet_PhaseCT/checkpoints/model_at_epoch_200.dat \\
    --data-dir /path/to/data --output ./results/

# ProjCT inference
python test.py --config experiments/denseunet_projct.yaml \\
    --checkpoint ./runs/DenseUNet_ProjCT/checkpoints/model_at_epoch_200.dat \\
    --data-dir /path/to/data --output ./results_proj/ --domain proj
```

### Data Preparation

```bash
# Generate image-domain data indices
python prepare_data.py --data-dir /path/to/UIH_data --output ./

# Generate projection-domain data indices
python prepare_proj_data.py --data-dir /path/to/UIH_data --output ./
```

## Data Domains

| Domain | Input → Output | Data Shape | Normalization |
|--------|---------------|------------|--------------|
| **PhaseCT** (image) | `rec_pbi_*.raw` → `rec_no_pbi.raw` | (160, 512, 512) | HU [-1000, +1000] → [0, 1] |
| **ProjCT** (projection) | `proj_pbi_*.raw` → `proj_no_pbi.raw` | (720, 128, 512) | Clip [0, 0.28] → [0, 1] |

## Migration from Original AICT

If you have existing checkpoints from the original AICT project:

```python
# Original checkpoints are directly compatible
from src.models import DenseUNet2d
from src.utils.io_utils import load_model

model = DenseUNet2d()
checkpoint = torch.load("path/to/model_at_epoch_200.dat")
load_model(model, checkpoint)
```

## License

MIT
