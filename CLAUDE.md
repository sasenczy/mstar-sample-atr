# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MSTAR SAR Automatic Target Recognition (ATR) system. A PyTorch deep learning pipeline that classifies 10 target types in Synthetic Aperture Radar imagery using a ResNet-18 model adapted for single-channel SAR input. Supports two datasets:
- **MSTAR**: Standard Operating Conditions (SOC) protocol with pre-split train/test directories
- **SAMPLE**: Synthetic and Measured Paired Labeled Experiment — paired real/synthetic SAR images with elevation-based train/test split and configurable measured/synthetic mixing ratio (`k` parameter)

## Commands

### Training (MSTAR)
```bash
python train.py                                          # Run with default config (MSTAR)
python train.py training.lr=0.0005                       # Override a parameter
python train.py model.pretrained=False training.epochs=200  # Multiple overrides
```

### Training (SAMPLE)
```bash
python train.py dataset=sample                           # SAMPLE, k=1.0 (all measured)
python train.py dataset=sample dataset.k=0.5             # 50/50 measured/synthetic mix
python train.py dataset=sample dataset.k=0.0             # All synthetic training data
python train.py dataset=sample dataset.normalization=decibel  # Decibel normalization
```

### Evaluation
```bash
python src/evaluate.py --checkpoint outputs/best_model.pth
```

### Compute Dataset Statistics
```bash
python scripts/compute_stats.py              # MSTAR
python scripts/compute_sample_stats.py       # SAMPLE (both QPM and decibel)
```

### Install Dependencies
```bash
uv sync   # Uses uv with pyproject.toml and uv.lock
```

**No test suite, linter, or CI/CD pipeline is configured.**

## Architecture

### Entry Points
- **`train.py`** — Main entry point. Uses `@hydra.main` to load composed config from `configs/`, then orchestrates the full train/eval loop. Imports `hydra_zen` solely to patch Hydra for Python 3.14 compatibility.
- **`src/evaluate.py`** — Standalone evaluation. Loads a checkpoint (which embeds the config), rebuilds the model, and runs detailed per-class metrics + confusion matrix.

### Source Modules (`src/`)
- **`model.py`** — ResNet-18 adapted for SAR: first conv changed from 3→1 input channels (pretrained RGB weights averaged), final FC layer outputs `num_classes` (10), optional dropout before classifier. Factory: `build_model(cfg)`.
- **`dataset.py`** — Two dataset classes: `GrayscaleImageFolder` (MSTAR, directory-based split) and `SAMPLEDataset` (SAMPLE, elevation-based split with `k` parameter for measured/synthetic mixing). `build_dataloaders(cfg)` dispatches on `cfg.dataset.name` and returns `(train_loader, test_loader, class_names)`.
- **`engine.py`** — Training loop with early stopping (patience=20), model checkpointing, W&B logging, LR scheduling (cosine/step/none). Contains `train()`, `train_one_epoch()`, `evaluate()`, `plot_confusion_matrix()`, `build_optimizer()`, `build_scheduler()`.
- **`utils.py`** — `set_seed()` for reproducibility (random/numpy/torch/cuda deterministic). `get_device()` for auto device selection (CUDA → MPS → CPU).

### Configuration (`configs/`)
Hydra composes `config.yaml` from four sub-configs:
- `dataset/mstar.yaml` — data paths (`./data/train`, `./data/test`), image size (128), normalization stats (mean=0.1305, std=0.1250), augmentation params, dataloader settings
- `dataset/sample.yaml` — root dir (`./data/sample/png_images`), normalization type (qpm/decibel), k parameter (measured/synthetic ratio), test elevation (017), dataset-specific mean/std
- `model/resnet18.yaml` — pretrained flag, dropout (0.3), num_classes (10)
- `training/default.yaml` — epochs (100), optimizer (adam), lr (0.001), scheduler (cosine), early stopping patience (20)
- `logging/wandb.yaml` — W&B project name, logging frequency, confusion matrix logging

All config values are overridable via Hydra CLI syntax (e.g., `python train.py training.lr=0.01`).

### Data Flow
- **MSTAR**: images in `./data/{train,test}/` (ImageFolder layout) → `GrayscaleImageFolder` → transforms → DataLoaders
- **SAMPLE**: images in `./data/sample/png_images/{qpm,decibel}/{real,synth}/{class}/` → `SAMPLEDataset` (splits by elevation, mixes real/synth by `k`) → transforms → DataLoaders

Both paths → ResNet-18 (1-ch input, 10-class output) → CrossEntropyLoss → Adam/SGD optimizer → best checkpoint saved to `./outputs/best_model.pth`.

### Key Design Decisions
- **Single-channel adaptation**: SAR images are grayscale. When using pretrained ImageNet weights, the three RGB conv1 weight channels are averaged into one.
- **SAR-specific augmentation**: Random rotation (targets at arbitrary azimuth), random flips (no canonical orientation in SAR), additive Gaussian noise (approximates speckle variation).
- **Experiment tracking**: Weights & Biases integration throughout training and evaluation for metrics, confusion matrices, and model artifacts.
