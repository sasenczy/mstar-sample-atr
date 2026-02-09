"""
ResNet-18 model adapted for single-channel SAR ATR.

Key modifications from the standard torchvision ResNet-18:
1. First convolutional layer accepts 1 input channel instead of 3
2. Final fully connected layer outputs num_classes instead of 1000
3. Optional dropout before the classifier for regularization
"""

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchvision import models


def build_model(cfg: DictConfig) -> nn.Module:
    """
    Build a ResNet-18 adapted for single-channel SAR input.

    If pretrained=True, we load ImageNet weights and then adapt the first
    conv layer by averaging the three RGB channel weights into a single
    channel. This preserves as much of the pretrained feature extraction
    capability as possible while accepting grayscale input.

    Args:
        cfg: Full Hydra config (expects cfg.model sub-config).

    Returns:
        A PyTorch nn.Module ready for training.
    """
    # Load base ResNet-18 (with or without pretrained ImageNet weights)
    if cfg.model.pretrained:
        print("Loading ImageNet-pretrained ResNet-18 weights...")
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        print("Initializing ResNet-18 from scratch (no pretraining)...")
        model = models.resnet18(weights=None)

    # ---------------------------------------------------------------
    # Adaptation 1: Modify first conv layer for single-channel input
    # ---------------------------------------------------------------
    # The original conv1 expects 3-channel (RGB) input:
    #   Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # We replace it with a 1-channel version.
    original_conv1 = model.conv1

    model.conv1 = nn.Conv2d(
        in_channels=1,  # Single-channel SAR input
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    if cfg.model.pretrained:
        # Average the 3 RGB channel weights into 1 channel weight.
        # Shape: [64, 3, 7, 7] → mean over dim=1 → [64, 1, 7, 7]
        # This is a common technique that works well in practice because
        # the averaged weights still capture edges and textures.
        with torch.no_grad():
            model.conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )

    # ---------------------------------------------------------------
    # Adaptation 2: Replace final FC layer for our number of classes
    # ---------------------------------------------------------------
    # Original: Linear(512, 1000) for ImageNet
    # New: Optional dropout + Linear(512, num_classes)
    num_features = model.fc.in_features  # 512 for ResNet-18

    model.fc = nn.Sequential(  # type: ignore[assignment]
        nn.Dropout(p=cfg.model.dropout),
        nn.Linear(num_features, cfg.model.num_classes),
    )

    return model


def count_parameters(model: nn.Module) -> int:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total:,} total, {trainable:,} trainable")
    return trainable
