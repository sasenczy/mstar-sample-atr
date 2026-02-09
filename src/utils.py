"""
Utility functions for reproducibility and common operations.
"""

import os
import random

import numpy as np
import torch
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    """
    Set random seeds across all libraries for reproducibility.

    Note: Full determinism also requires setting CUBLAS_WORKSPACE_CONFIG
    and using torch.use_deterministic_algorithms(True), which can significantly
    slow down training. The seed setting below provides strong reproducibility
    for most practical purposes.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (slight performance cost)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set to {seed}")


def get_device() -> torch.device:
    """Select the best available device (CUDA GPU > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    return device
