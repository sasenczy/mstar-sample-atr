"""
MSTAR and SAMPLE dataset loading, preprocessing, and augmentation.

This module provides:
- Transform pipelines for training (with augmentation) and testing (deterministic)
- GrayscaleImageFolder for MSTAR single-channel SAR image loading
- SAMPLEDataset for the SAMPLE dataset with elevation-based train/test split
  and configurable measured/synthetic mixing ratio (k parameter)
- A factory function for creating DataLoaders from Hydra config
"""

import os
import random
import re

import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class AddGaussianNoise:
    """
    Custom transform that adds zero-mean Gaussian noise to a tensor.

    This is a simple approximation of speckle variation. Real SAR speckle is
    multiplicative and follows a different distribution, but additive Gaussian
    noise is a standard augmentation baseline that helps regularize training.

    Args:
        std: Standard deviation of the Gaussian noise.
    """

    def __init__(self, std: float = 0.02):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            return tensor + noise
        return tensor

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class GrayscaleImageFolder(datasets.ImageFolder):
    """
    ImageFolder subclass that forces all images to single-channel grayscale.

    torchvision's default ImageFolder opens images in RGB mode, which triples
    memory usage and is incorrect for SAR data. This override ensures every
    image is loaded as a grayscale ('L' mode) PIL image.
    """

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        # Load directly as grayscale to avoid RGB conversion overhead
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


# Regex for SAMPLE filenames, e.g.:
# 2s1_real_A_elevDeg_015_azCenter_010_22_serial_b01.png
_SAMPLE_FILENAME_RE = re.compile(
    r"(?P<target>\w+?)_(?P<domain>real|synth)_A_elevDeg_(?P<elevation>\d{3})_azCenter_(?P<azimuth>\d+)_\d+_serial_\w+\.png"
)


def parse_sample_filename(filename: str) -> dict[str, str] | None:
    """Parse metadata from a SAMPLE PNG filename.

    Returns a dict with keys: target, domain ('real'/'synth'), elevation, azimuth.
    Returns None if the filename doesn't match the expected pattern.
    """
    m = _SAMPLE_FILENAME_RE.match(filename)
    if m is None:
        return None
    return m.groupdict()


class SAMPLEDataset(Dataset):
    """SAMPLE dataset with elevation-based train/test split and measured/synthetic mixing.

    Split protocol (per sample_public.pdf):
    - Test:  all measured (real) images at the test elevation angle
    - Train: remaining non-test-elevation images, with k controlling the
             fraction of measured vs synthetic images per class

    Args:
        root_dir: Path to png_images directory (contains qpm/ and decibel/ subdirs).
        normalization: "qpm" or "decibel".
        split: "train" or "test".
        k: Measured/synthetic mixing ratio for training (1.0 = all measured,
           0.0 = all synthetic). Ignored for test split.
        test_elevation: 3-digit elevation string to withhold for testing (e.g. "017").
        transform: Optional torchvision transform pipeline.
        seed: Random seed for reproducible subset selection.
    """

    def __init__(
        self,
        root_dir: str,
        normalization: str,
        split: str,
        k: float = 1.0,
        test_elevation: str = "017",
        transform=None,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.normalization = normalization
        self.split = split
        self.k = k
        self.test_elevation = test_elevation
        self.transform = transform

        real_dir = os.path.join(root_dir, normalization, "real")
        synth_dir = os.path.join(root_dir, normalization, "synth")

        # Build class list from real directory (sorted for consistent indexing)
        self.classes = sorted(
            d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Scan and partition files by class, domain, and elevation
        real_train_pool: dict[str, list[str]] = {cls: [] for cls in self.classes}
        real_test_pool: dict[str, list[str]] = {cls: [] for cls in self.classes}
        synth_train_pool: dict[str, list[str]] = {cls: [] for cls in self.classes}

        for cls in self.classes:
            # Scan real images
            cls_real_dir = os.path.join(real_dir, cls)
            for fname in sorted(os.listdir(cls_real_dir)):
                meta = parse_sample_filename(fname)
                if meta is None:
                    continue
                filepath = os.path.join(cls_real_dir, fname)
                if meta["elevation"] == test_elevation:
                    real_test_pool[cls].append(filepath)
                else:
                    real_train_pool[cls].append(filepath)

            # Scan synthetic images (exclude test elevation per paper protocol)
            cls_synth_dir = os.path.join(synth_dir, cls)
            for fname in sorted(os.listdir(cls_synth_dir)):
                meta = parse_sample_filename(fname)
                if meta is None:
                    continue
                if meta["elevation"] == test_elevation:
                    continue
                synth_train_pool[cls].append(os.path.join(cls_synth_dir, fname))

        # Build final sample list based on split
        self.samples: list[tuple[str, int]] = []

        if split == "test":
            for cls in self.classes:
                idx = self.class_to_idx[cls]
                for path in real_test_pool[cls]:
                    self.samples.append((path, idx))
        elif split == "train":
            rng = random.Random(seed)
            for cls in self.classes:
                idx = self.class_to_idx[cls]
                real_pool = real_train_pool[cls]
                synth_pool = synth_train_pool[cls]
                pool_size = len(real_pool)

                n_measured = round(k * pool_size)
                n_synthetic = pool_size - n_measured

                # Sample measured images
                measured = rng.sample(real_pool, min(n_measured, len(real_pool)))
                # Sample synthetic images
                synthetic = rng.sample(synth_pool, min(n_synthetic, len(synth_pool)))

                for path in measured:
                    self.samples.append((path, idx))
                for path in synthetic:
                    self.samples.append((path, idx))
        else:
            raise ValueError(f"Unknown split: {split!r}. Expected 'train' or 'test'.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, target = self.samples[index]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def build_train_transforms(cfg: DictConfig) -> transforms.Compose:
    """
    Build the training transform pipeline with data augmentation.

    The pipeline:
    1. Resize to a consistent dimension (handles slight size variation in chips)
    2. Random augmentations (rotation, flips) — physically meaningful for SAR
    3. Convert to tensor (scales [0,255] -> [0.0, 1.0])
    4. Normalize with dataset mean/std
    5. Optional additive Gaussian noise

    Args:
        cfg: Full Hydra config (expects cfg.dataset sub-config).

    Returns:
        A composed transform pipeline.
    """
    transform_list: list = [
        transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
    ]

    # --- Data augmentation (SAR-appropriate) ---
    aug = cfg.dataset.augmentation

    # Random rotation: vehicles appear at all azimuth angles in SAR imagery,
    # so rotation is a physically justified augmentation.
    if aug.random_rotation > 0:
        transform_list.append(
            transforms.RandomRotation(
                degrees=aug.random_rotation,
                fill=0,  # Fill border pixels with black (low backscatter)
            )
        )

    # Random flips: horizontal and vertical flips are valid because SAR
    # images don't have a fixed "up" direction (azimuth varies).
    if aug.horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if aug.vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    # Convert PIL image to tensor (also scales to [0, 1])
    transform_list.append(transforms.ToTensor())

    # Normalize using training set statistics
    transform_list.append(
        transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std)
    )

    # Optional additive Gaussian noise (applied after normalization)
    if aug.gaussian_noise_std > 0:
        transform_list.append(AddGaussianNoise(std=aug.gaussian_noise_std))

    return transforms.Compose(transform_list)


def build_test_transforms(cfg: DictConfig) -> transforms.Compose:
    """
    Build the test/validation transform pipeline (deterministic, no augmentation).

    Args:
        cfg: Full Hydra config (expects cfg.dataset sub-config).

    Returns:
        A composed transform pipeline.
    """
    return transforms.Compose(
        [
            transforms.Resize((cfg.dataset.image_size, cfg.dataset.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.dataset.mean, std=cfg.dataset.std),
        ]
    )


def build_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, list[str]]:
    """
    Create training and test DataLoaders from Hydra config.

    Dispatches on cfg.dataset.name:
    - "mstar": loads from train_dir/test_dir using GrayscaleImageFolder
    - "sample": loads from root_dir using SAMPLEDataset with elevation-based
      train/test split and measured/synthetic mixing ratio k

    Args:
        cfg: Full Hydra config.

    Returns:
        Tuple of (train_loader, test_loader, class_names) where class_names
        is a list of the 10 target class names in alphabetical order
        (matching ImageFolder's default class-to-index mapping).
    """
    train_transforms = build_train_transforms(cfg)
    test_transforms = build_test_transforms(cfg)

    if cfg.dataset.name == "sample":
        train_dataset = SAMPLEDataset(
            root_dir=cfg.dataset.root_dir,
            normalization=cfg.dataset.normalization,
            split="train",
            k=cfg.dataset.k,
            test_elevation=cfg.dataset.test_elevation,
            transform=train_transforms,
            seed=cfg.seed,
        )
        test_dataset = SAMPLEDataset(
            root_dir=cfg.dataset.root_dir,
            normalization=cfg.dataset.normalization,
            split="test",
            test_elevation=cfg.dataset.test_elevation,
            transform=test_transforms,
            seed=cfg.seed,
        )
        print(
            f"SAMPLE dataset | normalization: {cfg.dataset.normalization} | "
            f"k: {cfg.dataset.k} | test elevation: {cfg.dataset.test_elevation}°"
        )
    else:
        # MSTAR: load from pre-split train/test directories
        train_dataset = GrayscaleImageFolder(
            root=cfg.dataset.train_dir,
            transform=train_transforms,
        )
        test_dataset = GrayscaleImageFolder(
            root=cfg.dataset.test_dir,
            transform=test_transforms,
        )

    # Extract class names (sorted alphabetically by ImageFolder convention)
    class_names = train_dataset.classes
    print(
        f"Found {len(train_dataset)} training images across {len(class_names)} classes"
    )
    print(f"Found {len(test_dataset)} test images")
    print(f"Classes: {class_names}")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=True,  # Shuffle training data each epoch
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,  # Drop incomplete last batch for stable batch norm
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=False,  # Deterministic order for evaluation
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=False,  # Evaluate every sample
    )

    return train_loader, test_loader, class_names
