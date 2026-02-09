# scripts/compute_stats.py
"""
Compute the per-channel mean and standard deviation of the MSTAR training set.
Run once and paste the results into configs/dataset/mstar.yaml.
"""

import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def compute_dataset_statistics(data_dir: str) -> tuple[float, float]:
    """
    Compute mean and std of all images in a directory tree.

    We accumulate running sums rather than loading everything into memory,
    which keeps this script usable even on machines with limited RAM.

    Args:
        data_dir: Root directory containing class subfolders with images.

    Returns:
        Tuple of (mean, std) for pixel values scaled to [0, 1].
    """
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0

    # Walk through all class subfolders
    for class_name in sorted(os.listdir(data_dir)):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in tqdm(os.listdir(class_dir), desc=class_name):
            fpath = os.path.join(class_dir, fname)
            # Open as grayscale, convert to float in [0, 1]
            img = np.array(Image.open(fpath).convert("L")) / 255.0
            pixel_sum += img.sum()
            pixel_sq_sum += (img**2).sum()
            num_pixels += img.size

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean**2)
    return mean, std


if __name__ == "__main__":
    train_dir = "./data/train"
    mean, std = compute_dataset_statistics(train_dir)
    print("Dataset statistics (paste into configs/dataset/mstar.yaml):")
    print(f"  mean: [{mean:.4f}]")
    print(f"  std:  [{std:.4f}]")
