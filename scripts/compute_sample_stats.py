# scripts/compute_sample_stats.py
"""
Compute per-channel mean and standard deviation of the SAMPLE training partition.

Uses only real (measured) images at non-17° elevation angles, matching the
training set definition from sample_public.pdf. Computes stats for both
QPM and decibel normalizations.

Run once and paste the results into configs/dataset/sample.yaml.
"""

import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

# Add project root to path so we can import the filename parser
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import parse_sample_filename


def compute_sample_statistics(
    root_dir: str, normalization: str, test_elevation: str = "017"
) -> tuple[float, float]:
    """
    Compute mean and std of SAMPLE training partition (real, non-test-elevation).

    Args:
        root_dir: Path to png_images directory.
        normalization: "qpm" or "decibel".
        test_elevation: Elevation string to exclude (test set).

    Returns:
        Tuple of (mean, std) for pixel values scaled to [0, 1].
    """
    real_dir = os.path.join(root_dir, normalization, "real")
    pixel_sum = 0.0
    pixel_sq_sum = 0.0
    num_pixels = 0
    num_images = 0

    for class_name in sorted(os.listdir(real_dir)):
        class_dir = os.path.join(real_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for fname in tqdm(
            sorted(os.listdir(class_dir)), desc=f"{normalization}/{class_name}"
        ):
            meta = parse_sample_filename(fname)
            if meta is None:
                continue
            # Skip test elevation
            if meta["elevation"] == test_elevation:
                continue

            fpath = os.path.join(class_dir, fname)
            img = np.array(Image.open(fpath).convert("L")) / 255.0
            pixel_sum += img.sum()
            pixel_sq_sum += (img**2).sum()
            num_pixels += img.size
            num_images += 1

    mean = pixel_sum / num_pixels
    std = np.sqrt(pixel_sq_sum / num_pixels - mean**2)
    return mean, std


if __name__ == "__main__":
    root_dir = "./data/sample/png_images"

    for norm in ("qpm", "decibel"):
        mean, std = compute_sample_statistics(root_dir, norm)
        print(f"\n{norm.upper()} statistics (paste into configs/dataset/sample.yaml):")
        print(f"  mean: [{mean:.4f}]")
        print(f"  std:  [{std:.4f}]")
