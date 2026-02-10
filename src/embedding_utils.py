"""
Shared utilities for embedding space analysis.

Provides:
- SAMPLEEvalDataset: loads both measured and synthetic SAMPLE images at a
  given elevation, returning (tensor, class_idx, domain_label)
- Feature extraction via forward hook on model.avgpool
- Checkpoint loading and evaluation dataloader construction
- Checkpoint discovery by k-value directory convention
"""

import os

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.dataset import build_test_transforms, parse_sample_filename
from src.model import build_model

DOMAIN_MEASURED = 0
DOMAIN_SYNTHETIC = 1


class SAMPLEEvalDataset(Dataset):
    """SAMPLE evaluation dataset loading both measured and synthetic images.

    Unlike SAMPLEDataset (which filters by k for training and only returns
    measured images for testing), this dataset loads ALL images at a given
    elevation from both real/ and synth/ directories. Each sample returns
    a 3-tuple including a domain label.

    Args:
        root_dir: Path to png_images directory (contains qpm/ and decibel/ subdirs).
        normalization: "qpm" or "decibel".
        elevation: 3-digit elevation string to load (e.g. "017").
        transform: Optional torchvision transform pipeline.
    """

    def __init__(
        self,
        root_dir: str,
        normalization: str,
        elevation: str = "017",
        transform=None,
    ):
        self.transform = transform

        real_dir = os.path.join(root_dir, normalization, "real")
        synth_dir = os.path.join(root_dir, normalization, "synth")

        # Build class list (sorted for consistent indexing, matches SAMPLEDataset)
        self.classes = sorted(
            d for d in os.listdir(real_dir) if os.path.isdir(os.path.join(real_dir, d))
        )
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect all images at the target elevation from both domains
        self.samples: list[tuple[str, int, int]] = []  # (path, class_idx, domain)

        for cls in self.classes:
            idx = self.class_to_idx[cls]

            # Measured (real) images
            cls_real_dir = os.path.join(real_dir, cls)
            if os.path.isdir(cls_real_dir):
                for fname in sorted(os.listdir(cls_real_dir)):
                    meta = parse_sample_filename(fname)
                    if meta is not None and meta["elevation"] == elevation:
                        self.samples.append(
                            (os.path.join(cls_real_dir, fname), idx, DOMAIN_MEASURED)
                        )

            # Synthetic images
            cls_synth_dir = os.path.join(synth_dir, cls)
            if os.path.isdir(cls_synth_dir):
                for fname in sorted(os.listdir(cls_synth_dir)):
                    meta = parse_sample_filename(fname)
                    if meta is not None and meta["elevation"] == elevation:
                        self.samples.append(
                            (os.path.join(cls_synth_dir, fname), idx, DOMAIN_SYNTHETIC)
                        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        path, class_idx, domain = self.samples[index]
        img = Image.open(path).convert("L")
        if self.transform is not None:
            img = self.transform(img)
        return img, class_idx, domain


def extract_features(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract L2-normalized 512-dim features from model's avgpool layer.

    Uses a forward hook on model.avgpool — does NOT modify the model.

    Args:
        model: ResNet-18 model (already on device, eval mode set internally).
        dataloader: DataLoader yielding (images, class_idx, domain_label).
        device: Torch device.

    Returns:
        features: [N, 512] L2-normalized feature array.
        class_labels: [N] integer class indices.
        domain_labels: [N] integer domain labels (0=measured, 1=synthetic).
    """
    hook_output: dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        hook_output["feat"] = output

    handle = model.avgpool.register_forward_hook(hook_fn)
    model.eval()

    features_list = []
    class_labels_list = []
    domain_labels_list = []

    with torch.no_grad():
        for images, class_idx, domain_label in dataloader:
            images = images.to(device)
            _ = model(images)

            feat = hook_output["feat"].flatten(1)  # [B, 512]
            features_list.append(feat.cpu().numpy())
            class_labels_list.append(class_idx.numpy())
            domain_labels_list.append(domain_label.numpy())

    handle.remove()

    features = np.concatenate(features_list, axis=0)
    class_labels = np.concatenate(class_labels_list, axis=0)
    domain_labels = np.concatenate(domain_labels_list, axis=0)

    # L2 normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    features = features / norms

    return features, class_labels, domain_labels


def load_checkpoint_and_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Module, DictConfig]:
    """Load a checkpoint, rebuild the model, and load weights.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        device: Torch device.

    Returns:
        model: Model with loaded weights, on device, in eval mode.
        cfg: Hydra config recovered from the checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, cfg


def build_eval_dataloader(
    cfg: DictConfig,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[DataLoader, list[str]]:
    """Build a DataLoader with SAMPLEEvalDataset (both domains, test elevation).

    Args:
        cfg: Hydra config (from checkpoint, expects cfg.dataset sub-config).
        batch_size: Batch size for the dataloader.
        num_workers: Number of data loading workers.

    Returns:
        dataloader: DataLoader yielding (tensor, class_idx, domain_label).
        class_names: List of class name strings in alphabetical order.
    """
    test_transforms = build_test_transforms(cfg)
    dataset = SAMPLEEvalDataset(
        root_dir=cfg.dataset.root_dir,
        normalization=cfg.dataset.normalization,
        elevation=cfg.dataset.test_elevation,
        transform=test_transforms,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, dataset.classes


def discover_checkpoints(
    checkpoint_dir: str,
    k_values: list[float],
) -> dict[float, str]:
    """Find checkpoints for requested k-values.

    Expects directory layout: checkpoint_dir/k_{value:.2f}/best_model.pth

    Args:
        checkpoint_dir: Parent directory containing k_* subdirectories.
        k_values: List of k-values to look for.

    Returns:
        Dict mapping k_value -> checkpoint_path for found checkpoints.
    """
    found = {}
    missing = []
    for k in k_values:
        path = os.path.join(checkpoint_dir, f"k_{k:.2f}", "best_model.pth")
        if os.path.isfile(path):
            found[k] = path
        else:
            missing.append(k)
    if missing:
        print(f"Warning: Missing checkpoints for k={missing}")
    return found
