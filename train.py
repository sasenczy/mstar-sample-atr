"""
Main training entry point for MSTAR SAR ATR.

Usage:
    python train.py                                     # Run with defaults
    python train.py training.lr=0.0005                  # Override learning rate
    python train.py model.pretrained=False training.epochs=200  # Multiple overrides

All configuration is managed by Hydra — see configs/ directory.
"""

import hydra
import hydra_zen  # noqa: F401 - patches hydra for Python 3.14 compatibility
import wandb
from omegaconf import DictConfig

from src.dataset import build_dataloaders
from src.engine import train
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function, decorated with Hydra for config management."""

    # 1. Reproducibility
    set_seed(cfg.seed)

    # 2. Device selection
    device = get_device()

    # 3. Data
    train_loader, test_loader, class_names = build_dataloaders(cfg)

    # 4. Model
    model = build_model(cfg)
    count_parameters(model)

    # 5. Train
    model = train(model, train_loader, test_loader, class_names, device, cfg)

    # 6. Cleanup
    if cfg.logging.enabled:
        wandb.finish()

    print("Done!")


if __name__ == "__main__":
    main()
