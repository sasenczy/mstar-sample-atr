"""
Standalone evaluation script for a trained MSTAR ATR model.

Loads a saved checkpoint and runs full evaluation with:
- Per-class precision, recall, and F1 scores
- Confusion matrix visualization
- Detailed analysis of misclassifications

Usage:
    python evaluate.py --checkpoint outputs/best_model.pth
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import build_dataloaders
from src.engine import evaluate, plot_confusion_matrix
from src.model import build_model
from src.utils import get_device, set_seed


def detailed_evaluation(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
    cfg: DictConfig,
) -> None:
    """
    Run comprehensive evaluation and print/log detailed results.

    Args:
        model: Trained model with best weights loaded.
        test_loader: Test DataLoader.
        class_names: List of class name strings.
        device: Torch device.
        cfg: Hydra config.
    """
    criterion = torch.nn.CrossEntropyLoss()

    # Run evaluation
    test_loss, test_acc, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device, class_names
    )

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print(f"Overall Test Loss:     {test_loss:.4f}")

    # ---------------------------------------------------------------
    # Per-class metrics using scikit-learn
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("PER-CLASS METRICS")
    print(f"{'=' * 60}")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
    )
    print(report)

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------
    fig = plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        title=f"MSTAR SOC 10-Class — Test Accuracy: {test_acc:.2f}%",
    )
    fig.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Confusion matrix saved to outputs/confusion_matrix.png")

    # Log to W&B if enabled
    if cfg.logging.enabled:
        wandb.log(
            {
                "eval/accuracy": test_acc,
                "eval/loss": test_loss,
                "eval/confusion_matrix": wandb.Image(fig),
            }
        )
    plt.close(fig)

    # ---------------------------------------------------------------
    # Misclassification analysis
    # ---------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("MISCLASSIFICATION ANALYSIS")
    print(f"{'=' * 60}")

    cm = confusion_matrix(all_labels, all_preds)
    misclassified = all_preds != all_labels
    num_errors = misclassified.sum()
    print(f"Total misclassifications: {num_errors} / {len(all_labels)}")

    # Find the most confused pairs
    print("\nMost confused class pairs:")
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((class_names[i], class_names[j], cm[i, j]))
    confusions.sort(key=lambda x: x[2], reverse=True)

    for true_cls, pred_cls, count in confusions[:10]:
        print(f"  {true_cls:>8s} → {pred_cls:<8s}: {count} errors")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained MSTAR ATR model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/best_model.pth",
        help="Path to saved model checkpoint",
    )
    args = parser.parse_args()

    # Load checkpoint and recover config
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(checkpoint["config"])

    # Setup
    set_seed(cfg.seed)
    device = get_device()

    # Build model and load weights
    model = build_model(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    print(
        f"Loaded model from epoch {checkpoint['epoch']} "
        f"(test acc: {checkpoint['test_acc']:.1f}%)"
    )

    # Build test DataLoader
    _, test_loader, class_names = build_dataloaders(cfg)

    # Initialize W&B for logging (optional)
    if cfg.logging.enabled:
        wandb.init(
            project=cfg.logging.project,
            name=f"eval-{cfg.logging.run_name or 'checkpoint'}",
            tags=["evaluation"],
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    # Run evaluation
    detailed_evaluation(model, test_loader, class_names, device, cfg)

    if cfg.logging.enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
