"""
Training and evaluation loops with W&B integration.

This module contains the core training logic: the single-epoch train step,
the validation step, and the full training loop with early stopping,
learning rate scheduling, and comprehensive metric logging.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: DictConfig,
) -> tuple[float, float]:
    """
    Run one training epoch.

    Args:
        model: The neural network.
        loader: Training DataLoader.
        criterion: Loss function (CrossEntropyLoss).
        optimizer: Optimizer (Adam or SGD).
        device: Torch device.
        epoch: Current epoch number (for logging).
        cfg: Hydra config.

    Returns:
        Tuple of (average_loss, accuracy) for the epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg.training.epochs} [Train]")
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.1f}%",
        )

        # Log batch-level metrics to W&B
        if cfg.logging.enabled and (batch_idx + 1) % cfg.logging.log_frequency == 0:
            wandb.log(
                {
                    "train/batch_loss": loss.item(),
                    "train/batch_acc": 100.0 * correct / total,
                }
            )

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
    epoch: int = -1,
    cfg: DictConfig = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate the model on a dataset.

    Args:
        model: The neural network.
        loader: Test/validation DataLoader.
        criterion: Loss function.
        device: Torch device.
        class_names: List of class name strings.
        epoch: Current epoch (for logging context).
        cfg: Hydra config.

    Returns:
        Tuple of (average_loss, accuracy, all_predictions, all_labels).
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Eval]" if epoch >= 0 else "[Eval]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return epoch_loss, epoch_acc, all_preds, all_labels


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Create a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        class_names: List of class name strings.
        title: Plot title.

    Returns:
        Matplotlib Figure object.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def build_optimizer(model: nn.Module, cfg: DictConfig) -> torch.optim.Optimizer:
    """Create optimizer from config."""
    if cfg.training.optimizer.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.training.lr,
            momentum=cfg.training.momentum,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, cfg: DictConfig
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Create learning rate scheduler from config."""
    if cfg.training.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.epochs
        )
    elif cfg.training.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.training.step_size,
            gamma=cfg.training.gamma,
        )
    elif cfg.training.scheduler == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {cfg.training.scheduler}")


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    cfg: DictConfig,
) -> nn.Module:
    """
    Full training loop with early stopping, scheduling, and W&B logging.

    This is the main training function that orchestrates everything:
    training epochs, validation, metric logging, model checkpointing,
    and early stopping.

    Args:
        model: The neural network to train.
        train_loader: Training DataLoader.
        test_loader: Test/validation DataLoader.
        class_names: List of class name strings.
        device: Torch device.
        cfg: Full Hydra config.

    Returns:
        The trained model (with best weights loaded).
    """
    model = model.to(device)

    # Loss function — standard cross-entropy for multi-class classification
    criterion = nn.CrossEntropyLoss()

    # Optimizer and scheduler
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    # Initialize W&B
    if cfg.logging.enabled:
        wandb.init(
            project=cfg.logging.project,
            entity=cfg.logging.entity,
            name=cfg.logging.run_name,
            tags=cfg.logging.tags,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        # Watch model gradients and parameters
        wandb.watch(model, log="all", log_freq=100)

    # Tracking variables for early stopping and checkpointing
    best_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(cfg.output_dir, "best_model.pth")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ===================== Training Loop =====================
    for epoch in range(cfg.training.epochs):
        # --- Train ---
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, cfg
        )

        # --- Evaluate ---
        test_loss, test_acc, all_preds, all_labels = evaluate(
            model, test_loader, criterion, device, class_names, epoch, cfg
        )

        # --- Learning rate scheduling ---
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        # --- Print epoch summary ---
        print(
            f"Epoch {epoch + 1}/{cfg.training.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.1f}% | "
            f"LR: {current_lr:.6f}"
        )

        # --- Log to W&B ---
        if cfg.logging.enabled:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/accuracy": train_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "learning_rate": current_lr,
            }

            # Log confusion matrix as a W&B plot
            if cfg.logging.log_confusion_matrix:
                fig = plot_confusion_matrix(
                    all_labels,
                    all_preds,
                    class_names,
                    title=f"Epoch {epoch + 1} — Test Acc: {test_acc:.1f}%",
                )
                log_dict["test/confusion_matrix"] = wandb.Image(fig)
                plt.close(fig)

            wandb.log(log_dict)

        # --- Checkpointing: save best model ---
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                best_model_path,
            )
            print(f"  ★ New best model saved (accuracy: {best_acc:.1f}%)")

            # Also save to W&B
            if cfg.logging.enabled and cfg.logging.save_model:
                wandb.save(best_model_path)
        else:
            patience_counter += 1

        # --- Early stopping ---
        if patience_counter >= cfg.training.patience:
            print(
                f"\nEarly stopping triggered after {patience_counter} epochs "
                f"without improvement. Best accuracy: {best_acc:.1f}% at epoch {best_epoch}"
            )
            break

    # ===================== End of Training =====================
    print(
        f"\nTraining complete. Best test accuracy: {best_acc:.1f}% at epoch {best_epoch}"
    )

    # Load best model weights
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if cfg.logging.enabled:
        wandb.summary["best_test_accuracy"] = best_acc
        wandb.summary["best_epoch"] = best_epoch

    return model
