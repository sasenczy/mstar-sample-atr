"""
Compute embedding space metrics across k-values.

Metrics computed per k-value:
- Class silhouette score: how well-separated are the 10 target classes?
- Domain silhouette score: how separable are measured vs synthetic samples?
- Per-class domain centroid distance: Euclidean distance between measured and
  synthetic centroids for each class in 512-dim feature space

Usage:
    python scripts/embedding_metrics.py
    python scripts/embedding_metrics.py checkpoint_dir=./outputs
"""

import os
import sys

import hydra
import hydra_zen  # noqa: F401 - patches hydra for Python 3.14 compatibility
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics import silhouette_score

import wandb

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.embedding_utils import (
    DOMAIN_MEASURED,
    DOMAIN_SYNTHETIC,
    build_eval_dataloader,
    discover_checkpoints,
    extract_features,
    load_checkpoint_and_model,
)
from src.utils import get_device, set_seed


def compute_silhouette_scores(
    features: np.ndarray,
    class_labels: np.ndarray,
    domain_labels: np.ndarray,
) -> tuple[float, float]:
    """Compute class and domain silhouette scores.

    Args:
        features: [N, 512] L2-normalized feature array.
        class_labels: [N] integer class indices.
        domain_labels: [N] integer domain labels.

    Returns:
        (class_silhouette, domain_silhouette). Domain silhouette is NaN if
        only one domain is present.
    """
    class_sil = silhouette_score(features, class_labels, metric="euclidean")

    unique_domains = np.unique(domain_labels)
    if len(unique_domains) < 2:
        domain_sil = float("nan")
    else:
        domain_sil = silhouette_score(features, domain_labels, metric="euclidean")

    return class_sil, domain_sil


def compute_per_class_centroid_distances(
    features: np.ndarray,
    class_labels: np.ndarray,
    domain_labels: np.ndarray,
    num_classes: int = 10,
) -> dict[int, float]:
    """Compute per-class Euclidean distance between measured and synthetic centroids.

    Args:
        features: [N, 512] L2-normalized feature array.
        class_labels: [N] integer class indices.
        domain_labels: [N] integer domain labels.
        num_classes: Number of classes.

    Returns:
        Dict mapping class_idx -> centroid distance. NaN if a domain is missing
        for that class.
    """
    distances = {}
    for c in range(num_classes):
        mask_c = class_labels == c
        measured_mask = mask_c & (domain_labels == DOMAIN_MEASURED)
        synthetic_mask = mask_c & (domain_labels == DOMAIN_SYNTHETIC)

        if measured_mask.sum() == 0 or synthetic_mask.sum() == 0:
            distances[c] = float("nan")
            continue

        centroid_measured = features[measured_mask].mean(axis=0)
        centroid_synthetic = features[synthetic_mask].mean(axis=0)
        distances[c] = float(np.linalg.norm(centroid_measured - centroid_synthetic))

    return distances


def plot_silhouette_vs_k(
    k_values: list[float],
    class_silhouettes: list[float],
    domain_silhouettes: list[float],
    output_path: str,
    dpi: int = 150,
) -> plt.Figure:
    """Plot class and domain silhouette scores vs k-value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        k_values,
        class_silhouettes,
        "o-",
        color="#0072B2",
        linewidth=2,
        markersize=6,
        label="Class silhouette",
    )
    ax.plot(
        k_values,
        domain_silhouettes,
        "s-",
        color="#D55E00",
        linewidth=2,
        markersize=6,
        label="Domain silhouette",
    )

    ax.set_xlabel("k (measured/synthetic ratio)", fontsize=12)
    ax.set_ylabel("Silhouette score", fontsize=12)
    ax.set_title("Silhouette Scores vs Training Data Composition", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig


def plot_centroid_distances_vs_k(
    k_values: list[float],
    per_class_distances: list[dict[int, float]],
    class_names: list[str],
    output_path: str,
    dpi: int = 150,
) -> plt.Figure:
    """Plot per-class domain centroid distances vs k-value."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.tab10

    for c in range(len(class_names)):
        dists = [d[c] for d in per_class_distances]
        ax.plot(
            k_values,
            dists,
            "o-",
            color=cmap(c),
            linewidth=1.5,
            markersize=4,
            label=class_names[c],
        )

    ax.set_xlabel("k (measured/synthetic ratio)", fontsize=12)
    ax.set_ylabel("Centroid distance (Euclidean)", fontsize=12)
    ax.set_title(
        "Per-Class Domain Centroid Distance vs Training Data Composition", fontsize=14
    )
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {output_path}")
    return fig


@hydra.main(version_base=None, config_path="../configs/analysis", config_name="metrics")
def main(cfg: DictConfig) -> None:
    set_seed(42)
    device = get_device()

    # Build k-values list
    n_steps = round((cfg.k_values_end - cfg.k_values_start) / cfg.k_values_step) + 1
    k_values = [
        round(cfg.k_values_start + i * cfg.k_values_step, 2) for i in range(n_steps)
    ]

    # Discover checkpoints
    checkpoints = discover_checkpoints(cfg.checkpoint_dir, k_values)
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    available_k = sorted(checkpoints.keys())
    print(f"Found checkpoints for k = {available_k}")

    # Compute metrics for each k-value
    results_k: list[float] = []
    class_sils: list[float] = []
    domain_sils: list[float] = []
    centroid_dists: list[dict[int, float]] = []
    class_names: list[str] | None = None

    for k_val in available_k:
        print(f"\nProcessing k = {k_val:.2f}...")
        model, model_cfg = load_checkpoint_and_model(checkpoints[k_val], device)
        eval_loader, cls_names = build_eval_dataloader(model_cfg)
        if class_names is None:
            class_names = cls_names

        features, class_labels, domain_labels = extract_features(
            model, eval_loader, device
        )
        print(f"  Extracted {features.shape[0]} features of dim {features.shape[1]}")

        c_sil, d_sil = compute_silhouette_scores(features, class_labels, domain_labels)
        dists = compute_per_class_centroid_distances(
            features, class_labels, domain_labels, num_classes=len(class_names)
        )

        results_k.append(k_val)
        class_sils.append(c_sil)
        domain_sils.append(d_sil)
        centroid_dists.append(dists)

        print(f"  Class silhouette: {c_sil:.4f}")
        print(f"  Domain silhouette: {d_sil:.4f}")

        # Free model memory
        del model
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Plot silhouette scores vs k
    fig1 = plot_silhouette_vs_k(
        results_k,
        class_sils,
        domain_sils,
        os.path.join(cfg.output_dir, "silhouette_vs_k.png"),
        dpi=cfg.dpi,
    )

    # Plot centroid distances vs k
    fig2 = plot_centroid_distances_vs_k(
        results_k,
        centroid_dists,
        class_names,
        os.path.join(cfg.output_dir, "centroid_distance_vs_k.png"),
        dpi=cfg.dpi,
    )

    # Log to W&B
    if cfg.wandb_enabled:
        wandb.init(
            project=cfg.wandb_project,
            name="embedding-metrics",
            tags=["analysis", "metrics"],
        )
        for i, k_val in enumerate(results_k):
            log_dict: dict = {
                "k": k_val,
                "metrics/class_silhouette": class_sils[i],
                "metrics/domain_silhouette": domain_sils[i],
            }
            for c in range(len(class_names)):
                dist = centroid_dists[i].get(c, float("nan"))
                if not np.isnan(dist):
                    log_dict[f"metrics/centroid_dist/{class_names[c]}"] = dist
            wandb.log(log_dict)

        wandb.log(
            {
                "plots/silhouette_vs_k": wandb.Image(fig1),
                "plots/centroid_distance_vs_k": wandb.Image(fig2),
            }
        )
        wandb.finish()

    plt.close("all")
    print("\nDone!")


if __name__ == "__main__":
    main()
