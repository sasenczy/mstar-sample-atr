"""
UMAP visualization of learned embedding spaces across k-values.

Generates two figures:
1. 1xN subplot grid colored by target class with domain-differentiated markers
2. 1xN subplot grid colored by domain only (measured vs synthetic)

Usage:
    python scripts/visualize_umap.py
    python scripts/visualize_umap.py checkpoint_dir=./outputs k_values=[0.0,0.5,1.0]
"""

import os
import sys

import hydra
import hydra_zen  # noqa: F401 - patches hydra for Python 3.14 compatibility
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import umap
from omegaconf import DictConfig

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

# Colorblind-friendly colors for domain plots (Wong palette)
COLOR_MEASURED = "#0072B2"
COLOR_SYNTHETIC = "#D55E00"


def compute_umap(
    features: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
) -> np.ndarray:
    """Run UMAP on L2-normalized features, returning [N, 2] embeddings."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2,
    )
    return reducer.fit_transform(features)


def plot_umap_by_class(
    fig: plt.Figure,
    axes: np.ndarray,
    k_values: list[float],
    all_embeddings: dict,
    class_names: list[str],
    cfg: DictConfig,
) -> None:
    """Plot 1xN UMAP subplots colored by class with domain-differentiated markers."""
    cmap = plt.cm.tab10
    marker_size = 15
    alpha = 0.7

    for i, k_val in enumerate(k_values):
        ax = axes[i]
        umap_2d, class_labels, domain_labels = all_embeddings[k_val]

        for c in range(len(class_names)):
            mask_c = class_labels == c
            color = cmap(c)

            # Measured samples (circles)
            mask_m = mask_c & (domain_labels == DOMAIN_MEASURED)
            if mask_m.any():
                ax.scatter(
                    umap_2d[mask_m, 0],
                    umap_2d[mask_m, 1],
                    c=[color],
                    marker="o",
                    s=marker_size,
                    alpha=alpha,
                    label=class_names[c] if i == 0 else None,
                    edgecolors="none",
                )

            # Synthetic samples (crosses)
            mask_s = mask_c & (domain_labels == DOMAIN_SYNTHETIC)
            if mask_s.any():
                ax.scatter(
                    umap_2d[mask_s, 0],
                    umap_2d[mask_s, 1],
                    c=[color],
                    marker="x",
                    s=marker_size,
                    alpha=alpha,
                )

        ax.set_title(f"k = {k_val:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    # Build combined legend: class colors + domain markers
    class_handles = [
        mlines.Line2D(
            [],
            [],
            color=cmap(c),
            marker="o",
            linestyle="None",
            markersize=6,
            label=class_names[c],
        )
        for c in range(len(class_names))
    ]
    domain_handles = [
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="o",
            linestyle="None",
            markersize=6,
            label="Measured",
        ),
        mlines.Line2D(
            [],
            [],
            color="black",
            marker="x",
            linestyle="None",
            markersize=6,
            label="Synthetic",
        ),
    ]
    fig.legend(
        handles=class_handles + domain_handles,
        loc="center right",
        fontsize=8,
        framealpha=0.9,
        bbox_to_anchor=(1.12, 0.5),
    )


def plot_umap_by_domain(
    fig: plt.Figure,
    axes: np.ndarray,
    k_values: list[float],
    all_embeddings: dict,
    cfg: DictConfig,
) -> None:
    """Plot 1xN UMAP subplots colored by domain only."""
    marker_size = 15
    alpha = 0.7

    for i, k_val in enumerate(k_values):
        ax = axes[i]
        umap_2d, _, domain_labels = all_embeddings[k_val]

        mask_m = domain_labels == DOMAIN_MEASURED
        mask_s = domain_labels == DOMAIN_SYNTHETIC

        if mask_m.any():
            ax.scatter(
                umap_2d[mask_m, 0],
                umap_2d[mask_m, 1],
                c=COLOR_MEASURED,
                marker="o",
                s=marker_size,
                alpha=alpha,
                label="Measured" if i == 0 else None,
                edgecolors="none",
            )
        if mask_s.any():
            ax.scatter(
                umap_2d[mask_s, 0],
                umap_2d[mask_s, 1],
                c=COLOR_SYNTHETIC,
                marker="x",
                s=marker_size,
                alpha=alpha,
                label="Synthetic" if i == 0 else None,
            )

        ax.set_title(f"k = {k_val:.2f}", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])

    domain_handles = [
        mlines.Line2D(
            [],
            [],
            color=COLOR_MEASURED,
            marker="o",
            linestyle="None",
            markersize=6,
            label="Measured",
        ),
        mlines.Line2D(
            [],
            [],
            color=COLOR_SYNTHETIC,
            marker="x",
            linestyle="None",
            markersize=6,
            label="Synthetic",
        ),
    ]
    fig.legend(
        handles=domain_handles,
        loc="center right",
        fontsize=9,
        framealpha=0.9,
        bbox_to_anchor=(1.08, 0.5),
    )


@hydra.main(version_base=None, config_path="../configs/analysis", config_name="umap")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.random_state)
    device = get_device()

    # Discover checkpoints
    k_values = list(cfg.k_values)
    checkpoints = discover_checkpoints(cfg.checkpoint_dir, k_values)
    if not checkpoints:
        print("No checkpoints found. Exiting.")
        return

    available_k = sorted(checkpoints.keys())
    print(f"Found checkpoints for k = {available_k}")

    # Extract features and compute UMAP for each k-value
    all_embeddings: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
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

        umap_2d = compute_umap(
            features, cfg.n_neighbors, cfg.min_dist, cfg.metric, cfg.random_state
        )
        all_embeddings[k_val] = (umap_2d, class_labels, domain_labels)

        # Free model memory
        del model
        if device.type == "cuda":
            import torch

            torch.cuda.empty_cache()

    os.makedirs(cfg.output_dir, exist_ok=True)
    n_plots = len(available_k)

    # Figure 1: colored by class
    fig1, axes1 = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
    axes1 = axes1[0]
    plot_umap_by_class(fig1, axes1, available_k, all_embeddings, class_names, cfg)
    fig1.suptitle("UMAP Embeddings Colored by Target Class", fontsize=14, y=1.02)
    fig1.tight_layout()
    path1 = os.path.join(cfg.output_dir, "umap_by_class.png")
    fig1.savefig(path1, dpi=cfg.dpi, bbox_inches="tight")
    print(f"\nSaved: {path1}")

    # Figure 2: colored by domain
    fig2, axes2 = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5), squeeze=False)
    axes2 = axes2[0]
    plot_umap_by_domain(fig2, axes2, available_k, all_embeddings, cfg)
    fig2.suptitle("UMAP Embeddings Colored by Domain", fontsize=14, y=1.02)
    fig2.tight_layout()
    path2 = os.path.join(cfg.output_dir, "umap_by_domain.png")
    fig2.savefig(path2, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved: {path2}")

    # Log to W&B
    if cfg.wandb_enabled:
        wandb.init(
            project=cfg.wandb_project,
            name="umap-visualization",
            tags=["analysis", "umap"],
        )
        wandb.log(
            {
                "umap/by_class": wandb.Image(fig1),
                "umap/by_domain": wandb.Image(fig2),
            }
        )
        wandb.finish()

    plt.close("all")
    print("Done!")


if __name__ == "__main__":
    main()
