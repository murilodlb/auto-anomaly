"""Diagnostic heatmaps, training loss curves, and anomaly score distribution plots."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _select_balanced_indices(defect_types: list[str], max_samples: int) -> list[int]:
    """Select up to max_samples indices round-robin across all defect_types."""
    if max_samples <= 0 or not defect_types:
        return []
    if len(defect_types) <= max_samples:
        return list(range(len(defect_types)))

    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, defect in enumerate(defect_types):
        buckets[defect].append(idx)

    selected: list[int] = []
    keys = sorted(buckets.keys())
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for key in keys:
            group = buckets[key]
            if cursor < len(group):
                selected.append(group[cursor])
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def _to_display_rgb(image: np.ndarray) -> np.ndarray:
    """Convert (H, W, 1) or (H, W, 3) uint8 image to (H, W, 3) for matplotlib display."""
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    return image


def save_diagnostic_samples(
    images: np.ndarray,
    masks: np.ndarray,
    reconstructions: np.ndarray,
    anomaly_maps: np.ndarray,
    image_scores: np.ndarray,
    defect_types: list[str],
    output_dir: Path,
    optimal_threshold: float,
    max_samples: int = 25,
) -> list[Path]:
    """Save 4-panel [Original | GT Mask | Recon | Heatmap Overlay] diagnostic images."""
    heatmaps_dir = output_dir / "heatmaps"
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    vmax = max(2.0 * float(optimal_threshold), 1e-8)
    selected_indices = _select_balanced_indices(defect_types, max_samples)
    saved_paths: list[Path] = []

    for idx in selected_indices:
        orig = _to_display_rgb(images[idx])
        gt_mask = masks[idx]
        recon = _to_display_rgb(reconstructions[idx])
        amap = anomaly_maps[idx]
        score = float(image_scores[idx])
        defect = defect_types[idx]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), constrained_layout=True)

        axes[0].imshow(orig)
        axes[0].set_title(f"Original ({defect})")

        axes[1].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(recon)
        axes[2].set_title("Reconstruction")

        axes[3].imshow(orig)
        im = axes[3].imshow(amap, cmap="inferno", alpha=0.55, vmin=0.0, vmax=vmax)
        axes[3].set_title(f"Anomaly Map (score={score:.4f})")
        fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

        for ax in axes:
            ax.axis("off")

        out_path = heatmaps_dir / f"{defect}_{idx:04d}_score_{score:.4f}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(out_path)

    return saved_paths


def save_training_and_score_plots(
    epoch_losses: list[float],
    labels: np.ndarray,
    image_scores: np.ndarray,
    optimal_threshold: float,
    output_dir: Path,
) -> None:
    """Save training loss curve and normal-vs-anomalous score distribution histogram."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if epoch_losses:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        epochs = np.arange(1, len(epoch_losses) + 1)
        ax.plot(epochs, epoch_losses, marker="o", linewidth=2, markersize=4)
        ax.set_title("Autoencoder Training Reconstruction Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Reconstruction Loss")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.savefig(output_dir / "loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    normal_scores = image_scores[labels == 0]
    anomaly_scores = image_scores[labels == 1]

    if normal_scores.size > 0:
        ax.hist(
            normal_scores,
            bins=25,
            alpha=0.65,
            label=f"Good (n={normal_scores.size})",
            color="#2b8cbe",
        )
    if anomaly_scores.size > 0:
        ax.hist(
            anomaly_scores,
            bins=25,
            alpha=0.65,
            label=f"Anomalous (n={anomaly_scores.size})",
            color="#de2d26",
        )

    ax.axvline(
        optimal_threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Opt Threshold ({optimal_threshold:.4f})",
    )
    ax.set_title("Test Image Anomaly Score Distribution")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.savefig(output_dir / "score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
