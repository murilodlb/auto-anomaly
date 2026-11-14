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
        im = axes[3].imshow(amap, cmap="viridis", alpha=0.55, vmin=0.0, vmax=vmax)
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


def save_pro_and_roc_plots(
    roc_fpr: np.ndarray,
    roc_tpr: np.ndarray,
    classification_au_roc_max: float,
    image_auroc_topk: float,
    pro_fpr: np.ndarray,
    pro_vals: np.ndarray,
    au_pro: float,
    pro_integration_limit: float,
    output_dir: Path,
) -> Path:
    """Save side-by-side Image Classification ROC and Pixel Localization PRO curves."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), constrained_layout=True)

    # Left: Image-level Classification ROC
    ax_roc = axes[0]
    roc_label = (
        f"max score (AU-ROC={classification_au_roc_max:.4f})"
        if np.isfinite(classification_au_roc_max)
        else "max score (N/A)"
    )
    ax_roc.plot(roc_fpr, roc_tpr, linewidth=2.0, color="#1f77b4", label=roc_label)
    ax_roc.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray", linewidth=1.0, label="Chance")
    if np.isfinite(image_auroc_topk):
        ax_roc.plot([], [], " ", label=f"top-k% AUROC={image_auroc_topk:.4f}")
    ax_roc.set_title("Image-Level Classification ROC")
    ax_roc.set_xlabel("False Positive Rate (FPR)")
    ax_roc.set_ylabel("True Positive Rate (TPR)")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle="--", alpha=0.35)

    # Right: Pixel-level Per-Region Overlap (PRO) Curve
    ax_pro = axes[1]
    mask = pro_fpr <= pro_integration_limit
    x_plot = pro_fpr[mask]
    y_plot = pro_vals[mask]
    if x_plot.size > 0 and x_plot[-1] < pro_integration_limit and pro_fpr.size > x_plot.size:
        idx = x_plot.size
        x0, x1 = pro_fpr[idx - 1], pro_fpr[idx]
        y0, y1 = pro_vals[idx - 1], pro_vals[idx]
        if x1 > x0:
            y_end = y0 + (y1 - y0) * (pro_integration_limit - x0) / (x1 - x0)
            x_plot = np.append(x_plot, pro_integration_limit)
            y_plot = np.append(y_plot, y_end)

    pro_label = (
        f"AU-PRO@{pro_integration_limit:g}={au_pro:.4f}"
        if np.isfinite(au_pro)
        else f"AU-PRO@{pro_integration_limit:g}=N/A"
    )
    ax_pro.plot(x_plot, y_plot, linewidth=2.0, color="#d62728", label=pro_label)
    if x_plot.size > 0:
        ax_pro.fill_between(x_plot, 0.0, y_plot, color="#d62728", alpha=0.18)
    ax_pro.axvline(pro_integration_limit, color="black", linestyle="--", linewidth=1.2)
    ax_pro.set_title(f"Pixel Localization PRO Curve (FPR <= {pro_integration_limit:g})")
    ax_pro.set_xlabel("False Positive Rate (FPR)")
    ax_pro.set_ylabel("Per-Region Overlap (PRO)")
    ax_pro.set_xlim(0.0, pro_integration_limit)
    ax_pro.set_ylim(0.0, 1.02)
    ax_pro.legend(loc="lower right")
    ax_pro.grid(True, linestyle="--", alpha=0.35)

    out_path = output_dir / "pro_roc_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
