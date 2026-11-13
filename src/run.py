"""Data-oriented anomaly detection pipeline orchestrator."""

from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PipelineConfig
from src.dataset import iter_batches, load_mvtec_category
from src.extractors import get_extractor
from src.metrics import compute_metrics
from src.models import get_model
from src.visualization import save_diagnostic_samples, save_training_and_score_plots


def _set_deterministic_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[WARN] Requested device {requested!r} is unavailable; falling back to 'cpu'.")
        return torch.device("cpu")
    return torch.device(requested)


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def _format_metric(val: float) -> str:
    return "  N/A " if math.isnan(val) else f"{val:.4f}"


def _print_summary_table(
    config: PipelineConfig,
    metrics: dict[str, Any],
    train_time_s: float,
    infer_time_s: float,
    num_test: int,
) -> None:
    fps = num_test / infer_time_s if infer_time_s > 0 else 0.0
    print("\n" + "=" * 68)
    print(
        f"Experiment: {config.output.experiment_name} | "
        f"Category: {config.dataset.category} | "
        f"Extractor: {config.extractor_name} | Model: {config.model_name}"
    )
    print("=" * 68)
    print(f"{'Image AUROC':<24}: {_format_metric(metrics['image_auroc'])}")
    print(f"{'Image AP':<24}: {_format_metric(metrics['image_ap'])}")
    print(f"{'Image F1-max':<24}: {_format_metric(metrics['f1_max'])}")
    print(f"{'Optimal Threshold':<24}: {_format_metric(metrics['optimal_threshold'])}")
    print(f"{'Accuracy @ Opt Thresh':<24}: {_format_metric(metrics['accuracy_at_opt_th'])}")
    print("-" * 68)
    print(f"{'Pixel AUROC':<24}: {_format_metric(metrics['pixel_auroc'])}")
    print(f"{'Pixel AP':<24}: {_format_metric(metrics['pixel_ap'])}")
    print(f"{'Pixel F1-max':<24}: {_format_metric(metrics['pixel_f1_max'])}")
    print("-" * 68)
    print(f"{'Train Time (s)':<24}: {train_time_s:.2f}")
    print(f"{'Inference Time (s)':<24}: {infer_time_s:.2f} ({fps:.1f} FPS)")

    per_defect = metrics.get("per_defect", {})
    if per_defect:
        print("-" * 68)
        print(f"{'Defect Type':<22} {'N':>4} {'Recall@Th':>10} {'AUROC':>10} {'Mean Score':>12}")
        for defect, stats in per_defect.items():
            print(
                f"{defect:<22} {stats['count']:>4d} "
                f"{_format_metric(stats['recall_at_opt_th']):>10} "
                f"{_format_metric(stats['image_auroc']):>10} "
                f"{stats['mean_score']:>12.5f}"
            )
    print("=" * 68 + "\n")


def _write_per_sample_csv(
    csv_path: Path,
    image_paths: list[Path],
    defect_types: list[str],
    labels: np.ndarray,
    image_scores: np.ndarray,
    optimal_threshold: float,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["index", "image_path", "defect_type", "label", "image_score", "predicted_label"]
        )
        for idx, (path, defect, label, score) in enumerate(
            zip(image_paths, defect_types, labels, image_scores, strict=True)
        ):
            pred = int(score >= optimal_threshold)
            writer.writerow([idx, str(path), defect, int(label), f"{float(score):.6f}", pred])


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Execute training/loading, inference, evaluation, and diagnostic artifact export."""
    _set_deterministic_seeds(config.seed)
    config.validate()
    device = _resolve_device(config.device)

    output_dir = config.output.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_split, test_split = load_mvtec_category(config.dataset)
    print(
        f"Loaded MVTec '{config.dataset.category}': "
        f"train={len(train_split.images)}, test={len(test_split.images)}, "
        f"image_size={config.dataset.image_size}, device={device}"
    )

    extractor = get_extractor(config, device)
    model = get_model(
        config,
        in_channels=extractor.out_channels,
        device=device,
        use_sigmoid=extractor.is_unit_range,
    )

    epoch_losses: list[float] = []
    train_time_s = 0.0
    if config.autoencoder.load_weights is not None:
        model.load(config.autoencoder.load_weights)
        print(f"Loaded weights from {config.autoencoder.load_weights}")
    else:
        t_train = time.perf_counter()
        epoch_losses = model.fit(train_split.images, extractor)
        train_time_s = time.perf_counter() - t_train
        if config.autoencoder.save_weights:
            weights_path = config.output.weights_dir / "model.pt"
            model.save(weights_path)
            print(f"Saved model weights to {weights_path}")

    t_infer = time.perf_counter()
    score_batches: list[np.ndarray] = []
    map_batches: list[np.ndarray] = []
    recon_batches: list[np.ndarray] = []
    target_hw = config.dataset.image_size

    for batch_np in iter_batches(
        test_split.images,
        batch_size=config.dataset.batch_size,
        shuffle=False,
    ):
        scores, amaps, recons = model.predict_batch(batch_np, extractor, target_hw)
        score_batches.append(scores)
        map_batches.append(amaps)
        recon_batches.append(recons)

    infer_time_s = time.perf_counter() - t_infer

    image_scores = np.concatenate(score_batches, axis=0)
    anomaly_maps = np.concatenate(map_batches, axis=0)
    reconstructions = np.concatenate(recon_batches, axis=0)

    metrics = compute_metrics(
        labels=test_split.labels,
        image_scores=image_scores,
        masks=test_split.masks,
        anomaly_maps=anomaly_maps,
        defect_types=test_split.defect_types,
    )

    _print_summary_table(config, metrics, train_time_s, infer_time_s, len(test_split.images))

    results_payload: dict[str, Any] = {
        "experiment_name": config.output.experiment_name,
        "category": config.dataset.category,
        "extractor": config.extractor_name,
        "model": config.model_name,
        "device": str(device),
        "image_size": list(config.dataset.image_size),
        "num_train": len(train_split.images),
        "num_test": len(test_split.images),
        "train_time_s": train_time_s,
        "infer_time_s": infer_time_s,
        "infer_fps": len(test_split.images) / infer_time_s if infer_time_s > 0 else 0.0,
        "epoch_losses": epoch_losses,
        "metrics": metrics,
    }

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(results_payload), f, indent=2)

    csv_path = output_dir / "per_sample_scores.csv"
    _write_per_sample_csv(
        csv_path=csv_path,
        image_paths=test_split.image_paths,
        defect_types=test_split.defect_types,
        labels=test_split.labels,
        image_scores=image_scores,
        optimal_threshold=float(metrics["optimal_threshold"]),
    )

    if config.output.save_visualizations:
        save_diagnostic_samples(
            images=test_split.images,
            masks=test_split.masks,
            reconstructions=reconstructions,
            anomaly_maps=anomaly_maps,
            image_scores=image_scores,
            defect_types=test_split.defect_types,
            output_dir=output_dir,
            optimal_threshold=float(metrics["optimal_threshold"]),
            max_samples=config.output.max_vis_samples,
        )
        save_training_and_score_plots(
            epoch_losses=epoch_losses,
            labels=test_split.labels,
            image_scores=image_scores,
            optimal_threshold=float(metrics["optimal_threshold"]),
            output_dir=output_dir,
        )

    return results_payload
