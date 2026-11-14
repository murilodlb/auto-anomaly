"""Data-oriented anomaly detection pipeline orchestrator."""

from __future__ import annotations

import csv
import gc
import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile
import torch

from src.config import (
    PipelineConfig,
    detect_aligned_native_size,
    discover_mvtec_categories,
)
from src.dataset import MVTecSplit, iter_batches, load_mvtec_category
from src.extractors import get_extractor
from src.metrics import compute_metrics_with_curves
from src.models import get_model
from src.visualization import (
    save_diagnostic_samples,
    save_pro_and_roc_plots,
    save_training_and_score_plots,
)


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
    device = torch.device(requested)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(device)
        print(
            f"[INFO] Using CUDA device: {gpu_name} "
            f"(torch={torch.__version__}, cuda={torch.version.cuda})"
        )
    return device


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
    pro_limit = metrics["pro_integration_limit"]
    print("\n" + "=" * 78)
    print(
        f"Experiment: {config.output.experiment_name} | "
        f"Category: {config.dataset.category} | "
        f"Extractor: {config.extractor_name} | Model: {config.model_name}"
    )
    print("=" * 78)
    print(f"{f'AU-PRO (FPR<={pro_limit:g})':<26}: {_format_metric(metrics['au_pro'])}")
    print(f"{'Image AU-ROC (max)':<26}: {_format_metric(metrics['classification_au_roc_max'])}")
    print(f"{'Image AUROC (top-k%)':<26}: {_format_metric(metrics['image_auroc'])}")
    print(f"{'Image AP':<26}: {_format_metric(metrics['image_ap'])}")
    print(f"{'Image F1-max':<26}: {_format_metric(metrics['f1_max'])}")
    print(f"{'Optimal Threshold':<26}: {_format_metric(metrics['optimal_threshold'])}")
    print(f"{'Accuracy @ Opt Thresh':<26}: {_format_metric(metrics['accuracy_at_opt_th'])}")
    print("-" * 78)
    print(f"{'Pixel AUROC':<26}: {_format_metric(metrics['pixel_auroc'])}")
    print(f"{'Pixel AP':<26}: {_format_metric(metrics['pixel_ap'])}")
    print(f"{'Pixel F1-max':<26}: {_format_metric(metrics['pixel_f1_max'])}")
    print("-" * 78)
    print(f"{'Train Time (s)':<26}: {train_time_s:.2f}")
    print(f"{'Inference Time (s)':<26}: {infer_time_s:.2f} ({fps:.1f} FPS)")

    per_defect = metrics.get("per_defect", {})
    if per_defect:
        print("-" * 78)
        print(
            f"{'Defect Type':<22} {'N':>4} {'Recall@Th':>10} "
            f"{'AUROC':>10} {'AU-PRO':>10} {'Mean Score':>12}"
        )
        for defect, stats in per_defect.items():
            print(
                f"{defect:<22} {stats['count']:>4d} "
                f"{_format_metric(stats['recall_at_opt_th']):>10} "
                f"{_format_metric(stats['image_auroc']):>10} "
                f"{_format_metric(stats['au_pro']):>10} "
                f"{stats['mean_score']:>12.5f}"
            )
    print("=" * 78 + "\n")


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


def _export_tiff_anomaly_maps(
    anomaly_maps: np.ndarray,
    test_split: MVTecSplit,
    category: str,
    anomaly_maps_dir: Path,
) -> int:
    """Export native-resolution float32 .tiff anomaly maps matching MVTec AD layout."""
    count = 0
    for idx, (img_path, defect, amap) in enumerate(
        zip(test_split.image_paths, test_split.defect_types, anomaly_maps, strict=True)
    ):
        out_path = anomaly_maps_dir / category / "test" / defect / f"{img_path.stem}.tiff"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        orig_hw = (
            test_split.original_sizes[idx] if idx < len(test_split.original_sizes) else amap.shape
        )
        orig_h, orig_w = int(orig_hw[0]), int(orig_hw[1])
        map_f32 = amap.astype(np.float32, copy=False)
        if map_f32.shape != (orig_h, orig_w):
            map_f32 = cv2.resize(map_f32, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        tifffile.imwrite(str(out_path), np.ascontiguousarray(map_f32, dtype=np.float32))
        count += 1
    return count


def _run_single_category(
    config: PipelineConfig,
    tiff_base_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute training/loading, inference, evaluation, and artifact export for one category."""
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

    metrics, curves = compute_metrics_with_curves(
        labels=test_split.labels,
        image_scores=image_scores,
        masks=test_split.masks,
        anomaly_maps=anomaly_maps,
        defect_types=test_split.defect_types,
        pro_integration_limit=config.pro_integration_limit,
    )

    _print_summary_table(config, metrics, train_time_s, infer_time_s, len(test_split.images))

    if config.output.save_tiff_maps:
        tiff_root = tiff_base_dir if tiff_base_dir is not None else config.output.anomaly_maps_dir
        exported = _export_tiff_anomaly_maps(
            anomaly_maps=anomaly_maps,
            test_split=test_split,
            category=config.dataset.category,
            anomaly_maps_dir=tiff_root,
        )
        print(f"Exported {exported} native-resolution .tiff anomaly maps to {tiff_root}")

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
        save_pro_and_roc_plots(
            roc_fpr=curves["roc_fpr"],
            roc_tpr=curves["roc_tpr"],
            classification_au_roc_max=float(metrics["classification_au_roc_max"]),
            image_auroc_topk=float(metrics["image_auroc"]),
            pro_fpr=curves["pro_fpr"],
            pro_vals=curves["pro_vals"],
            au_pro=float(metrics["au_pro"]),
            pro_integration_limit=config.pro_integration_limit,
            output_dir=output_dir,
        )

    return results_payload


def _print_multi_category_table(
    experiment_name: str,
    per_category: dict[str, dict[str, Any]],
    pro_limit: float,
) -> None:
    print("\n" + "=" * 78)
    print(f"MVTec AD Multi-Category Summary: {experiment_name} (PRO limit={pro_limit:g})")
    print("=" * 78)
    print(
        f"{'Category':<18} {'AU-PRO':>10} {'AU-ROC(max)':>12} "
        f"{'AUROC(top-k)':>13} {'Pixel AUROC':>12}"
    )
    print("-" * 78)

    au_pros, roc_maxs, roc_topks, pix_rocs = [], [], [], []
    for cat, payload in per_category.items():
        m = payload["metrics"]
        au_pros.append(m["au_pro"])
        roc_maxs.append(m["classification_au_roc_max"])
        roc_topks.append(m["image_auroc"])
        pix_rocs.append(m["pixel_auroc"])
        print(
            f"{cat:<18} {_format_metric(m['au_pro']):>10} "
            f"{_format_metric(m['classification_au_roc_max']):>12} "
            f"{_format_metric(m['image_auroc']):>13} "
            f"{_format_metric(m['pixel_auroc']):>12}"
        )

    print("-" * 78)
    print(
        f"{'Mean':<18} {_format_metric(float(np.nanmean(au_pros))):>10} "
        f"{_format_metric(float(np.nanmean(roc_maxs))):>12} "
        f"{_format_metric(float(np.nanmean(roc_topks))):>13} "
        f"{_format_metric(float(np.nanmean(pix_rocs))):>12}"
    )
    print("=" * 78 + "\n")


def run_multi_category_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the pipeline across all MVTec AD categories discovered in config.dataset.root_dir."""
    config.validate()
    categories = discover_mvtec_categories(config.dataset.root_dir)
    root_output_dir = config.output.output_dir
    root_output_dir.mkdir(parents=True, exist_ok=True)
    shared_tiff_dir = config.output.anomaly_maps_dir if config.output.save_tiff_maps else None

    stride_factor = 2**config.autoencoder.num_stages
    per_category: dict[str, dict[str, Any]] = {}

    for category in categories:
        cat_image_size = (
            detect_aligned_native_size(config.dataset.root_dir / category, stride_factor)
            if config.dataset.full_res
            else config.dataset.image_size
        )
        cat_dataset_cfg = replace(
            config.dataset,
            category=category,
            image_size=cat_image_size,
        )
        cat_output_cfg = replace(
            config.output,
            base_dir=root_output_dir,
            experiment_name=category,
        )
        cat_cfg = replace(
            config,
            dataset=cat_dataset_cfg,
            output=cat_output_cfg,
        )
        per_category[category] = _run_single_category(cat_cfg, tiff_base_dir=shared_tiff_dir)
        gc.collect()
        torch.cuda.empty_cache()

    _print_multi_category_table(
        config.output.experiment_name,
        per_category,
        config.pro_integration_limit,
    )

    summary_payload: dict[str, Any] = {
        "experiment_name": config.output.experiment_name,
        "categories": categories,
        "pro_integration_limit": config.pro_integration_limit,
        "mean_au_pro": float(np.nanmean([p["metrics"]["au_pro"] for p in per_category.values()])),
        "mean_classification_au_roc_max": float(
            np.nanmean([p["metrics"]["classification_au_roc_max"] for p in per_category.values()])
        ),
        "mean_image_auroc": float(
            np.nanmean([p["metrics"]["image_auroc"] for p in per_category.values()])
        ),
        "mean_pixel_auroc": float(
            np.nanmean([p["metrics"]["pixel_auroc"] for p in per_category.values()])
        ),
        "per_category": per_category,
    }

    summary_path = root_output_dir / "summary_metrics.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize_for_json(summary_payload), f, indent=2)

    return summary_payload


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the pipeline for a single category or all categories when category == 'all'."""
    if config.dataset.category == "all":
        return run_multi_category_pipeline(config)
    return _run_single_category(config)
