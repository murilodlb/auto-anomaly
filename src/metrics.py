"""Pure evaluation metrics for image-level, pixel-level, and per-defect anomaly detection."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _max_f1_and_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    if np.unique(y_true).size < 2:
        default_th = float(np.max(y_score)) if y_score.size > 0 else 0.0
        return 0.0, default_th

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if thresholds.size == 0:
        return 0.0, float(np.max(y_score))

    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.divide(2.0 * p * r, denom, out=np.zeros_like(denom), where=denom > 0.0)
    best_idx = int(np.argmax(f1))
    return float(f1[best_idx]), float(thresholds[best_idx])


def compute_metrics(
    labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    defect_types: list[str],
) -> dict[str, Any]:
    """Compute image-level, pixel-level, and per-defect-type anomaly detection metrics."""
    labels_arr = np.asarray(labels, dtype=np.int32).ravel()
    scores_arr = np.asarray(image_scores, dtype=np.float64).ravel()

    image_auroc = _safe_auroc(labels_arr, scores_arr)
    image_ap = _safe_ap(labels_arr, scores_arr)
    f1_max, optimal_threshold = _max_f1_and_threshold(labels_arr, scores_arr)

    preds = (scores_arr >= optimal_threshold).astype(np.int32)
    accuracy_at_opt_th = float(np.mean(preds == labels_arr)) if labels_arr.size > 0 else 0.0

    flat_masks = np.asarray(masks, dtype=np.uint8).ravel()
    flat_maps = np.asarray(anomaly_maps, dtype=np.float64).ravel()
    pixel_auroc = _safe_auroc(flat_masks, flat_maps)
    pixel_ap = _safe_ap(flat_masks, flat_maps)
    pixel_f1_max, _ = _max_f1_and_threshold(flat_masks, flat_maps)

    defect_arr = np.asarray(defect_types)
    good_mask = defect_arr == "good"
    per_defect: dict[str, dict[str, float | int]] = {}

    for defect in sorted(set(defect_types)):
        if defect == "good":
            continue
        d_mask = defect_arr == defect
        d_scores = scores_arr[d_mask]
        recall_at_th = float(np.mean(d_scores >= optimal_threshold)) if d_scores.size > 0 else 0.0

        if np.any(good_mask) and np.any(d_mask):
            pair_mask = good_mask | d_mask
            pair_labels = (defect_arr[pair_mask] == defect).astype(np.int32)
            pair_scores = scores_arr[pair_mask]
            d_auroc = _safe_auroc(pair_labels, pair_scores)
        else:
            d_auroc = float("nan")

        per_defect[defect] = {
            "count": int(np.sum(d_mask)),
            "recall_at_opt_th": recall_at_th,
            "image_auroc": d_auroc,
            "mean_score": float(np.mean(d_scores)) if d_scores.size > 0 else 0.0,
            "std_score": float(np.std(d_scores)) if d_scores.size > 0 else 0.0,
        }

    return {
        "image_auroc": image_auroc,
        "image_ap": image_ap,
        "f1_max": f1_max,
        "optimal_threshold": optimal_threshold,
        "accuracy_at_opt_th": accuracy_at_opt_th,
        "pixel_auroc": pixel_auroc,
        "pixel_ap": pixel_ap,
        "pixel_f1_max": pixel_f1_max,
        "per_defect": per_defect,
    }
