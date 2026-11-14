"""Pure evaluation metrics for image-level, pixel-level, AU-PRO, and per-defect evaluation."""

from __future__ import annotations

from bisect import bisect
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.ndimage import label as ndimage_label
from sklearn.metrics import precision_recall_curve, roc_auc_score


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _ap_and_max_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[float, float, float]:
    """Compute (average_precision, f1_max, optimal_threshold) in a single sort pass."""
    if np.unique(y_true).size < 2:
        default_th = float(np.max(y_score)) if y_score.size > 0 else 0.0
        return float("nan"), 0.0, default_th

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ap = float(-np.sum(np.diff(recall) * precision[:-1]))

    if thresholds.size == 0:
        return ap, 0.0, float(np.max(y_score))

    p = precision[:-1]
    r = recall[:-1]
    denom = p + r
    f1 = np.divide(2.0 * p * r, denom, out=np.zeros_like(denom), where=denom > 0.0)
    best_idx = int(np.argmax(f1))
    return ap, float(f1[best_idx]), float(thresholds[best_idx])


def trapezoid(x: np.ndarray, y: np.ndarray, x_max: float | None = None) -> float:
    """Compute definite integral using trapezoidal rule with optional upper integration limit."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    finite_mask = np.logical_and(np.isfinite(x_arr), np.isfinite(y_arr))
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]
    if x_arr.size < 2:
        return 0.0

    correction = 0.0
    if x_max is not None:
        if x_max <= x_arr[0]:
            return 0.0
        if x_max < x_arr[-1]:
            if x_max not in x_arr:
                ins = bisect(x_arr, x_max)
                y_interp = y_arr[ins - 1] + (
                    (y_arr[ins] - y_arr[ins - 1])
                    * (x_max - x_arr[ins - 1])
                    / (x_arr[ins] - x_arr[ins - 1])
                )
                correction = 0.5 * (y_interp + y_arr[ins - 1]) * (x_max - x_arr[ins - 1])
            mask = x_arr <= x_max
            x_arr = x_arr[mask]
            y_arr = y_arr[mask]

    if x_arr.size < 2:
        return float(correction)
    return float(np.sum(0.5 * (y_arr[1:] + y_arr[:-1]) * (x_arr[1:] - x_arr[:-1])) + correction)


def compute_pro_curve(
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the MVTec AD Per-Region Overlap (PRO) curve (fprs, pros)."""
    amaps = np.asarray(anomaly_maps)
    gt_masks = np.asarray(masks)
    if amaps.shape != gt_masks.shape:
        raise ValueError(
            f"anomaly_maps shape {amaps.shape} must match masks shape {gt_masks.shape}"
        )

    structure = np.ones((3, 3), dtype=int)
    n, h, w = amaps.shape
    total_pixels = n * h * w
    fp_dtype = np.uint32 if total_pixels < np.iinfo(np.uint32).max else np.uint64

    num_ok_pixels = 0
    num_gt_regions = 0
    fp_changes = np.zeros((n, h, w), dtype=fp_dtype)
    pro_changes = np.zeros((n, h, w), dtype=np.float32)

    for idx in range(n):
        gt_map = gt_masks[idx] > 0
        label_result: Any = ndimage_label(gt_map, structure=structure)
        labeled = np.asarray(label_result[0], dtype=np.int32)
        n_components = int(label_result[1])
        num_gt_regions += n_components

        ok_mask = labeled == 0
        num_ok_pixels += int(np.sum(ok_mask))
        fp_changes[idx, ok_mask] = 1

        if n_components > 0:
            region_sizes = np.bincount(labeled.ravel())
            weights = np.zeros(n_components + 1, dtype=np.float64)
            weights[1:] = 1.0 / region_sizes[1:]
            pro_changes[idx] = weights[labeled]

    if num_gt_regions == 0 or num_ok_pixels == 0:
        return np.array([0.0, 1.0], dtype=np.float64), np.array([0.0, 1.0], dtype=np.float64)

    scores_flat = np.array(amaps, copy=True).ravel()
    fp_flat = fp_changes.ravel()
    pro_flat = pro_changes.ravel()

    sort_dtype = np.uint32 if total_pixels < np.iinfo(np.uint32).max else np.uint64
    sort_idxs = np.argsort(scores_flat)[::-1].astype(sort_dtype, copy=False)

    np.take(scores_flat, sort_idxs, out=scores_flat)
    np.take(fp_flat, sort_idxs, out=fp_flat)
    np.take(pro_flat, sort_idxs, out=pro_flat)
    del sort_idxs

    fprs = np.cumsum(fp_flat, dtype=np.float64) / float(num_ok_pixels)
    pros = np.cumsum(pro_flat, dtype=np.float64) / float(num_gt_regions)

    keep_mask = np.append(np.diff(scores_flat) != 0, True)
    fprs = np.clip(fprs[keep_mask], 0.0, 1.0)
    pros = np.clip(pros[keep_mask], 0.0, 1.0)

    zero = np.array([0.0], dtype=np.float64)
    one = np.array([1.0], dtype=np.float64)
    return np.concatenate((zero, fprs, one)), np.concatenate((zero, pros, one))


def compute_au_pro(
    anomaly_maps: np.ndarray,
    masks: np.ndarray,
    integration_limit: float = 0.3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute normalized AU-PRO up to FPR integration_limit and return (au_pro, fprs, pros)."""
    if not (0.0 < integration_limit <= 1.0):
        raise ValueError(f"integration_limit must be in (0.0, 1.0], got {integration_limit}")

    gt_binary = np.asarray(masks) > 0
    if not np.any(gt_binary) or not np.any(~gt_binary):
        dummy = np.array([0.0, 1.0], dtype=np.float64)
        return float("nan"), dummy, dummy.copy()

    fprs, pros = compute_pro_curve(anomaly_maps, masks)
    au_pro = trapezoid(fprs, pros, x_max=integration_limit) / integration_limit
    return float(au_pro), fprs, pros


def compute_classification_roc(
    anomaly_maps: np.ndarray,
    labels: np.ndarray,
    scoring_fn: Callable[[np.ndarray], float] = np.max,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute image-level classification ROC curve and AU-ROC using scoring_fn."""
    labels_arr = np.asarray(labels, dtype=np.int32).ravel()
    amaps = np.asarray(anomaly_maps)
    if len(amaps) != len(labels_arr):
        raise ValueError("anomaly_maps and labels must have the same length")

    if np.unique(labels_arr).size < 2:
        dummy = np.array([0.0, 1.0], dtype=np.float64)
        return float("nan"), dummy, dummy.copy()

    if scoring_fn is np.max and amaps.ndim == 3:
        scores = amaps.max(axis=(1, 2)).astype(np.float64)
    else:
        scores = np.array([float(scoring_fn(amap)) for amap in amaps], dtype=np.float64)

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels_arr[order]

    num_nok = int(np.sum(labels_arr != 0))
    num_ok = int(np.sum(labels_arr == 0))

    fprs = [1.0]
    tprs = [1.0]
    num_fp = num_ok
    num_tp = num_nok
    n = len(sorted_scores)

    for i in range(n):
        if sorted_labels[i] == 0:
            num_fp -= 1
        else:
            num_tp -= 1

        if i == n - 1 or sorted_scores[i + 1] != sorted_scores[i]:
            fprs.append(num_fp / num_ok)
            tprs.append(num_tp / num_nok)

    fprs_arr = np.asarray(fprs[::-1], dtype=np.float64)
    tprs_arr = np.asarray(tprs[::-1], dtype=np.float64)
    au_roc = trapezoid(fprs_arr, tprs_arr)
    return float(au_roc), fprs_arr, tprs_arr


def compute_metrics_with_curves(
    labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    defect_types: list[str],
    pro_integration_limit: float = 0.3,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Compute all image, pixel, AU-PRO, and per-defect metrics along with ROC/PRO curves."""
    labels_arr = np.asarray(labels, dtype=np.int32).ravel()
    scores_arr = np.asarray(image_scores, dtype=np.float64).ravel()

    image_auroc = _safe_auroc(labels_arr, scores_arr)
    image_ap, f1_max, optimal_threshold = _ap_and_max_f1(labels_arr, scores_arr)

    preds = (scores_arr >= optimal_threshold).astype(np.int32)
    accuracy_at_opt_th = float(np.mean(preds == labels_arr)) if labels_arr.size > 0 else 0.0

    classification_au_roc_max, roc_fpr, roc_tpr = compute_classification_roc(
        anomaly_maps=anomaly_maps,
        labels=labels_arr,
        scoring_fn=np.max,
    )

    au_pro, pro_fpr, pro_vals = compute_au_pro(
        anomaly_maps=anomaly_maps,
        masks=masks,
        integration_limit=pro_integration_limit,
    )

    flat_masks = np.asarray(masks, dtype=np.uint8).ravel()
    flat_maps = np.asarray(anomaly_maps, dtype=np.float32).ravel()
    pixel_auroc = _safe_auroc(flat_masks, flat_maps)
    pixel_ap, pixel_f1_max, _ = _ap_and_max_f1(flat_masks, flat_maps)

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
            d_au_pro, _, _ = compute_au_pro(
                anomaly_maps=anomaly_maps[pair_mask],
                masks=masks[pair_mask],
                integration_limit=pro_integration_limit,
            )
        else:
            d_auroc = float("nan")
            d_au_pro = float("nan")

        per_defect[defect] = {
            "count": int(np.sum(d_mask)),
            "recall_at_opt_th": recall_at_th,
            "image_auroc": d_auroc,
            "au_pro": d_au_pro,
            "mean_score": float(np.mean(d_scores)) if d_scores.size > 0 else 0.0,
            "std_score": float(np.std(d_scores)) if d_scores.size > 0 else 0.0,
        }

    metrics: dict[str, Any] = {
        "au_pro": au_pro,
        "pro_integration_limit": float(pro_integration_limit),
        "classification_au_roc_max": classification_au_roc_max,
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
    curves = {
        "roc_fpr": roc_fpr,
        "roc_tpr": roc_tpr,
        "pro_fpr": pro_fpr,
        "pro_vals": pro_vals,
    }
    return metrics, curves


def compute_metrics(
    labels: np.ndarray,
    image_scores: np.ndarray,
    masks: np.ndarray,
    anomaly_maps: np.ndarray,
    defect_types: list[str],
    pro_integration_limit: float = 0.3,
) -> dict[str, Any]:
    """Compute image-level, pixel-level, AU-PRO, and per-defect-type anomaly detection metrics."""
    metrics, _ = compute_metrics_with_curves(
        labels=labels,
        image_scores=image_scores,
        masks=masks,
        anomaly_maps=anomaly_maps,
        defect_types=defect_types,
        pro_integration_limit=pro_integration_limit,
    )
    return metrics
