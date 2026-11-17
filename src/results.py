"""Results directory inspector, hyperparameter leaderboard, and CSV exporter."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_ARTIFACT_SUBDIRS = frozenset({"heatmaps", "weights", "anomaly_maps"})
_PLOT_NAMES = ("loss_curve.png", "score_distribution.png", "pro_roc_curves.png")
_NAN = float("nan")

_NAME_HP_PATTERN = re.compile(
    r"^(?P<prefix>.+?)_"
    r"(?:(?P<res>fullres|\d+x\d+)_)?"
    r"(?P<loss>ms_l1_ssim|l1_ssim|l1ssim|smooth_l1|l1|l2)_"
    r"s(?P<stages>\d+)"
    r"(?:_c(?P<base_ch>\d+))?"
    r"(?:_z(?P<latent>\d+))?"
    r"(?:_.*)?$"
)

_LOSS_ALIASES = {"l1ssim": "l1_ssim"}


@dataclass(frozen=True)
class FolderInventoryEntry:
    """High-level summary of one directory under results/."""

    folder_name: str
    folder_path: str
    status: str
    categories: tuple[str, ...]
    has_metrics: bool
    has_summary: bool
    has_scores_csv: bool
    has_weights: bool
    has_plots: bool
    num_heatmaps: int
    num_tiff_maps: int
    size_mb: float
    updated_at: str


@dataclass(frozen=True)
class ExperimentRecord:
    """Flattened metrics, hyperparameters, and artifact stats for one run."""

    experiment_name: str
    parent_experiment: str
    category: str
    status: str
    updated_at: str
    device: str
    image_size: str
    num_train: int | None
    num_test: int | None
    hp_source: str
    loss_type: str
    num_stages: int | None
    base_channels: int | None
    latent_dim: int | None
    epochs: int | None
    batch_size: int | None
    lr: float | None
    weight_decay: float | None
    gaussian_sigma: float | None
    score_top_k_pct: float | None
    spatial_dropout: float | None
    denoise_prob: float | None
    texture_damping: float | None
    loss_floor: float | None
    full_res: bool | None
    use_amp: bool | None
    pro_integration_limit: float
    seed: int | None
    calibrated: bool | None
    calib_err_mean: float
    calib_err_std: float
    au_pro: float
    classification_au_roc_max: float
    image_auroc: float
    image_ap: float
    f1_max: float
    optimal_threshold: float
    accuracy_at_opt_th: float
    pixel_auroc: float
    pixel_ap: float
    pixel_f1_max: float
    good_mean_score: float
    good_max_score: float
    defect_mean_score: float
    defect_min_score: float
    epochs_completed: int
    initial_loss: float
    final_loss: float
    best_loss: float
    train_time_s: float
    sec_per_epoch: float
    infer_fps: float
    has_weights: bool
    num_heatmaps: int
    num_tiff_maps: int
    size_mb: float
    result_path: str


@dataclass(frozen=True)
class DefectRecord:
    """Per-defect metrics paired with the run's core hyperparameters."""

    experiment_name: str
    category: str
    defect_type: str
    count: int
    recall_at_opt_th: float
    image_auroc: float
    au_pro: float
    mean_score: float
    std_score: float
    hp_source: str
    loss_type: str
    num_stages: int | None
    base_channels: int | None
    latent_dim: int | None
    spatial_dropout: float | None
    denoise_prob: float | None
    texture_damping: float | None


def _safe_float(val: Any, default: float = _NAN) -> float:
    if val is None:
        return default
    try:
        out = float(val)
        return out
    except (TypeError, ValueError):
        return default


def _safe_int(val: Any) -> int | None:
    if val is None or isinstance(val, bool):
        return None
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _safe_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return bool(val)


def _format_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _walk_artifact_stats(root: Path) -> tuple[int, int, int, float]:
    """Return (num_heatmaps, num_tiff_maps, total_bytes, latest_mtime) for a folder."""
    num_heatmaps = 0
    num_tiff_maps = 0
    total_bytes = 0
    try:
        latest_mtime = root.stat().st_mtime
    except OSError:
        return 0, 0, 0, 0.0

    for dirpath, _, filenames in os.walk(root):
        dir_rel = Path(dirpath).relative_to(root)
        in_heatmaps = "heatmaps" in dir_rel.parts
        in_anomaly_maps = "anomaly_maps" in dir_rel.parts
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            try:
                st = os.stat(fpath)
            except OSError:
                continue
            total_bytes += st.st_size
            if st.st_mtime > latest_mtime:
                latest_mtime = st.st_mtime
            low = fname.lower()
            if in_heatmaps and low.endswith(".png"):
                num_heatmaps += 1
            elif in_anomaly_maps and low.endswith((".tiff", ".tif")):
                num_tiff_maps += 1

    return num_heatmaps, num_tiff_maps, total_bytes, latest_mtime


def parse_experiment_name_hyperparams(name: str) -> dict[str, Any]:
    """Infer model/loss hyperparameters from standard test.sh experiment names."""
    match = _NAME_HP_PATTERN.match(name)
    if not match:
        return {}
    loss_raw = match.group("loss")
    loss_norm = _LOSS_ALIASES.get(loss_raw, loss_raw)
    res_raw = match.group("res")
    parsed: dict[str, Any] = {
        "loss_type": loss_norm,
        "num_stages": int(match.group("stages")),
    }
    if res_raw == "fullres":
        parsed["full_res"] = True
    if match.group("base_ch") is not None:
        parsed["base_channels"] = int(match.group("base_ch"))
    if match.group("latent") is not None:
        parsed["latent_dim"] = int(match.group("latent"))
    return parsed


def _read_checkpoint_meta(weights_path: Path) -> dict[str, Any]:
    """Best-effort inspection of meta dict inside weights/model.pt without failing."""
    if not weights_path.is_file():
        return {}
    try:
        import torch

        payload = torch.load(weights_path, map_location="cpu", weights_only=True)
        meta = payload.get("meta", {}) if isinstance(payload, dict) else {}
        return dict(meta) if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _compute_score_stats(csv_path: Path) -> tuple[float, float, float, float]:
    """Compute (good_mean, good_max, defect_mean, defect_min) from per_sample_scores.csv."""
    if not csv_path.is_file():
        return _NAN, _NAN, _NAN, _NAN

    good_scores: list[float] = []
    defect_scores: list[float] = []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                score = _safe_float(row.get("image_score"))
                label = _safe_int(row.get("label"))
                if math.isnan(score) or label is None:
                    continue
                if label == 0:
                    good_scores.append(score)
                else:
                    defect_scores.append(score)
    except OSError:
        return _NAN, _NAN, _NAN, _NAN

    good_mean = sum(good_scores) / len(good_scores) if good_scores else _NAN
    good_max = max(good_scores) if good_scores else _NAN
    defect_mean = sum(defect_scores) / len(defect_scores) if defect_scores else _NAN
    defect_min = min(defect_scores) if defect_scores else _NAN
    return good_mean, good_max, defect_mean, defect_min


def _discover_category_dirs(exp_dir: Path) -> list[Path]:
    """Return sorted nested category directories containing metrics.json."""
    cat_dirs: list[Path] = []
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir() or child.name in _ARTIFACT_SUBDIRS or child.name.startswith("."):
            continue
        if (child / "metrics.json").is_file():
            cat_dirs.append(child)
    return cat_dirs


def inspect_results_folders(results_dir: Path) -> list[FolderInventoryEntry]:
    """Summarize all top-level experiment folders and their contents under results_dir."""
    if not results_dir.is_dir():
        return []

    entries: list[FolderInventoryEntry] = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir() or exp_dir.name.startswith("."):
            continue

        top_metrics = (exp_dir / "metrics.json").is_file()
        has_summary = (exp_dir / "summary_metrics.json").is_file()
        nested_cat_dirs = [] if top_metrics else _discover_category_dirs(exp_dir)

        if top_metrics:
            status = "complete"
            try:
                data = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
                categories: tuple[str, ...] = (str(data.get("category", "")),)
            except Exception:
                categories = ()
            has_scores_csv = (exp_dir / "per_sample_scores.csv").is_file()
            has_weights = (exp_dir / "weights" / "model.pt").is_file()
            has_plots = all((exp_dir / p).is_file() for p in _PLOT_NAMES)
        elif nested_cat_dirs:
            status = "complete" if has_summary else "partial_multi"
            categories = tuple(d.name for d in nested_cat_dirs)
            has_scores_csv = any((d / "per_sample_scores.csv").is_file() for d in nested_cat_dirs)
            has_weights = any((d / "weights" / "model.pt").is_file() for d in nested_cat_dirs)
            has_plots = any(all((d / p).is_file() for p in _PLOT_NAMES) for d in nested_cat_dirs)
        else:
            status = "incomplete"
            categories = ()
            has_scores_csv = (exp_dir / "per_sample_scores.csv").is_file()
            has_weights = (exp_dir / "weights" / "model.pt").is_file()
            has_plots = any((exp_dir / p).is_file() for p in _PLOT_NAMES)

        num_heatmaps, num_tiff_maps, total_bytes, latest_mtime = _walk_artifact_stats(exp_dir)
        entries.append(
            FolderInventoryEntry(
                folder_name=exp_dir.name,
                folder_path=str(exp_dir),
                status=status,
                categories=categories,
                has_metrics=top_metrics or bool(nested_cat_dirs),
                has_summary=has_summary,
                has_scores_csv=has_scores_csv,
                has_weights=has_weights,
                has_plots=has_plots,
                num_heatmaps=num_heatmaps,
                num_tiff_maps=num_tiff_maps,
                size_mb=total_bytes / (1024.0 * 1024.0),
                updated_at=_format_timestamp(latest_mtime),
            )
        )
    return entries


def _resolve_hyperparams(
    payload: dict[str, Any],
    exp_dir: Path,
    name_candidates: Sequence[str],
) -> tuple[str, dict[str, Any]]:
    """Resolve hyperparameters from recorded JSON or fallback name/calibration/checkpoint."""
    recorded = payload.get("hyperparams")
    if isinstance(recorded, dict) and recorded:
        return "recorded", dict(recorded)

    merged: dict[str, Any] = {}
    for candidate in reversed(name_candidates):
        if candidate:
            merged.update(parse_experiment_name_hyperparams(candidate))

    epoch_losses = payload.get("epoch_losses", [])
    if isinstance(epoch_losses, list) and epoch_losses:
        merged.setdefault("epochs", len(epoch_losses))

    calib = payload.get("calibration")
    if isinstance(calib, dict) and "texture_damping" in calib:
        merged["texture_damping"] = calib["texture_damping"]

    needs_arch = any(k not in merged for k in ("num_stages", "base_channels", "latent_dim"))
    if needs_arch:
        ckpt_meta = _read_checkpoint_meta(exp_dir / "weights" / "model.pt")
        for key in ("num_stages", "base_channels", "latent_dim", "spatial_dropout"):
            if key in ckpt_meta:
                merged.setdefault(key, ckpt_meta[key])

    hp_source = "inferred" if merged else "none"
    return hp_source, merged


def _build_record_from_run_dir(
    run_dir: Path,
    experiment_name: str,
    parent_experiment: str,
) -> tuple[ExperimentRecord, list[DefectRecord]] | None:
    metrics_path = run_dir / "metrics.json"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    m = payload.get("metrics", {})
    if not isinstance(m, dict):
        m = {}
    calib = payload.get("calibration", {})
    if not isinstance(calib, dict):
        calib = {}

    hp_source, hp = _resolve_hyperparams(
        payload,
        run_dir,
        name_candidates=[experiment_name, parent_experiment],
    )

    epoch_losses_raw = payload.get("epoch_losses", [])
    epoch_losses = (
        [float(v) for v in epoch_losses_raw if isinstance(v, (int, float))]
        if isinstance(epoch_losses_raw, list)
        else []
    )
    epochs_completed = len(epoch_losses)
    initial_loss = epoch_losses[0] if epoch_losses else _NAN
    final_loss = epoch_losses[-1] if epoch_losses else _NAN
    best_loss = min(epoch_losses) if epoch_losses else _NAN

    train_time_s = _safe_float(payload.get("train_time_s"))
    sec_per_epoch = (
        train_time_s / float(epochs_completed)
        if epochs_completed > 0 and not math.isnan(train_time_s)
        else _NAN
    )

    good_mean, good_max, defect_mean, defect_min = _compute_score_stats(
        run_dir / "per_sample_scores.csv"
    )
    num_heatmaps, num_tiff_maps, total_bytes, latest_mtime = _walk_artifact_stats(run_dir)
    if parent_experiment and num_tiff_maps == 0:
        parent_cat = str(payload.get("category", run_dir.name))
        tiff_dir = run_dir.parent / "anomaly_maps" / parent_cat
        if tiff_dir.is_dir():
            _, num_tiff_maps, _, _ = _walk_artifact_stats(tiff_dir)

    img_sz = payload.get("image_size")
    image_size_str = (
        f"{img_sz[0]}x{img_sz[1]}"
        if isinstance(img_sz, list) and len(img_sz) == 2
        else ""
    )
    category = str(payload.get("category", "")).strip() or (
        run_dir.name if parent_experiment else ""
    )

    rec = ExperimentRecord(
        experiment_name=experiment_name,
        parent_experiment=parent_experiment,
        category=category,
        status="complete",
        updated_at=_format_timestamp(latest_mtime),
        device=str(payload.get("device", "")),
        image_size=image_size_str,
        num_train=_safe_int(payload.get("num_train")),
        num_test=_safe_int(payload.get("num_test")),
        hp_source=hp_source,
        loss_type=str(hp.get("loss_type", "")),
        num_stages=_safe_int(hp.get("num_stages")),
        base_channels=_safe_int(hp.get("base_channels")),
        latent_dim=_safe_int(hp.get("latent_dim")),
        epochs=_safe_int(hp.get("epochs")) or (epochs_completed if epochs_completed else None),
        batch_size=_safe_int(hp.get("batch_size")),
        lr=_safe_float(hp.get("lr")) if "lr" in hp else None,
        weight_decay=_safe_float(hp.get("weight_decay")) if "weight_decay" in hp else None,
        gaussian_sigma=(
            _safe_float(hp.get("gaussian_sigma")) if "gaussian_sigma" in hp else None
        ),
        score_top_k_pct=(
            _safe_float(hp.get("score_top_k_pct")) if "score_top_k_pct" in hp else None
        ),
        spatial_dropout=(
            _safe_float(hp.get("spatial_dropout")) if "spatial_dropout" in hp else None
        ),
        denoise_prob=_safe_float(hp.get("denoise_prob")) if "denoise_prob" in hp else None,
        texture_damping=(
            _safe_float(hp.get("texture_damping")) if "texture_damping" in hp else None
        ),
        loss_floor=_safe_float(hp.get("loss_floor")) if "loss_floor" in hp else None,
        full_res=_safe_bool(hp.get("full_res")),
        use_amp=_safe_bool(hp.get("use_amp")),
        pro_integration_limit=_safe_float(
            m.get("pro_integration_limit", hp.get("pro_integration_limit", 0.3)), default=0.3
        ),
        seed=_safe_int(hp.get("seed")),
        calibrated=_safe_bool(calib.get("calibrated")),
        calib_err_mean=_safe_float(calib.get("err_mean")),
        calib_err_std=_safe_float(calib.get("err_std")),
        au_pro=_safe_float(m.get("au_pro")),
        classification_au_roc_max=_safe_float(m.get("classification_au_roc_max")),
        image_auroc=_safe_float(m.get("image_auroc")),
        image_ap=_safe_float(m.get("image_ap")),
        f1_max=_safe_float(m.get("f1_max")),
        optimal_threshold=_safe_float(m.get("optimal_threshold")),
        accuracy_at_opt_th=_safe_float(m.get("accuracy_at_opt_th")),
        pixel_auroc=_safe_float(m.get("pixel_auroc")),
        pixel_ap=_safe_float(m.get("pixel_ap")),
        pixel_f1_max=_safe_float(m.get("pixel_f1_max")),
        good_mean_score=good_mean,
        good_max_score=good_max,
        defect_mean_score=defect_mean,
        defect_min_score=defect_min,
        epochs_completed=epochs_completed,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        train_time_s=train_time_s,
        sec_per_epoch=sec_per_epoch,
        infer_fps=_safe_float(payload.get("infer_fps")),
        has_weights=(run_dir / "weights" / "model.pt").is_file(),
        num_heatmaps=num_heatmaps,
        num_tiff_maps=num_tiff_maps,
        size_mb=total_bytes / (1024.0 * 1024.0),
        result_path=str(run_dir),
    )

    defect_records: list[DefectRecord] = []
    per_defect = m.get("per_defect", {})
    if isinstance(per_defect, dict):
        for defect_name, dstats in per_defect.items():
            if not isinstance(dstats, dict):
                continue
            defect_records.append(
                DefectRecord(
                    experiment_name=experiment_name,
                    category=category,
                    defect_type=str(defect_name),
                    count=int(dstats.get("count", 0)),
                    recall_at_opt_th=_safe_float(dstats.get("recall_at_opt_th")),
                    image_auroc=_safe_float(dstats.get("image_auroc")),
                    au_pro=_safe_float(dstats.get("au_pro")),
                    mean_score=_safe_float(dstats.get("mean_score")),
                    std_score=_safe_float(dstats.get("std_score")),
                    hp_source=hp_source,
                    loss_type=rec.loss_type,
                    num_stages=rec.num_stages,
                    base_channels=rec.base_channels,
                    latent_dim=rec.latent_dim,
                    spatial_dropout=rec.spatial_dropout,
                    denoise_prob=rec.denoise_prob,
                    texture_damping=rec.texture_damping,
                )
            )

    return rec, defect_records


def _build_incomplete_record(inv: FolderInventoryEntry) -> ExperimentRecord:
    parsed = parse_experiment_name_hyperparams(inv.folder_name)
    inferred_cat = inv.folder_name.split("_", 1)[0] if "_" in inv.folder_name else ""
    return ExperimentRecord(
        experiment_name=inv.folder_name,
        parent_experiment="",
        category=inferred_cat,
        status=inv.status,
        updated_at=inv.updated_at,
        device="",
        image_size="",
        num_train=None,
        num_test=None,
        hp_source="inferred" if parsed else "none",
        loss_type=str(parsed.get("loss_type", "")),
        num_stages=_safe_int(parsed.get("num_stages")),
        base_channels=_safe_int(parsed.get("base_channels")),
        latent_dim=_safe_int(parsed.get("latent_dim")),
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        gaussian_sigma=None,
        score_top_k_pct=None,
        spatial_dropout=None,
        denoise_prob=None,
        texture_damping=None,
        loss_floor=None,
        full_res=_safe_bool(parsed.get("full_res")),
        use_amp=None,
        pro_integration_limit=0.3,
        seed=None,
        calibrated=None,
        calib_err_mean=_NAN,
        calib_err_std=_NAN,
        au_pro=_NAN,
        classification_au_roc_max=_NAN,
        image_auroc=_NAN,
        image_ap=_NAN,
        f1_max=_NAN,
        optimal_threshold=_NAN,
        accuracy_at_opt_th=_NAN,
        pixel_auroc=_NAN,
        pixel_ap=_NAN,
        pixel_f1_max=_NAN,
        good_mean_score=_NAN,
        good_max_score=_NAN,
        defect_mean_score=_NAN,
        defect_min_score=_NAN,
        epochs_completed=0,
        initial_loss=_NAN,
        final_loss=_NAN,
        best_loss=_NAN,
        train_time_s=_NAN,
        sec_per_epoch=_NAN,
        infer_fps=_NAN,
        has_weights=inv.has_weights,
        num_heatmaps=inv.num_heatmaps,
        num_tiff_maps=inv.num_tiff_maps,
        size_mb=inv.size_mb,
        result_path=inv.folder_path,
    )


def scan_results(
    results_dir: Path,
    category: str | None = None,
    include_incomplete: bool = False,
) -> tuple[list[FolderInventoryEntry], list[ExperimentRecord], list[DefectRecord]]:
    """Scan results_dir and return (inventory, experiment_records, defect_records)."""
    inventory = inspect_results_folders(results_dir)
    records: list[ExperimentRecord] = []
    defects: list[DefectRecord] = []

    for inv in inventory:
        exp_dir = Path(inv.folder_path)
        if (exp_dir / "metrics.json").is_file():
            built = _build_record_from_run_dir(
                exp_dir,
                experiment_name=exp_dir.name,
                parent_experiment="",
            )
            if built is not None:
                rec, defs = built
                if category is None or rec.category == category:
                    records.append(rec)
                    defects.extend(defs)
        elif inv.status in {"complete", "partial_multi"}:
            for sub in _discover_category_dirs(exp_dir):
                built = _build_record_from_run_dir(
                    sub,
                    experiment_name=f"{exp_dir.name}/{sub.name}",
                    parent_experiment=exp_dir.name,
                )
                if built is not None:
                    rec, defs = built
                    if category is None or rec.category == category:
                        records.append(rec)
                        defects.extend(defs)
        elif include_incomplete:
            inc_rec = _build_incomplete_record(inv)
            if category is None or inc_rec.category == category:
                records.append(inc_rec)

    return inventory, records, defects


def sort_experiment_records(
    records: list[ExperimentRecord],
    sort_by: str = "au_pro",
    group_by_category: bool = True,
) -> list[ExperimentRecord]:
    """Sort experiment records with NaNs/incomplete runs always placed at the bottom."""
    valid_fields = {f.name for f in fields(ExperimentRecord)}
    if sort_by not in valid_fields:
        raise ValueError(
            f"Unknown sort_by column {sort_by!r}; expected one of {sorted(valid_fields)}"
        )

    def _desc_metric_key(val: Any) -> tuple[int, float]:
        if val is None:
            return (1, 0.0)
        if isinstance(val, (int, float)):
            fval = float(val)
            if math.isnan(fval):
                return (1, 0.0)
            return (0, -fval)
        return (1, 0.0)

    def _key(rec: ExperimentRecord) -> tuple[Any, ...]:
        incomplete_rank = 0 if rec.status == "complete" else 1
        cat_key = rec.category or "~"
        if sort_by == "updated_at":
            # Sort ISO timestamps newest-first via negated character ordinals
            ts_inv = tuple(-ord(ch) for ch in rec.updated_at)
            cat_sort = cat_key if group_by_category else ""
            return (incomplete_rank, cat_sort, ts_inv, rec.experiment_name)
        val = getattr(rec, sort_by)
        if isinstance(val, str):
            primary: Any = (0 if val else 1, val)
        else:
            primary = _desc_metric_key(val)
        secondary = _desc_metric_key(rec.classification_au_roc_max)
        if group_by_category:
            return (incomplete_rank, cat_key, primary, secondary, rec.experiment_name)
        return (incomplete_rank, primary, secondary, cat_key, rec.experiment_name)

    return sorted(records, key=_key)


def _csv_cell(val: Any) -> Any:
    if val is None:
        return ""
    if isinstance(val, float):
        if math.isnan(val):
            return ""
        return f"{val:.6g}"
    return val


def _write_dataclass_csv(rows: Sequence[Any], fieldnames: Sequence[str], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            raw = asdict(row)
            writer.writerow({k: _csv_cell(raw.get(k)) for k in fieldnames})
    return out_path


def write_experiments_csv(records: list[ExperimentRecord], out_path: Path) -> Path:
    """Write all ExperimentRecord rows to a clean CSV file."""
    cols = [f.name for f in fields(ExperimentRecord)]
    return _write_dataclass_csv(records, cols, out_path)


def write_per_defect_csv(defects: list[DefectRecord], out_path: Path) -> Path:
    """Write all per-defect breakdown rows to a clean CSV file."""
    cols = [f.name for f in fields(DefectRecord)]
    return _write_dataclass_csv(defects, cols, out_path)


def export_results_csvs(
    results_dir: Path,
    category: str | None = None,
    sort_by: str = "au_pro",
    include_incomplete: bool = False,
    csv_out: Path | None = None,
    defect_csv_out: Path | None = None,
) -> tuple[Path, Path]:
    """Scan results_dir and export experiment + per-defect comparison CSVs."""
    _, records, defects = scan_results(
        results_dir,
        category=category,
        include_incomplete=include_incomplete,
    )
    sorted_records = sort_experiment_records(records, sort_by=sort_by, group_by_category=True)
    exp_csv = csv_out if csv_out is not None else results_dir / "all_experiments_metrics.csv"
    def_csv = (
        defect_csv_out
        if defect_csv_out is not None
        else results_dir / "all_per_defect_metrics.csv"
    )
    write_experiments_csv(sorted_records, exp_csv)
    write_per_defect_csv(defects, def_csv)
    return exp_csv, def_csv


def _fmt_num(val: float | None, decimals: int = 4, width: int = 8) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'-':>{width}}"
    return f"{val:>{width}.{decimals}f}"


def _fmt_opt_int(val: int | None, width: int = 4) -> str:
    return f"{'-':>{width}}" if val is None else f"{val:>{width}d}"


def _fmt_opt_flt(val: float | None, width: int = 5) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return f"{'-':>{width}}"
    return f"{val:>{width}g}"


def format_inventory_table(entries: list[FolderInventoryEntry]) -> str:
    """Format an aligned table listing all folders under results/ and their artifacts."""
    if not entries:
        return "No experiment directories found under results/."

    name_w = max(24, min(42, max(len(e.folder_name) for e in entries) + 2))
    header = (
        f"{'Folder':<{name_w}} {'Status':<13} {'Cats':>4} {'Metrics':>7} "
        f"{'CSV':>3} {'Wts':>3} {'Plots':>5} {'HMap':>5} {'TIFF':>5} "
        f"{'Size(MB)':>9} {'Updated':<19}"
    )
    sep = "-" * len(header)
    lines = ["Results Folder Inventory", sep, header, sep]
    for e in entries:
        name = (
            e.folder_name
            if len(e.folder_name) <= name_w
            else e.folder_name[: name_w - 3] + "..."
        )
        lines.append(
            f"{name:<{name_w}} {e.status:<13} {len(e.categories):>4d} "
            f"{'yes' if e.has_metrics else 'no':>7} "
            f"{'yes' if e.has_scores_csv else 'no':>3} "
            f"{'yes' if e.has_weights else 'no':>3} "
            f"{'yes' if e.has_plots else 'no':>5} "
            f"{e.num_heatmaps:>5d} {e.num_tiff_maps:>5d} "
            f"{e.size_mb:>9.1f} {e.updated_at:<19}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_leaderboard_table(records: list[ExperimentRecord], top: int | None = None) -> str:
    """Format an aligned hyperparameter & metric comparison leaderboard."""
    shown = records[:top] if top is not None and top > 0 else records
    if not shown:
        return "No completed experiments matched the filter."

    name_w = max(26, min(42, max(len(r.experiment_name) for r in shown) + 2))
    header = (
        f"{'Experiment':<{name_w}} {'Category':<10} {'Loss':<10} {'S':>2} {'C':>3} {'Z':>3} "
        f"{'Ep':>3} {'Dmp':>4} {'Dns':>4} {'AU-PRO':>7} {'ROC(max)':>8} "
        f"{'ROC(k%)':>7} {'PixROC':>7} {'PixAP':>6} {'GoodMu':>7} {'DefMu':>7} {'s/ep':>6}"
    )
    sep = "-" * len(header)
    lines = ["Experiment Metrics & Hyperparameter Leaderboard", sep, header, sep]
    for r in shown:
        name = (
            r.experiment_name
            if len(r.experiment_name) <= name_w
            else r.experiment_name[: name_w - 3] + "..."
        )
        cat = r.category or "-"
        loss = r.loss_type or "-"
        lines.append(
            f"{name:<{name_w}} {cat:<10} {loss:<10} "
            f"{_fmt_opt_int(r.num_stages, 2)} {_fmt_opt_int(r.base_channels, 3)} "
            f"{_fmt_opt_int(r.latent_dim, 3)} {_fmt_opt_int(r.epochs, 3)} "
            f"{_fmt_opt_flt(r.texture_damping, 4)} {_fmt_opt_flt(r.denoise_prob, 4)} "
            f"{_fmt_num(r.au_pro, 4, 7)} {_fmt_num(r.classification_au_roc_max, 4, 8)} "
            f"{_fmt_num(r.image_auroc, 4, 7)} {_fmt_num(r.pixel_auroc, 4, 7)} "
            f"{_fmt_num(r.pixel_ap, 4, 6)} {_fmt_num(r.good_mean_score, 3, 7)} "
            f"{_fmt_num(r.defect_mean_score, 3, 7)} {_fmt_num(r.sec_per_epoch, 2, 6)}"
        )
    lines.append(sep)
    return "\n".join(lines)


def format_category_progress_summary(records: list[ExperimentRecord]) -> str:
    """Summarize best run per category and delta vs runner-up to guide hyperparameter choices."""
    complete = [r for r in records if r.status == "complete" and r.category]
    if not complete:
        return ""

    by_cat: dict[str, list[ExperimentRecord]] = {}
    for rec in complete:
        by_cat.setdefault(rec.category, []).append(rec)

    lines = ["Per-Category Best Run & Direction Summary", "-" * 78]
    for cat in sorted(by_cat):
        cat_runs = sort_experiment_records(by_cat[cat], sort_by="au_pro", group_by_category=True)
        best = cat_runs[0]
        if len(cat_runs) == 1:
            lines.append(
                f"  {cat:<12}: 1 run ({best.experiment_name}) -> "
                f"AU-PRO={best.au_pro:.4f}, ROC(max)={best.classification_au_roc_max:.4f}, "
                f"PixROC={best.pixel_auroc:.4f}"
            )
            continue

        prev = cat_runs[1]
        d_pro = best.au_pro - prev.au_pro
        d_roc = best.classification_au_roc_max - prev.classification_au_roc_max
        d_pix = best.pixel_auroc - prev.pixel_auroc
        lines.append(
            f"  {cat:<12}: {len(cat_runs)} runs | best={best.experiment_name} | "
            f"AU-PRO={best.au_pro:.4f} ({d_pro:+.4f}), "
            f"ROC(max)={best.classification_au_roc_max:.4f} ({d_roc:+.4f}), "
            f"PixROC={best.pixel_auroc:.4f} ({d_pix:+.4f} vs {prev.experiment_name})"
        )
    lines.append("-" * 78)
    return "\n".join(lines)


def summarize_and_export(
    results_dir: Path,
    category: str | None = None,
    sort_by: str = "au_pro",
    include_incomplete: bool = False,
    csv_out: Path | None = None,
    defect_csv_out: Path | None = None,
    write_csv: bool = True,
    top: int | None = None,
) -> tuple[list[FolderInventoryEntry], list[ExperimentRecord], list[DefectRecord]]:
    """Print inventory/leaderboard tables and optionally write comparison CSV files."""
    inventory, records, defects = scan_results(
        results_dir,
        category=category,
        include_incomplete=include_incomplete,
    )
    sorted_records = sort_experiment_records(
        records,
        sort_by=sort_by,
        group_by_category=(category is not None or sort_by == "au_pro"),
    )

    print()
    print(format_inventory_table(inventory))
    print()
    print(format_leaderboard_table(sorted_records, top=top))
    progress = format_category_progress_summary(records)
    if progress:
        print()
        print(progress)

    if write_csv:
        exp_path = csv_out if csv_out is not None else results_dir / "all_experiments_metrics.csv"
        def_path = (
            defect_csv_out
            if defect_csv_out is not None
            else results_dir / "all_per_defect_metrics.csv"
        )
        write_experiments_csv(sorted_records, exp_path)
        write_per_defect_csv(defects, def_path)
        print(f"\nSaved experiment metrics CSV : {exp_path} ({len(sorted_records)} rows)")
        print(f"Saved per-defect metrics CSV : {def_path} ({len(defects)} rows)")

    return inventory, sorted_records, defects


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for inspecting results/."""
    parser = argparse.ArgumentParser(
        prog="python -m src.results",
        description="Inspect results/ folder contents, compare hyperparameters, and export CSVs.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Root directory containing experiment result folders",
    )
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        default=None,
        help="Filter leaderboard and CSVs to a single MVTec category",
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="au_pro",
        help="Metric/column name to rank experiments by (e.g. au_pro, classification_au_roc_max)",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include incomplete/empty experiment directories in CSV and leaderboard",
    )
    parser.add_argument(
        "--csv-out",
        type=str,
        default=None,
        help="Output path for all_experiments_metrics.csv",
    )
    parser.add_argument(
        "--defect-csv-out",
        type=str,
        default=None,
        help="Output path for all_per_defect_metrics.csv",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Print terminal summary tables without writing CSV files",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit terminal leaderboard to top N rows",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for inspecting results/ and exporting comparison CSVs."""
    args = build_parser().parse_args(argv)
    summarize_and_export(
        results_dir=Path(args.results_dir),
        category=args.category,
        sort_by=args.sort_by,
        include_incomplete=bool(args.include_incomplete),
        csv_out=Path(args.csv_out) if args.csv_out else None,
        defect_csv_out=Path(args.defect_csv_out) if args.defect_csv_out else None,
        write_csv=not bool(args.no_csv),
        top=args.top,
    )


if __name__ == "__main__":
    main()
