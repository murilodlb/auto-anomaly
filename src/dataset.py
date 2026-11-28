"""Data-oriented MVTec AD disk loader and contiguous batch iterator."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.config import DatasetConfig

# Official MVTec AD counts per category: (train_good, test_total, test_defective)
EXPECTED_MVTEC_COUNTS: dict[str, tuple[int, int, int]] = {
    "bottle": (209, 83, 63),
    "cable": (224, 150, 92),
    "capsule": (219, 132, 109),
    "carpet": (280, 117, 89),
    "grid": (264, 78, 57),
    "hazelnut": (391, 110, 70),
    "leather": (245, 124, 92),
    "metal_nut": (220, 115, 93),
    "pill": (267, 167, 141),
    "screw": (320, 160, 119),
    "tile": (230, 117, 84),
    "toothbrush": (60, 42, 30),
    "transistor": (213, 100, 40),
    "wood": (247, 79, 60),
    "zipper": (240, 151, 119),
}


def _check_mvtec_split_sanity(
    category: str,
    train_paths: Sequence[Path],
    test_records: Sequence[tuple[Path, Path | None, int, str]],
) -> list[str]:
    """Warn if an MVTec AD category has unexpected image counts or numbering gaps."""
    warnings_out: list[str] = []
    n_train = len(train_paths)
    n_test = len(test_records)
    n_defective = sum(1 for rec in test_records if rec[2] == 1)

    expected = EXPECTED_MVTEC_COUNTS.get(category)
    if expected is not None and (n_train, n_test, n_defective) != expected:
        exp_tr, exp_te, exp_def = expected
        swapped_matches = [
            other
            for other, counts in EXPECTED_MVTEC_COUNTS.items()
            if other != category and counts == (n_train, n_test, n_defective)
        ]
        swap_hint = (
            f" — counts match category '{swapped_matches[0]}' (possible swapped directory!)"
            if swapped_matches
            else " — check for incomplete extraction/transfer."
        )
        msg = (
            f"[WARN] Dataset sanity check failed for MVTec AD category '{category}': "
            f"found train={n_train}, test={n_test}, gt={n_defective}; "
            f"expected train={exp_tr}, test={exp_te}, gt={exp_def}{swap_hint}"
        )
        warnings_out.append(msg)

    if train_paths and all(p.stem.isdigit() for p in train_paths):
        indices = sorted(int(p.stem) for p in train_paths)
        max_idx = indices[-1]
        missing = sorted(set(range(0, max_idx + 1)) - set(indices))
        if missing:
            first_start = missing[0]
            first_end = first_start
            for val in missing[1:]:
                if val == first_end + 1:
                    first_end = val
                else:
                    break
            range_str = (
                f"{first_start}..{first_end}" if first_end > first_start else f"{first_start}"
            )
            msg = (
                f"[WARN] Non-contiguous training filenames in '{category}/train/good': "
                f"missing {len(missing)} index(es) in 0..{max_idx} "
                f"(first gap: {range_str})."
            )
            warnings_out.append(msg)

    for msg in warnings_out:
        print(msg)
    return warnings_out


@dataclass(slots=True)
class MVTecSplit:
    """Contiguous in-memory representation of an MVTec AD split."""

    image_paths: list[Path]
    images: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    defect_types: list[str]
    original_sizes: list[tuple[int, int]] = field(default_factory=list)


def _read_and_resize_image(
    path: Path,
    target_hw: tuple[int, int],
    grayscale: bool,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Load an image from disk and return ((H, W, C) uint8, (src_h, src_w))."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    raw = cv2.imread(str(path), flag)
    if raw is None:
        raise ValueError(f"Failed to read image: {path}")

    if not grayscale:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    src_h, src_w = int(raw.shape[0]), int(raw.shape[1])
    dst_h, dst_w = target_hw
    if (src_h, src_w) != (dst_h, dst_w):
        interp = cv2.INTER_AREA if (src_h > dst_h or src_w > dst_w) else cv2.INTER_LINEAR
        raw = cv2.resize(raw, (dst_w, dst_h), interpolation=interp)

    if grayscale and raw.ndim == 2:
        raw = raw[:, :, None]

    return np.ascontiguousarray(raw, dtype=np.uint8), (src_h, src_w)


def _read_and_resize_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    """Load a binary ground-truth mask and resize with nearest-neighbor interpolation."""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise ValueError(f"Failed to read mask: {path}")

    dst_h, dst_w = target_hw
    if raw.shape[:2] != (dst_h, dst_w):
        raw = cv2.resize(raw, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)

    return np.ascontiguousarray((raw > 127).astype(np.uint8))


def _load_split(
    records: Sequence[tuple[Path, Path | None, int, str]],
    config: DatasetConfig,
) -> MVTecSplit:
    """Decode and pack a deterministic list of sample records into contiguous arrays."""
    n = len(records)
    h, w = config.image_size
    c = config.channels

    image_paths = [rec[0] for rec in records]
    defect_types = [rec[3] for rec in records]
    labels = np.array([rec[2] for rec in records], dtype=np.int32)
    images = np.empty((n, h, w, c), dtype=np.uint8)
    masks = np.zeros((n, h, w), dtype=np.uint8)
    original_sizes: list[tuple[int, int]] = [(0, 0)] * n

    def _load_one(idx: int) -> tuple[int, np.ndarray, np.ndarray | None, tuple[int, int]]:
        img_path, mask_path, _, _ = records[idx]
        img, orig_hw = _read_and_resize_image(img_path, (h, w), config.grayscale)
        mask = _read_and_resize_mask(mask_path, (h, w)) if mask_path is not None else None
        return idx, img, mask, orig_hw

    if config.num_workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
            for idx, img, mask, orig_hw in pool.map(_load_one, range(n)):
                images[idx] = img
                original_sizes[idx] = orig_hw
                if mask is not None:
                    masks[idx] = mask
    else:
        for idx in range(n):
            _, img, mask, orig_hw = _load_one(idx)
            images[idx] = img
            original_sizes[idx] = orig_hw
            if mask is not None:
                masks[idx] = mask

    return MVTecSplit(
        image_paths=image_paths,
        images=np.ascontiguousarray(images),
        masks=np.ascontiguousarray(masks),
        labels=np.ascontiguousarray(labels),
        defect_types=defect_types,
        original_sizes=original_sizes,
    )


def load_mvtec_category(config: DatasetConfig) -> tuple[MVTecSplit, MVTecSplit]:
    """Load MVTec AD train and test splits into contiguous numpy containers."""
    train_good_dir = config.train_dir / "good"
    train_paths = sorted(train_good_dir.glob("*.png"))
    if not train_paths:
        raise ValueError(f"No training images found in {train_good_dir}")
    train_records = [(p, None, 0, "good") for p in train_paths]

    test_records: list[tuple[Path, Path | None, int, str]] = []
    defect_dirs = sorted(d for d in config.test_dir.iterdir() if d.is_dir())
    for defect_dir in defect_dirs:
        defect_name = defect_dir.name
        img_paths = sorted(defect_dir.glob("*.png"))
        for img_path in img_paths:
            if defect_name == "good":
                test_records.append((img_path, None, 0, "good"))
            else:
                mask_path = config.ground_truth_dir / defect_name / f"{img_path.stem}_mask.png"
                if not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Missing ground-truth mask for defective image {img_path}: {mask_path}"
                    )
                test_records.append((img_path, mask_path, 1, defect_name))

    if not test_records:
        raise ValueError(f"No test images found in {config.test_dir}")

    _check_mvtec_split_sanity(config.category, train_paths, test_records)

    return _load_split(train_records, config), _load_split(test_records, config)


def iter_batches(
    images: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    seed: int = 42,
) -> Iterator[np.ndarray]:
    """Yield contiguous (B, H, W, C) image batches from an in-memory array."""
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    num_samples = images.shape[0]
    if num_samples == 0:
        return

    if shuffle:
        indices = np.random.default_rng(seed).permutation(num_samples)
        for start in range(0, num_samples, batch_size):
            yield np.ascontiguousarray(images[indices[start : start + batch_size]])
    else:
        for start in range(0, num_samples, batch_size):
            yield np.ascontiguousarray(images[start : start + batch_size])
