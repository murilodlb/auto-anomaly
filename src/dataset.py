"""Data-oriented MVTec AD disk loader and contiguous batch iterator."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.config import DatasetConfig


@dataclass(slots=True)
class MVTecSplit:
    """Contiguous in-memory representation of an MVTec AD split."""

    image_paths: list[Path]
    images: np.ndarray
    masks: np.ndarray
    labels: np.ndarray
    defect_types: list[str]


def _read_and_resize_image(
    path: Path,
    target_hw: tuple[int, int],
    grayscale: bool,
) -> np.ndarray:
    """Load an image from disk and resize it to (H, W, C) uint8."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    raw = cv2.imread(str(path), flag)
    if raw is None:
        raise ValueError(f"Failed to read image: {path}")

    if not grayscale:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    src_h, src_w = raw.shape[:2]
    dst_h, dst_w = target_hw
    if (src_h, src_w) != (dst_h, dst_w):
        interp = cv2.INTER_AREA if (src_h > dst_h or src_w > dst_w) else cv2.INTER_LINEAR
        raw = cv2.resize(raw, (dst_w, dst_h), interpolation=interp)

    if grayscale and raw.ndim == 2:
        raw = raw[:, :, None]

    return np.ascontiguousarray(raw, dtype=np.uint8)


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

    def _load_one(idx: int) -> tuple[int, np.ndarray, np.ndarray | None]:
        img_path, mask_path, _, _ = records[idx]
        img = _read_and_resize_image(img_path, (h, w), config.grayscale)
        mask = _read_and_resize_mask(mask_path, (h, w)) if mask_path is not None else None
        return idx, img, mask

    if config.num_workers > 1 and n > 1:
        with ThreadPoolExecutor(max_workers=config.num_workers) as pool:
            for idx, img, mask in pool.map(_load_one, range(n)):
                images[idx] = img
                if mask is not None:
                    masks[idx] = mask
    else:
        for idx in range(n):
            _, img, mask = _load_one(idx)
            images[idx] = img
            if mask is not None:
                masks[idx] = mask

    return MVTecSplit(
        image_paths=image_paths,
        images=np.ascontiguousarray(images),
        masks=np.ascontiguousarray(masks),
        labels=np.ascontiguousarray(labels),
        defect_types=defect_types,
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
