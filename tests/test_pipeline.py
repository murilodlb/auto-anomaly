"""Fast unit and end-to-end smoke tests for the anomaly detection pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
import torch

from src.config import (
    AutoencoderConfig,
    DatasetConfig,
    OutputConfig,
    PipelineConfig,
    RawExtractorConfig,
)
from src.dataset import iter_batches, load_mvtec_category
from src.extractors.raw import RawPixelExtractor
from src.metrics import (
    compute_au_pro,
    compute_classification_roc,
    compute_metrics,
    compute_pro_curve,
    trapezoid,
)
from src.models.autoencoder import ConvAutoencoder, RawAutoencoderModel
from src.run import run_pipeline


def _create_synthetic_mvtec(root: Path, category: str = "carpet") -> Path:
    """Create a tiny valid MVTec-style directory tree on disk."""
    cat_dir = root / category
    train_good = cat_dir / "train" / "good"
    test_good = cat_dir / "test" / "good"
    test_hole = cat_dir / "test" / "hole"
    gt_hole = cat_dir / "ground_truth" / "hole"

    for d in (train_good, test_good, test_hole, gt_hole):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    base_pattern = rng.integers(90, 130, size=(64, 64, 3), dtype=np.uint8)

    for i in range(4):
        img = np.clip(base_pattern.astype(np.int16) + rng.integers(-5, 6, size=(64, 64, 3)), 0, 255)
        cv2.imwrite(str(train_good / f"{i:03d}.png"), img.astype(np.uint8))

    for i in range(2):
        img = np.clip(base_pattern.astype(np.int16) + rng.integers(-5, 6, size=(64, 64, 3)), 0, 255)
        cv2.imwrite(str(test_good / f"{i:03d}.png"), img.astype(np.uint8))

    for i in range(2):
        img = base_pattern.copy()
        img[20:40, 20:40] = 250
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[20:40, 20:40] = 255
        cv2.imwrite(str(test_hole / f"{i:03d}.png"), img)
        cv2.imwrite(str(gt_hole / f"{i:03d}_mask.png"), mask)

    return root


def test_config_validation_and_from_args(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path)
    cfg = PipelineConfig.from_args(
        data_dir=str(data_root),
        category="carpet",
        image_size=[64, 64],
        epochs=2,
        batch_size=2,
        output="unit_test",
        output_dir=str(tmp_path / "results"),
        device="cpu",
    )
    cfg.validate()

    bad_size_cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, image_size=(65, 64)),
        autoencoder=AutoencoderConfig(num_stages=4),
        device="cpu",
    )
    with pytest.raises(ValueError, match="divisible"):
        bad_size_cfg.validate()


def test_dataset_loader_and_missing_mask(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path)
    ds_cfg = DatasetConfig(root_dir=data_root, image_size=(64, 64), num_workers=2)
    train_split, test_split = load_mvtec_category(ds_cfg)

    assert train_split.images.shape == (4, 64, 64, 3)
    assert train_split.images.dtype == np.uint8
    assert train_split.images.flags["C_CONTIGUOUS"]
    assert np.all(train_split.labels == 0)
    assert np.all(train_split.masks == 0)

    assert test_split.images.shape == (4, 64, 64, 3)
    assert test_split.masks.shape == (4, 64, 64)
    assert set(np.unique(test_split.masks)).issubset({0, 1})
    assert list(test_split.labels) == [0, 0, 1, 1]
    assert test_split.defect_types == ["good", "good", "hole", "hole"]

    batches = list(iter_batches(train_split.images, batch_size=3, shuffle=True, seed=7))
    assert [b.shape[0] for b in batches] == [3, 1]
    assert all(b.flags["C_CONTIGUOUS"] for b in batches)

    # Grayscale mode
    gray_cfg = DatasetConfig(root_dir=data_root, image_size=(32, 32), grayscale=True, num_workers=1)
    gray_train, _ = load_mvtec_category(gray_cfg)
    assert gray_train.images.shape == (4, 32, 32, 1)

    # Missing mask raises FileNotFoundError
    (data_root / "carpet" / "ground_truth" / "hole" / "000_mask.png").unlink()
    with pytest.raises(FileNotFoundError, match="Missing ground-truth mask"):
        load_mvtec_category(ds_cfg)


def test_raw_extractor_and_autoencoder_roundtrip(tmp_path: Path) -> None:
    ds_cfg = DatasetConfig(image_size=(64, 64), grayscale=False)
    raw_cfg = RawExtractorConfig(normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.25, 0.25, 0.25))
    device = torch.device("cpu")
    extractor = RawPixelExtractor(ds_cfg, raw_cfg, device)
    assert extractor.out_channels == 3
    assert not extractor.is_unit_range

    batch = np.full((2, 64, 64, 3), 128, dtype=np.uint8)
    tensor = extractor.extract_batch(batch)
    assert tensor.shape == (2, 3, 64, 64)
    assert tensor.dtype == torch.float32
    reconstructed_u8 = extractor.to_uint8_images(tensor)
    assert reconstructed_u8.shape == (2, 64, 64, 3)
    assert np.allclose(reconstructed_u8, 128, atol=1)

    ae = ConvAutoencoder(in_channels=3, base_channels=16, latent_dim=32, num_stages=3)
    out = ae(torch.zeros((2, 3, 64, 64)))
    assert out.shape == (2, 3, 64, 64)

    ae_cfg = AutoencoderConfig(
        latent_dim=16,
        base_channels=8,
        num_stages=3,
        epochs=1,
        gaussian_sigma=1.5,
    )
    model = RawAutoencoderModel(in_channels=3, config=ae_cfg, device=device, batch_size=2)
    losses = model.fit(batch, RawPixelExtractor(ds_cfg, RawExtractorConfig(), device))
    assert len(losses) == 1
    ckpt_path = tmp_path / "model.pt"
    model.save(ckpt_path)

    loaded = RawAutoencoderModel(in_channels=3, config=ae_cfg, device=device, batch_size=2)
    loaded.load(ckpt_path)
    scores, amaps, recons = loaded.predict_batch(
        batch, RawPixelExtractor(ds_cfg, RawExtractorConfig(), device), (64, 64)
    )
    assert scores.shape == (2,)
    assert amaps.shape == (2, 64, 64)
    assert recons.shape == (2, 64, 64, 3)


def test_compute_metrics() -> None:
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
    masks = np.zeros((4, 8, 8), dtype=np.uint8)
    masks[2:, 2:6, 2:6] = 1
    amaps = np.full((4, 8, 8), 0.05, dtype=np.float32)
    amaps[2:, 2:6, 2:6] = 0.95

    m = compute_metrics(labels, scores, masks, amaps, ["good", "good", "hole", "hole"])
    assert m["au_pro"] == pytest.approx(1.0)
    assert m["pro_integration_limit"] == pytest.approx(0.3)
    assert m["classification_au_roc_max"] == pytest.approx(1.0)
    assert m["image_auroc"] == pytest.approx(1.0)
    assert m["image_ap"] == pytest.approx(1.0)
    assert m["f1_max"] == pytest.approx(1.0)
    assert m["pixel_auroc"] == pytest.approx(1.0)
    assert m["pixel_ap"] == pytest.approx(1.0)
    assert m["per_defect"]["hole"]["count"] == 2
    assert m["per_defect"]["hole"]["recall_at_opt_th"] == pytest.approx(1.0)
    assert m["per_defect"]["hole"]["image_auroc"] == pytest.approx(1.0)
    assert m["per_defect"]["hole"]["au_pro"] == pytest.approx(1.0)


def test_official_pro_and_roc_parity() -> None:
    # Verify trapezoid linear interpolation at x_max=0.3
    x = np.array([0.0, 0.2, 0.4, 1.0])
    y = np.array([0.0, 0.6, 0.8, 1.0])
    # y(0.3) = 0.7 -> area = 0.5*0.2*(0+0.6) + 0.5*0.1*(0.6+0.7) = 0.06 + 0.065 = 0.125
    assert trapezoid(x, y, x_max=0.3) == pytest.approx(0.125)

    # Verify per-region equal weighting in compute_pro_curve:
    # One small 2x2 component (score=0.9) and one large 6x6 component (score=0.1)
    # Background pixels all have score=0.5 -> at FPR=0, PRO should equal 0.5 (1 of 2 regions)
    gt = np.zeros((1, 16, 16), dtype=np.uint8)
    gt[0, 0:2, 0:2] = 1
    gt[0, 8:14, 8:14] = 1
    amap = np.full((1, 16, 16), 0.5, dtype=np.float32)
    amap[0, 0:2, 0:2] = 0.9
    amap[0, 8:14, 8:14] = 0.1

    fprs, pros = compute_pro_curve(amap, gt)
    zero_fpr_pros = pros[fprs == 0.0]
    assert float(np.max(zero_fpr_pros)) == pytest.approx(0.5)

    au_pro, _, _ = compute_au_pro(amap, gt, integration_limit=0.3)
    assert au_pro == pytest.approx(0.5)

    # Verify compute_classification_roc matches official roc_curve_util behavior
    rng = np.random.default_rng(7)
    rand_maps = rng.random((20, 8, 8), dtype=np.float32)
    rand_labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    au_roc, roc_fpr, roc_tpr = compute_classification_roc(rand_maps, rand_labels)
    assert roc_fpr[0] == 0.0 and roc_fpr[-1] == 1.0
    assert roc_tpr[0] == 0.0 and roc_tpr[-1] == 1.0
    assert 0.0 <= au_roc <= 1.0


def test_end_to_end_pipeline_and_weight_reload(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path / "data")
    out_base = tmp_path / "results"

    cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, image_size=(64, 64), batch_size=2, num_workers=1),
        autoencoder=AutoencoderConfig(
            latent_dim=16,
            base_channels=8,
            num_stages=3,
            epochs=2,
            gaussian_sigma=1.0,
        ),
        output=OutputConfig(
            base_dir=out_base,
            experiment_name="smoke",
            max_vis_samples=4,
            save_tiff_maps=True,
        ),
        device="cpu",
    )
    payload = run_pipeline(cfg)

    exp_dir = out_base / "smoke"
    assert (exp_dir / "metrics.json").is_file()
    assert (exp_dir / "per_sample_scores.csv").is_file()
    assert (exp_dir / "loss_curve.png").is_file()
    assert (exp_dir / "score_distribution.png").is_file()
    assert (exp_dir / "pro_roc_curves.png").is_file()
    assert (exp_dir / "weights" / "model.pt").is_file()
    assert len(list((exp_dir / "heatmaps").glob("*.png"))) == 4

    tiff_good = exp_dir / "anomaly_maps" / "carpet" / "test" / "good" / "000.tiff"
    tiff_hole = exp_dir / "anomaly_maps" / "carpet" / "test" / "hole" / "000.tiff"
    assert tiff_good.is_file() and tiff_hole.is_file()
    loaded_tiff = tifffile.imread(str(tiff_hole))
    assert loaded_tiff.shape == (64, 64)
    assert loaded_tiff.dtype == np.float32

    with (exp_dir / "metrics.json").open(encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["num_train"] == 4
    assert saved["num_test"] == 4
    assert "au_pro" in saved["metrics"]
    assert "classification_au_roc_max" in saved["metrics"]
    assert "au_pro" in saved["metrics"]["per_defect"]["hole"]

    with (exp_dir / "per_sample_scores.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4
    assert payload["metrics"]["image_auroc"] >= 0.0

    # Reload saved weights
    reload_cfg = PipelineConfig(
        dataset=cfg.dataset,
        autoencoder=AutoencoderConfig(
            latent_dim=16,
            base_channels=8,
            num_stages=3,
            epochs=2,
            gaussian_sigma=1.0,
            load_weights=exp_dir / "weights" / "model.pt",
        ),
        output=OutputConfig(
            base_dir=out_base,
            experiment_name="smoke_reload",
            save_visualizations=False,
        ),
        device="cpu",
    )
    reload_payload = run_pipeline(reload_cfg)
    assert reload_payload["epoch_losses"] == []
    assert reload_payload["metrics"]["image_auroc"] == pytest.approx(
        payload["metrics"]["image_auroc"]
    )


def test_multi_category_pipeline(tmp_path: Path) -> None:
    data_root = tmp_path / "multi_data"
    _create_synthetic_mvtec(data_root, category="carpet")
    _create_synthetic_mvtec(data_root, category="leather")
    (data_root / "mvtec_ad_evaluation").mkdir(parents=True)  # should be ignored

    out_base = tmp_path / "results"
    cfg = PipelineConfig.from_args(
        data_dir=str(data_root),
        category="all",
        image_size=[64, 64],
        num_stages=3,
        base_channels=8,
        latent_dim=16,
        epochs=1,
        batch_size=2,
        num_workers=1,
        output="multi_smoke",
        output_dir=str(out_base),
        no_vis=True,
        save_tiff=True,
        device="cpu",
    )
    summary = run_pipeline(cfg)
    assert summary["categories"] == ["carpet", "leather"]
    assert (out_base / "multi_smoke" / "summary_metrics.json").is_file()
    assert (out_base / "multi_smoke" / "carpet" / "metrics.json").is_file()
    assert (out_base / "multi_smoke" / "leather" / "metrics.json").is_file()
    assert (
        out_base / "multi_smoke" / "anomaly_maps" / "leather" / "test" / "hole" / "000.tiff"
    ).is_file()
