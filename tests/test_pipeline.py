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

from src.__main__ import build_parser
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
from src.models.autoencoder import (
    ConvAutoencoder,
    RawAutoencoderModel,
    _corrupt_batch,
    _ms_l1_ssim_loss,
    _ssim_map,
)
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


def _naive_2d_ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    x_f32 = x.float()
    y_f32 = y.float()
    c1 = 0.01**2
    c2 = 0.03**2
    pad = window_size // 2
    mu_x = torch.nn.functional.avg_pool2d(x_f32, window_size, stride=1, padding=pad)
    mu_y = torch.nn.functional.avg_pool2d(y_f32, window_size, stride=1, padding=pad)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    sigma_x_sq = (
        torch.nn.functional.avg_pool2d(x_f32 * x_f32, window_size, stride=1, padding=pad) - mu_x_sq
    ).clamp(min=0.0)
    sigma_y_sq = (
        torch.nn.functional.avg_pool2d(y_f32 * y_f32, window_size, stride=1, padding=pad) - mu_y_sq
    ).clamp(min=0.0)
    sigma_xy = (
        torch.nn.functional.avg_pool2d(x_f32 * y_f32, window_size, stride=1, padding=pad) - mu_xy
    )
    ssim_n = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return ((ssim_n / ssim_d) + 1.0).mul(0.5).clamp(0.0, 1.0)


def test_separable_ssim_matches_2d_and_bfloat16_finite() -> None:
    torch.manual_seed(0)
    x = torch.rand((2, 3, 64, 64), dtype=torch.float32)
    y = torch.rand((2, 3, 64, 64), dtype=torch.float32)

    sep = _ssim_map(x, y, window_size=11)
    ref = _naive_2d_ssim_map(x, y, window_size=11)
    assert sep.shape == ref.shape
    assert torch.allclose(sep, ref, atol=1e-5, rtol=1e-5)

    # Constant inputs + bfloat16 precision should produce strictly finite bounded outputs
    ones_bf16 = torch.ones((1, 3, 32, 32), dtype=torch.bfloat16)
    zeros_bf16 = torch.zeros((1, 3, 32, 32), dtype=torch.bfloat16)
    bf_same = _ssim_map(ones_bf16, ones_bf16)
    bf_rand = _ssim_map(x.to(torch.bfloat16), y.to(torch.bfloat16))
    bf_extreme = _ssim_map(ones_bf16, zeros_bf16)
    for out in (bf_same, bf_rand, bf_extreme):
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min().item()) >= 0.0
        assert float(out.max().item()) <= 1.0


def test_ms_l1_ssim_denoising_training_and_calibration() -> None:
    # Verify batch corruption behavior
    clean = torch.rand((4, 3, 64, 64), dtype=torch.float32)
    assert torch.equal(_corrupt_batch(clean, denoise_prob=0.0, seed=1), clean)
    corrupted_a = _corrupt_batch(clean, denoise_prob=1.0, seed=42)
    corrupted_b = _corrupt_batch(clean, denoise_prob=1.0, seed=42)
    assert torch.equal(corrupted_a, corrupted_b)
    assert not torch.equal(corrupted_a, clean)
    assert float(corrupted_a.min().item()) >= float(clean.min().item()) - 1e-6
    assert float(corrupted_a.max().item()) <= float(clean.max().item()) + 1e-6

    # Verify multi-scale loss decreases with better reconstruction
    assert float(_ms_l1_ssim_loss(clean, clean).item()) < float(
        _ms_l1_ssim_loss(corrupted_a, clean).item()
    )

    rng = np.random.default_rng(9)
    # Create textured background so texture damping can also be verified
    grid_y, grid_x = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    checker = (((grid_y // 2 + grid_x // 2) % 2) * 80 + 70).astype(np.int16)
    base_rgb = np.stack([checker, checker, checker], axis=-1)
    train_u8 = np.clip(
        base_rgb[None, ...] + rng.integers(-4, 5, size=(4, 64, 64, 3), dtype=np.int16),
        0,
        255,
    ).astype(np.uint8)

    device = torch.device("cpu")
    extractor = RawPixelExtractor(
        DatasetConfig(image_size=(64, 64)),
        RawExtractorConfig(),
        device=device,
    )
    cfg = AutoencoderConfig(
        latent_dim=16,
        base_channels=16,
        num_stages=3,
        loss_type="ms_l1_ssim",
        epochs=6,
        lr=5e-3,
        score_top_k_pct=0.02,
        gaussian_sigma=1.0,
        spatial_dropout=0.1,
        denoise_prob=0.5,
        texture_damping=2.0,
    )
    model = RawAutoencoderModel(
        in_channels=3,
        config=cfg,
        device=device,
        batch_size=2,
        seed=42,
    )
    losses = model.fit(train_u8, extractor)
    assert len(losses) == 6
    assert losses[-1] < losses[0]
    assert bool(model.net.calibrated.item()) is True
    assert torch.isfinite(model.net.err_mean).all()
    assert float(model.net.err_std.item()) >= 1e-4
    assert model.net.normal_mu_map.shape == (1, 1, 16, 16)
    assert model.net.normal_std_map.shape == (1, 1, 16, 16)

    normal_test = train_u8[:1].copy()
    defect_test = train_u8[:1].copy()
    defect_test[0, 20:40, 20:40, :] = 250
    both = np.concatenate([normal_test, defect_test], axis=0)
    scores, amaps, recons = model.predict_batch(both, extractor, target_hw=(64, 64))
    assert scores.shape == (2,)
    assert amaps.shape == (2, 64, 64)
    assert recons.shape == (2, 64, 64, 3)
    assert np.isfinite(scores).all() and np.isfinite(amaps).all()
    assert float(amaps.min()) >= 0.0
    assert scores[1] > scores[0]

    # Check texture damping reduces raw error over textured reconstructions when uncalibrated
    with torch.inference_mode():
        model.net.eval()
        x_t = extractor.extract_batch(normal_test)
        _, raw_damped = model._score_batch_tensor(x_t, target_hw=(64, 64), apply_calibration=False)
        model.config = AutoencoderConfig(
            latent_dim=16,
            base_channels=16,
            num_stages=3,
            loss_type="ms_l1_ssim",
            epochs=6,
            lr=5e-3,
            score_top_k_pct=0.02,
            gaussian_sigma=1.0,
            spatial_dropout=0.1,
            denoise_prob=0.5,
            texture_damping=0.0,
        )
        _, raw_undamped = model._score_batch_tensor(
            x_t, target_hw=(64, 64), apply_calibration=False
        )
        assert float(raw_damped.mean().item()) < float(raw_undamped.mean().item())


def test_save_load_preserves_weights_metadata_and_buffers(tmp_path: Path) -> None:
    rng = np.random.default_rng(5)
    train_u8 = rng.integers(80, 140, size=(2, 32, 32, 3), dtype=np.uint8)
    device = torch.device("cpu")
    extractor = RawPixelExtractor(
        DatasetConfig(image_size=(32, 32)),
        RawExtractorConfig(),
        device=device,
    )
    cfg = AutoencoderConfig(
        latent_dim=16,
        base_channels=8,
        num_stages=3,
        loss_type="ms_l1_ssim",
        epochs=2,
        spatial_dropout=0.25,
        denoise_prob=0.5,
        texture_damping=2.0,
    )
    model = RawAutoencoderModel(in_channels=3, config=cfg, device=device, batch_size=2, seed=7)
    model.fit(train_u8, extractor)
    scores_before, maps_before, recons_before = model.predict_batch(
        train_u8, extractor, target_hw=(32, 32)
    )

    ckpt_path = tmp_path / "weights" / "model.pt"
    model.save(ckpt_path)
    raw_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert raw_ckpt["meta"]["spatial_dropout"] == pytest.approx(0.25)
    assert raw_ckpt["meta"]["latent_dim"] == 16
    assert raw_ckpt["meta"]["num_stages"] == 3

    reloaded = RawAutoencoderModel(in_channels=3, config=cfg, device=device, batch_size=2, seed=7)
    reloaded.load(ckpt_path)

    for (k_a, v_a), (k_b, v_b) in zip(
        model.net.state_dict().items(), reloaded.net.state_dict().items(), strict=True
    ):
        assert k_a == k_b
        assert torch.equal(v_a, v_b)

    scores_after, maps_after, recons_after = reloaded.predict_batch(
        train_u8, extractor, target_hw=(32, 32)
    )
    assert np.allclose(scores_before, scores_after, atol=1e-6)
    assert np.allclose(maps_before, maps_after, atol=1e-6)
    assert np.array_equal(recons_before, recons_after)


def test_cli_new_flags_and_config_validation(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path)
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(data_root),
            "--loss",
            "ms_l1_ssim",
            "--spatial-dropout",
            "0.2",
            "--denoise-prob",
            "0.75",
            "--texture-damping",
            "1.5",
            "--device",
            "cpu",
        ]
    )
    cfg = PipelineConfig.from_args(**vars(args))
    cfg.validate()
    assert cfg.autoencoder.loss_type == "ms_l1_ssim"
    assert cfg.autoencoder.spatial_dropout == pytest.approx(0.2)
    assert cfg.autoencoder.denoise_prob == pytest.approx(0.75)
    assert cfg.autoencoder.texture_damping == pytest.approx(1.5)

    for bad_ae in (
        AutoencoderConfig(spatial_dropout=1.0),
        AutoencoderConfig(denoise_prob=1.5),
        AutoencoderConfig(texture_damping=-0.1),
    ):
        bad_cfg = PipelineConfig(
            dataset=DatasetConfig(root_dir=data_root),
            autoencoder=bad_ae,
            device="cpu",
        )
        with pytest.raises(ValueError):
            bad_cfg.validate()
