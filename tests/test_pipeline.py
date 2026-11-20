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
    DINOv3ExtractorConfig,
    OutputConfig,
    PipelineConfig,
    RawExtractorConfig,
    WTAConfig,
)
from src.dataset import iter_batches, load_mvtec_category
from src.extractors.dinov3 import DINOv3Extractor
from src.extractors.raw import RawPixelExtractor
from src.metrics import (
    compute_au_pro,
    compute_classification_roc,
    compute_metrics,
    compute_pro_curve,
    trapezoid,
)
from src.models import get_model
from src.models.dcae import (
    ConvAutoencoder,
    RawAutoencoderModel,
    _compute_raw_error_map,
    _corrupt_batch,
    _gms_map,
    _iter_msgms_error_scales,
    _morphological_open_2d,
    _ms_l1_ssim_loss,
    _ssim_map,
)
from src.models.wta_ae import WTAAutoencoderModel, WTAConvAutoencoder, WTASparsity2d
from src.results import inspect_results_folders, scan_results
from src.results import main as results_main
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

    bottleneck_1x1_cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, image_size=(64, 64)),
        autoencoder=AutoencoderConfig(num_stages=6),
        device="cpu",
    )
    with pytest.raises(ValueError, match="1x1 bottleneck"):
        bottleneck_1x1_cfg.validate()


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
    assert saved["hyperparams"]["loss_type"] == "l2"
    assert (out_base / "all_experiments_metrics.csv").is_file()
    assert (out_base / "all_per_defect_metrics.csv").is_file()

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
    return (ssim_n / ssim_d).clamp(0.0, 1.0)


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
    # Full [0, 1] dynamic range: ones vs zeros in interior should be near 0.0 (not 0.5)
    assert float(bf_extreme[:, :, 11:-11, 11:-11].mean().item()) < 0.01


def test_gms_map_finite_and_sensitive_to_thin_lines() -> None:
    torch.manual_seed(1)
    x = torch.rand((2, 3, 64, 64), dtype=torch.float32)
    y = torch.rand((2, 3, 64, 64), dtype=torch.float32)

    # Identical inputs -> similarity == 1.0 everywhere (including borders), error == 0.0
    same_f32 = _gms_map(x, x)
    assert same_f32.dtype == torch.float32
    assert same_f32.shape == x.shape
    assert torch.allclose(same_f32, torch.ones_like(same_f32), atol=1e-6)

    # bfloat16 constant, random, and extreme inputs must remain finite and in [0, 1]
    ones_bf16 = torch.ones((1, 3, 32, 32), dtype=torch.bfloat16)
    zeros_bf16 = torch.zeros((1, 3, 32, 32), dtype=torch.bfloat16)
    for out in (
        _gms_map(ones_bf16, ones_bf16),
        _gms_map(x.to(torch.bfloat16), y.to(torch.bfloat16)),
        _gms_map(ones_bf16, zeros_bf16),
    ):
        assert out.dtype == torch.float32
        assert torch.isfinite(out).all()
        assert float(out.min().item()) >= 0.0
        assert float(out.max().item()) <= 1.0

    # Small spatial edge case (1x1 and 2x2) does not crash on reflect padding
    tiny = torch.rand((1, 3, 1, 1), dtype=torch.float32)
    tiny_out = _gms_map(tiny, tiny)
    assert torch.isfinite(tiny_out).all()

    # Thin high-gradient line simulating metal_contamination / thread on uniform grey background
    recon = torch.full((1, 3, 64, 64), 0.5, dtype=torch.float32)
    orig = recon.clone()
    orig[:, :, 31:33, 20:44] = 0.95  # 2-pixel-thick bright staple/line
    gms_err = 1.0 - _gms_map(recon, orig)
    line_peak = float(gms_err[:, :, 28:36, 20:44].max().item())
    bg_mean = float(gms_err[:, :, :15, :15].mean().item())
    assert line_peak > 0.3
    assert bg_mean == pytest.approx(0.0, abs=1e-6)

    # Multi-scale MSGMS weights and shapes
    scales = _iter_msgms_error_scales(recon, orig)
    assert [w for w, _ in scales] == pytest.approx([0.15, 0.45, 0.40])
    assert [tuple(m.shape[-2:]) for _, m in scales] == [(64, 64), (32, 32), (16, 16)]
    assert all(torch.isfinite(m).all() for _, m in scales)

    # Raw error map under ms_l1_ssim on odd/tiny shapes and bfloat16
    for hw in ((65, 65), (1, 1)):
        xh_t = torch.rand((2, 3, *hw), dtype=torch.bfloat16)
        x_t = torch.rand((2, 3, *hw), dtype=torch.bfloat16)
        raw_em = _compute_raw_error_map(xh_t, x_t, "ms_l1_ssim")
        assert raw_em.dtype == torch.float32
        assert raw_em.shape == (2, 1, *hw)
        assert torch.isfinite(raw_em).all()

    # ConvAutoencoder with InstanceNorm2d under channels_last + bfloat16 autocast
    net = ConvAutoencoder(in_channels=3, base_channels=8, latent_dim=16, num_stages=3).to(
        memory_format=torch.channels_last
    )
    x_cl = torch.rand((2, 3, 32, 32), dtype=torch.float32).to(memory_format=torch.channels_last)
    with torch.amp.autocast("cpu", dtype=torch.bfloat16):
        y_cl = net(x_cl)
        loss_cl = _ms_l1_ssim_loss(y_cl, x_cl)
    loss_cl.backward()
    assert torch.isfinite(loss_cl)
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


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
    assert float(model.net.err_p99.item()) >= float(model.net.err_mean.item())
    assert model.net.normal_mu_map.shape == (1, 1, 16, 16)
    assert model.net.normal_std_map.shape == (1, 1, 16, 16)
    assert any(isinstance(m, torch.nn.InstanceNorm2d) for m in model.net.encoder)

    # Multi-seed corruption covers all modes (including smooth Gaussian blob stain) within bounds
    for s in range(10):
        c_out = _corrupt_batch(clean, denoise_prob=1.0, seed=s)
        assert torch.isfinite(c_out).all()
        assert float(c_out.min().item()) >= float(clean.min().item()) - 1e-6
        assert float(c_out.max().item()) <= float(clean.max().item()) + 1e-6

    normal_test = train_u8[:1].copy()
    defect_test = train_u8[:1].copy()
    defect_test[0, 20:40, 20:40, :] = 250
    both = np.concatenate([normal_test, defect_test], axis=0)
    scores, amaps, recons = model.predict_batch(both, extractor, target_hw=(64, 64))
    assert scores.shape == (2,)
    assert amaps.shape == (2, 64, 64)
    assert recons.shape == (2, 64, 64, 3)
    assert np.isfinite(scores).all() and np.isfinite(amaps).all()
    assert float(amaps.min()) > 0.0  # smooth softplus avoids flat zero ties
    assert scores[1] > scores[0]
    amaps_t = torch.from_numpy(amaps).unsqueeze(1)
    score_map = torch.nn.functional.avg_pool2d(amaps_t, kernel_size=3, stride=1, padding=1)
    flat_amaps = np.sort(score_map.squeeze(1).numpy().reshape(2, -1), axis=1)[:, ::-1]
    k_expected = max(1, int(np.ceil(flat_amaps.shape[1] * cfg.score_top_k_pct)))
    k_core_expected = max(1, int(np.ceil(k_expected * 0.1)))
    expected_scores = 0.5 * flat_amaps[:, :k_core_expected].mean(axis=1) + 0.5 * flat_amaps[
        :, :k_expected
    ].mean(axis=1)
    assert np.allclose(scores, expected_scores, atol=1e-5)

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
    assert cfg.autoencoder.score_top_k_pct == pytest.approx(0.0005)
    assert AutoencoderConfig().score_top_k_pct == pytest.approx(0.0005)
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


def test_results_scan_mixed_layout_and_csv_export(tmp_path: Path) -> None:
    res_dir = tmp_path / "results"

    # 1) Recorded new-format single-category run with per_sample_scores.csv
    rec_dir = res_dir / "carpet_fullres_ms_l1_ssim_s6_c64_z128"
    (rec_dir / "heatmaps").mkdir(parents=True)
    (rec_dir / "heatmaps" / "000.png").write_bytes(b"png")
    (rec_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_name": "carpet_fullres_ms_l1_ssim_s6_c64_z128",
                "category": "carpet",
                "device": "cuda",
                "image_size": [1024, 1024],
                "num_train": 76,
                "num_test": 4,
                "train_time_s": 60.0,
                "infer_fps": 30.0,
                "hyperparams": {
                    "loss_type": "ms_l1_ssim",
                    "num_stages": 6,
                    "base_channels": 64,
                    "latent_dim": 128,
                    "epochs": 120,
                    "batch_size": 16,
                    "lr": 1e-3,
                    "weight_decay": 1e-5,
                    "gaussian_sigma": 10.0,
                    "score_top_k_pct": 0.002,
                    "spatial_dropout": 0.1,
                    "denoise_prob": 0.5,
                    "texture_damping": 2.0,
                    "full_res": True,
                    "use_amp": True,
                    "pro_integration_limit": 0.3,
                    "seed": 42,
                },
                "calibration": {"calibrated": True, "err_mean": 0.18, "err_std": 0.03},
                "epoch_losses": [0.4, 0.25],
                "metrics": {
                    "au_pro": 0.75,
                    "pro_integration_limit": 0.3,
                    "classification_au_roc_max": 0.72,
                    "image_auroc": 0.70,
                    "image_ap": 0.85,
                    "f1_max": 0.90,
                    "optimal_threshold": 3.0,
                    "accuracy_at_opt_th": 0.88,
                    "pixel_auroc": 0.91,
                    "pixel_ap": 0.25,
                    "pixel_f1_max": 0.33,
                    "per_defect": {
                        "hole": {
                            "count": 2,
                            "recall_at_opt_th": 1.0,
                            "image_auroc": 0.8,
                            "au_pro": 0.78,
                            "mean_score": 5.5,
                            "std_score": 0.4,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (rec_dir / "per_sample_scores.csv").write_text(
        "index,image_path,defect_type,label,image_score,predicted_label\n"
        "0,a.png,good,0,1.0,0\n"
        "1,b.png,good,0,2.0,0\n"
        "2,c.png,hole,1,5.0,1\n"
        "3,d.png,hole,1,6.0,1\n",
        encoding="utf-8",
    )

    # 2) Legacy run without "hyperparams" block
    leg_dir = res_dir / "hazelnut_fullres_l1_ssim_s5_c48_z128"
    leg_dir.mkdir(parents=True)
    (leg_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_name": "hazelnut_fullres_l1_ssim_s5_c48_z128",
                "category": "hazelnut",
                "device": "cuda",
                "image_size": [1024, 1024],
                "num_train": 391,
                "num_test": 110,
                "train_time_s": 300.0,
                "infer_fps": 15.0,
                "epoch_losses": [0.3, 0.2, 0.1],
                "metrics": {
                    "au_pro": 0.88,
                    "pro_integration_limit": 0.3,
                    "classification_au_roc_max": 0.66,
                    "image_auroc": 0.65,
                    "image_ap": 0.79,
                    "f1_max": 0.80,
                    "optimal_threshold": 0.14,
                    "accuracy_at_opt_th": 0.70,
                    "pixel_auroc": 0.95,
                    "pixel_ap": 0.30,
                    "pixel_f1_max": 0.31,
                    "per_defect": {},
                },
            }
        ),
        encoding="utf-8",
    )

    # 3) Nested multi-category partial run
    multi_bottle = res_dir / "mvtec_fullres_l1ssim_s5" / "bottle"
    multi_bottle.mkdir(parents=True)
    (multi_bottle / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_name": "bottle",
                "category": "bottle",
                "device": "cuda",
                "image_size": [896, 896],
                "num_train": 209,
                "num_test": 83,
                "train_time_s": 100.0,
                "infer_fps": 20.0,
                "epoch_losses": [0.2, 0.1],
                "metrics": {"au_pro": 0.81, "classification_au_roc_max": 0.79, "per_defect": {}},
            }
        ),
        encoding="utf-8",
    )

    # 4) Empty / incomplete folder
    (res_dir / "carpet_fullres_l1_ssim_s4_c64_z256").mkdir(parents=True)

    inv = inspect_results_folders(res_dir)
    by_name = {entry.folder_name: entry for entry in inv}
    assert by_name["carpet_fullres_ms_l1_ssim_s6_c64_z128"].status == "complete"
    assert by_name["carpet_fullres_ms_l1_ssim_s6_c64_z128"].num_heatmaps == 1
    assert by_name["mvtec_fullres_l1ssim_s5"].status == "partial_multi"
    assert by_name["carpet_fullres_l1_ssim_s4_c64_z256"].status == "incomplete"

    _, records, defects = scan_results(res_dir, include_incomplete=False)
    assert len(records) == 3
    rec_map = {r.experiment_name: r for r in records}
    assert rec_map["carpet_fullres_ms_l1_ssim_s6_c64_z128"].hp_source == "recorded"
    assert rec_map["carpet_fullres_ms_l1_ssim_s6_c64_z128"].good_mean_score == pytest.approx(1.5)
    assert rec_map["carpet_fullres_ms_l1_ssim_s6_c64_z128"].defect_mean_score == pytest.approx(5.5)

    leg = rec_map["hazelnut_fullres_l1_ssim_s5_c48_z128"]
    assert leg.hp_source == "inferred"
    assert leg.loss_type == "l1_ssim"
    assert (leg.num_stages, leg.base_channels, leg.latent_dim, leg.epochs) == (5, 48, 128, 3)

    multi = rec_map["mvtec_fullres_l1ssim_s5/bottle"]
    assert multi.parent_experiment == "mvtec_fullres_l1ssim_s5"
    assert multi.loss_type == "l1_ssim"
    assert multi.num_stages == 5
    assert len(defects) == 1

    # CLI execution writes both CSVs
    results_main(["--results-dir", str(res_dir), "--include-incomplete"])
    exp_csv = res_dir / "all_experiments_metrics.csv"
    def_csv = res_dir / "all_per_defect_metrics.csv"
    assert exp_csv.is_file() and def_csv.is_file()
    with exp_csv.open(encoding="utf-8") as f:
        csv_rows = list(csv.DictReader(f))
    assert len(csv_rows) == 4
    with def_csv.open(encoding="utf-8") as f:
        def_rows = list(csv.DictReader(f))
    assert len(def_rows) == 1
    assert def_rows[0]["defect_type"] == "hole"


def test_morphological_open_2d_and_block_masking() -> None:
    # Isolated 1-pixel spike is erased, while contiguous 5x5 defect block is preserved
    err = torch.zeros((1, 1, 32, 32), dtype=torch.float32)
    err[0, 0, 6, 6] = 10.0  # isolated single-pixel spike
    err[0, 0, 18:23, 18:23] = 5.0  # contiguous 5x5 region
    opened = _morphological_open_2d(err, kernel_size=3)
    assert float(opened[0, 0, 6, 6].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(opened[0, 0, 20, 20].item()) == pytest.approx(5.0, abs=1e-6)

    # Structured grid masking + smooth stain modes across a batch remain bounded and finite
    clean = torch.rand((8, 3, 64, 64), dtype=torch.float32)
    corrupted = _corrupt_batch(clean, denoise_prob=1.0, seed=123)
    assert torch.isfinite(corrupted).all()
    assert not torch.equal(corrupted, clean)
    assert float(corrupted.min().item()) >= float(clean.min().item()) - 1e-6
    assert float(corrupted.max().item()) <= float(clean.max().item()) + 1e-6


def test_loss_floor_early_stopping_and_cli_flag(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path)
    args = build_parser().parse_args(
        ["--data-dir", str(data_root), "--loss-floor", "0.01", "--device", "cpu"]
    )
    cfg_cli = PipelineConfig.from_args(**vars(args))
    cfg_cli.validate()
    assert cfg_cli.autoencoder.loss_floor == pytest.approx(0.01)

    bad_cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root),
        autoencoder=AutoencoderConfig(loss_floor=-0.01),
        device="cpu",
    )
    with pytest.raises(ValueError, match="loss_floor"):
        bad_cfg.validate()

    # Train 25 max epochs with high loss_floor so early stopping triggers at epoch 21 (2 hits >= 20)
    rng = np.random.default_rng(3)
    train_u8 = rng.integers(100, 150, size=(2, 32, 32, 3), dtype=np.uint8)
    device = torch.device("cpu")
    extractor = RawPixelExtractor(
        DatasetConfig(image_size=(32, 32)),
        RawExtractorConfig(),
        device=device,
    )
    ae_cfg = AutoencoderConfig(
        latent_dim=8,
        base_channels=8,
        num_stages=3,
        loss_type="l2",
        epochs=25,
        loss_floor=0.5,
        denoise_prob=0.0,
    )
    model = RawAutoencoderModel(in_channels=3, config=ae_cfg, device=device, batch_size=2, seed=42)
    losses = model.fit(train_u8, extractor)
    assert len(losses) == 21
    assert bool(model.net.calibrated.item()) is True


class _MockRoPE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_hw: tuple[int, int] | None = None

    def forward(self, pixel_values: torch.Tensor) -> tuple[int, int]:
        self.last_hw = (pixel_values.shape[-2], pixel_values.shape[-1])
        return self.last_hw


class _MockDINOv3Embeddings(torch.nn.Module):
    def __init__(self, hidden_size: int = 32, num_registers: int = 4) -> None:
        super().__init__()
        self.patch_embeddings = torch.nn.Conv2d(3, hidden_size, kernel_size=16, stride=16)
        self.rope_embeddings = _MockRoPE()
        self.num_registers = num_registers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.rope_embeddings(pixel_values)
        feats = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        b, _, d = feats.shape
        cls_and_regs = torch.ones((b, 1 + self.num_registers, d), device=feats.device)
        return torch.cat([cls_and_regs, feats], dim=1)


class _MockDINOv3Model(torch.nn.Module):
    def __init__(self, hidden_size: int = 32, num_registers: int = 4) -> None:
        super().__init__()
        from types import SimpleNamespace

        self.config = SimpleNamespace(hidden_size=hidden_size, num_register_tokens=num_registers)
        self.embeddings = _MockDINOv3Embeddings(
            hidden_size=hidden_size, num_registers=num_registers
        )
        self.layernorm = torch.nn.LayerNorm(hidden_size)

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> object:
        from types import SimpleNamespace

        tokens = self.embeddings(pixel_values)
        normed = self.layernorm(tokens)
        if output_hidden_states:
            return SimpleNamespace(
                last_hidden_state=normed,
                hidden_states=[tokens * (0.5 * i + 1.0) for i in range(5)],
            )
        return SimpleNamespace(last_hidden_state=normed, hidden_states=None)


def test_dinov3_extractor_features_and_autoencoder_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    mock_proc = SimpleNamespace(image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])

    def _fake_load(self: DINOv3Extractor) -> tuple[_MockDINOv3Model, SimpleNamespace]:
        m = _MockDINOv3Model(hidden_size=32, num_registers=4)
        m.eval()
        m._native_num_registers = 4  # type: ignore[attr-defined]
        self._apply_patch_stride(m)
        self._configure_register_pruning(m)
        return m, mock_proc

    monkeypatch.setattr(DINOv3Extractor, "_load_model", _fake_load)

    ds_cfg = DatasetConfig(image_size=(128, 128), batch_size=2)
    dv3_cfg = DINOv3ExtractorConfig(
        model_name="mock/dinov3-small",
        stride=8,
        layers=(2, 4),
        prune_registers=True,
        mean_centering_alpha=0.5,
        plain_last_hidden_state=False,
    )
    device = torch.device("cpu")
    extractor = DINOv3Extractor(ds_cfg, dv3_cfg, device=device, stride_factor=4)

    # (128 - 16) // 8 + 1 = 15 -> aligned down to multiple of 4 => 12x12 patches
    assert extractor.spatial_shape == (12, 12)
    assert extractor.out_channels == 64  # 32 * 2 layers
    assert extractor.is_unit_range is False

    rng = np.random.default_rng(11)
    imgs = rng.integers(0, 255, size=(2, 128, 128, 3), dtype=np.uint8)
    feats = extractor.extract_batch(imgs)
    assert feats.shape == (2, 64, 12, 12)
    assert feats.dtype == torch.float32
    norms = torch.linalg.vector_norm(feats, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    # Check RoPE received dummy pixel dimensions matching (h_p * patch_size)
    model_obj, _ = extractor._components
    assert model_obj.embeddings.rope_embeddings.last_hw == (12 * 16, 12 * 16)

    # Autoencoder roundtrip with DINOv3 feature maps (use_sigmoid=False)
    ae_cfg = AutoencoderConfig(
        latent_dim=16,
        base_channels=16,
        num_stages=2,
        loss_type="l2",
        epochs=2,
        denoise_prob=0.5,
    )
    ae = RawAutoencoderModel(
        in_channels=extractor.out_channels,
        config=ae_cfg,
        device=device,
        batch_size=2,
        use_sigmoid=extractor.is_unit_range,
    )
    losses = ae.fit(imgs, extractor)
    assert len(losses) == 2
    scores, amaps, recons = ae.predict_batch(imgs, extractor, target_hw=(128, 128))
    assert scores.shape == (2,)
    assert amaps.shape == (2, 128, 128)
    assert recons.shape == (2, 128, 128, 3) and recons.dtype == np.uint8


def test_dinov3_cli_and_config_validation(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path)
    args = build_parser().parse_args(
        [
            "--data-dir",
            str(data_root),
            "--extractor",
            "dinov3",
            "--image-size",
            "256",
            "256",
            "--num-stages",
            "3",
            "--dinov3-stride",
            "14",
            "--dinov3-layers",
            "6",
            "12",
            "--dinov3-prune-registers",
            "--dinov3-mean-center-alpha",
            "0.5",
            "--device",
            "cpu",
        ]
    )
    cfg = PipelineConfig.from_args(**vars(args))
    cfg.validate()
    assert cfg.extractor_name == "dinov3"
    assert cfg.extractors.dinov3.stride == 14
    assert cfg.extractors.dinov3.layers == (6, 12)
    assert cfg.extractors.dinov3.prune_registers is True
    assert cfg.extractors.dinov3.mean_centering_alpha == pytest.approx(0.5)

    # Grayscale rejected for DINOv3
    gray_cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, grayscale=True, image_size=(256, 256)),
        extractor_name="dinov3",
        autoencoder=AutoencoderConfig(num_stages=3),
        device="cpu",
    )
    with pytest.raises(ValueError, match="RGB"):
        gray_cfg.validate()

    # Too many AE stages for DINOv3 patch grid rejected
    deep_cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, image_size=(256, 256)),
        extractor_name="dinov3",
        autoencoder=AutoencoderConfig(num_stages=4),
        device="cpu",
    )
    with pytest.raises(ValueError, match="patch grid"):
        deep_cfg.validate()


def test_wta_sparsity_spatial_and_lifetime_exactness() -> None:
    torch.manual_seed(0)
    b, ch, hb, wb = 4, 8, 16, 16
    # Strictly distinct positive activations per slot to verify exact winner counts
    z = (torch.rand((b, ch, hb, wb), dtype=torch.float32) + 0.1).requires_grad_(True)
    wta = WTASparsity2d(num_channels=ch, spatial_window=4, lifetime_sparsity=0.25)
    wta.train()

    out = wta(z)
    assert out.shape == z.shape

    # Each 4x4 window must contain at most 1 non-zero winner
    blocks = out.unfold(2, 4, 4).unfold(3, 4, 4)  # Shape: (B, C, 4, 4, 4, 4)
    per_block_nz = (blocks > 0.0).sum(dim=(-1, -2))
    assert int(per_block_nz.max().item()) <= 1

    # Lifetime sparsity across N_slots = 4 * 4 * 4 = 64 -> k = ceil(64 * 0.25) = 16 per channel
    per_channel_nz = (out > 0.0).sum(dim=(0, 2, 3))
    assert torch.all(per_channel_nz == 16)
    assert bool(wta.threshold_initialized.item()) is True
    assert bool((wta.running_threshold > 0.0).all().item())

    # Verify gradients flow exclusively through active winners
    out.sum().backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()
    inactive = out.detach() == 0.0
    assert torch.all(z.grad[inactive] == 0.0)


def test_wta_eval_batch_size_independence() -> None:
    torch.manual_seed(3)
    net = WTAConvAutoencoder(
        in_channels=3,
        base_channels=8,
        latent_dim=16,
        num_stages=3,
        spatial_window=4,
        lifetime_sparsity=0.25,
    )
    x = torch.rand((4, 3, 64, 64), dtype=torch.float32)
    net.train()
    _ = net(x)
    assert bool(net.wta.threshold_initialized.item()) is True

    net.eval()
    with torch.inference_mode():
        out_single = net(x[:1])
        out_batch = net(x)
    assert out_single.shape == (1, 3, 64, 64)
    assert torch.allclose(out_single, out_batch[:1], atol=1e-6)


def test_wta_ae_pipeline_checkpoint_and_dcae_alias(tmp_path: Path) -> None:
    data_root = _create_synthetic_mvtec(tmp_path / "data")
    out_base = tmp_path / "results"

    cfg = PipelineConfig(
        dataset=DatasetConfig(root_dir=data_root, image_size=(64, 64), batch_size=2, num_workers=1),
        model_name="wta_ae",
        autoencoder=AutoencoderConfig(
            latent_dim=16,
            base_channels=8,
            num_stages=3,
            loss_type="l2",
            epochs=2,
            gaussian_sigma=1.0,
        ),
        wta=WTAConfig(spatial_window=2, lifetime_sparsity=0.25, eval_threshold_scale=0.8),
        output=OutputConfig(
            base_dir=out_base,
            experiment_name="wta_smoke",
            save_visualizations=False,
        ),
        device="cpu",
    )
    payload = run_pipeline(cfg)
    ckpt_path = out_base / "wta_smoke" / "weights" / "model.pt"
    assert ckpt_path.is_file()

    saved = json.loads((out_base / "wta_smoke" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["hyperparams"]["model"] == "wta_ae"
    assert saved["hyperparams"]["wta_spatial_window"] == 2
    assert saved["hyperparams"]["wta_lifetime_sparsity"] == pytest.approx(0.25)

    reload_cfg = PipelineConfig(
        dataset=cfg.dataset,
        model_name="wta_ae",
        autoencoder=AutoencoderConfig(
            latent_dim=16,
            base_channels=8,
            num_stages=3,
            loss_type="l2",
            epochs=2,
            gaussian_sigma=1.0,
            load_weights=ckpt_path,
        ),
        wta=cfg.wta,
        output=OutputConfig(
            base_dir=out_base,
            experiment_name="wta_smoke_reload",
            save_visualizations=False,
        ),
        device="cpu",
    )
    reloaded_payload = run_pipeline(reload_cfg)
    assert reloaded_payload["metrics"]["image_auroc"] == pytest.approx(
        payload["metrics"]["image_auroc"]
    )

    reloaded_model = WTAAutoencoderModel(
        in_channels=3,
        config=reload_cfg.autoencoder,
        device=torch.device("cpu"),
        wta_config=cfg.wta,
        batch_size=2,
    )
    reloaded_model.load(ckpt_path)
    assert bool(reloaded_model.net.wta.threshold_initialized.item()) is True

    # Backward-compatible 'autoencoder' alias resolves cleanly to DCAE RawAutoencoderModel
    alias_cfg = PipelineConfig(
        dataset=cfg.dataset,
        model_name="autoencoder",
        autoencoder=AutoencoderConfig(latent_dim=16, base_channels=8, num_stages=3, epochs=1),
        device="cpu",
    )
    alias_cfg.validate()
    alias_model = get_model(alias_cfg, in_channels=3, device=torch.device("cpu"))
    assert type(alias_model) is RawAutoencoderModel
