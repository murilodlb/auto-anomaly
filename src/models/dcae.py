"""Denoising 2D Convolutional Autoencoder (DCAE), training loop, and anomaly map scorer."""

from __future__ import annotations

import functools
import math
import random
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import AutoencoderConfig

_MAX_GPU_STAGE_BYTES = 8 * 1024**3


def _separable_avg_pool2d(t: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Apply fast separable 1D box pooling equivalent to 2D avg_pool2d."""
    pad = window_size // 2
    pooled_h = torch.nn.functional.avg_pool2d(t, (window_size, 1), stride=1, padding=(pad, 0))
    return torch.nn.functional.avg_pool2d(pooled_h, (1, window_size), stride=1, padding=(0, pad))


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute per-pixel (B, C, H, W) Structural Similarity (SSIM) map in [0, 1]."""
    x_f32 = x.float()
    y_f32 = y.float()
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = _separable_avg_pool2d(x_f32, window_size)
    mu_y = _separable_avg_pool2d(y_f32, window_size)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = (_separable_avg_pool2d(x_f32 * x_f32, window_size) - mu_x_sq).clamp(min=0.0)
    sigma_y_sq = (_separable_avg_pool2d(y_f32 * y_f32, window_size) - mu_y_sq).clamp(min=0.0)
    sigma_xy = _separable_avg_pool2d(x_f32 * y_f32, window_size) - mu_xy

    ssim_n = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    ssim_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    return (ssim_n / ssim_d).clamp(0.0, 1.0)


def _to_luminance(x: torch.Tensor) -> torch.Tensor:
    """Convert (B, C, H, W) tensor to 1-channel float32 luminance (B, 1, H, W)."""
    x_f32 = x.float()  # Shape: (B, C, H, W)
    if x_f32.shape[1] == 3:
        return (
            0.299 * x_f32[:, 0:1, :, :]
            + 0.587 * x_f32[:, 1:2, :, :]
            + 0.114 * x_f32[:, 2:3, :, :]
        )  # Shape: (B, 1, H, W)
    return x_f32.mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute depthwise horizontal and vertical 3x3 Sobel gradients (gx, gy)."""
    x_f32 = x.float()  # Shape: (B, C, H, W)
    c_ch = x_f32.shape[1]
    kx_base = (
        torch.tensor(
            [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
            dtype=torch.float32,
            device=x_f32.device,
        )
        / 4.0
    )  # Shape: (3, 3)
    ky_base = kx_base.t().contiguous()  # Shape: (3, 3)
    kx = kx_base.view(1, 1, 3, 3).expand(c_ch, 1, 3, 3).contiguous()  # Shape: (C, 1, 3, 3)
    ky = ky_base.view(1, 1, 3, 3).expand(c_ch, 1, 3, 3).contiguous()  # Shape: (C, 1, 3, 3)

    pad_mode = "reflect" if min(x_f32.shape[-2], x_f32.shape[-1]) > 1 else "replicate"
    x_pad = torch.nn.functional.pad(
        x_f32, (1, 1, 1, 1), mode=pad_mode
    )  # Shape: (B, C, H + 2, W + 2)
    gx = torch.nn.functional.conv2d(x_pad, kx, groups=c_ch)  # Shape: (B, C, H, W)
    gy = torch.nn.functional.conv2d(x_pad, ky, groups=c_ch)  # Shape: (B, C, H, W)
    return gx, gy


def _gms_map(
    x: torch.Tensor,
    y: torch.Tensor,
    c: float = 0.0026,
    window_size: int = 11,
) -> torch.Tensor:
    """Compute smoothed per-pixel (B, C, H, W) Gradient Magnitude Similarity (GMS) in [0, 1]."""
    gx_x, gy_x = _sobel_gradients(x)  # Shape: (B, C, H, W) each
    gx_y, gy_y = _sobel_gradients(y)  # Shape: (B, C, H, W) each

    grad_x = torch.sqrt(gx_x * gx_x + gy_x * gy_x + 1e-12)  # Shape: (B, C, H, W)
    grad_y = torch.sqrt(gx_y * gx_y + gy_y * gy_y + 1e-12)  # Shape: (B, C, H, W)

    gms = (2.0 * grad_x * grad_y + c) / (grad_x * grad_x + grad_y * grad_y + c)
    gms_err = _separable_avg_pool2d(
        (1.0 - gms).clamp(0.0, 1.0), window_size=window_size
    )  # Shape: (B, C, H, W)
    return (1.0 - gms_err).clamp(0.0, 1.0)  # Shape: (B, C, H, W)


def _iter_ms_grad_l1_scales(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> list[tuple[float, torch.Tensor]]:
    """Return weighted (weight, grad_l1_map) pairs on grayscale luminance at 1x, 0.5x, 0.25x."""
    y_hat = _to_luminance(x_hat)  # Shape: (B, 1, H, W)
    y = _to_luminance(x)  # Shape: (B, 1, H, W)
    h, w = y.shape[-2], y.shape[-1]
    maps: list[tuple[float, torch.Tensor]] = []
    for scale, weight in ((1, 0.40), (2, 0.35), (4, 0.25)):
        if min(h, w) < scale:
            continue
        if scale == 1:
            y_hat_s, y_s = y_hat, y  # Shape: (B, 1, H, W)
        else:
            y_hat_s = torch.nn.functional.avg_pool2d(
                y_hat, kernel_size=scale
            )  # Shape: (B, 1, H//scale, W//scale)
            y_s = torch.nn.functional.avg_pool2d(
                y, kernel_size=scale
            )  # Shape: (B, 1, H//scale, W//scale)
        gx_hat, gy_hat = _sobel_gradients(y_hat_s)  # Shape: (B, 1, H_s, W_s) each
        gx, gy = _sobel_gradients(y_s)  # Shape: (B, 1, H_s, W_s) each
        grad_err = 0.5 * ((gx - gx_hat).abs() + (gy - gy_hat).abs())  # Shape: (B, 1, H_s, W_s)
        maps.append((weight, grad_err))
    return maps


def _iter_msgms_error_scales(
    x_hat: torch.Tensor,
    x: torch.Tensor,
) -> list[tuple[float, torch.Tensor]]:
    """Return weighted (weight, error_map) pairs combining SSIM and GMS at 1x, 0.5x, 0.25x."""
    h, w = x.shape[-2], x.shape[-1]
    maps: list[tuple[float, torch.Tensor]] = []
    for scale, weight, win in ((1, 0.35, 11), (2, 0.40, 7), (4, 0.25, 5)):
        if min(h, w) < scale:
            continue
        if scale == 1:
            xh_s, x_s = x_hat, x  # Shape: (B, C, H, W)
        else:
            xh_s = torch.nn.functional.avg_pool2d(
                x_hat.float(), kernel_size=scale
            )  # Shape: (B, C, H//scale, W//scale)
            x_s = torch.nn.functional.avg_pool2d(
                x.float(), kernel_size=scale
            )  # Shape: (B, C, H//scale, W//scale)
        ssim_err = (1.0 - _ssim_map(xh_s, x_s, window_size=win)).mean(
            dim=1, keepdim=True
        )  # Shape: (B, 1, H_s, W_s)
        gms_err = (1.0 - _gms_map(xh_s, x_s, window_size=win)).mean(
            dim=1, keepdim=True
        )  # Shape: (B, 1, H_s, W_s)
        scale_err = 0.5 * ssim_err + 0.5 * gms_err  # Shape: (B, 1, H_s, W_s)
        maps.append((weight, scale_err))
    return maps


def _iter_ssim_error_scales(x_hat: torch.Tensor, x: torch.Tensor) -> list[torch.Tensor]:
    """Return unweighted multi-scale SSIM+GMS error maps at 1.0x, 0.5x, and 0.25x scales."""
    return [err_map for _, err_map in _iter_msgms_error_scales(x_hat, x)]


def _l1_ssim_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Combined L1 + SSIM loss (0.2 * L1 + 0.8 * (1 - SSIM)) for sharp textures."""
    l1 = torch.nn.functional.l1_loss(x_hat.float(), x.float())
    ssim_err = 1.0 - _ssim_map(x_hat, x).mean()
    return 0.2 * l1 + 0.8 * ssim_err


def _ms_l1_ssim_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multi-scale L1 + SSIM + GMS loss across 1.0x, 0.5x, and 0.25x spatial resolutions."""
    l1 = torch.nn.functional.l1_loss(x_hat.float(), x.float())
    weighted_scales = _iter_msgms_error_scales(x_hat, x)
    total_weight = sum(w for w, _ in weighted_scales)
    struct_err = torch.stack([w * s.mean() for w, s in weighted_scales]).sum() / total_weight
    return 0.2 * l1 + 0.8 * struct_err


def _l1_grad_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    grad_weight: float = 2.0,
) -> torch.Tensor:
    """Combined RGB L1 + multi-scale grayscale luminance Sobel directional gradient L1 loss."""
    l1 = torch.nn.functional.l1_loss(x_hat.float(), x.float())  # Shape: ()
    scales = _iter_ms_grad_l1_scales(x_hat, x)
    total_w = sum(w for w, _ in scales)
    grad_l1 = torch.stack([w * s.mean() for w, s in scales]).sum() / total_w  # Shape: ()
    return (l1 + grad_weight * grad_l1) / (1.0 + grad_weight)  # Shape: ()


def _cosine_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Compute channel-dimension-invariant cosine distance loss in [0, 2]."""
    # Shape: x_hat and x are (B, C, H, W)
    cos = torch.nn.functional.cosine_similarity(
        x_hat.float(), x.float(), dim=1, eps=1e-6
    )  # Shape: (B, H, W)
    return (1.0 - cos).mean()  # Shape: ()


_LOSS_FUNCTIONS: dict[str, Any] = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
    "l1_ssim": _l1_ssim_loss,
    "ms_l1_ssim": _ms_l1_ssim_loss,
    "l1_grad": _l1_grad_loss,
    "cosine": _cosine_loss,
}


def _compute_raw_error_map(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    loss_type: str,
    grad_weight: float = 2.0,
) -> torch.Tensor:
    """Compute per-pixel (B, 1, H, W) reconstruction error map in float32."""
    x_hat_f32 = x_hat.float()  # Shape: (B, C, H, W)
    x_f32 = x.float()  # Shape: (B, C, H, W)
    if loss_type == "l1":
        return (x_f32 - x_hat_f32).abs().mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    if loss_type == "smooth_l1":
        return torch.nn.functional.smooth_l1_loss(
            x_hat_f32, x_f32, reduction="none"
        ).mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
    if loss_type == "cosine":
        cos = torch.nn.functional.cosine_similarity(
            x_hat_f32, x_f32, dim=1, eps=1e-6
        )  # Shape: (B, H, W)
        return (1.0 - cos).unsqueeze(1)  # Shape: (B, 1, H, W)
    if loss_type == "l1_ssim":
        l1_map = (x_f32 - x_hat_f32).abs().mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        ssim_err_map = (1.0 - _ssim_map(x_hat_f32, x_f32)).mean(
            dim=1, keepdim=True
        )  # Shape: (B, 1, H, W)
        return 0.2 * l1_map + 0.8 * ssim_err_map  # Shape: (B, 1, H, W)
    if loss_type == "ms_l1_ssim":
        l1_map = (x_f32 - x_hat_f32).abs().mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        target_hw = (x_f32.shape[-2], x_f32.shape[-1])
        weighted_scales = _iter_msgms_error_scales(x_hat_f32, x_f32)
        total_weight = sum(w for w, _ in weighted_scales)
        upsampled_scales = [
            w
            * (
                s
                if s.shape[-2:] == target_hw
                else torch.nn.functional.interpolate(
                    s, size=target_hw, mode="bilinear", align_corners=False
                )
            )
            for w, s in weighted_scales
        ]
        ssim_err_map = (
            torch.stack(upsampled_scales, dim=0).sum(dim=0) / total_weight
        )  # Shape: (B, 1, H, W)
        return 0.2 * l1_map + 0.8 * ssim_err_map  # Shape: (B, 1, H, W)
    if loss_type == "l1_grad":
        l1_map = (x_f32 - x_hat_f32).abs().mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        target_hw = (x_f32.shape[-2], x_f32.shape[-1])
        weighted_scales = _iter_ms_grad_l1_scales(x_hat_f32, x_f32)
        total_w = sum(w for w, _ in weighted_scales)
        upsampled_scales = [
            w
            * (
                s
                if s.shape[-2:] == target_hw
                else torch.nn.functional.interpolate(
                    s, size=target_hw, mode="bilinear", align_corners=False
                )
            )
            for w, s in weighted_scales
        ]
        grad_map = (
            torch.stack(upsampled_scales, dim=0).sum(dim=0) / total_w
        )  # Shape: (B, 1, H, W)
        return (l1_map + grad_weight * grad_map) / (1.0 + grad_weight)  # Shape: (B, 1, H, W)
    return ((x_f32 - x_hat_f32) ** 2).mean(dim=1, keepdim=True)  # Shape: (B, 1, H, W)


def _build_riad_disjoint_masks(
    h: int,
    w: int,
    patch_size: int,
    num_masks: int,
    device: torch.device,
    shift_hw: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Build exact disjoint (M, 1, H, W) binary RIAD masks summing to 1.0 at every pixel."""
    k = min(patch_size, max(2, min(h, w) // 2))
    dy = shift_hw[0] % k
    dx = shift_hw[1] % k
    n_rows = math.ceil((h + k) / k)
    n_cols = math.ceil((w + k) / k)

    side = math.isqrt(num_masks)
    if side * side == num_masks:
        rows = torch.arange(n_rows, device=device).view(n_rows, 1)  # Shape: (n_rows, 1)
        cols = torch.arange(n_cols, device=device).view(1, n_cols)  # Shape: (1, n_cols)
        bucket = (rows % side) * side + (cols % side)  # Shape: (n_rows, n_cols)
    else:
        total_cells = n_rows * n_cols
        perm_gen = torch.Generator(device="cpu").manual_seed(k * 1009 + num_masks)
        perm = torch.randperm(total_cells, generator=perm_gen)  # Shape: (n_rows * n_cols,)
        bucket = (
            (perm % num_masks).view(n_rows, n_cols).to(device=device)
        )  # Shape: (n_rows, n_cols)

    pixel_buckets = (
        bucket.repeat_interleave(k, dim=0)
        .repeat_interleave(k, dim=1)[dy : dy + h, dx : dx + w]
    )  # Shape: (H, W)
    mask_ids = torch.arange(num_masks, device=device).view(num_masks, 1, 1)  # Shape: (M, 1, 1)
    masks = (
        (pixel_buckets.unsqueeze(0) == mask_ids).unsqueeze(1).to(torch.float32)
    )  # Shape: (M, 1, H, W)
    return masks


def _riad_corrupt_batch(
    x: torch.Tensor,
    patch_sizes: tuple[int, ...],
    num_masks: int,
    prob: float,
    seed: int,
) -> torch.Tensor:
    """Apply RIAD-style disjoint grid masking with per-image channel mean fill during fit()."""
    # Shape: x is (B, C, H, W)
    if prob <= 0.0 or len(patch_sizes) == 0 or x.numel() == 0:
        return x

    b, _, h, w = x.shape
    x_noisy = x.clone()  # Shape: (B, C, H, W)
    fill = x.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
    rng = random.Random(seed)
    max_k = max(2, min(h, w) // 2)

    for i in range(b):
        if rng.random() >= prob:
            continue
        k = min(rng.choice(patch_sizes), max_k)
        dy = rng.randrange(k)
        dx = rng.randrange(k)
        masks = _build_riad_disjoint_masks(
            h,
            w,
            patch_size=k,
            num_masks=num_masks,
            device=x.device,
            shift_hw=(dy, dx),
        )  # Shape: (M, 1, H, W)
        m = rng.randrange(num_masks)
        mask_m = masks[m].to(dtype=x.dtype)  # Shape: (1, H, W)
        x_noisy[i] = x[i] * (1.0 - mask_m) + fill[i] * mask_m  # Shape: (C, H, W)

    return x_noisy  # Shape: (B, C, H, W)


def _corrupt_batch(
    x: torch.Tensor,
    denoise_prob: float,
    seed: int | None = None,
) -> torch.Tensor:
    """Apply CutPaste, streak, jitter, and Gaussian noise corruptions for denoising training."""
    if denoise_prob <= 0.0 or x.numel() == 0:
        return x

    b, c, h, w = x.shape
    x_noisy = x.clone()
    device = x.device
    dtype = x.dtype
    rng = random.Random(seed)
    torch_gen = torch.Generator(device=device).manual_seed(seed) if seed is not None else None

    # Keep dynamic range bounds on-device so CUDA training never stalls on scalar syncs.
    x_min_t = x.amin()
    x_max_t = x.amax()
    val_span_t = (x_max_t - x_min_t).clamp(min=1e-3)

    min_log_ar = math.log(0.15)
    max_log_ar = math.log(6.5)

    for i in range(b):
        if rng.random() >= denoise_prob:
            continue
        if rng.random() < 0.35:
            cell_size = min(rng.choice((16, 32)), max(4, min(h, w) // 4))
            n_rows = max(1, (h + cell_size - 1) // cell_size)
            n_cols = max(1, (w + cell_size - 1) // cell_size)
            total_cells = n_rows * n_cols
            num_masked = max(1, min(total_cells, round(total_cells * rng.uniform(0.15, 0.25))))
            ch_mean = x_noisy[i].mean(dim=(1, 2), keepdim=True)
            for cell_idx in rng.sample(range(total_cells), k=num_masked):
                y0 = (cell_idx // n_cols) * cell_size
                y1 = min(h, y0 + cell_size)
                x0 = (cell_idx % n_cols) * cell_size
                x1 = min(w, x0 + cell_size)
                jitter = (
                    torch.randn(
                        (c, y1 - y0, x1 - x0),
                        device=device,
                        dtype=dtype,
                        generator=torch_gen,
                    )
                    * (0.05 * val_span_t)
                )
                x_noisy[i, :, y0:y1, x0:x1] = (ch_mean + jitter).clamp(x_min_t, x_max_t)
            continue

        num_patches = rng.randint(1, 3)
        for _ in range(num_patches):
            target_area = rng.uniform(0.01, 0.16) * float(h * w)
            aspect = math.exp(rng.uniform(min_log_ar, max_log_ar))
            ph = min(h, max(2, round(math.sqrt(target_area * aspect))))
            pw = min(w, max(2, round(math.sqrt(target_area / aspect))))

            y0 = rng.randint(0, max(0, h - ph))
            x0 = rng.randint(0, max(0, w - pw))
            mode = rng.randint(0, 3)

            if mode == 0:
                streak_mode = rng.randint(0, 2)
                region = x_noisy[i, :, y0 : y0 + ph, x0 : x0 + pw]
                if streak_mode == 0:
                    scale = rng.uniform(0.1, 0.4)
                    patch = region * scale - 0.1 * val_span_t
                elif streak_mode == 1:
                    delta = rng.uniform(0.25, 0.75) * val_span_t
                    patch = region + delta
                else:
                    jitter = (
                        torch.rand((c, 1, 1), device=device, dtype=dtype, generator=torch_gen)
                        - 0.5
                    ) * (0.8 * val_span_t)
                    patch = region + jitter
            elif mode == 1:
                src_i = rng.randrange(b)
                sy0 = rng.randint(0, max(0, h - ph))
                sx0 = rng.randint(0, max(0, w - pw))
                patch = x[src_i, :, sy0 : sy0 + ph, sx0 : sx0 + pw].clone()
            elif mode == 2:
                sigma = rng.uniform(0.15, 0.45) * val_span_t
                noise = torch.randn(
                    (c, ph, pw), device=device, dtype=dtype, generator=torch_gen
                )
                patch = x_noisy[i, :, y0 : y0 + ph, x0 : x0 + pw] + noise * sigma
            else:
                yy = torch.linspace(-1.0, 1.0, steps=ph, device=device, dtype=dtype).view(ph, 1)
                xx = torch.linspace(-1.0, 1.0, steps=pw, device=device, dtype=dtype).view(1, pw)
                sigma_y = rng.uniform(0.35, 0.55)
                sigma_x = rng.uniform(0.35, 0.55)
                gauss_2d = torch.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
                stain_factor = (
                    rng.uniform(0.25, 0.65) if rng.random() < 0.5 else rng.uniform(1.35, 1.75)
                )
                stain_scale = torch.tensor(
                    [max(0.15, stain_factor * rng.uniform(0.85, 1.15)) for _ in range(c)],
                    device=device,
                    dtype=dtype,
                ).view(c, 1, 1)
                mult = 1.0 + (stain_scale - 1.0) * gauss_2d.unsqueeze(0)
                patch = x_noisy[i, :, y0 : y0 + ph, x0 : x0 + pw] * mult

            x_noisy[i, :, y0 : y0 + ph, x0 : x0 + pw] = patch.clamp(x_min_t, x_max_t)

    return x_noisy


class ConvAutoencoder(torch.nn.Module):
    """Symmetric fully-convolutional 2D autoencoder operating on (B, C, H, W) tensors.

    Example shape flow for H=W=1024, in_channels=3, base_channels=64, num_stages=6, latent_dim=64:
      Encoder stage 0: (B,   3, 1024, 1024) -> (B,  64, 512, 512)
      Encoder stage 1: (B,  64,  512,  512) -> (B, 128, 256, 256)
      Encoder stage 2: (B, 128,  256,  256) -> (B, 256, 128, 128)
      Encoder stage 3: (B, 256,  128,  128) -> (B, 512,  64,  64)
      Encoder stage 4: (B, 512,   64,   64) -> (B, 512,  32,  32)
      Encoder stage 5: (B, 512,   32,   32) -> (B, 512,  16,  16)
      Bottleneck:      (B, 512,   16,   16) -> (B,  64,  16,  16)  [3x3 + IN2d + Dropout2d + 3x3]
      Decoder stages:  (B,  64,   16,   16) -> (B, 512,  16,  16) -> ... -> (B,   3, 1024, 1024)
    """

    err_mean: torch.Tensor
    err_std: torch.Tensor
    err_p99: torch.Tensor
    calibrated: torch.Tensor
    normal_mu_map: torch.Tensor
    normal_std_map: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        latent_dim: int = 128,
        num_stages: int = 4,
        spatial_dropout: float = 0.1,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        if not (0.0 <= spatial_dropout < 1.0):
            raise ValueError(f"spatial_dropout must be in [0.0, 1.0), got {spatial_dropout}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.num_stages = num_stages
        self.spatial_dropout = spatial_dropout
        self.use_sigmoid = use_sigmoid

        # Persistent calibration buffers (saved/loaded automatically inside state_dict)
        self.register_buffer("err_mean", torch.zeros(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("err_std", torch.ones(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("err_p99", torch.zeros(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("calibrated", torch.zeros(1, dtype=torch.bool))  # Shape: (1,)
        self.register_buffer(
            "normal_mu_map", torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        )  # Shape: (1, 1, 16, 16)
        self.register_buffer(
            "normal_std_map", torch.ones((1, 1, 16, 16), dtype=torch.float32)
        )  # Shape: (1, 1, 16, 16)

        stage_channels = [min(base_channels * (2**i), 512) for i in range(num_stages)]

        enc_layers: list[torch.nn.Module] = []
        ch_in = in_channels  # Shape at input: (B, C_in, H, W)
        for ch_out in stage_channels:
            enc_layers.extend(
                [
                    # Shape: (B, ch_in, H/2^i, W/2^i) -> (B, ch_out, H/2^(i+1), W/2^(i+1))
                    torch.nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                    # Shape: (B, ch_out, H/2^(i+1), W/2^(i+1))
                    torch.nn.BatchNorm2d(ch_out),
                    # Shape: (B, ch_out, H/2^(i+1), W/2^(i+1))
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch_in = ch_out  # Next stage input channels: ch_out at H_b = H // 2**num_stages
        enc_layers.extend(
            [
                # Shape: (B, stage_channels[-1], H_b, W_b) -> (B, latent_dim, H_b, W_b)
                torch.nn.Conv2d(ch_in, latent_dim, kernel_size=3, padding=1, bias=False),
                # Shape: (B, latent_dim, H_b, W_b) — per-image InstanceNorm2d strips stain bias
                torch.nn.InstanceNorm2d(latent_dim, affine=True),
                # Shape: (B, latent_dim, H_b, W_b)
                torch.nn.LeakyReLU(0.2, inplace=True),
                # Shape: (B, latent_dim, H_b, W_b) — drops whole feature maps during training
                torch.nn.Dropout2d(p=spatial_dropout),
                # Shape: (B, latent_dim, H_b, W_b) -> (B, latent_dim, H_b, W_b)
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                # Shape: (B, latent_dim, H_b, W_b)
                torch.nn.InstanceNorm2d(latent_dim, affine=True),
                # Shape: (B, latent_dim, H_b, W_b)
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.encoder = torch.nn.Sequential(*enc_layers)

        dec_layers: list[torch.nn.Module] = [
            # Shape: (B, latent_dim, H_b, W_b) -> (B, stage_channels[-1], H_b, W_b)
            torch.nn.Conv2d(latent_dim, stage_channels[-1], kernel_size=1, bias=False),
            # Shape: (B, stage_channels[-1], H_b, W_b)
            torch.nn.BatchNorm2d(stage_channels[-1]),
            # Shape: (B, stage_channels[-1], H_b, W_b)
            torch.nn.ReLU(inplace=True),
        ]
        for idx in range(num_stages - 1, 0, -1):
            c_in = stage_channels[idx]
            c_out = stage_channels[idx - 1]
            dec_layers.extend(
                [
                    # Shape: (B, c_in, H_s, W_s) -> (B, c_out, 2*H_s, 2*W_s)
                    torch.nn.ConvTranspose2d(
                        c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False
                    ),
                    # Shape: (B, c_out, 2*H_s, 2*W_s)
                    torch.nn.BatchNorm2d(c_out),
                    # Shape: (B, c_out, 2*H_s, 2*W_s)
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        dec_layers.extend(
            [
                # Shape: (B, stage_channels[0], H/2, W/2) -> (B, in_channels, H, W)
                torch.nn.ConvTranspose2d(
                    stage_channels[0], in_channels, kernel_size=4, stride=2, padding=1, bias=True
                ),
                # Shape: (B, in_channels, H, W) in [0, 1] (Sigmoid) or unbounded (Identity)
                torch.nn.Sigmoid() if use_sigmoid else torch.nn.Identity(),
            ]
        )
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: x is (B, C_in, H, W)
        n_down = 3 * self.num_stages
        h_enc = self.encoder[:n_down](x)  # Shape: (B, C_last, H_b, W_b), H_b = H // 2**num_stages
        z = self.encoder[n_down:](h_enc)  # Shape: (B, Z, H_b, W_b) after 3x3 context + Dropout2d
        h_dec = self.decoder[:3](z)  # Shape: (B, C_last, H_b, W_b) after 1x1 decoder projection
        x_hat = self.decoder[3:](h_dec)  # Shape: (B, C_in, H, W) after num_stages ConvTranspose2d
        return x_hat


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable 2D Gaussian smoothing to a (B, 1, H, W) tensor."""
    if sigma <= 0.0:
        return tensor

    radius = max(1, math.ceil(3.0 * sigma))
    coords = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)

    min_dim = min(tensor.shape[-2], tensor.shape[-1])
    pad_mode = "reflect" if radius < min_dim else "replicate"
    padded = torch.nn.functional.pad(tensor, (radius, radius, radius, radius), mode=pad_mode)
    blurred = torch.nn.functional.conv2d(padded, kernel_x)
    return torch.nn.functional.conv2d(blurred, kernel_y)


def _morphological_open_2d(tensor: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """Apply greyscale 2D morphological opening (erosion followed by dilation)."""
    pad = kernel_size // 2
    eroded = -torch.nn.functional.max_pool2d(
        -tensor, kernel_size=kernel_size, stride=1, padding=pad
    )
    return torch.nn.functional.max_pool2d(
        eroded, kernel_size=kernel_size, stride=1, padding=pad
    )


def _guided_filter_2d(
    guide_rgb: torch.Tensor,
    src: torch.Tensor,
    radius: int,
    eps: float,
) -> torch.Tensor:
    """Apply Fast Guided Filter using guide_rgb luminance edges to sharpen anomaly map src."""
    # Shape: guide_rgb is (B, C_g, H_g, W_g), src is (B, 1, H, W)
    if radius <= 0:
        return src  # Shape: (B, 1, H, W)

    guide = _to_luminance(guide_rgb)  # Shape: (B, 1, H_g, W_g)
    src_f32 = src.float()  # Shape: (B, 1, H, W)

    h, w = src_f32.shape[-2], src_f32.shape[-1]
    if guide.shape[-2:] != (h, w):
        guide = torch.nn.functional.interpolate(
            guide, size=(h, w), mode="bilinear", align_corners=False
        )  # Shape: (B, 1, H, W)

    subsample = 2 if (min(h, w) >= 512 and radius >= 4) else 1
    r_eff = max(1, radius // subsample)
    win = 2 * r_eff + 1

    if subsample > 1:
        g_s = torch.nn.functional.interpolate(
            guide, scale_factor=1.0 / subsample, mode="bilinear", align_corners=False
        )  # Shape: (B, 1, H_s, W_s)
        p_s = torch.nn.functional.interpolate(
            src_f32, scale_factor=1.0 / subsample, mode="bilinear", align_corners=False
        )  # Shape: (B, 1, H_s, W_s)
    else:
        g_s = guide  # Shape: (B, 1, H, W)
        p_s = src_f32  # Shape: (B, 1, H, W)

    mean_g = _separable_avg_pool2d(g_s, window_size=win)  # Shape: (B, 1, H_s, W_s)
    mean_p = _separable_avg_pool2d(p_s, window_size=win)  # Shape: (B, 1, H_s, W_s)
    corr_g = _separable_avg_pool2d(g_s * g_s, window_size=win)  # Shape: (B, 1, H_s, W_s)
    corr_gp = _separable_avg_pool2d(g_s * p_s, window_size=win)  # Shape: (B, 1, H_s, W_s)

    var_g = (corr_g - mean_g * mean_g).clamp(min=0.0)  # Shape: (B, 1, H_s, W_s)
    cov_gp = corr_gp - mean_g * mean_p  # Shape: (B, 1, H_s, W_s)

    a = cov_gp / (var_g + eps)  # Shape: (B, 1, H_s, W_s)
    b = mean_p - a * mean_g  # Shape: (B, 1, H_s, W_s)

    mean_a = _separable_avg_pool2d(a, window_size=win)  # Shape: (B, 1, H_s, W_s)
    mean_b = _separable_avg_pool2d(b, window_size=win)  # Shape: (B, 1, H_s, W_s)

    if subsample > 1:
        mean_a = torch.nn.functional.interpolate(
            mean_a, size=(h, w), mode="bilinear", align_corners=False
        )  # Shape: (B, 1, H, W)
        mean_b = torch.nn.functional.interpolate(
            mean_b, size=(h, w), mode="bilinear", align_corners=False
        )  # Shape: (B, 1, H, W)

    return (mean_a * guide + mean_b).clamp(min=0.0)  # Shape: (B, 1, H, W)


def _uint8_nhwc_to_unit_nchw(batch_images: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert uint8 (B, H, W, C) numpy array to float32 (B, C, H, W) in [0.0, 1.0]."""
    contiguous = np.ascontiguousarray(batch_images)  # Shape: (B, H, W, C)
    tensor_u8 = torch.from_numpy(contiguous).to(
        device=device, non_blocking=True
    )  # Shape: (B, H, W, C)
    return tensor_u8.permute(0, 3, 1, 2).to(dtype=torch.float32).div_(255.0)  # Shape: (B, C, H, W)


class RawAutoencoderModel:
    """Training, inference, and anomaly-map scoring wrapper around ConvAutoencoder."""

    def __init__(
        self,
        in_channels: int,
        config: AutoencoderConfig,
        device: torch.device,
        batch_size: int = 16,
        seed: int = 42,
        use_sigmoid: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.config = config
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.use_sigmoid = use_sigmoid

        self.net: Any = self._build_network(
            in_channels=in_channels, use_sigmoid=use_sigmoid
        ).to(device=device)
        if device.type == "cuda":
            self.net.to(memory_format=torch.channels_last)

        self._use_amp = bool(
            config.use_amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
        )

        if config.loss_type not in _LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_type: {config.loss_type!r}")
        self._loss_fn: Any = (
            functools.partial(_l1_grad_loss, grad_weight=config.grad_loss_weight)
            if config.loss_type == "l1_grad"
            else _LOSS_FUNCTIONS[config.loss_type]
        )
        self._riad_mask_cache: dict[tuple[int, int, int, int], torch.Tensor] = {}

    def _build_network(self, in_channels: int, use_sigmoid: bool) -> Any:
        """Instantiate the underlying autoencoder network."""
        return ConvAutoencoder(
            in_channels=in_channels,
            base_channels=self.config.base_channels,
            latent_dim=self.config.latent_dim,
            num_stages=self.config.num_stages,
            spatial_dropout=self.config.spatial_dropout,
            use_sigmoid=use_sigmoid,
        )

    def _checkpoint_meta(self) -> dict[str, Any]:
        """Build architecture metadata dictionary for checkpoint persistence."""
        return {
            "model_type": "dcae",
            "in_channels": self.in_channels,
            "base_channels": self.config.base_channels,
            "latent_dim": self.config.latent_dim,
            "num_stages": self.config.num_stages,
            "spatial_dropout": getattr(self.net, "spatial_dropout", self.config.spatial_dropout),
            "use_sigmoid": self.net.use_sigmoid,
        }

    def _expected_checkpoint_meta(self) -> dict[str, Any]:
        """Return key-value pairs that must match when loading a checkpoint."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.config.base_channels,
            "latent_dim": self.config.latent_dim,
            "num_stages": self.config.num_stages,
        }

    def _on_epoch_start(self, epoch: int) -> None:
        """Optional hook called at the start of each training epoch."""
        _ = epoch

    def _maybe_cache_train_features(
        self,
        train_images: np.ndarray,
        extractor: Any,
    ) -> torch.Tensor | None:
        """Pre-extract frozen extractor feature maps once before the training epoch loop."""
        if not getattr(extractor, "cache_features_for_training", False) or len(train_images) == 0:
            return None

        cache_dtype = torch.bfloat16 if self._use_amp else torch.float32
        chunks: list[torch.Tensor] = []
        num_samples = len(train_images)
        with torch.inference_mode():
            for start in range(0, num_samples, self.batch_size):
                batch_np = np.ascontiguousarray(
                    train_images[start : start + self.batch_size]
                )  # Shape: (B, H, W, C)
                feats = extractor.extract_batch(batch_np)  # Shape: (B, C_f, H_p, W_p)
                chunks.append(
                    feats.to(device="cpu", dtype=cache_dtype)
                )  # Shape: (B, C_f, H_p, W_p)

        cached = torch.cat(chunks, dim=0).contiguous()  # Shape: (N, C_f, H_p, W_p)
        if self.device.type == "cuda":
            try:
                free_bytes, _ = torch.cuda.mem_get_info(self.device)
                max_stage_bytes = min(_MAX_GPU_STAGE_BYTES, int(free_bytes * 0.5))
                if cached.nbytes < max_stage_bytes:
                    cached = cached.to(
                        device=self.device, memory_format=torch.channels_last
                    )  # Shape: (N, C_f, H_p, W_p)
            except RuntimeError:
                pass

        print(
            f"[Cache] Pre-extracted {num_samples} feature maps of shape {tuple(cached.shape)} "
            f"({cached.nbytes / 1e6:.1f} MB) on {cached.device}",
            flush=True,
        )
        return cached

    def _maybe_preload_train_gpu(
        self,
        train_images: np.ndarray,
        extractor: Any,
    ) -> torch.Tensor | None:
        """Cache uint8 training images in VRAM when CUDA is active and memory permits."""
        if getattr(extractor, "cache_features_for_training", False):
            return None
        if self.device.type != "cuda" or not hasattr(extractor, "normalize_uint8_tensor"):
            return None
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return None
        max_stage_bytes = min(_MAX_GPU_STAGE_BYTES, int(free_bytes * 0.5))
        if train_images.nbytes >= max_stage_bytes:
            return None
        return (
            torch.from_numpy(np.ascontiguousarray(train_images))
            .to(device=self.device)
            .permute(0, 3, 1, 2)
            .contiguous(memory_format=torch.channels_last)
        )  # Shape: (N, C, H, W)

    def _iter_train_batches(
        self,
        train_images: np.ndarray,
        extractor: Any,
        train_gpu_u8: torch.Tensor | None,
        epoch: int,
        shuffle: bool = True,
        cached_features: torch.Tensor | None = None,
    ) -> Iterator[tuple[np.ndarray, torch.Tensor]]:
        """Yield (indices, (B, C, H, W) float32 batch) from feature cache, GPU uint8, or CPU."""
        num_samples = len(train_images)
        indices = (
            np.random.default_rng(self.seed + epoch).permutation(num_samples)
            if shuffle
            else np.arange(num_samples)
        )

        if cached_features is not None:
            mem_fmt = torch.channels_last if self.device.type == "cuda" else torch.contiguous_format
            for start in range(0, num_samples, self.batch_size):
                idx_slice = indices[start : start + self.batch_size]
                idx_t = torch.from_numpy(idx_slice).to(device=cached_features.device)
                batch_feat = cached_features[idx_t]  # Shape: (B, C_f, H_p, W_p)
                x = batch_feat.to(
                    device=self.device,
                    dtype=torch.float32,
                    memory_format=mem_fmt,
                    non_blocking=True,
                )  # Shape: (B, C_f, H_p, W_p)
                yield idx_slice, x
            return

        if train_gpu_u8 is not None:
            for start in range(0, num_samples, self.batch_size):
                idx_slice = indices[start : start + self.batch_size]
                idx_t = torch.from_numpy(idx_slice).to(device=self.device)
                batch_u8 = train_gpu_u8[idx_t]  # Shape: (B, C, H, W)
                yield idx_slice, extractor.normalize_uint8_tensor(batch_u8)
            return

        for start in range(0, num_samples, self.batch_size):
            idx_slice = indices[start : start + self.batch_size]
            batch_np = np.ascontiguousarray(train_images[idx_slice])  # Shape: (B, H, W, C)
            yield idx_slice, extractor.extract_batch(batch_np)

    def _get_riad_masks(
        self,
        h: int,
        w: int,
        patch_size: int,
        num_masks: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Fetch or build cached disjoint (M, 1, H, W) RIAD masks on device."""
        key = (h, w, patch_size, num_masks)
        cached = self._riad_mask_cache.get(key)
        if cached is None or cached.device != device:
            cached = _build_riad_disjoint_masks(
                h, w, patch_size=patch_size, num_masks=num_masks, device=device, shift_hw=(0, 0)
            )  # Shape: (M, 1, H, W)
            self._riad_mask_cache[key] = cached
        return cached

    def _forward_riad_eval(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-scale disjoint RIAD masked inpainting and average per-scale error maps."""
        # Shape: x is (B, C, H, W)
        h, w = x.shape[-2], x.shape[-1]
        fill = x.mean(dim=(2, 3), keepdim=True)  # Shape: (B, C, 1, 1)
        num_masks = self.config.riad_num_masks
        scale_recons: list[torch.Tensor] = []
        scale_err_maps: list[torch.Tensor] = []

        for patch_size in self.config.riad_patch_sizes:
            masks = self._get_riad_masks(
                h, w, patch_size=patch_size, num_masks=num_masks, device=x.device
            )  # Shape: (M, 1, H, W)
            x_hat_scale = torch.zeros_like(x, dtype=torch.float32)  # Shape: (B, C, H, W)
            for m in range(num_masks):
                mask_m = masks[m : m + 1].to(dtype=x.dtype)  # Shape: (1, 1, H, W)
                x_masked = x * (1.0 - mask_m) + fill * mask_m  # Shape: (B, C, H, W)
                if self.device.type == "cuda":
                    x_masked = x_masked.contiguous(memory_format=torch.channels_last)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                    x_hat_m = self.net(x_masked)  # Shape: (B, C, H, W)
                x_hat_scale = (
                    x_hat_scale + x_hat_m.float() * mask_m.float()
                )  # Shape: (B, C, H, W)

            err_scale = _compute_raw_error_map(
                x_hat_scale,
                x,
                self.config.loss_type,
                grad_weight=self.config.grad_loss_weight,
            )  # Shape: (B, 1, H, W)
            scale_recons.append(x_hat_scale)
            scale_err_maps.append(err_scale)

        x_hat_mean = torch.stack(scale_recons, dim=0).mean(dim=0)  # Shape: (B, C, H, W)
        err_map = torch.stack(scale_err_maps, dim=0).mean(dim=0)  # Shape: (B, 1, H, W)
        return x_hat_mean, err_map

    def _score_batch_tensor(
        self,
        x: torch.Tensor,
        target_hw: tuple[int, int] | None,
        apply_calibration: bool = True,
        guide_rgb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction (float32) and damped/calibrated error map (B, 1, H, W)."""
        if len(self.config.riad_patch_sizes) > 0:
            x_hat_f32, err_map = self._forward_riad_eval(
                x
            )  # Shape: (B, C, H_in, W_in), (B, 1, H_in, W_in)
        else:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                x_hat = self.net(x)  # Shape: (B, C, H_in, W_in)
            x_hat_f32 = x_hat.float()  # Shape: (B, C, H_in, W_in)

            err_map = _compute_raw_error_map(
                x_hat_f32,
                x,
                self.config.loss_type,
                grad_weight=self.config.grad_loss_weight,
            )  # Shape: (B, 1, H_in, W_in)

        # Apply texture variance damping only in [0, 1] pixel space (not ViT patch grids)
        if self.use_sigmoid and self.config.texture_damping > 0.0:
            mu_hat = _separable_avg_pool2d(x_hat_f32, window_size=11)  # Shape: (B, C, H_in, W_in)
            sq_hat = _separable_avg_pool2d(
                x_hat_f32 * x_hat_f32, window_size=11
            )  # Shape: (B, C, H_in, W_in)
            var_hat = (sq_hat - mu_hat * mu_hat).clamp(min=0.0)  # Shape: (B, C, H_in, W_in)
            texture_std = torch.sqrt(
                var_hat.mean(dim=1, keepdim=True).clamp(min=0.0)
            )  # Shape: (B, 1, H_in, W_in)
            err_map = err_map / (
                1.0 + self.config.texture_damping * texture_std
            )  # Shape: (B, 1, H_in, W_in)

        if target_hw is not None and err_map.shape[-2:] != target_hw:
            err_map = torch.nn.functional.interpolate(
                err_map,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )  # Shape: (B, 1, H_out, W_out)

        if min(err_map.shape[-2], err_map.shape[-1]) >= 32:
            err_map = _morphological_open_2d(err_map, kernel_size=3)  # Shape: (B, 1, H_out, W_out)

        err_map = _gaussian_blur_2d(
            err_map, self.config.gaussian_sigma
        )  # Shape: (B, 1, H_out, W_out)

        if self.config.guided_filter_radius > 0 and guide_rgb is not None:
            err_map = _guided_filter_2d(
                guide_rgb,
                err_map,
                radius=self.config.guided_filter_radius,
                eps=self.config.guided_filter_eps,
            )  # Shape: (B, 1, H_out, W_out)

        if apply_calibration and bool(self.net.calibrated.item()):
            mu_map = 0.5 * self.net.err_mean + 0.5 * torch.nn.functional.interpolate(
                self.net.normal_mu_map,
                size=err_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )  # Shape: (1, 1, H_out, W_out)
            std_map = (
                0.5 * self.net.err_std
                + 0.5
                * torch.nn.functional.interpolate(
                    self.net.normal_std_map,
                    size=err_map.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            ).clamp(min=1e-4)  # Shape: (1, 1, H_out, W_out)
            z_map = (err_map - mu_map) / std_map  # Shape: (B, 1, H_out, W_out)
            err_map = torch.nn.functional.softplus(
                z_map - 1.5, beta=2.0
            )  # Shape: (B, 1, H_out, W_out)

        return x_hat_f32, err_map

    def _calibrate_normal_baseline(
        self,
        train_images: np.ndarray,
        extractor: Any,
        train_gpu_u8: torch.Tensor | None,
        cached_features: torch.Tensor | None = None,
    ) -> None:
        """Estimate defect-free training pixel error baseline statistics in eval mode."""
        self.net.eval()
        sum_err = torch.zeros((), dtype=torch.float64, device=self.device)
        sum_err_sq = torch.zeros((), dtype=torch.float64, device=self.device)
        spatial_sum = torch.zeros((1, 1, 16, 16), dtype=torch.float64, device=self.device)
        spatial_sum_sq = torch.zeros((1, 1, 16, 16), dtype=torch.float64, device=self.device)
        total_pixels = 0
        total_images = 0

        ds_cfg = getattr(extractor, "dataset_cfg", None)
        if ds_cfg is not None and hasattr(ds_cfg, "image_size"):
            target_hw = (int(ds_cfg.image_size[0]), int(ds_cfg.image_size[1]))
        else:
            target_hw = (int(train_images.shape[1]), int(train_images.shape[2]))

        with torch.inference_mode():
            for idx_slice, x in self._iter_train_batches(
                train_images,
                extractor,
                train_gpu_u8,
                epoch=0,
                shuffle=False,
                cached_features=cached_features,
            ):
                guide_rgb: torch.Tensor | None = None
                if self.config.guided_filter_radius > 0:
                    if self.use_sigmoid and cached_features is None and x.shape[-2:] == target_hw:
                        guide_rgb = x  # Shape: (B, C, H, W)
                    elif train_gpu_u8 is not None:
                        idx_t = torch.from_numpy(idx_slice).to(device=train_gpu_u8.device)
                        guide_rgb = (
                            train_gpu_u8[idx_t]
                            .to(device=self.device, dtype=torch.float32)
                            .div_(255.0)
                        )  # Shape: (B, C_img, H, W)
                    else:
                        guide_rgb = _uint8_nhwc_to_unit_nchw(
                            train_images[idx_slice], self.device
                        )  # Shape: (B, C_img, H, W)

                _, err_map = self._score_batch_tensor(
                    x,
                    target_hw=target_hw,
                    apply_calibration=False,
                    guide_rgb=guide_rgb,
                )  # Shape: (B, 1, H_t, W_t)
                err_f64 = err_map.to(torch.float64)  # Shape: (B, 1, H_t, W_t)
                sum_err += err_f64.sum()
                sum_err_sq += (err_f64 * err_f64).sum()
                total_pixels += err_map.numel()

                pooled = torch.nn.functional.adaptive_avg_pool2d(err_map, (16, 16)).to(
                    torch.float64
                )  # Shape: (B, 1, 16, 16)
                spatial_sum += pooled.sum(dim=0, keepdim=True)
                spatial_sum_sq += (pooled * pooled).sum(dim=0, keepdim=True)
                total_images += x.shape[0]

        if total_pixels == 0 or total_images == 0:
            return

        mean_val = sum_err / float(total_pixels)
        var_val = (sum_err_sq / float(total_pixels) - mean_val * mean_val).clamp(min=0.0)
        std_val = torch.sqrt(var_val).clamp(min=1e-4)
        p99_val = mean_val + 2.5 * std_val

        sp_mean = spatial_sum / float(total_images)
        sp_var = (spatial_sum_sq / float(total_images) - sp_mean * sp_mean).clamp(min=0.0)
        sp_std = torch.sqrt(sp_var).clamp(min=1e-4)

        self.net.err_mean.copy_(mean_val.view(1).to(torch.float32))
        self.net.err_std.copy_(std_val.view(1).to(torch.float32))
        self.net.err_p99.copy_(p99_val.view(1).to(torch.float32))
        self.net.normal_mu_map.copy_(sp_mean.to(torch.float32))
        self.net.normal_std_map.copy_(sp_std.to(torch.float32))
        self.net.calibrated.fill_(True)
        print(
            f"[Calibration] err_mean={float(self.net.err_mean.item()):.6f} "
            f"err_std={float(self.net.err_std.item()):.6f} "
            f"err_p99={float(self.net.err_p99.item()):.6f}",
            flush=True,
        )

    def fit(self, train_images: np.ndarray, extractor: Any) -> list[float]:
        """Train the autoencoder on normal images and return per-epoch average losses."""
        if (
            not self.use_sigmoid
            and self.config.loss_type == "l2"
            and self.config.loss_floor > 0.0
            and self.config.loss_floor >= (1.0 / max(1, self.in_channels))
        ):
            print(
                f"[WARN] loss_floor={self.config.loss_floor:.6f} >= 1/C "
                f"({1.0 / max(1, self.in_channels):.6f}) with feature-space 'l2' loss; "
                "early stopping may trigger at epoch 21. Consider --loss cosine or --loss-floor 0.",
                flush=True,
            )

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * 0.05,
        )
        num_samples = len(train_images)
        epoch_losses: list[float] = []
        consecutive_floor_hits = 0
        cached_features = self._maybe_cache_train_features(train_images, extractor)
        train_gpu_u8 = (
            None
            if cached_features is not None
            else self._maybe_preload_train_gpu(train_images, extractor)
        )

        try:
            for epoch in range(1, self.config.epochs + 1):
                self._on_epoch_start(epoch)
                self.net.train()
                t0 = time.perf_counter()
                running_loss = torch.zeros((), dtype=torch.float64, device=self.device)

                for step_idx, (_idx_slice, x) in enumerate(
                    self._iter_train_batches(
                        train_images,
                        extractor,
                        train_gpu_u8,
                        epoch,
                        shuffle=True,
                        cached_features=cached_features,
                    )
                ):
                    optimizer.zero_grad(set_to_none=True)
                    if self.config.denoise_prob > 0.0:
                        step_seed = self.seed * 100003 + epoch * 1009 + step_idx
                        if len(self.config.riad_patch_sizes) > 0:
                            x_in = _riad_corrupt_batch(
                                x,
                                patch_sizes=self.config.riad_patch_sizes,
                                num_masks=self.config.riad_num_masks,
                                prob=self.config.denoise_prob,
                                seed=step_seed,
                            )  # Shape: (B, C, H_in, W_in)
                        else:
                            x_in = _corrupt_batch(
                                x,
                                self.config.denoise_prob,
                                seed=step_seed,
                            )  # Shape: (B, C, H_in, W_in)
                    else:
                        x_in = x  # Shape: (B, C, H_in, W_in)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=self._use_amp):
                        x_hat = self.net(x_in)  # Shape: (B, C, H_in, W_in)
                        loss = self._loss_fn(x_hat, x)  # Shape: ()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach().to(torch.float64) * x.shape[0]

                scheduler.step()
                avg_loss = float(running_loss.item()) / max(1, num_samples)
                epoch_losses.append(avg_loss)
                elapsed = time.perf_counter() - t0
                print(
                    f"Epoch [{epoch:03d}/{self.config.epochs:03d}] "
                    f"loss={avg_loss:.6f} ({elapsed:.2f}s)",
                    flush=True,
                )

                if self.config.loss_floor > 0.0 and epoch >= 20:
                    if avg_loss <= self.config.loss_floor:
                        consecutive_floor_hits += 1
                        if consecutive_floor_hits >= 2:
                            print(
                                f"[EarlyStop] Reached loss_floor={self.config.loss_floor:.6f} "
                                f"at epoch {epoch}",
                                flush=True,
                            )
                            break
                    else:
                        consecutive_floor_hits = 0

            self._calibrate_normal_baseline(
                train_images,
                extractor,
                train_gpu_u8,
                cached_features=cached_features,
            )
        finally:
            del train_gpu_u8, cached_features

        return epoch_losses

    def predict_batch(
        self,
        batch_images: np.ndarray,
        extractor: Any,
        target_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on one batch and return (image_scores, anomaly_maps, recons_uint8)."""
        self.net.eval()
        with torch.inference_mode():
            x = extractor.extract_batch(batch_images)  # Shape: (B, C_in, H_in, W_in)
            guide_rgb: torch.Tensor | None = None
            if self.config.guided_filter_radius > 0:
                if self.use_sigmoid and x.shape[-2:] == target_hw:
                    guide_rgb = x  # Shape: (B, C, H_t, W_t)
                else:
                    guide_rgb = _uint8_nhwc_to_unit_nchw(
                        batch_images, self.device
                    )  # Shape: (B, C_img, H, W)
            x_hat, err_map = self._score_batch_tensor(
                x, target_hw=target_hw, apply_calibration=True, guide_rgb=guide_rgb
            )  # Shape: (B, C_in, H_in, W_in), (B, 1, H_t, W_t)

            score_map = torch.nn.functional.avg_pool2d(
                err_map, kernel_size=3, stride=1, padding=1
            )  # Shape: (B, 1, H_t, W_t)
            flat = score_map.flatten(start_dim=1)  # Shape: (B, H_t * W_t)
            k = max(1, math.ceil(flat.shape[1] * self.config.score_top_k_pct))
            top_vals = torch.topk(flat, k=k, dim=1).values  # Shape: (B, k)
            k_core = max(1, math.ceil(k * 0.1))
            image_scores = (
                0.5 * top_vals[:, :k_core].mean(dim=1) + 0.5 * top_vals.mean(dim=1)
            )  # Shape: (B,)

            if hasattr(extractor, "to_uint8_images"):
                recons_uint8 = extractor.to_uint8_images(x_hat)
            else:
                vis = x_hat.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
                recons_uint8 = np.ascontiguousarray(vis.permute(0, 2, 3, 1).cpu().numpy())

            return (
                np.ascontiguousarray(image_scores.cpu().numpy(), dtype=np.float32),
                np.ascontiguousarray(err_map.squeeze(1).cpu().numpy(), dtype=np.float32),
                recons_uint8,
            )

    def save(self, path: Path) -> None:
        """Persist model weights and architecture metadata to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.net.state_dict(),
            "meta": self._checkpoint_meta(),
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Load model weights from disk after validating architecture compatibility."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        meta = payload.get("meta", {})
        expected = self._expected_checkpoint_meta()
        for key, val in expected.items():
            if key in meta and meta[key] != val:
                raise ValueError(
                    f"Checkpoint mismatch for {key}: expected {val}, found {meta[key]}"
                )
        self.net.load_state_dict(payload["state_dict"])
        self.net.to(device=self.device)
        if self.device.type == "cuda":
            self.net.to(memory_format=torch.channels_last)
        self.net.eval()


DCAE = ConvAutoencoder
DCAEModel = RawAutoencoderModel
