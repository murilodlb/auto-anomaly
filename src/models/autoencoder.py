"""2D Convolutional Autoencoder model, training loop, and anomaly map scorer."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import AutoencoderConfig
from src.dataset import iter_batches


def _l1_ssim_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Combined L1 + SSIM loss (0.2 * L1 + 0.8 * (1 - SSIM)) for sharp textures."""
    l1 = torch.nn.functional.l1_loss(x_hat, x)
    ssim_err = 1.0 - _ssim_map(x_hat, x).mean()
    return 0.2 * l1 + 0.8 * ssim_err


_LOSS_FUNCTIONS = {
    "l1": torch.nn.functional.l1_loss,
    "l2": torch.nn.functional.mse_loss,
    "smooth_l1": torch.nn.functional.smooth_l1_loss,
    "l1_ssim": _l1_ssim_loss,
}


def _ssim_map(x: torch.Tensor, y: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """Compute per-pixel (B, C, H, W) Structural Similarity (SSIM) map in [0, 1]."""
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


class ConvAutoencoder(torch.nn.Module):
    """Symmetric fully-convolutional 2D autoencoder operating on (B, C, H, W) tensors."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        latent_dim: int = 128,
        num_stages: int = 4,
        use_sigmoid: bool = True,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.num_stages = num_stages
        self.use_sigmoid = use_sigmoid

        stage_channels = [min(base_channels * (2**i), 512) for i in range(num_stages)]

        enc_layers: list[torch.nn.Module] = []
        ch_in = in_channels
        for ch_out in stage_channels:
            enc_layers.extend(
                [
                    torch.nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                    torch.nn.BatchNorm2d(ch_out),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch_in = ch_out
        enc_layers.extend(
            [
                torch.nn.Conv2d(ch_in, latent_dim, kernel_size=1, bias=True),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.encoder = torch.nn.Sequential(*enc_layers)

        dec_layers: list[torch.nn.Module] = [
            torch.nn.Conv2d(latent_dim, stage_channels[-1], kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(stage_channels[-1]),
            torch.nn.ReLU(inplace=True),
        ]
        for idx in range(num_stages - 1, 0, -1):
            c_in = stage_channels[idx]
            c_out = stage_channels[idx - 1]
            dec_layers.extend(
                [
                    torch.nn.ConvTranspose2d(
                        c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False
                    ),
                    torch.nn.BatchNorm2d(c_out),
                    torch.nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        dec_layers.extend(
            [
                torch.nn.ConvTranspose2d(
                    stage_channels[0], in_channels, kernel_size=4, stride=2, padding=1, bias=True
                ),
                torch.nn.Sigmoid() if use_sigmoid else torch.nn.Identity(),
            ]
        )
        self.decoder = torch.nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


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

        self.net = ConvAutoencoder(
            in_channels=in_channels,
            base_channels=config.base_channels,
            latent_dim=config.latent_dim,
            num_stages=config.num_stages,
            use_sigmoid=use_sigmoid,
        ).to(device)

        if config.loss_type not in _LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss_type: {config.loss_type!r}")
        self._loss_fn = _LOSS_FUNCTIONS[config.loss_type]

    def fit(self, train_images: np.ndarray, extractor: Any) -> list[float]:
        """Train the autoencoder on normal images and return per-epoch average losses."""
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        _ = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
            eta_min=self.config.lr * 0.05,
        )
        num_samples = len(train_images)
        epoch_losses: list[float] = []
        use_amp = (
            self.config.use_amp and self.device.type == "cuda" and torch.cuda.is_bf16_supported()
        )

        self.net.train()
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()
            running_loss = torch.zeros((), dtype=torch.float64, device=self.device)

            for batch_np in iter_batches(
                train_images,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed + epoch,
            ):
                x = extractor.extract_batch(batch_np)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    x_hat = self.net(x)
                    loss = self._loss_fn(x_hat, x)
                loss.backward()
                optimizer.step()
                running_loss += loss.detach().to(torch.float64) * len(batch_np)

            avg_loss = float(running_loss.item()) / max(1, num_samples)
            epoch_losses.append(avg_loss)
            elapsed = time.perf_counter() - t0
            print(
                f"Epoch [{epoch:03d}/{self.config.epochs:03d}] "
                f"loss={avg_loss:.6f} ({elapsed:.2f}s)",
                flush=True,
            )

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
            x = extractor.extract_batch(batch_images)
            x_hat = self.net(x)

            if self.config.loss_type == "l1_ssim":
                l1_map = (x - x_hat).abs().mean(dim=1, keepdim=True)
                ssim_err_map = (1.0 - _ssim_map(x_hat, x)).mean(dim=1, keepdim=True)
                err_map = 0.2 * l1_map + 0.8 * ssim_err_map
            elif self.config.loss_type == "l1":
                err_map = (x - x_hat).abs().mean(dim=1, keepdim=True)
            else:
                err_map = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
            if err_map.shape[-2:] != target_hw:
                err_map = torch.nn.functional.interpolate(
                    err_map,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            err_map = _gaussian_blur_2d(err_map, self.config.gaussian_sigma)

            flat = err_map.flatten(start_dim=1)
            k = max(1, math.ceil(flat.shape[1] * self.config.score_top_k_pct))
            top_vals = torch.topk(flat, k=k, dim=1).values
            image_scores = top_vals.mean(dim=1)

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
            "meta": {
                "in_channels": self.in_channels,
                "base_channels": self.config.base_channels,
                "latent_dim": self.config.latent_dim,
                "num_stages": self.config.num_stages,
                "use_sigmoid": self.net.use_sigmoid,
            },
        }
        torch.save(payload, path)

    def load(self, path: Path) -> None:
        """Load model weights from disk after validating architecture compatibility."""
        payload = torch.load(path, map_location=self.device, weights_only=True)
        meta = payload.get("meta", {})
        expected = {
            "in_channels": self.in_channels,
            "base_channels": self.config.base_channels,
            "latent_dim": self.config.latent_dim,
            "num_stages": self.config.num_stages,
        }
        for key, val in expected.items():
            if key in meta and meta[key] != val:
                raise ValueError(
                    f"Checkpoint mismatch for {key}: expected {val}, found {meta[key]}"
                )
        self.net.load_state_dict(payload["state_dict"])
        self.net.to(self.device)
        self.net.eval()
