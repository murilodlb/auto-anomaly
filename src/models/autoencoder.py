"""2D Convolutional Autoencoder model, training loop, and anomaly map scorer."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import AutoencoderConfig
from src.dataset import iter_batches

_LOSS_FUNCTIONS = {
    "l1": F.l1_loss,
    "l2": F.mse_loss,
    "smooth_l1": F.smooth_l1_loss,
}


class ConvAutoencoder(nn.Module):
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

        enc_layers: list[nn.Module] = []
        ch_in = in_channels
        for ch_out in stage_channels:
            enc_layers.extend(
                [
                    nn.Conv2d(ch_in, ch_out, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(ch_out),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
            ch_in = ch_out
        enc_layers.extend(
            [
                nn.Conv2d(ch_in, latent_dim, kernel_size=1, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        )
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = [
            nn.Conv2d(latent_dim, stage_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(stage_channels[-1]),
            nn.ReLU(inplace=True),
        ]
        for idx in range(num_stages - 1, 0, -1):
            c_in = stage_channels[idx]
            c_out = stage_channels[idx - 1]
            dec_layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(c_out),
                    nn.ReLU(inplace=True),
                ]
            )
        dec_layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(stage_channels[0], in_channels, kernel_size=3, padding=1, bias=True),
                nn.Sigmoid() if use_sigmoid else nn.Identity(),
            ]
        )
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable 2D Gaussian smoothing to a (B, 1, H, W) tensor."""
    if sigma <= 0.0:
        return tensor

    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=tensor.dtype, device=tensor.device)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    kernel_x = kernel_1d.view(1, 1, 1, -1)
    kernel_y = kernel_1d.view(1, 1, -1, 1)

    min_dim = min(tensor.shape[-2], tensor.shape[-1])
    pad_mode = "reflect" if radius < min_dim else "replicate"
    padded = F.pad(tensor, (radius, radius, radius, radius), mode=pad_mode)
    blurred = F.conv2d(padded, kernel_x)
    return F.conv2d(blurred, kernel_y)


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
        num_samples = len(train_images)
        epoch_losses: list[float] = []

        self.net.train()
        for epoch in range(1, self.config.epochs + 1):
            t0 = time.perf_counter()
            running_loss = 0.0

            for batch_np in iter_batches(
                train_images,
                batch_size=self.batch_size,
                shuffle=True,
                seed=self.seed + epoch,
            ):
                x = extractor.extract_batch(batch_np)
                optimizer.zero_grad(set_to_none=True)
                x_hat = self.net(x)
                loss = self._loss_fn(x_hat, x)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(batch_np)

            avg_loss = running_loss / max(1, num_samples)
            epoch_losses.append(avg_loss)
            elapsed = time.perf_counter() - t0
            print(
                f"Epoch [{epoch:03d}/{self.config.epochs:03d}] "
                f"loss={avg_loss:.6f} ({elapsed:.2f}s)"
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

            err_map = ((x - x_hat) ** 2).mean(dim=1, keepdim=True)
            if err_map.shape[-2:] != target_hw:
                err_map = F.interpolate(
                    err_map,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )

            err_map = _gaussian_blur_2d(err_map, self.config.gaussian_sigma)

            flat = err_map.flatten(start_dim=1)
            k = max(1, int(math.ceil(flat.shape[1] * self.config.score_top_k_pct)))
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
