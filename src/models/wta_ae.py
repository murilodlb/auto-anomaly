"""Bottleneck Winner-Take-All Convolutional Autoencoder (Makhzani & Frey, NIPS 2015)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from src.config import AutoencoderConfig, WTAConfig
from src.models.dcae import RawAutoencoderModel


class WTASparsity2d(torch.nn.Module):
    """Apply non-overlapping spatial WTA + mini-batch lifetime sparsity (with eval EMA).

    Example shape flow for 1024x1024 input downsampled by 2^6=64 (H_b=W_b=16, Z=64, win=4):
      Input z:            (B, 64, 16, 16)
      max_pool2d(win=4):  (B, 64,  4,  4)   [16 spatial blocks per feature map]
      max_unpool2d:       (B, 64, 16, 16)   [at most 1 winner per 4x4 block = 6.25% density]
      slots (train):      (64, B * 4 * 4)   [rank across batch x spatial blocks per channel]
      batch_thresh:       (1, 64,  1,  1)   [k-th largest slot where k=ceil(N_slots*lifetime)]
      Output z_wta:       (B, 64, 16, 16)
    """

    running_threshold: torch.Tensor
    threshold_initialized: torch.Tensor

    def __init__(
        self,
        num_channels: int,
        spatial_window: int = 4,
        lifetime_sparsity: float = 0.2,
        ema_momentum: float = 0.05,
        eval_threshold_scale: float = 0.8,
    ) -> None:
        super().__init__()
        if num_channels < 1:
            raise ValueError(f"num_channels must be >= 1, got {num_channels}")
        if spatial_window < 1:
            raise ValueError(f"spatial_window must be >= 1, got {spatial_window}")
        if not (0.0 < lifetime_sparsity <= 1.0):
            raise ValueError(f"lifetime_sparsity must be in (0.0, 1.0], got {lifetime_sparsity}")
        if not (0.0 < ema_momentum <= 1.0):
            raise ValueError(f"ema_momentum must be in (0.0, 1.0], got {ema_momentum}")
        if eval_threshold_scale < 0.0:
            raise ValueError(f"eval_threshold_scale must be >= 0.0, got {eval_threshold_scale}")

        self.num_channels = num_channels
        self.spatial_window = spatial_window
        self.lifetime_sparsity = lifetime_sparsity
        self.ema_momentum = ema_momentum
        self.eval_threshold_scale = eval_threshold_scale

        self.register_buffer(
            "running_threshold", torch.zeros((1, num_channels, 1, 1), dtype=torch.float32)
        )  # Shape: (1, Z, 1, 1)
        self.register_buffer(
            "threshold_initialized", torch.zeros(1, dtype=torch.bool)
        )  # Shape: (1,)

        # Non-persistent accumulators (excluded from state_dict for checkpoint compatibility)
        self._train_density_sum: torch.Tensor = torch.zeros((), dtype=torch.float64)
        self._train_density_steps: int = 0
        self._eval_density_sum: torch.Tensor = torch.zeros((), dtype=torch.float64)
        self._eval_density_steps: int = 0

    def reset_density_stats(self, mode: str = "all") -> None:
        """Reset zero-sync bottleneck active density accumulators for 'train', 'eval', or 'all'."""
        if mode in {"train", "all"}:
            self._train_density_sum = torch.zeros((), dtype=torch.float64)
            self._train_density_steps = 0
        if mode in {"eval", "all"}:
            self._eval_density_sum = torch.zeros((), dtype=torch.float64)
            self._eval_density_steps = 0
        if mode not in {"train", "eval", "all"}:
            raise ValueError(
                f"Unknown density reset mode {mode!r}; expected 'train', 'eval', or 'all'"
            )

    def get_density_stats(self) -> dict[str, float]:
        """Return average active fraction during train/eval and learned EMA threshold stats."""
        train_d = (
            float(self._train_density_sum.item()) / float(self._train_density_steps)
            if self._train_density_steps > 0
            else float("nan")
        )
        eval_d = (
            float(self._eval_density_sum.item()) / float(self._eval_density_steps)
            if self._eval_density_steps > 0
            else float("nan")
        )
        return {
            "train_active_density": train_d,
            "eval_active_density": eval_d,
            "threshold_mean": float(self.running_threshold.mean().item()),
            "threshold_max": float(self.running_threshold.max().item()),
        }

    def _effective_window(self, h_b: int, w_b: int) -> int:
        g = math.gcd(h_b, w_b)
        for d in range(min(self.spatial_window, g), 0, -1):
            if g % d == 0:
                return d
        return 1

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # Shape: z is (B, Z, H_b, W_b), non-negative after ReLU
        b, z_ch, h_b, w_b = z.shape
        z_f32 = z.float()  # Shape: (B, Z, H_b, W_b) — float32 prevents bfloat16 unpool failures
        win = self._effective_window(h_b, w_b)

        if win > 1:
            pooled, indices = torch.nn.functional.max_pool2d(
                z_f32, kernel_size=(win, win), stride=(win, win), return_indices=True
            )  # Shape: pooled and indices are both (B, Z, H_b // win, W_b // win)
            z_spatial = torch.nn.functional.max_unpool2d(
                pooled,
                indices,
                kernel_size=(win, win),
                stride=(win, win),
                output_size=(b, z_ch, h_b, w_b),
            )  # Shape: (B, Z, H_b, W_b) with <= 1 active winner per (win x win) block
        else:
            pooled = z_f32  # Shape: (B, Z, H_b, W_b)
            z_spatial = z_f32  # Shape: (B, Z, H_b, W_b)

        if self.training and 0.0 < self.lifetime_sparsity < 1.0:
            slots = pooled.permute(1, 0, 2, 3).reshape(
                z_ch, -1
            )  # Shape: (Z, N_slots), N_slots = B * (H_b // win) * (W_b // win)
            n_slots = slots.shape[1]
            k = max(1, math.ceil(n_slots * self.lifetime_sparsity))
            topk_vals = torch.topk(slots, k=k, dim=1).values  # Shape: (Z, k)
            batch_thresh = topk_vals[:, -1].view(1, z_ch, 1, 1)  # Shape: (1, Z, 1, 1)

            with torch.no_grad():
                if not bool(self.threshold_initialized.item()):
                    # Shape: (1, Z, 1, 1)
                    self.running_threshold.copy_(batch_thresh.detach().float())
                    self.threshold_initialized.fill_(True)
                else:
                    self.running_threshold.lerp_(
                        batch_thresh.detach().float(), self.ema_momentum
                    )  # Shape: (1, Z, 1, 1)

            mask = (z_spatial >= batch_thresh) & (z_spatial > 0.0)  # Shape: (B, Z, H_b, W_b)
            z_wta = (z_spatial * mask.to(z_spatial.dtype)).to(z.dtype)  # Shape: (B, Z, H_b, W_b)
        elif (
            not self.training
            and bool(self.threshold_initialized.item())
            and 0.0 < self.lifetime_sparsity < 1.0
        ):
            eval_thresh = (
                self.running_threshold * self.eval_threshold_scale
            )  # Shape: (1, Z, 1, 1) — batch-independent eval threshold
            mask = (z_spatial >= eval_thresh) & (z_spatial > 0.0)  # Shape: (B, Z, H_b, W_b)
            z_wta = (z_spatial * mask.to(z_spatial.dtype)).to(z.dtype)  # Shape: (B, Z, H_b, W_b)
        else:
            z_wta = z_spatial.to(z.dtype)  # Shape: (B, Z, H_b, W_b)

        with torch.no_grad():
            density = (z_wta > 0.0).float().mean().to(torch.float64)  # Shape: ()
            if self.training:
                if self._train_density_sum.device != density.device:
                    self._train_density_sum = self._train_density_sum.to(device=density.device)
                self._train_density_sum = self._train_density_sum + density  # Shape: ()
                self._train_density_steps += 1
            else:
                if self._eval_density_sum.device != density.device:
                    self._eval_density_sum = self._eval_density_sum.to(device=density.device)
                self._eval_density_sum = self._eval_density_sum + density  # Shape: ()
                self._eval_density_steps += 1

        return z_wta  # Shape: (B, Z, H_b, W_b)


class WTAConvAutoencoder(torch.nn.Module):
    """Symmetric 2D ConvAE with Spatial + Lifetime Winner-Take-All bottleneck sparsity.

    Example shape flow for H=W=1024, in_channels=3, base_channels=64, num_stages=6, latent_dim=64:
      Encoder stages 0..5: (B, 3, 1024, 1024) -> ... -> (B, 512, 16, 16)
      Bottleneck pre-WTA:  (B, 512, 16, 16) -> (B, 64, 16, 16)  [3x3 + InstanceNorm2d + ReLU]
      WTA sparsity:        (B,  64, 16, 16) -> (B, 64, 16, 16)  [4x4 spatial WTA + lifetime EMA]
      Decoder proj:        (B,  64, 16, 16) -> (B, 512, 16, 16) [3x3 + BatchNorm2d + LeakyReLU]
      Decoder stages:      (B, 512, 16, 16) -> ... -> (B, 3, 1024, 1024)
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
        spatial_window: int = 4,
        lifetime_sparsity: float = 0.2,
        ema_momentum: float = 0.05,
        eval_threshold_scale: float = 0.8,
        use_sigmoid: bool = True,
        calib_spatial_size: int = 64,
    ) -> None:
        super().__init__()
        if num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {num_stages}")
        if calib_spatial_size < 1:
            raise ValueError(f"calib_spatial_size must be >= 1, got {calib_spatial_size}")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_dim = latent_dim
        self.num_stages = num_stages
        self.use_sigmoid = use_sigmoid
        self.calib_spatial_size = calib_spatial_size

        # Persistent calibration buffers matching ConvAutoencoder state_dict contract
        self.register_buffer("err_mean", torch.zeros(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("err_std", torch.ones(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("err_p99", torch.zeros(1, dtype=torch.float32))  # Shape: (1,)
        self.register_buffer("calibrated", torch.zeros(1, dtype=torch.bool))  # Shape: (1,)
        self.register_buffer(
            "normal_mu_map",
            torch.zeros((1, 1, calib_spatial_size, calib_spatial_size), dtype=torch.float32),
        )  # Shape: (1, 1, S, S)
        self.register_buffer(
            "normal_std_map",
            torch.ones((1, 1, calib_spatial_size, calib_spatial_size), dtype=torch.float32),
        )  # Shape: (1, 1, S, S)

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
        self.encoder_stages = torch.nn.Sequential(*enc_layers)

        self.bottleneck_proj = torch.nn.Sequential(
            # Shape: (B, stage_channels[-1], H_b, W_b) -> (B, latent_dim, H_b, W_b)
            torch.nn.Conv2d(ch_in, latent_dim, kernel_size=3, padding=1, bias=False),
            # Shape: (B, latent_dim, H_b, W_b) — per-image InstanceNorm2d strips stain bias
            torch.nn.InstanceNorm2d(latent_dim, affine=True),
            # Shape: (B, latent_dim, H_b, W_b) — strictly >= 0 for WTA winner selection
            torch.nn.ReLU(inplace=True),
        )

        # Shape: (B, latent_dim, H_b, W_b) -> (B, latent_dim, H_b, W_b) sparse winners
        self.wta = WTASparsity2d(
            num_channels=latent_dim,
            spatial_window=spatial_window,
            lifetime_sparsity=lifetime_sparsity,
            ema_momentum=ema_momentum,
            eval_threshold_scale=eval_threshold_scale,
        )

        self.decoder_proj = torch.nn.Sequential(
            # Shape: (B, latent_dim, H_b, W_b) -> (B, stage_channels[-1], H_b, W_b)
            torch.nn.Conv2d(latent_dim, stage_channels[-1], kernel_size=3, padding=1, bias=False),
            # Shape: (B, stage_channels[-1], H_b, W_b)
            torch.nn.BatchNorm2d(stage_channels[-1]),
            # Shape: (B, stage_channels[-1], H_b, W_b)
            torch.nn.LeakyReLU(0.2, inplace=True),
        )

        dec_layers: list[torch.nn.Module] = []
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
        self.decoder_stages = torch.nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shape: x is (B, C_in, H, W)
        h_enc = self.encoder_stages(x)  # Shape: (B, C_last, H_b, W_b), H_b = H // 2**num_stages
        z_dense = self.bottleneck_proj(h_enc)  # Shape: (B, Z, H_b, W_b), activations >= 0
        z_wta = self.wta(z_dense)  # Shape: (B, Z, H_b, W_b), sparse spatial + lifetime winners
        h_dec = self.decoder_proj(z_wta)  # Shape: (B, C_last, H_b, W_b) after 3x3 context
        x_hat = self.decoder_stages(h_dec)  # Shape: (B, C_in, H, W) reconstructed output
        return x_hat


class WTAAutoencoderModel(RawAutoencoderModel):
    """Training, inference, and anomaly scoring wrapper around WTAConvAutoencoder."""

    def __init__(
        self,
        in_channels: int,
        config: AutoencoderConfig,
        device: torch.device,
        wta_config: WTAConfig | None = None,
        batch_size: int = 16,
        seed: int = 42,
        use_sigmoid: bool = True,
    ) -> None:
        self.wta_config = wta_config if wta_config is not None else WTAConfig()
        super().__init__(
            in_channels=in_channels,
            config=config,
            device=device,
            batch_size=batch_size,
            seed=seed,
            use_sigmoid=use_sigmoid,
        )

    def _build_network(self, in_channels: int, use_sigmoid: bool) -> Any:
        return WTAConvAutoencoder(
            in_channels=in_channels,
            base_channels=self.config.base_channels,
            latent_dim=self.config.latent_dim,
            num_stages=self.config.num_stages,
            spatial_window=self.wta_config.spatial_window,
            lifetime_sparsity=self.wta_config.lifetime_sparsity,
            ema_momentum=self.wta_config.ema_momentum,
            eval_threshold_scale=self.wta_config.eval_threshold_scale,
            use_sigmoid=use_sigmoid,
            calib_spatial_size=self.config.calib_spatial_size,
        )

    def _checkpoint_meta(self) -> dict[str, Any]:
        return {
            "model_type": "wta_ae",
            "in_channels": self.in_channels,
            "base_channels": self.config.base_channels,
            "latent_dim": self.config.latent_dim,
            "num_stages": self.config.num_stages,
            "use_sigmoid": self.net.use_sigmoid,
            "calib_spatial_size": int(
                getattr(self.net, "calib_spatial_size", self.config.calib_spatial_size)
            ),
            "l1_channel_max_weight": self.config.l1_channel_max_weight,
            "calib_spatial_weight": self.config.calib_spatial_weight,
            "calib_z_shift": self.config.calib_z_shift,
            "wta_spatial_window": self.wta_config.spatial_window,
            "wta_lifetime_sparsity": self.wta_config.lifetime_sparsity,
            "wta_eval_threshold_scale": self.wta_config.eval_threshold_scale,
        }

    def _on_epoch_start(self, epoch: int) -> None:
        """Reset train density accumulators each epoch so final-epoch density is reported."""
        _ = epoch
        self.net.wta.reset_density_stats(mode="train")

    def fit(self, train_images: np.ndarray, extractor: Any) -> list[float]:
        """Train WTA autoencoder and log bottleneck active density diagnostics."""
        self.net.wta.reset_density_stats(mode="all")
        epoch_losses = super().fit(train_images, extractor)
        stats = self.net.wta.get_density_stats()
        print(
            f"[WTA Density] train_active={stats['train_active_density']:.4f} | "
            f"eval_calib_active={stats['eval_active_density']:.4f} | "
            f"thresh_mean={stats['threshold_mean']:.4f}",
            flush=True,
        )
        return epoch_losses

    def _expected_checkpoint_meta(self) -> dict[str, Any]:
        return {
            "model_type": "wta_ae",
            "in_channels": self.in_channels,
            "base_channels": self.config.base_channels,
            "latent_dim": self.config.latent_dim,
            "num_stages": self.config.num_stages,
            "wta_spatial_window": self.wta_config.spatial_window,
            "wta_lifetime_sparsity": self.wta_config.lifetime_sparsity,
            "wta_eval_threshold_scale": self.wta_config.eval_threshold_scale,
        }
