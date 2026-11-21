"""Raw pixel spatial tensor extractor producing (B, C, H, W) batches."""

from __future__ import annotations

import numpy as np
import torch

from src.config import DatasetConfig, RawExtractorConfig


class RawPixelExtractor:
    """Convert uint8 (B, H, W, C) image batches into contiguous (B, C, H, W) float32 tensors."""

    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        extractor_cfg: RawExtractorConfig,
        device: torch.device,
    ) -> None:
        self.dataset_cfg = dataset_cfg
        self.extractor_cfg = extractor_cfg
        self.device = device

        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None
        if extractor_cfg.normalize_mean is not None and extractor_cfg.normalize_std is not None:
            self._mean = torch.tensor(
                extractor_cfg.normalize_mean,
                dtype=torch.float32,
                device=device,
            ).view(1, -1, 1, 1)
            self._std = torch.tensor(
                extractor_cfg.normalize_std,
                dtype=torch.float32,
                device=device,
            ).view(1, -1, 1, 1)

    @property
    def out_channels(self) -> int:
        return self.dataset_cfg.channels

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.dataset_cfg.image_size

    @property
    def is_unit_range(self) -> bool:
        """Whether extracted tensors lie in [0.0, 1.0]."""
        return self._mean is None and self._std is None

    @property
    def cache_features_for_training(self) -> bool:
        """Raw pixels are inexpensive to normalize on the fly."""
        return False

    def normalize_uint8_tensor(self, tensor_nchw_u8: torch.Tensor) -> torch.Tensor:
        """Normalize an on-device (B, C, H, W) uint8 tensor into float32 (channels_last on CUDA)."""
        mem_fmt = torch.channels_last if self.device.type == "cuda" else torch.contiguous_format
        tensor = tensor_nchw_u8.to(
            device=self.device,
            dtype=torch.float32,
            memory_format=mem_fmt,
        )
        tensor.div_(255.0)

        if self._mean is not None and self._std is not None:
            tensor = (tensor - self._mean) / self._std

        return tensor

    def extract_batch(self, batch_images: np.ndarray) -> torch.Tensor:
        """Convert (B, H, W, C) uint8 numpy batch to (B, C, H, W) float32 tensor on device."""
        if batch_images.ndim != 4 or batch_images.shape[-1] != self.out_channels:
            raise ValueError(
                f"Expected (B, H, W, {self.out_channels}) batch, got shape {batch_images.shape}"
            )

        contiguous = np.ascontiguousarray(batch_images)
        tensor_u8 = torch.from_numpy(contiguous).to(device=self.device, non_blocking=True)
        return self.normalize_uint8_tensor(tensor_u8.permute(0, 3, 1, 2))

    def to_uint8_images(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a (B, C, H, W) float tensor back to (B, H, W, C) uint8 numpy images."""
        vis = tensor.detach()
        if self._mean is not None and self._std is not None:
            vis = vis * self._std + self._mean
        vis = vis.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8)
        return np.ascontiguousarray(vis.permute(0, 2, 3, 1).cpu().numpy())
