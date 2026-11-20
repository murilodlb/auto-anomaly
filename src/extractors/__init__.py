"""Feature extractor dispatcher."""

from __future__ import annotations

import torch

from src.config import PipelineConfig
from src.extractors.dinov3 import DINOv3Extractor
from src.extractors.raw import RawPixelExtractor

EXTRACTOR_NAMES = {"raw", "dinov3"}


def get_extractor(
    config: PipelineConfig,
    device: torch.device,
) -> RawPixelExtractor | DINOv3Extractor:
    """Instantiate the configured spatial feature extractor."""
    if config.extractor_name == "raw":
        return RawPixelExtractor(config.dataset, config.extractors.raw, device)
    if config.extractor_name == "dinov3":
        return DINOv3Extractor(
            config.dataset,
            config.extractors.dinov3,
            device,
            stride_factor=2**config.autoencoder.num_stages,
        )
    raise ValueError(
        f"Unknown extractor {config.extractor_name!r}; available: {sorted(EXTRACTOR_NAMES)}"
    )
