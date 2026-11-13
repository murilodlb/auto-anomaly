"""Feature extractor dispatcher."""

from __future__ import annotations

import torch

from src.config import PipelineConfig
from src.extractors.raw import RawPixelExtractor

EXTRACTOR_NAMES = {"raw"}


def get_extractor(config: PipelineConfig, device: torch.device) -> RawPixelExtractor:
    """Instantiate the configured spatial feature extractor."""
    if config.extractor_name == "raw":
        return RawPixelExtractor(config.dataset, config.extractors.raw, device)
    raise ValueError(
        f"Unknown extractor {config.extractor_name!r}; available: {sorted(EXTRACTOR_NAMES)}"
    )
