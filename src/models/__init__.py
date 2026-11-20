"""Anomaly detection model dispatcher."""

from __future__ import annotations

import torch

from src.config import PipelineConfig
from src.models.dcae import RawAutoencoderModel
from src.models.wta_ae import WTAAutoencoderModel

MODEL_NAMES = {"dcae", "wta_ae", "autoencoder"}


def get_model(
    config: PipelineConfig,
    in_channels: int,
    device: torch.device,
    use_sigmoid: bool = True,
) -> RawAutoencoderModel:
    """Instantiate the configured anomaly detection model."""
    if config.model_name in {"dcae", "autoencoder"}:
        return RawAutoencoderModel(
            in_channels=in_channels,
            config=config.autoencoder,
            device=device,
            batch_size=config.dataset.batch_size,
            seed=config.seed,
            use_sigmoid=use_sigmoid,
        )
    if config.model_name == "wta_ae":
        return WTAAutoencoderModel(
            in_channels=in_channels,
            config=config.autoencoder,
            wta_config=config.wta,
            device=device,
            batch_size=config.dataset.batch_size,
            seed=config.seed,
            use_sigmoid=use_sigmoid,
        )
    raise ValueError(f"Unknown model {config.model_name!r}; available: {sorted(MODEL_NAMES)}")
