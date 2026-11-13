"""Single source of truth for all pipeline configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_LOSS_TYPES = frozenset({"l1", "l2", "smooth_l1"})
VALID_EXTRACTORS = frozenset({"raw"})
VALID_MODELS = frozenset({"autoencoder"})


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for MVTec AD dataset loading and batching."""

    root_dir: Path = Path("data")
    category: str = "carpet"
    image_size: tuple[int, int] = (256, 256)
    grayscale: bool = False
    batch_size: int = 16
    num_workers: int = 4

    @property
    def category_dir(self) -> Path:
        return self.root_dir / self.category

    @property
    def train_dir(self) -> Path:
        return self.category_dir / "train"

    @property
    def test_dir(self) -> Path:
        return self.category_dir / "test"

    @property
    def ground_truth_dir(self) -> Path:
        return self.category_dir / "ground_truth"

    @property
    def channels(self) -> int:
        return 1 if self.grayscale else 3


@dataclass(frozen=True)
class RawExtractorConfig:
    """Configuration for the raw normalized pixel extractor."""

    normalize_mean: tuple[float, ...] | None = None
    normalize_std: tuple[float, ...] | None = None


@dataclass(frozen=True)
class ResNetExtractorConfig:
    """Stub configuration for future ResNet intermediate feature extractor."""

    backbone: str = "resnet18"
    layers: tuple[str, ...] = ("layer2", "layer3")


@dataclass(frozen=True)
class DINOv3ExtractorConfig:
    """Stub configuration for future DINOv3 patch-token spatial extractor."""

    model_name: str = "dinov3_vitb16"
    layer: int = -1


@dataclass(frozen=True)
class ExtractorsConfig:
    """Container holding per-extractor configurations."""

    raw: RawExtractorConfig = field(default_factory=RawExtractorConfig)
    resnet: ResNetExtractorConfig = field(default_factory=ResNetExtractorConfig)
    dinov3: DINOv3ExtractorConfig = field(default_factory=DINOv3ExtractorConfig)


@dataclass(frozen=True)
class AutoencoderConfig:
    """Configuration for the 2D convolutional reconstruction autoencoder."""

    latent_dim: int = 128
    base_channels: int = 32
    num_stages: int = 4
    loss_type: str = "l2"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    score_top_k_pct: float = 0.01
    gaussian_sigma: float = 4.0
    save_weights: bool = True
    load_weights: Path | None = None


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for experiment artifacts, checkpoints, and plots."""

    base_dir: Path = Path("results")
    experiment_name: str = "carpet_raw_ae"
    save_visualizations: bool = True
    max_vis_samples: int = 25

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.experiment_name

    @property
    def heatmaps_dir(self) -> Path:
        return self.output_dir / "heatmaps"

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating dataset, extractor, model, and output."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extractor_name: str = "raw"
    model_name: str = "autoencoder"
    extractors: ExtractorsConfig = field(default_factory=ExtractorsConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    seed: int = 42
    device: str = "cuda"

    def validate(self) -> None:
        """Validate directory structure on disk and hyperparameter invariants."""
        if not self.dataset.category_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset category directory does not exist: {self.dataset.category_dir}"
            )
        train_good = self.dataset.train_dir / "good"
        if not train_good.is_dir():
            raise FileNotFoundError(f"Training 'good' directory does not exist: {train_good}")
        if not self.dataset.test_dir.is_dir():
            raise FileNotFoundError(f"Test directory does not exist: {self.dataset.test_dir}")

        if self.extractor_name not in VALID_EXTRACTORS:
            raise ValueError(
                f"Unsupported extractor {self.extractor_name!r}; expected one of {VALID_EXTRACTORS}"
            )
        if self.model_name not in VALID_MODELS:
            raise ValueError(
                f"Unsupported model {self.model_name!r}; expected one of {VALID_MODELS}"
            )

        if len(self.dataset.image_size) != 2:
            raise ValueError(f"image_size must have 2 elements, got {self.dataset.image_size}")
        h, w = self.dataset.image_size
        if h <= 0 or w <= 0:
            raise ValueError(f"image_size dimensions must be positive, got {(h, w)}")
        if self.dataset.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.dataset.batch_size}")
        if self.dataset.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.dataset.num_workers}")

        ae = self.autoencoder
        if ae.num_stages < 1:
            raise ValueError(f"num_stages must be >= 1, got {ae.num_stages}")
        if ae.latent_dim < 1 or ae.base_channels < 1:
            raise ValueError("latent_dim and base_channels must be >= 1")
        stride_factor = 2**ae.num_stages
        if h % stride_factor != 0 or w % stride_factor != 0:
            raise ValueError(
                f"image_size {(h, w)} must be divisible by 2**num_stages ({stride_factor})"
            )
        if ae.loss_type not in VALID_LOSS_TYPES:
            raise ValueError(
                f"Unsupported loss_type {ae.loss_type!r}; expected one of {VALID_LOSS_TYPES}"
            )
        if ae.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {ae.epochs}")
        if ae.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {ae.lr}")
        if ae.weight_decay < 0.0:
            raise ValueError(f"weight_decay must be >= 0, got {ae.weight_decay}")
        if not (0.0 < ae.score_top_k_pct <= 1.0):
            raise ValueError(
                f"score_top_k_pct must be in (0.0, 1.0], got {ae.score_top_k_pct}"
            )
        if ae.gaussian_sigma < 0.0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {ae.gaussian_sigma}")
        if ae.load_weights is not None and not ae.load_weights.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ae.load_weights}")

        raw_cfg = self.extractors.raw
        if (raw_cfg.normalize_mean is None) != (raw_cfg.normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must both be set or both be None")
        if raw_cfg.normalize_mean is not None and raw_cfg.normalize_std is not None:
            expected_ch = self.dataset.channels
            if (
                len(raw_cfg.normalize_mean) != expected_ch
                or len(raw_cfg.normalize_std) != expected_ch
            ):
                raise ValueError(
                    f"normalize_mean and normalize_std must have length {expected_ch}"
                )
            if any(s <= 0.0 for s in raw_cfg.normalize_std):
                raise ValueError("All normalize_std entries must be strictly positive")

    @classmethod
    def from_args(cls, **kwargs: Any) -> PipelineConfig:
        """Construct a PipelineConfig cleanly from flat CLI keyword arguments."""
        image_size_raw = kwargs.get("image_size", (256, 256))
        image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

        dataset_cfg = DatasetConfig(
            root_dir=Path(kwargs.get("data_dir", "data")),
            category=str(kwargs.get("category", "carpet")),
            image_size=image_size,
            grayscale=bool(kwargs.get("grayscale", False)),
            batch_size=int(kwargs.get("batch_size", 16)),
            num_workers=int(kwargs.get("num_workers", 4)),
        )

        load_weights_raw = kwargs.get("load_weights")
        load_weights = Path(load_weights_raw) if load_weights_raw is not None else None

        ae_cfg = AutoencoderConfig(
            latent_dim=int(kwargs.get("latent_dim", 128)),
            base_channels=int(kwargs.get("base_channels", 32)),
            num_stages=int(kwargs.get("num_stages", 4)),
            loss_type=str(kwargs.get("loss", "l2")),
            epochs=int(kwargs.get("epochs", 50)),
            lr=float(kwargs.get("lr", 1e-3)),
            weight_decay=float(kwargs.get("weight_decay", 1e-5)),
            score_top_k_pct=float(kwargs.get("top_k_pct", 0.01)),
            gaussian_sigma=float(kwargs.get("gaussian_sigma", 4.0)),
            save_weights=not bool(kwargs.get("no_save_weights", False)),
            load_weights=load_weights,
        )

        output_cfg = OutputConfig(
            base_dir=Path(kwargs.get("output_dir", "results")),
            experiment_name=str(kwargs.get("output", "carpet_raw_ae")),
            save_visualizations=not bool(kwargs.get("no_vis", False)),
            max_vis_samples=int(kwargs.get("max_vis_samples", 25)),
        )

        return cls(
            dataset=dataset_cfg,
            extractor_name=str(kwargs.get("extractor", "raw")),
            model_name=str(kwargs.get("model", "autoencoder")),
            autoencoder=ae_cfg,
            output=output_cfg,
            seed=int(kwargs.get("seed", 42)),
            device=str(kwargs.get("device", "cuda")),
        )
