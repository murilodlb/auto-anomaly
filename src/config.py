"""Single source of truth for all pipeline configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

VALID_LOSS_TYPES = frozenset({"l1", "l2", "smooth_l1", "l1_ssim", "ms_l1_ssim", "cosine"})
VALID_EXTRACTORS = frozenset({"raw", "dinov3"})
VALID_MODELS = frozenset({"dcae", "wta_ae", "autoencoder"})
MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


def detect_aligned_native_size(
    category_dir: Path,
    stride_factor: int,
) -> tuple[int, int]:
    """Detect native (H, W) from the first training image and align to stride_factor."""
    train_good = category_dir / "train" / "good"
    first_img = next(iter(sorted(train_good.glob("*.png"))), None)
    if first_img is None:
        raise FileNotFoundError(f"No PNG images found in {train_good} to infer native resolution")
    img = cv2.imread(str(first_img), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image for native size detection: {first_img}")
    raw_h, raw_w = int(img.shape[0]), int(img.shape[1])
    aligned_h = max(stride_factor, (raw_h // stride_factor) * stride_factor)
    aligned_w = max(stride_factor, (raw_w // stride_factor) * stride_factor)
    return aligned_h, aligned_w


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for MVTec AD dataset loading and batching."""

    root_dir: Path = Path("data")
    category: str = "carpet"
    image_size: tuple[int, int] = (256, 256)
    grayscale: bool = False
    batch_size: int = 16
    num_workers: int = 4
    full_res: bool = False

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
    """Configuration for the DINOv3 patch-token spatial feature extractor."""

    model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    patch_size: int = 16
    stride: int = 16
    layers: tuple[int, ...] | None = None
    prune_registers: bool = False
    mean_centering_alpha: float = 0.0
    target_short_edge: int | None = None
    plain_last_hidden_state: bool = False


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
    score_top_k_pct: float = 0.0005
    gaussian_sigma: float = 4.0
    spatial_dropout: float = 0.1
    denoise_prob: float = 0.5
    texture_damping: float = 2.0
    loss_floor: float = 0.006
    guided_filter_radius: int = 8
    guided_filter_eps: float = 0.01
    use_amp: bool = True
    save_weights: bool = True
    load_weights: Path | None = None


@dataclass(frozen=True)
class WTAConfig:
    """Configuration for the Bottleneck Winner-Take-All (WTA) autoencoder."""

    spatial_window: int = 4
    lifetime_sparsity: float = 0.2
    ema_momentum: float = 0.05
    eval_threshold_scale: float = 0.8


@dataclass(frozen=True)
class OutputConfig:
    """Configuration for experiment artifacts, checkpoints, and plots."""

    base_dir: Path = Path("results")
    experiment_name: str = "carpet_raw_ae"
    save_visualizations: bool = True
    max_vis_samples: int = 25
    save_tiff_maps: bool = False

    @property
    def output_dir(self) -> Path:
        return self.base_dir / self.experiment_name

    @property
    def heatmaps_dir(self) -> Path:
        return self.output_dir / "heatmaps"

    @property
    def weights_dir(self) -> Path:
        return self.output_dir / "weights"

    @property
    def anomaly_maps_dir(self) -> Path:
        return self.output_dir / "anomaly_maps"


def discover_mvtec_categories(root_dir: Path) -> list[str]:
    """Return sorted valid MVTec AD category names present under root_dir."""
    if not root_dir.is_dir():
        return []
    categories: list[str] = []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        if (entry / "train" / "good").is_dir() and (entry / "test").is_dir():
            categories.append(entry.name)
    return categories


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration aggregating dataset, extractor, model, and output."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    extractor_name: str = "raw"
    model_name: str = "dcae"
    extractors: ExtractorsConfig = field(default_factory=ExtractorsConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    wta: WTAConfig = field(default_factory=WTAConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pro_integration_limit: float = 0.3
    seed: int = 42
    device: str = "cuda"

    def validate(self) -> None:
        """Validate directory structure on disk and hyperparameter invariants."""
        if self.dataset.category == "all":
            if not self.dataset.root_dir.is_dir():
                raise FileNotFoundError(
                    f"Dataset root directory does not exist: {self.dataset.root_dir}"
                )
            if not discover_mvtec_categories(self.dataset.root_dir):
                raise FileNotFoundError(
                    f"No valid MVTec AD category directories found under {self.dataset.root_dir}"
                )
            if self.autoencoder.load_weights is not None:
                raise ValueError("--load-weights cannot be used when --category is 'all'")
        else:
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
        if self.extractor_name == "raw":
            if h % stride_factor != 0 or w % stride_factor != 0:
                raise ValueError(
                    f"image_size {(h, w)} must be divisible by 2**num_stages ({stride_factor})"
                )
            if (h // stride_factor) * (w // stride_factor) < 2:
                raise ValueError(
                    f"image_size {(h, w)} with num_stages={ae.num_stages} yields a 1x1 bottleneck; "
                    "InstanceNorm2d requires bottleneck spatial area >= 2"
                )
        elif self.extractor_name == "dinov3":
            if self.dataset.grayscale:
                raise ValueError("DINOv3 extractor requires 3-channel RGB images (grayscale=False)")
            dv3 = self.extractors.dinov3
            if dv3.patch_size < 1 or dv3.stride < 1:
                raise ValueError("dinov3 patch_size and stride must be >= 1")
            if dv3.mean_centering_alpha < 0.0:
                raise ValueError("dinov3 mean_centering_alpha must be >= 0.0")
            if dv3.target_short_edge is not None and dv3.target_short_edge < dv3.patch_size:
                raise ValueError(
                    f"dinov3 target_short_edge ({dv3.target_short_edge}) "
                    f"must be >= patch_size ({dv3.patch_size})"
                )
            if dv3.layers is not None and any(layer < 1 for layer in dv3.layers):
                raise ValueError("All dinov3 layer indices must be >= 1")
            if dv3.target_short_edge is not None:
                scale = float(dv3.target_short_edge) / float(min(h, w))
                scaled_h, scaled_w = round(h * scale), round(w * scale)
            else:
                scaled_h, scaled_w = h, w
            n_h = max(1, (scaled_h - dv3.patch_size) // dv3.stride + 1)
            n_w = max(1, (scaled_w - dv3.patch_size) // dv3.stride + 1)
            if (n_h // stride_factor) * (n_w // stride_factor) < 2:
                raise ValueError(
                    f"DINOv3 patch grid {(n_h, n_w)} with num_stages={ae.num_stages} "
                    f"(stride_factor={stride_factor}) is too small for bottleneck spatial area >= 2"
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
            raise ValueError(f"score_top_k_pct must be in (0.0, 1.0], got {ae.score_top_k_pct}")
        if ae.gaussian_sigma < 0.0:
            raise ValueError(f"gaussian_sigma must be >= 0, got {ae.gaussian_sigma}")
        if not (0.0 <= ae.spatial_dropout < 1.0):
            raise ValueError(f"spatial_dropout must be in [0.0, 1.0), got {ae.spatial_dropout}")
        if not (0.0 <= ae.denoise_prob <= 1.0):
            raise ValueError(f"denoise_prob must be in [0.0, 1.0], got {ae.denoise_prob}")
        if ae.texture_damping < 0.0:
            raise ValueError(f"texture_damping must be >= 0, got {ae.texture_damping}")
        if ae.loss_floor < 0.0:
            raise ValueError(f"loss_floor must be >= 0, got {ae.loss_floor}")
        if ae.guided_filter_radius < 0:
            raise ValueError(
                f"guided_filter_radius must be >= 0, got {ae.guided_filter_radius}"
            )
        if ae.guided_filter_eps <= 0.0:
            raise ValueError(f"guided_filter_eps must be > 0.0, got {ae.guided_filter_eps}")
        if ae.load_weights is not None and not ae.load_weights.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ae.load_weights}")

        wta = self.wta
        if wta.spatial_window < 1:
            raise ValueError(f"wta.spatial_window must be >= 1, got {wta.spatial_window}")
        if not (0.0 < wta.lifetime_sparsity <= 1.0):
            raise ValueError(
                f"wta.lifetime_sparsity must be in (0.0, 1.0], got {wta.lifetime_sparsity}"
            )
        if not (0.0 < wta.ema_momentum <= 1.0):
            raise ValueError(f"wta.ema_momentum must be in (0.0, 1.0], got {wta.ema_momentum}")
        if wta.eval_threshold_scale < 0.0:
            raise ValueError(
                f"wta.eval_threshold_scale must be >= 0.0, got {wta.eval_threshold_scale}"
            )

        if not (0.0 < self.pro_integration_limit <= 1.0):
            raise ValueError(
                f"pro_integration_limit must be in (0.0, 1.0], got {self.pro_integration_limit}"
            )

        raw_cfg = self.extractors.raw
        if (raw_cfg.normalize_mean is None) != (raw_cfg.normalize_std is None):
            raise ValueError("normalize_mean and normalize_std must both be set or both be None")
        if raw_cfg.normalize_mean is not None and raw_cfg.normalize_std is not None:
            expected_ch = self.dataset.channels
            if (
                len(raw_cfg.normalize_mean) != expected_ch
                or len(raw_cfg.normalize_std) != expected_ch
            ):
                raise ValueError(f"normalize_mean and normalize_std must have length {expected_ch}")
            if any(s <= 0.0 for s in raw_cfg.normalize_std):
                raise ValueError("All normalize_std entries must be strictly positive")

    @classmethod
    def from_args(cls, **kwargs: Any) -> PipelineConfig:
        """Construct a PipelineConfig cleanly from flat CLI keyword arguments."""
        root_dir = Path(kwargs.get("data_dir", "data"))
        category = str(kwargs.get("category", "carpet"))
        num_stages = int(kwargs.get("num_stages", 4))

        full_res = bool(kwargs.get("full_res", False))
        if full_res and category != "all":
            stride_factor = 2**num_stages
            image_size = detect_aligned_native_size(root_dir / category, stride_factor)
        else:
            image_size_raw = kwargs.get("image_size", (256, 256))
            image_size = (int(image_size_raw[0]), int(image_size_raw[1]))

        dataset_cfg = DatasetConfig(
            root_dir=root_dir,
            category=category,
            image_size=image_size,
            grayscale=bool(kwargs.get("grayscale", False)),
            batch_size=int(kwargs.get("batch_size", 16)),
            num_workers=int(kwargs.get("num_workers", 4)),
            full_res=full_res,
        )

        load_weights_raw = kwargs.get("load_weights")
        load_weights = Path(load_weights_raw) if load_weights_raw is not None else None

        ae_cfg = AutoencoderConfig(
            latent_dim=int(kwargs.get("latent_dim", 128)),
            base_channels=int(kwargs.get("base_channels", 32)),
            num_stages=num_stages,
            loss_type=str(kwargs.get("loss", "l2")),
            epochs=int(kwargs.get("epochs", 50)),
            lr=float(kwargs.get("lr", 1e-3)),
            weight_decay=float(kwargs.get("weight_decay", 1e-5)),
            score_top_k_pct=float(kwargs.get("top_k_pct", 0.0005)),
            gaussian_sigma=float(kwargs.get("gaussian_sigma", 4.0)),
            spatial_dropout=float(kwargs.get("spatial_dropout", 0.1)),
            denoise_prob=float(kwargs.get("denoise_prob", 0.5)),
            texture_damping=float(kwargs.get("texture_damping", 2.0)),
            loss_floor=float(kwargs.get("loss_floor", 0.006)),
            guided_filter_radius=int(kwargs.get("guided_filter_radius", 8)),
            guided_filter_eps=float(kwargs.get("guided_filter_eps", 0.01)),
            use_amp=not bool(kwargs.get("no_amp", False)),
            save_weights=not bool(kwargs.get("no_save_weights", False)),
            load_weights=load_weights,
        )

        output_cfg = OutputConfig(
            base_dir=Path(kwargs.get("output_dir", "results")),
            experiment_name=str(kwargs.get("output", "carpet_raw_ae")),
            save_visualizations=not bool(kwargs.get("no_vis", False)),
            max_vis_samples=int(kwargs.get("max_vis_samples", 25)),
            save_tiff_maps=bool(kwargs.get("save_tiff", False)),
        )

        dinov3_layers_raw = kwargs.get("dinov3_layers")
        dinov3_layers = (
            tuple(int(v) for v in dinov3_layers_raw)
            if dinov3_layers_raw is not None
            else None
        )
        short_edge_raw = kwargs.get("dinov3_target_short_edge")
        extractors_cfg = ExtractorsConfig(
            dinov3=DINOv3ExtractorConfig(
                model_name=str(
                    kwargs.get("dinov3_model", "facebook/dinov3-vits16-pretrain-lvd1689m")
                ),
                stride=int(kwargs.get("dinov3_stride", 16)),
                layers=dinov3_layers,
                prune_registers=bool(kwargs.get("dinov3_prune_registers", False)),
                mean_centering_alpha=float(kwargs.get("dinov3_mean_center_alpha", 0.0)),
                target_short_edge=int(short_edge_raw) if short_edge_raw is not None else None,
                plain_last_hidden_state=bool(kwargs.get("dinov3_plain_last_hidden_state", False)),
            )
        )

        wta_cfg = WTAConfig(
            spatial_window=int(kwargs.get("wta_spatial_window", 4)),
            lifetime_sparsity=float(kwargs.get("wta_lifetime_sparsity", 0.2)),
            ema_momentum=float(kwargs.get("wta_ema_momentum", 0.05)),
            eval_threshold_scale=float(kwargs.get("wta_eval_scale", 0.8)),
        )

        return cls(
            dataset=dataset_cfg,
            extractor_name=str(kwargs.get("extractor", "raw")),
            model_name=str(kwargs.get("model", "dcae")),
            extractors=extractors_cfg,
            autoencoder=ae_cfg,
            wta=wta_cfg,
            output=output_cfg,
            pro_integration_limit=float(kwargs.get("pro_limit", 0.3)),
            seed=int(kwargs.get("seed", 42)),
            device=str(kwargs.get("device", "cuda")),
        )
