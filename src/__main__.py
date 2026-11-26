"""Command-line entry point for the anomaly detection pipeline (python -m src ...)."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import VALID_LOSS_TYPES, PipelineConfig
from src.extractors import EXTRACTOR_NAMES
from src.models import MODEL_NAMES
from src.results import summarize_and_export
from src.run import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Data-Oriented Anomaly Detection Pipeline (MVTec AD)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory")
    parser.add_argument(
        "-c",
        "--category",
        type=str,
        default="carpet",
        help="MVTec AD category name, or 'all' to evaluate all categories in --data-dir",
    )
    parser.add_argument(
        "-e",
        "--extractor",
        type=str,
        choices=sorted(EXTRACTOR_NAMES),
        default="raw",
        help="Feature extractor name",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        choices=sorted(MODEL_NAMES),
        default="dcae",
        help="Anomaly detection model name (dcae, wta_ae, or legacy alias autoencoder)",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=[256, 256],
        metavar=("H", "W"),
        help="Target image height and width",
    )
    parser.add_argument("--grayscale", action="store_true", help="Load images as 1-channel gray")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=128, help="Autoencoder bottleneck dims")
    parser.add_argument(
        "--loss",
        type=str,
        choices=sorted(VALID_LOSS_TYPES),
        default="l2",
        help="Reconstruction loss function",
    )
    parser.add_argument(
        "--top-k-pct",
        type=float,
        default=0.0005,
        help="Top percentage of pixels averaged for image-level score",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="carpet_raw_ae",
        help="Experiment output folder name under results/",
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to pretrained checkpoint to skip training",
    )
    parser.add_argument("--no-vis", action="store_true", help="Disable diagnostic plots/heatmaps")
    parser.add_argument("--device", type=str, default="cuda", help="Compute device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Use native category image resolution (aligned to 2**num_stages)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel image decoding threads",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=32,
        help="Base channel width of the first encoder stage",
    )
    parser.add_argument(
        "--num-stages",
        type=int,
        default=4,
        help="Number of stride-2 encoder/decoder stages",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay",
    )
    parser.add_argument(
        "--gaussian-sigma",
        type=float,
        default=4.0,
        help="Gaussian blur sigma for anomaly map smoothing",
    )
    parser.add_argument(
        "--spatial-dropout",
        type=float,
        default=0.1,
        help="Dropout2d probability in the encoder spatial context bottleneck",
    )
    parser.add_argument(
        "--denoise-prob",
        type=float,
        default=0.5,
        help="Per-image probability of CutPaste/streak/noise denoising corruption during fit()",
    )
    parser.add_argument(
        "--texture-damping",
        type=float,
        default=2.0,
        help="Weight alpha damping high-variance textures using sqrt(Var(x_hat))",
    )
    parser.add_argument(
        "--loss-floor",
        type=float,
        default=0.006,
        help="Early-stopping reconstruction loss floor after epoch 20 (0 disables)",
    )
    parser.add_argument(
        "--guided-filter-radius",
        type=int,
        default=8,
        help="Fast Guided Filter radius using RGB guide edges (0 disables)",
    )
    parser.add_argument(
        "--guided-filter-eps",
        type=float,
        default=0.01,
        help="Fast Guided Filter regularization epsilon (> 0)",
    )
    parser.add_argument(
        "--grad-loss-weight",
        type=float,
        default=2.0,
        help="Weight of multi-scale luminance Sobel gradient L1 term in l1_grad loss",
    )
    parser.add_argument(
        "--riad-patch-sizes",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Multi-scale RIAD disjoint grid mask sizes in pixels (e.g. 32 64 128); "
            "omit or pass empty to disable"
        ),
    )
    parser.add_argument(
        "--riad-num-masks",
        type=int,
        default=4,
        help="Number of disjoint RIAD masks per patch scale (>= 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (bfloat16) on CUDA",
    )
    parser.add_argument(
        "--pro-limit",
        type=float,
        default=0.3,
        help="FPR integration limit in (0.0, 1.0] for official MVTec AU-PRO",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Export native-resolution .tiff anomaly maps compatible with official MVTec evaluator",
    )
    parser.add_argument(
        "--summarize-results",
        action="store_true",
        help="Inspect --output-dir contents, print leaderboard, and export comparison CSVs",
    )
    parser.add_argument(
        "--dinov3-model",
        type=str,
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="HuggingFace model ID for DINOv3 extractor",
    )
    parser.add_argument(
        "--dinov3-stride",
        type=int,
        default=16,
        help="Patch embedding stride for DINOv3 (e.g. 16 default, 12/14 overlap)",
    )
    parser.add_argument(
        "--dinov3-layers",
        nargs="+",
        type=int,
        default=None,
        help="Optional 1-based transformer layer indices for multi-layer DINOv3 fusion",
    )
    parser.add_argument(
        "--dinov3-prune-registers",
        action="store_true",
        help="Prune DINOv3 register tokens via embeddings forward hook",
    )
    parser.add_argument(
        "--dinov3-mean-center-alpha",
        type=float,
        default=0.0,
        help="Hypersphere mean-centering alpha before L2 normalization",
    )
    parser.add_argument(
        "--dinov3-target-short-edge",
        type=int,
        default=None,
        help="Optional short-edge target size before DINOv3 patch grid alignment",
    )
    parser.add_argument(
        "--dinov3-plain-last-hidden-state",
        action="store_true",
        help="Disable 3x3 neighborhood AvgPool2d smoothing on DINOv3 patch tokens",
    )
    parser.add_argument(
        "--wta-spatial-window",
        type=int,
        default=4,
        help="Non-overlapping spatial winner window size for wta_ae bottleneck",
    )
    parser.add_argument(
        "--wta-lifetime-sparsity",
        type=float,
        default=0.2,
        help="Mini-batch lifetime winner fraction in (0.0, 1.0] for wta_ae bottleneck",
    )
    parser.add_argument(
        "--wta-eval-scale",
        type=float,
        default=0.8,
        help="Test-time EMA threshold scale factor for wta_ae eval() mode",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the anomaly detection pipeline."""
    args = build_parser().parse_args(argv)
    if args.summarize_results:
        summarize_and_export(results_dir=Path(args.output_dir))
        return
    config = PipelineConfig.from_args(**vars(args))
    run_pipeline(config)


if __name__ == "__main__":
    main()
