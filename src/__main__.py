"""Command-line entry point for the anomaly detection pipeline (python -m src ...)."""

from __future__ import annotations

import argparse

from src.config import PipelineConfig
from src.extractors import EXTRACTOR_NAMES
from src.models import MODEL_NAMES
from src.run import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Data-Oriented Anomaly Detection Pipeline (MVTec AD)",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory")
    parser.add_argument("-c", "--category", type=str, default="carpet", help="MVTec AD category")
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
        default="autoencoder",
        help="Anomaly detection model name",
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
        choices=["l1", "l2", "smooth_l1"],
        default="l2",
        help="Reconstruction loss function",
    )
    parser.add_argument(
        "--top-k-pct",
        type=float,
        default=0.01,
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
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and run the anomaly detection pipeline."""
    args = build_parser().parse_args(argv)
    config = PipelineConfig.from_args(**vars(args))
    run_pipeline(config)


if __name__ == "__main__":
    main()
