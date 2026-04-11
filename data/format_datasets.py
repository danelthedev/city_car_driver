from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.taxonomy import DATASET_TARGETS
from pipeline.config import PipelineConfig
from pipeline.pipeline import parse_requested_datasets, run_pipeline


def parse_args() -> argparse.Namespace:
    default_data_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build hybrid YOLO datasets with one unified class space across iterative stages."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma or space-separated targets to generate. "
            f"Allowed: all, {', '.join(DATASET_TARGETS)}. "
            "Aliases: mapillary, gtsdb, romanian, bstld"
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root,
        help="Root data directory (default: script directory)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_data_root / "traffic_sign_pipeline",
        help="Output directory for generated datasets and reports",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gtsdb-val-ratio",
        type=float,
        default=0.20,
        help="Fraction of GTSDB images reserved for validation (default: 0.20)",
    )
    parser.add_argument(
        "--bstld-val-ratio",
        type=float,
        default=0.15,
        help="Fraction of BSTLD training bags reserved for validation (default: 0.15)",
    )
    parser.add_argument(
        "--bstld-negatives",
        type=int,
        default=2000,
        help="Number of empty Mapillary frames added as hard negatives to iter_bstld train (0 to disable)",
    )
    parser.add_argument(
        "--synthetic-min-instances",
        type=int,
        default=250,
        help="Minimum target count per class when computing synthetic deficits",
    )
    parser.add_argument(
        "--synthetic-max-per-class",
        type=int,
        default=400,
        help="Cap of generated synthetic samples per class",
    )
    parser.add_argument(
        "--exclude-gtsrb-test",
        action="store_true",
        help="Use only GTSRB Train.csv as crop source (exclude Test.csv)",
    )
    parser.add_argument(
        "--no-overwrite-output",
        action="store_true",
        help="Fail if a selected output dataset already exists instead of replacing it",
    )
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    if not 0.0 < float(args.gtsdb_val_ratio) < 1.0:
        raise ValueError("--gtsdb-val-ratio must be between 0 and 1")
    if not 0.0 < float(args.bstld_val_ratio) < 1.0:
        raise ValueError("--bstld-val-ratio must be between 0 and 1")
    if args.synthetic_min_instances <= 0:
        raise ValueError("--synthetic-min-instances must be > 0")
    if args.synthetic_max_per_class <= 0:
        raise ValueError("--synthetic-max-per-class must be > 0")

    return PipelineConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        seed=args.seed,
        gtsdb_val_ratio=args.gtsdb_val_ratio,
        bstld_val_ratio=args.bstld_val_ratio,
        bstld_negatives=args.bstld_negatives,
        overwrite_output=not args.no_overwrite_output,
        include_gtsrb_test=not args.exclude_gtsrb_test,
        synthetic_min_instances=args.synthetic_min_instances,
        synthetic_max_per_class=args.synthetic_max_per_class,
    )


def main() -> None:
    args = parse_args()
    requested_datasets = parse_requested_datasets(args.datasets)
    config = build_config_from_args(args)
    run_pipeline(config, requested_datasets)


if __name__ == "__main__":
    main()