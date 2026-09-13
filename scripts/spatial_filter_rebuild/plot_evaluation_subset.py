#!/usr/bin/env python3
"""Regenerate the full-metric boxplot for a readable subset of models."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts.spatial_filter_rebuild.evaluate_full import boxplots


DEFAULT_MODELS = [
    "baseline",
    "molecular_quality_balanced",
    "spatial_quality_balanced",
    "calicost_quality_balanced",
    "integrated_quality_balanced",
    "consensus_quality_balanced",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty plot directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    tables = {
        "silver": pd.read_csv(args.evaluation / "silver_recovery_with_uncertainty.csv"),
        "intrinsic": pd.read_csv(args.evaluation / "intrinsic_metrics.csv"),
        "viewer": pd.read_csv(args.evaluation / "viewer_tumor_clustering.csv"),
        "replicate": pd.read_csv(args.evaluation / "replicate_consistency.csv"),
    }
    for name in tables:
        tables[name] = tables[name].loc[tables[name]["model"].isin(args.models)]
    boxplots(
        tables["silver"], tables["intrinsic"], tables["viewer"],
        tables["replicate"], args.models, args.output,
    )
    print(args.output)


if __name__ == "__main__":
    main()
