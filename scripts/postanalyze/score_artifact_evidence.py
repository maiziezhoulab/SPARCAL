#!/usr/bin/env python3
"""Create a transparent, non-destructive artifact-evidence table.

This is intentionally *not* a supervised somatic caller and its scores are not
posterior probabilities. It converts a site-level feature table from
``collect_spatial_artifact_features.py`` into explicit annotation, technical,
and low-information flags. The result is meant to be used before—not instead
of—a later spatial/CalicoST model.

Technical thresholds are estimated from the high-confidence ``defined``
provenance stratum in the same section when possible. That stratum calibrates
the assay's read quality only; it is never treated as somatic truth. All output
is written to a new directory and no input VCF/matrix is modified.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


QUALITY_COLUMNS = {
    "alt_mean_bq": ("min_alt_bq", 20.0, 0.05, "low"),
    "alt_mean_mq": ("min_alt_mq", 30.0, 0.05, "low"),
    "alt_terminal_frac": ("max_alt_terminal_frac", 0.50, 0.95, "high"),
    "alt_multi_map_frac": ("max_alt_multi_map_frac", 0.50, 0.95, "high"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path,
                        help="Feature-collector directory containing site_features.tsv.gz.")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="New output directory; must not already contain results.")
    parser.add_argument("--calibration-source", default="defined",
                        help="Candidate provenance label used only for technical threshold calibration.")
    parser.add_argument("--min-calibration-sites", type=int, default=100,
                        help="Minimum calibration rows needed to use empirical quality quantiles.")
    parser.add_argument("--min-alt-umis", type=int, default=2,
                        help="Minimum alt UMIs for an observation to be called adequately supported.")
    parser.add_argument("--min-alt-spots", type=int, default=2,
                        help="Minimum spots with an alt read for spatial support evidence.")
    parser.add_argument("--strand-min-reads", type=int, default=4,
                        help="Alt reads required before testing strand imbalance.")
    parser.add_argument("--max-strand-imbalance", type=float, default=0.90,
                        help="Flag if >= this fraction of sufficiently many alt reads are one orientation.")
    parser.add_argument("--quality-drop-bq", type=float, default=10.0,
                        help="Flag if alt mean BQ is this much lower than reference mean BQ.")
    return parser.parse_args()


def contains_source(value: object, label: str) -> bool:
    return label in {token for token in str(value).split(",") if token}


def finite_quantile(series: pd.Series, quantile: float) -> float | None:
    values = pd.to_numeric(series, errors="coerce")
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    return float(values.quantile(quantile))


def choose_thresholds(df: pd.DataFrame, args: argparse.Namespace) -> tuple[dict[str, float], dict[str, object]]:
    provenance = df["candidate_sources"].map(lambda value: contains_source(value, args.calibration_source))
    calibration = df.loc[provenance & (pd.to_numeric(df["alt_n"], errors="coerce") > 0)].copy()
    thresholds: dict[str, float] = {}
    metadata: dict[str, object] = {
        "source": args.calibration_source,
        "n_calibration_sites": int(len(calibration)),
        "used_empirical_quantiles": bool(len(calibration) >= args.min_calibration_sites),
    }
    for column, (name, fallback, quantile, _) in QUALITY_COLUMNS.items():
        value = None
        if len(calibration) >= args.min_calibration_sites and column in calibration:
            value = finite_quantile(calibration[column], quantile)
        thresholds[name] = fallback if value is None else value
    metadata["thresholds"] = thresholds
    return thresholds, metadata


def number(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in df:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def main() -> int:
    args = parse_args()
    if args.min_alt_umis < 1 or args.min_alt_spots < 1:
        raise ValueError("Minimum alt UMI and spot support must both be >= 1.")
    source = args.input_dir / "site_features.tsv.gz"
    if not source.is_file():
        raise FileNotFoundError(source)
    out_dir = args.out_dir
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to replace existing artifact evidence: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source, sep="\t", na_values="NA", keep_default_na=True)
    required = {"candidate_id", "candidate_sources", "alt_n", "alt_umi_n", "n_spots_alt"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input lacks required collector columns: {sorted(missing)}")
    for column in ("known_rna_editing", "external_pon", "editing_compatible_substitution"):
        if column not in df:
            df[column] = 0

    thresholds, calibration = choose_thresholds(df, args)
    alt_n = number(df, "alt_n", 0).fillna(0)
    alt_umis = number(df, "alt_umi_n", 0).fillna(0)
    alt_spots = number(df, "n_spots_alt", 0).fillna(0)
    alt_bq = number(df, "alt_mean_bq")
    alt_mq = number(df, "alt_mean_mq")
    alt_terminal = number(df, "alt_terminal_frac")
    alt_multimap = number(df, "alt_multi_map_frac")
    alt_forward = number(df, "alt_forward_frac")
    bq_drop = number(df, "alt_minus_ref_mean_bq")

    # External masks are annotations, not automatic deletions: their portability
    # has already been shown to vary by platform and tissue in this project.
    df["flag_known_rna_editing"] = number(df, "known_rna_editing", 0).fillna(0).astype(int).eq(1)
    df["flag_external_pon"] = number(df, "external_pon", 0).fillna(0).astype(int).eq(1)
    df["flag_editing_compatible_substitution"] = (
        number(df, "editing_compatible_substitution", 0).fillna(0).astype(int).eq(1)
    )
    df["flag_low_alt_bq"] = alt_bq.lt(thresholds["min_alt_bq"])
    df["flag_low_alt_mq"] = alt_mq.lt(thresholds["min_alt_mq"])
    df["flag_terminal_bias"] = alt_terminal.gt(thresholds["max_alt_terminal_frac"])
    df["flag_multi_mapping"] = alt_multimap.gt(thresholds["max_alt_multi_map_frac"])
    df["flag_strand_imbalance"] = (
        alt_n.ge(args.strand_min_reads)
        & np.maximum(alt_forward, 1.0 - alt_forward).ge(args.max_strand_imbalance)
    )
    df["flag_alt_bq_drop"] = bq_drop.lt(-abs(args.quality_drop_bq))
    df["flag_low_umi_support"] = alt_umis.lt(args.min_alt_umis)
    df["flag_low_spatial_support"] = alt_spots.lt(args.min_alt_spots)

    annotation_flags = ["flag_known_rna_editing", "flag_external_pon"]
    technical_flags = [
        "flag_low_alt_bq", "flag_low_alt_mq", "flag_terminal_bias", "flag_multi_mapping",
        "flag_strand_imbalance", "flag_alt_bq_drop",
    ]
    support_flags = ["flag_low_umi_support", "flag_low_spatial_support"]
    df["annotation_signal_count"] = df[annotation_flags].sum(axis=1).astype(int)
    df["technical_artifact_signal_count"] = df[technical_flags].sum(axis=1).astype(int)
    df["low_information_signal_count"] = df[support_flags].sum(axis=1).astype(int)
    df["artifact_signal_count"] = (
        df["annotation_signal_count"] + df["technical_artifact_signal_count"]
    ).astype(int)
    # A bounded rank statistic only—not a calibrated probability.
    df["artifact_risk_fraction"] = df["artifact_signal_count"] / (len(annotation_flags) + len(technical_flags))
    df["read_evidence_score"] = (
        np.minimum(1.0, alt_umis / args.min_alt_umis)
        + np.minimum(1.0, alt_spots / args.min_alt_spots)
        + (~df["flag_low_alt_bq"]).astype(float)
        + (~df["flag_low_alt_mq"]).astype(float)
        + (~df["flag_terminal_bias"]).astype(float)
        + (~df["flag_multi_mapping"]).astype(float)
    ) / 6.0
    df["artifact_evidence_status"] = np.select(
        [
            df["annotation_signal_count"].gt(0),
            df["technical_artifact_signal_count"].ge(2),
            df["low_information_signal_count"].gt(0),
        ],
        ["external_annotation", "multiple_technical_flags", "low_information"],
        default="read_evidence_pass",
    )

    output = out_dir / "site_artifact_evidence.tsv.gz"
    df.to_csv(output, sep="\t", index=False, compression="gzip")
    summary = {
        "input": str(source.resolve()),
        "n_sites": int(len(df)),
        "calibration": calibration,
        "support_parameters": {
            "min_alt_umis": args.min_alt_umis,
            "min_alt_spots": args.min_alt_spots,
            "strand_min_reads": args.strand_min_reads,
            "max_strand_imbalance": args.max_strand_imbalance,
            "quality_drop_bq": args.quality_drop_bq,
        },
        "status_counts": {str(key): int(value) for key, value in df["artifact_evidence_status"].value_counts().items()},
        "important_interpretation": (
            "artifact_risk_fraction is an interpretable ranking statistic, not a calibrated probability; "
            "no VCF records were filtered or rewritten."
        ),
    }
    with (out_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
