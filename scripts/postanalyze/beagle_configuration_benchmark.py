#!/usr/bin/env python3
"""Evaluate WES-free Beagle routes with matched-DNA evidence and run stability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts/2_beagle_filtering"))
from beagle_evidence_router import (  # noqa: E402
    meets_threshold,
    normalize_gt,
    parse_gp,
)

Key = Tuple[str, int, str, str]
SAMPLES = ("P4", "P6")
ROUTES = (
    "beagle_nonref_highconf", "beagle_ref_highconf",
    "beagle_uncertain", "panel_absent",
)


def key_from_row(row: Mapping[str, object]) -> Key:
    return str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"])


def bool_value(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "t", "yes"})


def wilson(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_audits(root: Path) -> dict[tuple[str, str], Path]:
    """Discover only completed runs and resolve the known dual-router layout."""
    found = {}
    priority = {}
    for path in sorted(root.rglob("beagle_evidence_audit.tsv.gz")):
        parts = path.relative_to(root).parts
        indices = [
            i
            for i, part in enumerate(parts)
            if part in {"P4_sec1", "P6_sec1"}
        ]
        if len(indices) != 1:
            continue
        index = indices[0]
        sample_root = root.joinpath(*parts[: index + 1])
        if not (
            (sample_root / "RUN_COMPLETE").is_file()
            or (sample_root / "RUN_COMPLETE.json").is_file()
        ):
            continue
        sample = parts[index].split("_", 1)[0]
        config = "/".join(parts[:index]) or "candidate"
        key = config, sample
        integrated = (
            "output_VCFs" in parts[index + 1 :]
            and "beagle" in parts[index + 1 :]
        )
        path_priority = 1 if integrated else 0
        if key in found:
            if priority[key] == path_priority:
                raise ValueError(
                    f"duplicate audit for {config}/{sample}: "
                    f"{found[key]} and {path}"
                )
            if path_priority < priority[key]:
                found[key] = path
                priority[key] = path_priority
        else:
            found[key] = path
            priority[key] = path_priority
    if not found:
        raise FileNotFoundError(
            f"no completed beagle_evidence_audit.tsv.gz under {root}"
        )
    return found


def read_evidence(path: Path) -> dict[Key, dict]:
    frame = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    required = {
        "chrom", "pos", "ref", "alt", "beagle_gt", "beagle_gp",
        "beagle_called_gp", "beagle_p_nonref", "route",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    frame["pos"] = frame["pos"].astype(int)
    if frame.duplicated(["chrom", "pos", "ref", "alt"]).any():
        raise ValueError(f"duplicate exact alleles in {path}")
    out = {}
    for _, row in frame.iterrows():
        called_gp: Optional[float] = None
        if not pd.isna(row.beagle_called_gp) and str(row.beagle_called_gp) != ".":
            called_gp = float(row.beagle_called_gp)
        p_nonref: Optional[float] = None
        if not pd.isna(row.beagle_p_nonref) and str(row.beagle_p_nonref) != ".":
            p_nonref = float(row.beagle_p_nonref)
        gp = parse_gp(None if pd.isna(row.beagle_gp) else str(row.beagle_gp))
        out[key_from_row(row)] = {
            "route": str(row.route),
            "gt": normalize_gt(None if pd.isna(row.beagle_gt) else str(row.beagle_gt)),
            "called_gp": called_gp,
            "p_nonref": p_nonref,
            "p_ref": gp[0] if gp is not None else None,
        }
    return out


def reroute_evidence(
    evidence: dict[Key, dict],
    anchor_score: str,
    min_anchor_probability: float,
) -> dict[Key, dict]:
    """Recompute routes from stored posteriors so configurations are comparable."""
    if anchor_score not in {"recorded_route", "p_nonref", "called_gp"}:
        raise ValueError(f"unknown anchor_score: {anchor_score}")
    out = {}
    for key, original in evidence.items():
        value = dict(original)
        if anchor_score == "recorded_route":
            out[key] = value
            continue
        if value["route"] == "panel_absent":
            route = "panel_absent"
        elif anchor_score == "p_nonref":
            if meets_threshold(value["p_nonref"], min_anchor_probability):
                route = "beagle_nonref_highconf"
            elif meets_threshold(value["p_ref"], min_anchor_probability):
                route = "beagle_ref_highconf"
            else:
                route = "beagle_uncertain"
        elif meets_threshold(value["called_gp"], min_anchor_probability):
            route = (
                "beagle_nonref_highconf"
                if value["gt"] in {"0/1", "1/1"}
                else "beagle_ref_highconf"
                if value["gt"] == "0/0"
                else "beagle_uncertain"
            )
        else:
            route = "beagle_uncertain"
        value["route"] = route
        out[key] = value
    return out


def rows_at_keys(frame: pd.DataFrame, sample: str, keys: set[Key]) -> dict[Key, pd.Series]:
    out = {}
    for _, row in frame.loc[frame["sample"] == sample].iterrows():
        key = key_from_row(row)
        if key in keys:
            out[key] = row
    return out


def metric(config: str, sample: str, name: str, numerator: int, denominator: int) -> dict:
    low, high = wilson(numerator, denominator)
    return {
        "config": config, "sample": sample, "metric": name,
        "numerator": numerator, "denominator": denominator,
        "value": numerator / denominator if denominator else math.nan,
        "ci95_low": low, "ci95_high": high,
    }


def evaluate(config, sample, evidence, normal, support, somatic, table_s1):
    def sample_rows(frame: pd.DataFrame) -> dict[Key, pd.Series]:
        return {
            key_from_row(row): row
            for _, row in frame.loc[frame["sample"] == sample].iterrows()
        }

    keys = set(evidence)
    anchors = {key for key, value in evidence.items() if value["route"] == ROUTES[0]}
    normal_all = sample_rows(normal)
    normal_rows = {key: row for key, row in normal_all.items() if key in keys}
    support_all = sample_rows(support.loc[support["wes_pileup_dp"] >= 10])
    support_rows = {key: row for key, row in support_all.items() if key in keys}
    somatic_all = sample_rows(somatic.loc[bool_value(somatic["rna_covered"])])
    somatic_rows = {key: row for key, row in somatic_all.items() if key in keys}
    curated = table_s1.rename(columns={"chrom_hg19": "chrom", "pos_hg19": "pos"})
    curated_all = sample_rows(curated)
    curated_rows = {key: row for key, row in curated_all.items() if key in keys}
    normal_anchor = anchors & set(normal_rows)
    support_anchor = anchors & set(support_rows)
    somatic_anchor = anchors & set(somatic_rows)
    curated_anchor = anchors & set(curated_rows)
    gt_correct = sum(
        evidence[key]["gt"] == normalize_gt(str(normal_rows[key]["wes_gt"]))
        for key in normal_anchor
    )
    supported = sum(
        int(support_rows[key]["high_conf_wes_exact"]) == 1 for key in support_anchor
    )
    somatic_candidates_retained = (keys - anchors) & set(somatic_all)
    curated_candidates_retained = (keys - anchors) & set(curated_all)
    metrics = [
        metric(config, sample, "raw_candidate_normal_wes_recall", len(normal_rows), len(normal_all)),
        metric(config, sample, "normal_wes_end_to_end_anchor_recall", len(normal_anchor), len(normal_all)),
        metric(config, sample, "normal_wes_anchor_sensitivity", len(normal_anchor), len(normal_rows)),
        metric(config, sample, "normal_wes_anchor_gt_accuracy", gt_correct, len(normal_anchor)),
        metric(config, sample, "selected_anchor_normal_wes_support", supported, len(support_anchor)),
        metric(config, sample, "raw_candidate_wes_somatic_recall", len(somatic_rows), len(somatic_all)),
        metric(config, sample, "wes_somatic_end_to_end_retention", len(somatic_candidates_retained), len(somatic_all)),
        metric(config, sample, "wes_somatic_anchor_leakage", len(somatic_anchor), len(somatic_rows)),
        metric(config, sample, "wes_somatic_retention", len(somatic_rows) - len(somatic_anchor), len(somatic_rows)),
        metric(config, sample, "raw_candidate_table_s1_recall", len(curated_rows), len(curated_all)),
        metric(config, sample, "table_s1_end_to_end_retention", len(curated_candidates_retained), len(curated_all)),
        metric(config, sample, "table_s1_somatic_anchor_leakage", len(curated_anchor), len(curated_rows)),
    ]
    route_counts = pd.Series([value["route"] for value in evidence.values()]).value_counts()
    counts = {
        "config": config, "sample": sample, "raw_candidate_count": len(keys),
        "hard_anchor_count": len(anchors),
        "hard_anchor_fraction": len(anchors) / len(keys) if keys else math.nan,
        **{route: int(route_counts.get(route, 0)) for route in ROUTES},
    }
    return metrics, counts


def settings_without_seed(config: str) -> str:
    return re.sub(r"([_/])seed-?\d+", "", config).replace("__", "_").strip("_/")


def pairwise_stability(configs, sample: str) -> list[dict]:
    rows = []
    for left, right in combinations(sorted(configs), 2):
        a, b = configs[left], configs[right]
        raw_a, raw_b = set(a), set(b)
        common, raw_union = raw_a & raw_b, raw_a | raw_b
        anchor_a = {key for key, value in a.items() if value["route"] == ROUTES[0]}
        anchor_b = {key for key, value in b.items() if value["route"] == ROUTES[0]}
        anchor_union = anchor_a | anchor_b
        gt_common = [key for key in common if a[key]["gt"] is not None and b[key]["gt"] is not None]
        gp_common = [key for key in common if a[key]["called_gp"] is not None and b[key]["called_gp"] is not None]
        p_nonref_common = [
            key for key in common
            if a[key].get("p_nonref") is not None and b[key].get("p_nonref") is not None
        ]
        rows.append({
            "sample": sample, "config_a": left, "config_b": right,
            "same_settings_except_seed": settings_without_seed(left) == settings_without_seed(right),
            "raw_a_n": len(raw_a), "raw_b_n": len(raw_b),
            "raw_jaccard": len(common) / len(raw_union) if raw_union else math.nan,
            "common_raw_n": len(common),
            "route_agreement": sum(a[key]["route"] == b[key]["route"] for key in common) / len(common) if common else math.nan,
            "anchor_a_n": len(anchor_a), "anchor_b_n": len(anchor_b),
            "anchor_jaccard": len(anchor_a & anchor_b) / len(anchor_union) if anchor_union else math.nan,
            "anchor_symmetric_difference_n": len(anchor_a ^ anchor_b),
            "gt_compared_n": len(gt_common),
            "gt_concordance": sum(normalize_gt(a[key]["gt"]) == normalize_gt(b[key]["gt"]) for key in gt_common) / len(gt_common) if gt_common else math.nan,
            "gp_compared_n": len(gp_common),
            "called_gp_mae": float(np.mean([abs(a[key]["called_gp"] - b[key]["called_gp"]) for key in gp_common])) if gp_common else math.nan,
            "p_nonref_compared_n": len(p_nonref_common),
            "p_nonref_mae": float(np.mean([
                abs(a[key]["p_nonref"] - b[key]["p_nonref"]) for key in p_nonref_common
            ])) if p_nonref_common else math.nan,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-score", choices=("p_nonref", "called_gp", "recorded_route"),
        default="p_nonref",
        help="Recompute every audit with the same evidence rule (default: p_nonref)",
    )
    parser.add_argument("--min-anchor-probability", type=float, default=0.90)
    args = parser.parse_args()
    if not 0 <= args.min_anchor_probability <= 1:
        raise ValueError("min_anchor_probability must be between 0 and 1")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    normal = pd.read_csv(PROJECT / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz")
    normal = normal.loc[(normal["detected"] == 1) & (normal["rna_dp"] >= 1)].copy()
    support = pd.read_csv(PROJECT / "data/germline_concordance_2026-08-23/sparcal_call_details.csv.gz")
    somatic = pd.read_csv(PROJECT / "data/somatic_evidence_2026-07-28/wes_confirmed_full_accounting.csv")
    table_s1 = pd.read_csv(PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv", sep="\t")
    metrics, counts = [], []
    evidence_by_sample = {sample: {} for sample in SAMPLES}
    provenance = []
    for (config, sample), path in sorted(discover_audits(args.experiment_root).items()):
        evidence = reroute_evidence(
            read_evidence(path), args.anchor_score, args.min_anchor_probability
        )
        evidence_by_sample[sample][config] = evidence
        sample_metrics, sample_counts = evaluate(
            config, sample, evidence, normal, support, somatic, table_s1
        )
        metrics.extend(sample_metrics)
        counts.append(sample_counts)
        provenance.append({
            "config": config, "sample": sample, "audit": str(path.resolve()),
            "audit_sha256": sha256(path),
        })
    metrics_frame, counts_frame = pd.DataFrame(metrics), pd.DataFrame(counts)
    pairwise = pd.DataFrame([
        row for sample in SAMPLES
        for row in pairwise_stability(evidence_by_sample[sample], sample)
    ])
    wide = metrics_frame.pivot(
        index=["config", "sample"], columns="metric", values="value"
    ).reset_index()
    summary = counts_frame.merge(wide, on=["config", "sample"], how="left")
    metrics_frame.to_csv(args.output_dir / "configuration_metrics_long.tsv", sep="\t", index=False)
    summary.to_csv(args.output_dir / "configuration_summary.tsv", sep="\t", index=False)
    pairwise.to_csv(args.output_dir / "pairwise_stability.tsv", sep="\t", index=False)
    with (args.output_dir / "PROVENANCE.json").open("wt") as handle:
        json.dump({
            "production_inputs_include_wes": False, "evaluation_uses_wes": True,
            "anchor_score": args.anchor_score,
            "min_anchor_probability": args.min_anchor_probability,
            "audits": provenance,
        }, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    print(summary.to_string(index=False))
    if len(pairwise):
        print("\nPairwise stability:\n" + pairwise.to_string(index=False))


if __name__ == "__main__":
    main()
