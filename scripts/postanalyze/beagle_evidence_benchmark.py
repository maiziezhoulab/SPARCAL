#!/usr/bin/env python3
"""WES-excluded matched-DNA benchmark of SPARCAL Beagle evidence routing.

WES never enters production inference. Because these evaluation results can
inform method selection, they are internal development evidence rather than an
untouched held-out test. Four SPARCAL routing rules are compared on the same
exact-allele raw-candidate universe. External Monopogen executions are audited
separately on their exact overlap with that universe because their mpileup input
contract differs. Outputs are immutable: an existing output directory is an error.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata as importlib_metadata
import json
import math
import platform
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
ROUTER_PATH = PROJECT / "scripts" / "2_beagle_filtering" / "beagle_evidence_router.py"
sys.path.insert(0, str(ROUTER_PATH.parent))
from beagle_evidence_router import (  # noqa: E402
    extract_format_value,
    meets_threshold,
    normalize_gt,
    parse_gp,
)


Key = Tuple[str, int, str, str]
NO_GT = {None, "", ".", "./.", ".|."}
SAMPLES = ("P4", "P6")
THRESHOLDS = (0.50, 0.80, 0.90, 0.95, 0.99, 0.999)
ANCHOR_THRESHOLD = 0.90
BENCHMARK_SCHEMA_VERSION = "11.0"
PINNED_MONOPOGEN_COMMIT = "2c0a4b0868cb86a77f196cba29b4d4dd3ce482f6"
MONOPOGEN_SOURCE_ROOT = (
    "https://github.com/KChen-lab/Monopogen/blob/"
    f"{PINNED_MONOPOGEN_COMMIT}"
)
DEFAULT_MONOPOGEN_AUDIT = (
    PROJECT
    / "data/experiments/monopogen_beagle_contract_audit_20260912_v3_corrected"
)
DEFAULT_TRUSTED_BENCHMARKS = (
    PROJECT / "data/experiments/beagle_evidence_benchmark_20260912_v4",
    PROJECT
    / "data/experiments/beagle_evidence_benchmark_20260912_v8_hardened_final",
)

LEGACY_PANEL_METHOD = "Legacy any panel record"
LEGACY_CONCORDANCE_METHOD = (
    "SPARCAL legacy paper rule (raw nonref=Beagle GT; called-GP>=0.90)"
)
CALLED_GP_METHOD = "SPARCAL called-GP>=0.90 ablation"
PNONREF_METHOD = "SPARCAL P(nonref)>=0.90"
MONO_RELEASED_COMPONENT = (
    "Monopogen released-code phased-record membership component"
)
MONO_PHASED_GT_DIAGNOSTIC = "Monopogen final-phased nonref-GT diagnostic"
MONO_PAPER_METHOD = "Monopogen paper concordant nonref"
MONO_PNONREF_METHOD = "Monopogen P(nonref)>=0.90 (derived)"
SAME_INPUT_METHODS = (
    LEGACY_PANEL_METHOD,
    LEGACY_CONCORDANCE_METHOD,
    CALLED_GP_METHOD,
    PNONREF_METHOD,
)
EXTERNAL_METHODS = (
    MONO_RELEASED_COMPONENT,
    MONO_PHASED_GT_DIAGNOSTIC,
    MONO_PAPER_METHOD,
    MONO_PNONREF_METHOD,
)

TRUSTED_LABEL_MAP = {
    "Legacy any panel record": LEGACY_PANEL_METHOD,
    "Three-route concordance": LEGACY_CONCORDANCE_METHOD,
    "SPARCAL raw/Beagle GT concordance (paper-specified rule)": (
        LEGACY_CONCORDANCE_METHOD
    ),
    "SPARCAL legacy paper rule (raw nonref=Beagle GT; called-GP>=0.90)": (
        LEGACY_CONCORDANCE_METHOD
    ),
    "Four-route nonref GP>=0.90": CALLED_GP_METHOD,
    "SPARCAL called-GP>=0.90 ablation": CALLED_GP_METHOD,
    "Four-route P(nonref)>=0.90": PNONREF_METHOD,
    "SPARCAL P(nonref)>=0.90": PNONREF_METHOD,
}


@dataclass(frozen=True)
class MethodDecision:
    anchor: bool
    gt: Optional[str]


@dataclass(frozen=True)
class MonopogenEvidence:
    """Evidence needed to distinguish the paper rule from released code."""

    raw_gl_gt: Optional[str]
    beagle_gp_gt: Optional[str]
    final_phased_gt: Optional[str]
    called_gp: Optional[float]
    p_nonref: Optional[float]
    final_phased_present: bool = False


def key_from_row(row: Mapping[str, object]) -> Key:
    return str(row["chrom"]), int(row["pos"]), str(row["ref"]), str(row["alt"])


def read_router_audit(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={"chrom": str})
    frame["pos"] = frame["pos"].astype(int)
    return frame


def decisions_from_audits(
    new_audit: Path, old_audit: Path
) -> Tuple[Dict[Key, Dict[str, MethodDecision]], pd.DataFrame]:
    new = read_router_audit(new_audit)
    old = pd.read_csv(old_audit, sep="\t", dtype={"chrom": str})
    old["pos"] = old["pos"].astype(int)
    old_status = {
        key_from_row(row): row.panel_status for _, row in old.iterrows()
    }
    out: Dict[Key, Dict[str, MethodDecision]] = {}
    for _, row in new.iterrows():
        key = key_from_row(row)
        gt = normalize_gt(None if pd.isna(row.beagle_gt) else str(row.beagle_gt))
        represented = gt is not None
        out[key] = {
            LEGACY_PANEL_METHOD: MethodDecision(represented, gt),
            LEGACY_CONCORDANCE_METHOD: MethodDecision(
                old_status.get(key) == "panel_concordant", gt
            ),
            CALLED_GP_METHOD: MethodDecision(
                gt in {"0/1", "1/1"}
                and not pd.isna(row.beagle_called_gp)
                and str(row.beagle_called_gp) != "."
                and meets_threshold(float(row.beagle_called_gp), ANCHOR_THRESHOLD),
                gt,
            ),
            PNONREF_METHOD: MethodDecision(
                gt in {"0/1", "1/1"}
                and not pd.isna(row.beagle_p_nonref)
                and str(row.beagle_p_nonref) != "."
                and meets_threshold(float(row.beagle_p_nonref), ANCHOR_THRESHOLD),
                gt,
            ),
        }
    return out, new


def genotype_from_pl(pl_text: Optional[str]) -> Optional[str]:
    """Return the maximum-likelihood diploid GT; PL ties use lowest VCF index."""
    if pl_text in NO_GT:
        return None
    try:
        values = tuple(float(value) for value in str(pl_text).split(","))
    except ValueError:
        return None
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        return None
    minimum = min(values)
    winner = next(index for index, value in enumerate(values) if value == minimum)
    return ("0/0", "0/1", "1/1")[winner]


def read_monopogen_evidence(
    run_dir: Path, raw_keys: Iterable[Key]
) -> Dict[Key, MonopogenEvidence]:
    """Read raw GL, first-pass GP and released-code final phased genotypes."""
    wanted = set(raw_keys)
    raw_gt: Dict[Key, Optional[str]] = {}
    gp_values: Dict[
        Key, Tuple[Optional[str], Optional[float], Optional[float]]
    ] = {}
    final_gt: Dict[Key, Optional[str]] = {}
    germline = run_dir / "out" / "germline"
    for chrom in range(1, 23):
        raw_path = germline / f"chr{chrom}.gl.vcf.gz"
        gp_path = germline / f"chr{chrom}.gp.vcf.gz"
        phased_path = germline / f"chr{chrom}.phased.vcf.gz"
        if raw_path.is_file():
            with gzip.open(raw_path, "rt") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 10 or len(fields[3]) != 1 or len(fields[4]) != 1:
                        continue
                    key = (fields[0], int(fields[1]), fields[3], fields[4])
                    if key in wanted:
                        raw_gt[key] = genotype_from_pl(
                            extract_format_value(fields[8], fields[9], "PL")
                        )
        if gp_path.is_file():
            with gzip.open(gp_path, "rt") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 10 or len(fields[3]) != 1 or len(fields[4]) != 1:
                        continue
                    key = (fields[0], int(fields[1]), fields[3], fields[4])
                    if key not in wanted:
                        continue
                    gt = normalize_gt(
                        extract_format_value(fields[8], fields[9], "GT")
                    )
                    gp = parse_gp(
                        extract_format_value(fields[8], fields[9], "GP")
                    )
                    index = {"0/0": 0, "0/1": 1, "1/1": 2}.get(gt)
                    called_gp = (
                        gp[index]
                        if gp is not None and index is not None
                        else None
                    )
                    p_nonref = gp[1] + gp[2] if gp is not None else None
                    gp_values[key] = (gt, called_gp, p_nonref)
        if phased_path.is_file():
            with gzip.open(phased_path, "rt") as handle:
                for line in handle:
                    if line.startswith("#"):
                        continue
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) < 10 or len(fields[3]) != 1 or len(fields[4]) != 1:
                        continue
                    key = (fields[0], int(fields[1]), fields[3], fields[4])
                    if key in wanted:
                        final_gt[key] = normalize_gt(
                            extract_format_value(fields[8], fields[9], "GT")
                        )

    out: Dict[Key, MonopogenEvidence] = {}
    observed = raw_gt.keys() | gp_values.keys() | final_gt.keys()
    for key in wanted.intersection(observed):
        gp_gt, called_gp, p_nonref = gp_values.get(
            key, (None, None, None)
        )
        out[key] = MonopogenEvidence(
            raw_gl_gt=raw_gt.get(key),
            beagle_gp_gt=gp_gt,
            final_phased_gt=final_gt.get(key),
            called_gp=called_gp,
            p_nonref=p_nonref,
            final_phased_present=key in final_gt,
        )
    return out


def add_monopogen_decisions(
    decisions: Dict[Key, Dict[str, MethodDecision]],
    evidence: Mapping[Key, MonopogenEvidence],
) -> None:
    missing = MonopogenEvidence(None, None, None, None, None)
    for key, methods in decisions.items():
        item = evidence.get(key, missing)
        methods[MONO_RELEASED_COMPONENT] = MethodDecision(
            item.final_phased_present, item.final_phased_gt
        )
        methods[MONO_PHASED_GT_DIAGNOSTIC] = MethodDecision(
            item.final_phased_gt in {"0/1", "1/1"}, item.final_phased_gt
        )
        methods[MONO_PAPER_METHOD] = MethodDecision(
            item.raw_gl_gt in {"0/1", "1/1"}
            and item.raw_gl_gt == item.beagle_gp_gt,
            item.beagle_gp_gt,
        )
        methods[MONO_PNONREF_METHOD] = MethodDecision(
            item.beagle_gp_gt in {"0/1", "1/1"}
            and meets_threshold(item.p_nonref, ANCHOR_THRESHOLD),
            item.beagle_gp_gt,
        )


def bool_value(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "t", "yes"})


def exact_rows(frame: pd.DataFrame, sample: str, decisions: Mapping[Key, object]) -> Dict[Key, pd.Series]:
    subset = frame.loc[frame["sample"] == sample]
    out: Dict[Key, pd.Series] = {}
    for _, row in subset.iterrows():
        key = key_from_row(row)
        if key in decisions:
            out[key] = row
    return out


def wilson(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total == 0:
        return math.nan, math.nan
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def metric_row(sample: str, method: str, metric: str, numerator: int, denominator: int) -> dict:
    low, high = wilson(numerator, denominator)
    return {
        "sample": sample,
        "method": method,
        "metric": metric,
        "numerator": numerator,
        "denominator": denominator,
        "value": numerator / denominator if denominator else math.nan,
        "ci95_low": low,
        "ci95_high": high,
    }


def evaluate_methods(
    sample: str,
    decisions: Mapping[Key, Mapping[str, MethodDecision]],
    normal_truth: pd.DataFrame,
    selected_support: pd.DataFrame,
    somatic_truth: pd.DataFrame,
    table_s1: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    normal = exact_rows(normal_truth, sample, decisions)
    support_frame = selected_support.loc[
        (selected_support["sample"] == sample) & (selected_support["wes_pileup_dp"] >= 10)
    ]
    support = exact_rows(support_frame, sample, decisions)
    somatic_frame = somatic_truth.loc[
        (somatic_truth["sample"] == sample) & bool_value(somatic_truth["rna_covered"])
    ]
    somatic = exact_rows(somatic_frame, sample, decisions)

    s1_subset = table_s1.loc[table_s1["sample"] == sample].copy()
    s1_subset = s1_subset.rename(columns={"chrom_hg19": "chrom", "pos_hg19": "pos"})
    curated = exact_rows(s1_subset, sample, decisions)

    method_names = list(next(iter(decisions.values())).keys())
    rows = []
    counts = []
    for method in method_names:
        anchors = {key for key, value in decisions.items() if value[method].anchor}
        counts.append({"sample": sample, "method": method, "anchor_count": len(anchors)})

        normal_anchor = anchors.intersection(normal)
        rows.append(metric_row(sample, method, "normal_wes_anchor_sensitivity", len(normal_anchor), len(normal)))
        gt_correct = sum(
            decisions[key][method].gt == normalize_gt(str(normal[key]["wes_gt"]))
            for key in normal_anchor
        )
        rows.append(metric_row(sample, method, "normal_wes_anchor_gt_accuracy", gt_correct, len(normal_anchor)))

        support_anchor = anchors.intersection(support)
        supported = sum(int(support[key]["high_conf_wes_exact"]) == 1 for key in support_anchor)
        rows.append(metric_row(sample, method, "selected_anchor_normal_wes_support", supported, len(support_anchor)))

        somatic_anchor = anchors.intersection(somatic)
        rows.append(metric_row(sample, method, "wes_somatic_anchor_leakage", len(somatic_anchor), len(somatic)))
        rows.append(metric_row(sample, method, "wes_somatic_retention", len(somatic) - len(somatic_anchor), len(somatic)))

        curated_anchor = anchors.intersection(curated)
        rows.append(metric_row(sample, method, "table_s1_somatic_anchor_leakage", len(curated_anchor), len(curated)))

    return pd.DataFrame(rows), pd.DataFrame(counts)


def evaluate_thresholds(
    sample: str,
    audit: pd.DataFrame,
    mono_evidence: Mapping[Key, MonopogenEvidence],
    normal_truth: pd.DataFrame,
    selected_support: pd.DataFrame,
    somatic_truth: pd.DataFrame,
) -> pd.DataFrame:
    raw_keys = {key_from_row(row) for _, row in audit.iterrows()}
    normal = exact_rows(normal_truth, sample, {key: None for key in raw_keys})
    support_frame = selected_support.loc[
        (selected_support["sample"] == sample) & (selected_support["wes_pileup_dp"] >= 10)
    ]
    support = exact_rows(support_frame, sample, {key: None for key in raw_keys})
    somatic_frame = somatic_truth.loc[
        (somatic_truth["sample"] == sample) & bool_value(somatic_truth["rna_covered"])
    ]
    somatic = exact_rows(somatic_frame, sample, {key: None for key in raw_keys})

    sparcal_ev, sparcal_nonref = {}, {}
    for _, row in audit.iterrows():
        key = key_from_row(row)
        gt = normalize_gt(None if pd.isna(row.beagle_gt) else str(row.beagle_gt))
        raw_called = row.beagle_called_gp
        called = (
            None
            if pd.isna(raw_called) or str(raw_called) == "."
            else float(raw_called)
        )
        sparcal_ev[key] = (gt, called)
        raw_nonref = row.beagle_p_nonref
        p_nonref = (
            None
            if pd.isna(raw_nonref) or str(raw_nonref) == "."
            else float(raw_nonref)
        )
        sparcal_nonref[key] = (gt, p_nonref)

    mono_called = {
        key: (item.beagle_gp_gt, item.called_gp)
        for key, item in mono_evidence.items()
    }
    mono_nonref = {
        key: (item.beagle_gp_gt, item.p_nonref)
        for key, item in mono_evidence.items()
    }

    rows = []
    for source, evidence in (
        ("SPARCAL Beagle called-GP", sparcal_ev),
        ("SPARCAL Beagle P(nonref)", sparcal_nonref),
        ("Monopogen Beagle called-GP", mono_called),
        ("Monopogen Beagle P(nonref)", mono_nonref),
    ):
        for threshold in THRESHOLDS:
            anchors = {
                key for key, (gt, called) in evidence.items()
                if key in raw_keys and gt in {"0/1", "1/1"}
                and meets_threshold(called, threshold)
            }
            normal_anchor = anchors.intersection(normal)
            support_anchor = anchors.intersection(support)
            somatic_anchor = anchors.intersection(somatic)
            gt_correct = sum(
                evidence[key][0] == normalize_gt(str(normal[key]["wes_gt"]))
                for key in normal_anchor
            )
            supported = sum(int(support[key]["high_conf_wes_exact"]) == 1 for key in support_anchor)
            rows.append({
                "sample": sample,
                "source": source,
                "threshold": threshold,
                "anchor_count": len(anchors),
                "normal_wes_n": len(normal),
                "normal_wes_anchor_n": len(normal_anchor),
                "normal_wes_anchor_sensitivity": len(normal_anchor) / len(normal) if normal else math.nan,
                "normal_wes_gt_accuracy": gt_correct / len(normal_anchor) if normal_anchor else math.nan,
                "selected_support_n": len(support_anchor),
                "selected_support_exact_n": supported,
                "selected_support_rate": supported / len(support_anchor) if support_anchor else math.nan,
                "wes_somatic_n": len(somatic),
                "wes_somatic_anchor_n": len(somatic_anchor),
                "wes_somatic_leakage": len(somatic_anchor) / len(somatic) if somatic else math.nan,
            })
    return pd.DataFrame(rows)


def plot_benchmark(metrics: pd.DataFrame, counts: pd.DataFrame, sweep: pd.DataFrame, output: Path) -> None:
    methods = list(counts["method"].drop_duplicates())
    colors = dict(zip(methods, plt.get_cmap("tab10").colors[: len(methods)]))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    markers = {"P4": "o", "P6": "s"}
    for (sample, source), frame in sweep.groupby(["sample", "source"]):
        ax.plot(
            frame["normal_wes_anchor_sensitivity"], frame["selected_support_rate"],
            marker=markers[sample], label=f"{source}, {sample}", alpha=0.85,
        )
        row90 = frame.loc[np.isclose(frame["threshold"], 0.90)].iloc[0]
        ax.annotate("0.90", (row90.normal_wes_anchor_sensitivity, row90.selected_support_rate), fontsize=7)
    ax.set_xlabel("Normal-WES germline sensitivity")
    ax.set_ylabel("Normal-WES support among selected anchors")
    ax.set_title("A  GP-threshold trade-off")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[0, 1]
    x = np.arange(len(methods))
    width = 0.36
    for offset, sample in ((-width / 2, "P4"), (width / 2, "P6")):
        values = [
            int(counts.loc[(counts["sample"] == sample) & (counts["method"] == method), "anchor_count"].iloc[0])
            for method in methods
        ]
        ax.bar(x + offset, values, width, label=sample)
    ax.set_yscale("log")
    ax.set_xticks(x, methods, rotation=28, ha="right", fontsize=8)
    ax.set_ylabel("Hard anchors (log scale)")
    ax.set_title("B  Number of raw candidates excluded as germline")
    ax.legend(frameon=False)

    def grouped_metric(ax, metric: str, title: str, ylabel: str) -> None:
        subset = metrics.loc[metrics["metric"] == metric]
        for offset, sample in ((-width / 2, "P4"), (width / 2, "P6")):
            vals, low, high = [], [], []
            for method in methods:
                row = subset.loc[(subset["sample"] == sample) & (subset["method"] == method)].iloc[0]
                vals.append(row.value)
                low.append(row.value - row.ci95_low)
                high.append(row.ci95_high - row.value)
            ax.bar(x + offset, vals, width, label=sample)
            ax.errorbar(x + offset, vals, yerr=np.array([low, high]), fmt="none", color="black", lw=0.8, capsize=2)
        ax.set_xticks(x, methods, rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, 1.04)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    grouped_metric(axes[1, 0], "normal_wes_anchor_gt_accuracy", "C  Genotype accuracy of germline anchors", "Exact-GT accuracy")
    grouped_metric(axes[1, 1], "wes_somatic_retention", "D  Matched-WES somatic retention (internal)", "Fraction retained downstream")
    axes[1, 1].text(0.01, 0.03, "P4 n=34; P6 n=22 RNA-observable exact alleles", transform=axes[1, 1].transAxes, fontsize=7)

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Beagle evidence routing ablation (WES excluded from production inference)", fontsize=13)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT).as_posix()
    except ValueError:
        return str(resolved)


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    shutil.copystat(source, destination)


def parse_checksum_manifest(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n")
            if not line:
                raise ValueError(f"{path}:{line_number}: empty checksum line")
            try:
                digest, relative = line.split("  ", 1)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: expected SHA256 and relative path"
                ) from exc
            relative_path = Path(relative)
            if (
                len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
                or relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative in entries
            ):
                raise ValueError(f"{path}:{line_number}: invalid checksum entry")
            entries[relative] = digest
    return entries


def verify_checksum_tree(root: Path, manifest_name: str) -> Dict[str, str]:
    manifest = root / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    entries = parse_checksum_manifest(manifest)
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != manifest
    }
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise ValueError(
            f"{manifest}: checksum coverage mismatch; missing={missing}, extra={extra}"
        )
    for relative, expected_digest in entries.items():
        observed = sha256(root / relative)
        if observed != expected_digest:
            raise ValueError(
                f"{manifest}: SHA256 mismatch for {relative}: "
                f"{observed} != {expected_digest}"
            )
    return entries


def validate_monopogen_audit(
    audit_root: Path, selected_runs: Mapping[str, Path]
) -> Dict[str, object]:
    audit_root = audit_root.resolve()
    entries = verify_checksum_tree(audit_root, "OUTPUT_SHA256SUMS.txt")
    provenance_path = audit_root / "PROVENANCE.json"
    input_manifest_path = audit_root / "input_manifest.tsv"
    method_sets_path = audit_root / "method_sets.tsv"
    summary_path = audit_root / "run_summary.tsv"
    provenance = json.loads(provenance_path.read_text())
    if provenance.get("audit_schema_version") != "3.0":
        raise ValueError("corrected Monopogen audit schema 3.0 is required")
    if provenance.get("monopogen_public_commit") != PINNED_MONOPOGEN_COMMIT:
        raise ValueError("Monopogen audit source commit does not match the benchmark pin")
    definitions = provenance.get("method_definitions", {})
    downstream = definitions.get(
        "monopogen_downstream_beagle_record_membership", {}
    )
    if downstream.get("role") != "released_downstream_beagle_component":
        raise ValueError("Monopogen audit lacks the corrected released-code component")
    if "regardless of final phased GT" not in downstream.get("definition", ""):
        raise ValueError("Monopogen released component must be record membership")
    scope_limit = str(provenance.get("scope_limit", ""))
    if "raw read support" not in scope_limit:
        raise ValueError("Monopogen audit must explicitly scope out raw read support")

    manifest_rows: List[Dict[str, str]] = []
    with input_manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "run", "chromosome", "kind", "path", "size_bytes", "sha256"
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{input_manifest_path}: incomplete schema")
        manifest_rows = list(reader)
    if len(manifest_rows) != 264:
        raise ValueError(
            f"{input_manifest_path}: expected 264 inputs, found {len(manifest_rows)}"
        )
    identities = {
        (row["run"], row["chromosome"], row["kind"])
        for row in manifest_rows
    }
    if len(identities) != 264:
        raise ValueError("Monopogen input manifest has duplicate run/chromosome/kind")
    expected_identities = {
        (run, f"chr{chrom}", kind)
        for run in ("P4_rep1", "P4_rep2", "P6_rep1", "P6_rep2")
        for chrom in range(1, 23)
        for kind in ("gl", "gp", "phased")
    }
    if identities != expected_identities:
        raise ValueError("Monopogen input manifest is not the expected 4x22x3 grid")
    for row in manifest_rows:
        input_path = Path(row["path"])
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if input_path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"Monopogen input size changed: {input_path}")
        observed = sha256(input_path)
        if observed != row["sha256"]:
            raise ValueError(f"Monopogen input SHA256 changed: {input_path}")

    audit_runs = {
        str(item["label"]): Path(str(item["path"])).resolve()
        for item in provenance.get("runs", [])
    }
    for label, run_dir in selected_runs.items():
        if audit_runs.get(label) != run_dir.resolve():
            raise ValueError(
                f"benchmark run {label} is not the path verified by audit v3"
            )

    method_sets = pd.read_csv(method_sets_path, sep="\t")
    summaries = pd.read_csv(summary_path, sep="\t")
    required_methods = {
        "monopogen_downstream_beagle_record_membership",
        "monopogen_paper_raw_beagle_gt_concordant_nonref",
        "sparcal_derived_pnonref_threshold",
        "diagnostic_final_phased_nonref_gt",
    }
    if set(method_sets["method"]) != required_methods:
        raise ValueError("Monopogen audit method set is incomplete or unexpected")
    if set(method_sets["run"]) != set(audit_runs):
        raise ValueError("Monopogen audit method rows do not cover all pinned runs")
    if not (summaries["gp_gt_strictly_below_printed_max"] == 0).all():
        raise ValueError("Monopogen GP GT is inconsistent with printed GP argmax")

    selected_manifest_rows = sum(
        row["run"] in selected_runs for row in manifest_rows
    )
    return {
        "path": str(audit_root),
        "audit_schema_version": provenance["audit_schema_version"],
        "public_commit": provenance["monopogen_public_commit"],
        "output_manifest_sha256": sha256(audit_root / "OUTPUT_SHA256SUMS.txt"),
        "output_files_verified": len(entries),
        "input_manifest_sha256": sha256(input_manifest_path),
        "input_files_verified": len(manifest_rows),
        "selected_input_files": selected_manifest_rows,
        "selected_runs": sorted(selected_runs),
        "full_run_method_counts": method_sets[
            ["run", "method", "count"]
        ].to_dict(orient="records"),
        "scope_limit": scope_limit,
    }


def method_definitions() -> List[Dict[str, str]]:
    return [
        {
            "method": LEGACY_PANEL_METHOD,
            "comparison_scope": "same_sparcal_input",
            "role": "historical_broad_panel_anchor",
            "definition": "Any SPARCAL raw exact allele represented by a Beagle GT.",
        },
        {
            "method": LEGACY_CONCORDANCE_METHOD,
            "comparison_scope": "same_sparcal_input",
            "role": "paper_specified_legacy_rule",
            "definition": (
                "Historical panel_concordant membership: raw non-reference GT "
                "equals the Beagle GT and the posterior probability of that "
                "called Beagle GT is at least 0.90."
            ),
        },
        {
            "method": CALLED_GP_METHOD,
            "comparison_scope": "same_sparcal_input",
            "role": "ablation",
            "definition": (
                "Beagle GT is non-reference and its called-genotype GP is at "
                "least 0.90 using absolute tolerance 1e-12."
            ),
        },
        {
            "method": PNONREF_METHOD,
            "comparison_scope": "same_sparcal_input",
            "role": "production_candidate",
            "definition": (
                "Beagle GT is non-reference and GP(0/1)+GP(1/1) is at least "
                "0.90 using absolute tolerance 1e-12."
            ),
        },
        {
            "method": MONO_RELEASED_COMPONENT,
            "comparison_scope": "external_monopogen_execution",
            "role": "released_downstream_beagle_component",
            "definition": (
                "Exact-allele membership in the final phased VCF regardless "
                "of final GT. Full Monopogen somatic calling additionally "
                "uses raw read support, which is outside this Beagle-only audit."
            ),
        },
        {
            "method": MONO_PHASED_GT_DIAGNOSTIC,
            "comparison_scope": "external_monopogen_execution",
            "role": "diagnostic_only",
            "definition": (
                "Final phased GT is non-reference; not the released-code "
                "record-membership contract."
            ),
        },
        {
            "method": MONO_PAPER_METHOD,
            "comparison_scope": "external_monopogen_execution",
            "role": "paper_described_concordance_rule",
            "definition": (
                "Raw PL argmin (lowest standard VCF genotype index on ties) "
                "equals first-pass Beagle GT and is non-reference."
            ),
        },
        {
            "method": MONO_PNONREF_METHOD,
            "comparison_scope": "external_monopogen_execution",
            "role": "sparcal_derived_rule_not_native_monopogen",
            "definition": (
                "First-pass Beagle GT is non-reference and "
                "GP(0/1)+GP(1/1) is at least 0.90 using the router's absolute "
                "tolerance of 1e-12."
            ),
        },
    ]


def validate_same_input_identity(
    current_metrics: pd.DataFrame,
    current_counts: pd.DataFrame,
    trusted_roots: Sequence[Path],
) -> List[Dict[str, object]]:
    current_metric_columns = [
        "sample", "method", "metric", "numerator", "denominator", "value",
        "ci95_low", "ci95_high",
    ]
    current_count_columns = ["sample", "method", "anchor_count"]
    current_m = current_metrics[current_metric_columns].copy()
    current_c = current_counts[current_count_columns].copy()
    results: List[Dict[str, object]] = []

    for root in trusted_roots:
        root = root.resolve()
        metric_path = root / "same_input_method_metrics.tsv"
        count_path = root / "same_input_anchor_counts.tsv"
        if not metric_path.is_file():
            metric_path = root / "method_metrics.tsv"
        if not count_path.is_file():
            count_path = root / "anchor_counts.tsv"
        reference_m = pd.read_csv(metric_path, sep="\t")
        reference_c = pd.read_csv(count_path, sep="\t")
        reference_m["method"] = reference_m["method"].map(TRUSTED_LABEL_MAP)
        reference_c["method"] = reference_c["method"].map(TRUSTED_LABEL_MAP)
        reference_m = reference_m.loc[reference_m["method"].notna(), current_metric_columns]
        reference_c = reference_c.loc[reference_c["method"].notna(), current_count_columns]

        sort_m = ["sample", "method", "metric"]
        sort_c = ["sample", "method"]
        left_m = current_m.sort_values(sort_m).reset_index(drop=True)
        right_m = reference_m.sort_values(sort_m).reset_index(drop=True)
        left_c = current_c.sort_values(sort_c).reset_index(drop=True)
        right_c = reference_c.sort_values(sort_c).reset_index(drop=True)
        if left_m[sort_m].to_dict("records") != right_m[sort_m].to_dict("records"):
            raise ValueError(f"same-input metric keys differ from {root}")
        if left_c[sort_c].to_dict("records") != right_c[sort_c].to_dict("records"):
            raise ValueError(f"same-input count keys differ from {root}")
        for column in ("numerator", "denominator"):
            if not np.array_equal(left_m[column].to_numpy(), right_m[column].to_numpy()):
                raise ValueError(f"same-input {column} differs from {root}")
        for column in ("value", "ci95_low", "ci95_high"):
            if not np.allclose(
                left_m[column].to_numpy(float),
                right_m[column].to_numpy(float),
                rtol=0.0,
                atol=1e-15,
                equal_nan=True,
            ):
                raise ValueError(f"same-input {column} differs from {root}")
        if not np.array_equal(
            left_c["anchor_count"].to_numpy(),
            right_c["anchor_count"].to_numpy(),
        ):
            raise ValueError(f"same-input anchor counts differ from {root}")
        results.append(
            {
                "trusted_root": str(root),
                "metric_file": str(metric_path),
                "metric_file_sha256": sha256(metric_path),
                "count_file": str(count_path),
                "count_file_sha256": sha256(count_path),
                "metric_rows_verified": len(left_m),
                "count_rows_verified": len(left_c),
                "label_map": dict(TRUSTED_LABEL_MAP),
                "status": "exact_integer_and_1e-15_float_identity",
            }
        )
    return results


def validate_external_pnonref_sweep(
    counts: pd.DataFrame, sweep: pd.DataFrame
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for sample in SAMPLES:
        main = counts.loc[
            (counts["sample"] == sample)
            & (counts["method"] == MONO_PNONREF_METHOD),
            "anchor_count",
        ]
        swept = sweep.loc[
            (sweep["sample"] == sample)
            & (sweep["source"] == "Monopogen Beagle P(nonref)")
            & np.isclose(sweep["threshold"], ANCHOR_THRESHOLD),
            "anchor_count",
        ]
        if len(main) != 1 or len(swept) != 1:
            raise ValueError(f"missing unique Monopogen P(nonref) count for {sample}")
        main_count, sweep_count = int(main.iloc[0]), int(swept.iloc[0])
        if main_count != sweep_count:
            raise ValueError(
                f"Monopogen derived P(nonref) mismatch for {sample}: "
                f"main={main_count}, sweep={sweep_count}"
            )
        results.append(
            {
                "sample": sample,
                "method": MONO_PNONREF_METHOD,
                "threshold": ANCHOR_THRESHOLD,
                "main_table_anchor_count": main_count,
                "threshold_sweep_anchor_count": sweep_count,
                "status": "identical",
            }
        )
    return results


def package_versions() -> Dict[str, str]:
    versions = {}
    for package in ("matplotlib", "numpy", "pandas"):
        versions[package] = importlib_metadata.version(package)
    return versions


def write_dataframe_exclusive(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep="\t", index=False, mode="x")


def write_json_exclusive(payload: Mapping[str, object], path: Path) -> None:
    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_code_and_input_hashes(
    output_dir: Path, artifacts: Mapping[str, Path]
) -> Tuple[Path, Dict[str, str]]:
    hashes = {label: sha256(path) for label, path in artifacts.items()}
    output = output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    with output.open("x") as handle:
        for label in sorted(hashes):
            handle.write(f"{hashes[label]}  {label}\n")
    return output, hashes


def write_output_checksums(
    output_dir: Path, future_files: Mapping[str, bytes]
) -> Path:
    checksum_path = output_dir / "OUTPUT_SHA256SUMS.txt"
    if checksum_path.exists():
        raise FileExistsError(checksum_path)
    entries = {
        path.relative_to(output_dir).as_posix(): sha256(path)
        for path in output_dir.rglob("*")
        if path.is_file() and path != checksum_path
    }
    for relative, payload in future_files.items():
        if relative in entries or (output_dir / relative).exists():
            raise FileExistsError(output_dir / relative)
        entries[relative] = sha256_bytes(payload)
    with checksum_path.open("x") as handle:
        for relative in sorted(entries):
            handle.write(f"{entries[relative]}  {relative}\n")
    return checksum_path


def locate_monopogen_runs(root: Path) -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    for sample in SAMPLES:
        candidates = (root / f"{sample}_rep1", root / f"Monopogen_{sample}_rep1")
        matches = [candidate.resolve() for candidate in candidates if candidate.is_dir()]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one Monopogen rep1 run for {sample} under {root}; "
                f"found {matches}"
            )
        runs[f"{sample}_rep1"] = matches[0]
    return runs

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--route-root", type=Path,
        default=PROJECT
        / "data/experiments/beagle_evidence_v2_4_pnonref090_hardened_20260912",
    )
    parser.add_argument(
        "--old-root", type=Path,
        default=PROJECT
        / "data/experiments/beagle_panel_routing_gp090_20260911_v3",
    )
    parser.add_argument("--monopogen-root", type=Path, default=PROJECT)
    parser.add_argument(
        "--monopogen-audit-root",
        type=Path,
        default=DEFAULT_MONOPOGEN_AUDIT,
    )
    parser.add_argument(
        "--trusted-benchmark-root",
        action="append",
        type=Path,
        help=(
            "Prior benchmark used only for same-input numerical identity; "
            "repeatable (default: trusted v4 and v8)."
        ),
    )
    args = parser.parse_args(argv)
    trusted_roots = tuple(
        args.trusted_benchmark_root or DEFAULT_TRUSTED_BENCHMARKS
    )

    started_wall = time.monotonic()
    started_at = datetime.now(timezone.utc)
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite existing output directory: {output_dir}"
        )

    script_path = Path(__file__).resolve()
    test_path = PROJECT / "tests" / "test_beagle_routing.py"
    code_sources = {
        "beagle_evidence_benchmark.py": script_path,
        "beagle_evidence_router.py": ROUTER_PATH.resolve(),
        "test_beagle_routing.py": test_path.resolve(),
    }
    for source in code_sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_dir = output_dir / "code_snapshot"
    snapshot_hashes: Dict[str, str] = {}
    for name, source in code_sources.items():
        destination = snapshot_dir / name
        copy_exclusive(source, destination)
        snapshot_hashes[name] = sha256(destination)
    snapshot_completed_at = datetime.now(timezone.utc)

    monopogen_runs = locate_monopogen_runs(args.monopogen_root.resolve())
    audit_validation = validate_monopogen_audit(
        args.monopogen_audit_root, monopogen_runs
    )

    normal_path = (
        PROJECT
        / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz"
    )
    support_path = (
        PROJECT
        / "data/germline_concordance_2026-08-23/sparcal_call_details.csv.gz"
    )
    somatic_path = (
        PROJECT
        / "data/somatic_evidence_2026-07-28/wes_confirmed_full_accounting.csv"
    )
    table_s1_path = (
        PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv"
    )
    direct_inputs: Dict[str, Path] = {
        display_path(normal_path): normal_path,
        display_path(support_path): support_path,
        display_path(somatic_path): somatic_path,
        display_path(table_s1_path): table_s1_path,
    }
    sample_inputs: Dict[str, Dict[str, Path]] = {}
    for sample in SAMPLES:
        section = f"{sample}_sec1"
        new_audit = (
            args.route_root / section / "beagle_evidence_audit.tsv.gz"
        ).resolve()
        old_audit = (
            args.old_root
            / section
            / "output_VCFs/beagle/baseQ0mapQ0/panel_routing.tsv.gz"
        ).resolve()
        sample_inputs[sample] = {
            "route_audit": new_audit,
            "legacy_audit": old_audit,
        }
        direct_inputs[display_path(new_audit)] = new_audit
        direct_inputs[display_path(old_audit)] = old_audit
    for path in direct_inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    normal = pd.read_csv(normal_path)
    normal = normal.loc[
        (normal["detected"] == 1) & (normal["rna_dp"] >= 1)
    ].copy()
    support = pd.read_csv(support_path)
    somatic = pd.read_csv(somatic_path)
    table_s1 = pd.read_csv(table_s1_path, sep="\t")

    metric_frames, count_frames, sweep_frames = [], [], []
    sample_provenance: Dict[str, Dict[str, object]] = {}
    for sample in SAMPLES:
        inputs = sample_inputs[sample]
        decisions, audit = decisions_from_audits(
            inputs["route_audit"], inputs["legacy_audit"]
        )
        mono_dir = monopogen_runs[f"{sample}_rep1"]
        mono_evidence = read_monopogen_evidence(mono_dir, decisions)
        add_monopogen_decisions(decisions, mono_evidence)
        metrics, counts = evaluate_methods(
            sample, decisions, normal, support, somatic, table_s1
        )
        metric_frames.append(metrics)
        count_frames.append(counts)
        sweep_frames.append(
            evaluate_thresholds(
                sample, audit, mono_evidence, normal, support, somatic
            )
        )
        sample_provenance[sample] = {
            "route_audit": str(inputs["route_audit"]),
            "legacy_audit": str(inputs["legacy_audit"]),
            "monopogen_run": str(mono_dir),
            "raw_candidate_count": len(decisions),
            "monopogen_evidence_exact_overlap": len(mono_evidence),
            "monopogen_released_code_component": (
                "exact-allele record membership in final phased VCF "
                "regardless of GT; full read-support logic is scoped out"
            ),
            "monopogen_final_phased_nonref_gt": "diagnostic only",
            "monopogen_paper_method": (
                "non-reference raw PL argmin equals first-pass Beagle GT"
            ),
            "monopogen_derived_pnonref": (
                "non-reference first-pass Beagle GT and "
                "P(nonref)>=0.90 with abs_tol=1e-12"
            ),
        }

    metrics = pd.concat(metric_frames, ignore_index=True)
    counts = pd.concat(count_frames, ignore_index=True)
    sweep = pd.concat(sweep_frames, ignore_index=True)
    metrics["comparison_scope"] = np.where(
        metrics["method"].isin(SAME_INPUT_METHODS),
        "same_sparcal_input", "external_monopogen_execution",
    )
    counts["comparison_scope"] = np.where(
        counts["method"].isin(SAME_INPUT_METHODS),
        "same_sparcal_input", "external_monopogen_execution",
    )
    sweep["comparison_scope"] = np.where(
        sweep["source"].str.startswith("SPARCAL"),
        "same_sparcal_input", "external_monopogen_execution",
    )
    if set(metrics["method"]) != set(SAME_INPUT_METHODS) | set(EXTERNAL_METHODS):
        raise ValueError("benchmark methods differ from the frozen method contract")

    same_metrics = metrics.loc[
        metrics["comparison_scope"] == "same_sparcal_input"
    ].copy()
    same_counts = counts.loc[
        counts["comparison_scope"] == "same_sparcal_input"
    ].copy()
    external_metrics = metrics.loc[
        metrics["comparison_scope"] == "external_monopogen_execution"
    ].copy()
    external_counts = counts.loc[
        counts["comparison_scope"] == "external_monopogen_execution"
    ].copy()

    historical_validation = validate_same_input_identity(
        same_metrics, same_counts, trusted_roots
    )
    sweep_validation = validate_external_pnonref_sweep(counts, sweep)
    validation = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "same_input_historical_identity": historical_validation,
        "external_derived_pnonref_main_vs_sweep": sweep_validation,
        "monopogen_corrected_audit": audit_validation,
        "status": "all_checks_passed",
    }

    write_dataframe_exclusive(metrics, output_dir / "method_metrics.tsv")
    write_dataframe_exclusive(counts, output_dir / "anchor_counts.tsv")
    write_dataframe_exclusive(sweep, output_dir / "gp_threshold_sweep.tsv")
    write_dataframe_exclusive(
        same_metrics, output_dir / "same_input_method_metrics.tsv"
    )
    write_dataframe_exclusive(
        same_counts, output_dir / "same_input_anchor_counts.tsv"
    )
    write_dataframe_exclusive(
        external_metrics, output_dir / "external_monopogen_overlap_metrics.tsv"
    )
    write_dataframe_exclusive(
        external_counts,
        output_dir / "external_monopogen_overlap_anchor_counts.tsv",
    )
    write_dataframe_exclusive(
        pd.DataFrame(method_definitions()), output_dir / "METHOD_DEFINITIONS.tsv"
    )
    write_json_exclusive(validation, output_dir / "VALIDATION.json")
    plot_benchmark(
        same_metrics,
        same_counts,
        sweep.loc[sweep["comparison_scope"] == "same_sparcal_input"],
        output_dir / "beagle_routing_ablation",
    )

    for root_info in historical_validation:
        metric_path = Path(str(root_info["metric_file"]))
        count_path = Path(str(root_info["count_file"]))
        direct_inputs[display_path(metric_path)] = metric_path
        direct_inputs[display_path(count_path)] = count_path
    audit_root = args.monopogen_audit_root.resolve()
    for name in (
        "OUTPUT_SHA256SUMS.txt",
        "PROVENANCE.json",
        "input_manifest.tsv",
        "method_sets.tsv",
        "run_summary.tsv",
    ):
        path = audit_root / name
        direct_inputs[display_path(path)] = path
    hash_artifacts = {
        display_path(source): source for source in code_sources.values()
    }
    hash_artifacts.update(direct_inputs)
    code_input_manifest, code_input_hashes = write_code_and_input_hashes(
        output_dir, hash_artifacts
    )

    for name, source in code_sources.items():
        if sha256(source) != snapshot_hashes[name]:
            raise RuntimeError(f"source changed after pre-computation snapshot: {source}")

    completed_at = datetime.now(timezone.utc)
    completion_payload = (
        json.dumps(
            {
                "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
                "completed_at_utc": completed_at.isoformat(),
                "status": "complete",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    provenance = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "started_at_utc": started_at.isoformat(),
        "code_snapshot_completed_at_utc": snapshot_completed_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
        "runtime_seconds": time.monotonic() - started_wall,
        "command": [str(script_path), *(sys.argv[1:] if argv is None else argv)],
        "working_directory": str(Path.cwd().resolve()),
        "software_versions": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "python_packages": package_versions(),
        },
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "matched_wes_role": (
            "internal development evidence, not untouched held-out evaluation"
        ),
        "comparison_scopes": {
            "same_sparcal_input": (
                "direct routing ablation on identical SPARCAL raw and Beagle evidence"
            ),
            "external_monopogen_execution": (
                "exact-overlap audit only; different mpileup input contract"
            ),
        },
        "method_definitions": method_definitions(),
        "monopogen_public_source_commit": PINNED_MONOPOGEN_COMMIT,
        "monopogen_beagle_source": f"{MONOPOGEN_SOURCE_ROOT}/src/Monopogen.py#L96-L120",
        "monopogen_downstream_source": f"{MONOPOGEN_SOURCE_ROOT}/src/somatic.py#L108-L156",
        "monopogen_audit_validation": audit_validation,
        "monopogen_input_hash_contract": (
            "All 264 source VCFs were re-hashed against corrected audit v3. "
            "The benchmark uses the 132 P4_rep1/P6_rep1 files; their hashes "
            "are pinned through the verified v3 input_manifest.tsv."
        ),
        "samples": sample_provenance,
        "historical_identity_validation": historical_validation,
        "external_threshold_validation": sweep_validation,
        "code_snapshot_created_before_computation": True,
        "code_snapshot_sha256": snapshot_hashes,
        "code_and_primary_input_sha256": code_input_hashes,
        "code_and_input_manifest": code_input_manifest.name,
        "output_checksum_manifest": "OUTPUT_SHA256SUMS.txt",
        "output_checksum_scope": (
            "Every regular output file except OUTPUT_SHA256SUMS.txt itself; "
            "the RUN_COMPLETE digest was calculated before and the marker "
            "was written last."
        ),
        "expected_run_complete_sha256": sha256_bytes(completion_payload),
    }
    write_json_exclusive(provenance, output_dir / "PROVENANCE.json")
    write_output_checksums(
        output_dir, {"RUN_COMPLETE": completion_payload}
    )
    with (output_dir / "RUN_COMPLETE").open("xb") as handle:
        handle.write(completion_payload)
    verify_checksum_tree(output_dir, "OUTPUT_SHA256SUMS.txt")

    print(counts.to_string(index=False))
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
