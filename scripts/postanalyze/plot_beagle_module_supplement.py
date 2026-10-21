#!/usr/bin/env python3
"""Render the final Beagle-v2.3 supplementary validation dossier.

This program consumes completed, immutable assessment dossiers.  It never
recomputes a biological benchmark or clustering result.  All input contracts
and checksum ledgers are validated before a new output directory is created.

Interpretation boundaries are part of the data contract:

* matched WES is internal development evaluation, never a production input;
* DLPFC STAGATE -> fixed-EEE mclust is functional no-harm, not caller accuracy;
* the primary DLPFC interface is ``all_panel_present``; the P(nonref)>=0.90
  anchor matrix is a secondary diagnostic and is not plotted as the primary arm;
* Monopogen is an audited Beagle mechanism from a different input universe and
  is never placed on a same-universe performance axis with SPARCAL.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as font_manager  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ft2font import FT2Font  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
DEFAULT_ARIAL_DIR = PROJECT.parent / "SPARCAL_pnas_2026/fonts/arial"
REQUIRED_ARIAL_STYLES = {"Regular", "Bold", "Italic", "Bold Italic"}
SAMPLES = ("P4", "P6")
SECTIONS = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)
DONOR_BY_SECTION = {
    **{section: "Br5292" for section in SECTIONS[:4]},
    **{section: "Br5595" for section in SECTIONS[4:8]},
    **{section: "Br8100" for section in SECTIONS[8:]},
}
SAMPLE_COLORS = {"P4": "#0072B2", "P6": "#D55E00"}
DONOR_COLORS = {
    "Br5292": "#0072B2", "Br5595": "#D55E00", "Br8100": "#009E73",
}

EVIDENCE_METHODS = (
    "Legacy any panel record",
    "SPARCAL legacy paper rule (raw nonref=Beagle GT; called-GP>=0.90)",
    "SPARCAL called-GP>=0.90 ablation",
    "SPARCAL P(nonref)>=0.90",
)
EVIDENCE_METHOD_LABELS = {
    EVIDENCE_METHODS[0]: "Historical: any panel record",
    EVIDENCE_METHODS[1]: (
        "Legacy paper: raw GT nonref; raw GT = Beagle GT;\n"
        "called-GP >= 0.90"
    ),
    EVIDENCE_METHODS[2]: "Called-GP >= 0.90 ablation",
    EVIDENCE_METHODS[3]: "P(nonref) = GP(0/1) + GP(1/1) >= 0.90",
}
PRODUCTION_RULE = EVIDENCE_METHODS[3]
EVIDENCE_ALL_METRICS = (
    "normal_wes_anchor_sensitivity",
    "normal_wes_anchor_gt_accuracy",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
    "wes_somatic_retention",
    "table_s1_somatic_anchor_leakage",
)
EVIDENCE_PLOT_METRICS = (
    "normal_wes_anchor_sensitivity",
    "normal_wes_anchor_gt_accuracy",
    "wes_somatic_retention",
)
QUALITY_CONFIGS = ("baseQ13mapQ20", "baseQ20mapQ20")
QUALITY_PLOT_METRICS = (
    "raw_candidate_normal_wes_recall",
    "normal_wes_end_to_end_anchor_recall",
    "wes_somatic_end_to_end_retention",
)
REPLICATION_METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
ENSEMBLE_METHODS = ("ensemble_mean", "ensemble_median")
ENSEMBLE_REPRESENTATIONS = ("single_seed", *ENSEMBLE_METHODS)
STRICT_PROTOCOL = "sparcal_beagle_v2_3_final_dlpfc_regression_v1"
STRICT_METHOD = "beagle_panel_present_compat"
STRICT_GATES = {
    "mean_delta_ari_nonnegative",
    "minimum_donor_mean_delta_ari_above_minus_0.01",
    "no_section_mean_ari_drop_gt_0.05",
    "nmi_not_worse_by_more_than_0.01",
    "at_least_9_paired_runs_each",
    "no_candidate_only_failures",
    "candidate_run_success_not_lower_than_legacy",
    "candidate_used_requested_mclust_model_for_all_runs",
}
MONO_RUNS = ("P4_rep1", "P4_rep2", "P6_rep1", "P6_rep2")
MONO_METHODS = {
    "monopogen_downstream_beagle_record_membership",
    "monopogen_paper_raw_beagle_gt_concordant_nonref",
    "sparcal_derived_pnonref_threshold",
    "diagnostic_final_phased_nonref_gt",
}
MONO_COMMIT = "2c0a4b0868cb86a77f196cba29b4d4dd3ce482f6"

EVIDENCE_LEDGER_MEMBERS = {
    "CODE_AND_INPUT_SHA256SUMS.txt", "METHOD_DEFINITIONS.tsv", "PROVENANCE.json",
    "RUN_COMPLETE", "VALIDATION.json", "anchor_counts.tsv",
    "beagle_routing_ablation.pdf", "beagle_routing_ablation.png",
    "code_snapshot/beagle_evidence_benchmark.py",
    "code_snapshot/beagle_evidence_router.py",
    "code_snapshot/test_beagle_routing.py",
    "external_monopogen_overlap_anchor_counts.tsv",
    "external_monopogen_overlap_metrics.tsv", "gp_threshold_sweep.tsv",
    "method_metrics.tsv", "same_input_anchor_counts.tsv",
    "same_input_method_metrics.tsv",
}
MONO_LEDGER_MEMBERS = {
    "PROVENANCE.json", "RUN_COMPLETE",
    "code_snapshot/monopogen_beagle_contract_audit.py",
    "code_snapshot/test_monopogen_beagle_contract_audit.py",
    "input_manifest.tsv", "method_intersections.tsv", "method_sets.tsv",
    "run_summary.tsv", "transition_counts.tsv",
}
REPLICATION_LEDGER_MEMBERS = {
    "replicated_configuration_metrics_long.tsv",
    "replicated_configuration_summary.tsv",
    "replicated_pairwise_stability.tsv", "replication_gate_results.tsv",
    "FINAL_BASE_CONFIGURATION.json", "PROVENANCE.json",
    "CODE_AND_INPUT_SHA256SUMS.txt",
}
ENSEMBLE_LEDGER_MEMBERS = {
    "ensemble_gate_results.tsv", "FINAL_BEAGLE_REPRESENTATION.json",
    "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
}
STRICT_LEDGER_MEMBERS = {
    "NONINFERIORITY_RESULT.json", "STRICT_RUN_VALIDATION.json",
    "clustering_metrics_long.tsv", "donor_paired_deltas.tsv",
    "method_success_counts.tsv", "section_paired_deltas.tsv",
    "seed_paired_deltas.tsv",
}


@dataclass(frozen=True)
class EvidenceData:
    metrics: pd.DataFrame
    counts: pd.DataFrame
    provenance: Mapping[str, object]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class QualityData:
    metrics: pd.DataFrame
    pairwise: pd.DataFrame
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class ReplicationData:
    base: str
    seeds: tuple[int, ...]
    stability: pd.DataFrame
    metrics: pd.DataFrame
    selected_gate: Mapping[str, object]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class EnsembleData:
    base: str
    representation: str
    gates: pd.DataFrame
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class DlpfcData:
    sections: pd.DataFrame
    donors: pd.DataFrame
    result: Mapping[str, object]
    semantic_contract: Mapping[str, object]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class MonopogenData:
    summary: pd.DataFrame
    provenance: Mapping[str, object]
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class FigureInputs:
    evidence: EvidenceData
    quality: QualityData
    replication: ReplicationData
    ensemble: EnsembleData
    dlpfc: DlpfcData
    monopogen: MonopogenData

    @property
    def paths(self) -> tuple[Path, ...]:
        ordered = (
            *self.evidence.paths, *self.quality.paths, *self.replication.paths,
            *self.ensemble.paths, *self.dlpfc.paths, *self.monopogen.paths,
        )
        return tuple(dict.fromkeys(Path(path).resolve() for path in ordered))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def require_files(root: Path, names: Sequence[str], label: str) -> list[Path]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} root is absent: {root}")
    paths = [root / name for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete: {missing}")
    return paths


def require_marker(path: Path, label: str, *, marker_format: str = "plain") -> None:
    """Validate the package-specific marker format used by its frozen producer."""
    marker = path.read_text()
    if marker_format == "plain":
        valid = marker.strip() == "complete"
    elif marker_format == "json":
        try:
            payload = json.loads(marker)
        except json.JSONDecodeError:
            payload = None
        valid = isinstance(payload, dict) and payload.get("status") == "complete"
    elif marker_format == "monopogen_v3":
        valid = bool(re.fullmatch(
            r"Monopogen Beagle contract audit completed successfully at "
            r"\d{4}-\d{2}-\d{2}T[^\n]+\.\n?",
            marker,
        ))
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"unknown completion marker format: {marker_format}")
    if not valid:
        raise ValueError(f"{label} has an invalid completion marker: {path}")


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], label: str, *, exact: bool = False,
) -> None:
    required = set(columns)
    observed = set(frame.columns)
    if exact and observed != required:
        raise ValueError(
            f"{label} column set differs: missing={sorted(required-observed)}, "
            f"extra={sorted(observed-required)}"
        )
    missing = required - observed
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def strict_bool(value: object, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{label} is not a strict boolean: {value!r}")


def parse_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} is not an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer: {value!r}") from error
    if str(parsed) != str(value) or parsed < minimum:
        raise ValueError(f"{label} is not a canonical integer: {value!r}")
    return parsed


def finite_float(value: object, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} is not finite: {value!r}")
    return parsed


def validate_fraction_rows(frame: pd.DataFrame, label: str) -> None:
    require_columns(
        frame,
        ("numerator", "denominator", "value", "ci95_low", "ci95_high"),
        label,
    )
    values = frame[
        ["numerator", "denominator", "value", "ci95_low", "ci95_high"]
    ].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError(f"{label} contains non-finite fractions")
    if (
        (values["numerator"] < 0).any()
        or (values["denominator"] <= 0).any()
        or (values["numerator"] > values["denominator"]).any()
    ):
        raise ValueError(f"{label} contains invalid counts")
    expected = values["numerator"] / values["denominator"]
    if not np.allclose(expected, values["value"], rtol=0, atol=1e-12):
        raise ValueError(f"{label} values do not equal numerator/denominator")
    if (
        (values["ci95_low"] < -1e-12).any()
        or (values["ci95_high"] > 1 + 1e-12).any()
        or (values["ci95_low"] > values["value"] + 1e-12).any()
        or (values["ci95_high"] < values["value"] - 1e-12).any()
    ):
        raise ValueError(f"{label} confidence limits are invalid")


def verify_identity(record: object, expected: Path, label: str) -> None:
    if not isinstance(record, Mapping):
        raise TypeError(f"{label} identity is not an object")
    expected = Path(expected).resolve()
    stat = expected.stat()
    observed = {
        "path": str(expected), "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "sha256": sha256(expected),
    }
    for key, value in observed.items():
        if record.get(key) != value:
            raise ValueError(f"{label} identity mismatch for {key}: {expected}")


def verify_ledger(
    path: Path, root: Path, expected_members: set[str] | None, label: str,
) -> list[Path]:
    root = Path(root).resolve()
    records: dict[str, Path] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or not re.fullmatch(r"[0-9a-f]{64}", parts[0]):
            raise ValueError(f"malformed {label} row {line_number}")
        digest, relative = parts
        relative_path = Path(relative)
        if relative_path.is_absolute() or relative in records:
            raise ValueError(f"absolute/duplicate member in {label}: {relative}")
        member = (root / relative_path).resolve()
        try:
            member.relative_to(root)
        except ValueError as error:
            raise ValueError(f"member escapes {label} root: {relative}") from error
        if not member.is_file() or sha256(member) != digest:
            raise ValueError(f"missing or checksum-mismatched {label} member: {relative}")
        records[relative] = member
    if expected_members is not None and set(records) != set(expected_members):
        raise ValueError(
            f"{label} member set differs: missing={sorted(expected_members-set(records))}, "
            f"extra={sorted(set(records)-expected_members)}"
        )
    return [records[name] for name in sorted(records)]


def load_evidence(root: Path) -> EvidenceData:
    names = (
        "same_input_method_metrics.tsv", "same_input_anchor_counts.tsv",
        "gp_threshold_sweep.tsv", "METHOD_DEFINITIONS.tsv", "PROVENANCE.json",
        "VALIDATION.json", "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE",
    )
    paths = require_files(root, names, "v11 same-input evidence")
    root = Path(root).resolve()
    require_marker(root / "RUN_COMPLETE", "v11 same-input evidence", marker_format="json")
    ledger_paths = verify_ledger(
        root / "OUTPUT_SHA256SUMS.txt", root, EVIDENCE_LEDGER_MEMBERS,
        "v11 output ledger",
    )
    provenance = read_json(root / "PROVENANCE.json")
    validation = read_json(root / "VALIDATION.json")
    if (
        provenance.get("benchmark_schema_version") != "11.0"
        or validation.get("benchmark_schema_version") != "11.0"
        or validation.get("status") != "all_checks_passed"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("matched_wes_role")
        != "internal development evidence, not untouched held-out evaluation"
        or provenance.get("code_snapshot_created_before_computation") is not True
    ):
        raise ValueError("v11 evidence status, WES role, or freeze contract is invalid")

    definitions = pd.read_csv(root / "METHOD_DEFINITIONS.tsv", sep="\t")
    require_columns(
        definitions, ("method", "comparison_scope", "role", "definition"),
        "v11 method definitions", exact=True,
    )
    expected_roles = {
        EVIDENCE_METHODS[0]: ("same_sparcal_input", "historical_broad_panel_anchor"),
        EVIDENCE_METHODS[1]: ("same_sparcal_input", "paper_specified_legacy_rule"),
        EVIDENCE_METHODS[2]: ("same_sparcal_input", "ablation"),
        EVIDENCE_METHODS[3]: ("same_sparcal_input", "production_candidate"),
        "Monopogen released-code phased-record membership component": (
            "external_monopogen_execution", "released_downstream_beagle_component"
        ),
        "Monopogen final-phased nonref-GT diagnostic": (
            "external_monopogen_execution", "diagnostic_only"
        ),
        "Monopogen paper concordant nonref": (
            "external_monopogen_execution", "paper_described_concordance_rule"
        ),
        "Monopogen P(nonref)>=0.90 (derived)": (
            "external_monopogen_execution", "sparcal_derived_rule_not_native_monopogen"
        ),
    }
    observed_roles = {
        str(row.method): (str(row.comparison_scope), str(row.role))
        for row in definitions.itertuples(index=False)
    }
    if observed_roles != expected_roles or len(definitions) != len(expected_roles):
        raise ValueError("v11 method definitions are not the exact frozen method set")

    metrics = pd.read_csv(root / "same_input_method_metrics.tsv", sep="\t")
    require_columns(
        metrics,
        ("sample", "method", "metric", "numerator", "denominator", "value",
         "ci95_low", "ci95_high", "comparison_scope"),
        "v11 same-input metrics", exact=True,
    )
    expected_grid = set(itertools.product(SAMPLES, EVIDENCE_METHODS, EVIDENCE_ALL_METRICS))
    observed_grid = set(
        metrics[["sample", "method", "metric"]].itertuples(index=False, name=None)
    )
    if observed_grid != expected_grid or len(metrics) != len(expected_grid):
        raise ValueError("v11 same-input metric grid is not exact")
    if set(metrics["comparison_scope"]) != {"same_sparcal_input"}:
        raise ValueError("v11 same-input table mixes external input universes")
    validate_fraction_rows(metrics, "v11 same-input metrics")

    counts = pd.read_csv(root / "same_input_anchor_counts.tsv", sep="\t")
    require_columns(
        counts, ("sample", "method", "anchor_count", "comparison_scope"),
        "v11 anchor counts", exact=True,
    )
    expected_counts = set(itertools.product(SAMPLES, EVIDENCE_METHODS))
    observed_counts = set(
        counts[["sample", "method"]].itertuples(index=False, name=None)
    )
    if observed_counts != expected_counts or len(counts) != len(expected_counts):
        raise ValueError("v11 anchor-count grid is not exact")
    counts["anchor_count"] = pd.to_numeric(counts["anchor_count"], errors="raise").astype(int)
    if (counts["anchor_count"] <= 0).any():
        raise ValueError("v11 anchor counts must be positive")

    sweep = pd.read_csv(root / "gp_threshold_sweep.tsv", sep="\t")
    require_columns(
        sweep,
        ("sample", "source", "threshold", "anchor_count",
         "normal_wes_anchor_sensitivity", "normal_wes_gt_accuracy",
         "wes_somatic_leakage", "comparison_scope"),
        "v11 threshold sweep",
    )
    crosswalk = {
        "SPARCAL Beagle called-GP": EVIDENCE_METHODS[2],
        "SPARCAL Beagle P(nonref)": EVIDENCE_METHODS[3],
    }
    at_090 = sweep.loc[
        sweep["source"].isin(crosswalk)
        & np.isclose(pd.to_numeric(sweep["threshold"]), 0.90, rtol=0, atol=1e-12)
        & sweep["comparison_scope"].eq("same_sparcal_input")
    ]
    if len(at_090) != 4:
        raise ValueError("v11 sweep lacks the exact SPARCAL score x sample 0.90 grid")
    metric_lookup = metrics.set_index(["sample", "method", "metric"])
    count_lookup = counts.set_index(["sample", "method"])
    for row in at_090.itertuples(index=False):
        method = crosswalk[str(row.source)]
        if int(row.anchor_count) != int(count_lookup.loc[(row.sample, method), "anchor_count"]):
            raise ValueError("v11 threshold/count cross-check failed")
        comparisons = {
            "normal_wes_anchor_sensitivity": float(row.normal_wes_anchor_sensitivity),
            "normal_wes_anchor_gt_accuracy": float(row.normal_wes_gt_accuracy),
            "wes_somatic_retention": 1 - float(row.wes_somatic_leakage),
        }
        for metric, expected_value in comparisons.items():
            observed = float(metric_lookup.loc[(row.sample, method, metric), "value"])
            if not math.isclose(expected_value, observed, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"v11 threshold/metric cross-check failed: {row.sample}/{method}/{metric}")

    plotted = metrics.loc[metrics["metric"].isin(EVIDENCE_PLOT_METRICS)].merge(
        counts[["sample", "method", "anchor_count"]],
        on=["sample", "method"], validate="many_to_one",
    )
    plotted["method_label"] = plotted["method"].map(EVIDENCE_METHOD_LABELS)
    plotted["selected"] = plotted["method"].eq(PRODUCTION_RULE)
    return EvidenceData(plotted, counts, provenance, tuple(dict.fromkeys([*paths, *ledger_paths])))


def parse_quality(config: str) -> tuple[int, int]:
    match = re.fullmatch(r"baseQ(\d+)mapQ(\d+)", str(config))
    if not match:
        raise ValueError(f"invalid BQ/MQ configuration: {config}")
    return int(match.group(1)), int(match.group(2))


def load_quality(root: Path) -> QualityData:
    names = (
        "configuration_metrics_long.tsv", "configuration_summary.tsv",
        "pairwise_stability.tsv", "PROVENANCE.json", "RUN_COMPLETE",
    )
    paths = require_files(root, names, "BQ/MQ assessment")
    root = Path(root).resolve()
    require_marker(root / "RUN_COMPLETE", "BQ/MQ assessment")
    provenance = read_json(root / "PROVENANCE.json")
    if (
        provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("anchor_score") != "p_nonref"
        or not math.isclose(float(provenance.get("min_anchor_probability")), 0.90, rel_tol=0, abs_tol=1e-12)
    ):
        raise ValueError("BQ/MQ assessment has an invalid WES/routing contract")
    metrics = pd.read_csv(root / "configuration_metrics_long.tsv", sep="\t")
    require_columns(
        metrics,
        ("config", "sample", "metric", "numerator", "denominator", "value",
         "ci95_low", "ci95_high"),
        "BQ/MQ metrics",
    )
    chosen = metrics.loc[metrics["metric"].isin(QUALITY_PLOT_METRICS)].copy()
    expected = set(itertools.product(QUALITY_CONFIGS, SAMPLES, QUALITY_PLOT_METRICS))
    observed = set(chosen[["config", "sample", "metric"]].itertuples(index=False, name=None))
    if observed != expected or len(chosen) != len(expected):
        raise ValueError("BQ/MQ plot metric grid is not exact")
    validate_fraction_rows(chosen, "BQ/MQ plotted metrics")
    chosen[["base_quality", "mapping_quality"]] = pd.DataFrame(
        [parse_quality(config) for config in chosen["config"]], index=chosen.index,
    )
    chosen["quality_label"] = (
        "BQ" + chosen["base_quality"].astype(str)
        + "/MQ" + chosen["mapping_quality"].astype(str)
    )
    summary = pd.read_csv(root / "configuration_summary.tsv", sep="\t")
    require_columns(summary, ("config", "sample", *QUALITY_PLOT_METRICS), "BQ/MQ summary")
    summary_lookup = summary.set_index(["config", "sample"])
    if summary_lookup.index.duplicated().any():
        raise ValueError("BQ/MQ summary contains duplicate config/sample rows")
    for row in chosen.itertuples(index=False):
        value = float(summary_lookup.loc[(row.config, row.sample), row.metric])
        if not math.isclose(value, float(row.value), rel_tol=0, abs_tol=1e-12):
            raise ValueError("BQ/MQ long and summary tables disagree")
    pairwise = pd.read_csv(root / "pairwise_stability.tsv", sep="\t")
    require_columns(pairwise, ("sample", "config_a", "config_b", "anchor_jaccard"), "BQ/MQ pairwise")
    if len(pairwise) != 2 or set(pairwise["sample"]) != set(SAMPLES):
        raise ValueError("BQ/MQ pairwise table must contain one row per sample")
    for row in pairwise.itertuples(index=False):
        if {str(row.config_a), str(row.config_b)} != set(QUALITY_CONFIGS):
            raise ValueError("BQ/MQ pairwise comparison uses the wrong settings")
        value = finite_float(row.anchor_jaccard, "BQ/MQ anchor Jaccard")
        if not 0 <= value <= 1:
            raise ValueError("BQ/MQ anchor Jaccard is not a fraction")
    return QualityData(chosen, pairwise, tuple(paths))


def load_replication(root: Path) -> ReplicationData:
    names = (
        "replicated_configuration_metrics_long.tsv",
        "replicated_configuration_summary.tsv", "replicated_pairwise_stability.tsv",
        "replication_gate_results.tsv", "FINAL_BASE_CONFIGURATION.json",
        "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
        "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE", "RUN_COMPLETE.json",
    )
    paths = require_files(root, names, "final replication assessment")
    root = Path(root).resolve()
    require_marker(root / "RUN_COMPLETE", "final replication assessment")
    ledger_paths = verify_ledger(
        root / "OUTPUT_SHA256SUMS.txt", root, REPLICATION_LEDGER_MEMBERS,
        "replication output ledger",
    )
    provenance = read_json(root / "PROVENANCE.json")
    completion = read_json(root / "RUN_COMPLETE.json")
    selection = read_json(root / "FINAL_BASE_CONFIGURATION.json")
    base = str(selection.get("selected_base_configuration", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base):
        raise ValueError("replication selected base is empty/unsafe")
    seeds_raw = provenance.get("seeds")
    if (
        not isinstance(seeds_raw, list) or len(seeds_raw) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in seeds_raw)
        or len(set(seeds_raw)) != 3
    ):
        raise ValueError("replication provenance must declare three distinct integer seeds")
    seeds = tuple(int(value) for value in seeds_raw)
    selected_configs = provenance.get("selected_configurations")
    if not isinstance(selected_configs, list) or base not in selected_configs or len(selected_configs) != len(set(selected_configs)):
        raise ValueError("replication provenance has an invalid selected-configuration set")
    if (
        selection.get("schema") != "sparcal_beagle_v2_3_final_base_selection_v1"
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or selection.get("final_ensemble_selected") is not False
        or provenance.get("schema") != "sparcal_beagle_v2_3_replication_assessment_v1"
        or provenance.get("status") != "base_configuration_selected"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("development_set_evaluation") is not True
        or provenance.get("samples") != list(SAMPLES)
        or provenance.get("selected_base_configuration") != base
        or completion.get("schema") != "sparcal_beagle_v2_3_replication_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or completion.get("selected_base_configuration") != base
    ):
        raise ValueError("replication selection/status/WES contract is inconsistent")
    verify_identity(
        completion.get("output_sha256sums"), root / "OUTPUT_SHA256SUMS.txt",
        "replication output ledger",
    )

    gate_columns = {
        "base_config", "comparator", "mean_normal_wes_anchor_sensitivity",
        "delta_mean_sensitivity_vs_comparator", "replicated_sensitivity_no_harm",
        "mean_selected_anchor_normal_wes_support", "delta_mean_support_vs_comparator",
        "replicated_support_no_harm", "total_wes_somatic_anchor_leakage",
        "total_leakage_reduction_vs_comparator", "replicated_material_improvement",
        "minimum_anchor_jaccard", "minimum_anchor_jaccard_pass",
        "minimum_route_agreement", "minimum_route_agreement_pass",
        "median_anchor_jaccard", "median_anchor_jaccard_vs_comparator_pass",
        "median_p_nonref_mae", "median_p_nonref_mae_vs_comparator_pass",
        "matched_seed_sample_leakage_nonincrease", "mean_beagle_runtime_seconds",
        "simplicity_rank", "eligible_as_base_default", "selected_base_default",
    }
    gates = pd.read_csv(root / "replication_gate_results.tsv", sep="\t")
    require_columns(gates, gate_columns, "replication gates", exact=True)
    if set(gates["base_config"]) != set(selected_configs) or len(gates) != len(selected_configs):
        raise ValueError("replication gate rows do not match selected configurations")
    bool_columns = {
        "comparator", "replicated_sensitivity_no_harm", "replicated_support_no_harm",
        "replicated_material_improvement", "minimum_anchor_jaccard_pass",
        "minimum_route_agreement_pass", "median_anchor_jaccard_vs_comparator_pass",
        "median_p_nonref_mae_vs_comparator_pass",
        "matched_seed_sample_leakage_nonincrease", "eligible_as_base_default",
        "selected_base_default",
    }
    for column in bool_columns:
        gates[column] = [strict_bool(value, f"replication {column}") for value in gates[column]]
    selected_rows = gates.loc[gates["selected_base_default"]]
    if len(selected_rows) != 1 or str(selected_rows.iloc[0]["base_config"]) != base:
        raise ValueError("replication gate and selection JSON disagree")
    if not bool(selected_rows.iloc[0]["eligible_as_base_default"]):
        raise ValueError("selected replication base is not eligible")

    metrics = pd.read_csv(root / "replicated_configuration_metrics_long.tsv", sep="\t")
    require_columns(
        metrics,
        ("config", "sample", "metric", "numerator", "denominator", "value",
         "ci95_low", "ci95_high"),
        "replication metrics",
    )
    configs = tuple(f"{base}_seed{seed}" for seed in seeds)
    chosen_metrics = metrics.loc[
        metrics["config"].isin(configs) & metrics["metric"].isin(REPLICATION_METRICS)
    ].copy()
    expected_metric_grid = set(itertools.product(configs, SAMPLES, REPLICATION_METRICS))
    observed_metric_grid = set(
        chosen_metrics[["config", "sample", "metric"]].itertuples(index=False, name=None)
    )
    if observed_metric_grid != expected_metric_grid or len(chosen_metrics) != len(expected_metric_grid):
        raise ValueError("selected-base replication metric grid is not exact")
    validate_fraction_rows(chosen_metrics, "selected-base replication metrics")

    pairwise = pd.read_csv(root / "replicated_pairwise_stability.tsv", sep="\t")
    require_columns(
        pairwise,
        ("sample", "config_a", "config_b", "anchor_jaccard", "route_agreement", "p_nonref_mae"),
        "replication pairwise stability",
    )
    within = pairwise.loc[pairwise["config_a"].isin(configs) & pairwise["config_b"].isin(configs)].copy()
    expected_pairs = {
        (sample, *sorted(pair))
        for sample in SAMPLES for pair in itertools.combinations(configs, 2)
    }
    observed_pairs = {
        (str(row.sample), *sorted((str(row.config_a), str(row.config_b))))
        for row in within.itertuples(index=False)
    }
    if observed_pairs != expected_pairs or len(within) != len(expected_pairs):
        raise ValueError("selected-base replication seed-pair grid is not exact")
    for column in ("anchor_jaccard", "route_agreement", "p_nonref_mae"):
        within[column] = pd.to_numeric(within[column], errors="raise")
        if not np.isfinite(within[column]).all():
            raise ValueError(f"replication {column} contains non-finite values")
    if not within["anchor_jaccard"].between(0, 1).all() or not within["route_agreement"].between(0, 1).all():
        raise ValueError("replication stability fractions are outside [0,1]")

    selected_gate = selected_rows.iloc[0].to_dict()
    lookup = chosen_metrics.set_index(["config", "sample", "metric"])
    mean_sensitivity = np.mean([
        float(lookup.loc[(config, sample, "normal_wes_anchor_sensitivity"), "value"])
        for config in configs for sample in SAMPLES
    ])
    mean_support = np.mean([
        float(lookup.loc[(config, sample, "selected_anchor_normal_wes_support"), "value"])
        for config in configs for sample in SAMPLES
    ])
    total_leakage = sum(
        int(lookup.loc[(config, sample, "wes_somatic_anchor_leakage"), "numerator"])
        for config in configs for sample in SAMPLES
    )
    checks = {
        "mean_normal_wes_anchor_sensitivity": mean_sensitivity,
        "mean_selected_anchor_normal_wes_support": mean_support,
        "total_wes_somatic_anchor_leakage": total_leakage,
        "minimum_anchor_jaccard": float(within["anchor_jaccard"].min()),
        "minimum_route_agreement": float(within["route_agreement"].min()),
        "median_anchor_jaccard": float(within["anchor_jaccard"].median()),
        "median_p_nonref_mae": float(within["p_nonref_mae"].median()),
    }
    for key, expected in checks.items():
        if not math.isclose(float(selected_gate[key]), float(expected), rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"replication selected-gate aggregate mismatch: {key}")
    within["pair"] = within["config_a"].astype(str) + " vs " + within["config_b"].astype(str)
    return ReplicationData(
        base, seeds, within, chosen_metrics, selected_gate,
        tuple(dict.fromkeys([*paths, *ledger_paths])),
    )


def load_ensemble(root: Path, replication: ReplicationData) -> EnsembleData:
    names = (
        "ensemble_gate_results.tsv", "FINAL_BEAGLE_REPRESENTATION.json",
        "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
        "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE", "RUN_COMPLETE.json",
    )
    paths = require_files(root, names, "final ensemble assessment")
    root = Path(root).resolve()
    require_marker(root / "RUN_COMPLETE", "final ensemble assessment")
    ledger_paths = verify_ledger(
        root / "OUTPUT_SHA256SUMS.txt", root, ENSEMBLE_LEDGER_MEMBERS,
        "ensemble output ledger",
    )
    selection = read_json(root / "FINAL_BEAGLE_REPRESENTATION.json")
    provenance = read_json(root / "PROVENANCE.json")
    completion = read_json(root / "RUN_COMPLETE.json")
    base = str(selection.get("base_configuration", ""))
    representation = str(selection.get("selected_representation", ""))
    if representation not in ENSEMBLE_REPRESENTATIONS:
        raise ValueError("ensemble selected an unknown representation")
    expected_seed = replication.seeds[0] if representation == "single_seed" else None
    expected_multiplier = 1 if representation == "single_seed" else 3
    if (
        selection.get("schema") != "sparcal_beagle_v2_3_final_representation_v1"
        or base != replication.base
        or selection.get("selected_seed") != expected_seed
        or selection.get("beagle_run_multiplier") != expected_multiplier
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or provenance.get("schema") != "sparcal_beagle_v2_3_ensemble_selection_v1"
        or provenance.get("status") != "final_representation_selected"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("development_set_evaluation") is not True
        or provenance.get("base_configuration") != base
        or provenance.get("selected_representation") != representation
        or provenance.get("frozen_seeds") != list(replication.seeds)
        or completion.get("schema") != "sparcal_beagle_v2_3_ensemble_selection_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or completion.get("base_configuration") != base
        or completion.get("selected_representation") != representation
    ):
        raise ValueError("ensemble selection/status/WES contract is inconsistent")
    verify_identity(
        completion.get("output_sha256sums"), root / "OUTPUT_SHA256SUMS.txt",
        "ensemble output ledger",
    )
    gate_columns = {
        "method", "single_seed_comparator", "mean_normal_wes_anchor_sensitivity",
        "delta_mean_sensitivity", "sensitivity_no_harm",
        "mean_selected_anchor_normal_wes_support", "delta_mean_support",
        "support_no_harm", "P4_somatic_leakage_n", "P6_somatic_leakage_n",
        "total_somatic_leakage_n", "leakage_nonincrease_both",
        "total_leakage_reduction", "material_improvement",
        "minimum_leave_one_out_anchor_jaccard", "leave_one_out_stability_pass",
        "beagle_run_multiplier", "advance_over_single_seed",
        "selected_final_representation",
    }
    gates = pd.read_csv(root / "ensemble_gate_results.tsv", sep="\t")
    require_columns(gates, gate_columns, "ensemble gates", exact=True)
    if set(gates["method"]) != set(ENSEMBLE_METHODS) or len(gates) != 2:
        raise ValueError("ensemble gate table is not the exact mean/median grid")
    bool_columns = {
        "sensitivity_no_harm", "support_no_harm", "leakage_nonincrease_both",
        "material_improvement", "leave_one_out_stability_pass",
        "advance_over_single_seed", "selected_final_representation",
    }
    for column in bool_columns:
        gates[column] = [strict_bool(value, f"ensemble {column}") for value in gates[column]]
    comparator = f"{base}_seed{replication.seeds[0]}"
    if set(gates["single_seed_comparator"]) != {comparator}:
        raise ValueError("ensemble comparator differs from selected base/first seed")
    selected_methods = gates.loc[gates["selected_final_representation"], "method"].tolist()
    expected_selected = [] if representation == "single_seed" else [representation]
    if selected_methods != expected_selected:
        raise ValueError("ensemble gate and final representation selection disagree")
    if representation == "single_seed":
        if gates["advance_over_single_seed"].any():
            raise ValueError("single seed selected despite an advancing ensemble")
    else:
        row = gates.set_index("method").loc[representation]
        if not bool(row["advance_over_single_seed"]):
            raise ValueError("selected ensemble did not pass its advance gate")
    for column in (
        "mean_normal_wes_anchor_sensitivity", "delta_mean_sensitivity",
        "mean_selected_anchor_normal_wes_support", "delta_mean_support",
        "minimum_leave_one_out_anchor_jaccard",
    ):
        gates[column] = pd.to_numeric(gates[column], errors="raise")
        if not np.isfinite(gates[column]).all():
            raise ValueError(f"ensemble {column} contains non-finite values")
    if not gates["minimum_leave_one_out_anchor_jaccard"].between(0, 1).all():
        raise ValueError("ensemble leave-one-out Jaccard is not a fraction")
    for column in ("P4_somatic_leakage_n", "P6_somatic_leakage_n", "total_somatic_leakage_n", "total_leakage_reduction", "beagle_run_multiplier"):
        gates[column] = pd.to_numeric(gates[column], errors="raise").astype(int)
    if not (gates["P4_somatic_leakage_n"] + gates["P6_somatic_leakage_n"]).equals(gates["total_somatic_leakage_n"]):
        raise ValueError("ensemble P4/P6 leakage counts do not sum to total")
    return EnsembleData(
        base, representation, gates,
        tuple(dict.fromkeys([*paths, *ledger_paths])),
    )


def _read_indexed_tsv(path: Path, index_name: str = "section") -> pd.DataFrame:
    frame = pd.read_csv(path, sep="\t", dtype={index_name: str})
    if index_name not in frame.columns and len(frame.columns):
        frame = frame.rename(columns={frame.columns[0]: index_name})
        frame[index_name] = frame[index_name].astype(str)
    return frame


def load_dlpfc(root: Path) -> DlpfcData:
    names = (
        "NONINFERIORITY_RESULT.json", "STRICT_RUN_VALIDATION.json",
        "clustering_metrics_long.tsv", "donor_paired_deltas.tsv",
        "method_success_counts.tsv", "section_paired_deltas.tsv",
        "seed_paired_deltas.tsv", "OUTPUT_SHA256SUMS", "ASSESSMENT_COMPLETE",
    )
    paths = require_files(root, names, "final strict DLPFC assessment")
    root = Path(root).resolve()
    require_marker(root / "ASSESSMENT_COMPLETE", "final strict DLPFC assessment")
    accepted_marker = (root / "ACCEPTED").is_file()
    rejected_marker = (root / "REJECTED").is_file()
    if accepted_marker == rejected_marker:
        raise ValueError("strict DLPFC assessment must have exactly one decision marker")
    decision_path = root / ("ACCEPTED" if accepted_marker else "REJECTED")
    ledger_paths = verify_ledger(
        root / "OUTPUT_SHA256SUMS", root, STRICT_LEDGER_MEMBERS,
        "strict DLPFC output ledger",
    )
    result = read_json(root / "NONINFERIORITY_RESULT.json")
    validation = read_json(root / "STRICT_RUN_VALIDATION.json")
    gates = result.get("gates")
    if not isinstance(gates, dict) or set(gates) != STRICT_GATES:
        raise ValueError("strict DLPFC result does not have the exact eight-gate set")
    if any(type(value) is not bool for value in gates.values()):
        raise TypeError("strict DLPFC gate values must be booleans")
    accepted = result.get("accept_replacement")
    if type(accepted) is not bool or accepted != all(gates.values()) or accepted != accepted_marker:
        raise ValueError("strict DLPFC decision, gates, and marker disagree")
    if (
        result.get("schema_version") != "3.0"
        or result.get("protocol_version") != STRICT_PROTOCOL
        or result.get("baseline_method") != "legacy_any_panel"
        or result.get("candidate_method") != STRICT_METHOD
        or result.get("mandatory_execution_even_if_inputs_exactly_equal") is not True
        or result.get("expected_section_count") != 12
        or result.get("expected_runs_per_section") != 10
        or result.get("expected_paired_run_count") != 120
        or "not SNV-caller accuracy" not in str(result.get("interpretation_scope"))
        or result.get("candidate_only_failures") != []
    ):
        raise ValueError("strict DLPFC result envelope/interpretation is invalid")
    if (
        validation.get("status") != "all_checks_passed"
        or validation.get("protocol_version") != STRICT_PROTOCOL
        or validation.get("validated_sections") != list(SECTIONS)
        or validation.get("candidate_only_failures") != []
        or validation.get("mandatory_execution_even_if_exactly_equal") is not True
    ):
        raise ValueError("strict DLPFC run validation is incomplete")
    run_root = Path(str(result.get("candidate_root", ""))).resolve(strict=True)
    if Path(str(result.get("baseline_root", ""))).resolve(strict=True) != run_root:
        raise ValueError("strict DLPFC arms are not bound to one paired run root")
    if Path(str(validation.get("root", ""))).resolve(strict=True) != run_root:
        raise ValueError("strict DLPFC result and validation roots disagree")
    strict_manifest_path = run_root / "STRICT_INPUT_MANIFEST.json"
    if not strict_manifest_path.is_file():
        raise FileNotFoundError(strict_manifest_path)
    manifest_sha = sha256(strict_manifest_path)
    if (
        result.get("strict_input_manifest_sha256") != manifest_sha
        or validation.get("strict_input_manifest_sha256") != manifest_sha
    ):
        raise ValueError("strict DLPFC manifest hash is not transitively bound")
    strict_manifest = read_json(strict_manifest_path)
    semantic = strict_manifest.get("candidate_semantic_contract")
    if not isinstance(semantic, dict):
        raise TypeError("strict DLPFC semantic contract is absent")
    expected_semantic = {
        "method": STRICT_METHOD,
        "materializer_schema": "sparcal_beagle_v2_3_final_dlpfc_materialization_v1",
        "scientific_role": "final all_panel_present feature-support compatibility",
        "selected_routes": [
            "beagle_nonref_highconf", "beagle_ref_highconf", "beagle_uncertain",
        ],
        "multiallelic_policy": "any",
        "all_candidate_panel_positions_representable": True,
        "complete_column_subset_of_baseline": True,
        "exact_dataframe_equal_to_baseline_required": False,
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
    }
    for key, expected in expected_semantic.items():
        if semantic.get(key) != expected:
            raise ValueError(f"strict DLPFC all_panel_present semantic mismatch: {key}")
    if (
        strict_manifest.get("schema_version") != "1.0"
        or strict_manifest.get("status") != "prepared"
        or strict_manifest.get("protocol_version") != STRICT_PROTOCOL
        or strict_manifest.get("sections") != list(SECTIONS)
        or strict_manifest.get("methods") != ["legacy_any_panel", STRICT_METHOD]
        or strict_manifest.get("runs_per_section") != 10
        or strict_manifest.get("wes_used_as_input") is not False
        or strict_manifest.get("mandatory_execution_even_if_exactly_equal") is not True
    ):
        raise ValueError("strict DLPFC input manifest envelope is invalid")

    sections = _read_indexed_tsv(root / "section_paired_deltas.tsv")
    require_columns(sections, ("section", "delta_ari", "delta_nmi", "donor"), "strict section deltas", exact=True)
    if tuple(sections["section"]) != SECTIONS:
        raise ValueError("strict DLPFC sections are missing or out of order")
    if list(sections["donor"]) != [DONOR_BY_SECTION[section] for section in SECTIONS]:
        raise ValueError("strict DLPFC donor mapping differs from the frozen design")
    for metric in ("delta_ari", "delta_nmi"):
        sections[metric] = pd.to_numeric(sections[metric], errors="raise")
        if not np.isfinite(sections[metric]).all():
            raise ValueError(f"strict DLPFC {metric} is non-finite")
    seeds = _read_indexed_tsv(root / "seed_paired_deltas.tsv")
    require_columns(seeds, ("section", "run", "delta_ari", "delta_nmi"), "strict seed deltas", exact=True)
    grouped = seeds.groupby("section", sort=False)[["delta_ari", "delta_nmi"]].mean().reindex(SECTIONS)
    if grouped.isna().any().any() or not np.allclose(
        grouped.to_numpy(dtype=float),
        sections.set_index("section")[["delta_ari", "delta_nmi"]].to_numpy(dtype=float),
        rtol=0, atol=1e-12,
    ):
        raise ValueError("strict section deltas do not equal paired-seed means")
    donors = _read_indexed_tsv(root / "donor_paired_deltas.tsv", index_name="donor")
    require_columns(donors, ("donor", "delta_ari", "delta_nmi"), "strict donor deltas", exact=True)
    if tuple(donors["donor"]) != tuple(DONOR_COLORS):
        raise ValueError("strict donor summary is not ordered Br5292/Br5595/Br8100")
    expected_donors = sections.groupby("donor", sort=False)[["delta_ari", "delta_nmi"]].mean().reindex(DONOR_COLORS)
    if not np.allclose(
        donors.set_index("donor")[["delta_ari", "delta_nmi"]].to_numpy(dtype=float),
        expected_donors.to_numpy(dtype=float), rtol=0, atol=1e-12,
    ):
        raise ValueError("strict donor deltas disagree with section means")
    if not math.isclose(float(result["mean_delta_ari"]), float(sections["delta_ari"].mean()), rel_tol=0, abs_tol=1e-12):
        raise ValueError("strict mean delta ARI disagrees with sections")
    if not math.isclose(float(result["mean_delta_nmi"]), float(sections["delta_nmi"].mean()), rel_tol=0, abs_tol=1e-12):
        raise ValueError("strict mean delta NMI disagrees with sections")
    if not math.isclose(float(result["minimum_donor_mean_delta_ari"]), float(donors["delta_ari"].min()), rel_tol=0, abs_tol=1e-12):
        raise ValueError("strict minimum donor mean delta ARI disagrees")
    donor_range = result.get("donor_mean_delta_ari_range_descriptive")
    expected_range = [float(donors["delta_ari"].min()), float(donors["delta_ari"].max())]
    if (
        not isinstance(donor_range, list) or len(donor_range) != 2
        or not np.allclose(donor_range, expected_range, rtol=0, atol=1e-12)
    ):
        raise ValueError("strict descriptive donor-mean delta-ARI range disagrees")
    section_ci = result.get("section_bootstrap_delta_ari_ci95_descriptive")
    if not isinstance(section_ci, list) or len(section_ci) != 2:
        raise ValueError("strict descriptive section-bootstrap delta-ARI interval is absent")
    ci_low, ci_high = map(float, section_ci)
    mean_ari = float(result["mean_delta_ari"])
    if not all(map(math.isfinite, (ci_low, ci_high))) or not ci_low <= mean_ari <= ci_high:
        raise ValueError("strict descriptive section-bootstrap delta-ARI interval is invalid")

    success = _read_indexed_tsv(root / "method_success_counts.tsv")
    require_columns(success, ("section", "legacy_any_panel", STRICT_METHOD), "strict success counts", exact=True)
    if int(pd.to_numeric(success["legacy_any_panel"]).sum()) != int(result["baseline_successful_runs"]):
        raise ValueError("strict baseline success count mismatch")
    if int(pd.to_numeric(success[STRICT_METHOD]).sum()) != int(result["candidate_successful_runs"]):
        raise ValueError("strict candidate success count mismatch")
    runs = pd.read_csv(root / "clustering_metrics_long.tsv", sep="\t", dtype={"section": str})
    require_columns(runs, ("section", "method", "run", "status", "ari", "nmi"), "strict run metrics")
    return DlpfcData(
        sections, donors, result, semantic,
        tuple(dict.fromkeys([*paths, decision_path, strict_manifest_path, *ledger_paths])),
    )


def load_monopogen(root: Path) -> MonopogenData:
    names = (
        "run_summary.tsv", "method_sets.tsv", "method_intersections.tsv",
        "transition_counts.tsv", "input_manifest.tsv", "PROVENANCE.json",
        "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE",
    )
    paths = require_files(root, names, "corrected Monopogen v3 audit")
    root = Path(root).resolve()
    require_marker(
        root / "RUN_COMPLETE", "corrected Monopogen v3 audit",
        marker_format="monopogen_v3",
    )
    ledger_paths = verify_ledger(
        root / "OUTPUT_SHA256SUMS.txt", root, MONO_LEDGER_MEMBERS,
        "Monopogen-v3 output ledger",
    )
    provenance = read_json(root / "PROVENANCE.json")
    if (
        provenance.get("audit_schema_version") != "3.0"
        or provenance.get("wes_used") is not False
        or provenance.get("monopogen_public_commit") != MONO_COMMIT
        or not math.isclose(float(provenance.get("pnonref_threshold")), 0.90, rel_tol=0, abs_tol=1e-12)
        or "does not claim that every phased record is a final Monopogen somatic call"
        not in str(provenance.get("scope_limit"))
    ):
        raise ValueError("Monopogen-v3 provenance/scope contract is invalid")
    summary = pd.read_csv(root / "run_summary.tsv", sep="\t")
    required = {
        "run", "gp_records", "phased_records", "gp_phased_record_sets_equal",
        "gp_keys_absent_from_phased", "phased_keys_absent_from_gp",
        "gp_phased_comparable", "gp_phased_unordered_gt_changes",
        "phased_heterozygous", "phased_0_pipe_1", "phased_1_pipe_0",
    }
    require_columns(summary, required, "Monopogen-v3 run summary")
    if tuple(summary["run"]) != MONO_RUNS:
        raise ValueError("Monopogen-v3 run grid/order is not exact")
    for row in summary.itertuples(index=False):
        gp_n = parse_int(row.gp_records, "Monopogen GP records")
        phased_n = parse_int(row.phased_records, "Monopogen phased records")
        comparable = parse_int(row.gp_phased_comparable, "Monopogen comparable records")
        changes = parse_int(row.gp_phased_unordered_gt_changes, "Monopogen unordered changes")
        absent_forward = parse_int(row.gp_keys_absent_from_phased, "Monopogen GP-only keys")
        absent_reverse = parse_int(row.phased_keys_absent_from_gp, "Monopogen phased-only keys")
        heterozygous = parse_int(row.phased_heterozygous, "Monopogen phased heterozygous")
        phase_01 = parse_int(row.phased_0_pipe_1, "Monopogen 0|1")
        phase_10 = parse_int(row.phased_1_pipe_0, "Monopogen 1|0")
        if (
            not strict_bool(row.gp_phased_record_sets_equal, "Monopogen GP/phased equality")
            or gp_n != phased_n or comparable != gp_n
            or absent_forward != 0 or absent_reverse != 0 or changes != 0
            or heterozygous <= 0 or phase_01 + phase_10 != heterozygous
        ):
            raise ValueError(f"Monopogen second-pass mechanism contract failed: {row.run}")
    method_sets = pd.read_csv(root / "method_sets.tsv", sep="\t")
    require_columns(method_sets, ("run", "method", "method_role", "count", "definition"), "Monopogen method sets")
    expected_sets = set(itertools.product(MONO_RUNS, MONO_METHODS))
    observed_sets = set(method_sets[["run", "method"]].itertuples(index=False, name=None))
    if observed_sets != expected_sets or len(method_sets) != len(expected_sets):
        raise ValueError("Monopogen-v3 method-set grid is not exact")
    summary_lookup = summary.set_index("run")
    summary_count_fields = {
        "monopogen_downstream_beagle_record_membership": "monopogen_downstream_beagle_records",
        "monopogen_paper_raw_beagle_gt_concordant_nonref": "monopogen_paper_raw_beagle_gt_concordant_nonref",
        "sparcal_derived_pnonref_threshold": "sparcal_derived_pnonref_threshold_count",
        "diagnostic_final_phased_nonref_gt": "diagnostic_final_phased_nonref_gt",
    }
    require_columns(summary, summary_count_fields.values(), "Monopogen summary method counts")
    for row in method_sets.itertuples(index=False):
        expected = int(summary_lookup.loc[row.run, summary_count_fields[row.method]])
        if int(row.count) != expected:
            raise ValueError(f"Monopogen method-set count mismatch: {row.run}/{row.method}")
    intersections = pd.read_csv(root / "method_intersections.tsv", sep="\t")
    require_columns(intersections, ("run", "left_method", "right_method", "intersection_count", "jaccard"), "Monopogen intersections")
    transitions = pd.read_csv(root / "transition_counts.tsv", sep="\t")
    require_columns(transitions, ("run", "raw_pl_gt", "beagle_gp_gt", "count"), "Monopogen transitions")
    input_manifest = pd.read_csv(root / "input_manifest.tsv", sep="\t")
    require_columns(input_manifest, ("run", "chromosome", "kind", "path", "size_bytes", "mtime_ns", "sha256"), "Monopogen input manifest", exact=True)
    if len(input_manifest) != 264:
        raise ValueError("Monopogen-v3 input manifest must contain 264 files")
    return MonopogenData(summary, provenance, tuple(dict.fromkeys([*paths, *ledger_paths])))


def validate_evidence_monopogen_binding(evidence: EvidenceData, monopogen: MonopogenData, mono_root: Path) -> None:
    binding = evidence.provenance.get("monopogen_audit_validation")
    if not isinstance(binding, Mapping):
        raise TypeError("v11 evidence lacks corrected Monopogen audit binding")
    mono_root = Path(mono_root).resolve()
    expected_counts = [
        {"run": str(row.run), "method": str(row.method), "count": int(row.count)}
        for row in pd.read_csv(mono_root / "method_sets.tsv", sep="\t").itertuples(index=False)
    ]
    if (
        Path(str(binding.get("path", ""))).resolve() != mono_root
        or binding.get("audit_schema_version") != "3.0"
        or binding.get("public_commit") != MONO_COMMIT
        or binding.get("input_manifest_sha256") != sha256(mono_root / "input_manifest.tsv")
        or binding.get("output_manifest_sha256") != sha256(mono_root / "OUTPUT_SHA256SUMS.txt")
        or binding.get("full_run_method_counts") != expected_counts
    ):
        raise ValueError("v11 evidence and corrected Monopogen-v3 audit are not identically bound")


def load_inputs(
    evidence_dir: Path, replication_dir: Path, ensemble_dir: Path,
    quality_dir: Path, dlpfc_dir: Path, monopogen_dir: Path,
) -> FigureInputs:
    evidence = load_evidence(evidence_dir)
    quality = load_quality(quality_dir)
    replication = load_replication(replication_dir)
    ensemble = load_ensemble(ensemble_dir, replication)
    dlpfc = load_dlpfc(dlpfc_dir)
    monopogen = load_monopogen(monopogen_dir)
    validate_evidence_monopogen_binding(evidence, monopogen, monopogen_dir)
    return FigureInputs(evidence, quality, replication, ensemble, dlpfc, monopogen)


PLOTTED_COLUMNS = (
    "panel", "scope", "element", "sample", "run", "donor", "section",
    "method", "method_label", "config", "seed_pair", "metric", "numerator",
    "denominator", "value", "ci95_low", "ci95_high", "threshold", "selected",
    "primary_interface", "note",
)


def build_plotted_data(inputs: FigureInputs) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in inputs.evidence.metrics.itertuples(index=False):
        rows.append({
            "panel": "A", "scope": "same_sparcal_input",
            "element": "development_metric", "sample": row.sample,
            "method": row.method, "method_label": row.method_label,
            "metric": row.metric, "numerator": int(row.numerator),
            "denominator": int(row.denominator), "value": float(row.value),
            "ci95_low": float(row.ci95_low), "ci95_high": float(row.ci95_high),
            "selected": bool(row.selected),
            "note": "matched-WES internal development evaluation; not a production input",
        })
    for row in inputs.quality.metrics.itertuples(index=False):
        rows.append({
            "panel": "B", "scope": "same_sparcal_input",
            "element": "input_quality_metric", "sample": row.sample,
            "config": row.config, "metric": row.metric,
            "numerator": int(row.numerator), "denominator": int(row.denominator),
            "value": float(row.value), "ci95_low": float(row.ci95_low),
            "ci95_high": float(row.ci95_high),
            "note": "BQ/MQ end-to-end matched-WES development ablation",
        })
    for row in inputs.quality.pairwise.itertuples(index=False):
        rows.append({
            "panel": "B", "scope": "same_sparcal_input",
            "element": "quality_pairwise", "sample": row.sample,
            "config": f"{row.config_a} vs {row.config_b}",
            "metric": "anchor_jaccard", "value": float(row.anchor_jaccard),
            "note": "anchor-set overlap across input-quality settings",
        })
    for row in inputs.replication.stability.itertuples(index=False):
        for metric in ("anchor_jaccard", "route_agreement"):
            rows.append({
                "panel": "C", "scope": "same_sparcal_input",
                "element": "selected_base_seed_pair", "sample": row.sample,
                "config": inputs.replication.base, "seed_pair": row.pair,
                "metric": metric, "value": float(getattr(row, metric)),
                "selected": True,
                "note": "three-seed replication of the selected base configuration",
            })
    rows.append({
        "panel": "C", "scope": "same_sparcal_input", "element": "base_selection",
        "config": inputs.replication.base, "metric": "eligible_as_base_default",
        "value": int(bool(inputs.replication.selected_gate["eligible_as_base_default"])),
        "selected": True,
        "note": "WES is development evaluation only",
    })
    for row in inputs.ensemble.gates.itertuples(index=False):
        for metric in ("delta_mean_sensitivity", "delta_mean_support"):
            rows.append({
                "panel": "D", "scope": "same_sparcal_input",
                "element": "ensemble_delta", "method": row.method,
                "metric": metric, "value": float(getattr(row, metric)),
                "selected": bool(row.selected_final_representation),
                "note": "delta versus the selected base's first frozen seed; WES development evaluation",
            })
        rows.extend([
            {
                "panel": "D", "scope": "same_sparcal_input",
                "element": "ensemble_diagnostic", "method": row.method,
                "metric": "total_leakage_reduction", "value": int(row.total_leakage_reduction),
                "selected": bool(row.selected_final_representation),
                "note": "positive means fewer matched-WES somatic alleles absorbed as anchors",
            },
            {
                "panel": "D", "scope": "same_sparcal_input",
                "element": "ensemble_diagnostic", "method": row.method,
                "metric": "minimum_leave_one_out_anchor_jaccard",
                "value": float(row.minimum_leave_one_out_anchor_jaccard),
                "selected": bool(row.selected_final_representation),
                "note": "leave-one-seed-out ensemble stability",
            },
        ])
    rows.append({
        "panel": "D", "scope": "same_sparcal_input", "element": "final_selection",
        "config": inputs.replication.base, "method": inputs.ensemble.representation,
        "metric": "selected_final_representation", "value": 1, "selected": True,
        "note": "final Beagle-v2.3 representation",
    })
    for row in inputs.dlpfc.sections.itertuples(index=False):
        for metric in ("delta_ari", "delta_nmi"):
            rows.append({
                "panel": "E", "scope": "DLPFC_functional_no_harm",
                "element": "section_mean", "donor": row.donor,
                "section": row.section, "method": STRICT_METHOD,
                "method_label": "final all_panel_present compatibility interface",
                "metric": metric, "value": float(getattr(row, metric)),
                "primary_interface": True,
                "note": "paired STAGATE/fixed-EEE mclust; not SNV-caller accuracy",
            })
    for metric in ("mean_delta_ari", "mean_delta_nmi", "minimum_donor_mean_delta_ari"):
        rows.append({
            "panel": "E", "scope": "DLPFC_functional_no_harm",
            "element": "registered_summary", "method": STRICT_METHOD,
            "method_label": "final all_panel_present compatibility interface",
            "metric": metric, "value": float(inputs.dlpfc.result[metric]),
            "primary_interface": True,
            "note": "P(nonref)>=0.90 anchor matrix is a secondary diagnostic, not this primary arm",
        })
    ci_low, ci_high = inputs.dlpfc.result[
        "section_bootstrap_delta_ari_ci95_descriptive"
    ]
    rows.append({
        "panel": "E", "scope": "DLPFC_functional_no_harm",
        "element": "registered_interval", "method": STRICT_METHOD,
        "method_label": "final all_panel_present compatibility interface",
        "metric": "mean_delta_ari_section_bootstrap_ci95_descriptive",
        "value": float(inputs.dlpfc.result["mean_delta_ari"]),
        "ci95_low": float(ci_low), "ci95_high": float(ci_high),
        "primary_interface": True,
        "note": "descriptive resampling of 12 sections; not a donor-block confidence interval",
    })
    for row in inputs.dlpfc.donors.itertuples(index=False):
        for metric in ("delta_ari", "delta_nmi"):
            rows.append({
                "panel": "E", "scope": "DLPFC_functional_no_harm",
                "element": "donor_mean", "donor": row.donor,
                "method": STRICT_METHOD,
                "method_label": "final all_panel_present compatibility interface",
                "metric": metric, "value": float(getattr(row, metric)),
                "primary_interface": True,
                "note": "descriptive donor mean; only three donors, so no donor-block confidence interval",
            })
    for gate, passed in sorted(inputs.dlpfc.result["gates"].items()):
        rows.append({
            "panel": "E", "scope": "DLPFC_functional_no_harm",
            "element": "registered_gate", "method": STRICT_METHOD,
            "metric": gate, "value": int(bool(passed)), "primary_interface": True,
            "note": "frozen strict DLPFC gate",
        })
    for row in inputs.monopogen.summary.itertuples(index=False):
        gp_records = int(row.gp_records)
        comparable = int(row.gp_phased_comparable)
        heterozygous = int(row.phased_heterozygous)
        mono_values = {
            "gp_to_phased_record_set_retention": int(row.phased_records) / gp_records,
            "unordered_gt_concordance": 1 - int(row.gp_phased_unordered_gt_changes) / comparable,
            "phased_0_pipe_1_fraction_of_heterozygotes": int(row.phased_0_pipe_1) / heterozygous,
            "phased_1_pipe_0_fraction_of_heterozygotes": int(row.phased_1_pipe_0) / heterozygous,
        }
        for metric, value in mono_values.items():
            rows.append({
                "panel": "F", "scope": "external_monopogen_different_input_universe",
                "element": "mechanism_audit", "run": row.run,
                "method": "Monopogen Beagle component", "metric": metric,
                "value": float(value),
                "note": "second pass adds phase orientation, not new evidence; full somatic caller also uses raw read support",
            })
    return pd.DataFrame(rows).reindex(columns=PLOTTED_COLUMNS)


def resolve_font(
    font_family: str,
    font_file: Path | None,
    font_dir: Path | None = None,
) -> tuple[str, tuple[Path, ...]]:
    """Register an exact font family; Arial requires all four manuscript faces."""
    if font_file is not None and font_dir is not None:
        raise ValueError("pass only one of --font-file and --font-dir")
    candidates: list[Path]
    if font_file is not None:
        font_file = Path(font_file).resolve()
        if not font_file.is_file():
            raise FileNotFoundError(font_file)
        candidates = [font_file]
        candidates.extend(
            Path(path).resolve()
            for path in font_manager.findSystemFonts(fontpaths=[str(font_file.parent)])
        )
    else:
        if font_dir is None and font_family.casefold() == "arial":
            font_dir = DEFAULT_ARIAL_DIR
        if font_dir is not None:
            font_dir = Path(font_dir).resolve()
            if not font_dir.is_dir():
                raise FileNotFoundError(font_dir)
            candidates = [
                Path(path).resolve()
                for path in font_manager.findSystemFonts(fontpaths=[str(font_dir)])
            ]
        else:
            try:
                candidates = [Path(font_manager.findfont(
                    font_family, fallback_to_default=False,
                )).resolve()]
            except ValueError as error:
                raise RuntimeError(
                    f"font {font_family!r} is unavailable; pass --font-file or --font-dir"
                ) from error
    matches: dict[Path, str] = {}
    for path in sorted(set(candidates), key=str):
        if not path.is_file():
            continue
        try:
            face = FT2Font(str(path))
        except (RuntimeError, OSError):
            continue
        if face.family_name.casefold() == font_family.casefold():
            matches[path] = str(face.style_name)
    if not matches:
        raise RuntimeError(
            f"no exact {font_family!r} font files found; refusing silent fallback"
        )
    if font_family.casefold() == "arial":
        observed = set(matches.values())
        missing = sorted(REQUIRED_ARIAL_STYLES - observed)
        if missing:
            raise RuntimeError(
                "Arial family is incomplete; missing required faces: "
                + ", ".join(missing)
            )
    for path in matches:
        font_manager.fontManager.addfont(str(path))
    regular = next(
        (path for path, style in matches.items() if style == "Regular"),
        next(iter(matches)),
    )
    actual = font_manager.FontProperties(fname=str(regular)).get_name()
    if actual.casefold() != font_family.casefold():
        raise RuntimeError(
            f"font {font_family!r} resolved to {actual!r}; refusing silent fallback"
        )
    return actual, tuple(matches)


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def horizontal_fraction(ax: plt.Axes, row: pd.Series, y: float, **kwargs: object) -> None:
    value = float(row["value"])
    errors = np.array([[
        value - float(row["ci95_low"])
    ], [
        float(row["ci95_high"]) - value
    ]])
    ax.errorbar(value, y, xerr=errors, capsize=1.8, elinewidth=0.7, **kwargs)


def draw_evidence(ax: plt.Axes, data: EvidenceData) -> None:
    frame = data.metrics
    y = np.arange(len(EVIDENCE_METHODS), dtype=float)
    ax.axhspan(2.52, 3.48, color="#E6F4EA", zorder=0)
    metric_style = {
        "normal_wes_anchor_sensitivity": ("o", "germline sensitivity"),
        "normal_wes_anchor_gt_accuracy": ("D", "exact-GT accuracy"),
        "wes_somatic_retention": ("s", "somatic retention"),
    }
    sample_offset = {"P4": -0.13, "P6": 0.13}
    metric_offset = dict(zip(EVIDENCE_PLOT_METRICS, (-0.045, 0.0, 0.045)))
    for sample in SAMPLES:
        for method_index, method in enumerate(EVIDENCE_METHODS):
            for metric, (marker, _) in metric_style.items():
                row = frame.loc[
                    frame["sample"].eq(sample) & frame["method"].eq(method)
                    & frame["metric"].eq(metric)
                ].iloc[0]
                horizontal_fraction(
                    ax, row, method_index + sample_offset[sample] + metric_offset[metric],
                    fmt=marker, ms=3.5, color=SAMPLE_COLORS[sample],
                    mec=SAMPLE_COLORS[sample],
                    mfc=("white" if marker == "D" else SAMPLE_COLORS[sample]), zorder=3,
                )
    ax.set_yticks(y, [EVIDENCE_METHOD_LABELS[value] for value in EVIDENCE_METHODS])
    ax.invert_yaxis()
    ax.set_xlim(max(0, float(frame["ci95_low"].min()) - 0.035), 1.015)
    ax.set_xlabel("Fraction (Wilson 95% CI)")
    ax.set_title("a", loc="left", fontweight="bold")
    handles = [
        *(Line2D([0], [0], marker="o", lw=0, color=SAMPLE_COLORS[sample], label=sample, ms=4) for sample in SAMPLES),
        *(Line2D([0], [0], marker=marker, lw=0, color="black", mfc=("white" if marker == "D" else "black"), label=label, ms=4)
          for marker, label in metric_style.values()),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, ncol=2, loc="lower left")
    style_axis(ax)


def draw_quality(ax: plt.Axes, data: QualityData) -> None:
    frame = data.metrics
    y = np.arange(len(QUALITY_CONFIGS), dtype=float)
    metric_style = {
        "raw_candidate_normal_wes_recall": ("o", "raw germline recall"),
        "normal_wes_end_to_end_anchor_recall": ("D", "anchor recall"),
        "wes_somatic_end_to_end_retention": ("s", "somatic retention"),
    }
    sample_offset = {"P4": -0.13, "P6": 0.13}
    metric_offset = dict(zip(QUALITY_PLOT_METRICS, (-0.045, 0.0, 0.045)))
    for sample in SAMPLES:
        for config_index, config in enumerate(QUALITY_CONFIGS):
            for metric, (marker, _) in metric_style.items():
                row = frame.loc[
                    frame["sample"].eq(sample) & frame["config"].eq(config)
                    & frame["metric"].eq(metric)
                ].iloc[0]
                horizontal_fraction(
                    ax, row, config_index + sample_offset[sample] + metric_offset[metric],
                    fmt=marker, ms=3.5, color=SAMPLE_COLORS[sample],
                    mec=SAMPLE_COLORS[sample],
                    mfc=("white" if marker == "D" else SAMPLE_COLORS[sample]),
                )
    ax.set_yticks(y, [config.replace("baseQ", "BQ").replace("mapQ", "/MQ") for config in QUALITY_CONFIGS])
    ax.invert_yaxis()
    ax.set_xlim(max(0, float(frame["ci95_low"].min()) - 0.04), 1.015)
    ax.set_xlabel("End-to-end fraction (Wilson 95% CI)")
    ax.set_title("b", loc="left", fontweight="bold")
    jaccard = data.pairwise.set_index("sample")["anchor_jaccard"]
    ax.text(
        0.99, 0.04,
        "Anchor-set Jaccard: " + "; ".join(f"{sample} {float(jaccard[sample]):.3f}" for sample in SAMPLES),
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
    )
    handles = [
        *(Line2D([0], [0], marker="o", lw=0, color=SAMPLE_COLORS[sample], label=sample, ms=4) for sample in SAMPLES),
        *(Line2D([0], [0], marker=marker, lw=0, color="black", mfc=("white" if marker == "D" else "black"), label=label, ms=4)
          for marker, label in metric_style.values()),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, ncol=2, loc="lower left")
    style_axis(ax)


def draw_replication(ax: plt.Axes, data: ReplicationData) -> None:
    y = np.arange(len(SAMPLES), dtype=float)
    metric_style = {
        "anchor_jaccard": ("o", "anchor-set Jaccard"),
        "route_agreement": ("D", "four-route agreement"),
    }
    offsets = {"anchor_jaccard": -0.10, "route_agreement": 0.10}
    point_offsets = (-0.025, 0.0, 0.025)
    for sample_index, sample in enumerate(SAMPLES):
        subset = data.stability.loc[data.stability["sample"].eq(sample)]
        for metric, (marker, _) in metric_style.items():
            values = np.sort(subset[metric].to_numpy(dtype=float))
            center = sample_index + offsets[metric]
            for jitter, value in zip(point_offsets, values):
                ax.plot(value, center + jitter, marker=marker, color=SAMPLE_COLORS[sample], ms=3.2, alpha=0.62)
            ax.plot(float(np.median(values)), center, marker="|", color="black", ms=9, mew=1.4)
    minimum = min(float(data.stability[metric].min()) for metric in metric_style)
    ax.set_xlim(max(0, minimum - max(0.002, (1 - minimum) * 0.16)), 1.001)
    ax.set_yticks(y, SAMPLES)
    ax.invert_yaxis()
    ax.set_xlabel("Agreement across the three seed pairs")
    ax.set_title("c", loc="left", fontweight="bold")
    ax.text(
        0.01, 0.04,
        f"Base: {data.base}\nSeeds: {', '.join(map(str, data.seeds))}\n"
        f"Base gate: {'PASS' if data.selected_gate['eligible_as_base_default'] else 'FAIL'}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.1,
    )
    handles = [
        Line2D([0], [0], marker=marker, lw=0, color="black", mfc=("white" if marker == "D" else "black"), label=label, ms=4)
        for marker, label in metric_style.values()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    style_axis(ax)


def draw_ensemble(ax: plt.Axes, data: EnsembleData) -> None:
    gates = data.gates.set_index("method").reindex(ENSEMBLE_METHODS)
    y = np.arange(len(ENSEMBLE_METHODS), dtype=float)
    ax.axvline(0, color="black", lw=0.7)
    styles = {
        "delta_mean_sensitivity": ("o", "delta germline sensitivity"),
        "delta_mean_support": ("D", "delta anchor support"),
    }
    offsets = {"delta_mean_sensitivity": -0.09, "delta_mean_support": 0.09}
    for index, method in enumerate(ENSEMBLE_METHODS):
        row = gates.loc[method]
        if bool(row["selected_final_representation"]):
            ax.axhspan(index - 0.36, index + 0.36, color="#E6F4EA", zorder=0)
        for metric, (marker, _) in styles.items():
            ax.plot(
                float(row[metric]), index + offsets[metric], marker=marker,
                color="#5B4B8A", mfc=("white" if marker == "D" else "#5B4B8A"), ms=4.2,
            )
        ax.text(
            0.99, index,
            f"leakage delta n={int(row['total_leakage_reduction']):+d}; "
            f"LOO J={float(row['minimum_leave_one_out_anchor_jaccard']):.3f}; "
            f"advance={'yes' if row['advance_over_single_seed'] else 'no'}",
            transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=5.8,
        )
    values = gates[["delta_mean_sensitivity", "delta_mean_support"]].to_numpy(dtype=float)
    extent = max(0.006, float(np.max(np.abs(values))) * 1.35)
    ax.set_xlim(-extent, extent)
    ax.set_yticks(y, ["posterior mean", "posterior median"])
    ax.invert_yaxis()
    ax.set_xlabel("Rate delta versus selected single seed")
    ax.set_title("d", loc="left", fontweight="bold")
    ax.text(
        0.01, 0.04,
        f"Final representation: {data.representation}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2, fontweight="bold",
    )
    handles = [
        Line2D([0], [0], marker=marker, lw=0, color="#5B4B8A", mfc=("white" if marker == "D" else "#5B4B8A"), label=label, ms=4)
        for marker, label in styles.values()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="lower right")
    style_axis(ax)


def draw_dlpfc(ax: plt.Axes, data: DlpfcData) -> None:
    sections = data.sections
    y = np.arange(len(SECTIONS), dtype=float)
    ci_low, ci_high = map(
        float, data.result["section_bootstrap_delta_ari_ci95_descriptive"]
    )
    ax.axvspan(ci_low, ci_high, color="#777777", alpha=0.09, zorder=0)
    for position, row in zip(y, sections.itertuples(index=False)):
        color = DONOR_COLORS[row.donor]
        ax.hlines(position, min(row.delta_ari, row.delta_nmi), max(row.delta_ari, row.delta_nmi), color=color, alpha=0.45, lw=0.8)
        ax.plot(row.delta_ari, position, "o", color=color, ms=3.7)
        ax.plot(row.delta_nmi, position, "D", mec=color, mfc="white", ms=3.5, mew=0.9)
    for boundary in (3.5, 7.5):
        ax.axhline(boundary, color="#BBBBBB", lw=0.55)
    ax.axvline(0, color="black", lw=0.7)
    ax.axvline(float(data.result["mean_delta_ari"]), color="#333333", lw=0.9)
    ax.axvline(float(data.result["mean_delta_nmi"]), color="#333333", lw=0.9, linestyle=":")
    values = np.r_[sections["delta_ari"], sections["delta_nmi"], 0]
    pad = max(0.006, (float(values.max()) - float(values.min())) * 0.12)
    ax.set_xlim(float(values.min()) - pad, float(values.max()) + pad)
    ax.set_yticks(y, SECTIONS)
    ax.invert_yaxis()
    ax.set_xlabel("Final interface - historical matrix")
    ax.set_title("e", loc="left", fontweight="bold")
    status = "PASS" if data.result["accept_replacement"] else "FAIL"
    color = "#007A3D" if data.result["accept_replacement"] else "#B22222"
    ax.text(
        0.99, 0.02,
        f"Strict gates: {status}\nmean delta ARI {float(data.result['mean_delta_ari']):+.3f}; "
        f"section-bootstrap 95% interval [{ci_low:+.3f}, {ci_high:+.3f}]\n"
        f"minimum donor mean {float(data.result['minimum_donor_mean_delta_ari']):+.3f}; "
        f"mean delta NMI {float(data.result['mean_delta_nmi']):+.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=5.9, color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )
    metric_handles = [
        Line2D([0], [0], marker="o", lw=0, color="black", label="delta ARI", ms=4),
        Line2D([0], [0], marker="D", lw=0, mec="black", mfc="white", color="black", label="delta NMI", ms=4),
    ]
    donor_handles = [
        Line2D([0], [0], marker="o", lw=0, color=value, label=key, ms=4)
        for key, value in DONOR_COLORS.items()
    ]
    first = ax.legend(handles=metric_handles, frameon=False, fontsize=5.7, loc="upper left", ncol=2)
    ax.add_artist(first)
    ax.legend(handles=donor_handles, frameon=False, fontsize=5.7, loc="upper right", ncol=3)
    style_axis(ax)


def draw_monopogen(ax: plt.Axes, data: MonopogenData) -> None:
    frame = data.summary
    y = np.arange(len(MONO_RUNS), dtype=float)
    ax.set_facecolor("#F4F1FA")
    styles = {
        "record": ("o", "GP/phased record-set retention"),
        "genotype": ("D", "unordered-GT concordance"),
    }
    for position, row in zip(y, frame.itertuples(index=False)):
        record_retention = int(row.phased_records) / int(row.gp_records)
        genotype_concordance = 1 - int(row.gp_phased_unordered_gt_changes) / int(row.gp_phased_comparable)
        ax.plot(record_retention, position - 0.08, "o", color="#5B4B8A", ms=4)
        ax.plot(genotype_concordance, position + 0.08, "D", mec="#5B4B8A", mfc="white", ms=4)
        ax.text(
            0.01, position,
            f"phase orientation: 0|1 {int(row.phased_0_pipe_1):,}; 1|0 {int(row.phased_1_pipe_0):,}",
            transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=5.7,
        )
    ax.set_xlim(0.965, 1.004)
    ax.set_yticks(y, MONO_RUNS)
    ax.invert_yaxis()
    ax.set_xlabel("Preserved fraction")
    ax.set_title("f", loc="left", fontweight="bold")
    ax.text(
        0.99, 0.04,
        "GP records = phased records; unordered GT changes = 0\n"
        "Second pass adds orientation, not evidence",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.1, fontweight="bold",
    )
    handles = [
        Line2D([0], [0], marker=marker, lw=0, color="#5B4B8A", mfc=("white" if marker == "D" else "#5B4B8A"), label=label, ms=4)
        for marker, label in styles.values()
    ]
    ax.legend(handles=handles, frameon=False, fontsize=5.8, loc="lower right")
    style_axis(ax)


def compose_figure(inputs: FigureInputs) -> tuple[plt.Figure, np.ndarray]:
    """Compose caption-addressable panels with bare letters and no prose titles."""
    fig, axes = plt.subplots(3, 2, figsize=(11.2, 10.7))
    draw_evidence(axes[0, 0], inputs.evidence)
    draw_quality(axes[0, 1], inputs.quality)
    draw_replication(axes[1, 0], inputs.replication)
    draw_ensemble(axes[1, 1], inputs.ensemble)
    draw_dlpfc(axes[2, 0], inputs.dlpfc)
    draw_monopogen(axes[2, 1], inputs.monopogen)
    fig.subplots_adjust(
        top=0.985, bottom=0.06, left=0.145, right=0.985,
        hspace=0.40, wspace=0.33,
    )
    return fig, axes


def render(inputs: FigureInputs, pdf: Path, png: Path, font_family: str) -> None:
    rc = {
        "font.family": font_family, "font.size": 7.2,
        "axes.labelsize": 7.2, "axes.titlesize": 8.8,
        "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
        "legend.fontsize": 5.9, "axes.linewidth": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }
    with plt.rc_context(rc):
        fig, _ = compose_figure(inputs)
        fig.savefig(
            pdf, format="pdf", bbox_inches="tight",
            metadata={
                "Title": "Beagle-v2.3 evidence routing supplementary validation",
                "Creator": "plot_beagle_module_supplement.py",
                "CreationDate": None, "ModDate": None,
            },
        )
        fig.savefig(
            png, format="png", dpi=300, bbox_inches="tight",
            metadata={"Software": "plot_beagle_module_supplement.py"},
        )
        plt.close(fig)


def write_hashes(path: Path, files: Iterable[Path], *, relative_to: Path | None = None) -> None:
    rows = []
    for member in sorted(set(Path(value).resolve() for value in files), key=str):
        label: Path = member
        if relative_to is not None:
            try:
                label = member.relative_to(relative_to.resolve())
            except ValueError:
                pass
        rows.append(f"{sha256(member)}  {label}")
    path.write_text("\n".join(rows) + "\n")


def generate_bundle(
    *, evidence_dir: Path, replication_dir: Path, ensemble_dir: Path,
    quality_dir: Path, dlpfc_dir: Path, monopogen_dir: Path, output_dir: Path,
    output_stem: str = "beagle_v2_3_supplement", font_family: str = "Arial",
    font_file: Path | None = None, font_dir: Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_stem):
        raise ValueError("unsafe output stem")
    inputs = load_inputs(
        evidence_dir, replication_dir, ensemble_dir, quality_dir, dlpfc_dir,
        monopogen_dir,
    )
    resolved_family, resolved_fonts = resolve_font(font_family, font_file, font_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    plotted_path = output_dir / "plotted_data.tsv"
    build_plotted_data(inputs).to_csv(plotted_path, sep="\t", index=False, na_rep="NA")
    pdf = output_dir / f"{output_stem}.pdf"
    png = output_dir / f"{output_stem}.png"
    render(inputs, pdf, png, resolved_family)
    script = Path(__file__).resolve()
    code_input_ledger = output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_hashes(code_input_ledger, (script, *resolved_fonts, *inputs.paths))
    provenance = {
        "schema": "sparcal_beagle_v2_3_supplement_figure_v1",
        "status": "figure_rendered",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "matched_wes_role": "internal development evaluation, not untouched held-out validation",
        "dlpfc_role": "functional STAGATE-to-fixed-EEE-mclust no-harm, not SNV-caller accuracy",
        "dlpfc_primary_interface": "final all_panel_present feature-support compatibility",
        "pnonref_anchor_role": "secondary diagnostic; not the primary DLPFC clustering arm",
        "monopogen_scope": "separate input-universe Beagle mechanism audit; not a same-input performance comparison",
        "monopogen_mechanism_conclusion": "GP and phased record sets are equal and unordered GT changes are zero; second pass adds orientation, not evidence",
        "same_input_methods": list(EVIDENCE_METHODS),
        "production_routing_rule": "P(nonref)=GP(0/1)+GP(1/1)>=0.90",
        "selected_base_configuration": inputs.replication.base,
        "replication_seeds": list(inputs.replication.seeds),
        "selected_final_representation": inputs.ensemble.representation,
        "strict_dlpfc_gates": inputs.dlpfc.result["gates"],
        "strict_dlpfc_accept_replacement": inputs.dlpfc.result["accept_replacement"],
        "font": {
            "requested_family": font_family, "resolved_family": resolved_family,
            "faces": [
                {
                    "path": str(path), "style": str(FT2Font(str(path)).style_name),
                    "sha256": sha256(path),
                }
                for path in resolved_fonts
            ],
        },
        "software": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
        },
        "code": {"path": str(script), "sha256": sha256(script)},
        "inputs": [
            {"path": str(path), "sha256": sha256(path)} for path in inputs.paths
        ],
        "outputs": {
            "pdf": pdf.name, "png_300dpi": png.name,
            "plotted_data": plotted_path.name,
        },
    }
    provenance_path = output_dir / "PROVENANCE.json"
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    output_ledger = output_dir / "OUTPUT_SHA256SUMS.txt"
    write_hashes(
        output_ledger,
        (pdf, png, plotted_path, provenance_path, code_input_ledger),
        relative_to=output_dir,
    )
    (output_dir / "RUN_COMPLETE").write_text("complete\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path, required=True)
    parser.add_argument("--ensemble-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--dlpfc-dir", type=Path, required=True)
    parser.add_argument("--monopogen-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default="beagle_v2_3_supplement")
    parser.add_argument("--font-family", default="Arial")
    parser.add_argument("--font-file", type=Path)
    parser.add_argument("--font-dir", type=Path)
    args = parser.parse_args()
    generate_bundle(
        evidence_dir=args.evidence_dir, replication_dir=args.replication_dir,
        ensemble_dir=args.ensemble_dir, quality_dir=args.quality_dir,
        dlpfc_dir=args.dlpfc_dir, monopogen_dir=args.monopogen_dir,
        output_dir=args.output_dir, output_stem=args.output_stem,
        font_family=args.font_family, font_file=args.font_file,
        font_dir=args.font_dir,
    )


if __name__ == "__main__":
    main()
