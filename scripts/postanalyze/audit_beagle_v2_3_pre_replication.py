#!/usr/bin/env python3
"""Recompute and certify all Beagle-v2.3 seed-screen decisions.

This is an independent, fail-closed bridge between the already-frozen screen
selectors and the replication planner.  It never trusts their metric tables:
canonical v2.3 run bundles and routing audits are validated again, frozen truth
is re-read, all gate metrics and decisions are recomputed, and the three
existing selector bundles must agree exactly before an admitted configuration
list is emitted.

Matched WES is used only for development-set assessment.  It is never a Beagle
or production-router input.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
POSTANALYZE = Path(__file__).resolve().parent
if str(POSTANALYZE) not in sys.path:
    sys.path.insert(0, str(POSTANALYZE))
import beagle_selection_integrity as INTEGRITY  # noqa: E402


SAMPLES = ("P4", "P6")
SEED = 20260915
COMPARATOR = f"ms2_niter0_seed{SEED}"
FACTORIAL = {
    COMPARATOR: {
        "base": "ms2_niter0", "modelscale": 2.0, "niterations": 0,
        "maxlr": 5000.0, "one_factor": False,
    },
    f"ms2_niter5_seed{SEED}": {
        "base": "ms2_niter5", "modelscale": 2.0, "niterations": 5,
        "maxlr": 5000.0, "one_factor": True,
    },
    f"ms0p8_niter0_seed{SEED}": {
        "base": "ms0p8_niter0", "modelscale": 0.8, "niterations": 0,
        "maxlr": 5000.0, "one_factor": True,
    },
    f"ms0p8_niter5_seed{SEED}": {
        "base": "ms0p8_niter5", "modelscale": 0.8, "niterations": 5,
        "maxlr": 5000.0, "one_factor": False,
    },
}
MODELSCALE = f"ms4_niter0_seed{SEED}"
MAXLR = f"ms2_niter0_maxlr50000_seed{SEED}"
SETTINGS = {
    **FACTORIAL,
    MODELSCALE: {
        "base": "ms4_niter0", "modelscale": 4.0, "niterations": 0,
        "maxlr": 5000.0, "one_factor": False,
    },
    MAXLR: {
        "base": "ms2_niter0_maxlr50000", "modelscale": 2.0,
        "niterations": 0, "maxlr": 50000.0, "one_factor": False,
    },
}
GATE_METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
ALL_METRICS = (
    "raw_candidate_normal_wes_recall",
    "normal_wes_end_to_end_anchor_recall",
    "normal_wes_anchor_sensitivity",
    "normal_wes_anchor_gt_accuracy",
    "selected_anchor_normal_wes_support",
    "raw_candidate_wes_somatic_recall",
    "wes_somatic_end_to_end_retention",
    "wes_somatic_anchor_leakage",
    "wes_somatic_retention",
    "raw_candidate_table_s1_recall",
    "table_s1_end_to_end_retention",
    "table_s1_somatic_anchor_leakage",
)
SENSITIVITY_MARGIN = 0.005
SUPPORT_MARGIN = 0.005
IMPROVEMENT_MARGIN = 0.002
FLOAT_TOLERANCE = 1e-12
TRUTH_INPUTS = {
    "normal": PROJECT
    / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz",
    "support": PROJECT
    / "data/germline_concordance_2026-08-23/sparcal_call_details.csv.gz",
    "somatic": PROJECT
    / "data/somatic_evidence_2026-07-28/wes_confirmed_full_accounting.csv",
    "table_s1": PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv",
}
CANDIDATE_DIR = (
    PROJECT / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
)
DEFAULT_CONTRACTS = CANDIDATE_DIR / "contracts.py"
DEFAULT_BENCHMARK = PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py"
DEFAULT_REFERENCE_LEDGER = (
    PROJECT / "data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen"
)
PROTOCOLS = {
    "factorial": CANDIDATE_DIR / "BENCHMARK_PROTOCOL.md",
    "modelscale": CANDIDATE_DIR / "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md",
    "maxlr": CANDIDATE_DIR / "BENCHMARK_PROTOCOL_MAXLR.md",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    path = Path(path).resolve()
    stat = path.stat()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def load_module(path: Path, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def stable_content(value: object) -> object:
    """Remove run-local locations/timestamps while retaining content identity."""
    if isinstance(value, Mapping):
        return {
            key: stable_content(item)
            for key, item in sorted(value.items())
            if key not in {"path", "mtime_ns", "raw_panel_exact_alleles"}
        }
    if isinstance(value, list):
        return [stable_content(item) for item in value]
    return value


def read_raw_alleles(path: Path) -> set[tuple[str, int, str, str]]:
    """Read the exact biallelic raw allele universe from a bgzipped VCF."""
    alleles: set[tuple[str, int, str, str]] = set()
    header = False
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                if header:
                    raise ValueError(f"duplicate raw VCF header: {path}")
                header = True
                continue
            if line.startswith("#"):
                continue
            if not header:
                raise ValueError(f"raw VCF record precedes header: {path}:{line_number}")
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 5 or "," in fields[4]:
                raise ValueError(f"raw VCF is not biallelic: {path}:{line_number}")
            try:
                key = (fields[0], int(fields[1]), fields[3], fields[4])
            except ValueError as error:
                raise ValueError(f"invalid raw VCF position: {path}:{line_number}") from error
            if key[1] <= 0 or key[2] in {"", "."} or key[3] in {"", "."}:
                raise ValueError(f"invalid raw VCF allele: {path}:{line_number}")
            if key in alleles:
                raise ValueError(f"duplicate raw exact allele: {path}:{line_number}")
            alleles.add(key)
    if not header or not alleles:
        raise ValueError(f"raw VCF has no exact alleles: {path}")
    return alleles


def allele_digest(values: set[tuple[str, int, str, str]]) -> str:
    digest = hashlib.sha256()
    for chrom, pos, ref, alt in sorted(values):
        digest.update(f"{chrom}\t{pos}\t{ref}\t{alt}\n".encode())
    return digest.hexdigest()


def require_exact_allele_universe(
    *,
    sample: str,
    config: str,
    raw: set[tuple[str, int, str, str]],
    audit: set[tuple[str, int, str, str]],
    prior_audit: set[tuple[str, int, str, str]] | None,
) -> None:
    """Require raw/audit equality and one exact universe across screen arms."""
    if audit != raw:
        missing = len(raw - audit)
        extra = len(audit - raw)
        raise ValueError(
            "canonical audit exact-allele universe differs from raw VCF: "
            f"{config}/{sample}; missing={missing}, extra={extra}"
        )
    if prior_audit is not None and audit != prior_audit:
        raise ValueError(
            f"canonical audit exact-allele universe differs across arms: {sample}"
        )


def load_truth() -> tuple[dict[str, pd.DataFrame], list[Path]]:
    for path in TRUTH_INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    normal = pd.read_csv(TRUTH_INPUTS["normal"])
    normal = normal.loc[(normal["detected"] == 1) & (normal["rna_dp"] >= 1)].copy()
    return (
        {
            "normal": normal,
            "support": pd.read_csv(TRUTH_INPUTS["support"]),
            "somatic": pd.read_csv(TRUTH_INPUTS["somatic"]),
            "table_s1": pd.read_csv(TRUTH_INPUTS["table_s1"], sep="\t"),
        },
        list(TRUTH_INPUTS.values()),
    )


def source_root(
    config: str,
    sample: str,
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
) -> Path:
    parent = (
        extension_root if config == MODELSCALE
        else maxlr_root if config == MAXLR
        else factorial_root
    )
    return Path(parent).resolve() / config / f"{sample}_sec1"


def validate_and_recompute(
    *,
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
    contracts,
    benchmark,
    reference_ledger_root: Path,
) -> tuple[pd.DataFrame, list[dict[str, object]], list[Path]]:
    truth, truth_paths = load_truth()
    metrics: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    bound_paths: list[Path] = []
    invariant_signature: object | None = None
    raw_identity: dict[str, tuple[tuple[object, ...], tuple[object, ...]]] = {}
    raw_universe: dict[str, set[tuple[str, int, str, str]]] = {}
    audit_universe: dict[str, set[tuple[str, int, str, str]]] = {}
    reference_ledger_root = Path(reference_ledger_root).resolve()

    for config, setting in SETTINGS.items():
        for sample in SAMPLES:
            root = source_root(
                config, sample, factorial_root, extension_root, maxlr_root
            )
            if (root / "RUN_FAILED.txt").exists():
                raise ValueError(f"run carries a failure marker: {root}")
            validation = contracts.validate_v2_3_run(root)
            if validation.get("status") != "verified":
                raise ValueError(f"v2.3 run is not verified: {root}")
            manifest_path = root / "V2_3_MANIFEST.json"
            completion_path = root / "RUN_COMPLETE.json"
            manifest = json.loads(manifest_path.read_text())
            params = manifest.get("beagle_parameters", {})
            expected_params = {
                "MODEL_SCALE": setting["modelscale"],
                "ITERATIONS": setting["niterations"],
                "SEED": SEED,
                "IMPUTE": False,
                "GPROBS": True,
                "MAX_LR": setting["maxlr"],
                "GENETIC_MAP": None,
                "THREADS": 30,
                "PARALLEL_CHROMOSOMES": 1,
                "MEMORY": "10g",
            }
            if (
                manifest.get("production_inputs_include_wes") is not False
                or manifest.get("evaluation_uses_wes") is not False
                or manifest.get("dataset") != f"{sample}_TUMOR"
                or str(manifest.get("section_id")) != "1"
                or manifest.get("quality_filter") != "baseQ0mapQ0"
                or manifest.get("anchor_score") != "p_nonref"
                or not math.isclose(
                    float(manifest.get("min_anchor_probability")),
                    0.90,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(f"invalid WES/routing/run contract: {root}")
            for key, expected in expected_params.items():
                if params.get(key) != expected:
                    raise ValueError(
                        f"parameter mismatch {root}/{key}: "
                        f"{params.get(key)!r} != {expected!r}"
                    )
            if (
                Path(str(params.get("REFERENCE_LEDGER", ""))).resolve()
                != reference_ledger_root
                or Path(str(manifest["reference_ledger"]["root"])).resolve()
                != reference_ledger_root
            ):
                raise ValueError(f"reference ledger root mismatch: {root}")

            signature = stable_content({
                "code": manifest.get("code"),
                "executables": manifest.get("executables"),
                "reference_ledger": manifest.get("reference_ledger"),
                "reference_panels": manifest.get("reference_panels"),
            })
            if invariant_signature is None:
                invariant_signature = signature
            elif signature != invariant_signature:
                raise ValueError(
                    "screen runs do not share one code/executable/reference identity"
                )

            raw_path = Path(str(manifest["raw_vcf"]["path"])).resolve()
            raw_index_path = Path(str(manifest["raw_vcf_index"]["path"])).resolve()
            raw_record = identity(raw_path)
            raw_index_record = identity(raw_index_path)
            for label, expected, observed in (
                ("raw VCF", manifest["raw_vcf"], raw_record),
                ("raw VCF index", manifest["raw_vcf_index"], raw_index_record),
            ):
                for field in ("path", "size_bytes", "mtime_ns", "sha256"):
                    if expected.get(field) != observed[field]:
                        raise ValueError(
                            f"{label} manifest identity mismatch for {field}: {root}"
                        )
            observed_raw = (
                tuple(
                    raw_record[field]
                    for field in ("path", "size_bytes", "mtime_ns", "sha256")
                ),
                tuple(
                    raw_index_record[field]
                    for field in ("path", "size_bytes", "mtime_ns", "sha256")
                ),
            )
            if sample in raw_identity and raw_identity[sample] != observed_raw:
                raise ValueError(f"raw VCF or index identity differs for {sample}")
            raw_identity[sample] = observed_raw
            if sample not in raw_universe:
                raw_universe[sample] = read_raw_alleles(raw_path)

            audit = (
                root / "output_VCFs" / "beagle" / "baseQ0mapQ0"
                / "beagle_evidence_audit.tsv.gz"
            ).resolve()
            if not audit.is_file():
                raise FileNotFoundError(audit)
            evidence = benchmark.reroute_evidence(
                benchmark.read_evidence(audit), "p_nonref", 0.90
            )
            keys = set(evidence)
            require_exact_allele_universe(
                sample=sample,
                config=config,
                raw=raw_universe[sample],
                audit=keys,
                prior_audit=audit_universe.get(sample),
            )
            audit_universe[sample] = keys
            rows, _ = benchmark.evaluate(
                config,
                sample,
                evidence,
                truth["normal"],
                truth["support"],
                truth["somatic"],
                truth["table_s1"],
            )
            metrics.extend(rows)
            bound_paths.extend(
                (manifest_path, completion_path, audit, raw_path, raw_index_path)
            )
            records.append({
                "config": config,
                "sample": sample,
                "run_root": str(root),
                "manifest": identity(manifest_path),
                "completion": identity(completion_path),
                "audit": identity(audit),
                "raw_vcf": identity(raw_path),
                "raw_vcf_index": identity(raw_index_path),
                "exact_allele_count": len(keys),
                "exact_allele_sha256": allele_digest(keys),
                "validation": validation,
            })

    frame = pd.DataFrame(metrics)
    validated = validate_metric_grid(frame, tuple(SETTINGS), ALL_METRICS)
    chosen = validated.loc[
        validated["metric"].isin(GATE_METRICS)
    ].copy()
    return chosen, records, [*truth_paths, *bound_paths]


def validate_metric_grid(
    metrics: pd.DataFrame,
    configs: Sequence[str],
    expected_metrics: Sequence[str] = GATE_METRICS,
) -> pd.DataFrame:
    required = (
        "config",
        "sample",
        "metric",
        "numerator",
        "denominator",
        "value",
        "ci95_low",
        "ci95_high",
    )
    if set(metrics.columns) != set(required) or len(metrics.columns) != len(required):
        raise ValueError("recomputed metric columns are not exact")
    expected = {
        (config, sample, metric)
        for config in configs
        for sample in SAMPLES
        for metric in expected_metrics
    }
    observed = set(
        metrics[["config", "sample", "metric"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(metrics) != len(expected):
        raise ValueError("recomputed metric grid is not exact")
    chosen = metrics.copy()
    for column in (
        "numerator", "denominator", "value", "ci95_low", "ci95_high"
    ):
        chosen[column] = pd.to_numeric(chosen[column], errors="raise")
    counts = [*chosen["numerator"], *chosen["denominator"]]
    if (
        not all(
            math.isfinite(float(count)) and float(count).is_integer()
            for count in counts
        )
        or (chosen["denominator"] < 0).any()
        or (chosen["numerator"] < 0).any()
        or (chosen["numerator"] > chosen["denominator"]).any()
        or (
            chosen.loc[
                chosen["metric"].isin(GATE_METRICS), "denominator"
            ] <= 0
        ).any()
    ):
        raise ValueError("recomputed metrics contain invalid counts")

    def value_matches_counts(row: pd.Series) -> bool:
        denominator = int(row["denominator"])
        if denominator == 0:
            return pd.isna(row["value"])
        return (
            math.isfinite(float(row["value"]))
            and math.isclose(
                float(row["value"]),
                int(row["numerator"]) / denominator,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )

    if not chosen.apply(value_matches_counts, axis=1).all():
        raise ValueError("recomputed metric values do not equal counts")

    def confidence_interval_is_valid(row: pd.Series) -> bool:
        denominator = int(row["denominator"])
        if denominator == 0:
            return pd.isna(row["ci95_low"]) and pd.isna(row["ci95_high"])
        value = float(row["value"])
        low = float(row["ci95_low"])
        high = float(row["ci95_high"])
        return (
            math.isfinite(low)
            and math.isfinite(high)
            # Wilson endpoints can cross an exact 0/1 point estimate by a
            # few ulps (for example, low=2.78e-17 for 0/11). Accept only
            # numerical round-off here; the counts and point estimate remain
            # validated independently and exactly above.
            and -FLOAT_TOLERANCE <= low <= value + FLOAT_TOLERANCE
            and value - FLOAT_TOLERANCE <= high <= 1.0 + FLOAT_TOLERANCE
        )

    if not chosen.apply(confidence_interval_is_valid, axis=1).all():
        raise ValueError("recomputed metric confidence intervals are invalid")
    return chosen


def compute_gate_tables(
    metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = validate_metric_grid(metrics, tuple(SETTINGS))
    lookup = metrics.set_index(["config", "sample", "metric"])

    def value(config: str, sample: str, metric: str) -> float:
        return float(lookup.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "numerator"])

    def denominator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "denominator"])

    comparator_sensitivity = sum(
        value(COMPARATOR, sample, GATE_METRICS[0]) for sample in SAMPLES
    ) / len(SAMPLES)
    comparator_support = sum(
        value(COMPARATOR, sample, GATE_METRICS[1]) for sample in SAMPLES
    ) / len(SAMPLES)
    comparator_leakage = {
        sample: numerator(COMPARATOR, sample, GATE_METRICS[2]) for sample in SAMPLES
    }

    def gate(config: str, allow_one_factor: bool) -> dict[str, object]:
        for sample in SAMPLES:
            if denominator(config, sample, GATE_METRICS[2]) != denominator(
                COMPARATOR, sample, GATE_METRICS[2]
            ):
                raise ValueError(f"somatic denominator differs for {config}/{sample}")
        setting = SETTINGS[config]
        sensitivity = sum(
            value(config, sample, GATE_METRICS[0]) for sample in SAMPLES
        ) / len(SAMPLES)
        support = sum(
            value(config, sample, GATE_METRICS[1]) for sample in SAMPLES
        ) / len(SAMPLES)
        leakage = {
            sample: numerator(config, sample, GATE_METRICS[2]) for sample in SAMPLES
        }
        sensitivity_delta = sensitivity - comparator_sensitivity
        support_delta = support - comparator_support
        leakage_reduction = sum(comparator_leakage.values()) - sum(leakage.values())
        leakage_nonincrease = all(
            leakage[sample] <= comparator_leakage[sample] for sample in SAMPLES
        )
        sensitivity_no_harm = sensitivity_delta >= -SENSITIVITY_MARGIN
        support_no_harm = support_delta >= -SUPPORT_MARGIN
        material = (
            sensitivity_delta >= IMPROVEMENT_MARGIN
            or support_delta >= IMPROVEMENT_MARGIN
            or leakage_reduction > 0
        )
        one_factor = bool(setting["one_factor"]) and allow_one_factor
        advance = config == COMPARATOR or (
            leakage_nonincrease
            and sensitivity_no_harm
            and support_no_harm
            and (material or one_factor)
        )
        return {
            "config": config,
            "modelscale": setting["modelscale"],
            "niterations": setting["niterations"],
            "comparator": config == COMPARATOR,
            "required_one_factor_ablation": bool(setting["one_factor"]),
            "mean_normal_wes_anchor_sensitivity": sensitivity,
            "delta_mean_sensitivity_vs_comparator": sensitivity_delta,
            "sensitivity_no_harm": sensitivity_no_harm,
            "mean_selected_anchor_normal_wes_support": support,
            "delta_mean_support_vs_comparator": support_delta,
            "support_no_harm": support_no_harm,
            "P4_somatic_leakage_n": leakage["P4"],
            "P6_somatic_leakage_n": leakage["P6"],
            "P4_leakage_nonincrease": leakage["P4"] <= comparator_leakage["P4"],
            "P6_leakage_nonincrease": leakage["P6"] <= comparator_leakage["P6"],
            "leakage_nonincrease_both": leakage_nonincrease,
            "total_leakage_reduction": leakage_reduction,
            "material_improvement": material,
            "advance_to_replication": advance,
        }

    factorial = pd.DataFrame([gate(config, True) for config in FACTORIAL])

    def extension_row(config: str) -> dict[str, object]:
        full = gate(config, False)
        return {
            "config": config,
            "comparator": COMPARATOR,
            **{
                key: full[key]
                for key in (
                    "mean_normal_wes_anchor_sensitivity",
                    "delta_mean_sensitivity_vs_comparator",
                    "sensitivity_no_harm",
                    "mean_selected_anchor_normal_wes_support",
                    "delta_mean_support_vs_comparator",
                    "support_no_harm",
                    "P4_somatic_leakage_n",
                    "P6_somatic_leakage_n",
                    "P4_leakage_nonincrease",
                    "P6_leakage_nonincrease",
                    "leakage_nonincrease_both",
                    "total_leakage_reduction",
                    "material_improvement",
                    "advance_to_replication",
                )
            },
        }

    modelscale = pd.DataFrame([extension_row(MODELSCALE)])
    maxlr = pd.DataFrame([extension_row(MAXLR)])
    return factorial, modelscale, maxlr


SELECTION_KINDS = {
    "factorial": {
        "gate": "screen_gate_results.tsv",
        "completion_schema": "sparcal_beagle_v2_3_screen_selection_completion_v1",
        "provenance_schema": "sparcal_beagle_v2_3_screen_selection_v1",
        "extra_outputs": (),
    },
    "modelscale": {
        "gate": "extension_gate_result.tsv",
        "completion_schema": "sparcal_beagle_v2_3_modelscale_extension_completion_v1",
        "provenance_schema": "sparcal_beagle_v2_3_modelscale_extension_selection_v1",
        "extra_outputs": (),
    },
    "maxlr": {
        "gate": "maxlr_gate_result.tsv",
        "completion_schema": "sparcal_beagle_v2_3_maxlr_extension_completion_v1",
        "provenance_schema": "sparcal_beagle_v2_3_maxlr_extension_selection_v1",
        "extra_outputs": (
            "matched_configuration_metrics_long.tsv",
            "matched_configuration_summary.tsv",
        ),
    },
}


def verify_selection_bundle(
    root: Path, kind: str
) -> tuple[pd.DataFrame, Mapping[str, object], Mapping[str, object], list[Path]]:
    root = Path(root).resolve()
    spec = SELECTION_KINDS[kind]
    names = (
        str(spec["gate"]),
        "PROVENANCE.json",
        "CODE_AND_INPUT_SHA256SUMS.txt",
        "OUTPUT_SHA256SUMS.txt",
        "RUN_COMPLETE",
        "RUN_COMPLETE.json",
        *spec["extra_outputs"],
    )
    paths = [root / name for name in names]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"{kind} selector is incomplete: {missing}")
    if (root / "RUN_COMPLETE").read_text().strip() != "complete":
        raise ValueError(f"{kind} selector has invalid RUN_COMPLETE")
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    completion = json.loads((root / "RUN_COMPLETE.json").read_text())
    if (
        provenance.get("schema") != spec["provenance_schema"]
        or completion.get("schema") != spec["completion_schema"]
        or completion.get("status") != "complete"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("final_default_selected") is not False
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
    ):
        raise ValueError(f"{kind} selector status/WES contract is invalid")
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=(
            str(spec["gate"]),
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
            *spec["extra_outputs"],
        ),
        label=f"{kind} selector",
    )
    INTEGRITY.parse_and_verify_ledger(
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label=f"{kind} selector code/input ledger",
    )
    return (
        pd.read_csv(root / str(spec["gate"]), sep="\t"),
        provenance,
        completion,
        paths,
    )


def exact_gate_compare(observed: pd.DataFrame, expected: pd.DataFrame, label: str) -> None:
    if list(observed.columns) != list(expected.columns) or len(observed) != len(expected):
        raise ValueError(f"{label} gate table schema/row count differs from recomputation")
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            # TSV round-tripping can change the last binary digit of a
            # derived mean (for example, 0.9650524501466777 becomes
            # 0.9650524501466776). Decisions and nonnumeric fields remain
            # exact; numeric drift beyond floating-point round-off fails.
            check_exact=False,
            rtol=0,
            atol=FLOAT_TOLERANCE,
        )
    except AssertionError as error:
        raise ValueError(f"{label} gate table differs from recomputation") from error


def validate_selector_decisions(
    *,
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
    factorial_gate: pd.DataFrame,
    modelscale_gate: pd.DataFrame,
    maxlr_gate: pd.DataFrame,
) -> tuple[list[str], list[Path]]:
    inputs: list[Path] = []
    bundles = {}
    for kind, root in (
        ("factorial", factorial_root),
        ("modelscale", extension_root),
        ("maxlr", maxlr_root),
    ):
        gate, provenance, completion, paths = verify_selection_bundle(root, kind)
        bundles[kind] = (gate, provenance, completion)
        inputs.extend(paths)
    exact_gate_compare(bundles["factorial"][0], factorial_gate, "factorial")
    exact_gate_compare(bundles["modelscale"][0], modelscale_gate, "modelscale")
    exact_gate_compare(bundles["maxlr"][0], maxlr_gate, "maxlr")

    factorial_selected = factorial_gate.loc[
        factorial_gate["advance_to_replication"], "config"
    ].tolist()
    if COMPARATOR not in factorial_selected:
        raise ValueError("recomputed factorial decision lacks comparator")
    fact_prov, fact_complete = bundles["factorial"][1:]
    if (
        fact_prov.get("comparator") != COMPARATOR
        or fact_prov.get("replicate_configurations") != factorial_selected
        or fact_complete.get("replicate_configurations") != factorial_selected
    ):
        raise ValueError("factorial selector decisions differ from recomputation")

    admitted = [str(SETTINGS[config]["base"]) for config in factorial_selected]
    for kind, config, gate in (
        ("modelscale", MODELSCALE, modelscale_gate),
        ("maxlr", MAXLR, maxlr_gate),
    ):
        advance = bool(gate.iloc[0]["advance_to_replication"])
        expected = [config] if advance else []
        provenance, completion = bundles[kind][1:]
        if (
            provenance.get("comparator") != COMPARATOR
            or provenance.get("advance_to_replication") is not advance
            or completion.get("replicate_configurations") != expected
            or (
                kind == "modelscale"
                and provenance.get("extension") != config
            )
            or (
                kind == "maxlr"
                and (
                    provenance.get("candidate") != config
                    or provenance.get("replicate_configurations") != expected
                )
            )
        ):
            raise ValueError(f"{kind} selector decisions differ from recomputation")
        if advance:
            admitted.append(str(SETTINGS[config]["base"]))
    if len(admitted) != len(set(admitted)):
        raise ValueError("recomputed admitted configuration list contains duplicates")
    return admitted, inputs

AUDIT_MEMBER_NAMES = (
    "recomputed_gate_metrics_long.tsv",
    "factorial_gate_results_recomputed.tsv",
    "modelscale4_gate_result_recomputed.tsv",
    "maxlr_gate_result_recomputed.tsv",
    "ADMITTED_CONFIGURATIONS.json",
    "PROVENANCE.json",
    "CODE_AND_INPUT_SHA256SUMS.txt",
)


def verify_embedded_identities(value: object, label: str = "provenance") -> None:
    """Verify every exact identity record embedded in a provenance tree."""
    if isinstance(value, Mapping):
        if set(value) == {"path", "size_bytes", "mtime_ns", "sha256"}:
            INTEGRITY.verify_identity(value, label=label)
            return
        for key, item in value.items():
            verify_embedded_identities(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            verify_embedded_identities(item, f"{label}[{index}]")


def verify_audit_bundle(root: Path) -> list[str]:
    """Fail closed on a completed pre-replication recomputation bundle."""
    root = Path(root)
    if root.is_symlink():
        raise ValueError("pre-replication audit root must not be a symlink")
    expected_files = {
        *AUDIT_MEMBER_NAMES,
        "OUTPUT_SHA256SUMS.txt",
        "RUN_COMPLETE",
        "RUN_COMPLETE.json",
    }
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries = list(root.iterdir())
    observed_files = {path.name for path in entries if path.is_file()}
    if (
        any(path.is_symlink() or not path.is_file() for path in entries)
        or observed_files != expected_files
        or len(entries) != len(expected_files)
    ):
        raise ValueError("pre-replication audit bundle file set is not exact")
    if (root / "RUN_COMPLETE").read_text().strip() != "complete":
        raise ValueError("pre-replication audit RUN_COMPLETE marker is invalid")

    completion = json.loads((root / "RUN_COMPLETE.json").read_text())
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    admitted_payload = json.loads(
        (root / "ADMITTED_CONFIGURATIONS.json").read_text()
    )
    if (
        completion.get("schema")
        != "sparcal_beagle_v2_3_pre_replication_audit_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or completion.get("screen_seed") != SEED
        or provenance.get("schema")
        != "sparcal_beagle_v2_3_pre_replication_audit_v1"
        or provenance.get("status") != "selector_decisions_verified"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("development_set_evaluation") is not True
        or provenance.get("screen_seed") != SEED
        or provenance.get("samples") != list(SAMPLES)
        or provenance.get("expected_configurations") != list(SETTINGS)
        or admitted_payload.get("schema")
        != "sparcal_beagle_v2_3_pre_replication_admitted_configurations_v1"
        or admitted_payload.get("status")
        != "screen_decisions_independently_recomputed"
        or admitted_payload.get("production_inputs_include_wes") is not False
        or admitted_payload.get("evaluation_uses_wes") is not True
        or admitted_payload.get("screen_seed") != SEED
    ):
        raise ValueError("pre-replication audit schema/status contract is invalid")

    INTEGRITY.verify_identity(
        completion.get("admitted_configurations"),
        label="pre-replication admitted configurations",
        expected_path=root / "ADMITTED_CONFIGURATIONS.json",
    )
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=AUDIT_MEMBER_NAMES,
        label="pre-replication audit",
    )
    INTEGRITY.parse_and_verify_ledger(
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label="pre-replication audit code/input ledger",
    )
    verify_embedded_identities(provenance)

    metrics = pd.read_csv(
        root / "recomputed_gate_metrics_long.tsv", sep="\t"
    )
    expected_gates = compute_gate_tables(metrics)
    gate_paths = (
        root / "factorial_gate_results_recomputed.tsv",
        root / "modelscale4_gate_result_recomputed.tsv",
        root / "maxlr_gate_result_recomputed.tsv",
    )
    for label, path, expected in zip(
        ("factorial", "modelscale", "maxlr"), gate_paths, expected_gates
    ):
        exact_gate_compare(pd.read_csv(path, sep="\t"), expected, label)

    factorial, modelscale, maxlr = expected_gates
    selected_screen = factorial.loc[
        factorial["advance_to_replication"], "config"
    ].tolist()
    admitted = [str(SETTINGS[config]["base"]) for config in selected_screen]
    for config, gate in ((MODELSCALE, modelscale), (MAXLR, maxlr)):
        if bool(gate.iloc[0]["advance_to_replication"]):
            admitted.append(str(SETTINGS[config]["base"]))
    if (
        COMPARATOR not in selected_screen
        or len(admitted) != len(set(admitted))
        or admitted_payload.get("admitted_base_configurations") != admitted
        or provenance.get("admitted_base_configurations") != admitted
        or completion.get("admitted_base_configurations") != admitted
    ):
        raise ValueError("pre-replication admitted configuration list is invalid")
    return admitted



def write_ledger(paths: Sequence[Path], output: Path, relative_root: Path | None = None) -> None:
    resolved = sorted(set(Path(path).resolve() for path in paths), key=str)
    rows = []
    for path in resolved:
        if relative_root is None:
            label: Path | str = path
        else:
            try:
                label = path.relative_to(Path(relative_root).resolve())
            except ValueError:
                label = path
        rows.append(f"{sha256(path)}  {label}")
    output.write_text("\n".join(rows) + "\n")


def run_audit(
    *,
    factorial_experiment_root: Path,
    factorial_selection_root: Path,
    extension_experiment_root: Path,
    extension_selection_root: Path,
    maxlr_experiment_root: Path,
    maxlr_selection_root: Path,
    output_dir: Path,
    contracts_path: Path = DEFAULT_CONTRACTS,
    benchmark_path: Path = DEFAULT_BENCHMARK,
    reference_ledger_root: Path = DEFAULT_REFERENCE_LEDGER,
) -> Mapping[str, object]:
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    required_code = [contracts_path, benchmark_path, *PROTOCOLS.values()]
    for path in required_code:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not (Path(reference_ledger_root) / "RUN_COMPLETE.json").is_file():
        raise FileNotFoundError(reference_ledger_root)
    contracts = load_module(contracts_path, "_beagle_v23_prerep_audit_contracts")
    benchmark = load_module(benchmark_path, "_beagle_v23_prerep_audit_benchmark")
    metrics, run_records, run_inputs = validate_and_recompute(
        factorial_root=factorial_experiment_root,
        extension_root=extension_experiment_root,
        maxlr_root=maxlr_experiment_root,
        contracts=contracts,
        benchmark=benchmark,
        reference_ledger_root=reference_ledger_root,
    )
    factorial_gate, modelscale_gate, maxlr_gate = compute_gate_tables(metrics)
    admitted, selector_inputs = validate_selector_decisions(
        factorial_root=factorial_selection_root,
        extension_root=extension_selection_root,
        maxlr_root=maxlr_selection_root,
        factorial_gate=factorial_gate,
        modelscale_gate=modelscale_gate,
        maxlr_gate=maxlr_gate,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "recomputed_gate_metrics_long.tsv"
    factorial_path = output_dir / "factorial_gate_results_recomputed.tsv"
    modelscale_path = output_dir / "modelscale4_gate_result_recomputed.tsv"
    maxlr_path = output_dir / "maxlr_gate_result_recomputed.tsv"
    admitted_path = output_dir / "ADMITTED_CONFIGURATIONS.json"
    provenance_path = output_dir / "PROVENANCE.json"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    factorial_gate.to_csv(factorial_path, sep="\t", index=False)
    modelscale_gate.to_csv(modelscale_path, sep="\t", index=False)
    maxlr_gate.to_csv(maxlr_path, sep="\t", index=False)
    admitted_payload = {
        "schema": "sparcal_beagle_v2_3_pre_replication_admitted_configurations_v1",
        "status": "screen_decisions_independently_recomputed",
        "screen_seed": SEED,
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "admitted_base_configurations": admitted,
    }
    admitted_path.write_text(json.dumps(admitted_payload, indent=2, sort_keys=True) + "\n")
    provenance = {
        "schema": "sparcal_beagle_v2_3_pre_replication_audit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "selector_decisions_verified",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "screen_seed": SEED,
        "samples": list(SAMPLES),
        "expected_configurations": list(SETTINGS),
        "admitted_base_configurations": admitted,
        "gate_thresholds": {
            "sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
            "support_no_harm_margin": SUPPORT_MARGIN,
            "material_improvement_margin": IMPROVEMENT_MARGIN,
            "per_sample_somatic_leakage_nonincrease": True,
        },
        "audit": identity(Path(__file__)),
        "integrity_module": identity(Path(INTEGRITY.__file__)),
        "contracts": identity(contracts_path),
        "metric_evaluator": identity(benchmark_path),
        "protocols": {
            name: identity(path) for name, path in PROTOCOLS.items()
        },
        "reference_ledger_completion": identity(
            Path(reference_ledger_root) / "RUN_COMPLETE.json"
        ),
        "truth_inputs": {
            name: identity(path) for name, path in TRUTH_INPUTS.items()
        },
        "selector_inputs": [identity(path) for path in selector_inputs],
        "validated_runs": run_records,
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    code_ledger = output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_ledger(
        [
            Path(__file__), Path(INTEGRITY.__file__), *required_code,
            Path(reference_ledger_root) / "RUN_COMPLETE.json",
            *TRUTH_INPUTS.values(), *selector_inputs, *run_inputs,
        ],
        code_ledger,
    )
    output_ledger = output_dir / "OUTPUT_SHA256SUMS.txt"
    output_members = (
        metrics_path,
        factorial_path,
        modelscale_path,
        maxlr_path,
        admitted_path,
        provenance_path,
        code_ledger,
    )
    write_ledger(output_members, output_ledger, output_dir)
    (output_dir / "RUN_COMPLETE").write_text("complete\n")
    completion = {
        "schema": "sparcal_beagle_v2_3_pre_replication_audit_completion_v1",
        "status": "complete",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "screen_seed": SEED,
        "admitted_base_configurations": admitted,
        "admitted_configurations": identity(admitted_path),
        "output_sha256sums": identity(output_ledger),
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    if verify_audit_bundle(output_dir) != admitted:
        raise ValueError("fresh pre-replication audit failed self-verification")
    return completion


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-experiment-root", type=Path, required=True)
    parser.add_argument("--factorial-selection-root", type=Path, required=True)
    parser.add_argument("--extension-experiment-root", type=Path, required=True)
    parser.add_argument("--extension-selection-root", type=Path, required=True)
    parser.add_argument("--maxlr-experiment-root", type=Path, required=True)
    parser.add_argument("--maxlr-selection-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=DEFAULT_CONTRACTS)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--reference-ledger-root", type=Path, default=DEFAULT_REFERENCE_LEDGER
    )
    args = parser.parse_args()
    result = run_audit(
        factorial_experiment_root=args.factorial_experiment_root,
        factorial_selection_root=args.factorial_selection_root,
        extension_experiment_root=args.extension_experiment_root,
        extension_selection_root=args.extension_selection_root,
        maxlr_experiment_root=args.maxlr_experiment_root,
        maxlr_selection_root=args.maxlr_selection_root,
        output_dir=args.output_dir,
        contracts_path=args.contracts.resolve(),
        benchmark_path=args.benchmark.resolve(),
        reference_ledger_root=args.reference_ledger_root.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
