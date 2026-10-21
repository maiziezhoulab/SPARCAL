#!/usr/bin/env python3
"""Gate the frozen mean/median Beagle ensemble against one selected seed."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import statistics
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
import build_selected_beagle_ensemble as SELECTED_BUILDER  # noqa: E402

SAMPLES = ("P4", "P6")
SEEDS = (20260915, 20260916, 20260917)
SINGLE_SEED = SEEDS[0]
ENSEMBLES = ("ensemble_mean", "ensemble_median")
LOO_CONFIGS = tuple(f"ensemble_mean_loo_seed{seed}" for seed in SEEDS)
ALL_ENSEMBLE_CONFIGS = (*ENSEMBLES, *LOO_CONFIGS)
METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
SENSITIVITY_MARGIN = 0.005
SUPPORT_MARGIN = 0.005
IMPROVEMENT_MARGIN = 0.002
MIN_LOO_JACCARD = 0.995
TRUTH_INPUTS = {
    "normal": PROJECT
    / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz",
    "support": PROJECT
    / "data/germline_concordance_2026-08-23/sparcal_call_details.csv.gz",
    "somatic": PROJECT
    / "data/somatic_evidence_2026-07-28/wes_confirmed_full_accounting.csv",
    "table_s1": PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv",
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
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def load_module(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def require_completed_assessment(
    root: Path, required: Sequence[str], context: str
) -> list[Path]:
    paths = [root / name for name in required]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"{context} is incomplete: {missing}")
    if (root / "RUN_COMPLETE").read_text().strip() != "complete":
        raise ValueError(f"{context} has invalid RUN_COMPLETE")
    completion = json.loads((root / "RUN_COMPLETE.json").read_text())
    if (
        completion.get("schema")
        != "sparcal_beagle_v2_3_replication_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
    ):
        raise ValueError(f"{context} has invalid status/WES contract")
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=(
            "replicated_configuration_metrics_long.tsv",
            "replicated_configuration_summary.tsv",
            "replicated_pairwise_stability.tsv",
            "replication_gate_results.tsv",
            "FINAL_BASE_CONFIGURATION.json",
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
        ),
        label=context,
    )
    INTEGRITY.parse_and_verify_ledger(
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label=f"{context} code/input ledger",
    )
    return paths


def load_base_assessment(
    root: Path,
) -> tuple[str, pd.DataFrame, Mapping[str, object], list[Path]]:
    paths = require_completed_assessment(
        root,
        (
            "replicated_configuration_metrics_long.tsv",
            "replicated_configuration_summary.tsv",
            "replicated_pairwise_stability.tsv",
            "replication_gate_results.tsv",
            "FINAL_BASE_CONFIGURATION.json",
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
            "OUTPUT_SHA256SUMS.txt",
            "RUN_COMPLETE",
            "RUN_COMPLETE.json",
        ),
        "base-configuration assessment",
    )
    selection = json.loads((root / "FINAL_BASE_CONFIGURATION.json").read_text())
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    base = str(selection.get("selected_base_configuration", ""))
    if (
        selection.get("schema")
        != "sparcal_beagle_v2_3_final_base_selection_v1"
        or provenance.get("schema")
        != "sparcal_beagle_v2_3_replication_assessment_v1"
        or not base
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or selection.get("final_ensemble_selected") is not False
        or provenance.get("selected_base_configuration") != base
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
    ):
        raise ValueError("base selection is invalid or inconsistent")
    verified_base, _, _ = SELECTED_BUILDER.load_selected_sources(root)
    if verified_base != base:
        raise ValueError("independent base verifier disagrees with selection")
    metrics = pd.read_csv(
        root / "replicated_configuration_metrics_long.tsv", sep="\t"
    )
    return base, metrics, provenance, paths


def load_ensemble_assessment(
    root: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Mapping[str, object],
    list[Path],
]:
    paths = [
        root / "configuration_metrics_long.tsv",
        root / "configuration_summary.tsv",
        root / "pairwise_stability.tsv",
        root / "PROVENANCE.json",
        root / "RUN_COMPLETE",
    ]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"ensemble metric assessment is incomplete: {missing}")
    if paths[-1].read_text().strip() != "complete":
        raise ValueError("ensemble metric assessment has invalid RUN_COMPLETE")
    provenance = json.loads(paths[3].read_text())
    if (
        provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("anchor_score") != "p_nonref"
        or not math.isclose(
            float(provenance.get("min_anchor_probability")),
            0.90,
            rel_tol=0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("ensemble metric assessment has invalid method/WES contract")
    audit_keys = {
        (str(row.get("config")), str(row.get("sample")))
        for row in provenance.get("audits", [])
    }
    expected = {
        (config, sample)
        for config in ALL_ENSEMBLE_CONFIGS
        for sample in SAMPLES
    }
    if audit_keys != expected or len(provenance.get("audits", [])) != len(expected):
        raise ValueError("ensemble assessment does not bind the exact 5x2 audit grid")
    for row in provenance["audits"]:
        if set(row) != {"config", "sample", "audit", "audit_sha256"}:
            raise ValueError("ensemble assessment audit identity fields are not exact")
        audit = Path(str(row["audit"])).resolve()
        if not audit.is_file() or sha256(audit) != row["audit_sha256"]:
            raise ValueError(f"ensemble assessment audit changed: {audit}")
    return (
        pd.read_csv(paths[0], sep="\t"),
        pd.read_csv(paths[1], sep="\t"),
        pd.read_csv(paths[2], sep="\t"),
        provenance,
        paths,
    )


def validate_ensemble_bundles(
    *,
    root: Path,
    base: str,
    base_provenance: Mapping[str, object],
    base_paths: Sequence[Path],
    ensemble_provenance: Mapping[str, object],
    contracts,
    builder_path: Path,
    ensemble_module_path: Path,
    protocol_path: Path,
    ensemble_module,
) -> list[dict[str, object]]:
    root = Path(root).resolve()
    top_verification = contracts.verify_complete_bundle(root)
    top_manifest_path = root / "MANIFEST.json"
    if Path(str(top_verification["manifest"])).resolve() != top_manifest_path.resolve():
        raise ValueError("top completion does not bind the canonical manifest")
    top_manifest = json.loads(top_manifest_path.read_text())
    if (
        top_manifest.get("schema")
        != "sparcal_beagle_selected_posterior_ensemble_v1"
        or top_manifest.get("status") != "outputs_ready_for_checksum"
        or top_manifest.get("selected_base_configuration") != base
        or top_manifest.get("production_inputs_include_wes") is not False
        or top_manifest.get("evaluation_uses_wes") is not False
        or top_manifest.get("samples") != list(SAMPLES)
        or top_manifest.get("frozen_seeds") != list(SEEDS)
        or top_manifest.get("methods") != list(ALL_ENSEMBLE_CONFIGS)
    ):
        raise ValueError(f"invalid top-level ensemble manifest: {top_manifest_path}")
    INTEGRITY.verify_identity(
        top_manifest.get("builder"),
        label="top ensemble builder",
        expected_path=builder_path,
    )
    INTEGRITY.verify_identity(
        top_manifest.get("integrity_module"),
        label="top ensemble integrity module",
        expected_path=Path(INTEGRITY.__file__),
    )
    INTEGRITY.verify_identity(
        top_manifest.get("ensemble_module"),
        label="top ensemble module",
        expected_path=ensemble_module_path,
    )
    INTEGRITY.verify_identity(
        top_manifest.get("contracts"),
        label="top ensemble contracts",
        expected_path=Path(contracts.__file__),
    )
    INTEGRITY.verify_identity(
        top_manifest.get("protocol"),
        label="top ensemble protocol",
        expected_path=protocol_path,
    )
    recorded_base_inputs = top_manifest.get("base_assessment_inputs")
    if not isinstance(recorded_base_inputs, list):
        raise ValueError("top ensemble manifest lacks base assessment identities")
    recorded_base_paths = {
        INTEGRITY.verify_identity(record, label=f"base assessment input {index}")
        for index, record in enumerate(recorded_base_inputs)
    }
    if recorded_base_paths != {Path(path).resolve() for path in base_paths}:
        raise ValueError("top ensemble manifest binds the wrong base assessment set")
    expected_source = {}
    for row in base_provenance.get("validated_runs", []):
        config = str(row.get("config"))
        sample = str(row.get("sample"))
        for seed in SEEDS:
            if config == f"{base}_seed{seed}" and sample in SAMPLES:
                expected_source[(seed, sample)] = row
    if len(expected_source) != len(SEEDS) * len(SAMPLES):
        raise ValueError("base assessment lacks all selected three-seed source runs")
    for (seed, sample), row in expected_source.items():
        validation = contracts.validate_v2_3_run(Path(str(row["run_root"])))
        if validation.get("status") != "verified":
            raise ValueError(f"selected source run failed validation: {sample}/{seed}")
    audit_lookup = {
        (str(row["config"]), str(row["sample"])): row
        for row in ensemble_provenance["audits"]
    }

    expected_children = {
        (root / method / f"{sample}_sec1" / "RUN_COMPLETE.json").resolve()
        for method in ALL_ENSEMBLE_CONFIGS
        for sample in SAMPLES
    }
    child_records = top_manifest.get("child_completions")
    if not isinstance(child_records, list):
        raise ValueError("top ensemble manifest lacks child completion identities")
    recorded_children = {
        INTEGRITY.verify_identity(record, label=f"child completion {index}")
        for index, record in enumerate(child_records)
    }
    if recorded_children != expected_children or len(child_records) != len(
        expected_children
    ):
        raise ValueError("top ensemble child completion set is not exact")

    records = [{
        "method": "top_level_bundle",
        "sample": ".",
        "sample_root": str(root.resolve()),
        "manifest": identity(top_manifest_path),
        "completion": identity(root / "RUN_COMPLETE.json"),
        "verification": top_verification,
    }]
    ensemble_audits = {sample: {} for sample in SAMPLES}
    ensemble_manifests = {}
    for method in ALL_ENSEMBLE_CONFIGS:
        for sample in SAMPLES:
            sample_root = root / method / f"{sample}_sec1"
            verification = contracts.verify_complete_bundle(sample_root)
            manifest_path = (
                sample_root
                / "beagle_evidence_v2_3_ensemble"
                / "MANIFEST.json"
            )
            manifest = json.loads(manifest_path.read_text())
            if Path(str(verification["manifest"])).resolve() != manifest_path.resolve():
                raise ValueError(
                    f"child completion does not bind canonical manifest: {sample_root}"
                )
            expected_included = (
                list(SEEDS)
                if method in ENSEMBLES
                else [seed for seed in SEEDS if str(seed) not in method]
            )
            if (
                manifest.get("schema")
                != "sparcal_beagle_posterior_ensemble_v2_3_candidate_1"
                or manifest.get("sample") != sample
                or manifest.get("method") != method
                or manifest.get("aggregation")
                != ("median" if method == "ensemble_median" else "mean")
                or manifest.get("included_seeds") != expected_included
                or manifest.get("frozen_source_seeds") != list(SEEDS)
                or manifest.get("production_inputs_include_wes") is not False
                or manifest.get("evaluation_uses_wes") is not False
                or manifest.get("anchor_score") != "p_nonref"
                or manifest.get("min_anchor_probability") != 0.90
                or manifest.get("p_nonref_aggregation")
                != "direct_across_seed_GP01_plus_GP11"
                or manifest.get("gp_aggregation")
                != "componentwise_for_descriptive_genotype_fields"
                or manifest.get("invalid_gp_policy") != "beagle_uncertain"
                or manifest.get("panel_absent_policy")
                != "must_agree_and_remain_panel_absent"
            ):
                raise ValueError(f"invalid ensemble manifest: {manifest_path}")
            INTEGRITY.verify_identity(
                manifest.get("source_contracts_module"),
                label=f"{method}/{sample} contracts",
                expected_path=Path(contracts.__file__),
            )
            INTEGRITY.verify_identity(
                manifest.get("source_ensemble_module"),
                label=f"{method}/{sample} ensemble module",
                expected_path=ensemble_module_path,
            )
            snapshot = manifest.get("code_snapshot")
            if not isinstance(snapshot, list) or len(snapshot) != 3:
                raise ValueError(f"invalid code snapshot: {method}/{sample}")
            snapshot_paths = {
                INTEGRITY.verify_identity(
                    record, label=f"{method}/{sample} snapshot {index}"
                )
                for index, record in enumerate(snapshot)
            }
            snapshot_by_name = {path.name: sha256(path) for path in snapshot_paths}
            expected_snapshot = {
                ensemble_module_path.name: sha256(ensemble_module_path),
                Path(contracts.__file__).name: sha256(Path(contracts.__file__)),
                protocol_path.name: sha256(protocol_path),
            }
            if snapshot_by_name != expected_snapshot:
                raise ValueError(f"code snapshot identity mismatch: {method}/{sample}")
            if set(manifest.get("source_audits", {})) != {
                str(seed) for seed in SEEDS
            }:
                raise ValueError(f"source audit seed set is not exact: {method}/{sample}")
            for seed in SEEDS:
                expected_audit = expected_source[(seed, sample)]["audit"]
                recorded = manifest["source_audits"][str(seed)]
                if (
                    str(Path(recorded["path"]).resolve())
                    != str(Path(expected_audit["path"]).resolve())
                    or recorded["sha256"] != expected_audit["sha256"]
                    or sha256(Path(recorded["path"])) != recorded["sha256"]
                ):
                    raise ValueError(
                        f"ensemble source audit differs from selected base: "
                        f"{method}/{sample}/seed{seed}"
                    )
            audit = (
                sample_root
                / "beagle_evidence_v2_3_ensemble"
                / "beagle_evidence_audit.tsv.gz"
            )
            assessed = audit_lookup[(method, sample)]
            if (
                Path(str(assessed["audit"])).resolve() != audit.resolve()
                or str(assessed["audit_sha256"]) != sha256(audit)
            ):
                raise ValueError(f"ensemble assessment audit mismatch: {audit}")
            ensemble_audits[sample][method] = audit
            ensemble_manifests[(method, sample)] = manifest
            records.append({
                "method": method,
                "sample": sample,
                "sample_root": str(sample_root.resolve()),
                "manifest": identity(manifest_path),
                "audit": identity(audit),
                "completion": identity(sample_root / "RUN_COMPLETE.json"),
                "verification": verification,
            })
    for sample in SAMPLES:
        source_audits = {
            seed: Path(str(expected_source[(seed, sample)]["audit"]["path"]))
            for seed in SEEDS
        }
        recomputed = ensemble_module.validate_ensemble_audits(
            audits=ensemble_audits[sample],
            source_audits=source_audits,
        )
        for method in ALL_ENSEMBLE_CONFIGS:
            manifest = ensemble_manifests[(method, sample)]
            if (
                recomputed[method]["raw_record_count"]
                != manifest.get("raw_record_count")
                or recomputed[method]["counts"] != manifest.get("counts")
            ):
                raise ValueError(
                    f"recomputed ensemble counts disagree: {method}/{sample}"
                )
    return records


def recompute_ensemble_assessment(
    provenance: Mapping[str, object], benchmark
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    for path in TRUTH_INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    normal = pd.read_csv(TRUTH_INPUTS["normal"])
    normal = normal.loc[
        (normal["detected"] == 1) & (normal["rna_dp"] >= 1)
    ].copy()
    support = pd.read_csv(TRUTH_INPUTS["support"])
    somatic = pd.read_csv(TRUTH_INPUTS["somatic"])
    table_s1 = pd.read_csv(TRUTH_INPUTS["table_s1"], sep="\t")

    audits = {
        (str(row["config"]), str(row["sample"])): Path(
            str(row["audit"])
        ).resolve()
        for row in provenance["audits"]
    }
    metrics = []
    counts = []
    evidence_by_sample = {sample: {} for sample in SAMPLES}
    for config in ALL_ENSEMBLE_CONFIGS:
        for sample in SAMPLES:
            evidence = benchmark.reroute_evidence(
                benchmark.read_evidence(audits[(config, sample)]),
                "p_nonref",
                0.90,
            )
            evidence_by_sample[sample][config] = evidence
            sample_metrics, sample_counts = benchmark.evaluate(
                config,
                sample,
                evidence,
                normal,
                support,
                somatic,
                table_s1,
            )
            metrics.extend(sample_metrics)
            counts.append(sample_counts)
    metrics_frame = pd.DataFrame(metrics)
    counts_frame = pd.DataFrame(counts)
    pairwise = pd.DataFrame([
        row
        for sample in SAMPLES
        for row in benchmark.pairwise_stability(
            evidence_by_sample[sample], sample
        )
    ])
    wide = metrics_frame.pivot(
        index=["config", "sample"], columns="metric", values="value"
    ).reset_index()
    summary = counts_frame.merge(wide, on=["config", "sample"], how="left")
    return metrics_frame, summary, pairwise, list(TRUTH_INPUTS.values())


def require_recomputed_table(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: Sequence[str],
    label: str,
) -> None:
    if set(observed.columns) != set(expected.columns):
        raise ValueError(f"{label} columns differ from recomputation")
    columns = list(expected.columns)
    observed_sorted = observed[columns].sort_values(list(keys)).reset_index(drop=True)
    expected_sorted = expected[columns].sort_values(list(keys)).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(
            observed_sorted,
            expected_sorted,
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
    except AssertionError as error:
        raise ValueError(f"{label} differs from audit recomputation") from error


def selected_metric_table(
    metrics: pd.DataFrame, expected: set[tuple[str, str, str]], context: str
) -> pd.DataFrame:
    required = {"config", "sample", "metric", "numerator", "denominator", "value"}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"{context} lacks columns: {sorted(missing)}")
    chosen = metrics.loc[
        metrics[["config", "sample", "metric"]]
        .apply(tuple, axis=1)
        .isin(expected)
    ].copy()
    observed = set(
        chosen[["config", "sample", "metric"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(chosen) != len(expected):
        raise ValueError(f"{context} does not contain the exact metric grid")
    for column in ("numerator", "denominator", "value"):
        chosen[column] = pd.to_numeric(chosen[column], errors="raise")
    if (
        (chosen["denominator"] <= 0).any()
        or (chosen["numerator"] < 0).any()
        or (chosen["numerator"] > chosen["denominator"]).any()
    ):
        raise ValueError(f"{context} has invalid counts")
    if not all(
        math.isclose(a / b, value, rel_tol=0, abs_tol=1e-12)
        for a, b, value in zip(
            chosen["numerator"], chosen["denominator"], chosen["value"]
        )
    ):
        raise ValueError(f"{context} values do not equal counts")
    return chosen


def assess_ensembles(
    *,
    base: str,
    base_metrics: pd.DataFrame,
    ensemble_metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    single = f"{base}_seed{SINGLE_SEED}"
    expected_single = {
        (single, sample, metric)
        for sample in SAMPLES
        for metric in METRICS
    }
    single_rows = selected_metric_table(
        base_metrics, expected_single, "single-seed comparator metrics"
    )
    expected_ensemble = {
        (config, sample, metric)
        for config in ENSEMBLES
        for sample in SAMPLES
        for metric in METRICS
    }
    ensemble_rows = selected_metric_table(
        ensemble_metrics, expected_ensemble, "ensemble metrics"
    )
    frame = pd.concat([single_rows, ensemble_rows], ignore_index=True).set_index(
        ["config", "sample", "metric"]
    )

    required_pairwise = {"sample", "config_a", "config_b", "anchor_jaccard"}
    if missing := required_pairwise.difference(pairwise.columns):
        raise ValueError(f"ensemble pairwise table lacks columns: {sorted(missing)}")
    loo = pairwise.loc[
        pairwise["config_a"].isin(LOO_CONFIGS)
        & pairwise["config_b"].isin(LOO_CONFIGS)
    ].copy()
    expected_loo_pairs = {
        (sample, *sorted((left, right)))
        for sample in SAMPLES
        for index, left in enumerate(LOO_CONFIGS)
        for right in LOO_CONFIGS[index + 1 :]
    }
    observed_loo_pairs = {
        (str(row.sample), *sorted((str(row.config_a), str(row.config_b))))
        for row in loo.itertuples(index=False)
    }
    if observed_loo_pairs != expected_loo_pairs or len(loo) != len(expected_loo_pairs):
        raise ValueError("leave-one-out pairwise grid is not exact")
    loo["anchor_jaccard"] = pd.to_numeric(
        loo["anchor_jaccard"], errors="raise"
    )
    if not all(
        math.isfinite(float(value)) and 0 <= float(value) <= 1
        for value in loo["anchor_jaccard"]
    ):
        raise ValueError("leave-one-out Jaccard contains invalid values")
    loo_min = float(loo["anchor_jaccard"].min())

    def value(config: str, sample: str, metric: str) -> float:
        return float(frame.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(frame.loc[(config, sample, metric), "numerator"])

    baseline_sensitivity = statistics.mean(
        value(single, sample, "normal_wes_anchor_sensitivity")
        for sample in SAMPLES
    )
    baseline_support = statistics.mean(
        value(single, sample, "selected_anchor_normal_wes_support")
        for sample in SAMPLES
    )
    baseline_leakage = {
        sample: numerator(single, sample, "wes_somatic_anchor_leakage")
        for sample in SAMPLES
    }
    rows = []
    for method in ENSEMBLES:
        sensitivity = statistics.mean(
            value(method, sample, "normal_wes_anchor_sensitivity")
            for sample in SAMPLES
        )
        support = statistics.mean(
            value(method, sample, "selected_anchor_normal_wes_support")
            for sample in SAMPLES
        )
        leakage = {
            sample: numerator(method, sample, "wes_somatic_anchor_leakage")
            for sample in SAMPLES
        }
        sensitivity_delta = sensitivity - baseline_sensitivity
        support_delta = support - baseline_support
        leakage_reduction = sum(baseline_leakage.values()) - sum(leakage.values())
        leakage_nonincrease = all(
            leakage[sample] <= baseline_leakage[sample] for sample in SAMPLES
        )
        material = (
            sensitivity_delta >= IMPROVEMENT_MARGIN
            or support_delta >= IMPROVEMENT_MARGIN
            or leakage_reduction > 0
        )
        advances = (
            leakage_nonincrease
            and sensitivity_delta >= -SENSITIVITY_MARGIN
            and support_delta >= -SUPPORT_MARGIN
            and material
            and loo_min >= MIN_LOO_JACCARD
        )
        rows.append({
            "method": method,
            "single_seed_comparator": single,
            "mean_normal_wes_anchor_sensitivity": sensitivity,
            "delta_mean_sensitivity": sensitivity_delta,
            "sensitivity_no_harm": sensitivity_delta >= -SENSITIVITY_MARGIN,
            "mean_selected_anchor_normal_wes_support": support,
            "delta_mean_support": support_delta,
            "support_no_harm": support_delta >= -SUPPORT_MARGIN,
            "P4_somatic_leakage_n": leakage["P4"],
            "P6_somatic_leakage_n": leakage["P6"],
            "total_somatic_leakage_n": sum(leakage.values()),
            "leakage_nonincrease_both": leakage_nonincrease,
            "total_leakage_reduction": leakage_reduction,
            "material_improvement": material,
            "minimum_leave_one_out_anchor_jaccard": loo_min,
            "leave_one_out_stability_pass": loo_min >= MIN_LOO_JACCARD,
            "beagle_run_multiplier": 3,
            "advance_over_single_seed": advances,
        })
    result = pd.DataFrame(rows)
    advancing = result.loc[result["advance_over_single_seed"]].copy()
    if advancing.empty:
        winner = "single_seed"
    else:
        advancing["mean_preference"] = advancing["method"].ne("ensemble_mean")
        advancing = advancing.sort_values(
            [
                "total_somatic_leakage_n",
                "mean_normal_wes_anchor_sensitivity",
                "mean_selected_anchor_normal_wes_support",
                "minimum_leave_one_out_anchor_jaccard",
                "mean_preference",
            ],
            ascending=[True, False, False, False, True],
            kind="mergesort",
        )
        winner = str(advancing.iloc[0]["method"])
    result["selected_final_representation"] = result["method"].eq(winner)
    return result, winner


def write_sums(paths: Sequence[Path], root: Path, output: Path) -> None:
    rows = []
    for path in sorted(set(Path(path).resolve() for path in paths), key=str):
        try:
            label = path.relative_to(root.resolve())
        except ValueError:
            label = path
        rows.append(f"{sha256(path)}  {label}")
    output.write_text("\n".join(rows) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assessment-root", type=Path, required=True)
    parser.add_argument("--ensemble-root", type=Path, required=True)
    parser.add_argument("--ensemble-assessment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contracts",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
        "BENCHMARK_PROTOCOL_ENSEMBLE.md",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py",
    )
    parser.add_argument(
        "--ensemble-module",
        type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_posterior_ensemble.py",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    for path in (
        args.protocol,
        args.contracts,
        args.benchmark,
        args.ensemble_module,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    base, base_metrics, base_provenance, base_paths = load_base_assessment(
        args.base_assessment_root
    )
    (
        observed_ensemble_metrics,
        observed_ensemble_summary,
        observed_pairwise,
        ensemble_provenance,
        assessment_paths,
    ) = (
        load_ensemble_assessment(args.ensemble_assessment_root)
    )
    contracts = load_module(args.contracts, "_beagle_ensemble_gate_contracts")
    ensemble_module = load_module(
        args.ensemble_module, "_beagle_ensemble_gate_recompute"
    )
    benchmark = load_module(
        args.benchmark, "_beagle_ensemble_gate_benchmark"
    )
    bundle_records = validate_ensemble_bundles(
        root=args.ensemble_root,
        base=base,
        base_provenance=base_provenance,
        base_paths=base_paths,
        ensemble_provenance=ensemble_provenance,
        contracts=contracts,
        builder_path=Path(SELECTED_BUILDER.__file__),
        ensemble_module_path=args.ensemble_module.resolve(),
        protocol_path=args.protocol.resolve(),
        ensemble_module=ensemble_module,
    )
    ensemble_metrics, ensemble_summary, pairwise, truth_paths = (
        recompute_ensemble_assessment(ensemble_provenance, benchmark)
    )
    require_recomputed_table(
        observed_ensemble_metrics,
        ensemble_metrics,
        keys=("config", "sample", "metric"),
        label="ensemble metric table",
    )
    require_recomputed_table(
        observed_ensemble_summary,
        ensemble_summary,
        keys=("config", "sample"),
        label="ensemble summary table",
    )
    require_recomputed_table(
        observed_pairwise,
        pairwise,
        keys=("sample", "config_a", "config_b"),
        label="ensemble pairwise table",
    )
    gates, winner = assess_ensembles(
        base=base,
        base_metrics=base_metrics,
        ensemble_metrics=ensemble_metrics,
        pairwise=pairwise,
    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    gate_path = args.output_dir / "ensemble_gate_results.tsv"
    gates.to_csv(gate_path, sep="\t", index=False)
    selection_path = args.output_dir / "FINAL_BEAGLE_REPRESENTATION.json"
    selection_path.write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_final_representation_v1",
                "base_configuration": base,
                "selected_representation": winner,
                "selected_seed": SINGLE_SEED if winner == "single_seed" else None,
                "beagle_run_multiplier": 1 if winner == "single_seed" else 3,
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "evidence_router": "four mutually exclusive routes: P(nonref)>=0.90 hard anchor, ref-high-confidence, uncertain, and panel-absent",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    provenance_path = args.output_dir / "PROVENANCE.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_ensemble_selection_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "final_representation_selected",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "development_set_evaluation": True,
                "base_configuration": base,
                "selected_representation": winner,
                "frozen_seeds": list(SEEDS),
                "gate_thresholds": {
                    "sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
                    "support_no_harm_margin": SUPPORT_MARGIN,
                    "material_improvement_margin": IMPROVEMENT_MARGIN,
                    "minimum_leave_one_out_anchor_jaccard": MIN_LOO_JACCARD,
                    "no_per_sample_somatic_leakage_increase": True,
                },
                "cost_rule": "three Beagle runs versus one; reject without material gain",
                "assessor": identity(Path(__file__)),
                "contracts": identity(args.contracts),
                "protocol": identity(args.protocol),
                "benchmark": identity(args.benchmark),
                "ensemble_module": identity(args.ensemble_module),
                "selected_ensemble_builder": identity(
                    Path(SELECTED_BUILDER.__file__)
                ),
                "integrity_module": identity(Path(INTEGRITY.__file__)),
                "truth_inputs": {
                    name: identity(path)
                    for name, path in zip(TRUTH_INPUTS, truth_paths)
                },
                "base_assessment_inputs": [identity(path) for path in base_paths],
                "ensemble_assessment_inputs": [
                    identity(path) for path in assessment_paths
                ],
                "validated_ensemble_bundles": bundle_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    code_sums = args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_sums(
        [
            Path(__file__),
            args.contracts,
            args.protocol,
            args.benchmark,
            args.ensemble_module,
            Path(SELECTED_BUILDER.__file__),
            Path(INTEGRITY.__file__),
            *base_paths,
            *assessment_paths,
            *truth_paths,
        ],
        PROJECT,
        code_sums,
    )
    output_sums = args.output_dir / "OUTPUT_SHA256SUMS.txt"
    write_sums(
        [gate_path, selection_path, provenance_path, code_sums],
        args.output_dir,
        output_sums,
    )
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    (args.output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_ensemble_selection_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "base_configuration": base,
                "selected_representation": winner,
                "output_sha256sums": identity(output_sums),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(gates.to_string(index=False))
    print(f"Selected final representation: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
