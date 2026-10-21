#!/usr/bin/env python3
"""Apply the frozen three-seed Beagle v2.3 stability and selection gates.

This assessor is intentionally frozen before seed-replication results exist.
Matched WES is loaded only for development-set evaluation and is never passed
to Beagle or to the production routing rule.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
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
import audit_beagle_v2_3_pre_replication as PREAUDIT  # noqa: E402

SAMPLES = ("P4", "P6")
SEEDS = (20260915, 20260916, 20260917)
COMPARATOR = "ms2_niter0"
SETTINGS = {
    "ms2_niter0": {
        "modelscale": 2.0, "niterations": 0, "maxlr": 5000.0, "simplicity": 0,
    },
    "ms2_niter5": {
        "modelscale": 2.0, "niterations": 5, "maxlr": 5000.0, "simplicity": 2,
    },
    "ms0p8_niter0": {
        "modelscale": 0.8, "niterations": 0, "maxlr": 5000.0, "simplicity": 1,
    },
    "ms0p8_niter5": {
        "modelscale": 0.8, "niterations": 5, "maxlr": 5000.0, "simplicity": 3,
    },
    "ms4_niter0": {
        "modelscale": 4.0, "niterations": 0, "maxlr": 5000.0, "simplicity": 1,
    },
    "ms2_niter0_maxlr50000": {
        "modelscale": 2.0,
        "niterations": 0,
        "maxlr": 50000.0,
        "simplicity": 1,
    },
}
METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
MIN_ANCHOR_JACCARD = 0.99
MIN_ROUTE_AGREEMENT = 0.998
MAX_MEDIAN_JACCARD_DEFICIT = 0.002
MAX_MEDIAN_PNONREF_MAE_EXCESS = 0.002
SENSITIVITY_MARGIN = 0.005
SUPPORT_MARGIN = 0.005
IMPROVEMENT_MARGIN = 0.002
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


def verify_pre_replication_audit_binding(
    provenance: Mapping[str, object],
    selected: Sequence[str],
) -> list[Path]:
    value = provenance.get("pre_replication_audit_root")
    if not isinstance(value, str) or not value:
        raise ValueError("replication plan lacks a pre-replication audit root")
    audit_root = Path(value).resolve()
    admitted = PREAUDIT.verify_audit_bundle(audit_root)
    if admitted != list(selected):
        raise ValueError(
            "replication plan admissions differ from the pre-replication audit"
        )
    INTEGRITY.verify_identity(
        provenance.get("pre_replication_audit_module"),
        label="pre-replication audit module",
        expected_path=Path(PREAUDIT.__file__),
    )
    paths = sorted(audit_root.iterdir(), key=lambda path: path.name)
    expected = [identity(path) for path in paths]
    if provenance.get("pre_replication_audit_inputs") != expected:
        raise ValueError(
            "replication plan pre-replication audit identities are not exact"
        )
    selection_inputs = provenance.get("selection_inputs")
    if not isinstance(selection_inputs, list) or any(
        record not in selection_inputs for record in expected
    ):
        raise ValueError(
            "replication plan selection inputs do not bind the complete audit"
        )
    return paths


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


def verify_plan(root: Path) -> tuple[list[str], list[dict[str, object]], list[Path]]:
    root = Path(root).resolve()
    paths = [
        root / "replication_tasks.tsv",
        root / "PROVENANCE.json",
        root / "OUTPUT_SHA256SUMS.txt",
        root / "RUN_COMPLETE",
        root / "RUN_COMPLETE.json",
    ]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"replication plan is incomplete: {missing}")
    if paths[3].read_text().strip() != "complete":
        raise ValueError("replication plan has invalid RUN_COMPLETE")
    provenance = json.loads(paths[1].read_text())
    completion = json.loads(paths[4].read_text())
    if (
        provenance.get("schema") != "sparcal_beagle_v2_3_replication_plan_v1"
        or provenance.get("status") != "tasks_frozen"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("development_set_evaluation") is not True
        or provenance.get("screen_seed") != SEEDS[0]
        or provenance.get("replication_seeds") != list(SEEDS[1:])
        or completion.get("schema")
        != "sparcal_beagle_v2_3_replication_plan_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
    ):
        raise ValueError("replication plan has an invalid status/WES contract")
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=("replication_tasks.tsv", "PROVENANCE.json"),
        label="replication plan",
    )
    INTEGRITY.verify_identity(
        provenance.get("planner"),
        label="replication planner",
        expected_path=PROJECT
        / "scripts/postanalyze/plan_beagle_v2_3_replication.py",
    )
    INTEGRITY.verify_identity(
        provenance.get("integrity_module"),
        label="replication plan integrity module",
        expected_path=Path(INTEGRITY.__file__),
    )
    selection_inputs = provenance.get("selection_inputs")
    if not isinstance(selection_inputs, list) or not selection_inputs:
        raise ValueError("replication plan lacks selection input identities")
    for index, record in enumerate(selection_inputs):
        INTEGRITY.verify_identity(
            record, label=f"replication plan selection input {index}"
        )

    frame = pd.read_csv(paths[0], sep="\t")
    required = {
        "task_id",
        "config",
        "modelscale",
        "niterations",
        "maxlr",
        "seed",
        "sample",
        "sample_root",
        "dataset",
    }
    if set(frame.columns) != required:
        raise ValueError("replication task table has an unexpected schema")
    frame["task_id"] = pd.to_numeric(frame["task_id"], errors="raise").astype(int)
    if frame["task_id"].tolist() != list(range(len(frame))):
        raise ValueError("replication task IDs are not contiguous")
    selected = [str(value) for value in provenance["selected_configurations"]]
    if selected != [str(value) for value in completion["selected_configurations"]]:
        raise ValueError("replication plan selected-config lists disagree")
    if COMPARATOR not in selected or len(selected) != len(set(selected)):
        raise ValueError("replication plan lacks one unique comparator")
    if any(config not in SETTINGS for config in selected):
        raise ValueError("replication plan contains an unknown configuration")
    audit_paths = verify_pre_replication_audit_binding(provenance, selected)
    expected = {
        (config, seed, sample)
        for config in selected
        for seed in SEEDS[1:]
        for sample in SAMPLES
    }
    observed = set(
        frame[["config", "seed", "sample"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(frame) != len(expected):
        raise ValueError("replication task grid is not exact")
    if (
        provenance.get("task_count") != len(frame)
        or completion.get("task_count") != len(frame)
        or completion.get("array_max") != len(frame) - 1
    ):
        raise ValueError("replication plan task count/array range is inconsistent")
    for row in frame.itertuples(index=False):
        setting = SETTINGS[str(row.config)]
        if (
            float(row.modelscale) != setting["modelscale"]
            or int(row.niterations) != setting["niterations"]
            or float(row.maxlr) != setting["maxlr"]
            or str(row.sample_root) != f"{row.sample}_sec1"
            or str(row.dataset) != f"{row.sample}_TUMOR"
        ):
            raise ValueError(f"invalid replication task parameters: {row}")
    return selected, frame.to_dict("records"), [*paths, *audit_paths]


def run_config(base: str, seed: int) -> str:
    return f"{base}_seed{seed}"


def base_config(config: str) -> str:
    match = re.fullmatch(r"(.+)_seed(\d+)", str(config))
    if not match or int(match.group(2)) not in SEEDS:
        raise ValueError(f"invalid replicated configuration label: {config}")
    return match.group(1)


def source_run_root(
    *,
    base: str,
    seed: int,
    sample: str,
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
    replication_root: Path,
) -> Path:
    config = run_config(base, seed)
    if seed != SEEDS[0]:
        parent = replication_root
    elif base == "ms4_niter0":
        parent = extension_root
    elif base == "ms2_niter0_maxlr50000":
        parent = maxlr_root
    else:
        parent = factorial_root
    return parent / config / f"{sample}_sec1"


def manifest_signature(manifest: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "code": sorted(
            (Path(str(row["path"])).name, str(row["sha256"]))
            for row in manifest["code"]
        ),
        "executables": sorted(
            (name, str(row["sha256"]))
            for name, row in manifest["executables"].items()
        ),
        "reference": sorted(
            (
                str(row["contig"]),
                str(row["panel"]["sha256"]),
                str(row["panel_index"]["sha256"]),
            )
            for row in manifest["reference_panels"]
        ),
        "ledger": sorted(
            (name, str(value["sha256"]))
            for name, value in manifest["reference_ledger"].items()
            if isinstance(value, dict) and "sha256" in value
        ),
    }


def read_runtime(root: Path, manifest: Mapping[str, object]) -> tuple[float, Path]:
    relative = manifest.get("metrics_file")
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError(f"invalid metrics path in {root}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"metrics path escapes run root: {relative}") from error
    frame = pd.read_csv(path)
    required = {"chromosome", "duration", "status"}
    if not required.issubset(frame.columns) or len(frame) != 22:
        raise ValueError(f"runtime metrics are not a 22-contig table: {path}")
    duration = pd.to_numeric(frame["duration"], errors="raise")
    if (
        not all(math.isfinite(float(value)) and float(value) >= 0 for value in duration)
        or set(frame["status"]).difference({"completed", "panel_absent", "raw_empty"})
    ):
        raise ValueError(f"invalid runtime values/statuses: {path}")
    return float(duration.sum()), path


def validate_replication_snapshot(
    *,
    root: Path,
    plan_root: Path,
    protocol: Path,
    extension_protocol: Path,
    maxlr_protocol: Path,
    replication_protocol: Path,
) -> Mapping[str, object]:
    root = Path(root).resolve()
    plan_root = Path(plan_root).resolve()
    ledger = root / "CODE_SNAPSHOT_SHA256SUMS.txt"
    records = INTEGRITY.parse_and_verify_ledger(
        ledger,
        roots=(root,),
        label="replication code snapshot ledger",
        require_contained_in=root / "code_snapshot",
    )
    by_relative = {
        str(path.relative_to(root)): path for path in records.values()
    }
    sources = {
        "code_snapshot/scripts/2_beagle_filtering/run_beagle.py": PROJECT
        / "scripts/2_beagle_filtering/run_beagle.py",
        "code_snapshot/scripts/2_beagle_filtering/beagle_evidence_router.py": PROJECT
        / "scripts/2_beagle_filtering/beagle_evidence_router.py",
        "code_snapshot/scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py": PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py",
        "code_snapshot/scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py": PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py",
        "code_snapshot/run_beagle_v2_3_replication.sh": PROJECT
        / "run_slurm/run_beagle_v2_3_replication.sh",
        "code_snapshot/beagle_selection_integrity.py": Path(INTEGRITY.__file__),
        "code_snapshot/BENCHMARK_PROTOCOL.md": protocol,
        "code_snapshot/BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md": extension_protocol,
        "code_snapshot/BENCHMARK_PROTOCOL_MAXLR.md": maxlr_protocol,
        "code_snapshot/BENCHMARK_PROTOCOL_REPLICATION.md": replication_protocol,
        "code_snapshot/replication_tasks.tsv": plan_root
        / "replication_tasks.tsv",
        "code_snapshot/PROVENANCE.json": plan_root / "PROVENANCE.json",
        "code_snapshot/OUTPUT_SHA256SUMS.txt": plan_root
        / "OUTPUT_SHA256SUMS.txt",
        "code_snapshot/RUN_COMPLETE.json": plan_root / "RUN_COMPLETE.json",
    }
    if set(by_relative) != set(sources):
        raise ValueError("replication code snapshot artifact set is not exact")
    for relative, source in sources.items():
        if sha256(by_relative[relative]) != sha256(Path(source)):
            raise ValueError(f"replication snapshot differs from source: {relative}")
    return {
        "schema": "sparcal_beagle_v2_3_replication_snapshot_v1",
        "status": "verified",
        "artifact_count": len(sources),
        "checksum_ledger": identity(ledger),
    }


def load_truth() -> tuple[dict[str, pd.DataFrame], list[Path]]:
    for path in TRUTH_INPUTS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    normal = pd.read_csv(TRUTH_INPUTS["normal"])
    normal = normal.loc[
        (normal["detected"] == 1) & (normal["rna_dp"] >= 1)
    ].copy()
    return (
        {
            "normal": normal,
            "support": pd.read_csv(TRUTH_INPUTS["support"]),
            "somatic": pd.read_csv(TRUTH_INPUTS["somatic"]),
            "table_s1": pd.read_csv(TRUTH_INPUTS["table_s1"], sep="\t"),
        },
        list(TRUTH_INPUTS.values()),
    )


def collect(
    *,
    selected: Sequence[str],
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
    replication_root: Path,
    plan_root: Path,
    protocol: Path,
    extension_protocol: Path,
    maxlr_protocol: Path,
    replication_protocol: Path,
    contracts,
    benchmark,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, object]], list[Path]]:
    truth, truth_paths = load_truth()
    metrics = []
    summaries = []
    run_records = []
    runtime_paths = []
    evidence_by_sample = {sample: {} for sample in SAMPLES}
    signatures = set()
    raw_by_sample: dict[str, str] = {}
    runtimes: dict[tuple[str, str], float] = {}

    for base in selected:
        setting = SETTINGS[base]
        for seed in SEEDS:
            config = run_config(base, seed)
            for sample in SAMPLES:
                root = source_run_root(
                    base=base,
                    seed=seed,
                    sample=sample,
                    factorial_root=factorial_root,
                    extension_root=extension_root,
                    maxlr_root=maxlr_root,
                    replication_root=replication_root,
                )
                if (root / "RUN_FAILED.txt").exists():
                    raise ValueError(f"run carries failure marker: {root}")
                validation = contracts.validate_v2_3_run(root)
                if validation.get("status") != "verified":
                    raise ValueError(f"run is not verified: {root}")
                manifest_path = root / "V2_3_MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest.get("production_inputs_include_wes") is not False
                    or manifest.get("evaluation_uses_wes") is not False
                    or manifest.get("dataset") != f"{sample}_TUMOR"
                    or manifest.get("quality_filter") != "baseQ0mapQ0"
                    or manifest.get("anchor_score") != "p_nonref"
                    or not math.isclose(
                        float(manifest.get("min_anchor_probability")),
                        0.90,
                        rel_tol=0,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(f"invalid run contract: {root}")
                params = manifest["beagle_parameters"]
                expected_params = {
                    "MODEL_SCALE": setting["modelscale"],
                    "ITERATIONS": setting["niterations"],
                    "SEED": seed,
                    "IMPUTE": False,
                    "GPROBS": True,
                    "MAX_LR": setting["maxlr"],
                    "GENETIC_MAP": None,
                    "THREADS": 30,
                    "PARALLEL_CHROMOSOMES": 1,
                    "MEMORY": "10g",
                }
                for key, expected in expected_params.items():
                    if params.get(key) != expected:
                        raise ValueError(f"parameter mismatch {root}/{key}")
                signatures.add(json.dumps(manifest_signature(manifest), sort_keys=True))
                raw_hash = str(manifest["raw_vcf"]["sha256"])
                if sample in raw_by_sample and raw_by_sample[sample] != raw_hash:
                    raise ValueError(f"raw input differs across runs for {sample}")
                raw_by_sample[sample] = raw_hash

                quality = str(manifest["quality_filter"])
                audit = (
                    root
                    / "output_VCFs"
                    / "beagle"
                    / quality
                    / "beagle_evidence_audit.tsv.gz"
                )
                evidence = benchmark.reroute_evidence(
                    benchmark.read_evidence(audit), "p_nonref", 0.90
                )
                evidence_by_sample[sample][config] = evidence
                sample_metrics, sample_summary = benchmark.evaluate(
                    config,
                    sample,
                    evidence,
                    truth["normal"],
                    truth["support"],
                    truth["somatic"],
                    truth["table_s1"],
                )
                metrics.extend(sample_metrics)
                summaries.append(sample_summary)
                runtime, runtime_path = read_runtime(root, manifest)
                runtimes[(config, sample)] = runtime
                runtime_paths.append(runtime_path)
                snapshot_validation = (
                    validate_replication_snapshot(
                        root=root,
                        plan_root=plan_root,
                        protocol=protocol,
                        extension_protocol=extension_protocol,
                        maxlr_protocol=maxlr_protocol,
                        replication_protocol=replication_protocol,
                    )
                    if seed != SEEDS[0]
                    else None
                )
                run_records.append({
                    "base_config": base,
                    "config": config,
                    "sample": sample,
                    "run_root": str(root.resolve()),
                    "manifest": identity(manifest_path),
                    "completion": identity(root / "RUN_COMPLETE.json"),
                    "audit": identity(audit),
                    "runtime_metrics": identity(runtime_path),
                    "beagle_runtime_seconds": runtime,
                    "validation": validation,
                    "replication_snapshot_validation": snapshot_validation,
                })
    if len(signatures) != 1:
        raise ValueError("replicated runs do not share code/executable/reference identity")

    # The frozen replication contract requires one exact allele universe per
    # sample across every setting and seed. Equal raw-VCF hashes are necessary
    # but not sufficient: a malformed/stale routed audit could contain a
    # different set of keys. Compare the audited keys before any endpoint is
    # calculated.
    for sample, by_config in evidence_by_sample.items():
        expected_configs = {
            run_config(base, seed) for base in selected for seed in SEEDS
        }
        if set(by_config) != expected_configs:
            raise ValueError(
                f"replication evidence grid is incomplete for {sample}"
            )
        ordered = sorted(by_config)
        reference_config = ordered[0]
        reference_keys = set(by_config[reference_config])
        for config in ordered[1:]:
            observed_keys = set(by_config[config])
            if observed_keys != reference_keys:
                raise ValueError(
                    "replication exact-allele universe differs for "
                    f"{sample}: {config} versus {reference_config}; "
                    f"missing={len(reference_keys - observed_keys)}, "
                    f"extra={len(observed_keys - reference_keys)}"
                )

    metrics_frame = pd.DataFrame(metrics)
    pairwise = pd.DataFrame(
        [
            row
            for sample in SAMPLES
            for row in benchmark.pairwise_stability(
                evidence_by_sample[sample], sample
            )
        ]
    )
    wide = metrics_frame.pivot(
        index=["config", "sample"], columns="metric", values="value"
    ).reset_index()
    summary_frame = pd.DataFrame(summaries).merge(
        wide, on=["config", "sample"], how="left"
    )
    summary_frame["beagle_runtime_seconds"] = [
        runtimes[(str(row.config), str(row.sample))]
        for row in summary_frame.itertuples(index=False)
    ]
    return metrics_frame, summary_frame, pairwise, run_records, truth_paths


def _metric_lookup(metrics: pd.DataFrame):
    required = {"config", "sample", "metric", "numerator", "denominator", "value"}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"replication metrics lack columns: {sorted(missing)}")
    chosen = metrics.loc[metrics["metric"].isin(METRICS)].copy()
    for column in ("numerator", "denominator", "value"):
        chosen[column] = pd.to_numeric(chosen[column], errors="raise")
    if (
        (chosen["denominator"] <= 0).any()
        or (chosen["numerator"] < 0).any()
        or (chosen["numerator"] > chosen["denominator"]).any()
    ):
        raise ValueError("replication metrics contain invalid counts")
    expected_values = chosen["numerator"] / chosen["denominator"]
    if not all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
        for a, b in zip(expected_values, chosen["value"])
    ):
        raise ValueError("replication metric values do not equal counts")
    if chosen.duplicated(["config", "sample", "metric"]).any():
        raise ValueError("replication metrics contain duplicate rows")
    return chosen.set_index(["config", "sample", "metric"])


def assess_replication(
    selected: Sequence[str],
    metrics: pd.DataFrame,
    pairwise: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    expected_metric_rows = {
        (run_config(base, seed), sample, metric)
        for base in selected
        for seed in SEEDS
        for sample in SAMPLES
        for metric in METRICS
    }
    lookup = _metric_lookup(metrics)
    observed_metric_rows = set(lookup.index)
    if observed_metric_rows != expected_metric_rows:
        raise ValueError(
            "replication metric grid is not exact; "
            f"missing={len(expected_metric_rows - observed_metric_rows)}, "
            f"extra={len(observed_metric_rows - expected_metric_rows)}"
        )
    required_pairwise = {
        "sample",
        "config_a",
        "config_b",
        "raw_jaccard",
        "anchor_jaccard",
        "route_agreement",
        "p_nonref_mae",
    }
    if missing := required_pairwise.difference(pairwise.columns):
        raise ValueError(f"pairwise stability lacks columns: {sorted(missing)}")

    within = pairwise.loc[
        pairwise.apply(
            lambda row: base_config(row["config_a"])
            == base_config(row["config_b"]),
            axis=1,
        )
    ].copy()
    expected_pairs = {
        (
            sample,
            base,
            *sorted((run_config(base, left), run_config(base, right))),
        )
        for sample in SAMPLES
        for base in selected
        for index, left in enumerate(SEEDS)
        for right in SEEDS[index + 1 :]
    }
    observed_pairs = {
        (
            str(row.sample),
            base_config(str(row.config_a)),
            *sorted((str(row.config_a), str(row.config_b))),
        )
        for row in within.itertuples(index=False)
    }
    if observed_pairs != expected_pairs or len(within) != len(expected_pairs):
        raise ValueError("within-setting stability grid is not exact")
    for column in (
        "raw_jaccard",
        "anchor_jaccard",
        "route_agreement",
        "p_nonref_mae",
    ):
        within[column] = pd.to_numeric(within[column], errors="raise")
        if not all(math.isfinite(float(value)) for value in within[column]):
            raise ValueError(f"non-finite pairwise metric: {column}")
    for column in ("raw_jaccard", "anchor_jaccard", "route_agreement"):
        if ((within[column] < 0) | (within[column] > 1)).any():
            raise ValueError(f"pairwise fraction is outside [0,1]: {column}")
    if (within["p_nonref_mae"] < 0).any():
        raise ValueError("pairwise p_nonref_mae is negative")
    if not all(
        math.isclose(float(value), 1.0, rel_tol=0, abs_tol=1e-15)
        for value in within["raw_jaccard"]
    ):
        raise ValueError(
            "replication exact-allele universe differs within a setting"
        )

    def value(config: str, sample: str, metric: str) -> float:
        return float(lookup.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "numerator"])

    required_summary = {"config", "sample", "beagle_runtime_seconds"}
    if missing := required_summary.difference(summary.columns):
        raise ValueError(f"runtime summary lacks columns: {sorted(missing)}")
    runtime_frame = summary.loc[:, sorted(required_summary)].copy()
    expected_runtime_rows = {
        (run_config(base, seed), sample)
        for base in selected
        for seed in SEEDS
        for sample in SAMPLES
    }
    observed_runtime_rows = set(
        runtime_frame[["config", "sample"]].itertuples(index=False, name=None)
    )
    if (
        observed_runtime_rows != expected_runtime_rows
        or len(runtime_frame) != len(expected_runtime_rows)
    ):
        raise ValueError("runtime summary grid is not exact")
    runtime_frame["beagle_runtime_seconds"] = pd.to_numeric(
        runtime_frame["beagle_runtime_seconds"], errors="raise"
    )
    if any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in runtime_frame["beagle_runtime_seconds"]
    ):
        raise ValueError("runtime summary contains an invalid duration")
    runtime_lookup = runtime_frame.set_index(["config", "sample"])[
        "beagle_runtime_seconds"
    ]
    if runtime_lookup.index.duplicated().any():
        raise ValueError("runtime summary contains duplicate config/sample rows")

    aggregates = {}
    for base in selected:
        stability = within.loc[
            within["config_a"].map(base_config).eq(base)
        ]
        if len(stability) != len(SAMPLES) * 3:
            raise ValueError(f"incomplete pairwise stability for {base}")
        aggregates[base] = {
            "mean_sensitivity": statistics.mean(
                value(run_config(base, seed), sample, "normal_wes_anchor_sensitivity")
                for seed in SEEDS
                for sample in SAMPLES
            ),
            "mean_support": statistics.mean(
                value(
                    run_config(base, seed),
                    sample,
                    "selected_anchor_normal_wes_support",
                )
                for seed in SEEDS
                for sample in SAMPLES
            ),
            "total_leakage": sum(
                numerator(
                    run_config(base, seed),
                    sample,
                    "wes_somatic_anchor_leakage",
                )
                for seed in SEEDS
                for sample in SAMPLES
            ),
            "min_anchor_jaccard": float(stability["anchor_jaccard"].min()),
            "min_route_agreement": float(stability["route_agreement"].min()),
            "median_anchor_jaccard": float(stability["anchor_jaccard"].median()),
            "median_p_nonref_mae": float(stability["p_nonref_mae"].median()),
            "mean_runtime_seconds": statistics.mean(
                float(runtime_lookup.loc[(run_config(base, seed), sample)])
                for seed in SEEDS
                for sample in SAMPLES
            ),
        }

    comparator = aggregates[COMPARATOR]
    rows = []
    for base in selected:
        item = aggregates[base]
        leakage_nonincrease = all(
            numerator(
                run_config(base, seed),
                sample,
                "wes_somatic_anchor_leakage",
            )
            <= numerator(
                run_config(COMPARATOR, seed),
                sample,
                "wes_somatic_anchor_leakage",
            )
            for seed in SEEDS
            for sample in SAMPLES
        )
        min_anchor_pass = item["min_anchor_jaccard"] >= MIN_ANCHOR_JACCARD
        min_route_pass = item["min_route_agreement"] >= MIN_ROUTE_AGREEMENT
        median_anchor_pass = (
            item["median_anchor_jaccard"]
            >= comparator["median_anchor_jaccard"] - MAX_MEDIAN_JACCARD_DEFICIT
        )
        median_mae_pass = (
            item["median_p_nonref_mae"]
            <= comparator["median_p_nonref_mae"]
            + MAX_MEDIAN_PNONREF_MAE_EXCESS
        )
        sensitivity_delta = (
            item["mean_sensitivity"] - comparator["mean_sensitivity"]
        )
        support_delta = item["mean_support"] - comparator["mean_support"]
        sensitivity_pass = sensitivity_delta >= -SENSITIVITY_MARGIN
        support_pass = support_delta >= -SUPPORT_MARGIN
        leakage_reduction = comparator["total_leakage"] - item["total_leakage"]
        material_improvement = (
            sensitivity_delta >= IMPROVEMENT_MARGIN
            or support_delta >= IMPROVEMENT_MARGIN
            or leakage_reduction > 0
        )
        is_comparator = base == COMPARATOR
        eligible = is_comparator or (
            min_anchor_pass
            and min_route_pass
            and median_anchor_pass
            and median_mae_pass
            and leakage_nonincrease
            and sensitivity_pass
            and support_pass
            and material_improvement
        )
        rows.append({
            "base_config": base,
            "comparator": is_comparator,
            "mean_normal_wes_anchor_sensitivity": item["mean_sensitivity"],
            "delta_mean_sensitivity_vs_comparator": sensitivity_delta,
            "replicated_sensitivity_no_harm": sensitivity_pass,
            "mean_selected_anchor_normal_wes_support": item["mean_support"],
            "delta_mean_support_vs_comparator": support_delta,
            "replicated_support_no_harm": support_pass,
            "total_wes_somatic_anchor_leakage": item["total_leakage"],
            "total_leakage_reduction_vs_comparator": leakage_reduction,
            "replicated_material_improvement": material_improvement,
            "minimum_anchor_jaccard": item["min_anchor_jaccard"],
            "minimum_anchor_jaccard_pass": min_anchor_pass,
            "minimum_route_agreement": item["min_route_agreement"],
            "minimum_route_agreement_pass": min_route_pass,
            "median_anchor_jaccard": item["median_anchor_jaccard"],
            "median_anchor_jaccard_vs_comparator_pass": median_anchor_pass,
            "median_p_nonref_mae": item["median_p_nonref_mae"],
            "median_p_nonref_mae_vs_comparator_pass": median_mae_pass,
            "matched_seed_sample_leakage_nonincrease": leakage_nonincrease,
            "mean_beagle_runtime_seconds": item["mean_runtime_seconds"],
            "simplicity_rank": SETTINGS[base]["simplicity"],
            "eligible_as_base_default": eligible,
        })
    result = pd.DataFrame(rows)
    eligible = result.loc[result["eligible_as_base_default"]].copy()
    eligible = eligible.sort_values(
        [
            "total_wes_somatic_anchor_leakage",
            "mean_normal_wes_anchor_sensitivity",
            "mean_selected_anchor_normal_wes_support",
            "median_anchor_jaccard",
            "mean_beagle_runtime_seconds",
            "simplicity_rank",
            "base_config",
        ],
        ascending=[True, False, False, False, True, True, True],
        kind="mergesort",
    )
    if eligible.empty:
        raise AssertionError("comparator fallback must always be eligible")
    winner = str(eligible.iloc[0]["base_config"])
    result["selected_base_default"] = result["base_config"].eq(winner)
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
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--factorial-experiment-root", type=Path, required=True)
    parser.add_argument("--extension-experiment-root", type=Path, required=True)
    parser.add_argument("--maxlr-experiment-root", type=Path, required=True)
    parser.add_argument("--replication-experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--contracts",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py",
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
        "BENCHMARK_PROTOCOL.md",
    )
    parser.add_argument(
        "--extension-protocol",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
        "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md",
    )
    parser.add_argument(
        "--replication-protocol",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
        "BENCHMARK_PROTOCOL_REPLICATION.md",
    )
    parser.add_argument(
        "--maxlr-protocol",
        type=Path,
        default=PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
        "BENCHMARK_PROTOCOL_MAXLR.md",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    for path in (
        args.protocol,
        args.extension_protocol,
        args.maxlr_protocol,
        args.replication_protocol,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected, _, plan_paths = verify_plan(args.plan_root)
    contracts = load_module(args.contracts, "_beagle_v23_replication_contracts")
    benchmark = load_module(args.benchmark, "_beagle_v23_replication_benchmark")
    metrics, summary, pairwise, run_records, truth_paths = collect(
        selected=selected,
        factorial_root=args.factorial_experiment_root,
        extension_root=args.extension_experiment_root,
        maxlr_root=args.maxlr_experiment_root,
        replication_root=args.replication_experiment_root,
        plan_root=args.plan_root,
        protocol=args.protocol,
        extension_protocol=args.extension_protocol,
        maxlr_protocol=args.maxlr_protocol,
        replication_protocol=args.replication_protocol,
        contracts=contracts,
        benchmark=benchmark,
    )
    gates, winner = assess_replication(selected, metrics, pairwise, summary)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_paths = {
        "metrics": args.output_dir / "replicated_configuration_metrics_long.tsv",
        "summary": args.output_dir / "replicated_configuration_summary.tsv",
        "pairwise": args.output_dir / "replicated_pairwise_stability.tsv",
        "gates": args.output_dir / "replication_gate_results.tsv",
        "selection": args.output_dir / "FINAL_BASE_CONFIGURATION.json",
        "provenance": args.output_dir / "PROVENANCE.json",
    }
    metrics.to_csv(output_paths["metrics"], sep="\t", index=False)
    summary.to_csv(output_paths["summary"], sep="\t", index=False)
    pairwise.to_csv(output_paths["pairwise"], sep="\t", index=False)
    gates.to_csv(output_paths["gates"], sep="\t", index=False)
    output_paths["selection"].write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_final_base_selection_v1",
                "selected_base_configuration": winner,
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "final_ensemble_selected": False,
                "selection_priority": [
                    "lower_somatic_leakage",
                    "higher_normal_wes_sensitivity",
                    "higher_selected_anchor_support",
                    "higher_seed_stability",
                    "lower_runtime",
                    "simpler_configuration",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    output_paths["provenance"].write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_replication_assessment_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "base_configuration_selected",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "development_set_evaluation": True,
                "selected_configurations": selected,
                "selected_base_configuration": winner,
                "seeds": list(SEEDS),
                "samples": list(SAMPLES),
                "gate_thresholds": {
                    "exact_allele_universe_required": True,
                    "minimum_anchor_jaccard": MIN_ANCHOR_JACCARD,
                    "minimum_route_agreement": MIN_ROUTE_AGREEMENT,
                    "maximum_median_anchor_jaccard_deficit": MAX_MEDIAN_JACCARD_DEFICIT,
                    "maximum_median_p_nonref_mae_excess": MAX_MEDIAN_PNONREF_MAE_EXCESS,
                    "no_matched_seed_sample_somatic_leakage_increase": True,
                    "replicated_sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
                    "replicated_support_no_harm_margin": SUPPORT_MARGIN,
                    "replicated_material_improvement_margin": IMPROVEMENT_MARGIN,
                },
                "assessor": identity(Path(__file__)),
                "integrity_module": identity(Path(INTEGRITY.__file__)),
                "pre_replication_audit_module": identity(
                    Path(PREAUDIT.__file__)
                ),
                "contracts": identity(args.contracts),
                "benchmark": identity(args.benchmark),
                "protocol": identity(args.protocol),
                "extension_protocol": identity(args.extension_protocol),
                "maxlr_protocol": identity(args.maxlr_protocol),
                "replication_protocol": identity(args.replication_protocol),
                "replication_launcher": identity(
                    PROJECT / "run_slurm/run_beagle_v2_3_replication.sh"
                ),
                "plan_inputs": [identity(path) for path in plan_paths],
                "truth_inputs": {
                    name: identity(path)
                    for name, path in zip(TRUTH_INPUTS, truth_paths)
                },
                "validated_runs": run_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    code_inputs = [
        Path(__file__),
        Path(INTEGRITY.__file__),
        Path(PREAUDIT.__file__),
        args.contracts,
        args.benchmark,
        args.protocol,
        args.extension_protocol,
        args.maxlr_protocol,
        args.replication_protocol,
        PROJECT / "run_slurm/run_beagle_v2_3_replication.sh",
        *plan_paths,
        *truth_paths,
    ]
    code_sums = args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_sums(code_inputs, PROJECT, code_sums)
    output_sums = args.output_dir / "OUTPUT_SHA256SUMS.txt"
    write_sums([*output_paths.values(), code_sums], args.output_dir, output_sums)
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    (args.output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_replication_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "selected_base_configuration": winner,
                "output_sha256sums": identity(output_sums),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(gates.to_string(index=False))
    print(f"Selected base configuration: {winner}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
