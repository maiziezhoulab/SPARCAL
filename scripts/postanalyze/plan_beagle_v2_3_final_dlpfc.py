#!/usr/bin/env python3
"""Freeze and verify the final Beagle-v2.3 DLPFC execution grid.

The selected numeric Beagle settings are read from the immutable seed-20260915
v2.3 manifests.  Configuration labels are identifiers only; they are never
parsed to recover parameters.  WES is part of the upstream development-set
assessment, but is not an input to this DLPFC production plan.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


PROJECT = Path(__file__).resolve().parents[2]
POSTANALYZE = Path(__file__).resolve().parent
if str(POSTANALYZE) not in sys.path:
    sys.path.insert(0, str(POSTANALYZE))
import beagle_selection_integrity as INTEGRITY  # noqa: E402


SECTIONS = (
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
)
SAMPLES = ("P4", "P6")
ALL_SEEDS = (20260915, 20260916, 20260917)
SINGLE_SEED = ALL_SEEDS[0]
REPRESENTATIONS = ("single_seed", "ensemble_mean", "ensemble_median")
TASK_COLUMNS = (
    "task_id",
    "section",
    "seed",
    "dataset",
    "modelscale",
    "niterations",
    "maxlr",
    "base_config",
    "representation",
    "section_root",
)
FINAL_REQUIRED = (
    "ensemble_gate_results.tsv",
    "FINAL_BEAGLE_REPRESENTATION.json",
    "PROVENANCE.json",
    "CODE_AND_INPUT_SHA256SUMS.txt",
    "OUTPUT_SHA256SUMS.txt",
    "RUN_COMPLETE",
    "RUN_COMPLETE.json",
)
BASE_REQUIRED = (
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
)
PARAMETER_KEYS = {
    "THREADS",
    "MEMORY",
    "MODEL_SCALE",
    "ITERATIONS",
    "IMPUTE",
    "GPROBS",
    "SEED",
    "PARALLEL_CHROMOSOMES",
    "MAX_LR",
    "GENETIC_MAP",
    "REFERENCE_LEDGER",
}
PROTOCOL = (
    PROJECT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
    "BENCHMARK_PROTOCOL_DLPFC_FINAL.md"
)
LAUNCHER = PROJECT / "run_slurm/run_beagle_v2_3_final_dlpfc.sh"
EXECUTION_CODE = (
    PROJECT / "scripts/2_beagle_filtering/run_beagle.py",
    PROJECT / "scripts/2_beagle_filtering/beagle_evidence_router.py",
    PROJECT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py",
    PROJECT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
    "run_beagle_v2_3.py",
)


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


def strict_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{label} is not a strict boolean: {value!r}")


def _required_paths(root: Path, names: Sequence[str], label: str) -> list[Path]:
    root = Path(root).resolve()
    paths = [root / name for name in names]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"{label} is incomplete: {missing}")
    if (root / "RUN_COMPLETE").read_text().strip() != "complete":
        raise ValueError(f"{label} has invalid RUN_COMPLETE")
    return paths


def _verify_assessment_bundle(
    root: Path,
    names: Sequence[str],
    *,
    expected_outputs: Sequence[str],
    label: str,
) -> tuple[Mapping[str, object], Mapping[str, object], list[Path]]:
    paths = _required_paths(root, names, label)
    root = Path(root).resolve()
    provenance = json.loads((root / "PROVENANCE.json").read_text())
    completion = json.loads((root / "RUN_COMPLETE.json").read_text())
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=expected_outputs,
        label=label,
    )
    INTEGRITY.parse_and_verify_ledger(
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label=f"{label} code/input ledger",
    )
    return provenance, completion, paths


def load_final_selection(
    root: Path,
) -> tuple[str, str, Mapping[str, object], list[Path]]:
    root = Path(root).resolve()
    provenance, completion, paths = _verify_assessment_bundle(
        root,
        FINAL_REQUIRED,
        expected_outputs=(
            "ensemble_gate_results.tsv",
            "FINAL_BEAGLE_REPRESENTATION.json",
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
        ),
        label="final Beagle representation selection",
    )
    selection = json.loads(
        (root / "FINAL_BEAGLE_REPRESENTATION.json").read_text()
    )
    base = str(selection.get("base_configuration", ""))
    representation = str(selection.get("selected_representation", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base):
        raise ValueError("final selection has an unsafe/empty base configuration")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown final Beagle representation: {representation}")
    expected_seed = SINGLE_SEED if representation == "single_seed" else None
    expected_multiplier = 1 if representation == "single_seed" else 3
    if (
        selection.get("schema")
        != "sparcal_beagle_v2_3_final_representation_v1"
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or selection.get("selected_seed") != expected_seed
        or selection.get("beagle_run_multiplier") != expected_multiplier
        or provenance.get("schema")
        != "sparcal_beagle_v2_3_ensemble_selection_v1"
        or provenance.get("status") != "final_representation_selected"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("base_configuration") != base
        or provenance.get("selected_representation") != representation
        or provenance.get("frozen_seeds") != list(ALL_SEEDS)
        or completion.get("schema")
        != "sparcal_beagle_v2_3_ensemble_selection_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or completion.get("base_configuration") != base
        or completion.get("selected_representation") != representation
    ):
        raise ValueError("final representation selection contract is inconsistent")

    with (root / "ensemble_gate_results.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if len(rows) != 2 or {row.get("method") for row in rows} != {
        "ensemble_mean", "ensemble_median"
    }:
        raise ValueError("ensemble gate table is not the exact two-method grid")
    selected_methods = [
        str(row["method"])
        for row in rows
        if strict_bool(
            row.get("selected_final_representation"),
            "ensemble final-selection flag",
        )
    ]
    expected_methods = [] if representation == "single_seed" else [representation]
    if selected_methods != expected_methods:
        raise ValueError("ensemble gate and final representation disagree")
    return base, representation, provenance, paths


def load_base_assessment(
    root: Path,
    expected_base: str,
) -> tuple[Mapping[str, object], Mapping[str, object], list[Path]]:
    root = Path(root).resolve()
    provenance, completion, paths = _verify_assessment_bundle(
        root,
        BASE_REQUIRED,
        expected_outputs=(
            "replicated_configuration_metrics_long.tsv",
            "replicated_configuration_summary.tsv",
            "replicated_pairwise_stability.tsv",
            "replication_gate_results.tsv",
            "FINAL_BASE_CONFIGURATION.json",
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
        ),
        label="base Beagle replication assessment",
    )
    selection = json.loads((root / "FINAL_BASE_CONFIGURATION.json").read_text())
    if (
        selection.get("schema")
        != "sparcal_beagle_v2_3_final_base_selection_v1"
        or selection.get("selected_base_configuration") != expected_base
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or selection.get("final_ensemble_selected") is not False
        or provenance.get("schema")
        != "sparcal_beagle_v2_3_replication_assessment_v1"
        or provenance.get("status") != "base_configuration_selected"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("selected_base_configuration") != expected_base
        or provenance.get("seeds") != list(ALL_SEEDS)
        or provenance.get("samples") != list(SAMPLES)
        or completion.get("schema")
        != "sparcal_beagle_v2_3_replication_completion_v1"
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or completion.get("selected_base_configuration") != expected_base
    ):
        raise ValueError("base replication assessment contract is inconsistent")
    selected = provenance.get("selected_configurations")
    if not isinstance(selected, list) or expected_base not in selected:
        raise ValueError("selected base is absent from the replicated configurations")
    return provenance, completion, paths


def selected_seed_parameters(
    base_provenance: Mapping[str, object],
    base: str,
) -> tuple[dict[str, object], list[dict[str, object]], Path]:
    validated = base_provenance.get("validated_runs")
    if not isinstance(validated, list):
        raise ValueError("base assessment lacks validated-run provenance")
    records: dict[str, Mapping[str, object]] = {}
    for row in validated:
        if not isinstance(row, Mapping):
            raise ValueError("base assessment contains an invalid validated-run row")
        if (
            row.get("config") == f"{base}_seed{SINGLE_SEED}"
            and row.get("sample") in SAMPLES
        ):
            sample = str(row["sample"])
            if sample in records:
                raise ValueError(f"duplicate selected seed manifest for {sample}")
            records[sample] = row
    if set(records) != set(SAMPLES):
        raise ValueError("base assessment lacks both selected seed manifests")

    manifest_records = []
    parameter_sets = []
    reference_roots = []
    for sample in SAMPLES:
        row = records[sample]
        run_root = Path(str(row.get("run_root", ""))).resolve()
        manifest_path = INTEGRITY.verify_identity(
            row.get("manifest"),
            label=f"selected seed manifest {sample}",
            expected_path=run_root / "V2_3_MANIFEST.json",
        )
        manifest = json.loads(manifest_path.read_text())
        params = manifest.get("beagle_parameters")
        if not isinstance(params, dict) or set(params) != PARAMETER_KEYS:
            raise ValueError(f"{sample} selected manifest has unexpected parameters")
        if (
            manifest.get("schema") != "sparcal_beagle_run_v2_3_candidate_1"
            or manifest.get("status") != "outputs_ready_for_checksum"
            or manifest.get("production_inputs_include_wes") is not False
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
            or params.get("SEED") != SINGLE_SEED
            or params.get("IMPUTE") is not False
            or params.get("GPROBS") is not True
            or params.get("GENETIC_MAP") is not None
            or params.get("THREADS") != 30
            or params.get("PARALLEL_CHROMOSOMES") != 1
            or params.get("MEMORY") != "10g"
        ):
            raise ValueError(f"{sample} selected seed manifest contract is invalid")
        for field in ("ITERATIONS", "SEED", "THREADS", "PARALLEL_CHROMOSOMES"):
            if isinstance(params[field], bool) or not isinstance(params[field], int):
                raise ValueError(f"invalid integer Beagle parameter {field}")
        if params["ITERATIONS"] < 0:
            raise ValueError("ITERATIONS must be nonnegative")
        for field in ("MODEL_SCALE", "MAX_LR"):
            value = params[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"invalid numeric Beagle parameter {field}")
        reference = manifest.get("reference_ledger")
        if not isinstance(reference, dict):
            raise ValueError("selected seed manifest lacks reference-ledger provenance")
        reference_root = Path(str(reference.get("root", ""))).resolve()
        requested_reference = Path(str(params.get("REFERENCE_LEDGER", ""))).resolve()
        if reference_root != requested_reference:
            raise ValueError("selected manifest parameter/reference roots disagree")
        manifest_records.append({"sample": sample, "manifest": identity(manifest_path)})
        parameter_sets.append(params)
        reference_roots.append(reference_root)
    if parameter_sets[0] != parameter_sets[1]:
        raise ValueError("P4/P6 selected seed manifests have different parameters")
    if reference_roots[0] != reference_roots[1]:
        raise ValueError("P4/P6 selected seed manifests use different references")
    return dict(parameter_sets[0]), manifest_records, reference_roots[0]


def _expected_seeds(representation: str) -> tuple[int, ...]:
    return (SINGLE_SEED,) if representation == "single_seed" else ALL_SEEDS


def build_plan(
    final_selection_root: Path,
    base_assessment_root: Path,
    run_root: Path,
    output_dir: Path,
    protocol: Path = PROTOCOL,
) -> Mapping[str, object]:
    output_dir = Path(output_dir)
    run_root = Path(run_root).resolve()
    protocol = Path(protocol).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite plan: {output_dir}")
    if run_root.exists():
        raise FileExistsError(f"final DLPFC run root must be fresh: {run_root}")
    if not protocol.is_file():
        raise FileNotFoundError(protocol)

    base, representation, _, final_paths = load_final_selection(
        final_selection_root
    )
    base_provenance, _, base_paths = load_base_assessment(
        base_assessment_root, base
    )
    parameters, manifests, reference_root = selected_seed_parameters(
        base_provenance, base
    )
    seeds = _expected_seeds(representation)
    rows = []
    for seed in seeds:
        for section in SECTIONS:
            section_root = run_root / f"{base}_seed{seed}" / section
            rows.append({
                "task_id": len(rows),
                "section": section,
                "seed": seed,
                "dataset": "DLPFC",
                "modelscale": parameters["MODEL_SCALE"],
                "niterations": parameters["ITERATIONS"],
                "maxlr": parameters["MAX_LR"],
                "base_config": base,
                "representation": representation,
                "section_root": str(section_root),
            })

    output_dir.mkdir(parents=True, exist_ok=False)
    task_path = output_dir / "dlpfc_tasks.tsv"
    with task_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TASK_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    final_identities = [identity(path) for path in final_paths]
    base_identities = [identity(path) for path in base_paths]
    shared = {
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "base_config": base,
        "representation": representation,
        "seeds": list(seeds),
        "sections": list(SECTIONS),
        "task_count": len(rows),
        "array_max": len(rows) - 1,
        "final_selection_inputs": final_identities,
        "base_assessment_inputs": base_identities,
        "selected_seed20260915_manifests": manifests,
    }
    provenance_path = output_dir / "PROVENANCE.json"
    provenance = {
        "schema": "sparcal_beagle_v2_3_final_dlpfc_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "tasks_frozen",
        **shared,
        "dataset": "DLPFC",
        "quality_filter": "baseQ0mapQ0",
        "anchor_score": "p_nonref",
        "min_anchor_probability": 0.90,
        "run_root": str(run_root),
        "final_selection_root": str(Path(final_selection_root).resolve()),
        "base_assessment_root": str(Path(base_assessment_root).resolve()),
        "beagle_parameters": parameters,
        "reference_ledger_root": str(reference_root),
        "planner": identity(Path(__file__)),
        "integrity_module": identity(Path(INTEGRITY.__file__)),
        "protocol": identity(protocol),
        "launcher": identity(LAUNCHER),
        "execution_code": [identity(path) for path in EXECUTION_CODE],
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    output_sums = output_dir / "OUTPUT_SHA256SUMS.txt"
    output_sums.write_text(
        f"{sha256(task_path)}  {task_path.name}\n"
        f"{sha256(provenance_path)}  {provenance_path.name}\n"
    )
    (output_dir / "RUN_COMPLETE").write_text("complete\n")
    completion = {
        "schema": "sparcal_beagle_v2_3_final_dlpfc_plan_completion_v1",
        "status": "complete",
        **shared,
        "output_sha256sums": identity(output_sums),
    }
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    validate_frozen_plan(output_dir, expected_run_root=run_root)
    return {
        "status": "complete",
        "task_count": len(rows),
        "array_max": len(rows) - 1,
        "base_config": base,
        "representation": representation,
        "seeds": list(seeds),
        "task_file": str(task_path.resolve()),
    }


def _verify_identity_list(records: object, label: str) -> list[Path]:
    if not isinstance(records, list) or not records:
        raise ValueError(f"{label} is missing/empty")
    return [
        INTEGRITY.verify_identity(record, label=f"{label} {index}")
        for index, record in enumerate(records)
    ]


def validate_frozen_plan(
    plan_root: Path,
    *,
    expected_run_root: Path | None = None,
    expected_final_selection_root: Path | None = None,
    expected_base_assessment_root: Path | None = None,
    expected_reference_ledger_root: Path | None = None,
) -> Mapping[str, object]:
    plan_root = Path(plan_root).resolve()
    paths = _required_paths(
        plan_root,
        (
            "dlpfc_tasks.tsv",
            "PROVENANCE.json",
            "OUTPUT_SHA256SUMS.txt",
            "RUN_COMPLETE",
            "RUN_COMPLETE.json",
        ),
        "final DLPFC plan",
    )
    provenance = json.loads((plan_root / "PROVENANCE.json").read_text())
    completion = json.loads((plan_root / "RUN_COMPLETE.json").read_text())
    if (
        provenance.get("schema")
        != "sparcal_beagle_v2_3_final_dlpfc_plan_v1"
        or provenance.get("status") != "tasks_frozen"
        or completion.get("schema")
        != "sparcal_beagle_v2_3_final_dlpfc_plan_completion_v1"
        or completion.get("status") != "complete"
    ):
        raise ValueError("invalid final DLPFC plan schema/status")
    shared_fields = (
        "production_inputs_include_wes",
        "evaluation_uses_wes",
        "base_config",
        "representation",
        "seeds",
        "sections",
        "task_count",
        "array_max",
        "final_selection_inputs",
        "base_assessment_inputs",
        "selected_seed20260915_manifests",
    )
    if any(provenance.get(field) != completion.get(field) for field in shared_fields):
        raise ValueError("DLPFC plan provenance/completion fields disagree")
    if (
        provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not False
        or provenance.get("dataset") != "DLPFC"
        or provenance.get("quality_filter") != "baseQ0mapQ0"
        or provenance.get("anchor_score") != "p_nonref"
        or not math.isclose(
            float(provenance.get("min_anchor_probability")),
            0.90,
            rel_tol=0,
            abs_tol=1e-12,
        )
        or provenance.get("sections") != list(SECTIONS)
        or provenance.get("representation") not in REPRESENTATIONS
    ):
        raise ValueError("invalid final DLPFC production/method contract")
    representation = str(provenance["representation"])
    seeds = _expected_seeds(representation)
    if provenance.get("seeds") != list(seeds):
        raise ValueError("DLPFC plan has the wrong seed set")
    base = str(provenance.get("base_config", ""))
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", base):
        raise ValueError("DLPFC plan has an unsafe base configuration")
    run_root = Path(str(provenance.get("run_root", ""))).resolve()
    if expected_run_root is not None and run_root != Path(expected_run_root).resolve():
        raise ValueError("DLPFC plan points to a different run root")
    final_root = Path(str(provenance.get("final_selection_root", ""))).resolve()
    base_root = Path(str(provenance.get("base_assessment_root", ""))).resolve()
    reference_root = Path(str(provenance.get("reference_ledger_root", ""))).resolve()
    if (
        expected_final_selection_root is not None
        and final_root != Path(expected_final_selection_root).resolve()
    ):
        raise ValueError("DLPFC plan points to a different final-selection root")
    if (
        expected_base_assessment_root is not None
        and base_root != Path(expected_base_assessment_root).resolve()
    ):
        raise ValueError("DLPFC plan points to a different base-assessment root")
    if (
        expected_reference_ledger_root is not None
        and reference_root != Path(expected_reference_ledger_root).resolve()
    ):
        raise ValueError("DLPFC plan points to a different reference-ledger root")

    INTEGRITY.verify_output_bundle(
        plan_root,
        completion,
        expected_names=("dlpfc_tasks.tsv", "PROVENANCE.json"),
        label="final DLPFC plan",
    )
    final_inputs = _verify_identity_list(
        provenance.get("final_selection_inputs"), "final-selection input"
    )
    base_inputs = _verify_identity_list(
        provenance.get("base_assessment_inputs"), "base-assessment input"
    )
    if (
        {path.name for path in final_inputs} != set(FINAL_REQUIRED)
        or len(final_inputs) != len(FINAL_REQUIRED)
        or {path.name for path in base_inputs} != set(BASE_REQUIRED)
        or len(base_inputs) != len(BASE_REQUIRED)
        or any(path.parent != final_root for path in final_inputs)
        or any(path.parent != base_root for path in base_inputs)
    ):
        raise ValueError("DLPFC plan binds an incorrect upstream input set")
    manifest_rows = provenance.get("selected_seed20260915_manifests")
    if (
        not isinstance(manifest_rows, list)
        or [row.get("sample") for row in manifest_rows if isinstance(row, Mapping)]
        != list(SAMPLES)
        or any(not isinstance(row, Mapping) or set(row) != {"sample", "manifest"} for row in manifest_rows)
    ):
        raise ValueError("DLPFC plan has an invalid selected-manifest list")
    for row in manifest_rows:
        INTEGRITY.verify_identity(
            row["manifest"], label=f"selected manifest {row['sample']}"
        )
    for key, expected in (
        ("planner", Path(__file__)),
        ("integrity_module", Path(INTEGRITY.__file__)),
        ("protocol", PROTOCOL),
        ("launcher", LAUNCHER),
    ):
        INTEGRITY.verify_identity(
            provenance.get(key), label=f"DLPFC plan {key}", expected_path=expected
        )
    execution_records = provenance.get("execution_code")
    if not isinstance(execution_records, list) or len(execution_records) != len(
        EXECUTION_CODE
    ):
        raise ValueError("DLPFC plan has an invalid execution-code identity list")
    for index, (record, expected) in enumerate(
        zip(execution_records, EXECUTION_CODE)
    ):
        INTEGRITY.verify_identity(
            record,
            label=f"DLPFC execution code {index}",
            expected_path=expected,
        )

    params = provenance.get("beagle_parameters")
    if not isinstance(params, dict) or set(params) != PARAMETER_KEYS:
        raise ValueError("DLPFC plan has an invalid Beagle parameter dictionary")
    if (
        params.get("SEED") != SINGLE_SEED
        or params.get("IMPUTE") is not False
        or params.get("GPROBS") is not True
        or params.get("GENETIC_MAP") is not None
        or params.get("THREADS") != 30
        or params.get("PARALLEL_CHROMOSOMES") != 1
        or params.get("MEMORY") != "10g"
        or Path(str(params.get("REFERENCE_LEDGER", ""))).resolve() != reference_root
    ):
        raise ValueError("DLPFC plan cannot replay the selected Beagle parameters")

    with (plan_root / "dlpfc_tasks.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if reader.fieldnames != list(TASK_COLUMNS):
        raise ValueError("DLPFC task table has an unexpected header")
    expected_count = len(SECTIONS) * len(seeds)
    if (
        len(rows) != expected_count
        or provenance.get("task_count") != expected_count
        or provenance.get("array_max") != expected_count - 1
    ):
        raise ValueError("DLPFC task count/array bounds are inconsistent")
    expected_grid = [(seed, section) for seed in seeds for section in SECTIONS]
    for task_id, (row, (seed, section)) in enumerate(zip(rows, expected_grid)):
        canonical = run_root / f"{base}_seed{seed}" / section
        try:
            observed_root = Path(row["section_root"]).resolve()
            modelscale = float(row["modelscale"])
            niterations = int(row["niterations"])
            maxlr = float(row["maxlr"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid DLPFC task row {task_id}") from error
        if (
            row["task_id"] != str(task_id)
            or row["section"] != section
            or row["seed"] != str(seed)
            or row["dataset"] != "DLPFC"
            or modelscale != float(params["MODEL_SCALE"])
            or niterations != int(params["ITERATIONS"])
            or maxlr != float(params["MAX_LR"])
            or row["base_config"] != base
            or row["representation"] != representation
            or observed_root != canonical
        ):
            raise ValueError(f"DLPFC task row {task_id} differs from the frozen grid")
    return {
        "status": "verified",
        "task_count": expected_count,
        "array_max": expected_count - 1,
        "base_config": base,
        "representation": representation,
        "seeds": list(seeds),
        "run_root": str(run_root),
        "paths": [str(path) for path in paths],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-selection-root", type=Path)
    parser.add_argument("--base-assessment-root", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--verify-plan-root", type=Path)
    parser.add_argument("--expected-run-root", type=Path)
    parser.add_argument("--expected-final-selection-root", type=Path)
    parser.add_argument("--expected-base-assessment-root", type=Path)
    parser.add_argument("--expected-reference-ledger-root", type=Path)
    args = parser.parse_args()
    if args.verify_plan_root is not None:
        if any(
            value is not None
            for value in (
                args.final_selection_root,
                args.base_assessment_root,
                args.run_root,
                args.output_dir,
            )
        ):
            parser.error("verification mode cannot include plan-build arguments")
        result = validate_frozen_plan(
            args.verify_plan_root,
            expected_run_root=args.expected_run_root,
            expected_final_selection_root=args.expected_final_selection_root,
            expected_base_assessment_root=args.expected_base_assessment_root,
            expected_reference_ledger_root=args.expected_reference_ledger_root,
        )
    else:
        required = {
            "--final-selection-root": args.final_selection_root,
            "--base-assessment-root": args.base_assessment_root,
            "--run-root": args.run_root,
            "--output-dir": args.output_dir,
        }
        if missing := [name for name, value in required.items() if value is None]:
            parser.error(f"plan-build mode requires: {', '.join(missing)}")
        result = build_plan(
            args.final_selection_root,
            args.base_assessment_root,
            args.run_root,
            args.output_dir,
            args.protocol,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
