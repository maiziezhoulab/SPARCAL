#!/usr/bin/env python3
"""Materialize the frozen final Beagle representation for DLPFC clustering.

The input is an immutable 12-row (single seed) or 36-row (three-seed
ensemble) DLPFC plan and the corresponding completed Beagle v2.3 run root.
Every run bundle is checksum- and contract-validated before an output root is
created.  No WES, histology annotation, or cluster label is accepted or read.

For each section this program creates two column-only projections of the
frozen historical position-only SPARCAL matrix:

* ``all_panel_present`` is the primary downstream compatibility interface;
* ``P(nonref)>=0.90`` is an anchor-only mechanistic diagnostic.

At multiallelic positions the prespecified rule is ``any``.  In contrast to
the historical generic matrix builder, this program fails closed if even one
candidate panel-present position is absent from the source matrix, rather than
silently dropping it.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
import math
import re
import shutil
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_materialization_v1"
SECTION_SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_section_v1"
COMPLETION_SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_materialization_completion_v1"
SECTION_COMPLETION_SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_section_completion_v1"
PLAN_SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_plan_v1"
PLAN_COMPLETION_SCHEMA = "sparcal_beagle_v2_3_final_dlpfc_plan_completion_v1"
SECTIONS = (
    "151507", "151508", "151509", "151510", "151669", "151670",
    "151671", "151672", "151673", "151674", "151675", "151676",
)
ENSEMBLE_SEEDS = (20260915, 20260916, 20260917)
SINGLE_SEEDS = (20260915,)
REPRESENTATIONS = {"single_seed", "ensemble_mean", "ensemble_median"}
QUALITY_FILTER = "baseQ0mapQ0"
PANEL_ROUTES = frozenset({
    "beagle_nonref_highconf",
    "beagle_ref_highconf",
    "beagle_uncertain",
})
ALL_ROUTES = PANEL_ROUTES | {"panel_absent"}
THRESHOLD = 0.90
PLAN_HEADER = (
    "task_id", "section", "seed", "dataset", "modelscale",
    "niterations", "maxlr", "base_config", "representation",
    "section_root",
)
PARAMETER_KEYS = {
    "THREADS", "MEMORY", "MODEL_SCALE", "ITERATIONS", "IMPUTE", "GPROBS",
    "SEED", "PARALLEL_CHROMOSOMES", "MAX_LR", "GENETIC_MAP",
    "REFERENCE_LEDGER",
}
CANONICAL_PROTOCOL = (
    PROJECT / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
    / "BENCHMARK_PROTOCOL_DLPFC_FINAL.md"
)
PROTOCOL_SHA256 = "e9a611e27d44436a402bdd561095a2a903f92a9366cb10a3a30a6846254e4e11"
CANONICAL_LAUNCHER = PROJECT / "run_slurm/run_beagle_v2_3_final_dlpfc.sh"
EXECUTION_CODE = (
    PROJECT / "scripts/2_beagle_filtering/run_beagle.py",
    PROJECT / "scripts/2_beagle_filtering/beagle_evidence_router.py",
    PROJECT / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py",
    PROJECT
    / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py",
)
PRIMARY_BASENAME = "DLPFC_{section}_SPARCAL_beagle_final_all_panel_present_matrix.pkl"
ANCHOR_BASENAME = "DLPFC_{section}_SPARCAL_beagle_final_pnonref_gp090_matrix.pkl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def verify_identity(record: object, label: str, expected: Path | None = None) -> Path:
    if not isinstance(record, Mapping) or set(record) != {
        "path", "size_bytes", "mtime_ns", "sha256"
    }:
        raise ValueError(f"{label} is not an exact full identity record")
    path = Path(str(record["path"])).resolve()
    if expected is not None and path != Path(expected).resolve():
        raise ValueError(f"{label} points to the wrong path: {path}")
    observed = identity(path)
    if any(observed[field] != record[field] for field in observed):
        raise ValueError(f"{label} identity mismatch: {path}")
    return path


def load_module(path: Path, name: str):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _strict_int(value: object, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer")
    try:
        result = int(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not an integer: {value!r}") from error
    if str(result) != str(value) or result < minimum:
        raise ValueError(f"{label} is not a canonical integer: {value!r}")
    return result


def _strict_float(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not numeric")
    try:
        result = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric: {value!r}") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _same_number(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return False
    try:
        return math.isclose(float(left), float(right), rel_tol=0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def verify_checksum_ledger(
    ledger: Path, root: Path, expected_names: Sequence[str]
) -> list[Path]:
    root = Path(root).resolve()
    observed: dict[str, Path] = {}
    for line_number, line in enumerate(ledger.read_text().splitlines(), start=1):
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not parts[1]:
            raise ValueError(f"malformed checksum row {ledger}:{line_number}")
        digest, name = parts
        if name in observed or Path(name).is_absolute():
            raise ValueError(f"duplicate/absolute checksum member: {name}")
        path = (root / name).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"checksum member escapes plan root: {name}") from error
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f"checksum mismatch: {path}")
        observed[name] = path
    if set(observed) != set(expected_names) or len(observed) != len(expected_names):
        raise ValueError(
            "plan checksum member set is not exact: "
            f"missing={sorted(set(expected_names) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected_names))}"
        )
    return [observed[name] for name in expected_names]


def _verify_identity_list(value: object, label: str) -> list[Path]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty identity list")
    paths = [verify_identity(record, f"{label}[{index}]") for index, record in enumerate(value)]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} contains duplicate paths")
    return paths


def verify_plan(
    plan_root: Path, run_root: Path, protocol_path: Path | None = None
) -> dict[str, object]:
    """Verify the frozen plan and return normalized task/provenance data."""
    plan_root = Path(plan_root).resolve()
    run_root = Path(run_root).resolve()
    names = (
        "dlpfc_tasks.tsv", "PROVENANCE.json", "OUTPUT_SHA256SUMS.txt",
        "RUN_COMPLETE", "RUN_COMPLETE.json",
    )
    missing = [plan_root / name for name in names if not (plan_root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete DLPFC plan: {missing}")
    if (plan_root / "RUN_COMPLETE").read_text() != "complete\n":
        raise ValueError("DLPFC plan has invalid RUN_COMPLETE marker")
    provenance = json.loads((plan_root / "PROVENANCE.json").read_text())
    completion = json.loads((plan_root / "RUN_COMPLETE.json").read_text())
    if (
        provenance.get("schema") != PLAN_SCHEMA
        or provenance.get("status") != "tasks_frozen"
        or completion.get("schema") != PLAN_COMPLETION_SCHEMA
        or completion.get("status") != "complete"
    ):
        raise ValueError("DLPFC plan has invalid schema/status")
    common = (
        "production_inputs_include_wes", "evaluation_uses_wes", "base_config",
        "representation", "seeds", "sections", "task_count", "array_max",
        "final_selection_inputs", "base_assessment_inputs",
        "selected_seed20260915_manifests",
    )
    for field in common:
        if completion.get(field) != provenance.get(field):
            raise ValueError(f"plan provenance/completion disagree on {field}")
    if (
        provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not False
        or provenance.get("dataset") != "DLPFC"
        or provenance.get("quality_filter") != QUALITY_FILTER
        or provenance.get("anchor_score") != "p_nonref"
        or not _same_number(provenance.get("min_anchor_probability"), THRESHOLD)
        or Path(str(provenance.get("run_root", ""))).resolve() != run_root
    ):
        raise ValueError("DLPFC plan violates its WES/dataset/run-root contract")
    representation = str(provenance.get("representation"))
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported final representation: {representation}")
    seeds = SINGLE_SEEDS if representation == "single_seed" else ENSEMBLE_SEEDS
    if provenance.get("seeds") != list(seeds):
        raise ValueError("DLPFC plan has the wrong seed set")
    if provenance.get("sections") != list(SECTIONS):
        raise ValueError("DLPFC plan has the wrong ordered section set")
    expected_count = len(SECTIONS) * len(seeds)
    if (
        provenance.get("task_count") != expected_count
        or provenance.get("array_max") != expected_count - 1
    ):
        raise ValueError("DLPFC plan has the wrong 12/36 task dimensions")
    base_config = provenance.get("base_config")
    if (
        not isinstance(base_config, str)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", base_config)
    ):
        raise ValueError("DLPFC plan has an invalid base configuration")
    parameters = provenance.get("beagle_parameters")
    if not isinstance(parameters, dict) or set(parameters) != PARAMETER_KEYS:
        raise ValueError("DLPFC plan lacks the exact selected Beagle parameters")
    for field in ("MODEL_SCALE", "ITERATIONS", "MAX_LR", "SEED"):
        if field not in parameters:
            raise ValueError(f"DLPFC plan parameters lack {field}")
    if parameters["SEED"] != 20260915:
        raise ValueError("selected source parameter seed must be 20260915")
    for field in ("THREADS", "ITERATIONS", "SEED", "PARALLEL_CHROMOSOMES"):
        if isinstance(parameters[field], bool) or not isinstance(parameters[field], int):
            raise ValueError(f"plan parameter {field} is not an exact integer")
    if parameters["ITERATIONS"] < 0:
        raise ValueError("plan ITERATIONS must be nonnegative")
    for field in ("MODEL_SCALE", "MAX_LR"):
        if (
            isinstance(parameters[field], bool)
            or not isinstance(parameters[field], (int, float))
            or not math.isfinite(float(parameters[field]))
            or float(parameters[field]) <= 0
        ):
            raise ValueError(f"plan parameter {field} is not finite and positive")
    reference_text = provenance.get("reference_ledger_root")
    if not isinstance(reference_text, str) or not Path(reference_text).is_absolute():
        raise ValueError("DLPFC plan has an invalid reference ledger root")
    reference_root = Path(reference_text).resolve()
    if Path(str(parameters["REFERENCE_LEDGER"])).resolve() != reference_root:
        raise ValueError("plan parameter and provenance reference roots disagree")
    if (
        parameters["IMPUTE"] is not False
        or parameters["GPROBS"] is not True
        or parameters["GENETIC_MAP"] is not None
        or parameters["THREADS"] != 30
        or parameters["PARALLEL_CHROMOSOMES"] != 1
        or parameters["MEMORY"] != "10g"
    ):
        raise ValueError("DLPFC plan cannot replay the selected v2.3 settings")

    bound_inputs = []
    for field in ("final_selection_inputs", "base_assessment_inputs"):
        bound_inputs.extend(_verify_identity_list(provenance.get(field), field))
    selected_manifests = provenance.get("selected_seed20260915_manifests")
    if not isinstance(selected_manifests, list) or len(selected_manifests) != 2:
        raise ValueError("plan must bind exactly the P4/P6 selected-seed manifests")
    if [row.get("sample") for row in selected_manifests if isinstance(row, dict)] != ["P4", "P6"]:
        raise ValueError("selected-seed manifests must be ordered P4 then P6")
    for index, row in enumerate(selected_manifests):
        if not isinstance(row, dict) or set(row) != {"sample", "manifest"}:
            raise ValueError("selected-seed manifest record fields are not exact")
        bound_inputs.append(verify_identity(row["manifest"], f"selected manifest {index}"))
    expected_planner = PROJECT / "scripts/postanalyze/plan_beagle_v2_3_final_dlpfc.py"
    expected_integrity = PROJECT / "scripts/postanalyze/beagle_selection_integrity.py"
    bound_inputs.append(verify_identity(
        provenance.get("planner"), "plan planner", expected_planner
    ))
    bound_inputs.append(verify_identity(
        provenance.get("integrity_module"), "plan integrity module", expected_integrity
    ))
    plan_protocol = verify_identity(
        provenance.get("protocol"), "plan protocol", CANONICAL_PROTOCOL
    )
    if sha256(plan_protocol) != PROTOCOL_SHA256:
        raise ValueError("canonical final DLPFC protocol checksum drifted")
    if protocol_path is not None and plan_protocol != Path(protocol_path).resolve():
        raise ValueError("materializer protocol differs from frozen plan protocol")
    bound_inputs.append(plan_protocol)
    bound_inputs.append(verify_identity(
        provenance.get("launcher"), "plan launcher", CANONICAL_LAUNCHER
    ))
    execution_records = provenance.get("execution_code")
    if not isinstance(execution_records, list) or len(execution_records) != len(EXECUTION_CODE):
        raise ValueError("plan has an invalid execution-code identity list")
    for index, (record, expected) in enumerate(zip(execution_records, EXECUTION_CODE)):
        bound_inputs.append(verify_identity(
            record, f"plan execution code {index}", expected
        ))

    final_root = Path(str(provenance.get("final_selection_root", ""))).resolve()
    base_root = Path(str(provenance.get("base_assessment_root", ""))).resolve()
    final_names = {
        "ensemble_gate_results.tsv", "FINAL_BEAGLE_REPRESENTATION.json",
        "PROVENANCE.json", "CODE_AND_INPUT_SHA256SUMS.txt",
        "OUTPUT_SHA256SUMS.txt", "RUN_COMPLETE", "RUN_COMPLETE.json",
    }
    base_names = {
        "replicated_configuration_metrics_long.tsv",
        "replicated_configuration_summary.tsv",
        "replicated_pairwise_stability.tsv", "replication_gate_results.tsv",
        "FINAL_BASE_CONFIGURATION.json", "PROVENANCE.json",
        "CODE_AND_INPUT_SHA256SUMS.txt", "OUTPUT_SHA256SUMS.txt",
        "RUN_COMPLETE", "RUN_COMPLETE.json",
    }
    final_bound = _verify_identity_list(
        provenance.get("final_selection_inputs"), "final_selection_inputs"
    )
    base_bound = _verify_identity_list(
        provenance.get("base_assessment_inputs"), "base_assessment_inputs"
    )
    if (
        {path.name for path in final_bound} != final_names
        or any(path.parent != final_root for path in final_bound)
        or {path.name for path in base_bound} != base_names
        or any(path.parent != base_root for path in base_bound)
    ):
        raise ValueError("plan binds an incorrect final/base assessment file set")

    ledger = plan_root / "OUTPUT_SHA256SUMS.txt"
    verify_identity(completion.get("output_sha256sums"), "plan output ledger", ledger)
    verify_checksum_ledger(
        ledger, plan_root, ("dlpfc_tasks.tsv", "PROVENANCE.json")
    )
    task_path = plan_root / "dlpfc_tasks.tsv"
    with task_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != PLAN_HEADER:
            raise ValueError("DLPFC task table header is not exact")
        rows = list(reader)
    if len(rows) != expected_count or any(None in row for row in rows):
        raise ValueError("DLPFC task row count/width is not exact")

    seen = set()
    normalized = []
    for index, row in enumerate(rows):
        task_id = _strict_int(row["task_id"], f"task {index} id")
        seed = _strict_int(row["seed"], f"task {index} seed")
        iterations = _strict_int(row["niterations"], f"task {index} niterations")
        modelscale = _strict_float(row["modelscale"], f"task {index} modelscale")
        maxlr = _strict_float(row["maxlr"], f"task {index} maxlr")
        section = row["section"]
        if task_id != index:
            raise ValueError("DLPFC task IDs must be contiguous and ordered from zero")
        if (
            section not in SECTIONS or seed not in seeds or row["dataset"] != "DLPFC"
            or row["base_config"] != base_config
            or row["representation"] != representation
            or not _same_number(modelscale, parameters["MODEL_SCALE"])
            or iterations != parameters["ITERATIONS"]
            or not _same_number(maxlr, parameters["MAX_LR"])
        ):
            raise ValueError(f"task {index} disagrees with frozen plan parameters")
        expected_root = run_root / f"{base_config}_seed{seed}" / section
        if Path(row["section_root"]).resolve() != expected_root.resolve():
            raise ValueError(f"task {index} section_root is not canonical")
        key = (section, seed)
        if key in seen:
            raise ValueError(f"duplicate DLPFC task: {key}")
        seen.add(key)
        normalized.append({
            **row, "task_id": task_id, "seed": seed,
            "niterations": iterations, "modelscale": modelscale,
            "maxlr": maxlr, "section_root": expected_root.resolve(),
        })
    expected_grid = {(section, seed) for section in SECTIONS for seed in seeds}
    if seen != expected_grid:
        raise ValueError("DLPFC task table is not the exact section-by-seed grid")
    expected_order = [(section, seed) for seed in seeds for section in SECTIONS]
    if [(row["section"], row["seed"]) for row in normalized] != expected_order:
        raise ValueError("DLPFC task table is not in the frozen seed-major order")
    return {
        "root": plan_root,
        "provenance": provenance,
        "completion": completion,
        "tasks": normalized,
        "base_config": base_config,
        "representation": representation,
        "seeds": seeds,
        "parameters": parameters,
        "execution_code": execution_records,
        "bound_inputs": bound_inputs,
        "bundle_paths": [plan_root / name for name in names],
    }


def _content_without_location(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _content_without_location(item)
            for key, item in sorted(value.items())
            if key not in {"path", "mtime_ns"}
        }
    if isinstance(value, list):
        return [_content_without_location(item) for item in value]
    return value


def require_execution_code_match(
    observed_records: object, planned_records: object
) -> None:
    """Require run-local snapshots to equal the four planned source files."""
    if (
        not isinstance(observed_records, list)
        or not isinstance(planned_records, list)
        or len(observed_records) != len(planned_records)
        or len(observed_records) != len(EXECUTION_CODE)
    ):
        raise ValueError("run/plan execution-code lists have the wrong size")

    def by_name(records: Sequence[object], label: str) -> dict[str, Mapping[str, object]]:
        result = {}
        for index, record in enumerate(records):
            path = verify_identity(record, f"{label} execution code {index}")
            if path.name in result:
                raise ValueError(f"duplicate {label} execution-code basename: {path.name}")
            result[path.name] = record  # type: ignore[assignment]
        return result

    observed = by_name(observed_records, "run")
    planned = by_name(planned_records, "plan")
    if set(observed) != set(planned):
        raise ValueError("run execution-code basenames differ from frozen plan")
    for name in planned:
        if any(
            observed[name].get(field) != planned[name].get(field)
            for field in ("size_bytes", "sha256")
        ):
            raise ValueError(f"run execution-code content drift: {name}")


def validate_runs(
    plan: Mapping[str, object], contracts, project: Path, run_root: Path
) -> tuple[dict[str, dict[int, dict[str, object]]], list[Path]]:
    """Validate every completed source run and its exact task parameters."""
    by_section: dict[str, dict[int, dict[str, object]]] = {
        section: {} for section in SECTIONS
    }
    bound_paths: list[Path] = []
    invariant_code = invariant_executables = invariant_reference = None
    plan_parameters = dict(plan["parameters"])  # type: ignore[arg-type]
    reference_root = Path(str(plan["provenance"]["reference_ledger_root"])).resolve()  # type: ignore[index]
    for task in plan["tasks"]:  # type: ignore[union-attr]
        section = str(task["section"])
        seed = int(task["seed"])
        root = Path(task["section_root"]).resolve()
        canonical = (
            Path(run_root).resolve()
            / f"{plan['base_config']}_seed{seed}"
            / section
        )
        if root != canonical:
            raise ValueError(f"source run escaped canonical grid: {root}")
        validation = contracts.validate_v2_3_run(root)
        manifest_path = root / "V2_3_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        expected_params = {**plan_parameters, "SEED": seed}
        if (
            manifest.get("dataset") != "DLPFC"
            or str(manifest.get("section_id")) != section
            or manifest.get("quality_filter") != QUALITY_FILTER
            or manifest.get("production_inputs_include_wes") is not False
            or manifest.get("evaluation_uses_wes") is not False
            or manifest.get("anchor_score") != "p_nonref"
            or manifest.get("beagle_parameters") != expected_params
            or not _same_number(manifest.get("min_anchor_probability"), THRESHOLD)
        ):
            raise ValueError(f"DLPFC run manifest disagrees with plan: {root}")
        if Path(str(manifest["reference_ledger"]["root"])).resolve() != reference_root:
            raise ValueError(f"DLPFC run uses the wrong reference ledger: {root}")
        expected_raw = (
            Path(project).resolve() / "data" / "dlpfc" / section
            / "output_VCFs" / "mpileup_multi_bam" / QUALITY_FILTER
            / "merged_sorted_gt.vcf.gz"
        ).resolve()
        if Path(str(manifest["raw_vcf"]["path"])).resolve() != expected_raw:
            raise ValueError(f"DLPFC run uses a noncanonical raw candidate VCF: {root}")
        expected_raw_index = Path(str(expected_raw) + ".tbi")
        if Path(str(manifest["raw_vcf_index"]["path"])).resolve() != expected_raw_index:
            raise ValueError(f"DLPFC run uses a noncanonical raw VCF index: {root}")
        require_execution_code_match(manifest.get("code"), plan["execution_code"])
        audit = (
            root / "output_VCFs" / "beagle" / QUALITY_FILTER
            / "beagle_evidence_audit.tsv.gz"
        ).resolve()
        if not audit.is_file():
            raise FileNotFoundError(audit)
        reference_content = {
            "reference_ledger": manifest.get("reference_ledger"),
            "reference_panels": [
                {
                    key: value for key, value in panel.items()
                    if key != "raw_panel_exact_alleles"
                }
                for panel in manifest.get("reference_panels", [])
            ],
        }
        comparison_values = (
            _content_without_location(manifest.get("code")),
            _content_without_location(manifest.get("executables")),
            _content_without_location(reference_content),
        )
        if invariant_code is None:
            invariant_code, invariant_executables, invariant_reference = comparison_values
        elif comparison_values != (invariant_code, invariant_executables, invariant_reference):
            raise ValueError("DLPFC run code/executable/reference identities are not invariant")
        record = {
            "seed": seed,
            "root": root,
            "audit": audit,
            "manifest": manifest,
            "validation": validation,
            "bundle_identities": {
                "manifest": identity(manifest_path),
                "completion": identity(root / "RUN_COMPLETE.json"),
                "output_sha256sums": identity(root / "OUTPUT_SHA256SUMS.txt"),
                "audit": identity(audit),
                "raw_vcf": identity(expected_raw),
                "raw_vcf_index": identity(expected_raw_index),
            },
        }
        by_section[section][seed] = record
        bound_paths.extend(
            [
                manifest_path, root / "RUN_COMPLETE.json",
                root / "OUTPUT_SHA256SUMS.txt", audit,
                expected_raw, expected_raw_index,
            ]
        )
    expected_seeds = set(plan["seeds"])  # type: ignore[arg-type]
    if any(set(records) != expected_seeds for records in by_section.values()):
        raise ValueError("validated DLPFC run grid is not exact")
    return by_section, bound_paths


def _gzip_text_writer(path: Path):
    raw = path.open("xb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def stream_selected_ensemble(
    source_audits: Mapping[int, Path], output: Path, representation: str, ensemble
) -> dict[str, object]:
    """Stream exactly the selected mean/median representation to one audit."""
    if representation not in {"ensemble_mean", "ensemble_median"}:
        raise ValueError("only the selected mean/median ensemble can be streamed")
    if set(source_audits) != set(ENSEMBLE_SEEDS):
        raise ValueError("ensemble source audit grid must contain exactly three seeds")
    method = "mean" if representation == "ensemble_mean" else "median"
    counts = {route: 0 for route in ALL_ROUTES}
    total = 0
    output.parent.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        readers = {}
        for seed in ENSEMBLE_SEEDS:
            handle = stack.enter_context(gzip.open(source_audits[seed], "rt"))
            reader = csv.DictReader(handle, delimiter="\t")
            if tuple(reader.fieldnames or ()) != tuple(ensemble.AUDIT_COLUMNS):
                raise ValueError(f"source audit {seed} has a non-exact schema")
            readers[seed] = reader
        raw, compressed, text_handle = _gzip_text_writer(output)
        stack.callback(raw.close)
        stack.callback(compressed.close)
        writer_handle = stack.enter_context(text_handle)
        writer = csv.writer(writer_handle, delimiter="\t", lineterminator="\n")
        writer.writerow(ensemble.AUDIT_COLUMNS)
        sentinel = object()
        streams: Iterable[tuple[object, ...]] = zip_longest(
            *(readers[seed] for seed in ENSEMBLE_SEEDS), fillvalue=sentinel
        )
        for row_number, aligned in enumerate(streams, start=2):
            if any(row is sentinel for row in aligned):
                raise ValueError(f"ensemble audits differ in length near row {row_number}")
            rows_by_seed = {
                seed: aligned[index]
                for index, seed in enumerate(ENSEMBLE_SEEDS)
            }
            fields, route = ensemble.summarize_row(
                rows_by_seed, method, ENSEMBLE_SEEDS
            )
            writer.writerow(fields)
            counts[route] += 1
            total += 1
    if total <= 0 or sum(counts.values()) != total:
        raise ValueError("selected ensemble audit is empty or does not conserve rows")
    return {
        "representation": representation,
        "aggregation": method,
        "seeds": list(ENSEMBLE_SEEDS),
        "row_count": total,
        "route_counts": counts,
        "audit": identity(output),
    }


def locus(chrom: object, pos: object) -> str:
    chrom_text = str(chrom)
    if chrom_text.startswith("chr"):
        chrom_text = chrom_text[3:]
    position = int(str(pos))
    if not chrom_text or position <= 0:
        raise ValueError(f"invalid audit locus: {chrom!r}:{pos!r}")
    return f"{chrom_text}_{position}"


def set_digest(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(set(values)):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def dataframe_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(frame.index, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(frame.columns, index=True).values.tobytes())
    digest.update("\0".join(map(str, frame.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def scan_audit(audit_path: Path, audit_columns: Sequence[str]) -> dict[str, object]:
    exact_seen = set()
    per_locus: dict[str, dict[str, int]] = {}
    counts = {route: 0 for route in ALL_ROUTES}
    schemas = set()
    with gzip.open(audit_path, "rt") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != tuple(audit_columns):
            raise ValueError("selected audit has a non-exact schema")
        for line_number, row in enumerate(reader, start=2):
            schemas.add(row["schema"])
            route = row["route"]
            if route not in ALL_ROUTES:
                raise ValueError(f"unknown route at audit row {line_number}: {route}")
            try:
                position = int(row["pos"])
            except ValueError as error:
                raise ValueError(f"invalid position at audit row {line_number}") from error
            if (
                position <= 0
                or row["chrom"] in {"", "."}
                or row["ref"] in {"", "."}
                or row["alt"] in {"", "."}
            ):
                raise ValueError(f"invalid allele fields at audit row {line_number}")
            key = (row["chrom"], position, row["ref"], row["alt"])
            if key in exact_seen:
                raise ValueError(f"duplicate exact allele at audit row {line_number}: {key}")
            exact_seen.add(key)
            loc = locus(row["chrom"], row["pos"])
            summary = per_locus.setdefault(loc, {"rows": 0, "panel": 0, "anchor": 0})
            summary["rows"] += 1
            panel = route in PANEL_ROUTES
            summary["panel"] += int(panel)
            score_text = row["beagle_p_nonref"]
            try:
                score = float(score_text)
            except ValueError:
                score = math.nan
            anchor = math.isfinite(score) and score >= THRESHOLD - 1e-12
            if route == "beagle_nonref_highconf" and not anchor:
                raise ValueError(f"nonref route lacks P(nonref)>=0.90 at {key}")
            if anchor and route != "beagle_nonref_highconf":
                raise ValueError(f"P(nonref)>=0.90 row is not in nonref route at {key}")
            if row["hard_anchor"] != ("1" if anchor else "0"):
                raise ValueError(f"hard_anchor field disagrees with posterior at {key}")
            summary["anchor"] += int(anchor)
            counts[route] += 1
    if not exact_seen:
        raise ValueError("selected audit is empty")
    if schemas not in (
        {"sparcal_beagle_evidence_v2_2"},
        {"sparcal_beagle_evidence_v2_3_ensemble"},
    ):
        raise ValueError(f"selected audit has an invalid row schema set: {sorted(schemas)}")
    panel_loci = {key for key, row in per_locus.items() if row["panel"] > 0}
    anchor_loci = {key for key, row in per_locus.items() if row["anchor"] > 0}
    multi = {key for key, row in per_locus.items() if row["rows"] > 1}
    panel_mixed = {
        key for key, row in per_locus.items()
        if row["rows"] > 1 and 0 < row["panel"] < row["rows"]
    }
    anchor_mixed = {
        key for key, row in per_locus.items()
        if row["rows"] > 1 and 0 < row["anchor"] < row["rows"]
    }
    return {
        "exact_allele_count": len(exact_seen),
        "exact_allele_sha256": hashlib.sha256(
            "".join(
                f"{chrom}\t{pos}\t{ref}\t{alt}\n"
                for chrom, pos, ref, alt in sorted(exact_seen)
            ).encode("utf-8")
        ).hexdigest(),
        "route_counts": counts,
        "panel_loci": panel_loci,
        "anchor_loci": anchor_loci,
        "multiallelic_loci": multi,
        "panel_mixed_loci": panel_mixed,
        "anchor_mixed_loci": anchor_mixed,
    }


def _matrix_summary(frame: pd.DataFrame, path: Path) -> dict[str, object]:
    return {
        "identity": identity(path),
        "shape": list(frame.shape),
        "nonzero": int((frame.to_numpy(copy=False) != 0).sum()),
        "dataframe_sha256": dataframe_digest(frame),
        "feature_set_sha256": set_digest(map(str, frame.columns)),
    }


def build_matrix_views(
    source_matrix: Path,
    audit_path: Path,
    primary_output: Path,
    anchor_output: Path,
    differences_output: Path,
    audit_columns: Sequence[str],
) -> dict[str, object]:
    """Build strict position-only projections with no unrepresented positions."""
    matrix = pd.read_pickle(source_matrix)
    if not matrix.columns.is_unique:
        raise ValueError("legacy source matrix has duplicate feature columns")
    if any(not isinstance(value, str) for value in matrix.columns):
        raise ValueError("legacy source matrix columns must be position-only strings")
    for value in matrix.columns:
        fields = value.rsplit("_", 1)
        if (
            len(fields) != 2
            or not fields[0]
            or fields[0].startswith("chr")
            or not fields[1].isdigit()
            or int(fields[1]) <= 0
        ):
            raise ValueError(f"invalid legacy position-only matrix feature: {value!r}")
    source_loci = set(matrix.columns)
    audit = scan_audit(audit_path, audit_columns)
    panel_loci = audit["panel_loci"]
    anchor_loci = audit["anchor_loci"]
    candidate_only = set(panel_loci) - source_loci
    if candidate_only:
        examples = ", ".join(sorted(candidate_only)[:10])
        raise ValueError(
            f"{len(candidate_only)} candidate panel-present positions cannot be "
            f"represented in the frozen source matrix (examples: {examples})"
        )
    primary_columns = [column for column in matrix.columns if column in panel_loci]
    anchor_columns = [column for column in matrix.columns if column in anchor_loci]
    if not primary_columns:
        raise ValueError("primary all_panel_present projection is empty")
    primary = matrix.loc[:, primary_columns].copy()
    anchor = matrix.loc[:, anchor_columns].copy()
    for label, frame, columns in (
        ("primary", primary, primary_columns), ("anchor", anchor, anchor_columns)
    ):
        if not frame.index.equals(matrix.index) or list(frame.columns) != columns:
            raise AssertionError(f"{label} projection changed row/column order")
        pd.testing.assert_frame_equal(
            frame, matrix.loc[:, columns], check_dtype=True, check_exact=True,
            check_index_type=True, check_column_type=True, check_names=True,
        )
    primary_output.parent.mkdir(parents=True, exist_ok=False)
    primary.to_pickle(primary_output)
    anchor.to_pickle(anchor_output)
    reloaded_primary = pd.read_pickle(primary_output)
    reloaded_anchor = pd.read_pickle(anchor_output)
    pd.testing.assert_frame_equal(reloaded_primary, primary, check_dtype=True, check_exact=True)
    pd.testing.assert_frame_equal(reloaded_anchor, anchor, check_dtype=True, check_exact=True)

    differences = {
        "all_panel_present": {
            "candidate_only": sorted(set(panel_loci) - source_loci),
            "legacy_only": sorted(source_loci - set(panel_loci)),
        },
        "p_nonref_gp090": {
            "candidate_only": sorted(set(anchor_loci) - source_loci),
            "legacy_only": sorted(source_loci - set(anchor_loci)),
        },
    }
    raw, compressed, text_handle = _gzip_text_writer(differences_output)
    try:
        with text_handle as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("view", "side", "locus"))
            for view in ("all_panel_present", "p_nonref_gp090"):
                for side in ("candidate_only", "legacy_only"):
                    for value in differences[view][side]:
                        writer.writerow((view, side, value))
    finally:
        compressed.close()
        raw.close()

    source_summary = {
        "identity": identity(source_matrix),
        "shape": list(matrix.shape),
        "nonzero": int((matrix.to_numpy(copy=False) != 0).sum()),
        "dataframe_sha256": dataframe_digest(matrix),
        "feature_set_sha256": set_digest(source_loci),
    }
    primary_summary = _matrix_summary(reloaded_primary, primary_output)
    anchor_summary = _matrix_summary(reloaded_anchor, anchor_output)
    exact_equal = False
    try:
        pd.testing.assert_frame_equal(
            reloaded_primary, matrix, check_dtype=True, check_exact=True,
            check_index_type=True, check_column_type=True, check_names=True,
        )
        exact_equal = True
    except AssertionError:
        pass
    del matrix, primary, anchor, reloaded_primary, reloaded_anchor
    return {
        "source": source_summary,
        "audit": {
            key: value for key, value in audit.items()
            if not isinstance(value, set)
        },
        "multiallelic_policy": "any",
        "multiallelic_locus_count": len(audit["multiallelic_loci"]),
        "panel_mixed_multiallelic_locus_count": len(audit["panel_mixed_loci"]),
        "anchor_mixed_multiallelic_locus_count": len(audit["anchor_mixed_loci"]),
        "primary": primary_summary,
        "anchor": anchor_summary,
        "anchor_status": (
            "nonempty" if anchor_summary["shape"][1] > 0
            else "empty_no_pnonref_gp090_loci"
        ),
        "primary_exact_dataframe_equal_to_legacy": exact_equal,
        "primary_candidate_only_feature_count": len(differences["all_panel_present"]["candidate_only"]),
        "primary_legacy_only_feature_count": len(differences["all_panel_present"]["legacy_only"]),
        "anchor_candidate_only_feature_count": len(differences["p_nonref_gp090"]["candidate_only"]),
        "anchor_legacy_only_feature_count": len(differences["p_nonref_gp090"]["legacy_only"]),
        "primary_candidate_only_feature_sha256": set_digest(differences["all_panel_present"]["candidate_only"]),
        "primary_legacy_only_feature_sha256": set_digest(differences["all_panel_present"]["legacy_only"]),
        "anchor_candidate_only_feature_sha256": set_digest(differences["p_nonref_gp090"]["candidate_only"]),
        "anchor_legacy_only_feature_sha256": set_digest(differences["p_nonref_gp090"]["legacy_only"]),
        "feature_differences": identity(differences_output),
    }


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_ledger(paths: Iterable[Path], ledger: Path, relative_root: Path | None = None) -> None:
    resolved = sorted(set(Path(path).resolve() for path in paths), key=str)
    with ledger.open("x", encoding="utf-8") as handle:
        for path in resolved:
            if relative_root is not None:
                try:
                    label = path.relative_to(Path(relative_root).resolve()).as_posix()
                except ValueError:
                    label = str(path)
            else:
                label = str(path)
            handle.write(f"{sha256(path)}  {label}\n")


def write_completion(
    root: Path, schema: str, manifest: Path, code_ledger: Path, output_ledger: Path
) -> Path:
    path = root / "RUN_COMPLETE.json"
    write_json(path, {
        "schema": schema,
        "status": "complete",
        "created_utc": utc_now(),
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "manifest": identity(manifest),
        "code_and_input_sha256sums": identity(code_ledger),
        "output_sha256sums": identity(output_ledger),
    })
    (root / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    return path


def materialize(
    *,
    plan_root: Path,
    run_root: Path,
    output_root: Path,
    project: Path,
    contracts_path: Path,
    ensemble_path: Path,
    protocol_path: Path,
) -> Mapping[str, object]:
    project = Path(project).resolve()
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    for path in (contracts_path, ensemble_path, protocol_path):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    plan = verify_plan(plan_root, run_root, protocol_path)
    contracts = load_module(contracts_path, "_final_dlpfc_contracts")
    runs, run_bundle_paths = validate_runs(plan, contracts, project, run_root)

    # No output exists until the complete source grid and all bound checksums pass.
    output_root.mkdir(parents=True, exist_ok=False)
    snapshot = output_root / "code_snapshot"
    snapshot.mkdir()
    snapshots = []
    for source in (Path(__file__), contracts_path, ensemble_path, protocol_path):
        destination = snapshot / Path(source).name
        shutil.copy2(source, destination)
        snapshots.append(destination)
    ensemble = load_module(snapshot / Path(ensemble_path).name, "_final_dlpfc_ensemble")

    section_records = {}
    comparison_rows = []
    all_external_inputs = list(plan["bundle_paths"]) + list(plan["bound_inputs"]) + run_bundle_paths
    for section in SECTIONS:
        section_root = output_root / section
        section_root.mkdir()
        records = runs[section]
        source_audits = {seed: Path(record["audit"]) for seed, record in records.items()}
        representation = str(plan["representation"])
        if representation == "single_seed":
            selected_audit = source_audits[20260915]
            audit_origin = {
                "kind": "canonical_single_seed_v2_3_audit",
                "seed": 20260915,
                "audit": identity(selected_audit),
            }
            local_audit = None
        else:
            selected_audit = section_root / "selected_audit" / "beagle_evidence_audit.tsv.gz"
            audit_origin = {
                "kind": "streamed_selected_three_seed_posterior_ensemble",
                **stream_selected_ensemble(
                    source_audits, selected_audit, representation, ensemble
                ),
            }
            local_audit = selected_audit

        matrix_source = (
            project / "data" / "dlpfc" / section / "matrix"
            / f"DLPFC_{section}_SPARCAL_1000G_matrix.pkl"
        ).resolve()
        if not matrix_source.is_file():
            raise FileNotFoundError(matrix_source)
        matrix_dir = section_root / "matrix"
        primary = matrix_dir / PRIMARY_BASENAME.format(section=section)
        anchor = matrix_dir / ANCHOR_BASENAME.format(section=section)
        differences = section_root / "FEATURE_SET_DIFFERENCES.tsv.gz"
        comparison = build_matrix_views(
            matrix_source, selected_audit, primary, anchor, differences,
            ensemble.AUDIT_COLUMNS,
        )
        source_run_records = []
        for seed in plan["seeds"]:
            record = records[int(seed)]
            source_run_records.append({
                "seed": int(seed),
                "root": str(record["root"]),
                "contract_validation": record["validation"],
                **record["bundle_identities"],
            })
        manifest_path = section_root / "MANIFEST.json"
        section_manifest = {
            "schema": SECTION_SCHEMA,
            "created_utc": utc_now(),
            "status": "outputs_ready_for_checksum",
            "production_inputs_include_wes": False,
            "evaluation_uses_wes": False,
            "section": section,
            "base_config": plan["base_config"],
            "representation": representation,
            "seeds": list(plan["seeds"]),
            "scientific_role": "all_panel_present feature-support compatibility; not germline",
            "primary_interface": "all_panel_present",
            "secondary_diagnostic": "P(nonref)>=0.90",
            "secondary_diagnostic_can_decide_replacement": False,
            "position_projection": "retain if any exact allele at a position is selected",
            "multiallelic_policy": "any",
            "selected_audit": identity(selected_audit),
            "selected_audit_origin": audit_origin,
            "source_runs": source_run_records,
            "matrix_comparison": comparison,
            "anchor_diagnostic_status": comparison["anchor_status"],
            "primary_matrix": identity(primary),
            "anchor_matrix": identity(anchor),
        }
        write_json(manifest_path, section_manifest)
        section_inputs = [
            Path(__file__).resolve(), Path(contracts_path).resolve(),
            Path(ensemble_path).resolve(), Path(protocol_path).resolve(),
            matrix_source, *plan["bundle_paths"], *plan["bound_inputs"],
        ]
        for record in records.values():
            section_inputs.extend([
                Path(record["root"]) / "V2_3_MANIFEST.json",
                Path(record["root"]) / "RUN_COMPLETE.json",
                Path(record["root"]) / "OUTPUT_SHA256SUMS.txt",
                Path(record["audit"]),
                Path(record["manifest"]["raw_vcf"]["path"]),
                Path(record["manifest"]["raw_vcf_index"]["path"]),
            ])
        code_ledger = section_root / "CODE_AND_INPUT_SHA256SUMS.txt"
        write_ledger(section_inputs, code_ledger)
        outputs = [manifest_path, primary, anchor, differences]
        if local_audit is not None:
            outputs.append(local_audit)
        output_ledger = section_root / "OUTPUT_SHA256SUMS.txt"
        write_ledger(outputs, output_ledger, section_root)
        completion_path = write_completion(
            section_root, SECTION_COMPLETION_SCHEMA, manifest_path,
            code_ledger, output_ledger,
        )
        section_records[section] = {
            "section_root": str(section_root),
            "manifest": identity(manifest_path),
            "completion": identity(completion_path),
            "primary_matrix": identity(primary),
            "anchor_matrix": identity(anchor),
            "primary_exact_dataframe_equal_to_legacy": comparison[
                "primary_exact_dataframe_equal_to_legacy"
            ],
            "source_feature_count": comparison["source"]["shape"][1],
            "primary_feature_count": comparison["primary"]["shape"][1],
            "anchor_feature_count": comparison["anchor"]["shape"][1],
            "anchor_diagnostic_status": comparison["anchor_status"],
            "primary_legacy_only_feature_count": comparison[
                "primary_legacy_only_feature_count"
            ],
        }
        comparison_rows.append({
            "section": section,
            "source_features": comparison["source"]["shape"][1],
            "all_panel_present_features": comparison["primary"]["shape"][1],
            "pnonref_gp090_features": comparison["anchor"]["shape"][1],
            "primary_candidate_only": comparison["primary_candidate_only_feature_count"],
            "primary_legacy_only": comparison["primary_legacy_only_feature_count"],
            "anchor_candidate_only": comparison["anchor_candidate_only_feature_count"],
            "anchor_legacy_only": comparison["anchor_legacy_only_feature_count"],
            "primary_exact_dataframe_equal_to_legacy": str(
                comparison["primary_exact_dataframe_equal_to_legacy"]
            ).lower(),
            "source_dataframe_sha256": comparison["source"]["dataframe_sha256"],
            "primary_dataframe_sha256": comparison["primary"]["dataframe_sha256"],
            "anchor_dataframe_sha256": comparison["anchor"]["dataframe_sha256"],
        })

    comparison_path = output_root / "MATRIX_COMPARISON.tsv"
    with comparison_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparison_rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison_rows)
    root_manifest_path = output_root / "MANIFEST.json"
    root_manifest = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "status": "outputs_ready_for_checksum",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": False,
        "scientific_role": "all_panel_present feature-support compatibility; not germline",
        "base_config": plan["base_config"],
        "representation": plan["representation"],
        "seeds": list(plan["seeds"]),
        "sections": list(SECTIONS),
        "source_run_count": len(plan["tasks"]),
        "exact_expected_source_run_count": 12 if plan["representation"] == "single_seed" else 36,
        "all_candidate_panel_positions_representable": True,
        "all_primary_matrices_exactly_equal_to_legacy": all(
            row["primary_exact_dataframe_equal_to_legacy"]
            for row in section_records.values()
        ),
        "primary_matrix_basename": PRIMARY_BASENAME,
        "anchor_matrix_basename": ANCHOR_BASENAME,
        "plan": identity(Path(plan_root) / "PROVENANCE.json"),
        "protocol": identity(protocol_path),
        "code_snapshot": [identity(path) for path in snapshots],
        "matrix_comparison": identity(comparison_path),
        "section_records": section_records,
    }
    write_json(root_manifest_path, root_manifest)
    root_code_ledger = output_root / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_ledger([
        Path(__file__).resolve(), Path(contracts_path).resolve(),
        Path(ensemble_path).resolve(), Path(protocol_path).resolve(),
        *all_external_inputs,
        *(project / "data" / "dlpfc" / section / "matrix"
          / f"DLPFC_{section}_SPARCAL_1000G_matrix.pkl" for section in SECTIONS),
    ], root_code_ledger)
    transitive_outputs = [root_manifest_path, comparison_path, *snapshots]
    for section in SECTIONS:
        section_root = output_root / section
        transitive_outputs.extend([
            section_root / "MANIFEST.json",
            section_root / "CODE_AND_INPUT_SHA256SUMS.txt",
            section_root / "OUTPUT_SHA256SUMS.txt",
            section_root / "RUN_COMPLETE",
            section_root / "RUN_COMPLETE.json",
        ])
    root_output_ledger = output_root / "OUTPUT_SHA256SUMS.txt"
    write_ledger(transitive_outputs, root_output_ledger, output_root)
    write_completion(
        output_root, COMPLETION_SCHEMA, root_manifest_path,
        root_code_ledger, root_output_ledger,
    )
    return root_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--contracts", type=Path, default=EXECUTION_CODE[2])
    parser.add_argument(
        "--ensemble-module", type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_posterior_ensemble.py",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=CANONICAL_PROTOCOL,
    )
    args = parser.parse_args()
    result = materialize(
        plan_root=args.plan_root, run_root=args.run_root,
        output_root=args.output_root, project=args.project,
        contracts_path=args.contracts.resolve(),
        ensemble_path=args.ensemble_module.resolve(),
        protocol_path=args.protocol.resolve(),
    )
    print(json.dumps({
        "schema": result["schema"],
        "base_config": result["base_config"],
        "representation": result["representation"],
        "source_run_count": result["source_run_count"],
        "all_candidate_panel_positions_representable": result[
            "all_candidate_panel_positions_representable"
        ],
        "all_primary_matrices_exactly_equal_to_legacy": result[
            "all_primary_matrices_exactly_equal_to_legacy"
        ],
        "output_root": str(args.output_root.resolve()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
