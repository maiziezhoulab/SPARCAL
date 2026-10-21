#!/usr/bin/env python3
"""Freeze the seed-replication task list from completed v2.3 screen gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping


PROJECT = Path(__file__).resolve().parents[2]
POSTANALYZE = Path(__file__).resolve().parent
if str(POSTANALYZE) not in sys.path:
    sys.path.insert(0, str(POSTANALYZE))
import beagle_selection_integrity as INTEGRITY  # noqa: E402
import audit_beagle_v2_3_pre_replication as PREAUDIT  # noqa: E402


CONFIGURATIONS = {
    "ms2_niter0": {"modelscale": 2.0, "niterations": 0, "maxlr": 5000.0},
    "ms2_niter5": {"modelscale": 2.0, "niterations": 5, "maxlr": 5000.0},
    "ms0p8_niter0": {"modelscale": 0.8, "niterations": 0, "maxlr": 5000.0},
    "ms0p8_niter5": {"modelscale": 0.8, "niterations": 5, "maxlr": 5000.0},
    "ms4_niter0": {"modelscale": 4.0, "niterations": 0, "maxlr": 5000.0},
    "ms2_niter0_maxlr50000": {
        "modelscale": 2.0,
        "niterations": 0,
        "maxlr": 50000.0,
    },
}
FACTORIAL_CONFIGURATIONS = (
    "ms2_niter0",
    "ms2_niter5",
    "ms0p8_niter0",
    "ms0p8_niter5",
)
SCREEN_SEED = 20260915
REPLICATION_SEEDS = (20260916, 20260917)
SAMPLES = {
    "P4": {"sample_root": "P4_sec1", "dataset": "P4_TUMOR"},
    "P6": {"sample_root": "P6_sec1", "dataset": "P6_TUMOR"},
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


def strict_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"{label} is not a strict boolean: {value!r}")


def require_complete(
    root: Path, label: str, kind: str
) -> tuple[Mapping[str, object], Mapping[str, object], list[Path]]:
    root = Path(root).resolve()
    if kind == "factorial":
        gate_name = "screen_gate_results.tsv"
        completion_schema = "sparcal_beagle_v2_3_screen_selection_completion_v1"
        provenance_schema = "sparcal_beagle_v2_3_screen_selection_v1"
        extra_output_names: tuple[str, ...] = ()
    elif kind == "extension":
        gate_name = "extension_gate_result.tsv"
        completion_schema = (
            "sparcal_beagle_v2_3_modelscale_extension_completion_v1"
        )
        provenance_schema = (
            "sparcal_beagle_v2_3_modelscale_extension_selection_v1"
        )
        extra_output_names = ()
    elif kind == "maxlr":
        gate_name = "maxlr_gate_result.tsv"
        completion_schema = "sparcal_beagle_v2_3_maxlr_extension_completion_v1"
        provenance_schema = "sparcal_beagle_v2_3_maxlr_extension_selection_v1"
        extra_output_names = (
            "matched_configuration_metrics_long.tsv",
            "matched_configuration_summary.tsv",
        )
    else:
        raise ValueError(f"unknown selection kind: {kind}")
    paths = [
        root / "RUN_COMPLETE",
        root / "RUN_COMPLETE.json",
        root / "PROVENANCE.json",
        root / "OUTPUT_SHA256SUMS.txt",
        root / gate_name,
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        *(root / name for name in extra_output_names),
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} is incomplete: {missing}")
    if paths[0].read_text().strip() != "complete":
        raise ValueError(f"{label} has an invalid RUN_COMPLETE marker")
    completion = json.loads(paths[1].read_text())
    provenance = json.loads(paths[2].read_text())
    if (
        completion.get("schema") != completion_schema
        or provenance.get("schema") != provenance_schema
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or provenance.get("final_default_selected") is not False
    ):
        raise ValueError(f"{label} has an invalid WES/status contract")
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=(
            gate_name,
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
            *extra_output_names,
        ),
        label=label,
    )
    INTEGRITY.parse_and_verify_ledger(
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label=f"{label} code/input ledger",
    )
    return completion, provenance, paths


def read_gate_rows(path: Path, label: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows or reader.fieldnames is None:
        raise ValueError(f"{label} gate table is empty")
    return rows


def base_config(config: object) -> str:
    suffix = f"_seed{SCREEN_SEED}"
    text = str(config)
    if not text.endswith(suffix):
        raise ValueError(f"screen configuration has unexpected seed label: {text}")
    base = text[: -len(suffix)]
    if base not in CONFIGURATIONS:
        raise ValueError(f"unknown screen configuration: {text}")
    return base


def selected_configs(
    factorial_root: Path, extension_root: Path, maxlr_root: Path
) -> tuple[list[str], list[Path]]:
    factorial, factorial_provenance, factorial_paths = require_complete(
        factorial_root, "factorial selection", "factorial"
    )
    expected_factorial = {
        f"{config}_seed{SCREEN_SEED}"
        for config in FACTORIAL_CONFIGURATIONS
    }
    factorial_rows = read_gate_rows(
        Path(factorial_root) / "screen_gate_results.tsv", "factorial"
    )
    observed_factorial = {str(row.get("config")) for row in factorial_rows}
    if observed_factorial != expected_factorial or len(factorial_rows) != len(
        expected_factorial
    ):
        raise ValueError("factorial gate table does not contain the exact screen grid")
    gate_selected = [
        str(row["config"])
        for row in factorial_rows
        if strict_bool(row.get("advance_to_replication"), "factorial advancement")
    ]
    completion_selected = [
        str(config) for config in factorial.get("replicate_configurations", [])
    ]
    provenance_selected = [
        str(config)
        for config in factorial_provenance.get("replicate_configurations", [])
    ]
    if gate_selected != completion_selected or gate_selected != provenance_selected:
        raise ValueError("factorial gate/provenance/completion decisions disagree")
    if factorial_provenance.get("comparator") != (
        f"ms2_niter0_seed{SCREEN_SEED}"
    ):
        raise ValueError("factorial selection has the wrong comparator")
    selected = [base_config(config) for config in gate_selected]
    if "ms2_niter0" not in selected:
        raise ValueError("factorial selection did not replicate the comparator")
    if len(selected) != len(set(selected)):
        raise ValueError("factorial selection contains duplicate configurations")

    extension, extension_provenance, extension_paths = require_complete(
        extension_root, "modelscale extension selection", "extension"
    )
    extension_rows = read_gate_rows(
        Path(extension_root) / "extension_gate_result.tsv", "extension"
    )
    if len(extension_rows) != 1:
        raise ValueError("extension gate table must contain exactly one row")
    row = extension_rows[0]
    expected_extension = f"ms4_niter0_seed{SCREEN_SEED}"
    if (
        row.get("config") != expected_extension
        or extension_provenance.get("extension") != expected_extension
        or extension_provenance.get("comparator")
        != f"ms2_niter0_seed{SCREEN_SEED}"
    ):
        raise ValueError("extension gate identifies the wrong configuration")
    advance = strict_bool(
        row.get("advance_to_replication"), "extension advancement"
    )
    if extension_provenance.get("advance_to_replication") is not advance:
        raise ValueError("extension gate/provenance decisions disagree")
    expected_replicate = [expected_extension] if advance else []
    if extension.get("replicate_configurations") != expected_replicate:
        raise ValueError("extension gate/completion decisions disagree")
    if advance:
        selected.append("ms4_niter0")

    maxlr, maxlr_provenance, maxlr_paths = require_complete(
        maxlr_root, "maxlr extension selection", "maxlr"
    )
    maxlr_rows = read_gate_rows(
        Path(maxlr_root) / "maxlr_gate_result.tsv", "maxlr"
    )
    if len(maxlr_rows) != 1:
        raise ValueError("maxlr gate table must contain exactly one row")
    row = maxlr_rows[0]
    expected_maxlr = f"ms2_niter0_maxlr50000_seed{SCREEN_SEED}"
    if (
        row.get("config") != expected_maxlr
        or row.get("comparator") != f"ms2_niter0_seed{SCREEN_SEED}"
        or maxlr_provenance.get("candidate") != expected_maxlr
        or maxlr_provenance.get("comparator")
        != f"ms2_niter0_seed{SCREEN_SEED}"
    ):
        raise ValueError("maxlr gate identifies the wrong configuration")
    maxlr_advance = strict_bool(
        row.get("advance_to_replication"), "maxlr advancement"
    )
    if maxlr_provenance.get("advance_to_replication") is not maxlr_advance:
        raise ValueError("maxlr gate/provenance decisions disagree")
    expected_maxlr_replicate = [expected_maxlr] if maxlr_advance else []
    if (
        maxlr.get("replicate_configurations") != expected_maxlr_replicate
        or maxlr_provenance.get("replicate_configurations")
        != expected_maxlr_replicate
    ):
        raise ValueError("maxlr gate/provenance/completion decisions disagree")
    if maxlr_advance:
        selected.append("ms2_niter0_maxlr50000")
    if len(selected) != len(set(selected)):
        raise ValueError("combined selectors contain duplicate configurations")
    return selected, factorial_paths + extension_paths + maxlr_paths


def build_plan(
    factorial_root: Path,
    extension_root: Path,
    maxlr_root: Path,
    pre_replication_audit_root: Path,
    output_dir: Path,
) -> Mapping[str, object]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {output_dir}")
    selected, source_paths = selected_configs(
        factorial_root, extension_root, maxlr_root
    )
    admitted = PREAUDIT.verify_audit_bundle(pre_replication_audit_root)
    if admitted != selected:
        raise ValueError(
            "selector admissions differ from the independently recomputed "
            "pre-replication audit"
        )
    audit_root = Path(pre_replication_audit_root).resolve()
    audit_paths = sorted(audit_root.iterdir(), key=lambda path: path.name)
    source_paths = [*source_paths, *audit_paths]
    rows = []
    for config in selected:
        setting = CONFIGURATIONS[config]
        for seed in REPLICATION_SEEDS:
            for sample, sample_info in SAMPLES.items():
                rows.append({
                    "task_id": len(rows),
                    "config": config,
                    "modelscale": setting["modelscale"],
                    "niterations": setting["niterations"],
                    "maxlr": setting["maxlr"],
                    "seed": seed,
                    "sample": sample,
                    "sample_root": sample_info["sample_root"],
                    "dataset": sample_info["dataset"],
                })
    if not rows:
        raise AssertionError("replication plan is empty")

    output_dir.mkdir(parents=True, exist_ok=False)
    task_path = output_dir / "replication_tasks.tsv"
    columns = (
        "task_id",
        "config",
        "modelscale",
        "niterations",
        "maxlr",
        "seed",
        "sample",
        "sample_root",
        "dataset",
    )
    with task_path.open("x") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(row[column]) for column in columns) + "\n")

    provenance_path = output_dir / "PROVENANCE.json"
    provenance = {
        "schema": "sparcal_beagle_v2_3_replication_plan_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "tasks_frozen",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "screen_seed": SCREEN_SEED,
        "replication_seeds": list(REPLICATION_SEEDS),
        "selected_configurations": selected,
        "task_count": len(rows),
        "factorial_selection_root": str(factorial_root.resolve()),
        "extension_selection_root": str(extension_root.resolve()),
        "maxlr_selection_root": str(maxlr_root.resolve()),
        "pre_replication_audit_root": str(audit_root),
        "selection_inputs": [identity(path) for path in source_paths],
        "pre_replication_audit_inputs": [
            identity(path) for path in audit_paths
        ],
        "planner": identity(Path(__file__)),
        "integrity_module": identity(Path(INTEGRITY.__file__)),
        "pre_replication_audit_module": identity(Path(PREAUDIT.__file__)),
    }
    provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum_path = output_dir / "OUTPUT_SHA256SUMS.txt"
    checksum_path.write_text(
        "\n".join(
            f"{sha256(path)}  {path.name}"
            for path in (task_path, provenance_path)
        )
        + "\n"
    )
    (output_dir / "RUN_COMPLETE").write_text("complete\n")
    (output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_replication_plan_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "task_count": len(rows),
                "array_max": len(rows) - 1,
                "selected_configurations": selected,
                "output_sha256sums": identity(checksum_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return {
        "status": "complete",
        "task_count": len(rows),
        "array_max": len(rows) - 1,
        "selected_configurations": selected,
        "task_file": str(task_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-selection-root", type=Path, required=True)
    parser.add_argument("--extension-selection-root", type=Path, required=True)
    parser.add_argument("--maxlr-selection-root", type=Path, required=True)
    parser.add_argument(
        "--pre-replication-audit-root", type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_plan(
        args.factorial_selection_root,
        args.extension_selection_root,
        args.maxlr_selection_root,
        args.pre_replication_audit_root,
        args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
