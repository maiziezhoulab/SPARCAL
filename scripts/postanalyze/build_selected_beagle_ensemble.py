#!/usr/bin/env python3
"""Build fixed posterior ensembles only for the selected v2.3 base setting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
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

SAMPLES = ("P4", "P6")
SEEDS = (20260915, 20260916, 20260917)


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


def load_selected_sources(
    assessment_root: Path,
) -> tuple[str, dict[str, dict[int, Path]], list[Path]]:
    assessment_root = Path(assessment_root).resolve()
    required = [
        assessment_root / "replicated_configuration_metrics_long.tsv",
        assessment_root / "replicated_configuration_summary.tsv",
        assessment_root / "replicated_pairwise_stability.tsv",
        assessment_root / "replication_gate_results.tsv",
        assessment_root / "FINAL_BASE_CONFIGURATION.json",
        assessment_root / "PROVENANCE.json",
        assessment_root / "CODE_AND_INPUT_SHA256SUMS.txt",
        assessment_root / "OUTPUT_SHA256SUMS.txt",
        assessment_root / "RUN_COMPLETE",
        assessment_root / "RUN_COMPLETE.json",
    ]
    if missing := [path for path in required if not path.is_file()]:
        raise FileNotFoundError(f"base assessment is incomplete: {missing}")
    if (assessment_root / "RUN_COMPLETE").read_text().strip() != "complete":
        raise ValueError("base assessment has invalid RUN_COMPLETE")
    selection = json.loads(
        (assessment_root / "FINAL_BASE_CONFIGURATION.json").read_text()
    )
    provenance = json.loads((assessment_root / "PROVENANCE.json").read_text())
    completion = json.loads((assessment_root / "RUN_COMPLETE.json").read_text())
    base = str(selection.get("selected_base_configuration", ""))
    if (
        selection.get("schema")
        != "sparcal_beagle_v2_3_final_base_selection_v1"
        or provenance.get("schema")
        != "sparcal_beagle_v2_3_replication_assessment_v1"
        or completion.get("schema")
        != "sparcal_beagle_v2_3_replication_completion_v1"
        or not base
        or selection.get("production_inputs_include_wes") is not False
        or selection.get("evaluation_uses_wes") is not True
        or selection.get("final_ensemble_selected") is not False
        or provenance.get("selected_base_configuration") != base
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not True
        or completion.get("selected_base_configuration") != base
        or completion.get("status") != "complete"
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not True
    ):
        raise ValueError("base assessment selection/status contract is invalid")
    INTEGRITY.verify_output_bundle(
        assessment_root,
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
        label="base assessment",
    )
    INTEGRITY.parse_and_verify_ledger(
        assessment_root / "CODE_AND_INPUT_SHA256SUMS.txt",
        roots=(PROJECT,),
        label="base assessment code/input ledger",
    )

    with (assessment_root / "replication_gate_results.tsv").open(
        newline=""
    ) as handle:
        gate_rows = list(csv.DictReader(handle, delimiter="\t"))
    selected_rows = [
        row
        for row in gate_rows
        if str(row.get("selected_base_default", "")).lower() == "true"
    ]
    if (
        len(selected_rows) != 1
        or selected_rows[0].get("base_config") != base
        or str(
            selected_rows[0].get("eligible_as_base_default", "")
        ).lower()
        != "true"
    ):
        raise ValueError("base gate/selection/provenance decisions disagree")
    selected_configs = provenance.get("selected_configurations")
    if (
        not isinstance(selected_configs, list)
        or base not in selected_configs
        or provenance.get("seeds") != list(SEEDS)
        or provenance.get("samples") != list(SAMPLES)
    ):
        raise ValueError("base provenance has an invalid selection grid")

    sources = {sample: {} for sample in SAMPLES}
    validated_runs = provenance.get("validated_runs")
    if not isinstance(validated_runs, list):
        raise ValueError("base assessment lacks validated run records")
    observed_runs = set()
    for row_index, row in enumerate(validated_runs):
        config = str(row.get("config"))
        sample = str(row.get("sample"))
        key = (config, sample)
        if key in observed_runs:
            raise ValueError(f"duplicate validated run: {config}/{sample}")
        observed_runs.add(key)
        for field in ("manifest", "completion", "audit", "runtime_metrics"):
            INTEGRITY.verify_identity(
                row.get(field),
                label=f"validated run {row_index} {field}",
            )
        for seed in SEEDS:
            if config == f"{base}_seed{seed}" and sample in SAMPLES:
                audit_identity = row.get("audit")
                path = Path(str(audit_identity.get("path", ""))).resolve()
                sources[sample][seed] = path
    expected_runs = {
        (f"{config}_seed{seed}", sample)
        for config in selected_configs
        for seed in SEEDS
        for sample in SAMPLES
    }
    if observed_runs != expected_runs or len(validated_runs) != len(expected_runs):
        raise ValueError("base assessment validated-run grid is not exact")
    for sample in SAMPLES:
        if set(sources[sample]) != set(SEEDS):
            raise ValueError(f"base assessment lacks three source seeds for {sample}")
    return base, sources, required


def build(
    *,
    base_assessment_root: Path,
    output_root: Path,
    ensemble_module_path: Path,
    contracts_path: Path,
    protocol_path: Path,
) -> Mapping[str, object]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite: {output_root}")
    base, sources, assessment_paths = load_selected_sources(base_assessment_root)
    ensemble = load_module(
        ensemble_module_path, "_beagle_selected_ensemble_builder"
    )
    contracts = load_module(
        contracts_path, "_beagle_selected_ensemble_contracts"
    )
    results = {}
    for sample in SAMPLES:
        results[sample] = ensemble.build(
            audits=sources[sample],
            output_root=output_root,
            sample=sample,
            contracts_path=contracts_path,
            protocol_path=protocol_path,
        )

    child_completions = [
        output_root / method / f"{sample}_sec1" / "RUN_COMPLETE.json"
        for method in ensemble.METHODS
        for sample in SAMPLES
    ]
    if missing := [path for path in child_completions if not path.is_file()]:
        raise AssertionError(f"ensemble builder missed child completions: {missing}")
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_selected_posterior_ensemble_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "outputs_ready_for_checksum",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": False,
                "selected_base_configuration": base,
                "samples": list(SAMPLES),
                "frozen_seeds": list(SEEDS),
                "methods": list(ensemble.METHODS),
                "results": results,
                "base_assessment_inputs": [
                    identity(path) for path in assessment_paths
                ],
                "builder": identity(Path(__file__)),
                "integrity_module": identity(Path(INTEGRITY.__file__)),
                "ensemble_module": identity(ensemble_module_path),
                "contracts": identity(contracts_path),
                "protocol": identity(protocol_path),
                "child_completions": [
                    identity(path) for path in child_completions
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    simple_completion = output_root / "RUN_COMPLETE"
    simple_completion.write_text("complete\n")
    checksum_path = contracts.write_sha256sums(
        output_root,
        (manifest_path, simple_completion, *child_completions),
    )
    completion_path = contracts.write_completion(
        output_root,
        manifest_path=manifest_path,
        checksum_path=checksum_path,
    )
    return {
        "status": "complete",
        "selected_base_configuration": base,
        "output_root": str(output_root.resolve()),
        "manifest": str(manifest_path),
        "completion": str(completion_path),
    }


def main() -> int:
    candidate = (
        PROJECT
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-assessment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--ensemble-module",
        type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_posterior_ensemble.py",
    )
    parser.add_argument(
        "--contracts", type=Path, default=candidate / "contracts.py"
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=candidate / "BENCHMARK_PROTOCOL_ENSEMBLE.md",
    )
    args = parser.parse_args()
    for path in (args.ensemble_module, args.contracts, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = build(
        base_assessment_root=args.base_assessment_root,
        output_root=args.output_root,
        ensemble_module_path=args.ensemble_module.resolve(),
        contracts_path=args.contracts.resolve(),
        protocol_path=args.protocol.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
