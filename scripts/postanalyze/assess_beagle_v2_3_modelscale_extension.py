#!/usr/bin/env python3
"""Gate the prespecified modelscale=4 arm against the clean ms2 comparator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SAMPLES = ("P4", "P6")
SEED = 20260915
COMPARATOR = f"ms2_niter0_seed{SEED}"
EXTENSION = f"ms4_niter0_seed{SEED}"
METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
SENSITIVITY_MARGIN = 0.005
SUPPORT_MARGIN = 0.005
IMPROVEMENT_MARGIN = 0.002


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.absolute()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": sha256(path),
    }


def require_file(path: Path, context: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {context}: {path}")
    return path


def load_contracts(path: Path):
    require_file(path, "v2.3 contracts module")
    spec = importlib.util.spec_from_file_location("beagle_v23_extension_contracts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import contracts module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_metrics(root: Path, expected_config: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = [
        require_file(root / "configuration_metrics_long.tsv", "metric table"),
        require_file(root / "configuration_summary.tsv", "summary table"),
        require_file(root / "PROVENANCE.json", "assessment provenance"),
        require_file(root / "RUN_COMPLETE", "assessment completion"),
    ]
    if paths[-1].read_text().strip() != "complete":
        raise ValueError(f"invalid assessment completion marker: {root}")
    provenance = json.loads(paths[2].read_text())
    if provenance.get("production_inputs_include_wes") is not False:
        raise ValueError("production_inputs_include_wes must be false")
    if provenance.get("evaluation_uses_wes") is not True:
        raise ValueError("evaluation_uses_wes must be true")
    if provenance.get("anchor_score") != "p_nonref" or not math.isclose(
        float(provenance.get("min_anchor_probability")),
        0.90,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("assessment routing rule is not P(nonref)>=0.90")
    metrics = pd.read_csv(paths[0], sep="\t")
    required = {"config", "sample", "metric", "numerator", "denominator", "value"}
    if missing := required.difference(metrics.columns):
        raise ValueError(f"metric table missing columns: {sorted(missing)}")
    selected = metrics.loc[
        (metrics["config"] == expected_config) & metrics["metric"].isin(METRICS)
    ].copy()
    expected = {
        (expected_config, sample, metric) for sample in SAMPLES for metric in METRICS
    }
    observed = set(
        selected[["config", "sample", "metric"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(selected) != len(expected):
        raise ValueError(f"assessment does not contain exact grid for {expected_config}")
    for column in ("numerator", "denominator", "value"):
        selected[column] = pd.to_numeric(selected[column], errors="raise")
    if (selected["denominator"] <= 0).any():
        raise ValueError("metric denominator must be positive")
    expected_values = selected["numerator"] / selected["denominator"]
    if not all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
        for a, b in zip(expected_values, selected["value"])
    ):
        raise ValueError("metric values do not equal numerator/denominator")
    target_audits = [
        row for row in provenance.get("audits", [])
        if str(row.get("config")) == expected_config
    ]
    audit_keys = {
        (str(row["config"]), str(row["sample"])) for row in target_audits
    }
    if (
        audit_keys != {(expected_config, sample) for sample in SAMPLES}
        or len(target_audits) != len(SAMPLES)
    ):
        raise ValueError(f"assessment audit provenance is incomplete for {expected_config}")
    return selected, paths


def assess_extension(comparator: pd.DataFrame, extension: pd.DataFrame) -> dict[str, object]:
    frame = pd.concat([comparator, extension], ignore_index=True)
    lookup = frame.set_index(["config", "sample", "metric"])

    def value(config: str, sample: str, metric: str) -> float:
        return float(lookup.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "numerator"])

    def denominator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "denominator"])

    for sample in SAMPLES:
        left = denominator(COMPARATOR, sample, "wes_somatic_anchor_leakage")
        right = denominator(EXTENSION, sample, "wes_somatic_anchor_leakage")
        if left != right:
            raise ValueError(f"somatic denominator differs for {sample}: {left} != {right}")
    sensitivity = {
        config: sum(
            value(config, sample, "normal_wes_anchor_sensitivity") for sample in SAMPLES
        ) / len(SAMPLES)
        for config in (COMPARATOR, EXTENSION)
    }
    support = {
        config: sum(
            value(config, sample, "selected_anchor_normal_wes_support")
            for sample in SAMPLES
        ) / len(SAMPLES)
        for config in (COMPARATOR, EXTENSION)
    }
    leakage = {
        config: {
            sample: numerator(config, sample, "wes_somatic_anchor_leakage")
            for sample in SAMPLES
        }
        for config in (COMPARATOR, EXTENSION)
    }
    sensitivity_delta = sensitivity[EXTENSION] - sensitivity[COMPARATOR]
    support_delta = support[EXTENSION] - support[COMPARATOR]
    leakage_reduction = (
        sum(leakage[COMPARATOR].values()) - sum(leakage[EXTENSION].values())
    )
    leakage_nonincrease = all(
        leakage[EXTENSION][sample] <= leakage[COMPARATOR][sample]
        for sample in SAMPLES
    )
    sensitivity_no_harm = sensitivity_delta >= -SENSITIVITY_MARGIN
    support_no_harm = support_delta >= -SUPPORT_MARGIN
    material_improvement = (
        sensitivity_delta >= IMPROVEMENT_MARGIN
        or support_delta >= IMPROVEMENT_MARGIN
        or leakage_reduction > 0
    )
    return {
        "config": EXTENSION,
        "comparator": COMPARATOR,
        "mean_normal_wes_anchor_sensitivity": sensitivity[EXTENSION],
        "delta_mean_sensitivity_vs_comparator": sensitivity_delta,
        "sensitivity_no_harm": sensitivity_no_harm,
        "mean_selected_anchor_normal_wes_support": support[EXTENSION],
        "delta_mean_support_vs_comparator": support_delta,
        "support_no_harm": support_no_harm,
        "P4_somatic_leakage_n": leakage[EXTENSION]["P4"],
        "P6_somatic_leakage_n": leakage[EXTENSION]["P6"],
        "P4_leakage_nonincrease": (
            leakage[EXTENSION]["P4"] <= leakage[COMPARATOR]["P4"]
        ),
        "P6_leakage_nonincrease": (
            leakage[EXTENSION]["P6"] <= leakage[COMPARATOR]["P6"]
        ),
        "leakage_nonincrease_both": leakage_nonincrease,
        "total_leakage_reduction": leakage_reduction,
        "material_improvement": material_improvement,
        "advance_to_replication": (
            leakage_nonincrease
            and sensitivity_no_harm
            and support_no_harm
            and material_improvement
        ),
    }


def hash_map(records: list[Mapping[str, object]], path_key: str) -> dict[str, str]:
    return {Path(str(row[path_key])).name: str(row["sha256"]) for row in records}


def manifest_signature(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "code": hash_map(manifest["code"], "path"),
        "executables": hash_map(list(manifest["executables"].values()), "path"),
        "reference": {
            str(row["contig"]): (
                row["panel"]["sha256"], row["panel_index"]["sha256"]
            )
            for row in manifest["reference_panels"]
        },
        "ledger": {
            key: value["sha256"]
            for key, value in manifest["reference_ledger"].items()
            if isinstance(value, dict) and "sha256" in value
        },
    }


def validate_matched_runs(
    comparator_root: Path,
    extension_root: Path,
    contracts_path: Path,
) -> list[dict[str, object]]:
    contracts = load_contracts(contracts_path)
    records = []
    for sample in SAMPLES:
        roots = {
            COMPARATOR: comparator_root / COMPARATOR / f"{sample}_sec1",
            EXTENSION: extension_root / EXTENSION / f"{sample}_sec1",
        }
        manifests = {}
        for config, root in roots.items():
            if (root / "RUN_FAILED.txt").exists():
                raise ValueError(f"run has failure marker: {root}")
            validation = contracts.validate_v2_3_run(root)
            if validation.get("status") != "verified":
                raise ValueError(f"run is not verified: {root}")
            manifest_path = require_file(root / "V2_3_MANIFEST.json", "v2.3 manifest")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("production_inputs_include_wes") is not False
                or manifest.get("evaluation_uses_wes") is not False
            ):
                raise ValueError(f"run has an invalid WES contract: {root}")
            expected_scale = 2.0 if config == COMPARATOR else 4.0
            params = manifest["beagle_parameters"]
            expected_params = {
                "MODEL_SCALE": expected_scale,
                "ITERATIONS": 0,
                "SEED": SEED,
                "IMPUTE": False,
                "GPROBS": True,
                "MAX_LR": 5000.0,
                "GENETIC_MAP": None,
            }
            for key, expected in expected_params.items():
                if params.get(key) != expected:
                    raise ValueError(f"parameter mismatch {root}/{key}")
            manifests[config] = manifest
            records.append({
                "config": config,
                "sample": sample,
                "root": str(root.absolute()),
                "manifest": identity(manifest_path),
                "completion": identity(root / "RUN_COMPLETE.json"),
                "validation": validation,
            })
        if manifests[COMPARATOR]["raw_vcf"]["sha256"] != manifests[EXTENSION]["raw_vcf"]["sha256"]:
            raise ValueError(f"raw input mismatch for {sample}")
        if manifest_signature(manifests[COMPARATOR]) != manifest_signature(manifests[EXTENSION]):
            raise ValueError(f"code/executable/reference identity mismatch for {sample}")
    return records


def sum_label(path: Path, base: Path) -> Path:
    try:
        return path.resolve().relative_to(base.resolve())
    except ValueError:
        return path.absolute()


def write_sums(paths: list[Path], base: Path, output: Path) -> None:
    output.write_text("\n".join(
        f"{sha256(path)}  {sum_label(path, base)}"
        for path in sorted(paths, key=lambda item: str(sum_label(item, base)))
    ) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-experiment-root", required=True, type=Path)
    parser.add_argument("--factorial-assessment-root", required=True, type=Path)
    parser.add_argument("--factorial-selection-root", required=True, type=Path)
    parser.add_argument("--extension-experiment-root", required=True, type=Path)
    parser.add_argument("--extension-assessment-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
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
        "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    require_file(args.protocol, "extension protocol")
    main_selection = require_file(
        args.factorial_selection_root / "RUN_COMPLETE.json", "factorial selection"
    )
    selection = json.loads(main_selection.read_text())
    if selection.get("status") != "complete" or COMPARATOR not in selection.get(
        "replicate_configurations", []
    ):
        raise ValueError("factorial selection did not advance the comparator")
    comparator, main_paths = load_metrics(args.factorial_assessment_root, COMPARATOR)
    extension, extension_paths = load_metrics(args.extension_assessment_root, EXTENSION)
    run_records = validate_matched_runs(
        args.factorial_experiment_root,
        args.extension_experiment_root,
        args.contracts,
    )
    result = assess_extension(comparator, extension)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = args.output_dir / "extension_gate_result.tsv"
    pd.DataFrame([result]).to_csv(result_path, sep="\t", index=False)
    provenance_path = args.output_dir / "PROVENANCE.json"
    provenance_path.write_text(json.dumps({
        "schema": "sparcal_beagle_v2_3_modelscale_extension_selection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "final_default_selected": False,
        "comparator": COMPARATOR,
        "extension": EXTENSION,
        "advance_to_replication": result["advance_to_replication"],
        "gate_thresholds": {
            "sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
            "support_no_harm_margin": SUPPORT_MARGIN,
            "material_improvement_margin": IMPROVEMENT_MARGIN,
        },
        "assessor": identity(Path(__file__)),
        "contracts": identity(args.contracts),
        "protocol": identity(args.protocol),
        "factorial_selection": identity(main_selection),
        "assessment_inputs": [
            identity(path) for path in [*main_paths, *extension_paths]
        ],
        "validated_runs": run_records,
    }, indent=2, sort_keys=True) + "\n")
    inputs = [
        Path(__file__),
        args.contracts,
        args.protocol,
        main_selection,
        *main_paths,
        *extension_paths,
    ]
    write_sums(inputs, PROJECT, args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt")
    write_sums(
        [result_path, provenance_path, args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"],
        args.output_dir,
        args.output_dir / "OUTPUT_SHA256SUMS.txt",
    )
    replicate = [EXTENSION] if result["advance_to_replication"] else []
    (args.output_dir / "RUN_COMPLETE.json").write_text(json.dumps({
        "schema": "sparcal_beagle_v2_3_modelscale_extension_completion_v1",
        "status": "complete",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "replicate_configurations": replicate,
        "output_sha256sums": identity(args.output_dir / "OUTPUT_SHA256SUMS.txt"),
    }, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    print(pd.DataFrame([result]).to_string(index=False))


if __name__ == "__main__":
    main()
