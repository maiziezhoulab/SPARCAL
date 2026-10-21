#!/usr/bin/env python3
"""Apply frozen Beagle v2.3 screen gates and bind all run identities.

The metric calculator and this decision layer are intentionally separate.
Matched WES is development-set evaluation only; it is never a production input.
"""

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
CONFIGURATIONS = {
    COMPARATOR: {"modelscale": 2.0, "niterations": 0, "one_factor": False},
    f"ms2_niter5_seed{SEED}": {
        "modelscale": 2.0, "niterations": 5, "one_factor": True,
    },
    f"ms0p8_niter0_seed{SEED}": {
        "modelscale": 0.8, "niterations": 0, "one_factor": True,
    },
    f"ms0p8_niter5_seed{SEED}": {
        "modelscale": 0.8, "niterations": 5, "one_factor": False,
    },
}
METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
SENSITIVITY_MARGIN = 0.005
SUPPORT_MARGIN = 0.005
IMPROVEMENT_MARGIN = 0.002
TRUTH_INPUTS = {
    "normal": PROJECT / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz",
    "support": PROJECT / "data/germline_concordance_2026-08-23/sparcal_call_details.csv.gz",
    "somatic": PROJECT / "data/somatic_evidence_2026-07-28/wes_confirmed_full_accounting.csv",
    "table_s1": PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv",
}


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


def require_false(value: object, context: str) -> None:
    if value is not False:
        raise ValueError(f"{context} must be false")


def load_contracts(path: Path):
    require_file(path, "v2.3 contracts module")
    spec = importlib.util.spec_from_file_location("beagle_v23_screen_contracts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import contracts module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_fraction_rows(frame: pd.DataFrame) -> None:
    required = {"config", "sample", "metric", "numerator", "denominator", "value"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"metric table missing columns: {sorted(missing)}")
    chosen = frame.loc[frame["metric"].isin(METRICS)].copy()
    expected = {
        (config, sample, metric)
        for config in CONFIGURATIONS for sample in SAMPLES for metric in METRICS
    }
    observed = set(chosen[["config", "sample", "metric"]].itertuples(index=False, name=None))
    if observed != expected or len(chosen) != len(expected):
        raise ValueError(
            f"screen metric grid is not exact; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )
    for column in ("numerator", "denominator", "value"):
        chosen[column] = pd.to_numeric(chosen[column], errors="raise")
    if (chosen["denominator"] <= 0).any():
        raise ValueError("screen metrics contain a non-positive denominator")
    if (chosen["numerator"] < 0).any() or (chosen["numerator"] > chosen["denominator"]).any():
        raise ValueError("screen metrics contain invalid counts")
    expected_value = chosen["numerator"] / chosen["denominator"]
    if not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
               for a, b in zip(expected_value, chosen["value"])):
        raise ValueError("screen metric value does not equal numerator/denominator")


def assess_screen(metrics: pd.DataFrame) -> pd.DataFrame:
    """Return one frozen gate row per configuration."""
    validate_fraction_rows(metrics)
    lookup = metrics.loc[metrics["metric"].isin(METRICS)].set_index(
        ["config", "sample", "metric"]
    )

    def value(config: str, sample: str, metric: str) -> float:
        return float(lookup.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "numerator"])

    def denominator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "denominator"])

    for sample in SAMPLES:
        baseline = denominator(COMPARATOR, sample, "wes_somatic_anchor_leakage")
        for config in CONFIGURATIONS:
            observed = denominator(config, sample, "wes_somatic_anchor_leakage")
            if observed != baseline:
                raise ValueError(
                    f"somatic denominator differs for {config}/{sample}: "
                    f"{observed} != {baseline}"
                )

    comparator_sensitivity = sum(
        value(COMPARATOR, sample, "normal_wes_anchor_sensitivity") for sample in SAMPLES
    ) / len(SAMPLES)
    comparator_support = sum(
        value(COMPARATOR, sample, "selected_anchor_normal_wes_support") for sample in SAMPLES
    ) / len(SAMPLES)
    comparator_by_sample = {
        sample: numerator(COMPARATOR, sample, "wes_somatic_anchor_leakage")
        for sample in SAMPLES
    }
    rows = []
    for config, setting in CONFIGURATIONS.items():
        mean_sensitivity = sum(
            value(config, sample, "normal_wes_anchor_sensitivity") for sample in SAMPLES
        ) / len(SAMPLES)
        mean_support = sum(
            value(config, sample, "selected_anchor_normal_wes_support") for sample in SAMPLES
        ) / len(SAMPLES)
        leakage = {
            sample: numerator(config, sample, "wes_somatic_anchor_leakage")
            for sample in SAMPLES
        }
        sensitivity_delta = mean_sensitivity - comparator_sensitivity
        support_delta = mean_support - comparator_support
        leakage_reduction = sum(comparator_by_sample.values()) - sum(leakage.values())
        leakage_nonincrease = all(
            leakage[sample] <= comparator_by_sample[sample] for sample in SAMPLES
        )
        sensitivity_no_harm = sensitivity_delta >= -SENSITIVITY_MARGIN
        support_no_harm = support_delta >= -SUPPORT_MARGIN
        material_improvement = (
            sensitivity_delta >= IMPROVEMENT_MARGIN
            or support_delta >= IMPROVEMENT_MARGIN
            or leakage_reduction > 0
        )
        comparator = config == COMPARATOR
        advances = comparator or (
            leakage_nonincrease
            and sensitivity_no_harm
            and support_no_harm
            and (material_improvement or bool(setting["one_factor"]))
        )
        rows.append({
            "config": config,
            "modelscale": setting["modelscale"],
            "niterations": setting["niterations"],
            "comparator": comparator,
            "required_one_factor_ablation": bool(setting["one_factor"]),
            "mean_normal_wes_anchor_sensitivity": mean_sensitivity,
            "delta_mean_sensitivity_vs_comparator": sensitivity_delta,
            "sensitivity_no_harm": sensitivity_no_harm,
            "mean_selected_anchor_normal_wes_support": mean_support,
            "delta_mean_support_vs_comparator": support_delta,
            "support_no_harm": support_no_harm,
            "P4_somatic_leakage_n": leakage["P4"],
            "P6_somatic_leakage_n": leakage["P6"],
            "P4_leakage_nonincrease": leakage["P4"] <= comparator_by_sample["P4"],
            "P6_leakage_nonincrease": leakage["P6"] <= comparator_by_sample["P6"],
            "leakage_nonincrease_both": leakage_nonincrease,
            "total_leakage_reduction": leakage_reduction,
            "material_improvement": material_improvement,
            "advance_to_replication": advances,
        })
    return pd.DataFrame(rows)


def validate_assessment(
    root: Path,
) -> tuple[pd.DataFrame, list[Path], Mapping[str, object]]:
    paths = [
        require_file(root / "configuration_metrics_long.tsv", "metric table"),
        require_file(root / "configuration_summary.tsv", "configuration summary"),
        require_file(root / "pairwise_stability.tsv", "pairwise table"),
        require_file(root / "PROVENANCE.json", "assessment provenance"),
        require_file(root / "RUN_COMPLETE", "assessment completion marker"),
    ]
    if paths[-1].read_text().strip() != "complete":
        raise ValueError("assessment RUN_COMPLETE marker is invalid")
    provenance = json.loads(paths[3].read_text())
    require_false(
        provenance.get("production_inputs_include_wes"),
        "production_inputs_include_wes",
    )
    if provenance.get("evaluation_uses_wes") is not True:
        raise ValueError("assessment must declare evaluation_uses_wes=true")
    if provenance.get("anchor_score") != "p_nonref":
        raise ValueError("assessment must use p_nonref routing")
    if not math.isclose(
        float(provenance.get("min_anchor_probability")),
        0.90,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("assessment must use P(nonref)>=0.90")
    metrics = pd.read_csv(paths[0], sep="\t")
    validate_fraction_rows(metrics)
    summary = pd.read_csv(paths[1], sep="\t")
    expected_rows = {
        (config, sample) for config in CONFIGURATIONS for sample in SAMPLES
    }
    observed_rows = set(summary[["config", "sample"]].itertuples(index=False, name=None))
    if observed_rows != expected_rows or len(summary) != len(expected_rows):
        raise ValueError("configuration summary is not an exact 4x2 screen")
    for row in metrics.loc[metrics["metric"].isin(METRICS)].itertuples(index=False):
        recorded = summary.loc[
            (summary["config"] == row.config) & (summary["sample"] == row.sample),
            row.metric,
        ]
        if len(recorded) != 1 or not math.isclose(
            float(recorded.iloc[0]), float(row.value), rel_tol=0, abs_tol=1e-12
        ):
            raise ValueError(
                f"long/summary mismatch: {row.config}/{row.sample}/{row.metric}"
            )
    audit_rows = provenance.get("audits", [])
    audit_keys = {
        (str(row.get("config")), str(row.get("sample"))) for row in audit_rows
    }
    if audit_keys != expected_rows or len(audit_rows) != len(expected_rows):
        raise ValueError("assessment provenance does not bind exactly eight audits")
    return metrics, paths, provenance


def named_hashes(records: list[Mapping[str, object]], path_key: str) -> dict[str, str]:
    return {Path(str(record[path_key])).name: str(record["sha256"]) for record in records}


def validate_runs(
    experiment_root: Path,
    assessment_provenance: Mapping[str, object],
    contracts_path: Path,
    reference_ledger_root: Path,
) -> list[dict[str, object]]:
    contracts = load_contracts(contracts_path)
    audit_lookup = {
        (str(row["config"]), str(row["sample"])): row
        for row in assessment_provenance["audits"]
    }
    validated = []
    common_code = common_executables = common_reference = common_ledger = None
    raw_by_sample: dict[str, str] = {}
    for config, setting in CONFIGURATIONS.items():
        for sample in SAMPLES:
            run_root = experiment_root / config / f"{sample}_sec1"
            if (run_root / "RUN_FAILED.txt").exists():
                raise ValueError(f"run carries failure marker: {run_root}")
            result = contracts.validate_v2_3_run(run_root)
            if result.get("status") != "verified":
                raise ValueError(f"v2.3 run is not verified: {run_root}")
            manifest_path = require_file(run_root / "V2_3_MANIFEST.json", "v2.3 manifest")
            manifest = json.loads(manifest_path.read_text())
            require_false(
                manifest.get("production_inputs_include_wes"), "manifest WES input flag"
            )
            require_false(
                manifest.get("evaluation_uses_wes"), "manifest WES evaluation flag"
            )
            if manifest.get("quality_filter") != "baseQ0mapQ0":
                raise ValueError(f"unexpected input quality for {run_root}")
            if manifest.get("anchor_score") != "p_nonref" or not math.isclose(
                float(manifest.get("min_anchor_probability")),
                0.90,
                rel_tol=0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"unexpected routing rule for {run_root}")
            params = manifest.get("beagle_parameters", {})
            expected_params = {
                "MODEL_SCALE": float(setting["modelscale"]),
                "ITERATIONS": int(setting["niterations"]),
                "SEED": SEED,
                "IMPUTE": False,
                "GPROBS": True,
                "MAX_LR": 5000.0,
                "GENETIC_MAP": None,
            }
            for key, expected in expected_params.items():
                if params.get(key) != expected:
                    raise ValueError(
                        f"parameter mismatch {run_root}/{key}: "
                        f"{params.get(key)!r} != {expected!r}"
                    )
            if (
                Path(str(params.get("REFERENCE_LEDGER"))).absolute()
                != reference_ledger_root.absolute()
            ):
                raise ValueError(f"reference ledger mismatch: {run_root}")

            audit = audit_lookup[(config, sample)]
            audit_path = Path(str(audit["audit"]))
            try:
                audit_path.resolve().relative_to(run_root.resolve())
            except ValueError as exc:
                raise ValueError(f"audit is outside its run root: {audit_path}") from exc
            if sha256(audit_path) != str(audit["audit_sha256"]):
                raise ValueError(f"audit hash mismatch: {audit_path}")

            code = named_hashes(manifest["code"], "path")
            executables = named_hashes(list(manifest["executables"].values()), "path")
            reference = {
                str(row["contig"]): (
                    row["panel"]["sha256"], row["panel_index"]["sha256"]
                )
                for row in manifest["reference_panels"]
            }
            ledger = {
                key: value["sha256"]
                for key, value in manifest["reference_ledger"].items()
                if isinstance(value, dict) and "sha256" in value
            }
            if common_code is None:
                common_code, common_executables = code, executables
                common_reference, common_ledger = reference, ledger
            elif (
                code != common_code
                or executables != common_executables
                or reference != common_reference
                or ledger != common_ledger
            ):
                raise ValueError(
                    "screen runs do not share one code/executable/reference identity"
                )
            raw_hash = str(manifest["raw_vcf"]["sha256"])
            if sample in raw_by_sample and raw_by_sample[sample] != raw_hash:
                raise ValueError(f"raw input differs across configurations for {sample}")
            raw_by_sample[sample] = raw_hash
            validated.append({
                "config": config,
                "sample": sample,
                "run_root": str(run_root.absolute()),
                "manifest": identity(manifest_path),
                "completion": identity(run_root / "RUN_COMPLETE.json"),
                "audit": identity(audit_path),
                "raw_vcf_sha256": raw_hash,
                "validation": result,
            })
    return validated


def write_sums(paths: list[Path], base: Path, output: Path) -> None:
    def label(path: Path) -> Path:
        try:
            return path.resolve().relative_to(base.resolve())
        except ValueError:
            return path.absolute()

    output.write_text("\n".join(
        f"{sha256(path)}  {label(path)}"
        for path in sorted(paths, key=lambda item: str(label(item)))
    ) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", required=True, type=Path)
    parser.add_argument("--assessment-root", required=True, type=Path)
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
        / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md",
    )
    parser.add_argument(
        "--reference-ledger-root",
        type=Path,
        default=PROJECT
        / "data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    require_file(args.protocol, "frozen benchmark protocol")
    require_file(
        args.reference_ledger_root / "RUN_COMPLETE.json", "frozen reference ledger"
    )
    for path in TRUTH_INPUTS.values():
        require_file(path, "development truth input")
    metrics, assessment_paths, assessment_provenance = validate_assessment(
        args.assessment_root
    )
    validated_runs = validate_runs(
        args.experiment_root,
        assessment_provenance,
        args.contracts,
        args.reference_ledger_root,
    )
    gates = assess_screen(metrics)
    replicate_configs = gates.loc[
        gates["advance_to_replication"], "config"
    ].tolist()
    if COMPARATOR not in replicate_configs:
        raise AssertionError("comparator must always advance")

    args.output_dir.mkdir(parents=True, exist_ok=False)
    gates_path = args.output_dir / "screen_gate_results.tsv"
    gates.to_csv(gates_path, sep="\t", index=False)
    provenance_path = args.output_dir / "PROVENANCE.json"
    provenance_path.write_text(json.dumps({
        "schema": "sparcal_beagle_v2_3_screen_selection_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "final_default_selected": False,
        "experiment_root": str(args.experiment_root.absolute()),
        "assessment_root": str(args.assessment_root.absolute()),
        "comparator": COMPARATOR,
        "replicate_configurations": replicate_configs,
        "gate_thresholds": {
            "sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
            "support_no_harm_margin": SUPPORT_MARGIN,
            "material_improvement_margin": IMPROVEMENT_MARGIN,
        },
        "assessor": identity(Path(__file__)),
        "metric_evaluator": identity(
            PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py"
        ),
        "contracts": identity(args.contracts),
        "protocol": identity(args.protocol),
        "reference_ledger_completion": identity(
            args.reference_ledger_root / "RUN_COMPLETE.json"
        ),
        "assessment_inputs": [identity(path) for path in assessment_paths],
        "truth_inputs": {
            name: identity(path) for name, path in sorted(TRUTH_INPUTS.items())
        },
        "validated_runs": validated_runs,
    }, indent=2, sort_keys=True) + "\n")
    code_and_inputs = [
        Path(__file__),
        PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py",
        args.contracts,
        args.protocol,
        args.reference_ledger_root / "RUN_COMPLETE.json",
        *assessment_paths,
        *TRUTH_INPUTS.values(),
    ]
    write_sums(
        code_and_inputs, PROJECT, args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    )
    write_sums(
        [gates_path, provenance_path, args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"],
        args.output_dir,
        args.output_dir / "OUTPUT_SHA256SUMS.txt",
    )
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    (args.output_dir / "RUN_COMPLETE.json").write_text(json.dumps({
        "schema": "sparcal_beagle_v2_3_screen_selection_completion_v1",
        "status": "complete",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "replicate_configurations": replicate_configs,
        "output_sha256sums": identity(
            args.output_dir / "OUTPUT_SHA256SUMS.txt"
        ),
    }, indent=2, sort_keys=True) + "\n")
    print(gates.to_string(index=False))
    print("Replicate configurations: " + ", ".join(replicate_configs))


if __name__ == "__main__":
    main()
