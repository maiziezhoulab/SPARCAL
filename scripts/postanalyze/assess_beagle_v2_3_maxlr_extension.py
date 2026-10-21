#!/usr/bin/env python3
"""Gate the frozen maxlr=50000 full-genome arm against maxlr=5000.

Metrics are recomputed directly from validated canonical audits. Matched WES
is development-set evaluation only and is never a production input.
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
CANDIDATE = f"ms2_niter0_maxlr50000_seed{SEED}"
SETTINGS = {
    COMPARATOR: {"modelscale": 2.0, "niterations": 0, "maxlr": 5000.0},
    CANDIDATE: {"modelscale": 2.0, "niterations": 0, "maxlr": 50000.0},
}
GATE_METRICS = (
    "normal_wes_anchor_sensitivity",
    "selected_anchor_normal_wes_support",
    "wes_somatic_anchor_leakage",
)
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


def require_file(path: Path, label: str) -> Path:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    return path


def load_module(path: Path, name: str):
    path = require_file(path, name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_prescreen(root: Path) -> list[Path]:
    root = Path(root).resolve()
    paths = [
        root / "maxlr_prescreen_summary.tsv",
        root / "PROVENANCE.json",
        root / "CODE_AND_INPUT_SHA256SUMS.txt",
        root / "OUTPUT_SHA256SUMS.txt",
        root / "RUN_COMPLETE",
        root / "RUN_COMPLETE.json",
    ]
    if missing := [path for path in paths if not path.is_file()]:
        raise FileNotFoundError(f"maxlr prescreen is incomplete: {missing}")
    if paths[4].read_text().strip() != "complete":
        raise ValueError("maxlr prescreen has invalid RUN_COMPLETE")
    provenance = json.loads(paths[1].read_text())
    completion = json.loads(paths[5].read_text())
    if (
        provenance.get("schema") != "sparcal_beagle_maxlr_prescreen_v1"
        or completion.get("schema")
        != "sparcal_beagle_maxlr_prescreen_completion_v1"
        or completion.get("status") != "complete"
        or provenance.get("production_inputs_include_wes") is not False
        or provenance.get("evaluation_uses_wes") is not False
        or completion.get("production_inputs_include_wes") is not False
        or completion.get("evaluation_uses_wes") is not False
        or provenance.get("advance_to_full_genome") is not True
        or completion.get("advance_to_full_genome") is not True
        or provenance.get("decision")
        != "advance_to_frozen_full_genome_comparison"
        or completion.get("decision")
        != "advance_to_frozen_full_genome_comparison"
    ):
        raise ValueError("maxlr prescreen did not satisfy the frozen advancement contract")
    INTEGRITY.verify_output_bundle(
        root,
        completion,
        expected_names=(
            "maxlr_prescreen_summary.tsv",
            "PROVENANCE.json",
            "CODE_AND_INPUT_SHA256SUMS.txt",
        ),
        label="maxlr prescreen",
    )
    INTEGRITY.parse_and_verify_ledger(
        paths[2], roots=(PROJECT,), label="maxlr prescreen code/input ledger"
    )
    for name in ("assessor", "contracts", "protocol"):
        INTEGRITY.verify_identity(
            provenance.get(name), label=f"maxlr prescreen {name}"
        )
    for side in ("comparator", "candidate"):
        record = provenance.get("inputs", {}).get(side, {})
        for name in ("manifest", "completion", "audit"):
            INTEGRITY.verify_identity(
                record.get(name), label=f"maxlr prescreen {side} {name}"
            )
    return paths


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
        "reference_ledger": sorted(
            (name, str(value["sha256"]))
            for name, value in manifest["reference_ledger"].items()
            if isinstance(value, Mapping) and "sha256" in value
        ),
    }


def canonical_audit(root: Path, manifest: Mapping[str, object]) -> Path:
    return require_file(
        root
        / "output_VCFs"
        / "beagle"
        / str(manifest["quality_filter"])
        / "beagle_evidence_audit.tsv.gz",
        "canonical Beagle audit",
    )


def load_truth() -> tuple[dict[str, pd.DataFrame], list[Path]]:
    for path in TRUTH_INPUTS.values():
        require_file(path, "development truth input")
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


def validate_and_evaluate(
    *,
    factorial_root: Path,
    maxlr_root: Path,
    contracts,
    benchmark,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], list[Path]]:
    truth, truth_paths = load_truth()
    metrics: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []
    input_paths: list[Path] = []
    signatures: set[str] = set()

    for sample in SAMPLES:
        sample_evidence = {}
        sample_manifests = {}
        roots = {
            COMPARATOR: Path(factorial_root) / COMPARATOR / f"{sample}_sec1",
            CANDIDATE: Path(maxlr_root) / CANDIDATE / f"{sample}_sec1",
        }
        for config, root in roots.items():
            root = root.resolve()
            if (root / "RUN_FAILED.txt").exists():
                raise ValueError(f"run has a failure marker: {root}")
            validation = contracts.validate_v2_3_run(root)
            if validation.get("status") != "verified":
                raise ValueError(f"run did not pass v2.3 validation: {root}")
            manifest_path = require_file(root / "V2_3_MANIFEST.json", "run manifest")
            completion_path = require_file(
                root / "RUN_COMPLETE.json", "run completion"
            )
            manifest = json.loads(manifest_path.read_text())
            setting = SETTINGS[config]
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
                raise ValueError(f"run has an invalid WES/routing contract: {root}")
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
            for key, expected in expected_params.items():
                if params.get(key) != expected:
                    raise ValueError(
                        f"parameter mismatch {root}/{key}: "
                        f"{params.get(key)!r} != {expected!r}"
                    )
            signature = manifest_signature(manifest)
            signatures.add(json.dumps(signature, sort_keys=True))
            sample_manifests[config] = manifest

            audit = canonical_audit(root, manifest)
            evidence = benchmark.reroute_evidence(
                benchmark.read_evidence(audit), "p_nonref", 0.90
            )
            sample_evidence[config] = evidence
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
            input_paths.extend((manifest_path, completion_path, audit))
            run_records.append({
                "config": config,
                "sample": sample,
                "run_root": str(root),
                "manifest": identity(manifest_path),
                "completion": identity(completion_path),
                "audit": identity(audit),
                "validation": validation,
            })

        comparator_manifest = sample_manifests[COMPARATOR]
        candidate_manifest = sample_manifests[CANDIDATE]
        if comparator_manifest["raw_vcf"]["sha256"] != candidate_manifest[
            "raw_vcf"
        ]["sha256"]:
            raise ValueError(f"raw input differs for {sample}")
        if comparator_manifest["raw_vcf_index"]["sha256"] != candidate_manifest[
            "raw_vcf_index"
        ]["sha256"]:
            raise ValueError(f"raw input index differs for {sample}")
        if set(sample_evidence[COMPARATOR]) != set(sample_evidence[CANDIDATE]):
            raise ValueError(f"exact audit allele universe differs for {sample}")

    if len(signatures) != 1:
        raise ValueError("runs do not share one code/executable/reference identity")
    metrics_frame = pd.DataFrame(metrics)
    summary_frame = pd.DataFrame(summaries)
    expected_metric_rows = {
        (config, sample, metric)
        for config in SETTINGS
        for sample in SAMPLES
        for metric in set(metrics_frame["metric"])
    }
    observed_metric_rows = set(
        metrics_frame[["config", "sample", "metric"]].itertuples(
            index=False, name=None
        )
    )
    if (
        observed_metric_rows != expected_metric_rows
        or metrics_frame.duplicated(["config", "sample", "metric"]).any()
    ):
        raise ValueError("recomputed metric grid is not exact")
    return metrics_frame, summary_frame, run_records, [*input_paths, *truth_paths]


def assess(metrics: pd.DataFrame) -> dict[str, object]:
    chosen = metrics.loc[metrics["metric"].isin(GATE_METRICS)].copy()
    expected = {
        (config, sample, metric)
        for config in SETTINGS
        for sample in SAMPLES
        for metric in GATE_METRICS
    }
    observed = set(
        chosen[["config", "sample", "metric"]].itertuples(index=False, name=None)
    )
    if observed != expected or len(chosen) != len(expected):
        raise ValueError("maxlr gate metric grid is not exact")
    for column in ("numerator", "denominator", "value"):
        chosen[column] = pd.to_numeric(chosen[column], errors="raise")
    if (
        (chosen["denominator"] <= 0).any()
        or (chosen["numerator"] < 0).any()
        or (chosen["numerator"] > chosen["denominator"]).any()
    ):
        raise ValueError("maxlr gate metrics have invalid counts")
    if not all(
        math.isclose(value, numerator / denominator, rel_tol=0, abs_tol=1e-12)
        for value, numerator, denominator in zip(
            chosen["value"], chosen["numerator"], chosen["denominator"]
        )
    ):
        raise ValueError("maxlr gate metric values do not equal counts")
    lookup = chosen.set_index(["config", "sample", "metric"])

    def value(config: str, sample: str, metric: str) -> float:
        return float(lookup.loc[(config, sample, metric), "value"])

    def numerator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "numerator"])

    def denominator(config: str, sample: str, metric: str) -> int:
        return int(lookup.loc[(config, sample, metric), "denominator"])

    for sample in SAMPLES:
        if denominator(COMPARATOR, sample, "wes_somatic_anchor_leakage") != denominator(
            CANDIDATE, sample, "wes_somatic_anchor_leakage"
        ):
            raise ValueError(f"somatic denominator differs for {sample}")
    sensitivity = {
        config: sum(
            value(config, sample, "normal_wes_anchor_sensitivity")
            for sample in SAMPLES
        )
        / len(SAMPLES)
        for config in SETTINGS
    }
    support = {
        config: sum(
            value(config, sample, "selected_anchor_normal_wes_support")
            for sample in SAMPLES
        )
        / len(SAMPLES)
        for config in SETTINGS
    }
    leakage = {
        config: {
            sample: numerator(config, sample, "wes_somatic_anchor_leakage")
            for sample in SAMPLES
        }
        for config in SETTINGS
    }
    sensitivity_delta = sensitivity[CANDIDATE] - sensitivity[COMPARATOR]
    support_delta = support[CANDIDATE] - support[COMPARATOR]
    leakage_reduction = sum(leakage[COMPARATOR].values()) - sum(
        leakage[CANDIDATE].values()
    )
    leakage_nonincrease = all(
        leakage[CANDIDATE][sample] <= leakage[COMPARATOR][sample]
        for sample in SAMPLES
    )
    sensitivity_no_harm = sensitivity_delta >= -SENSITIVITY_MARGIN
    support_no_harm = support_delta >= -SUPPORT_MARGIN
    material_improvement = (
        sensitivity_delta >= IMPROVEMENT_MARGIN
        or support_delta >= IMPROVEMENT_MARGIN
        or leakage_reduction > 0
    )
    advance = (
        leakage_nonincrease
        and sensitivity_no_harm
        and support_no_harm
        and material_improvement
    )
    return {
        "config": CANDIDATE,
        "comparator": COMPARATOR,
        "mean_normal_wes_anchor_sensitivity": sensitivity[CANDIDATE],
        "delta_mean_sensitivity_vs_comparator": sensitivity_delta,
        "sensitivity_no_harm": sensitivity_no_harm,
        "mean_selected_anchor_normal_wes_support": support[CANDIDATE],
        "delta_mean_support_vs_comparator": support_delta,
        "support_no_harm": support_no_harm,
        "P4_somatic_leakage_n": leakage[CANDIDATE]["P4"],
        "P6_somatic_leakage_n": leakage[CANDIDATE]["P6"],
        "P4_leakage_nonincrease": (
            leakage[CANDIDATE]["P4"] <= leakage[COMPARATOR]["P4"]
        ),
        "P6_leakage_nonincrease": (
            leakage[CANDIDATE]["P6"] <= leakage[COMPARATOR]["P6"]
        ),
        "leakage_nonincrease_both": leakage_nonincrease,
        "total_leakage_reduction": leakage_reduction,
        "material_improvement": material_improvement,
        "advance_to_replication": advance,
    }


def write_sums(paths: Sequence[Path], root: Path, output: Path) -> None:
    rows = []
    for path in sorted(set(Path(path).resolve() for path in paths), key=str):
        try:
            label: Path | str = path.relative_to(root.resolve())
        except ValueError:
            label = path
        rows.append(f"{sha256(path)}  {label}")
    output.write_text("\n".join(rows) + "\n")


def main() -> int:
    candidate_dir = (
        PROJECT / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factorial-experiment-root", type=Path, required=True)
    parser.add_argument("--maxlr-experiment-root", type=Path, required=True)
    parser.add_argument("--prescreen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contracts", type=Path, default=candidate_dir / "contracts.py")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=PROJECT / "scripts/postanalyze/beagle_configuration_benchmark.py",
    )
    parser.add_argument(
        "--protocol", type=Path, default=candidate_dir / "BENCHMARK_PROTOCOL_MAXLR.md"
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output_dir}")
    for path in (args.contracts, args.benchmark, args.protocol):
        require_file(path, "frozen maxlr assessment dependency")

    prescreen_paths = validate_prescreen(args.prescreen_root)
    contracts = load_module(args.contracts, "_beagle_v23_maxlr_contracts")
    benchmark = load_module(args.benchmark, "_beagle_v23_maxlr_benchmark")
    metrics, summary, run_records, run_and_truth_paths = validate_and_evaluate(
        factorial_root=args.factorial_experiment_root,
        maxlr_root=args.maxlr_experiment_root,
        contracts=contracts,
        benchmark=benchmark,
    )
    result = assess(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = args.output_dir / "matched_configuration_metrics_long.tsv"
    summary_path = args.output_dir / "matched_configuration_summary.tsv"
    gate_path = args.output_dir / "maxlr_gate_result.tsv"
    provenance_path = args.output_dir / "PROVENANCE.json"
    metrics.to_csv(metrics_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    pd.DataFrame([result]).to_csv(gate_path, sep="\t", index=False)
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_maxlr_extension_selection_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "status": "replication_decision_frozen",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "development_set_evaluation": True,
                "final_default_selected": False,
                "comparator": COMPARATOR,
                "candidate": CANDIDATE,
                "advance_to_replication": result["advance_to_replication"],
                "replicate_configurations": (
                    [CANDIDATE] if result["advance_to_replication"] else []
                ),
                "gate_thresholds": {
                    "sensitivity_no_harm_margin": SENSITIVITY_MARGIN,
                    "support_no_harm_margin": SUPPORT_MARGIN,
                    "material_improvement_margin": IMPROVEMENT_MARGIN,
                    "per_sample_somatic_leakage_nonincrease": True,
                },
                "assessor": identity(Path(__file__)),
                "integrity_module": identity(Path(INTEGRITY.__file__)),
                "contracts": identity(args.contracts),
                "benchmark": identity(args.benchmark),
                "protocol": identity(args.protocol),
                "prescreen_inputs": [identity(path) for path in prescreen_paths],
                "truth_inputs": {
                    name: identity(path) for name, path in TRUTH_INPUTS.items()
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
        args.contracts,
        args.benchmark,
        args.protocol,
        *prescreen_paths,
        *run_and_truth_paths,
    ]
    code_sums = args.output_dir / "CODE_AND_INPUT_SHA256SUMS.txt"
    write_sums(code_inputs, PROJECT, code_sums)
    output_sums = args.output_dir / "OUTPUT_SHA256SUMS.txt"
    write_sums(
        (metrics_path, summary_path, gate_path, provenance_path, code_sums),
        args.output_dir,
        output_sums,
    )
    (args.output_dir / "RUN_COMPLETE").write_text("complete\n")
    replicate = [CANDIDATE] if result["advance_to_replication"] else []
    (args.output_dir / "RUN_COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "sparcal_beagle_v2_3_maxlr_extension_completion_v1",
                "status": "complete",
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "replicate_configurations": replicate,
                "output_sha256sums": identity(output_sums),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(pd.DataFrame([result]).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
