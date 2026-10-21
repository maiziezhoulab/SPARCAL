from __future__ import annotations

import gzip
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "scripts/postanalyze/audit_beagle_v2_3_pre_replication.py"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODULE = load(MODULE_PATH, "audit_beagle_v2_3_pre_replication")
SCREEN = load(
    ROOT / "scripts/postanalyze/assess_beagle_v2_3_screen.py",
    "_audit_test_screen",
)
MODELSCALE = load(
    ROOT / "scripts/postanalyze/assess_beagle_v2_3_modelscale_extension.py",
    "_audit_test_modelscale",
)
MAXLR = load(
    ROOT / "scripts/postanalyze/assess_beagle_v2_3_maxlr_extension.py",
    "_audit_test_maxlr",
)


def metric_rows(overrides=None) -> pd.DataFrame:
    overrides = overrides or {}
    defaults = {
        "normal_wes_anchor_sensitivity": (960, 1000),
        "selected_anchor_normal_wes_support": (940, 1000),
        "wes_somatic_anchor_leakage": (5, 50),
    }
    rows = []
    for config in MODULE.SETTINGS:
        for sample in MODULE.SAMPLES:
            for metric in MODULE.GATE_METRICS:
                numerator, denominator = overrides.get(
                    (config, sample, metric), defaults[metric]
                )
                rows.append({
                    "config": config,
                    "sample": sample,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": numerator / denominator,
                    "ci95_low": 0.0,
                    "ci95_high": 1.0,
                })
    return pd.DataFrame(rows)


def full_metric_rows() -> pd.DataFrame:
    rows = []
    for config in MODULE.SETTINGS:
        for sample in MODULE.SAMPLES:
            for metric in MODULE.ALL_METRICS:
                if metric in MODULE.GATE_METRICS:
                    numerator, denominator = {
                        "normal_wes_anchor_sensitivity": (96, 100),
                        "selected_anchor_normal_wes_support": (94, 100),
                        "wes_somatic_anchor_leakage": (5, 50),
                    }[metric]
                    value = numerator / denominator
                else:
                    numerator, denominator, value = 0, 0, float("nan")
                rows.append({
                    "config": config,
                    "sample": sample,
                    "metric": metric,
                    "numerator": numerator,
                    "denominator": denominator,
                    "value": value,
                    "ci95_low": 0.0 if denominator else float("nan"),
                    "ci95_high": 1.0 if denominator else float("nan"),
                })
    return pd.DataFrame(rows)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_audit_bundle(root: Path) -> list[str]:
    root.mkdir()
    metrics = metric_rows()
    gates = MODULE.compute_gate_tables(metrics)
    metrics.to_csv(
        root / "recomputed_gate_metrics_long.tsv", sep="\t", index=False
    )
    for name, frame in zip(
        (
            "factorial_gate_results_recomputed.tsv",
            "modelscale4_gate_result_recomputed.tsv",
            "maxlr_gate_result_recomputed.tsv",
        ),
        gates,
    ):
        frame.to_csv(root / name, sep="\t", index=False)
    admitted = ["ms2_niter0", "ms2_niter5", "ms0p8_niter0"]
    write_json(root / "ADMITTED_CONFIGURATIONS.json", {
        "schema": (
            "sparcal_beagle_v2_3_pre_replication_"
            "admitted_configurations_v1"
        ),
        "status": "screen_decisions_independently_recomputed",
        "screen_seed": MODULE.SEED,
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "admitted_base_configurations": admitted,
    })
    write_json(root / "PROVENANCE.json", {
        "schema": "sparcal_beagle_v2_3_pre_replication_audit_v1",
        "status": "selector_decisions_verified",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "development_set_evaluation": True,
        "screen_seed": MODULE.SEED,
        "samples": list(MODULE.SAMPLES),
        "expected_configurations": list(MODULE.SETTINGS),
        "admitted_base_configurations": admitted,
        "audit": MODULE.identity(MODULE_PATH),
    })
    MODULE.write_ledger(
        [MODULE_PATH], root / "CODE_AND_INPUT_SHA256SUMS.txt"
    )
    MODULE.write_ledger(
        [root / name for name in MODULE.AUDIT_MEMBER_NAMES],
        root / "OUTPUT_SHA256SUMS.txt",
        root,
    )
    (root / "RUN_COMPLETE").write_text("complete\n")
    write_json(root / "RUN_COMPLETE.json", {
        "schema": (
            "sparcal_beagle_v2_3_pre_replication_audit_completion_v1"
        ),
        "status": "complete",
        "production_inputs_include_wes": False,
        "evaluation_uses_wes": True,
        "screen_seed": MODULE.SEED,
        "admitted_base_configurations": admitted,
        "admitted_configurations": MODULE.identity(
            root / "ADMITTED_CONFIGURATIONS.json"
        ),
        "output_sha256sums": MODULE.identity(
            root / "OUTPUT_SHA256SUMS.txt"
        ),
    })
    return admitted


class PreReplicationAuditTests(unittest.TestCase):
    def test_gate_recomputation_matches_all_three_frozen_selectors(self):
        metrics = metric_rows({
            (
                f"ms0p8_niter5_seed{MODULE.SEED}",
                "P4",
                "normal_wes_anchor_sensitivity",
            ): (963, 1000),
            (
                f"ms0p8_niter5_seed{MODULE.SEED}",
                "P6",
                "normal_wes_anchor_sensitivity",
            ): (963, 1000),
            (MODULE.MODELSCALE, "P4", "wes_somatic_anchor_leakage"): (4, 50),
            (MODULE.MAXLR, "P6", "wes_somatic_anchor_leakage"): (4, 50),
        })
        factorial, modelscale, maxlr = MODULE.compute_gate_tables(metrics)

        screen_metrics = metrics.loc[
            metrics["config"].isin(MODULE.FACTORIAL)
        ]
        expected_factorial = SCREEN.assess_screen(screen_metrics)
        MODULE.exact_gate_compare(
            factorial, expected_factorial, "factorial equivalence"
        )

        comparator = metrics.loc[metrics["config"] == MODULE.COMPARATOR]
        extension = metrics.loc[metrics["config"] == MODULE.MODELSCALE]
        expected_modelscale = pd.DataFrame([
            MODELSCALE.assess_extension(comparator, extension)
        ])
        MODULE.exact_gate_compare(
            modelscale, expected_modelscale, "modelscale equivalence"
        )

        maxlr_metrics = metrics.loc[
            metrics["config"].isin((MODULE.COMPARATOR, MODULE.MAXLR))
        ]
        expected_maxlr = pd.DataFrame([MAXLR.assess(maxlr_metrics)])
        MODULE.exact_gate_compare(maxlr, expected_maxlr, "maxlr equivalence")

    def test_full_frozen_metric_grid_is_exact(self):
        full = full_metric_rows()
        validated = MODULE.validate_metric_grid(
            full, tuple(MODULE.SETTINGS), MODULE.ALL_METRICS
        )
        self.assertEqual(len(validated), len(full))
        with self.assertRaisesRegex(ValueError, "grid is not exact"):
            MODULE.validate_metric_grid(
                full.iloc[:-1],
                tuple(MODULE.SETTINGS),
                MODULE.ALL_METRICS,
            )

    def test_metric_grid_is_exact_and_counts_are_bounded(self):
        base = metric_rows()
        with self.assertRaisesRegex(ValueError, "grid is not exact"):
            MODULE.validate_metric_grid(base.iloc[:-1], tuple(MODULE.SETTINGS))
        with self.assertRaisesRegex(ValueError, "grid is not exact"):
            MODULE.validate_metric_grid(
                pd.concat([base, base.iloc[[0]]], ignore_index=True),
                tuple(MODULE.SETTINGS),
            )
        for numerator, denominator in ((-1, 50), (51, 50), (0, 0)):
            changed = base.copy()
            changed.loc[0, ["numerator", "denominator"]] = [
                numerator,
                denominator,
            ]
            changed.loc[0, "value"] = 0
            with self.subTest(numerator=numerator, denominator=denominator):
                with self.assertRaisesRegex(ValueError, "invalid counts"):
                    MODULE.validate_metric_grid(
                        changed, tuple(MODULE.SETTINGS)
                    )

    def test_metric_values_must_exactly_match_finite_count_ratios(self):
        for bad_value in (0.5, float("nan"), float("inf")):
            changed = metric_rows()
            changed.loc[0, "value"] = bad_value
            with self.subTest(value=bad_value):
                with self.assertRaisesRegex(ValueError, "do not equal counts"):
                    MODULE.validate_metric_grid(
                        changed, tuple(MODULE.SETTINGS)
                    )

    def test_metric_confidence_intervals_must_be_ordered_and_bounded(self):
        for column, bad_value in (
            ("ci95_low", -0.01),
            ("ci95_low", 0.99),
            ("ci95_high", 0.50),
            ("ci95_high", 1.01),
            ("ci95_high", float("nan")),
        ):
            changed = metric_rows()
            changed.loc[0, column] = bad_value
            with self.subTest(column=column, value=bad_value):
                with self.assertRaisesRegex(
                    ValueError, "confidence intervals are invalid"
                ):
                    MODULE.validate_metric_grid(
                        changed, tuple(MODULE.SETTINGS)
                    )

    def test_metric_confidence_intervals_allow_wilson_roundoff_at_zero(self):
        changed = metric_rows()
        changed.loc[0, ["numerator", "value", "ci95_low"]] = [
            0,
            0.0,
            2.7755575615628914e-17,
        ]
        validated = MODULE.validate_metric_grid(
            changed, tuple(MODULE.SETTINGS)
        )
        self.assertEqual(len(validated), len(changed))

    def test_exact_allele_universe_checks_raw_and_cross_arm_sets(self):
        raw = {("1", 1, "A", "C"), ("2", 2, "G", "T")}
        MODULE.require_exact_allele_universe(
            sample="P4",
            config=MODULE.COMPARATOR,
            raw=raw,
            audit=set(raw),
            prior_audit=set(raw),
        )
        with self.assertRaisesRegex(ValueError, "differs from raw VCF"):
            MODULE.require_exact_allele_universe(
                sample="P4",
                config=MODULE.MODELSCALE,
                raw=raw,
                audit={("1", 1, "A", "C")},
                prior_audit=None,
            )
        with self.assertRaisesRegex(ValueError, "differs across arms"):
            MODULE.require_exact_allele_universe(
                sample="P4",
                config=MODULE.MODELSCALE,
                raw=raw,
                audit=set(raw),
                prior_audit={("1", 1, "A", "C")},
            )

    def test_raw_vcf_reader_rejects_duplicate_or_multiallelic_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.vcf.gz"
            with gzip.open(valid, "wt") as handle:
                handle.write(
                    "##fileformat=VCFv4.2\n"
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
                    "1\t1\t.\tA\tC\t.\tPASS\t.\n"
                )
            self.assertEqual(
                MODULE.read_raw_alleles(valid), {("1", 1, "A", "C")}
            )
            duplicate = root / "duplicate.vcf.gz"
            with gzip.open(duplicate, "wt") as handle:
                handle.write(
                    "#CHROM\tPOS\tID\tREF\tALT\n"
                    "1\t1\t.\tA\tC\n"
                    "1\t1\t.\tA\tC\n"
                )
            with self.assertRaisesRegex(ValueError, "duplicate raw exact allele"):
                MODULE.read_raw_alleles(duplicate)
            multiallelic = root / "multiallelic.vcf.gz"
            with gzip.open(multiallelic, "wt") as handle:
                handle.write(
                    "#CHROM\tPOS\tID\tREF\tALT\n"
                    "1\t1\t.\tA\tC,G\n"
                )
            with self.assertRaisesRegex(ValueError, "not biallelic"):
                MODULE.read_raw_alleles(multiallelic)

    def test_exact_gate_comparison_rejects_changed_decision(self):
        factorial = MODULE.compute_gate_tables(metric_rows())[0]
        changed = factorial.copy()
        changed.loc[0, "advance_to_replication"] = False
        with self.assertRaisesRegex(ValueError, "differs from recomputation"):
            MODULE.exact_gate_compare(changed, factorial, "factorial")

    def test_gate_comparison_allows_only_float_roundoff(self):
        factorial = MODULE.compute_gate_tables(metric_rows())[0]
        rounded = factorial.copy()
        column = "mean_normal_wes_anchor_sensitivity"
        rounded.loc[0, column] += 1e-16
        MODULE.exact_gate_compare(rounded, factorial, "factorial")

        drifted = factorial.copy()
        drifted.loc[0, column] += 1e-9
        with self.assertRaisesRegex(ValueError, "differs from recomputation"):
            MODULE.exact_gate_compare(drifted, factorial, "factorial")

    def test_completed_bundle_verifies_and_checksum_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "audit"
            admitted = build_audit_bundle(root)
            self.assertEqual(MODULE.verify_audit_bundle(root), admitted)
            gate = root / "factorial_gate_results_recomputed.tsv"
            gate.write_text(gate.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.verify_audit_bundle(root)

    def test_bundle_rejects_unlisted_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "audit"
            build_audit_bundle(root)
            (root / "unexpected.txt").write_text("unexpected\n")
            with self.assertRaisesRegex(ValueError, "file set is not exact"):
                MODULE.verify_audit_bundle(root)

    def test_run_audit_refuses_overwrite_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                MODULE.run_audit(
                    factorial_experiment_root=Path("/does/not/exist"),
                    factorial_selection_root=Path("/does/not/exist"),
                    extension_experiment_root=Path("/does/not/exist"),
                    extension_selection_root=Path("/does/not/exist"),
                    maxlr_experiment_root=Path("/does/not/exist"),
                    maxlr_selection_root=Path("/does/not/exist"),
                    output_dir=output,
                )


if __name__ == "__main__":
    unittest.main()
