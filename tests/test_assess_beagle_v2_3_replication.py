import importlib.util
import itertools
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_v2_3_replication.py"
)
SPEC = importlib.util.spec_from_file_location(
    "assess_beagle_v2_3_replication", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def fixtures(
    *,
    alt_anchor_jaccard=0.995,
    alt_route_agreement=0.999,
    alt_pnonref_mae=0.003,
    alt_leakage_change=None,
    alt_sensitivity_n=965,
    alt_support_n=945,
):
    selected = ["ms2_niter0", "ms0p8_niter0"]
    metrics = []
    summary = []
    for base, seed, sample in itertools.product(
        selected, MODULE.SEEDS, MODULE.SAMPLES
    ):
        config = MODULE.run_config(base, seed)
        sensitivity_n = (
            960 if base == MODULE.COMPARATOR else alt_sensitivity_n
        )
        support_n = 940 if base == MODULE.COMPARATOR else alt_support_n
        leakage_n = 4 if sample == "P4" else 1
        if (
            base != MODULE.COMPARATOR
            and alt_leakage_change is not None
            and (seed, sample) == (20260917, "P4")
        ):
            leakage_n += alt_leakage_change
        for metric, numerator, denominator in (
            ("normal_wes_anchor_sensitivity", sensitivity_n, 1000),
            ("selected_anchor_normal_wes_support", support_n, 1000),
            (
                "wes_somatic_anchor_leakage",
                leakage_n,
                34 if sample == "P4" else 22,
            ),
        ):
            metrics.append({
                "config": config,
                "sample": sample,
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
            })
        summary.append({
            "config": config,
            "sample": sample,
            "beagle_runtime_seconds": (
                120.0 if base == MODULE.COMPARATOR else 100.0
            ),
        })

    pairwise = []
    for base, sample in itertools.product(selected, MODULE.SAMPLES):
        configs = [MODULE.run_config(base, seed) for seed in MODULE.SEEDS]
        for left, right in itertools.combinations(configs, 2):
            comparator = base == MODULE.COMPARATOR
            pairwise.append({
                "sample": sample,
                "config_a": left,
                "config_b": right,
                "raw_jaccard": 1.0,
                "anchor_jaccard": (
                    0.996 if comparator else alt_anchor_jaccard
                ),
                "route_agreement": (
                    0.9995 if comparator else alt_route_agreement
                ),
                "p_nonref_mae": (
                    0.002 if comparator else alt_pnonref_mae
                ),
            })
    return (
        selected,
        pd.DataFrame(metrics),
        pd.DataFrame(pairwise),
        pd.DataFrame(summary),
    )


class ReplicationAssessmentTests(unittest.TestCase):
    def test_pre_replication_audit_binding_is_exact(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit_root = Path(tmp) / "audit"
            audit_root.mkdir()
            (audit_root / "a.txt").write_text("a\n")
            (audit_root / "b.txt").write_text("b\n")
            paths = sorted(audit_root.iterdir(), key=lambda path: path.name)
            records = [MODULE.identity(path) for path in paths]
            selected = ["ms2_niter0", "ms0p8_niter0"]
            provenance = {
                "pre_replication_audit_root": str(audit_root),
                "pre_replication_audit_module": MODULE.identity(
                    Path(MODULE.PREAUDIT.__file__)
                ),
                "pre_replication_audit_inputs": records,
                "selection_inputs": list(records),
            }
            with mock.patch.object(
                MODULE.PREAUDIT,
                "verify_audit_bundle",
                return_value=selected,
            ):
                self.assertEqual(
                    MODULE.verify_pre_replication_audit_binding(
                        provenance, selected
                    ),
                    paths,
                )

                incomplete = dict(provenance)
                incomplete["pre_replication_audit_inputs"] = records[:-1]
                with self.assertRaisesRegex(ValueError, "identities are not exact"):
                    MODULE.verify_pre_replication_audit_binding(
                        incomplete, selected
                    )

                unbound = dict(provenance)
                unbound["selection_inputs"] = records[:-1]
                with self.assertRaisesRegex(ValueError, "complete audit"):
                    MODULE.verify_pre_replication_audit_binding(
                        unbound, selected
                    )

            with mock.patch.object(
                MODULE.PREAUDIT,
                "verify_audit_bundle",
                return_value=["ms2_niter0"],
            ):
                with self.assertRaisesRegex(ValueError, "admissions differ"):
                    MODULE.verify_pre_replication_audit_binding(
                        provenance, selected
                    )

    def test_replication_snapshot_is_exact_and_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            root = temp / "run"
            snapshot = root / "code_snapshot"
            plan = temp / "plan"
            plan.mkdir()
            for name in (
                "replication_tasks.tsv",
                "PROVENANCE.json",
                "OUTPUT_SHA256SUMS.txt",
                "RUN_COMPLETE.json",
            ):
                (plan / name).write_text(f"{name}\n")
            protocol_dir = (
                MODULE.PROJECT
                / "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13"
            )
            sources = {
                "scripts/2_beagle_filtering/run_beagle.py": MODULE.PROJECT
                / "scripts/2_beagle_filtering/run_beagle.py",
                "scripts/2_beagle_filtering/beagle_evidence_router.py": MODULE.PROJECT
                / "scripts/2_beagle_filtering/beagle_evidence_router.py",
                "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py": protocol_dir
                / "contracts.py",
                "scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py": protocol_dir
                / "run_beagle_v2_3.py",
                "run_beagle_v2_3_replication.sh": MODULE.PROJECT
                / "run_slurm/run_beagle_v2_3_replication.sh",
                "beagle_selection_integrity.py": Path(MODULE.INTEGRITY.__file__),
                "BENCHMARK_PROTOCOL.md": protocol_dir / "BENCHMARK_PROTOCOL.md",
                "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md": protocol_dir
                / "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md",
                "BENCHMARK_PROTOCOL_MAXLR.md": protocol_dir
                / "BENCHMARK_PROTOCOL_MAXLR.md",
                "BENCHMARK_PROTOCOL_REPLICATION.md": protocol_dir
                / "BENCHMARK_PROTOCOL_REPLICATION.md",
                "replication_tasks.tsv": plan / "replication_tasks.tsv",
                "PROVENANCE.json": plan / "PROVENANCE.json",
                "OUTPUT_SHA256SUMS.txt": plan / "OUTPUT_SHA256SUMS.txt",
                "RUN_COMPLETE.json": plan / "RUN_COMPLETE.json",
            }
            copied = []
            for relative, source in sources.items():
                destination = snapshot / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.append(destination)
            ledger = root / "CODE_SNAPSHOT_SHA256SUMS.txt"
            ledger.write_text("".join(
                f"{MODULE.sha256(path)}  {path.resolve()}\n"
                for path in sorted(copied)
            ))
            kwargs = {
                "root": root,
                "plan_root": plan,
                "protocol": protocol_dir / "BENCHMARK_PROTOCOL.md",
                "extension_protocol": protocol_dir
                / "BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md",
                "maxlr_protocol": protocol_dir / "BENCHMARK_PROTOCOL_MAXLR.md",
                "replication_protocol": protocol_dir
                / "BENCHMARK_PROTOCOL_REPLICATION.md",
            }
            verified = MODULE.validate_replication_snapshot(**kwargs)
            self.assertEqual(verified["artifact_count"], len(sources))
            (snapshot / "replication_tasks.tsv").write_text("mutated\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                MODULE.validate_replication_snapshot(**kwargs)

    def test_stable_higher_accuracy_alternative_wins_priority(self):
        gates, winner = MODULE.assess_replication(*fixtures())
        self.assertEqual(winner, "ms0p8_niter0")
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertTrue(row["eligible_as_base_default"])
        self.assertTrue(row["selected_base_default"])

    def test_absolute_stability_failure_falls_back_to_comparator(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_anchor_jaccard=0.989)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertFalse(row["minimum_anchor_jaccard_pass"])
        self.assertFalse(row["eligible_as_base_default"])

    def test_any_matched_seed_sample_leakage_increase_rejects_alternative(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_leakage_change=1)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertFalse(row["matched_seed_sample_leakage_nonincrease"])
        self.assertFalse(row["eligible_as_base_default"])

    def test_median_mae_excess_rejects_alternative(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_pnonref_mae=0.0041)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertFalse(row["median_p_nonref_mae_vs_comparator_pass"])

    def test_replicated_mean_sensitivity_harm_rejects_alternative(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_sensitivity_n=954, alt_support_n=945)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertLess(
            row["delta_mean_sensitivity_vs_comparator"],
            -MODULE.SENSITIVITY_MARGIN,
        )
        self.assertFalse(row["replicated_sensitivity_no_harm"])
        self.assertTrue(row["replicated_material_improvement"])
        self.assertFalse(row["eligible_as_base_default"])

    def test_replicated_mean_support_harm_rejects_alternative(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_sensitivity_n=965, alt_support_n=934)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertLess(
            row["delta_mean_support_vs_comparator"],
            -MODULE.SUPPORT_MARGIN,
        )
        self.assertFalse(row["replicated_support_no_harm"])
        self.assertTrue(row["replicated_material_improvement"])
        self.assertFalse(row["eligible_as_base_default"])

    def test_no_replicated_material_gain_falls_back_to_comparator(self):
        gates, winner = MODULE.assess_replication(
            *fixtures(alt_sensitivity_n=960, alt_support_n=940)
        )
        self.assertEqual(winner, MODULE.COMPARATOR)
        row = gates.set_index("base_config").loc["ms0p8_niter0"]
        self.assertTrue(row["replicated_sensitivity_no_harm"])
        self.assertTrue(row["replicated_support_no_harm"])
        self.assertFalse(row["replicated_material_improvement"])
        self.assertFalse(row["eligible_as_base_default"])

    def test_incomplete_pairwise_grid_is_rejected(self):
        selected, metrics, pairwise, summary = fixtures()
        with self.assertRaisesRegex(ValueError, "stability grid"):
            MODULE.assess_replication(
                selected, metrics, pairwise.iloc[:-1], summary
            )

    def test_nonidentical_raw_universe_is_rejected(self):
        selected, metrics, pairwise, summary = fixtures()
        pairwise.loc[pairwise.index[0], "raw_jaccard"] = 0.999
        with self.assertRaisesRegex(ValueError, "exact-allele universe"):
            MODULE.assess_replication(selected, metrics, pairwise, summary)

    def test_pairwise_fraction_out_of_bounds_is_rejected(self):
        selected, metrics, pairwise, summary = fixtures()
        pairwise.loc[pairwise.index[0], "route_agreement"] = 1.01
        with self.assertRaisesRegex(ValueError, r"outside \[0,1\]"):
            MODULE.assess_replication(selected, metrics, pairwise, summary)

    def test_negative_posterior_mae_is_rejected(self):
        selected, metrics, pairwise, summary = fixtures()
        pairwise.loc[pairwise.index[0], "p_nonref_mae"] = -0.001
        with self.assertRaisesRegex(ValueError, "negative"):
            MODULE.assess_replication(selected, metrics, pairwise, summary)

    def test_runtime_grid_and_values_are_fail_closed(self):
        selected, metrics, pairwise, summary = fixtures()
        with self.assertRaisesRegex(ValueError, "runtime summary grid"):
            MODULE.assess_replication(
                selected, metrics, pairwise, summary.iloc[:-1]
            )
        invalid = summary.copy()
        invalid.loc[invalid.index[0], "beagle_runtime_seconds"] = -1
        with self.assertRaisesRegex(ValueError, "invalid duration"):
            MODULE.assess_replication(selected, metrics, pairwise, invalid)


if __name__ == "__main__":
    unittest.main()
