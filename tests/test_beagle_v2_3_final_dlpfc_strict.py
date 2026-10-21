from __future__ import annotations

import importlib.util
import csv
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = (
    PROJECT / "clustering_benchmark"
    / "beagle_v2_3_final_strict_regression_2026_09_13"
)


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


PROVENANCE = load(PACKAGE / "strict_provenance.py", "final_strict_provenance_test")
sys.modules["strict_provenance"] = PROVENANCE
ASSESS = load(PACKAGE / "assess_strict_noninferiority.py", "final_strict_assess_test")
FINALIZE = load(PACKAGE / "finalize_strict_assessment.py", "final_strict_finalize_test")
PREPARE = load(PACKAGE / "prepare_strict_inputs.py", "final_strict_prepare_test")
VERIFY = load(PACKAGE / "verify_strict_inputs.py", "final_strict_verify_test")
sys.modules["verify_strict_inputs"] = VERIFY
VALIDATE = load(PACKAGE / "validate_strict_run.py", "final_strict_validate_test")


class FinalDlpfcStrictProjectionTests(unittest.TestCase):
    def baseline(self):
        return pd.DataFrame(
            [[1, 0, 2], [0, 1, 0]],
            index=pd.Index(["spot_b", "spot_a"], name="barcode"),
            columns=pd.Index(["1_10", "1_20", "1_30"], name="locus"),
            dtype="int8",
        )

    def test_complete_ordered_subset_is_valid_without_exact_equality(self):
        baseline = self.baseline()
        candidate = baseline.loc[:, ["1_10", "1_30"]].copy()
        result = PROVENANCE.validate_projection(baseline, candidate, "151507")
        self.assertTrue(result["all_checks_passed"])
        self.assertEqual(result["candidate_only_feature_count"], 0)
        self.assertEqual(result["legacy_only_feature_count"], 1)
        self.assertNotEqual(candidate.shape, baseline.shape)

    def test_candidate_only_feature_is_fatal(self):
        baseline = self.baseline()
        candidate = baseline.iloc[:, :1].copy()
        candidate["1_99"] = 1
        with self.assertRaisesRegex(ValueError, "candidate-only features"):
            PROVENANCE.validate_projection(baseline, candidate, "151507")

    def test_reordered_or_modified_projection_is_fatal(self):
        baseline = self.baseline()
        reordered = baseline.loc[:, ["1_30", "1_10"]].copy()
        with self.assertRaisesRegex(ValueError, "ordered complete subset"):
            PROVENANCE.validate_projection(baseline, reordered, "151507")
        modified = baseline.loc[:, ["1_10", "1_30"]].copy()
        modified.iloc[0, 0] = 0
        with self.assertRaisesRegex(ValueError, "rows/order/dtypes/values"):
            PROVENANCE.validate_projection(baseline, modified, "151507")

    def test_checksum_ledger_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            member = root / "member.txt"
            member.write_text("frozen\n")
            ledger = root / "ledger.txt"
            ledger.write_text(f"{PROVENANCE.sha256(member)}  member.txt\n")
            self.assertEqual(
                PROVENANCE.verify_ledger(ledger, root, expected=[member]),
                {member.resolve()},
            )
            member.write_text("drift\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                PROVENANCE.verify_ledger(ledger, root, expected=[member])

    def test_selected_audit_reconstructs_multiallelic_any_panel_loci(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.tsv.gz"
            defaults = {column: "." for column in PROVENANCE.AUDIT_COLUMNS}
            rows = []
            for alt, route in (
                ("C", "panel_absent"),
                ("G", "beagle_ref_highconf"),
                ("T", "beagle_uncertain"),
            ):
                rows.append({
                    **defaults, "schema": "sparcal_beagle_evidence_v2_2",
                    "chrom": "chr1", "pos": "10" if alt != "T" else "20",
                    "ref": "A", "alt": alt, "route": route,
                })
            with gzip.open(path, "wt") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=PROVENANCE.AUDIT_COLUMNS, delimiter="\t"
                )
                writer.writeheader()
                writer.writerows(rows)
            result = PROVENANCE.scan_panel_loci(path)
            self.assertEqual(result["panel_loci"], {"1_10", "1_20"})
            self.assertEqual(result["route_counts"]["panel_absent"], 1)
            self.assertEqual(result["exact_allele_count"], 3)


class FinalDlpfcStrictAssessmentTests(unittest.TestCase):
    def make_equal_run_labels(self, root: Path) -> None:
        (root / "STRICT_INPUT_MANIFEST.json").write_text("{}\n")
        frame = pd.DataFrame({
            "original_clusters": ["1", "1", "2", "2"],
            "mclust": [1, 1, 2, 2],
        }, index=["a", "b", "c", "d"])
        for section in ASSESS.load_validated_core().SECTIONS:
            for method in (ASSESS.BASELINE_METHOD, ASSESS.CANDIDATE_METHOD):
                for run in range(10):
                    run_root = root / section / "clustering" / method / f"run{run}"
                    run_root.mkdir(parents=True)
                    frame.to_csv(run_root / "cluster_labels.csv")
                    if method == ASSESS.CANDIDATE_METHOD:
                        (run_root / "run_metadata.json").write_text(json.dumps({
                            "modality": method, "run": run, "seed": run,
                            "requested_model": "EEE", "selected_model": "EEE",
                            "fallback_used": False, "effective_rank": 2,
                            "embedding_dimensions": 30,
                        }) + "\n")

    def test_exact_equal_results_still_assess_all_120_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "run"
            output = Path(tmp) / "assessment"
            root.mkdir()
            self.make_equal_run_labels(root)
            result = ASSESS.assess(root, output)
            metrics = pd.read_csv(output / "clustering_metrics_long.tsv", sep="\t")
            pairs = pd.read_csv(output / "seed_paired_deltas.tsv", sep="\t")
            self.assertEqual(len(metrics), 240)
            self.assertEqual(len(pairs), 120)
            self.assertEqual(result["expected_paired_run_count"], 120)
            self.assertTrue(result["mandatory_execution_even_if_inputs_exactly_equal"])
            self.assertTrue(result["gates"]["no_candidate_only_failures"])
            self.assertTrue(result["accept_replacement"])
            (output / "STRICT_RUN_VALIDATION.json").write_text(json.dumps({
                "status": "all_checks_passed",
                "protocol_version": FINALIZE.PROTOCOL_VERSION,
                "validated_sections": list(FINALIZE.SECTIONS),
                "candidate_only_failures": [],
                "mandatory_execution_even_if_exactly_equal": True,
                "root": str(root.resolve()),
                "strict_input_manifest_sha256": PROVENANCE.sha256(
                    root / "STRICT_INPUT_MANIFEST.json"
                ),
            }) + "\n")
            self.assertTrue(FINALIZE.finalize(output))
            self.assertEqual((output / "ACCEPTED").read_text(),
                             "all registered final-DLPFC gates passed\n")
            self.assertEqual((output / "ASSESSMENT_COMPLETE").read_text(), "complete\n")

    def test_assessor_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "existing"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                ASSESS.assess(root / "missing", output)


class FinalDlpfcScopedValidationTests(unittest.TestCase):
    def test_single_section_smoke_skips_other_section_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "STRICT_INPUT_MANIFEST.json").write_text(json.dumps({
                "candidate_semantic_contract": {
                    "materializer_manifest_sha256": "a" * 64,
                },
            }) + "\n")
            original_calls = []

            class FakeCore:
                def __init__(self):
                    self.records = [
                        {"kind": "global"},
                        {"kind": "selected", "section": "151508"},
                        {"kind": "other", "section": "151507"},
                    ]
                    self.verify_record = self.original_verify_record

                @staticmethod
                def validate_candidate_semantic_contract(manifest):
                    raise AssertionError("must be replaced")

                @staticmethod
                def validate_candidate_provenance_grid(manifest):
                    raise AssertionError("must be replaced")

                @staticmethod
                def original_verify_record(record, record_root, *args, **kwargs):
                    original_calls.append(record["kind"])

                def validate_root(self, record_root, sections=None):
                    for record in self.records:
                        self.verify_record(record, record_root, external=True)
                    return {
                        "schema_version": "1.0",
                        "status": "all_checks_passed",
                        "root": str(record_root),
                        "validated_sections": list(sections or ()),
                        "candidate_only_failures": [],
                        "strict_input_manifest_sha256": "b" * 64,
                    }

            fake = FakeCore()
            with (
                mock.patch.object(VALIDATE, "verify_prepared_root") as prepared,
                mock.patch.object(VALIDATE, "load_validated_core", return_value=fake),
            ):
                result = VALIDATE.validate_root(root, sections=("151508",))
            prepared.assert_called_once_with(root.resolve(), section="151508")
            self.assertEqual(original_calls, ["global", "selected"])
            self.assertEqual(result["validated_sections"], ["151508"])
            self.assertTrue(result["mandatory_execution_even_if_exactly_equal"])


class FinalDlpfcStrictFreezeTests(unittest.TestCase):
    def test_package_declares_full_grid_and_frozen_gates(self):
        clustering = (PACKAGE / "run_clustering.slurm").read_text()
        readme = (PACKAGE / "README.md").read_text()
        prepare = (PACKAGE / "prepare_strict_inputs.py").read_text()
        self.assertIn("#SBATCH --array=1", clustering)
        self.assertIn("--array=0,2-11%2", readme)
        self.assertIn(
            "151507 151508 151509 151510 151669 151670 "
            "151671 151672 151673 151674 151675 151676",
            clustering,
        )
        self.assertIn("--n_runs 10", clustering)
        self.assertIn("120 paired", readme)
        self.assertIn("must run even when", readme)
        self.assertIn('"mandatory_execution_even_if_exactly_equal": True', prepare)
        self.assertIn('"candidate_exact_dataframe_equality_required": False', prepare)
        self.assertEqual(FINALIZE.EXPECTED_GATES, {
            "mean_delta_ari_nonnegative",
            "minimum_donor_mean_delta_ari_above_minus_0.01",
            "no_section_mean_ari_drop_gt_0.05",
            "nmi_not_worse_by_more_than_0.01",
            "at_least_9_paired_runs_each",
            "no_candidate_only_failures",
            "candidate_run_success_not_lower_than_legacy",
            "candidate_used_requested_mclust_model_for_all_runs",
        })

    def test_old_package_is_not_referenced_as_candidate_input(self):
        joined = "\n".join(
            path.read_text()
            for path in PACKAGE.iterdir()
            if path.is_file()
        )
        self.assertNotIn(
            "beagle_dlpfc_panel_present_compatibility_20260913_v1", joined
        )
        self.assertIn("sparcal_beagle_v2_3_final_dlpfc_materialization_v1", joined)

    def test_validated_numerical_sources_match_accepted_snapshot_hashes(self):
        old = (
            PROJECT / "clustering_benchmark"
            / "beagle_panel_present_strict_regression_2026_09_13"
        )
        sources = {
            "SPARCAL_clustering.py": PROJECT / "clustering_benchmark" / "SPARCAL_clustering.py",
            "mclust_contract.py": PROJECT / "clustering_benchmark" / "mclust_contract.py",
            "assess_strict_noninferiority.py": old / "assess_strict_noninferiority.py",
            "validate_strict_run.py": old / "validate_strict_run.py",
            "record_runtime.py": old / "record_runtime.py",
            "baseline_provenance.json": old / "baseline_provenance.json",
        }
        self.assertEqual(
            {name: PREPARE.sha256(path) for name, path in sources.items()},
            PREPARE.VALIDATED_NUMERICAL_SHA256,
        )
        stagate = PROJECT / "DISCGATE_imputation" / "STAGATE_pyG"
        observed = {
            path.name: PREPARE.sha256(path) for path in stagate.glob("*.py")
        }
        self.assertEqual(observed, PREPARE.VALIDATED_STAGATE_SHA256)


if __name__ == "__main__":
    unittest.main()
