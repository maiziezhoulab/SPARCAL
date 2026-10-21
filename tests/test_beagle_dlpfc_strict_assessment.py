import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clustering_benchmark"
    / "beagle_pnonref_strict_noninferiority_2026_09_13"
    / "assess_strict_noninferiority.py"
)
SPEC = importlib.util.spec_from_file_location("strict_dlpfc_assessment", MODULE_PATH)
ASSESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSESS
SPEC.loader.exec_module(ASSESS)


class StrictDlpfcAssessmentTests(unittest.TestCase):
    def test_donor_summary_is_descriptive_range_without_bootstrap(self):
        deltas = pd.DataFrame(
            {
                "delta_ari": [0.1] * 4 + [0.2] * 4 + [0.3] * 4,
                "delta_nmi": [0.01] * 4 + [0.02] * 4 + [0.03] * 4,
                "donor": [ASSESS.DONOR_BY_SECTION[section] for section in ASSESS.SECTIONS],
            },
            index=ASSESS.SECTIONS,
        )
        summary = ASSESS.donor_mean_summary(deltas)
        self.assertEqual(list(summary.index), ["Br5292", "Br5595", "Br8100"])
        self.assertEqual(list(summary["delta_ari"]), [0.1, 0.2, 0.3])
        source = MODULE_PATH.read_text()
        self.assertNotIn("donor_block_bootstrap", source)
        self.assertNotIn("donor_block_bootstrap_delta_ari_ci95", source)

    def test_strict_run_identity_binds_same_root_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "strict"
            other = Path(tmp) / "other"
            root.mkdir()
            other.mkdir()
            manifest = root / "STRICT_INPUT_MANIFEST.json"
            manifest.write_bytes(b"pinned manifest\n")
            baseline, candidate, digest = ASSESS.strict_run_identity(root, root)
            self.assertEqual(baseline, root.resolve())
            self.assertEqual(candidate, root.resolve())
            self.assertEqual(digest, hashlib.sha256(manifest.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "same run root"):
                ASSESS.strict_run_identity(root, other)

    def test_material_section_drop_fails_even_when_overall_mean_is_positive(self):
        deltas = pd.DataFrame(
            {
                "delta_ari": [-0.051] + [0.010] * 11,
                "delta_nmi": [0.0] * 12,
                "donor": [ASSESS.DONOR_BY_SECTION[section] for section in ASSESS.SECTIONS],
            },
            index=ASSESS.SECTIONS,
        )
        self.assertGreater(float(deltas["delta_ari"].mean()), 0.0)
        donors = ASSESS.donor_mean_summary(deltas)
        gates = ASSESS.statistical_gates(deltas, donors)
        self.assertTrue(gates["mean_delta_ari_nonnegative"])
        self.assertFalse(gates["no_section_mean_ari_drop_gt_0.05"])

    def test_assessor_writes_bound_v2_result_and_donor_table(self):
        candidate_method = "beagle_pnonref090_allalleles"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            strict_root = root / "strict"
            assessment_root = root / "assessment"
            strict_root.mkdir()
            manifest = strict_root / "STRICT_INPUT_MANIFEST.json"
            manifest.write_text("{\"status\":\"prepared\"}\n")
            labels = pd.DataFrame(
                {
                    "original_clusters": ["1", "1", "2", "2"],
                    "mclust": [1, 1, 2, 2],
                },
                index=["a", "b", "c", "d"],
            )
            for section in ASSESS.SECTIONS:
                for method in ("legacy_any_panel", candidate_method):
                    for run in range(10):
                        run_root = (
                            strict_root / section / "clustering" / method / f"run{run}"
                        )
                        run_root.mkdir(parents=True)
                        labels.to_csv(run_root / "cluster_labels.csv")
                        if method == candidate_method:
                            (run_root / "run_metadata.json").write_text(json.dumps({
                                "modality": method,
                                "run": run,
                                "seed": run,
                                "requested_model": "EEE",
                                "selected_model": "EEE",
                                "fallback_used": False,
                                "effective_rank": 2,
                                "embedding_dimensions": 30,
                            }))
            old_argv = sys.argv
            try:
                sys.argv = [
                    "assess", "--baseline-root", str(strict_root),
                    "--candidate-root", str(strict_root),
                    "--candidate-method", candidate_method,
                    "--require-candidate-metadata",
                    "--output-dir", str(assessment_root),
                ]
                ASSESS.main()
            finally:
                sys.argv = old_argv
            result = json.loads(
                (assessment_root / "NONINFERIORITY_RESULT.json").read_text()
            )
            self.assertEqual(result["schema_version"], "2.0")
            self.assertEqual(result["protocol_version"], ASSESS.PROTOCOL_VERSION)
            self.assertEqual(result["baseline_root"], str(strict_root.resolve()))
            self.assertEqual(result["candidate_root"], str(strict_root.resolve()))
            self.assertEqual(
                result["strict_input_manifest_sha256"],
                hashlib.sha256(manifest.read_bytes()).hexdigest(),
            )
            self.assertTrue(result["accept_replacement"])
            self.assertEqual(
                result["donor_mean_delta_ari_range_descriptive"], [0.0, 0.0]
            )
            self.assertNotIn("donor_block_bootstrap_delta_ari_ci95", result)
            donors = pd.read_csv(
                assessment_root / "donor_paired_deltas.tsv", sep="\t"
            )
            self.assertEqual(
                list(donors["donor"]), ["Br5292", "Br5595", "Br8100"]
            )


if __name__ == "__main__":
    unittest.main()
