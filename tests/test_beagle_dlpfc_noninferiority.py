import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clustering_benchmark"
    / "beagle_pnonref_noninferiority_2026_09_12"
    / "assess_noninferiority.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beagle_pnonref_noninferiority", MODULE_PATH
)
ASSESS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ASSESS
SPEC.loader.exec_module(ASSESS)


class BeagleDlpfcNoninferiorityTests(unittest.TestCase):
    def test_candidate_metadata_contract_fails_closed(self):
        valid = {
            "modality": "candidate",
            "run": 2,
            "seed": 2,
            "requested_model": "EEE",
            "selected_model": "EEE",
            "fallback_used": False,
            "effective_rank": 9,
            "embedding_dimensions": 30,
        }
        path = Path("run_metadata.json")
        ASSESS.validate_candidate_metadata(valid, path, "candidate", 2)
        for key in valid:
            broken = valid.copy()
            broken.pop(key)
            with self.assertRaisesRegex(ValueError, f"missing {key}"):
                ASSESS.validate_candidate_metadata(broken, path, "candidate", 2)
        for key, value in (
            ("seed", 3), ("selected_model", "VVV"), ("fallback_used", True)
        ):
            broken = valid.copy()
            broken[key] = value
            with self.assertRaises(ValueError):
                ASSESS.validate_candidate_metadata(broken, path, "candidate", 2)
        for key, value in (("run", 2.0), ("fallback_used", 0)):
            broken = valid.copy()
            broken[key] = value
            with self.assertRaises(TypeError):
                ASSESS.validate_candidate_metadata(broken, path, "candidate", 2)

    def test_label_contract_rejects_barcode_or_truth_mismatch(self):
        baseline = pd.DataFrame(
            {"original_clusters": ["L1", "L2"], "mclust": [1, 2]},
            index=["a", "b"],
        )
        with self.assertRaisesRegex(ValueError, "barcode index mismatch"):
            ASSESS.validate_label_contract(
                baseline, baseline.iloc[::-1], "synthetic"
            )
        candidate = baseline.copy()
        candidate.loc["b", "original_clusters"] = "L1"
        with self.assertRaisesRegex(ValueError, "truth labels mismatch"):
            ASSESS.validate_label_contract(baseline, candidate, "synthetic")

    def test_donor_block_bootstrap_preserves_donor_means(self):
        values = pd.Series(
            [0.1] * 4 + [0.2] * 4 + [0.3] * 4,
            index=ASSESS.SECTIONS,
        )
        low, high = ASSESS.donor_block_bootstrap(values, n_boot=5000)
        self.assertAlmostEqual(low, 0.1)
        self.assertAlmostEqual(high, 0.3)

    def test_unmatched_seed_is_excluded_before_averaging(self):
        runs = pd.DataFrame([
            {
                "section": "151507", "method": "legacy_any_panel", "run": 0,
                "status": "ok", "ari": 0.10, "nmi": 0.20,
            },
            {
                "section": "151507", "method": "beagle_pnonref090", "run": 0,
                "status": "ok", "ari": 0.30, "nmi": 0.40,
            },
            {
                "section": "151507", "method": "beagle_pnonref090", "run": 1,
                "status": "ok", "ari": 0.99, "nmi": 0.99,
            },
            {
                "section": "151507", "method": "legacy_any_panel", "run": 1,
                "status": "missing",
            },
        ])

        run_deltas, section_deltas, paired_counts = ASSESS.paired_run_deltas(runs)

        self.assertEqual(len(run_deltas), 1)
        self.assertEqual(int(paired_counts.loc["151507"]), 1)
        self.assertAlmostEqual(section_deltas.loc["151507", "delta_ari"], 0.20)
        self.assertAlmostEqual(section_deltas.loc["151507", "delta_nmi"], 0.20)


if __name__ == "__main__":
    unittest.main()
