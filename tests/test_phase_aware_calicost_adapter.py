#!/usr/bin/env python3
"""End-to-end contract test for the direct CalicoST phase-aware adapter."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


class TestCalicostAdapter(unittest.TestCase):
    def test_builds_finite_label_free_viewer_bundle(self):
        root = Path(__file__).parents[1]
        script = root / "scripts/viewer_phase_aware/build_from_calicost.py"
        with tempfile.TemporaryDirectory(prefix="phase_calicost_test_") as tmp:
            tmp = Path(tmp)
            parsed = tmp / "parsed_inputs"
            parsed.mkdir()
            bins = pd.DataFrame(
                {"CHR": [1, 1], "START": [1, 101], "END": [100, 200]}
            )
            bins.to_csv(parsed / "table_bininfo.csv.gz", sep="\t", index=False)
            counts = pd.DataFrame(
                {
                    "BARCODES": ["s1_SAMPLE", "s1_SAMPLE", "s2_SAMPLE", "s2_SAMPLE"],
                    "TOT": [12, 8, 10, 5],
                    "B": [1, 4, 2, 3],
                }
            )
            counts.to_csv(parsed / "table_rdrbaf.csv.gz", sep="\t", index=False)
            purity = pd.DataFrame(
                {"BARCODES": ["s1_SAMPLE", "s2_SAMPLE"], "Tumor": [0.8, np.nan]}
            )
            purity.to_csv(tmp / "purity.tsv", sep="\t", index=False)
            clones = pd.DataFrame(
                {"BARCODES": ["s1_SAMPLE", "s2_SAMPLE"], "clone_label": [1, 1]}
            )
            clones.to_csv(tmp / "clones.tsv", sep="\t", index=False)
            cnv = pd.DataFrame(
                {
                    "CHR": [1],
                    "START": [1],
                    "END": [100],
                    "clone1 A": [1],
                    "clone1 B": [0],
                }
            )
            cnv.to_csv(tmp / "cnv.tsv", sep="\t", index=False)
            output = tmp / "tumor_evidence.npz"
            subprocess.run(
                [
                    sys.executable,
                    str(script),
                    str(parsed),
                    str(tmp / "purity.tsv"),
                    str(tmp / "clones.tsv"),
                    str(tmp / "cnv.tsv"),
                    str(output),
                    "--confirm-label-free",
                    "--acknowledge-in-sample-calicost-evidence",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with np.load(output, allow_pickle=False) as bundle:
                self.assertEqual(str(bundle["schema"]), "SPARCAL_tumor_evidence_v1")
                self.assertEqual(bundle["barcodes"].tolist(), ["s1", "s2"])
                self.assertEqual(bundle["valid"].tolist(), [True, True])
                self.assertEqual(bundle["phase_available"].tolist(), [True, True])
                self.assertEqual(bundle["purity_available"].tolist(), [True, False])
                self.assertTrue(np.isfinite(bundle["tumor_probability"]).all())
                self.assertTrue(np.isfinite(bundle["purity_log_prior"]).all())
                provenance = json.loads(str(bundle["provenance"]))
                self.assertTrue(provenance["calicost_in_sample_descriptive_evidence"])
                self.assertFalse(provenance["independent_validation"])
                self.assertFalse(provenance["production_inputs_include_ground_truth"])


if __name__ == "__main__":
    unittest.main()
