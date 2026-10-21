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
    / "build_beagle_route_matrix.py"
)
SPEC = importlib.util.spec_from_file_location("build_beagle_route_matrix", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BuildBeagleRouteMatrixTests(unittest.TestCase):
    def test_selects_columns_without_changing_values_or_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            audit = root / "audit.tsv.gz"
            output = root / "result.pkl"
            frame = pd.DataFrame(
                [[1, 0, 1], [0, 1, 1]],
                index=["spot_b", "spot_a"],
                columns=["1_20", "1_10", "2_5"],
                dtype="int8",
            )
            frame.to_pickle(source)
            pd.DataFrame(
                [
                    ["chr1", 10, "A", "G", "beagle_ref_highconf"],
                    ["chr1", 20, "C", "T", "beagle_nonref_highconf"],
                    ["chr2", 5, "G", "A", "beagle_nonref_highconf"],
                ],
                columns=["chrom", "pos", "ref", "alt", "route"],
            ).to_csv(audit, sep="\t", index=False)
            MODULE.build(source, audit, output, {"beagle_nonref_highconf"})
            observed = pd.read_pickle(output)
            pd.testing.assert_frame_equal(observed, frame.loc[:, ["1_20", "2_5"]])
            manifest = json.loads(output.with_suffix(".pkl.manifest.json").read_text())
            self.assertEqual(manifest["operation"], "column subset only")

    def test_selects_by_nonref_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            audit = root / "audit.tsv.gz"
            output = root / "result.pkl"
            pd.DataFrame([[1, 2, 3]], columns=["1_10", "1_20", "2_30"]).to_pickle(source)
            pd.DataFrame({
                "chrom": ["chr1", "chr1", "chr2"],
                "pos": [10, 20, 30],
                "ref": ["A", "C", "G"],
                "alt": ["G", "T", "A"],
                "route": ["beagle_uncertain", "beagle_nonref_highconf", "panel_absent"],
                "beagle_p_nonref": [0.95, 0.29 + 0.61, 0.89],
            }).to_csv(audit, sep="\t", index=False)

            manifest = MODULE.build(
                source, audit, output, min_p_nonref=0.90,
            )
            result = pd.read_pickle(output)
            self.assertEqual(list(result.columns), ["1_10", "1_20"])
            self.assertEqual(manifest["selection"], {
                "score": "beagle_p_nonref", "minimum": 0.90,
            })

    def test_mixed_multiallelic_decisions_require_explicit_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            audit = root / "audit.tsv.gz"
            pd.DataFrame([[1, 2]], columns=["1_10", "1_20"]).to_pickle(source)
            pd.DataFrame({
                "chrom": ["chr1", "chr1", "chr1"],
                "pos": [10, 10, 20],
                "ref": ["A", "A", "C"],
                "alt": ["G", "T", "G"],
                "route": ["beagle_nonref_highconf"] * 3,
                "beagle_p_nonref": [0.95, 0.40, 0.95],
            }).to_csv(audit, sep="\t", index=False)

            with self.assertRaisesRegex(ValueError, "mixed exact-allele decisions"):
                MODULE.build(source, audit, root / "error.pkl", min_p_nonref=0.90)
            all_manifest = MODULE.build(
                source, audit, root / "all.pkl", min_p_nonref=0.90,
                multiallelic_policy="all",
            )
            any_manifest = MODULE.build(
                source, audit, root / "any.pkl", min_p_nonref=0.90,
                multiallelic_policy="any",
            )
            self.assertEqual(list(pd.read_pickle(root / "all.pkl").columns), ["1_20"])
            self.assertEqual(list(pd.read_pickle(root / "any.pkl").columns), ["1_10", "1_20"])
            self.assertEqual(all_manifest["mixed_decision_multiallelic_loci"], 1)
            self.assertEqual(any_manifest["mixed_decision_loci_observed_in_source_matrix"], 1)

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "exists.pkl"
            output.touch()
            with self.assertRaises(FileExistsError):
                MODULE.build(Path("missing"), Path("missing"), output, {"x"})


if __name__ == "__main__":
    unittest.main()
