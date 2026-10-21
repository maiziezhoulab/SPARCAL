import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "clustering_benchmark"
    / "beagle_panel_present_compatibility_2026_09_13"
    / "build_panel_present_compatibility.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_panel_present_compatibility", MODULE_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BeaglePanelPresentCompatibilityTests(unittest.TestCase):
    def test_exact_dataframe_equality_records_all_dimensions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            candidate = root / "candidate.pkl"
            frame = pd.DataFrame(
                [[1, 0], [0, 1]],
                index=pd.Index(["b", "a"], name="barcode"),
                columns=pd.Index(["1_10", "2_20"], name="locus"),
                dtype="int8",
            )
            frame.to_pickle(source)
            frame.copy().to_pickle(candidate)
            result = MODULE.assert_exact_dataframe_equality(source, candidate)
            self.assertTrue(result["all_checks_passed"])
            self.assertTrue(result["index_equal"])
            self.assertTrue(result["columns_equal"])
            self.assertTrue(result["dtypes_equal"])
            self.assertTrue(result["values_equal"])
            self.assertEqual(result["source_shape"], [2, 2])
            self.assertEqual(result["source_nonzero"], 2)
            self.assertEqual(
                result["source_dataframe_sha256"], result["output_dataframe_sha256"],
            )

    def test_exact_dataframe_equality_rejects_dtype_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.pkl"
            candidate = root / "candidate.pkl"
            pd.DataFrame([[1]], columns=["1_10"], dtype="int8").to_pickle(source)
            pd.DataFrame([[1]], columns=["1_10"], dtype="int64").to_pickle(candidate)
            with self.assertRaises(AssertionError):
                MODULE.assert_exact_dataframe_equality(source, candidate)

    def test_prepare_refuses_existing_root_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "already-there"
            output.mkdir()
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                MODULE.prepare(root, output, root / "audits", root / "builder.py")

    def test_contract_routes_exclude_only_panel_absent(self):
        self.assertEqual(MODULE.ROUTES, {
            "beagle_nonref_highconf",
            "beagle_ref_highconf",
            "beagle_uncertain",
        })
        self.assertNotIn("panel_absent", MODULE.ROUTES)


if __name__ == "__main__":
    unittest.main()
