import importlib.util
import sys
import unittest
from pathlib import Path
import json
import tempfile

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_v2_3_modelscale_extension.py"
)
SPEC = importlib.util.spec_from_file_location("beagle_ms4_extension", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def frame(config, sensitivity=0.96, support=0.94, leakage=0.10):
    rows = []
    for sample in MODULE.SAMPLES:
        for metric, value, denominator in (
            ("normal_wes_anchor_sensitivity", sensitivity, 10000),
            ("selected_anchor_normal_wes_support", support, 10000),
            ("wes_somatic_anchor_leakage", leakage, 50),
        ):
            numerator = round(value * denominator)
            rows.append({
                "config": config,
                "sample": sample,
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
            })
    return pd.DataFrame(rows)


class ModelscaleExtensionGateTests(unittest.TestCase):
    def test_equal_extension_does_not_advance(self):
        result = MODULE.assess_extension(
            frame(MODULE.COMPARATOR), frame(MODULE.EXTENSION)
        )
        self.assertTrue(result["leakage_nonincrease_both"])
        self.assertFalse(result["material_improvement"])
        self.assertFalse(result["advance_to_replication"])

    def test_sensitivity_gain_advances(self):
        result = MODULE.assess_extension(
            frame(MODULE.COMPARATOR),
            frame(MODULE.EXTENSION, sensitivity=0.963),
        )
        self.assertTrue(result["material_improvement"])
        self.assertTrue(result["advance_to_replication"])

    def test_leakage_increase_blocks(self):
        result = MODULE.assess_extension(
            frame(MODULE.COMPARATOR),
            frame(MODULE.EXTENSION, sensitivity=0.97, leakage=0.12),
        )
        self.assertFalse(result["leakage_nonincrease_both"])
        self.assertFalse(result["advance_to_replication"])

    def test_support_harm_blocks(self):
        result = MODULE.assess_extension(
            frame(MODULE.COMPARATOR),
            frame(MODULE.EXTENSION, sensitivity=0.97, support=0.93),
        )
        self.assertFalse(result["support_no_harm"])
        self.assertFalse(result["advance_to_replication"])

    def test_denominator_mismatch_is_rejected(self):
        extension = frame(MODULE.EXTENSION)
        mask = (
            (extension["sample"] == "P4")
            & (extension["metric"] == "wes_somatic_anchor_leakage")
        )
        extension.loc[mask, "denominator"] = 51
        extension.loc[mask, "value"] = extension.loc[mask, "numerator"] / 51
        with self.assertRaisesRegex(ValueError, "somatic denominator differs"):
            MODULE.assess_extension(frame(MODULE.COMPARATOR), extension)

    def test_comparator_loader_allows_other_registered_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = frame(MODULE.COMPARATOR)
            metrics.to_csv(root / "configuration_metrics_long.tsv", sep="\t", index=False)
            metrics.pivot(
                index=["config", "sample"], columns="metric", values="value"
            ).reset_index().to_csv(
                root / "configuration_summary.tsv", sep="\t", index=False
            )
            (root / "PROVENANCE.json").write_text(json.dumps({
                "production_inputs_include_wes": False,
                "evaluation_uses_wes": True,
                "anchor_score": "p_nonref",
                "min_anchor_probability": 0.90,
                "audits": [
                    {"config": MODULE.COMPARATOR, "sample": sample}
                    for sample in MODULE.SAMPLES
                ] + [{"config": "another_registered_arm", "sample": "P4"}],
            }))
            (root / "RUN_COMPLETE").write_text("complete\n")
            selected, _ = MODULE.load_metrics(root, MODULE.COMPARATOR)
            self.assertEqual(len(selected), len(MODULE.SAMPLES) * len(MODULE.METRICS))


if __name__ == "__main__":
    unittest.main()
