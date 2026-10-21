import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_v2_3_screen.py"
)
SPEC = importlib.util.spec_from_file_location("assess_beagle_v2_3_screen", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def metric_rows(values):
    rows = []
    denominators = {
        "normal_wes_anchor_sensitivity": 10000,
        "selected_anchor_normal_wes_support": 10000,
        "wes_somatic_anchor_leakage": 50,
    }
    for config in MODULE.CONFIGURATIONS:
        for sample in MODULE.SAMPLES:
            for metric in MODULE.METRICS:
                value = values.get((config, sample, metric))
                if value is None:
                    value = {
                        "normal_wes_anchor_sensitivity": 0.96,
                        "selected_anchor_normal_wes_support": 0.94,
                        "wes_somatic_anchor_leakage": 0.10,
                    }[metric]
                denominator = denominators[metric]
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


class ScreenGateTests(unittest.TestCase):
    def test_comparator_and_no_harm_one_factor_arms_advance(self):
        result = MODULE.assess_screen(metric_rows({})).set_index("config")
        self.assertTrue(result.loc[MODULE.COMPARATOR, "advance_to_replication"])
        self.assertTrue(
            result.loc[f"ms2_niter5_seed{MODULE.SEED}", "advance_to_replication"]
        )
        self.assertTrue(
            result.loc[f"ms0p8_niter0_seed{MODULE.SEED}", "advance_to_replication"]
        )
        self.assertFalse(
            result.loc[f"ms0p8_niter5_seed{MODULE.SEED}", "advance_to_replication"]
        )

    def test_leakage_increase_blocks_even_a_one_factor_arm(self):
        config = f"ms2_niter5_seed{MODULE.SEED}"
        result = MODULE.assess_screen(metric_rows({
            (config, "P4", "wes_somatic_anchor_leakage"): 0.12,
        })).set_index("config")
        self.assertFalse(result.loc[config, "leakage_nonincrease_both"])
        self.assertFalse(result.loc[config, "advance_to_replication"])

    def test_materially_better_interaction_arm_advances(self):
        config = f"ms0p8_niter5_seed{MODULE.SEED}"
        result = MODULE.assess_screen(metric_rows({
            (config, "P4", "normal_wes_anchor_sensitivity"): 0.963,
            (config, "P6", "normal_wes_anchor_sensitivity"): 0.963,
        })).set_index("config")
        self.assertTrue(result.loc[config, "material_improvement"])
        self.assertTrue(result.loc[config, "advance_to_replication"])

    def test_support_harm_blocks_material_sensitivity_gain(self):
        config = f"ms0p8_niter5_seed{MODULE.SEED}"
        values = {
            (config, sample, "normal_wes_anchor_sensitivity"): 0.97
            for sample in MODULE.SAMPLES
        }
        values.update({
            (config, sample, "selected_anchor_normal_wes_support"): 0.93
            for sample in MODULE.SAMPLES
        })
        result = MODULE.assess_screen(metric_rows(values)).set_index("config")
        self.assertTrue(result.loc[config, "material_improvement"])
        self.assertFalse(result.loc[config, "support_no_harm"])
        self.assertFalse(result.loc[config, "advance_to_replication"])

    def test_somatic_denominator_mismatch_is_rejected(self):
        frame = metric_rows({})
        mask = (
            (frame["config"] == f"ms2_niter5_seed{MODULE.SEED}")
            & (frame["sample"] == "P4")
            & (frame["metric"] == "wes_somatic_anchor_leakage")
        )
        frame.loc[mask, "denominator"] = 51
        frame.loc[mask, "value"] = frame.loc[mask, "numerator"] / 51
        with self.assertRaisesRegex(ValueError, "somatic denominator differs"):
            MODULE.assess_screen(frame)


if __name__ == "__main__":
    unittest.main()
