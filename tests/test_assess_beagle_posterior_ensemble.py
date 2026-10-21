import importlib.util
import itertools
import unittest
from pathlib import Path

import pandas as pd


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/postanalyze/assess_beagle_posterior_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location(
    "assess_beagle_posterior_ensemble", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def metric_rows(config, sensitivity_n, support_n, leakage):
    rows = []
    for sample in MODULE.SAMPLES:
        for metric, numerator, denominator in (
            ("normal_wes_anchor_sensitivity", sensitivity_n, 1000),
            ("selected_anchor_normal_wes_support", support_n, 1000),
            (
                "wes_somatic_anchor_leakage",
                leakage[sample],
                34 if sample == "P4" else 22,
            ),
        ):
            rows.append({
                "config": config,
                "sample": sample,
                "metric": metric,
                "numerator": numerator,
                "denominator": denominator,
                "value": numerator / denominator,
            })
    return rows


def fixtures(
    *,
    mean=(963, 943, {"P4": 4, "P6": 1}),
    median=(960, 940, {"P4": 4, "P6": 1}),
    loo_jaccard=0.996,
):
    base = "ms2_niter0"
    single = f"{base}_seed{MODULE.SINGLE_SEED}"
    base_metrics = pd.DataFrame(
        metric_rows(single, 960, 940, {"P4": 4, "P6": 1})
    )
    ensemble_metrics = pd.DataFrame(
        metric_rows("ensemble_mean", *mean)
        + metric_rows("ensemble_median", *median)
    )
    pairs = []
    for sample in MODULE.SAMPLES:
        for left, right in itertools.combinations(MODULE.LOO_CONFIGS, 2):
            pairs.append({
                "sample": sample,
                "config_a": left,
                "config_b": right,
                "anchor_jaccard": loo_jaccard,
            })
    return base, base_metrics, ensemble_metrics, pd.DataFrame(pairs)


class EnsembleAssessmentTests(unittest.TestCase):
    def test_material_mean_improvement_advances(self):
        gates, winner = MODULE.assess_ensembles(
            base=fixtures()[0],
            base_metrics=fixtures()[1],
            ensemble_metrics=fixtures()[2],
            pairwise=fixtures()[3],
        )
        self.assertEqual(winner, "ensemble_mean")
        mean = gates.set_index("method").loc["ensemble_mean"]
        self.assertTrue(mean["material_improvement"])
        self.assertTrue(mean["advance_over_single_seed"])
        self.assertEqual(mean["beagle_run_multiplier"], 3)

    def test_no_material_gain_rejects_threefold_cost(self):
        values = (960, 940, {"P4": 4, "P6": 1})
        base, base_metrics, ensemble_metrics, pairwise = fixtures(
            mean=values, median=values
        )
        gates, winner = MODULE.assess_ensembles(
            base=base,
            base_metrics=base_metrics,
            ensemble_metrics=ensemble_metrics,
            pairwise=pairwise,
        )
        self.assertEqual(winner, "single_seed")
        self.assertFalse(gates["advance_over_single_seed"].any())

    def test_leave_one_out_instability_rejects_ensemble(self):
        base, base_metrics, ensemble_metrics, pairwise = fixtures(
            loo_jaccard=0.994
        )
        gates, winner = MODULE.assess_ensembles(
            base=base,
            base_metrics=base_metrics,
            ensemble_metrics=ensemble_metrics,
            pairwise=pairwise,
        )
        self.assertEqual(winner, "single_seed")
        self.assertFalse(gates["leave_one_out_stability_pass"].any())

    def test_per_sample_leakage_increase_rejects_even_if_total_ties(self):
        base, base_metrics, ensemble_metrics, pairwise = fixtures(
            mean=(965, 945, {"P4": 5, "P6": 0})
        )
        gates, winner = MODULE.assess_ensembles(
            base=base,
            base_metrics=base_metrics,
            ensemble_metrics=ensemble_metrics,
            pairwise=pairwise,
        )
        mean = gates.set_index("method").loc["ensemble_mean"]
        self.assertFalse(mean["leakage_nonincrease_both"])
        self.assertFalse(mean["advance_over_single_seed"])
        self.assertEqual(winner, "single_seed")

    def test_tied_advancing_methods_prefer_mean(self):
        values = (963, 943, {"P4": 4, "P6": 1})
        base, base_metrics, ensemble_metrics, pairwise = fixtures(
            mean=values, median=values
        )
        _, winner = MODULE.assess_ensembles(
            base=base,
            base_metrics=base_metrics,
            ensemble_metrics=ensemble_metrics,
            pairwise=pairwise,
        )
        self.assertEqual(winner, "ensemble_mean")

    def test_mutated_metric_value_that_disagrees_with_counts_is_rejected(self):
        base, base_metrics, ensemble_metrics, pairwise = fixtures()
        ensemble_metrics = ensemble_metrics.copy()
        target = (
            ensemble_metrics["config"].eq("ensemble_mean")
            & ensemble_metrics["sample"].eq("P4")
            & ensemble_metrics["metric"].eq(
                "normal_wes_anchor_sensitivity"
            )
        )
        ensemble_metrics.loc[target, "value"] = 0.5
        with self.assertRaisesRegex(ValueError, "values do not equal counts"):
            MODULE.assess_ensembles(
                base=base,
                base_metrics=base_metrics,
                ensemble_metrics=ensemble_metrics,
                pairwise=pairwise,
            )


if __name__ == "__main__":
    unittest.main()
