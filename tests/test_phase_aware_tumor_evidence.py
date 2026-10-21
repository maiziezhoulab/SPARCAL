#!/usr/bin/env python3
"""Focused tests for label-free phase/CNV tumor evidence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts/viewer_phase_aware"))
from phase_aware_evidence import (  # noqa: E402
    aggregate_phase_evidence,
    combine_tumor_evidence,
    expected_phased_het_fraction,
    phase_cnv_log_bayes_factor,
    purity_log_prior_shift,
)


class TestPhaseCnvLikelihood(unittest.TestCase):
    def test_expected_fraction_matches_mixture_formula(self):
        self.assertAlmostEqual(float(expected_phased_het_fraction(0.8, 1, 1)), 0.5)
        self.assertAlmostEqual(float(expected_phased_het_fraction(0.8, 1, 0)), 0.2 / 1.2)
        self.assertAlmostEqual(
            float(expected_phased_het_fraction(0.8, 1, 0, counted_allele="A")),
            1.0 / 1.2,
        )

    def test_expected_imbalance_favors_tumor_model(self):
        result = phase_cnv_log_bayes_factor(
            [1], [12], [0.8], [1], [0], orientation_probability=[1.0]
        )
        self.assertTrue(result["valid"][0])
        self.assertTrue(result["phase_available"][0])
        self.assertGreater(result["log_bf"][0], 0)

    def test_unknown_orientation_is_symmetric(self):
        low = phase_cnv_log_bayes_factor(
            [1], [12], [0.8], [1], [0], orientation_probability=[0.5]
        )["log_bf"][0]
        high = phase_cnv_log_bayes_factor(
            [11], [12], [0.8], [1], [0], orientation_probability=[0.5]
        )["log_bf"][0]
        self.assertAlmostEqual(low, high)

    def test_missing_context_is_marginalized(self):
        result = phase_cnv_log_bayes_factor(
            [2], [10], [np.nan], [np.nan], [np.nan]
        )
        self.assertTrue(np.isfinite(result["log_bf"][0]))
        self.assertFalse(result["purity_available"][0])
        self.assertFalse(result["cnv_available"][0])


class TestCombination(unittest.TestCase):
    def test_purity_maps_back_to_itself_without_other_evidence(self):
        result = purity_log_prior_shift([0.2, 0.8, np.nan], prior_probability=0.2)
        self.assertAlmostEqual(result["log_prior"][0], 0.0)
        self.assertGreater(result["log_prior"][1], 0.0)
        self.assertEqual(result["log_prior"][2], 0.0)

    def test_information_shrinks_to_prior(self):
        result = combine_tumor_evidence(
            [8, 8], [0, 0], [0, 0], [0, 0], [0, 0], [0, 1], prior_probability=0.2
        )
        self.assertAlmostEqual(result["tumor_probability"][0], 0.2)
        self.assertGreater(result["tumor_probability"][1], 0.9)

    def test_correlated_observations_are_capped(self):
        aggregate = aggregate_phase_evidence(
            ["s1"] * 100, [2.0] * 100, [1.0] * 100, effective_observation_cap=8
        )["s1"]
        self.assertAlmostEqual(aggregate["log_bf"], 16.0)
        self.assertEqual(aggregate["n_observations"], 100.0)


if __name__ == "__main__":
    unittest.main()
