#!/usr/bin/env python3
"""Focused tests for production AD parsing and mechanistic VAF evidence."""

from __future__ import annotations

import unittest

import numpy as np

from mechanistic_vaf import (
    expected_germline_vaf,
    expected_somatic_vaf,
    parse_sample_counts,
    score_mechanistic_vaf,
)


class TestSampleCounts(unittest.TestCase):
    def test_alt_presence_comes_from_ad(self):
        counts = parse_sample_counts("GT:AD:DP", "0/1:7,3:10")
        self.assertIsNotNone(counts)
        self.assertEqual((counts.ref_count, counts.alt_count, counts.depth), (7, 3, 10))

    def test_reference_record_is_parseable_but_not_alt_positive(self):
        counts = parse_sample_counts("GT:AD:DP", "0/0:8,0:8")
        self.assertIsNotNone(counts)
        self.assertEqual(counts.alt_count, 0)

    def test_missing_or_multiallelic_ad_is_not_usable(self):
        self.assertIsNone(parse_sample_counts("GT:DP", "0/1:8"))
        self.assertIsNone(parse_sample_counts("GT:AD:DP", "1/2:2,3,4:9"))


class TestExpectedVaf(unittest.TestCase):
    def test_copy_number_changes_somatic_and_germline_expectations(self):
        purity = np.asarray([0.8])
        diploid = expected_somatic_vaf(
            purity, np.asarray([1.0]), np.asarray([1.0]), 1.0, np.asarray([1.0])
        )[0]
        gained = expected_somatic_vaf(
            purity, np.asarray([2.0]), np.asarray([1.0]), 1.0, np.asarray([1.0])
        )[0]
        germline_a = expected_germline_vaf(
            purity, np.asarray([2.0]), np.asarray([1.0]), "A"
        )[0]
        self.assertAlmostEqual(diploid, 0.4)
        self.assertAlmostEqual(gained, 0.8 / 2.8)
        self.assertAlmostEqual(germline_a, 1.8 / 2.8)

    def test_somatic_like_counts_rank_above_germline_like_counts(self):
        common = {
            "total_counts": [20, 20, 20, 20],
            "purities": [0.8, 0.8, 0.8, 0.8],
            "copy_a": [1, 1, 1, 1],
            "copy_b": [1, 1, 1, 1],
        }
        somatic = score_mechanistic_vaf(alt_counts=[8, 8, 8, 8], **common)
        germline = score_mechanistic_vaf(alt_counts=[10, 10, 10, 10], **common)
        self.assertGreater(somatic["log_bf"], germline["log_bf"])

    def test_missing_context_is_marginalized_and_shrunk_to_neutral(self):
        result = score_mechanistic_vaf(
            alt_counts=[1, 0, 2], total_counts=[3, 2, 4],
            purities=[np.nan, np.nan, np.nan],
            copy_a=[np.nan, np.nan, np.nan], copy_b=[np.nan, np.nan, np.nan],
        )
        self.assertTrue(np.isfinite(result["log_bf"]))
        self.assertEqual(result["context_reliability"], 0.0)
        self.assertAlmostEqual(result["score"], 0.5)
        self.assertEqual(result["cnv_scenario"], "missing")


if __name__ == "__main__":
    unittest.main()
