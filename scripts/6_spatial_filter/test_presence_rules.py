#!/usr/bin/env python3
"""Acceptance tests for the production G1 and reproducibility presence arms."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from run_spatial_snv_filter_enhanced import apply_presence_rule


def test_g1_keeps_covered_ref_spots_and_drops_zero_alt_locus():
    alt_locus = "1_10_A_T"
    zero_locus = "1_20_G_C"
    records = {
        ("ALT", alt_locus),
        ("REF", alt_locus),
        ("REF", zero_locus),
    }
    molecules = {
        ("ALT", alt_locus): 1,
        ("REF", alt_locus): 0,
        ("REF", zero_locus): 0,
    }
    assert apply_presence_rule(records, molecules, "G1") == {
        ("ALT", alt_locus),
        ("REF", alt_locus),
    }


def test_g0_and_g2_reproduce_2026_10_09_definitions():
    records = {("A", "1_10_A_T"), ("B", "1_10_A_T"), ("B", "1_20_G_C")}
    molecules = {pair: 0 for pair in records}
    molecules[("A", "1_10_A_T")] = 1
    assert apply_presence_rule(records, molecules, "G0") == records
    assert apply_presence_rule(records, molecules, "G2") == {("A", "1_10_A_T")}


def test_missing_or_stale_molecule_rows_fail_closed():
    records = {("A", "1_10_A_T")}
    with pytest.raises(ValueError, match="missing=1"):
        apply_presence_rule(records, {}, "G1")
