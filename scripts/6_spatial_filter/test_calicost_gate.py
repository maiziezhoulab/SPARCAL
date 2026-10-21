#!/usr/bin/env python3
"""Synthetic acceptance tests for the CalicoST gate and corrected theta."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from calicost_gate import evaluate_gate, section_rows
from run_spatial_snv_filter_enhanced import EnhancedSpatialSNVFilter


def cnv_frame(states, split=False):
    if split:
        frame = pd.DataFrame({"CHR": ["1", "1"], "START": [1, 51], "END": [50, 100]})
    else:
        frame = pd.DataFrame({"CHR": ["1"], "START": [1], "END": [100]})
    for clone, (a, b) in states.items():
        frame[f"clone{clone} A"] = a if isinstance(a, list) else [a] * len(frame)
        frame[f"clone{clone} B"] = b if isinstance(b, list) else [b] * len(frame)
    return frame


def gate_inputs(labels, states, purity=None, suffix="S1", umi=None):
    barcodes = [f"BC{i:03d}" for i in range(len(labels))]
    clone = pd.DataFrame({"BARCODES": [f"{bc}_{suffix}" for bc in barcodes],
                          "clone_label": labels})
    n_clones = max(max(labels), max(states)) + 1
    posterior = np.full((len(labels), n_clones), 0.01)
    posterior[np.arange(len(labels)), labels] = 0.99
    if umi is None:
        umi = np.tile([100, 200, 150, 250], math.ceil(len(labels) / 4))[:len(labels)]
    umi_series = pd.Series(umi, index=barcodes)
    purity_frame = None
    if purity is not None:
        purity_frame = pd.DataFrame({"BARCODES": [f"{bc}_{suffix}" for bc in barcodes],
                                     "Tumor": purity})
    return clone, cnv_frame(states), posterior, purity_frame, umi_series


def permissive_auc_thresholds():
    return {"min_umi_auc": 0.0, "max_umi_auc": 1.0}


def test_suffix_selection_and_joint_collision_error():
    frame = pd.DataFrame({"BARCODES": ["AA_P61", "AA_P62"], "clone_label": [0, 1]})
    selected, mask = section_rows(frame, "P62")
    assert selected["BARCODES"].tolist() == ["AA"]
    assert mask.tolist() == [False, True]
    with pytest.raises(ValueError, match="more than one sample suffix"):
        section_rows(frame, None)
    with pytest.raises(ValueError, match="duplicate barcodes"):
        section_rows(pd.DataFrame({"BARCODES": ["AA_P61", "AA_P61"]}), "P61")


def test_explicit_none_accepts_only_unsuffixed_single_section_labels():
    frame = pd.DataFrame({"BARCODES": ["AA-1", "BB-1"], "clone_label": [0, 1]})
    selected, mask = section_rows(frame, "NONE")
    assert selected["BARCODES"].tolist() == ["AA-1", "BB-1"]
    assert mask.tolist() == [True, True]
    with pytest.raises(ValueError, match="requires unsuffixed"):
        section_rows(pd.DataFrame({"BARCODES": ["AA-1_SAMPLE"]}), "NONE")


def test_reference_clone_is_selected_by_profile_not_label_zero():
    labels = [0] * 10 + [5] * 12
    inputs = gate_inputs(labels, {0: (2, 1), 5: (1, 1)})
    result = evaluate_gate(*inputs[:4], "S1", inputs[4], permissive_auc_thresholds())
    assert result["cnv_pass"]
    assert result["reference_clone"] == 5
    assert result["tumor_clones"] == [0]


def test_small_coverage_predicted_clone_is_dropped_and_unassigned():
    labels = [0] * 20 + [1] + [2] * 20
    umi = [100] * 20 + [100000] + [110] * 20
    inputs = gate_inputs(labels, {0: (1, 1), 1: (1, 2), 2: (2, 1)}, umi=umi)
    result = evaluate_gate(*inputs[:4], "S1", inputs[4])
    reasons = result["dropped_clones"]["1"]
    assert "spot_fraction_below_min" in reasons
    assert "umi_auc_outside_range" in reasons
    assert result["effective_labels"]["BC020"] is None


def test_purity_gate_failure_modes():
    labels = [0] * 12 + [1] * 12
    states = {0: (1, 1), 1: (2, 1)}
    # P1: fewer than half finite.
    purity = [np.nan] * 13 + list(np.linspace(0.2, 0.9, 11))
    inputs = gate_inputs(labels, states, purity=purity)
    result = evaluate_gate(*inputs[:4], "S1", inputs[4], permissive_auc_thresholds())
    assert not result["purity_checks"]["P1_finite_fraction"]
    assert not result["purity_pass"]
    # P2: constant purity has zero IQR and nothing strictly above its minimum.
    inputs = gate_inputs(labels, states, purity=[0.4] * 24)
    result = evaluate_gate(*inputs[:4], "S1", inputs[4], permissive_auc_thresholds())
    assert not result["purity_checks"]["P2_dynamic_range"]
    # P3: purity is almost perfectly explained by log UMI.
    umi = np.arange(1, 25) * 100
    purity = np.arange(1, 25) / 25
    inputs = gate_inputs(labels, states, purity=purity, umi=umi)
    result = evaluate_gate(*inputs[:4], "S1", inputs[4], permissive_auc_thresholds())
    assert not result["purity_checks"]["P3_umi_independence"]
    # P4: dynamic, UMI-independent purity with reversed clone discrimination.
    purity = list(np.tile([0.8, 0.9, 0.7], 4)) + list(np.tile([0.1, 0.2, 0.3], 4))
    inputs = gate_inputs(labels, states, purity=purity)
    result = evaluate_gate(*inputs[:4], "S1", inputs[4], permissive_auc_thresholds())
    assert result["cnv_pass"]
    assert not result["purity_checks"]["P4_clone_discrimination"]
    assert not result["purity_pass"]


def test_gate_on_requires_posterior_and_umi_arguments(tmp_path):
    with pytest.raises(ValueError, match="--clone_posterior.*--spot_umi_tsv"):
        EnhancedSpatialSNVFilter(
            dataset="p6_tumor", section_id="1", quality_filter="baseQ0mapQ0",
            tumor_purity_file="purity.tsv", clone_labels_file="clones.tsv",
            cnv_segments_file="cnv.tsv", calicost_sample_id="P61",
            output_dir=str(tmp_path / "unused"),
        )


def theta_filter(reference=0, tumors=(2,)):
    obj = object.__new__(EnhancedSpatialSNVFilter)
    obj.use_clone_cnv = True
    obj.reference_clone = reference
    obj.cancer_clones = list(tumors)
    obj.alt_presence_min_count = 1
    obj.clone_labels = {}
    obj.snv_depth_counts = defaultdict(dict)
    obj.snv_alt_counts = defaultdict(dict)
    return obj


def test_mh_theta_enrichment_neutral_and_sparse():
    obj = theta_filter()
    enriched = "1_10_A_T"
    neutral = "1_20_A_T"
    sparse = "1_30_A_T"
    for index in range(8):
        ref = f"R{index}"
        tumor = f"T{index}"
        obj.clone_labels[ref] = 0
        obj.clone_labels[tumor] = 2
        depth = [1, 2, 3, 5][index % 4]
        for variant in (enriched, neutral):
            obj.snv_depth_counts[variant][ref] = depth
            obj.snv_depth_counts[variant][tumor] = depth
        obj.snv_alt_counts[enriched][tumor] = 1 if index < 7 else 0
        obj.snv_alt_counts[enriched][ref] = 1 if index == 0 else 0
        obj.snv_alt_counts[neutral][tumor] = index % 2
        obj.snv_alt_counts[neutral][ref] = index % 2
    for barcode, clone in (("R0", 0), ("R1", 0), ("T0", 2), ("T1", 2), ("T2", 2)):
        obj.snv_depth_counts[sparse][barcode] = 2
        obj.snv_alt_counts[sparse][barcode] = 1
    assert obj.calculate_clone_enrichment_theta(enriched) > obj.calculate_clone_enrichment_theta(neutral)
    assert np.isnan(obj.calculate_clone_enrichment_theta(sparse))


def test_nan_theta_never_receives_a_vote():
    obj = theta_filter()
    obj.evidence_mode = "full"
    obj.mechanistic_vaf_mode = "off"
    obj.has_informative_purity = False
    obj.kept_variants_pool = set()
    variants = [f"1_{position}_A_T" for position in range(1, 11)]
    obj.spot_snvs = {"spot": set(variants)}
    obj.snv_race = {variant: "denovo" for variant in variants}
    obj.variant_scores = {}
    for name in ("germline_variants", "somatic_variants", "ambiguous_variants",
                 "germline_defined", "germline_denovo", "germline_kept",
                 "somatic_defined", "somatic_denovo", "ambiguous_defined", "ambiguous_denovo"):
        setattr(obj, name, set())
    obj.filtered_spot_snvs_germline = defaultdict(set)
    obj.filtered_spot_snvs_somatic = defaultdict(set)
    obj.filtered_spot_snvs_ambiguous = defaultdict(set)
    obj.calculate_spatial_uniformity_score = lambda variant: 0.0
    obj.calculate_global_prevalence_score = lambda variant: 0.0
    obj.calculate_spatial_clustering_score = lambda variant: float(variant.split("_")[1])
    nan_variant = variants[-1]
    obj.calculate_theta_score = lambda variant: np.nan if variant == nan_variant else float(11 - int(variant.split("_")[1]))
    obj.classify_variants()
    # It has the largest zeta and receives that vote, but must not get a theta vote.
    assert obj.variant_scores[nan_variant]["votes"] == 1


def test_legacy_theta_route_preserves_legacy_score():
    obj = theta_filter(reference=5, tumors=(7,))
    obj.cnv_feature = "legacy"
    obj.cnv_segments = pd.DataFrame({"x": [1]})
    obj.find_cnv_segment = lambda variant: pd.Series({"clone7_A": 2, "clone7_B": 1})
    variant = "1_10_A_T"
    for index in range(5):
        barcode = f"T{index}"
        obj.clone_labels[barcode] = 7
        obj.snv_depth_counts[variant][barcode] = 2
        obj.snv_alt_counts[variant][barcode] = int(index < 2)
    assert obj.calculate_cnv_consistency_score(variant) == pytest.approx(0.4)
    assert obj.calculate_theta_score(variant) == pytest.approx(0.4)
