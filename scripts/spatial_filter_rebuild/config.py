#!/usr/bin/env python3
"""Paths and immutable experiment settings for spatial-filter rebuild v1.

All model outputs live below ``data/spatial_filter_rebuild_2026-09-12``.
The current production filter and its viewer profiles are read-only baselines.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
STMUT = Path(
    "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/"
    "Datasets/ST_datasets/STmut_Data"
)
CALICOST = Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
BENCHMARKING = Path("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking")

RUN_LABEL = "multistate_v1_20260912"
OUTPUT_ROOT = ROOT / "data/spatial_filter_rebuild_2026-09-12" / RUN_LABEL
QUALITY_FILTER = "baseQ0mapQ0"

# These are hypotheses fixed before any WES/pathology evaluation.  ``baseline``
# is never regenerated; it points at the already completed production output.
MODEL_NAMES = (
    "molecular",
    "spatial",
    "calicost",
    "integrated",
    "consensus",
)
SELECTIONS = ("matched", "natural")


def _section(patient: str, rep: str, clone_dir: str, cnv_name: str) -> dict[str, Path | str]:
    canonical = ROOT / "data" / f"{patient}_tumor" / rep
    shadow = (
        ROOT / "data/spatial_filter_runs/ad_vaf_shadow_v1"
        / f"{patient}_tumor" / rep
    )
    ablation = (
        ROOT / "data/spatial_filter_runs/ad_coverage_no_calicost_v1"
        / f"{patient}_tumor" / rep
    )
    outs = (
        STMUT / f"{patient}_Visium" / f"spaceranger_align_rep{rep}_hg19"
        / f"{patient}_Tumor_output" / "outs"
    )
    cal = CALICOST / f"{patient}_sec{rep}"
    clone = cal / "calicost" / clone_dir
    return {
        "patient": patient,
        "rep": rep,
        "build": "GRCh37",
        "scores": shadow / "spatial_filter_purity" / QUALITY_FILTER / "all_variant_scores.txt",
        "ablation_scores": ablation / "spatial_filter_purity" / QUALITY_FILTER / "all_variant_scores.txt",
        "baseline_root": shadow,
        "vcf_dir": canonical / "output_VCFs/spotprofiles" / QUALITY_FILTER / "vcf_by_spot",
        "positions": outs / "spatial/tissue_positions.csv",
        "outs": outs,
        "purity": cal / "estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
        "clones": clone / "clone_labels.tsv",
        "cnv": clone / cnv_name,
    }


SECTIONS = {
    "P4_rep1": _section("P4", "1", "clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P4_rep2": _section("P4", "2", "clone2_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P6_rep1": _section("P6", "1", "clone3_rectangle0_w1.0", "cnv_seglevel.tsv"),
    "P6_rep2": _section("P6", "2", "clone3_rectangle0_w1.0", "cnv_diploid_seglevel.tsv"),
}

WES_TRUTH = ROOT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv"
P6_PATHOLOGY_STUDY = (
    ROOT / "data/spatialsnv_reanalysis_2026-07-17/p6_gt/extracted"
    / "P6_rep1_SPARCAL_somatic"
)


def section_dir(section: str) -> Path:
    return OUTPUT_ROOT / section


def cache_dir(section: str) -> Path:
    return section_dir(section) / "cache"


def score_dir(section: str) -> Path:
    return section_dir(section) / "scores"


def matrix_path(section: str, model: str, selection: str = "matched") -> Path:
    return section_dir(section) / "matrices" / f"{model}__{selection}.pkl"


def viewer_label(model: str, selection: str = "matched") -> str:
    return f"sf_multistate_v1__{model}__{selection}"

