#!/usr/bin/env python3
"""
Enhanced Spatial SNV Filter with Tumor Purity, Clone Labels, and CNV Integration
==================================================================================
Complete implementation with VCF outputs, visualizations, and pool handling.

Version 2.0: Added CalicoST clone labels and CNV segments integration
- Soft probabilistic scoring based on clone enrichment (cancer vs normal)
- CNV consistency scoring (per-clone CNV-variant pattern matching)
- Backward compatible: works with or without clone/CNV data

Author: Yuki & Claude
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional, List
import scipy.stats
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import glob
import json
import hashlib
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from datetime import datetime, timezone
import gzip
import csv
import warnings
from bisect import bisect_right
from pathlib import Path

from mechanistic_vaf import (
    DEFAULT_PURITY_PRIOR,
    parse_sample_counts,
    score_mechanistic_vaf,
)
from calicost_gate import evaluate_gate_files, section_rows
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SPOTPROFILE_SCHEMA = "sparcal_spotprofiles_v1"
SPATIAL_FILTER_SCHEMA = "sparcal_spatial_filter_v1"
LEGACY_MIN_EXPRESSION_DEFAULTS = (2, 1)
PRESENCE_RULES = ("G1", "G0", "G2")


def make_variant_key(chrom, pos, ref, alt) -> str:
    """Return the canonical allele-exact feature identifier.

    The spatial stages used to key records by ``chrom_pos`` and therefore
    collapsed different ALT alleles at the same locus. VCF SNV components do
    not contain underscores, so ``chrom_pos_ref_alt`` is human-readable and
    losslessly parseable.
    """
    chrom = str(chrom)
    if chrom.startswith("chr"):
        chrom = chrom[3:]
    fields = (chrom, str(pos), str(ref), str(alt))
    if any(not field or "_" in field for field in fields):
        raise ValueError(f"Invalid variant-key fields: {fields!r}")
    if "," in fields[3]:
        raise ValueError("Variant key requires one ALT allele, not a multiallelic field")
    return "_".join(fields)


def parse_variant_key(variant: str) -> Tuple[str, str, str, str]:
    """Parse a canonical ``chrom_pos_ref_alt`` feature identifier."""
    fields = str(variant).split("_", 3)
    if len(fields) != 4 or any(not field for field in fields):
        raise ValueError(f"Expected chrom_pos_ref_alt variant key, got {variant!r}")
    try:
        int(fields[1])
    except ValueError as exc:
        raise ValueError(f"Invalid position in variant key {variant!r}") from exc
    return tuple(fields)


def variant_sort_key(variant: str):
    """Deterministic genomic/allelic ordering for exact feature identifiers."""
    chrom, pos, ref, alt = parse_variant_key(variant)
    chrom_sort = int(chrom) if chrom.isdigit() else 100
    return chrom_sort, chrom, int(pos), ref, alt


def apply_presence_rule(record_pairs, alt_molecules, presence_rule):
    """Return exact ``(barcode, variant)`` pairs present under G0/G1/G2.

    ``record_pairs`` is the G0 universe: every in-tissue step-6 VCF record.
    ``alt_molecules`` must contain one entry for every such pair. Requiring an
    exact inventory match makes stale or partial molecule tables fail closed.
    """
    if presence_rule not in PRESENCE_RULES:
        raise ValueError(f"unknown presence rule: {presence_rule}")
    records = set(record_pairs)
    molecule_pairs = set(alt_molecules)
    missing = records - molecule_pairs
    extra = molecule_pairs - records
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing={len(missing)} first={sorted(missing)[:1]}")
        if extra:
            details.append(f"extra={len(extra)} first={sorted(extra)[:1]}")
        raise ValueError(
            "molecule-table inventory does not exactly match in-tissue step-6 "
            "VCF records: " + "; ".join(details)
        )
    if presence_rule == "G0":
        return records
    alt_positive = {
        pair for pair, count in alt_molecules.items() if int(count) >= 1
    }
    if presence_rule == "G2":
        return alt_positive
    eligible_loci = {variant for _, variant in alt_positive}
    return {pair for pair in records if pair[1] in eligible_loci}

# Visualization settings
FIGURE_SIZE = (12, 10)
DPI = 300
COLORMAP_GERMLINE = ['white', 'lightblue', 'blue', 'darkblue']
COLORMAP_SOMATIC = ['white', 'lightyellow', 'orange', 'red']
COLORMAP_COMBINED = ['white', 'lightgray', 'gray', 'darkgray']

# Dataset configurations
# REPLACE the whole DATASET_CONFIGS dict with this:
# DLPFC_PREDEDUP=1 -> read/write the pre-dedup tree (data/dlpfc_prededup) so the
# post-dedup results under data/dlpfc are left untouched. Default: unchanged.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc_prededup" if _PREDEDUP else "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc",
        "spatial_subdir": "spatial",
        "position_file": "tissue_positions_list.csv",   # no header
        "scale_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "has_header": False,
        "in_tissue_column": 1
    },
    "P4_TUMOR": {
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "P4_Tumor_output/outs/spatial",
        "position_file_patterns": {"1": "tissue_positions.csv", "2": "tissue_positions.csv"},
        "scale_file_patterns": {"1": "scalefactors_json.json", "2": "scalefactors_json.json"},
        "image_file_patterns": {"1": "tissue_hires_image.png", "2": "tissue_hires_image.png"},
        "has_header": True,
        "x_flip":     False,
    },
    "P6_TUMOR": {
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "P6_Tumor_output/outs/spatial",
        "position_file_patterns": {"1": "tissue_positions.csv", "2": "tissue_positions.csv"},
        "scale_file_patterns": {"1": "scalefactors_json.json", "2": "scalefactors_json.json"},
        "image_file_patterns": {"1": "tissue_hires_image.png", "2": "tissue_hires_image.png"},
        "has_header": True,
        "x_flip":     False,
    },
    "DCIS": {
        # hg38, two sections: dcis1 / dcis2
        # Position file lives directly under DCIS{N}/ (no header, integer coords)
        # Scale/image files live under the full spaceranger outs/spatial/ path
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data",
        "spaceranger_dir_patterns": {
            "dcis1": "DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs",
            "dcis2": "DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs",
        },
        "spatial_subdir": "spatial",
        "position_file": "tissue_positions.csv",        # has header, float coords (new spaceranger)
        "scale_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "in_tissue_column": 1,     # col[1] == 1 means in-tissue
        "flip_x": True,            # horizontal axis flip needed for correct orientation
    },
    "OVAR_P5": {
        # Ovarian cancer patient, GRCh38 chr-prefix. Spatial files live in the colleague's
        # read-only outs/ (base_path/{section_id}/outs/spatial); pipeline outputs go to
        # our data/ovar_p5/{section_id}. Standard headerless Visium V1 positions.
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/ovar_p5",
        "spatial_subdir": "outs/spatial",
        "position_file": "tissue_positions_list.csv",   # no header, integer coords
        "scale_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "has_header": False,
        "in_tissue_column": 1
    },
    "NCCE": {
        # Newer SpaceRanger outs with a header in tissue_positions.csv. Outputs,
        # including derived annotation metadata, remain under data/ncce/.
        "base_path": "/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/ncce",
        "spatial_subdir": "NCCE_{section_id}/outs/spatial",
        "position_file": "tissue_positions.csv",
        "scale_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "has_header": True,
        "in_tissue_column": 1
    }

}


class EnhancedSpatialSNVFilter:
    """
    Enhanced spatial filtering with tumor purity integration.
    """
    
    def __init__(self, 
                 dataset: str,
                 section_id: str,
                 quality_filter: str,
                 tumor_purity_file: str,
                 molecule_table_file: str = None,
                 presence_rule: str = "G1",
                 output_dir: str = None,
                 section_root: str = None,
                 neighbor_distance: float = 2.0,
                 min_expression_germline: int = 2,
                 min_expression_somatic: int = 1,
                 exclude_vcf_path: str = None,
                 include_vcf_path: str = None,
                 kept_variants_path: str = None,
                 clone_labels_file: str = None,
                 cnv_segments_file: str = None,
                 clone_posterior_file: str = None,
                 spot_umi_tsv: str = None,
                 calicost_sample_id: str = None,
                 calicost_gate: str = "on",
                 cnv_feature: str = "clone_enrichment",
                 evidence_mode: str = "full",
                 mechanistic_vaf_mode: str = "auto",
                 mechanistic_kappa: float = 8.0,
                 mechanistic_score_scale: float = 10.0,
                 mechanistic_min_information: float = 0.20,
                 germline_threshold: float = 0.3,
                 somatic_threshold: float = 0.2):
        """
        Initialize the enhanced spatial filter with all parameters.
        
        New in v2: Optional CalicoST clone labels and CNV segments for enhanced scoring.
        """
        self.dataset = dataset.upper()
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.tumor_purity_file = tumor_purity_file
        self.molecule_table_file = molecule_table_file
        self.presence_rule = presence_rule
        self.section_root = os.path.abspath(section_root) if section_root else None
        self.neighbor_distance = neighbor_distance
        # These two historical arguments never formed valid class-specific
        # gates: presence had already been defined as AD[ALT] >= 1. Preserve the
        # canonical CLI values during migration, reject any other value, and
        # record the real support rule explicitly.
        legacy_expression = (min_expression_germline, min_expression_somatic)
        if legacy_expression != LEGACY_MIN_EXPRESSION_DEFAULTS:
            raise ValueError(
                "--min_expression_germline/--min_expression_somatic are deprecated "
                "and only legacy values 2/1 are accepted; classification uses the "
                "explicit --presence_rule contract"
            )
        self.deprecated_min_expression_germline = min_expression_germline
        self.deprecated_min_expression_somatic = min_expression_somatic
        logger.warning(
            "Class-specific --min_expression_* options are deprecated; the frozen "
            "presence contract is per-spot ALT count >=1"
        )
        self.exclude_vcf_path = exclude_vcf_path
        self.include_vcf_path = include_vcf_path
        self.kept_variants_path = kept_variants_path
        self.germline_threshold = germline_threshold
        self.somatic_threshold = somatic_threshold
        self.evidence_mode = evidence_mode
        self.mechanistic_vaf_mode = mechanistic_vaf_mode
        self.mechanistic_kappa = mechanistic_kappa
        self.mechanistic_score_scale = mechanistic_score_scale
        self.mechanistic_min_information = mechanistic_min_information

        if evidence_mode not in {"full", "spatial_only_no_calicost"}:
            raise ValueError(f"Unknown evidence mode: {evidence_mode}")
        if mechanistic_vaf_mode not in {"auto", "off", "shadow", "vote"}:
            raise ValueError(f"Unknown mechanistic VAF mode: {mechanistic_vaf_mode}")
        if calicost_gate not in {"on", "off"}:
            raise ValueError(f"Unknown CalicoST gate mode: {calicost_gate}")
        if cnv_feature not in {"clone_enrichment", "legacy"}:
            raise ValueError(f"Unknown CNV feature: {cnv_feature}")
        if evidence_mode == "spatial_only_no_calicost" and any(
                (tumor_purity_file, clone_labels_file, cnv_segments_file,
                 clone_posterior_file, spot_umi_tsv, calicost_sample_id)):
            raise ValueError(
                "spatial_only_no_calicost must not receive tumor-purity, clone-label, "
                "or CNV inputs"
            )
        if evidence_mode == "spatial_only_no_calicost":
            self.mechanistic_vaf_mode = "off"
        elif mechanistic_vaf_mode == "auto":
            self.mechanistic_vaf_mode = (
                "shadow" if any((tumor_purity_file, cnv_segments_file)) else "off"
            )
        
        # CalicoST clone/CNV inputs and gate state.
        self.tumor_purity_file_requested = tumor_purity_file
        self.clone_labels_file = clone_labels_file
        self.cnv_segments_file = cnv_segments_file
        self.clone_posterior_file = clone_posterior_file
        self.spot_umi_tsv = spot_umi_tsv
        self.calicost_sample_id = calicost_sample_id
        self.calicost_gate_mode = calicost_gate
        self.cnv_feature = cnv_feature
        self.calicost_gate_result = None
        self.calicost_inputs_present = any((
            tumor_purity_file, clone_labels_file, cnv_segments_file,
            clone_posterior_file, spot_umi_tsv, calicost_sample_id,
        ))
        if evidence_mode == "full" and calicost_gate == "on" and self.calicost_inputs_present:
            required_gate_inputs = {
                "--clone_labels": clone_labels_file,
                "--cnv_segments": cnv_segments_file,
                "--clone_posterior": clone_posterior_file,
                "--spot_umi_tsv": spot_umi_tsv,
                "--calicost_sample_id": calicost_sample_id,
            }
            missing = [name for name, value in required_gate_inputs.items() if not value]
            if missing:
                raise ValueError(
                    "CalicoST gate is on but required inputs are missing: "
                    + ", ".join(missing)
                )
        if self.presence_rule not in PRESENCE_RULES:
            raise ValueError(f"Unknown presence rule: {self.presence_rule}")
        if not self.molecule_table_file:
            raise ValueError(
                "--molecule_table is required for the G0/G1/G2 presence contract"
            )
        self.use_clone_cnv = (clone_labels_file is not None and cnv_segments_file is not None)
        
        # Get dataset configuration
        if self.dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset}")
        self.config = DATASET_CONFIGS[self.dataset]
        
        # Set up paths
        self.setup_paths(output_dir)
        
        # Initialize data structures
        self.spot_positions = {}  # barcode -> (x, y)
        self.spot_snvs = defaultdict(set)  # barcode -> set of SNVs
        self.tumor_purity = {}  # barcode -> purity (0-1)
        self.tumor_purity_default = np.nan  # missing means unknown, never normal
        self.purity_prior = DEFAULT_PURITY_PRIOR.copy()
        self.has_informative_purity = False
        self.voting_features = []
        self.spot_neighbors = defaultdict(list)  # barcode -> list of neighbors
        self.snv_ref_alt_map = {}  # snv_key -> (ref, alt)
        self.snv_race = {}  # snv_key -> "defined" or "denovo"
        self.snv_ref_counts = defaultdict(dict)  # snv_key -> barcode -> AD[0]
        self.snv_alt_counts = defaultdict(dict)  # snv_key -> barcode -> AD[1]
        self.snv_depth_counts = defaultdict(dict)  # snv_key -> barcode -> sum(AD)
        self.vcf_evidence_stats = defaultdict(int)
        
        # NEW: Clone and CNV data structures
        self.clone_labels = {}  # barcode -> clone_id (0=normal, 1,2,3...=cancer)
        self.cnv_segments = None  # DataFrame with CNV segments
        self.cnv_interval_index = {}  # chrom -> (starts, ordered segment Series)
        self.all_clones = []  # List of effective clone IDs
        self.reference_clone = 0  # legacy fallback; gate mode selects by CNV profile
        self.molecule_alt_counts = {}
        self.molecule_alt_supported_variants = set()
        self.molecule_table_stats = {}
        self.cancer_clones = []  # Effective tumor clones selected by the gate
        self.cnv_clone_ids = []  # Clone IDs represented by allele-specific CNV columns
        
        # Pools for filtering
        self.exclude_snv_pool = set()
        self.include_snv_pool = set()
        self.kept_variants_pool = set()
        
        # Results storage - split by classification AND race
        self.germline_variants = set()
        self.somatic_variants = set()
        self.ambiguous_variants = set()
        
        # Race-specific variant sets
        self.germline_defined = set()
        self.germline_denovo = set()
        self.germline_kept = set()   # subset of germline_denovo: forced via kept_variants_pool
        self.somatic_defined = set()
        self.somatic_denovo = set()
        self.ambiguous_defined = set()
        self.ambiguous_denovo = set()
        
        self.variant_scores = {}  # variant -> {'germline': score, 'somatic': score}
        
        # Filtered results per spot
        self.filtered_spot_snvs_germline = defaultdict(set)
        self.filtered_spot_snvs_somatic = defaultdict(set)
        self.filtered_spot_snvs_ambiguous = defaultdict(set)
        
        # Score weights - dynamically adjusted based on available data
        self.weights_germline = {
            'alpha': 0.4,   # spatial uniformity
            'beta': 0.3,    # global prevalence  
            'gamma': 0.3    # purity independence
        }
        
        if self.evidence_mode == "spatial_only_no_calicost":
            self.weights_somatic = {'zeta': 1.0}
            logger.info("Using CalicoST-free spatial-only ablation scoring")
        elif self.use_clone_cnv:
            # When clone+CNV data available, adjust weights
            self.weights_somatic = {
                'delta': 0.25,    # purity correlation
                'epsilon': 0.15,  # clone-specific (spatial)
                'zeta': 0.20,     # spatial clustering
                'eta': 0.25,      # NEW: clone enrichment
                'theta': 0.15     # NEW: CNV consistency
            }
            logger.info("Using enhanced scoring with clone labels and CNV data")
        else:
            # Original weights when only purity available
            self.weights_somatic = {
                'delta': 0.5,   # purity correlation
                'epsilon': 0.2, # clone-specific
                'zeta': 0.3     # spatial clustering
            }
            logger.info("Using purity-only scoring (no clone/CNV data)")
        
    def setup_paths(self, output_dir=None):
        cfg = self.config

        # SNV per-barcode VCFs -- new path under spotprofiles/. An explicit
        # section root redirects both inputs and outputs for non-overwriting runs.
        section_root = self.section_root or os.path.join(
            cfg.get('output_base'), self.section_id)
        self.spotprofile_root = os.path.join(
            section_root, f"output_VCFs/spotprofiles/{self.quality_filter}"
        )
        self.snv_pos_dir = os.path.join(self.spotprofile_root, "vcf_by_spot")

        # Spatial files
        if self.dataset == "DLPFC":
            # /DLPFC12/<section_id>/spatial/...
            spatial_base = os.path.join(cfg["base_path"], self.section_id, "spatial")
            self.positions_file     = os.path.join(spatial_base, cfg["position_file"])
            self.scale_factor_file  = os.path.join(spatial_base, cfg["scale_file"])
            self.image_file         = os.path.join(spatial_base, cfg["image_file"])
        elif self.dataset == "DCIS":
            # All three files (positions, scale, image) live under the same spatial_base
            outs_dir     = cfg["spaceranger_dir_patterns"][self.section_id]
            spatial_base = os.path.join(cfg["base_path"], outs_dir, cfg["spatial_subdir"])
            self.positions_file    = os.path.join(spatial_base, cfg["position_file"])
            self.scale_factor_file = os.path.join(spatial_base, cfg["scale_file"])
            self.image_file        = os.path.join(spatial_base, cfg["image_file"])
        elif self.dataset == "OVAR_P5":
            # Colleague read-only outs: base_path/{section_id}/outs/spatial/...
            spatial_base = os.path.join(cfg["base_path"], self.section_id, cfg["spatial_subdir"])
            self.positions_file    = os.path.join(spatial_base, cfg["position_file"])
            self.scale_factor_file = os.path.join(spatial_base, cfg["scale_file"])
            self.image_file        = os.path.join(spatial_base, cfg["image_file"])
        elif self.dataset == "NCCE":
            spatial_base = os.path.join(
                cfg["base_path"], cfg["spatial_subdir"].format(section_id=self.section_id))
            self.positions_file    = os.path.join(spatial_base, cfg["position_file"])
            self.scale_factor_file = os.path.join(spatial_base, cfg["scale_file"])
            self.image_file        = os.path.join(spatial_base, cfg["image_file"])
        else:
            # /P4_Visium|P6_Visium/spaceranger_align_rep{section}_hg19/Meta_Data/...
            spaceranger_dir = cfg["spaceranger_dir_template"].format(section_id=self.section_id)
            spatial_base = os.path.join(cfg["base_path"], spaceranger_dir, cfg["spatial_subdir"])

            # Section-specific GSM files
            posfile   = cfg["position_file_patterns"][self.section_id]
            scalefile = cfg["scale_file_patterns"][self.section_id]
            imgfile   = cfg["image_file_patterns"][self.section_id]

            self.positions_file     = os.path.join(spatial_base, posfile)
            self.scale_factor_file  = os.path.join(spatial_base, scalefile)
            self.image_file         = os.path.join(spatial_base, imgfile)

            # Optional out-tissue list (same place)
            self.out_tissue_file = getattr(self, "out_tissue_file", None)
            if self.out_tissue_file is None:
                candidate = os.path.join(spatial_base, "missing_barcodes.txt")
                if os.path.exists(candidate):
                    self.out_tissue_file = candidate
                    logger.info(f"Using out-tissue file: {self.out_tissue_file}")
                else:
                    logger.warning(f"Out-tissue file not found at {candidate}")
                    self.out_tissue_file = None

        # Output dir
        if output_dir:
            self.output_dir = output_dir
        else:
            self.output_dir = os.path.join(
                section_root, f"spatial_filter_purity/{self.quality_filter}"
            )
        self._assert_output_is_empty()

        # Subdirs - now with race split (12 directories total)
        self.germline_dir = os.path.join(self.output_dir, "germline")
        self.somatic_dir  = os.path.join(self.output_dir, "somatic")
        self.ambiguous_dir = os.path.join(self.output_dir, "ambiguous")
        self.viz_dir      = os.path.join(self.output_dir, "visualizations")
        
        # Race-specific subdirectories
        self.germline_defined_dir = os.path.join(self.germline_dir, "defined")
        self.germline_denovo_dir = os.path.join(self.germline_dir, "denovo")
        self.somatic_defined_dir = os.path.join(self.somatic_dir, "defined")
        self.somatic_denovo_dir = os.path.join(self.somatic_dir, "denovo")
        self.ambiguous_defined_dir = os.path.join(self.ambiguous_dir, "defined")
        self.ambiguous_denovo_dir = os.path.join(self.ambiguous_dir, "denovo")
        
        logger.info(f"SNV VCF dir: {self.snv_pos_dir}")
        logger.info(f"Positions: {self.positions_file}")
        logger.info(f"Scale:     {self.scale_factor_file}")
        logger.info(f"Image:     {self.image_file}")
        logger.info(f"Output:    {self.output_dir}")


    
    def _assert_output_is_empty(self):
        """Require a fresh output root; even an empty pre-existing path is refused."""
        if os.path.lexists(self.output_dir):
            raise FileExistsError(
                "Spatial-filter output path already exists; refusing to overwrite. "
                f"Choose a new --section-root or --output_dir: {self.output_dir}"
            )

    def _initialize_output_dirs(self):
        """Atomically claim a fresh output root immediately before writing."""
        os.makedirs(self.output_dir, exist_ok=False)
        for directory in [
            self.germline_dir, self.somatic_dir, self.ambiguous_dir, self.viz_dir,
            self.germline_defined_dir, self.germline_denovo_dir,
            self.somatic_defined_dir, self.somatic_denovo_dir,
            self.ambiguous_defined_dir, self.ambiguous_denovo_dir,
        ]:
            os.makedirs(directory, exist_ok=False)

    @staticmethod
    def _sha256_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def validate_spotprofile_bundle(self):
        """Validate the completed Stage 5 inventory before reading any VCF."""
        root = os.path.abspath(self.spotprofile_root)
        manifest_path = os.path.join(root, "MANIFEST.json")
        complete_path = os.path.join(root, "RUN_COMPLETE")
        required = [manifest_path, complete_path, self.snv_pos_dir]
        missing = [path for path in required if not os.path.exists(path)]
        if missing:
            raise FileNotFoundError(
                "incomplete required Stage 5 spot-profile bundle; missing: "
                + ", ".join(missing)
            )
        if not os.path.isdir(self.snv_pos_dir):
            raise NotADirectoryError(self.snv_pos_dir)
        if Path(complete_path).read_text().strip() != "complete":
            raise ValueError(f"invalid Stage 5 RUN_COMPLETE sentinel: {complete_path}")
        try:
            manifest = json.loads(Path(manifest_path).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Stage 5 manifest {manifest_path}: {error}") from error
        if manifest.get("schema") != SPOTPROFILE_SCHEMA or manifest.get("status") != "complete":
            raise ValueError(f"unsupported/incomplete Stage 5 manifest: {manifest_path}")
        upstream_section = str(manifest.get("section_id"))
        expected_section = str(self.section_id)
        if self.dataset == "DCIS" and expected_section.lower().startswith("dcis"):
            expected_section = expected_section[4:]
        if (
            manifest.get("dataset") != self.dataset
            or upstream_section != expected_section
            or manifest.get("quality_filter") != self.quality_filter
        ):
            raise ValueError(
                "Stage 5 manifest dataset/section/quality does not match this run"
            )
        if (
            manifest.get("beagle_anchor_score") != "p_nonref"
            or not np.isclose(
                float(manifest.get("beagle_min_anchor_probability", np.nan)),
                0.90, rtol=0.0, atol=1e-12,
            )
        ):
            raise ValueError("Stage 5 manifest is not production P(nonref)>=0.90")
        try:
            expected_count = int(manifest["per_spot_vcf_count"])
            failed_count = int(manifest["failed_bam_count"])
            declared_files = list(manifest["per_spot_vcfs"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Stage 5 file inventory: {manifest_path}") from error
        if failed_count != 0:
            raise ValueError(f"Stage 5 manifest reports {failed_count} failed BAMs")
        if expected_count <= 0:
            raise ValueError("Stage 5 completed with zero required per-spot VCFs")
        actual_paths = sorted(glob.glob(os.path.join(self.snv_pos_dir, "*.vcf.gz")))
        actual_files = [os.path.basename(path) for path in actual_paths]
        if expected_count != len(actual_paths) or sorted(declared_files) != actual_files:
            raise ValueError(
                "Stage 5 per-spot VCF inventory disagrees with files on disk: "
                f"manifest={expected_count}, actual={len(actual_paths)}"
            )
        if len(declared_files) != len(set(declared_files)):
            raise ValueError("Stage 5 manifest contains duplicate per-spot VCF names")
        missing_indices = [path + ".tbi" for path in actual_paths if not os.path.isfile(path + ".tbi")]
        if missing_indices:
            raise FileNotFoundError(
                f"missing required per-spot VCF index: {missing_indices[0]}"
            )
        self.spotprofile_manifest = manifest
        self.spotprofile_manifest_path = manifest_path
        return manifest

    def apply_calicost_gate(self):
        """Evaluate and apply section-specific CalicoST evidence eligibility."""
        self.calicost_gate_result = evaluate_gate_files(
            self.clone_labels_file,
            self.cnv_segments_file,
            self.clone_posterior_file,
            self.tumor_purity_file,
            self.calicost_sample_id,
            self.spot_umi_tsv,
        )
        gate = self.calicost_gate_result
        if not gate["purity_pass"]:
            logger.warning("CalicoST purity gate failed; disabling purity features")
            self.tumor_purity_file = None
        self.use_clone_cnv = bool(gate["cnv_pass"])
        if self.use_clone_cnv:
            self.clone_labels = {
                barcode: int(label)
                for barcode, label in gate["effective_labels"].items()
                if label is not None
            }
            self.reference_clone = int(gate["reference_clone"])
            self.cancer_clones = [int(value) for value in gate["tumor_clones"]]
            self.all_clones = [int(value) for value in gate["surviving_clones"]]
        else:
            logger.warning("CalicoST CNV gate failed; disabling clone/CNV scoring")
            self.clone_labels = {}
            self.reference_clone = None
            self.cancer_clones = []
            self.all_clones = []
        logger.info(
            "CalicoST gate: CNV=%s purity=%s reference=%s tumors=%s dropped=%s",
            "PASS" if gate["cnv_pass"] else "FAIL",
            "PASS" if gate["purity_pass"] else "FAIL",
            gate["reference_clone"], gate["tumor_clones"], gate["dropped_clones"],
        )

    def load_tumor_purity(self):
        """Load tumor purity data from CalicoST output."""
        if self.tumor_purity_file is None:
            logger.info("No tumor purity file provided — purity is unavailable (not zero)")
            return

        logger.info(f"Loading tumor purity from: {self.tumor_purity_file}")

        try:
            purity_df = pd.read_csv(self.tumor_purity_file, sep='\t')
            purity_df, _ = section_rows(purity_df, self.calicost_sample_id)
            for _, row in purity_df.iterrows():
                barcode = str(row['BARCODES'])
                purity = float(row['Tumor']) if pd.notna(row['Tumor']) else np.nan
                self.tumor_purity[barcode] = purity
            
            purities = list(self.tumor_purity.values())
            finite_purities = np.asarray(purities, dtype=float)
            finite_purities = finite_purities[np.isfinite(finite_purities)]
            self.has_informative_purity = bool(
                finite_purities.size >= 2 and np.ptp(finite_purities) > 1e-8
            )
            if self.has_informative_purity:
                self.purity_prior = np.unique(np.quantile(
                    finite_purities, [0.10, 0.25, 0.50, 0.75, 0.90]
                ))
            logger.info(f"Loaded tumor purity for {len(self.tumor_purity)} spots")
            logger.info(
                f"Finite purity values: {finite_purities.size}/{len(purities)}; "
                + (
                    f"min={np.min(finite_purities):.3f}, max={np.max(finite_purities):.3f}, "
                    f"mean={np.mean(finite_purities):.3f}"
                    if finite_purities.size else "none"
                )
            )
            logger.info(
                "Missing purity values will be excluded from association tests "
                "and marginalized in the mechanistic VAF model"
            )
            if not self.has_informative_purity:
                logger.warning(
                    "Tumor-purity values are absent or constant; purity-derived "
                    "features delta/epsilon will be excluded from voting"
                )
            
        except Exception as e:
            logger.error(f"Error loading tumor purity: {e}")
            raise
    
    def load_clone_labels(self):
        """Load clone assignments from CalicoST."""
        if not self.clone_labels_file:
            return
        
        logger.info(f"Loading clone labels from: {self.clone_labels_file}")
        
        try:
            if self.calicost_gate_result is not None:
                # Effective labels were installed by the gate; dropped clones are absent.
                clone_df = None
            else:
                clone_df = pd.read_csv(self.clone_labels_file, sep='\t')
                clone_df, _ = section_rows(clone_df, self.calicost_sample_id)
                for _, row in clone_df.iterrows():
                    barcode = str(row['BARCODES'])
                    self.clone_labels[barcode] = int(row['clone_label'])
                self.all_clones = sorted(set(self.clone_labels.values()))
                self.reference_clone = 0 if 0 in self.all_clones else None
                self.cancer_clones = [c for c in self.all_clones if c != self.reference_clone]
            
            logger.info(f"Loaded {len(self.clone_labels)} clone assignments")
            logger.info(f"Clones detected: {self.all_clones}")
            for clone_id in self.all_clones:
                count = sum(1 for c in self.clone_labels.values() if c == clone_id)
                clone_type = ("Reference clone" if clone_id == self.reference_clone
                              else f"Tumor clone {clone_id}")
                logger.info(f"  {clone_type}: {count} spots ({count/len(self.clone_labels)*100:.1f}%)")
        
        except Exception as e:
            logger.error(f"Error loading clone labels: {e}")
            raise
    
    def load_cnv_segments(self):
        """Load CNV segments from CalicoST."""
        if not self.cnv_segments_file:
            return

        logger.info(f"Loading CNV segments from: {self.cnv_segments_file}")

        try:
            self.cnv_segments = pd.read_csv(self.cnv_segments_file, sep='\t')
            logger.info(f"Loaded {len(self.cnv_segments)} CNV segments")

            # ── Normalize column names ──────────────────────────────────────
            # CalicoST outputs 'clone1 A' / 'clone1 B' (space-separated),
            # but the scoring code expects 'clone1_A' / 'clone1_B'.
            # Replace spaces with underscores in all column names at load time
            # so all downstream code works without modification.
            original_cols = list(self.cnv_segments.columns)
            self.cnv_segments.columns = [c.replace(' ', '_') for c in original_cols]
            normalized_cols = list(self.cnv_segments.columns)
            if original_cols != normalized_cols:
                logger.info(f"Normalized CNV column names (space→underscore): "
                            f"{original_cols} → {normalized_cols}")

            # ── Detect available clone columns and log CNA structure ────────
            cn_cols = [c for c in normalized_cols if c.endswith('_A') or c.endswith('_B')]
            clone_ids_in_file = sorted(set(
                int(c.replace('clone', '').replace('_A', '').replace('_B', ''))
                for c in cn_cols if c.startswith('clone')
            ))
            self.cnv_clone_ids = clone_ids_in_file
            logger.info(f"CNV clone columns detected: {cn_cols}")
            logger.info(f"Cancer clones in CNV file: {clone_ids_in_file}")

            # Warn and reconcile if cancer_clones from labels don't match CNV file
            if self.cancer_clones and set(self.cancer_clones) != set(clone_ids_in_file):
                logger.warning(
                    f"Mismatch: cancer_clones from labels={self.cancer_clones} "
                    f"vs CNV file clones={clone_ids_in_file}. Using intersection."
                )
            self.cancer_clones = [c for c in self.cancer_clones
                                   if c in clone_ids_in_file]
            logger.info(f"Effective cancer_clones for scoring: {self.cancer_clones}")

            # Log CNA event summary per clone
            chroms = self.cnv_segments['CHR'].unique()
            logger.info(f"CNV data covers {len(chroms)} chromosomes")
            a_cols = [c for c in cn_cols if c.endswith('_A')]
            b_cols = [c for c in cn_cols if c.endswith('_B')]
            for col_a, col_b in zip(a_cols, b_cols):
                n_total   = len(self.cnv_segments)
                n_altered = ((self.cnv_segments[col_a] != 1) |
                             (self.cnv_segments[col_b] != 1)).sum()
                n_loh     = ((self.cnv_segments[col_a] == 0) |
                             (self.cnv_segments[col_b] == 0)).sum()
                logger.info(f"  {col_a}/{col_b}: {n_altered}/{n_total} altered "
                            f"segments ({n_loh} LOH)")

            # Build a chromosome interval index once.  Mechanistic scoring can
            # query tens of thousands of variants, so repeated DataFrame scans
            # would otherwise dominate runtime.
            self.cnv_segments['CHR'] = (
                self.cnv_segments['CHR'].astype(str).str.replace('chr', '', regex=False)
            )
            self.cnv_segments['START'] = pd.to_numeric(self.cnv_segments['START'])
            self.cnv_segments['END'] = pd.to_numeric(self.cnv_segments['END'])
            self.cnv_interval_index = {}
            for chrom, frame in self.cnv_segments.groupby('CHR', sort=False):
                ordered = frame.sort_values(['START', 'END'])
                rows = [row for _, row in ordered.iterrows()]
                self.cnv_interval_index[str(chrom)] = (
                    [int(row['START']) for row in rows], rows
                )

        except Exception as e:
            logger.error(f"Error loading CNV segments: {e}")
            raise
    
# REPLACE the whole load_spot_positions method with this:
    @staticmethod
    def _read_positions_csv(filepath):
        """
        Read a SpaceRanger tissue_positions CSV regardless of whether it has a header.
        Detection: if col[0] of the first row is not a barcode (ACGT... pattern),
        treat that row as a header and skip it.
        Returns a DataFrame with integer positional indexing (header=None).
        """
        import re
        BARCODE_RE = re.compile(r'^[ACGTacgt]+-\d+$')
        with open(filepath, 'r') as fh:
            first_line = fh.readline().strip()
        first_col = first_line.split(',')[0]
        has_header = not bool(BARCODE_RE.match(first_col))
        if has_header:
            logger.info(f"Header detected in {filepath} — skipping header row")
            df = pd.read_csv(filepath, header=0)
            df.columns = range(len(df.columns))   # re-index to 0,1,2... for uniform access
        else:
            df = pd.read_csv(filepath, header=None)
        return df

    def load_spot_positions(self):
        logger.info(f"Loading spot positions from: {self.positions_file}")
        if not os.path.exists(self.positions_file):
            raise FileNotFoundError(f"Position file not found: {self.positions_file}")

        out_tissue_barcodes = set()
        if getattr(self, "out_tissue_file", None) and os.path.exists(self.out_tissue_file):
            with open(self.out_tissue_file) as f:
                for line in f:
                    bc = line.strip().split()[0]
                    if bc:
                        out_tissue_barcodes.add(bc)
            logger.info(f"Loaded {len(out_tissue_barcodes)} out-tissue barcodes")

        if self.dataset in ("DLPFC", "OVAR_P5"):
            # no header; cols: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
            df = pd.read_csv(self.positions_file, header=None)
            for _, row in df.iterrows():
                bc = row[0]
                if int(row[1]) != 1:
                    continue
                if bc in out_tissue_barcodes:
                    continue
                pxl_row = float(row[4])
                pxl_col = float(row[5])
                self.spot_positions[bc] = (pxl_col, pxl_row)
        elif self.dataset in ("DCIS", "NCCE"):
            # cols: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
            # Auto-detect header (new SpaceRanger uses named columns).
            # x_flip (horizontal mirror) is applied at visualization time via ax.invert_xaxis().
            df = self._read_positions_csv(self.positions_file)
            for _, row in df.iterrows():
                bc = str(row[0])
                if int(row[1]) != 1:
                    continue
                if bc in out_tissue_barcodes:
                    continue
                pxl_row = float(row[4])
                pxl_col = float(row[5])
                self.spot_positions[bc] = (pxl_col, pxl_row)
        else:
            # P4/P6: accept both legacy headerless files and current
            # Space Ranger tissue_positions.csv files with named columns.
            df = self._read_positions_csv(self.positions_file)
            for _, row in df.iterrows():
                bc = row[0]
                if int(row[1]) != 1:
                    continue
                if bc in out_tissue_barcodes:
                    continue
                pxl_row = float(row[4])
                pxl_col = float(row[5])
                self.spot_positions[bc] = (pxl_col, pxl_row)

        logger.info(f"Loaded {len(self.spot_positions)} spot positions")

    
    def load_vcf_variants(self, vcf_path: str) -> Set[str]:
        """Load an explicitly requested optional pool VCF, failing on bad input."""
        variants = set()
        if not os.path.isfile(vcf_path):
            raise FileNotFoundError(f"requested optional VCF does not exist: {vcf_path}")
        open_func = gzip.open if vcf_path.endswith('.gz') else open
        mode = 'rt' if vcf_path.endswith('.gz') else 'r'
        saw_chrom_header = False
        try:
            with open_func(vcf_path, mode) as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.startswith('#CHROM'):
                        if len(line.rstrip('\n').split('\t')) < 8:
                            raise ValueError(
                                f"{vcf_path}:{line_number}: malformed #CHROM header"
                            )
                        saw_chrom_header = True
                        continue
                    if line.startswith('#'):
                        continue
                    if not saw_chrom_header:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: data before #CHROM header"
                        )
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 8:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: malformed VCF record"
                        )
                    try:
                        pos = int(parts[1])
                    except ValueError as error:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: invalid POS {parts[1]!r}"
                        ) from error
                    chrom, ref, alt = parts[0], parts[3], parts[4]
                    if pos <= 0 or chrom in {'', '.'} or ref in {'', '.'} or alt in {'', '.'}:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: invalid allele fields"
                        )
                    for alt_allele in alt.split(','):
                        if not alt_allele or alt_allele == '.':
                            raise ValueError(
                                f"{vcf_path}:{line_number}: invalid ALT allele"
                            )
                        snv_key = make_variant_key(chrom, pos, ref, alt_allele)
                        variants.add(snv_key)
                        self.snv_ref_alt_map[snv_key] = (ref, alt_allele)
        except (OSError, EOFError, UnicodeError) as error:
            raise ValueError(f"could not read requested VCF {vcf_path}: {error}") from error
        if not saw_chrom_header:
            raise ValueError(f"missing #CHROM header: {vcf_path}")
        logger.info(f"Loaded {len(variants)} variants from {vcf_path}")
        return variants

    def create_exclusion_pool(self):
        """Create exclusion SNV pool from VCF."""
        if self.exclude_vcf_path:
            self.exclude_snv_pool = self.load_vcf_variants(self.exclude_vcf_path)
            logger.info(f"Exclusion pool: {len(self.exclude_snv_pool)} variants")
    
    def create_inclusion_pool(self):
        """Create inclusion SNV pool from VCF."""
        if self.include_vcf_path:
            self.include_snv_pool = self.load_vcf_variants(self.include_vcf_path)
            logger.info(f"Inclusion pool: {len(self.include_snv_pool)} variants")
    
    def create_kept_variants_pool(self):
        """Create kept variants pool from VCF."""
        if self.kept_variants_path:
            self.kept_variants_pool = self.load_vcf_variants(self.kept_variants_path)
            logger.info(f"Kept variants pool: {len(self.kept_variants_pool)} variants")
    
    def load_molecule_table(self):
        """Load the S6+ molecule inventory for in-tissue step-6 records."""
        path = Path(self.molecule_table_file)
        if not path.is_file():
            raise FileNotFoundError(f"required molecule table is missing: {path}")
        opener = gzip.open if path.suffix == ".gz" else open
        required = {"barcode", "chrom", "pos", "ref", "alt", "alt_mol"}
        counts = {}
        rows_total = rows_in_tissue = alt_positive = 0
        with opener(path, "rt", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            missing = sorted(required - set(reader.fieldnames or ()))
            if missing:
                raise ValueError(
                    f"molecule table {path} is missing columns: {', '.join(missing)}"
                )
            for line_number, row in enumerate(reader, start=2):
                rows_total += 1
                barcode = row["barcode"]
                if barcode not in self.spot_positions:
                    continue
                rows_in_tissue += 1
                try:
                    position = int(row["pos"])
                    alt_mol = int(row["alt_mol"])
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid pos/alt_mol"
                    ) from error
                if position <= 0 or alt_mol < 0:
                    raise ValueError(
                        f"{path}:{line_number}: pos must be positive and alt_mol nonnegative"
                    )
                variant = make_variant_key(
                    row["chrom"], position, row["ref"].upper(), row["alt"].upper()
                )
                pair = (barcode, variant)
                if pair in counts:
                    raise ValueError(
                        f"{path}:{line_number}: duplicate molecule row for {pair}"
                    )
                counts[pair] = alt_mol
                alt_positive += alt_mol >= 1
        if not counts:
            raise ValueError(f"molecule table has no in-tissue rows: {path}")
        self.molecule_alt_counts = counts
        # Precomputed once: has_observed_alt_support() is called per variant.
        self.molecule_alt_supported_variants = {
            variant for (_, variant), count in counts.items() if count >= 1
        }
        self.molecule_table_stats = {
            "rows_total": rows_total,
            "rows_in_tissue": rows_in_tissue,
            "alt_positive_rows": int(alt_positive),
        }
        logger.info(
            "Loaded molecule table: %s in-tissue records; %s ALT-positive",
            f"{rows_in_tissue:,}", f"{alt_positive:,}",
        )

    def load_snv_data(self):
        """Load step-6 records, then apply the molecule-driven G0/G1/G2 rule."""
        vcf_files = sorted(glob.glob(os.path.join(self.snv_pos_dir, "*.vcf.gz")))
        logger.info(f"Looking for SNV VCFs in: {self.snv_pos_dir}")
        logger.info(f"Found {len(vcf_files)} VCFs")
        if not vcf_files:
            raise FileNotFoundError(
                f"zero required per-spot VCFs under {self.snv_pos_dir}"
            )
        self.snv_expression_counts = defaultdict(lambda: defaultdict(int))
        covered_record_pairs = set()
        self.vcf_evidence_stats['vcf_files_total'] = len(vcf_files)

        for vcf in vcf_files:
            bc = os.path.basename(vcf)[:-len(".vcf.gz")]
            in_tissue = bc in self.spot_positions
            if in_tissue:
                self.vcf_evidence_stats['vcf_files_in_tissue'] += 1
            saw_chrom_header = False
            saw_record = False
            try:
                with gzip.open(vcf, "rt") as handle:
                    for line_number, line in enumerate(handle, start=1):
                        if line.startswith("#CHROM"):
                            header_fields = line.rstrip("\n").split("\t")
                            if len(header_fields) != 10:
                                raise ValueError(
                                    f"{vcf}:{line_number}: expected exactly one sample"
                                )
                            if header_fields[9] != bc:
                                raise ValueError(
                                    f"{vcf}:{line_number}: sample {header_fields[9]!r} "
                                    f"does not match filename barcode {bc!r}"
                                )
                            saw_chrom_header = True
                            continue
                        if line.startswith("#"):
                            continue
                        if not saw_chrom_header:
                            raise ValueError(
                                f"{vcf}:{line_number}: data before #CHROM header"
                            )
                        saw_record = True
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) != 10:
                            raise ValueError(
                                f"{vcf}:{line_number}: malformed single-sample VCF "
                                f"record with {len(parts)} fields"
                            )
                        try:
                            pos_int = int(parts[1])
                        except ValueError as error:
                            raise ValueError(
                                f"{vcf}:{line_number}: invalid POS {parts[1]!r}"
                            ) from error
                        chrom, ref, alt = parts[0], parts[3], parts[4]
                        if (
                            pos_int <= 0 or chrom in {'', '.'} or ref in {'', '.'}
                            or alt in {'', '.'} or ',' in alt
                        ):
                            raise ValueError(
                                f"{vcf}:{line_number}: expected one valid biallelic allele"
                            )
                        info_items = {}
                        for item in parts[7].split(';'):
                            if '=' in item:
                                name, value = item.split('=', 1)
                                info_items[name] = value
                        if 'RACE' not in info_items:
                            raise ValueError(
                                f"{vcf}:{line_number}: required INFO/RACE is missing"
                            )
                        race = info_items['RACE']
                        if race not in {'defined', 'panel_discordant', 'denovo', 'unknown'}:
                            raise ValueError(
                                f"{vcf}:{line_number}: invalid RACE={race!r}"
                            )
                        counts = parse_sample_counts(parts[8], parts[9])
                        if counts is None:
                            raise ValueError(
                                f"{vcf}:{line_number}: required biallelic FORMAT/AD is missing"
                            )

                        # Out-of-tissue VCFs are still parsed and integrity-checked,
                        # but they do not contribute evidence to this spatial section.
                        if not in_tissue:
                            continue
                        self.vcf_evidence_stats['records_total'] += 1
                        snv_key = make_variant_key(chrom, pos_int, ref, alt)
                        covered_record_pairs.add((bc, snv_key))
                        self.snv_ref_alt_map[snv_key] = (ref, alt)
                        race_priority = {
                            "unknown": 0, "denovo": 1,
                            "panel_discordant": 2, "defined": 3,
                        }
                        previous_race = self.snv_race.get(snv_key, "unknown")
                        if race_priority[race] >= race_priority.get(previous_race, 0):
                            self.snv_race[snv_key] = race

                        ad_depth = counts.ref_count + counts.alt_count
                        if ad_depth <= 0:
                            self.vcf_evidence_stats['records_zero_ad_depth'] += 1
                            continue
                        previous_depth = self.snv_depth_counts[snv_key].get(bc, -1)
                        previous_evidence = (
                            previous_depth,
                            self.snv_alt_counts[snv_key].get(bc, -1),
                            self.snv_ref_counts[snv_key].get(bc, -1),
                        )
                        candidate_evidence = (
                            ad_depth, counts.alt_count, counts.ref_count
                        )
                        if previous_depth >= 0:
                            self.vcf_evidence_stats['records_exact_duplicate'] += 1
                        if candidate_evidence > previous_evidence:
                            self.snv_ref_counts[snv_key][bc] = counts.ref_count
                            self.snv_alt_counts[snv_key][bc] = counts.alt_count
                            self.snv_depth_counts[snv_key][bc] = ad_depth

                        if counts.alt_count >= 1:
                            self.vcf_evidence_stats['records_alt_positive'] += 1
                        else:
                            self.vcf_evidence_stats['records_ref_only'] += 1
            except (OSError, EOFError, UnicodeError) as error:
                raise ValueError(f"could not read required per-spot VCF {vcf}: {error}") from error
            if not saw_chrom_header:
                raise ValueError(f"missing #CHROM header: {vcf}")
            if not saw_record:
                raise ValueError(
                    f"required per-spot VCF contains no records: {vcf}"
                )

        present_pairs = apply_presence_rule(
            covered_record_pairs, self.molecule_alt_counts, self.presence_rule
        )
        self.spot_snvs = defaultdict(set)
        self.snv_expression_counts = defaultdict(lambda: defaultdict(int))
        for barcode, snv_key in sorted(
            present_pairs, key=lambda item: (item[0], variant_sort_key(item[1]))
        ):
            self.spot_snvs[barcode].add(snv_key)
            self.snv_expression_counts[snv_key][barcode] = 1
        self.molecule_table_stats.update({
            "g0_record_pairs": len(covered_record_pairs),
            "presence_pairs": len(present_pairs),
            "presence_loci": len({variant for _, variant in present_pairs}),
        })

        total_snvs = sum(len(snvs) for snvs in self.spot_snvs.values())
        spots_with = sum(1 for snvs in self.spot_snvs.values() if snvs)
        logger.info(f"Loaded {total_snvs} total SNV instances across {spots_with} spots")
        logger.info(
            "VCF AD audit: "
            f"ALT-positive={self.vcf_evidence_stats['records_alt_positive']:,}, "
            f"ref-only={self.vcf_evidence_stats['records_ref_only']:,}, "
            f"zero-depth={self.vcf_evidence_stats['records_zero_ad_depth']:,}, "
            f"exact-duplicate={self.vcf_evidence_stats['records_exact_duplicate']:,}"
        )
        race_counts = defaultdict(int)
        for race in self.snv_race.values():
            race_counts[race] += 1
        logger.info(
            "Race distribution - "
            + ", ".join(
                f"{race}: {race_counts[race]}" for race in sorted(race_counts)
            )
        )

    def apply_pools(self):
        """Apply exclusion and inclusion pools to filter SNVs."""
        before_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        
        # Apply exclusion pool
        if self.exclude_snv_pool:
            for barcode in self.spot_snvs:
                self.spot_snvs[barcode] = self.spot_snvs[barcode] - self.exclude_snv_pool
            logger.info(f"Excluded {len(self.exclude_snv_pool)} variants")
        
        # Apply inclusion pool
        if self.include_snv_pool:
            for barcode in self.spot_snvs:
                self.spot_snvs[barcode] = self.spot_snvs[barcode] & self.include_snv_pool
            logger.info(f"Restricted to {len(self.include_snv_pool)} variants")
        
        after_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        logger.info(f"SNVs after pool filtering: {before_total} -> {after_total}")
    
    def build_spatial_graph(self):
        if not self.spot_positions:
            raise ValueError("No spot positions loaded")

        barcodes = list(self.spot_positions.keys())
        positions = np.array([self.spot_positions[b] for b in barcodes])

        # Compute empirical spot pitch from nearest-neighbor distances
        from scipy.spatial import KDTree
        tree = KDTree(positions)
        nn_dists, _ = tree.query(positions, k=2)   # k=2: closest non-self neighbor
        median_pitch = float(np.median(nn_dists[:, 1]))
        neighbor_threshold = self.neighbor_distance * median_pitch
        logger.info(f"Empirical spot pitch: {median_pitch:.1f}px → neighbor threshold: {neighbor_threshold:.1f}px")

        # kNN first (k=13), then apply distance threshold
        k = min(13, len(positions))
        nn = NearestNeighbors(n_neighbors=k).fit(positions)
        dists, idxs = nn.kneighbors(positions)

        self.spot_neighbors = defaultdict(list)
        for i, bc in enumerate(barcodes):
            neighs = []
            for j, dist in zip(idxs[i][1:], dists[i][1:]):   # skip self
                if dist <= neighbor_threshold:
                    neighs.append(barcodes[j])
            self.spot_neighbors[bc] = neighs

        counts = [len(v) for v in self.spot_neighbors.values()]
        if counts:
            logger.info(f"Average neighbors per spot: {np.mean(counts):.2f} "
                        f"(min={np.min(counts)}, max={np.max(counts)}, thresh={neighbor_threshold:.1f}px)")

    
    def calculate_spatial_uniformity_score(self, variant: str) -> float:
        """Calculate spatial uniformity score for a variant."""
        spots_with_variant = [
            bc for bc, alt_count in self.snv_alt_counts.get(variant, {}).items()
            if alt_count > 0 and bc in self.spot_positions
        ]
        
        if len(spots_with_variant) < 2:
            return 0.0
        
        positions = np.array([self.spot_positions[bc] for bc in spots_with_variant])
        distances = distance_matrix(positions, positions)
        dist_flat = distances[np.triu_indices_from(distances, k=1)]
        
        if len(dist_flat) > 0 and np.mean(dist_flat) > 0:
            cv = np.std(dist_flat) / np.mean(dist_flat)
            uniformity_score = 1.0 / (1.0 + cv)
        else:
            uniformity_score = 0.0
        
        return uniformity_score
    
    def calculate_global_prevalence_score(self, variant: str) -> float:
        """Calculate prevalence among spots with read coverage at the locus."""
        covered = self.snv_depth_counts.get(variant, {})
        spots_with_variant = sum(
            1 for bc, depth in covered.items()
            if depth > 0 and self.snv_alt_counts[variant].get(bc, 0) > 0
        )
        total_spots = sum(1 for depth in covered.values() if depth > 0)
        
        if total_spots == 0:
            return 0.0
        
        prevalence = spots_with_variant / total_spots
        score = 1.0 / (1.0 + np.exp(-20 * (prevalence - 0.1)))
        
        return score
    
    def calculate_purity_independence_score(self, variant: str) -> float:
        """Calculate purity independence among locus-covered, finite-purity spots."""
        purities_with = []
        purities_without = []
        
        for barcode, depth in self.snv_depth_counts.get(variant, {}).items():
            if depth <= 0:
                continue
            purity = self.tumor_purity.get(barcode, self.tumor_purity_default)
            if not np.isfinite(purity):
                continue
            if self.snv_alt_counts[variant].get(barcode, 0) > 0:
                purities_with.append(purity)
            else:
                purities_without.append(purity)
        
        if len(purities_with) < 3 or len(purities_without) < 3:
            return 0.5
        
        statistic, pvalue = scipy.stats.ks_2samp(purities_with, purities_without)
        return pvalue
    
    def calculate_purity_correlation_score(self, variant: str) -> float:
        """Calculate positive Spearman correlation between observed VAF and purity."""
        observed_vafs = []
        purities = []
        
        for barcode, depth in self.snv_depth_counts.get(variant, {}).items():
            if depth <= 0:
                continue
            purity = self.tumor_purity.get(barcode, self.tumor_purity_default)
            if not np.isfinite(purity):
                continue
            alt_count = self.snv_alt_counts[variant].get(barcode, 0)
            purities.append(purity)
            observed_vafs.append(alt_count / depth)
        
        if len(observed_vafs) < 6 or len(set(observed_vafs)) < 2 or len(set(purities)) < 2:
            return 0.0
        
        correlation, _ = scipy.stats.spearmanr(observed_vafs, purities)
        score = max(0.0, float(correlation)) if np.isfinite(correlation) else 0.0
        
        return score
    
    def calculate_spatial_clustering_score(self, variant: str) -> float:
        """Calculate clustering using only neighbors covered at the locus."""
        spots_with_variant = [
            bc for bc, alt_count in self.snv_alt_counts.get(variant, {}).items()
            if alt_count > 0 and bc in self.spot_positions
        ]
        
        if len(spots_with_variant) < 2:
            return 0.0
        
        clustering_scores = []
        
        for barcode in spots_with_variant:
            neighbors = [
                neighbor for neighbor in self.spot_neighbors.get(barcode, [])
                if self.snv_depth_counts.get(variant, {}).get(neighbor, 0) > 0
            ]
            if len(neighbors) > 0:
                neighbors_with_variant = sum(
                    1 for neighbor in neighbors
                    if self.snv_alt_counts[variant].get(neighbor, 0) > 0
                )
                cluster_coef = neighbors_with_variant / len(neighbors)
                clustering_scores.append(cluster_coef)
        
        if clustering_scores:
            avg_clustering = np.mean(clustering_scores)
        else:
            avg_clustering = 0.0
        
        return avg_clustering
    
    def parse_variant_for_cnv(self, variant: str) -> Tuple[str, int]:
        """Parse variant to get chromosome and position for CNV lookup."""
        chrom, pos, _, _ = parse_variant_key(variant)
        pos = int(pos)
        return chrom, pos

    def find_cnv_segment(self, variant: str):
        """Return the indexed CalicoST segment containing ``variant``."""
        chrom, pos = self.parse_variant_for_cnv(variant)
        indexed = self.cnv_interval_index.get(str(chrom))
        if not indexed:
            return None
        starts, rows = indexed
        index = bisect_right(starts, pos) - 1
        if index < 0:
            return None
        segment = rows[index]
        return segment if int(segment['END']) >= pos else None

    def _segment_copy_numbers(self, segment, barcode: str) -> Tuple[float, float]:
        """Resolve tumor allele copies for a spot, returning NaN when unknown."""
        if segment is None:
            return np.nan, np.nan
        clone_id = self.clone_labels.get(barcode)
        if clone_id not in self.cnv_clone_ids:
            # CalicoST usually omits a clone0 CNV column because clone0 is the
            # normal compartment.  If there is exactly one tumor clone, its
            # state is the unambiguous tumor component for every mixture spot.
            tumor_clones = list(self.cancer_clones)
            clone_id = tumor_clones[0] if len(tumor_clones) == 1 else None
        if clone_id is None:
            return np.nan, np.nan
        a_col, b_col = f'clone{clone_id}_A', f'clone{clone_id}_B'
        if a_col not in segment.index or b_col not in segment.index:
            return np.nan, np.nan
        try:
            copy_a, copy_b = float(segment[a_col]), float(segment[b_col])
        except (TypeError, ValueError):
            return np.nan, np.nan
        if not (np.isfinite(copy_a) and np.isfinite(copy_b)):
            return np.nan, np.nan
        return copy_a, copy_b

    def calculate_mechanistic_vaf_score(self, variant: str) -> Dict[str, object]:
        """Integrate genotype, AD, purity, CCF and allele-specific CNV evidence."""
        covered = self.snv_depth_counts.get(variant, {})
        barcodes = [
            barcode for barcode, depth in covered.items()
            if depth > 0 and barcode in self.spot_positions
        ]
        segment = self.find_cnv_segment(variant)
        alt_counts, total_counts, purities, copy_a, copy_b = [], [], [], [], []
        for barcode in barcodes:
            alt_counts.append(self.snv_alt_counts[variant].get(barcode, 0))
            total_counts.append(covered[barcode])
            purity = self.tumor_purity.get(barcode, np.nan)
            # Constant/degenerate CalicoST outputs are not treated as measured
            # purity; the model integrates the section prior instead.
            purities.append(purity if self.has_informative_purity else np.nan)
            state_a, state_b = self._segment_copy_numbers(segment, barcode)
            copy_a.append(state_a)
            copy_b.append(state_b)
        return dict(score_mechanistic_vaf(
            alt_counts,
            total_counts,
            purities,
            copy_a,
            copy_b,
            purity_prior=self.purity_prior,
            kappa=self.mechanistic_kappa,
            score_scale=self.mechanistic_score_scale,
        ))
    
    def calculate_clone_enrichment_score(self, variant: str) -> float:
        """
        Calculate soft probabilistic score based on clone enrichment.
        
        Returns score 0-1 based on cancer/normal enrichment ratio.
        Higher score = more enriched in cancer clones.
        """
        if not self.use_clone_cnv or not self.clone_labels:
            return 0.5  # Neutral if no clone data
        
        prevalence_by_clone = {}
        
        # Calculate prevalence in each clone
        for clone_id in self.all_clones:
            spots_in_clone = [
                bc for bc, cid in self.clone_labels.items()
                if cid == clone_id and self.snv_depth_counts.get(variant, {}).get(bc, 0) > 0
            ]
            
            if len(spots_in_clone) == 0:
                prevalence_by_clone[clone_id] = 0.0
                continue
            
            spots_with_variant = sum(
                1 for bc in spots_in_clone if self.snv_alt_counts[variant].get(bc, 0) > 0
            )
            
            prevalence_by_clone[clone_id] = spots_with_variant / len(spots_in_clone)
        
        # Get normal vs cancer prevalence
        prev_normal = prevalence_by_clone.get(self.reference_clone, 0.0)
        
        if not self.cancer_clones:
            return 0.5
        
        prev_cancer_values = [prevalence_by_clone[cid] for cid in self.cancer_clones]
        prev_cancer_mean = np.mean(prev_cancer_values)
        
        # Calculate enrichment with smoothing
        epsilon = 0.05  # Pseudocount to avoid division by zero
        enrichment = (prev_cancer_mean + epsilon) / (prev_normal + epsilon)
        
        # Transform to 0-1 score
        if enrichment >= 1:
            # Cancer enrichment: smoothly scale from 0.5 to 1.0
            score = 0.5 + 0.5 * (1 - np.exp(-0.5 * (enrichment - 1)))
        else:
            # Normal enrichment: linearly scale from 0.5 to 0.0
            score = 0.5 * enrichment
        
        return score
    
    def calculate_cnv_consistency_score(self, variant: str) -> float:
        """
        Calculate score based on consistency with CNV patterns.
        
        Returns MAX consistency across cancer clones.
        """
        if not self.use_clone_cnv or self.cnv_segments is None:
            return 0.5  # Neutral if no CNV data
        
        segment = self.find_cnv_segment(variant)
        if segment is None:
            return 0.5  # No CNV info — segment not found for this chromosome/position

        # Analyze each cancer clone
        consistency_scores = []

        for clone_id in self.cancer_clones:
            # Column names are normalized to underscore at load time (was 'clone1 A')
            cn_a_col = f'clone{clone_id}_A'
            cn_b_col = f'clone{clone_id}_B'

            # Check if columns exist (should always pass after normalization)
            if cn_a_col not in segment.index or cn_b_col not in segment.index:
                logger.warning(f"CNV columns {cn_a_col}/{cn_b_col} not found in segment. "
                               f"Available: {list(segment.index)}")
                continue
            
            cn_a = segment[cn_a_col]
            cn_b = segment[cn_b_col]
            total_cn = cn_a + cn_b
            
            # Check if CNV-altered
            is_altered = (cn_a != 1 or cn_b != 1)
            is_loh = (cn_a == 0 or cn_b == 0) and total_cn > 0
            
            # Get variant prevalence in this clone
            spots_in_clone = [
                bc for bc, cid in self.clone_labels.items()
                if cid == clone_id and self.snv_depth_counts.get(variant, {}).get(bc, 0) > 0
            ]
            if len(spots_in_clone) < 3:
                continue
            
            spots_with_variant = sum(
                1 for bc in spots_in_clone if self.snv_alt_counts[variant].get(bc, 0) > 0
            )
            variant_prevalence = spots_with_variant / len(spots_in_clone)
            
            # Score based on CNV-variant association
            clone_score = 0.0
            
            if is_altered:
                # Variant in CNV-altered region
                clone_score = variant_prevalence
                
                # Boost for LOH (strong somatic evidence)
                if is_loh and variant_prevalence > 0.5:
                    clone_score *= 1.3
                elif total_cn > 2 and variant_prevalence > 0.4:
                    clone_score *= 1.2
                
                clone_score = min(1.0, clone_score)
            else:
                # Copy-neutral region - less informative
                clone_score = 0.5
            
            consistency_scores.append(clone_score)
        
        if not consistency_scores:
            return 0.5
        
        # Return MAX consistency (detect clone-specific events)
        return max(consistency_scores)

    def calculate_clone_enrichment_theta(self, variant: str) -> float:
        """Depth-stratified tumor-vs-reference ALT enrichment (MH log OR)."""
        if not self.use_clone_cnv or self.reference_clone is None or not self.cancer_clones:
            return np.nan
        group_rows = []
        for barcode, depth in self.snv_depth_counts.get(variant, {}).items():
            if depth <= 0:
                continue
            clone_id = self.clone_labels.get(barcode)
            if clone_id == self.reference_clone:
                group = 0
            elif clone_id in self.cancer_clones:
                group = 1
            else:
                continue
            alt_present = int(
                self.snv_alt_counts.get(variant, {}).get(barcode, 0)
                >= 1
            )
            stratum = 0 if depth == 1 else 1 if depth == 2 else 2 if depth <= 4 else 3
            group_rows.append((group, alt_present, stratum))
        if sum(group == 1 for group, _, _ in group_rows) < 3:
            return np.nan
        if sum(group == 0 for group, _, _ in group_rows) < 3:
            return np.nan
        numerator = 0.0
        denominator = 0.0
        for stratum in range(4):
            rows = [(group, alt) for group, alt, level in group_rows if level == stratum]
            if not rows:
                continue
            a = sum(group == 1 and alt == 1 for group, alt in rows) + 0.5
            b = sum(group == 1 and alt == 0 for group, alt in rows) + 0.5
            c = sum(group == 0 and alt == 1 for group, alt in rows) + 0.5
            d = sum(group == 0 and alt == 0 for group, alt in rows) + 0.5
            n = a + b + c + d
            numerator += a * d / n
            denominator += b * c / n
        if numerator <= 0 or denominator <= 0:
            return np.nan
        return float(np.log(numerator / denominator))

    def calculate_theta_score(self, variant: str) -> float:
        if self.cnv_feature == "legacy":
            return self.calculate_cnv_consistency_score(variant)
        return self.calculate_clone_enrichment_theta(variant)
    
    def calculate_germline_score(self, variant: str) -> float:
        """Calculate combined germline score."""
        s_uniform = self.calculate_spatial_uniformity_score(variant)
        s_prevalence = self.calculate_global_prevalence_score(variant)
        s_independence = self.calculate_purity_independence_score(variant)
        
        score = (self.weights_germline['alpha'] * s_uniform +
                self.weights_germline['beta'] * s_prevalence +
                self.weights_germline['gamma'] * s_independence)
        
        return score
    
    def calculate_somatic_score(self, variant: str) -> float:
        """Calculate combined somatic score with optional clone+CNV enhancement."""
        if self.evidence_mode == "spatial_only_no_calicost":
            return self.calculate_spatial_clustering_score(variant)

        # Base components (always calculated)
        s_purity = self.calculate_purity_correlation_score(variant)
        s_cluster = self.calculate_spatial_clustering_score(variant)
        s_clone_specific = s_cluster * s_purity  # Simplified clone-specific score
        
        if self.use_clone_cnv:
            # Enhanced scoring with clone+CNV data
            s_clone_enrich = self.calculate_clone_enrichment_score(variant)
            s_cnv_consist = self.calculate_theta_score(variant)
            
            score = (self.weights_somatic['delta'] * s_purity +
                    self.weights_somatic['epsilon'] * s_clone_specific +
                    self.weights_somatic['zeta'] * s_cluster +
                    self.weights_somatic['eta'] * s_clone_enrich +
                    self.weights_somatic['theta'] * s_cnv_consist)
        else:
            # Original scoring (purity-only)
            score = (self.weights_somatic['delta'] * s_purity +
                    self.weights_somatic['epsilon'] * s_clone_specific +
                    self.weights_somatic['zeta'] * s_cluster)
        
        return score
    
    def has_observed_alt_support(self, variant: str) -> bool:
        """Return whether S6+ observed an ALT molecule at this exact allele."""
        return variant in self.molecule_alt_supported_variants
    
    def classify_variants(self, 
                          germline_threshold: float = 0.5,
                          somatic_threshold: float = 0.2):
        """
        Classify variants using a two-stage germline-exclusion + somatic-voting pipeline.

        Stage 1 — Germline exclusion (feature threshold):
            Denovo variants with spatial_uniformity > germline_threshold (default 0.5)
            AND global_prevalence > somatic_threshold (default 0.2) are called germline.

        Stage 2 — Somatic voting (rank-based):
            Among remaining denovo variants, each of the somatic features
            (purity_correlation, spatial_clustering, clone_specific_proxy,
            cnv_consistency[if available]) casts a vote for its top-20% variants.
            The top-10% variants by total vote count are called somatic;
            the rest are ambiguous.
        """
        logger.info("Classifying variants (germline exclusion + somatic voting)...")

        # ── Collect all unique variants ──
        all_variants = set()
        for snvs in self.spot_snvs.values():
            all_variants.update(snvs)
        if self.kept_variants_pool:
            all_variants.update(self.kept_variants_pool)
        logger.info(f"Total unique variants to classify: {len(all_variants)}")

        # ── Step 0: Defined and kept variants (always germline, skip scoring) ──
        denovo_candidates = []
        for variant in sorted(all_variants, key=variant_sort_key):
            race = self.snv_race.get(variant, "unknown")
            if race == "defined":
                self.germline_variants.add(variant)
                self.germline_defined.add(variant)
                self.variant_scores[variant] = {
                    'alpha': 1.0, 'beta': 1.0,
                    'germline': 1.0, 'somatic': 0.0, 'votes': 0
                }
                continue
            if variant in self.kept_variants_pool:
                self.germline_variants.add(variant)
                self.germline_denovo.add(variant)
                self.germline_kept.add(variant)   # tracked separately — scores are forced, not computed
                self.variant_scores[variant] = {
                    'alpha': 1.0, 'beta': 1.0,
                    'germline': 1.0, 'somatic': 0.0, 'votes': 0
                }
                continue
            denovo_candidates.append(variant)

        # ── Step 1: Germline exclusion ──
        # Threshold: spatial_uniformity (alpha) > 0.5  AND  global_prevalence (beta) > 0.2
        GERMLINE_ALPHA_THRESHOLD = germline_threshold   # 0.5
        GERMLINE_BETA_THRESHOLD  = somatic_threshold    # 0.2

        non_germline = []
        for variant in denovo_candidates:
            alpha = self.calculate_spatial_uniformity_score(variant)
            beta  = self.calculate_global_prevalence_score(variant)
            is_germline = (alpha > GERMLINE_ALPHA_THRESHOLD and beta > GERMLINE_BETA_THRESHOLD)
            self.variant_scores[variant] = {
                'alpha': alpha, 'beta': beta,
                'germline': 1.0 if is_germline else 0.0,
                'somatic': 0.0, 'votes': 0
            }
            if is_germline:
                self.germline_variants.add(variant)
                self.germline_denovo.add(variant)
            else:
                non_germline.append(variant)

        logger.info(f"Stage 1 — Germline exclusion: {len(self.germline_denovo)} denovo germline, "
                    f"{len(non_germline)} proceed to somatic voting")

        # Compute the biological VAF evidence for every candidate, including
        # known germline controls.  In the default ``shadow`` mode these fields
        # are audit-only and therefore cannot change a classification.  ``vote``
        # is an explicit opt-in used for the planned impact comparison.
        if self.mechanistic_vaf_mode != "off":
            logger.info(
                f"Computing mechanistic CNV/purity/VAF evidence in "
                f"{self.mechanistic_vaf_mode!r} mode"
            )
            for index, variant in enumerate(sorted(all_variants), start=1):
                evidence = self.calculate_mechanistic_vaf_score(variant)
                self.variant_scores[variant].update({
                    'vaf_log_bf': evidence['log_bf'],
                    'vaf_score': evidence['score'],
                    'vaf_information': evidence['information'],
                    'vaf_context_reliability': evidence['context_reliability'],
                    'vaf_observed': evidence['observed_vaf'],
                    'vaf_somatic_low': evidence['somatic_vaf_low'],
                    'vaf_somatic_high': evidence['somatic_vaf_high'],
                    'vaf_germline_low': evidence['germline_vaf_low'],
                    'vaf_germline_high': evidence['germline_vaf_high'],
                    'vaf_cnv_scenario': evidence['cnv_scenario'],
                    'vaf_covered_spots': evidence['covered_spots'],
                    'vaf_total_depth': evidence['total_depth'],
                    'vaf_purity_fraction': evidence.get('purity_fraction', 0.0),
                    'vaf_cnv_fraction': evidence.get('cnv_fraction', 0.0),
                })
                if index % 5000 == 0:
                    logger.info(
                        f"  Mechanistic VAF evidence: {index:,}/{len(all_variants):,} variants"
                    )

        # ── Step 2: Somatic voting ──
        # Features used for voting (clone enrichment excluded by design)
        VOTE_TOP_FRACTION   = 0.20   # top 20% per feature → 1 vote
        SOMATIC_TOP_FRACTION = 0.10  # top 10% by votes → somatic

        # Determine which features participate in voting. The ablation uses
        # zeta only; full mode adds each CalicoST-derived feature only when its
        # underlying input is informative.
        voting_features = ['zeta']
        if self.has_informative_purity:
            voting_features = ['delta', 'epsilon', 'zeta']
        if self.use_clone_cnv:
            voting_features.append('theta')
        if self.mechanistic_vaf_mode == "vote":
            voting_features.append('vaf_mechanistic')
        self.voting_features = voting_features
        n_features = len(voting_features)
        logger.info(f"Stage 2 — Somatic voting with {n_features} features: {voting_features}")

        # Calculate somatic feature scores for all non-germline denovo variants
        feature_scores = {f: {} for f in voting_features}
        for variant in non_germline:
            s_cluster = self.calculate_spatial_clustering_score(variant)
            feature_scores['zeta'][variant] = s_cluster
            if self.has_informative_purity:
                s_purity = self.calculate_purity_correlation_score(variant)
                s_clone_specific = s_cluster * s_purity
                feature_scores['delta'][variant] = s_purity
                feature_scores['epsilon'][variant] = s_clone_specific
                self.variant_scores[variant]['delta'] = s_purity
                self.variant_scores[variant]['epsilon'] = s_clone_specific
            if self.use_clone_cnv:
                feature_scores['theta'][variant] = self.calculate_theta_score(variant)
            if self.mechanistic_vaf_mode == "vote":
                information = self.variant_scores[variant].get('vaf_information', 0.0)
                context = self.variant_scores[variant].get('vaf_context_reliability', 0.0)
                if information >= self.mechanistic_min_information and context > 0:
                    feature_scores['vaf_mechanistic'][variant] = self.variant_scores[variant]['vaf_score']
            # Store in variant_scores for downstream use / plotting
            self.variant_scores[variant]['zeta'] = s_cluster
            if self.use_clone_cnv:
                self.variant_scores[variant]['theta'] = feature_scores['theta'][variant]

        # Each feature votes for its STRICTLY top-20% variants by rank
        # (sorted slice avoids tie inflation at the boundary)
        vote_counts = defaultdict(int)
        for feat in voting_features:
            scores = feature_scores[feat]
            finite_scores = {
                variant: score for variant, score in scores.items() if np.isfinite(score)
            }
            if not finite_scores:
                continue
            n_voters = max(1, int(len(finite_scores) * VOTE_TOP_FRACTION))  # floor, not ceil
            top_variants = sorted(
                finite_scores,
                key=lambda variant: (-finite_scores[variant], variant),
            )[:n_voters]
            for variant in top_variants:
                vote_counts[variant] += 1

        # Log vote distribution for diagnostics
        vote_hist = defaultdict(int)
        for v in non_germline:
            vote_hist[vote_counts[v]] += 1
        logger.info(f"Vote distribution (votes: count): "
                    + ", ".join(f"{k}:{vote_hist[k]}" for k in sorted(vote_hist)))

        # Somatic selection:
        #   Gate 1 — must have received at least 1 vote
        #   Gate 2 — among voted variants, take at most top 10% of non_germline by vote count
        somatic_cap = max(1, int(len(non_germline) * SOMATIC_TOP_FRACTION))  # floor = strict 10%
        # Resolve vote-count ties by summed feature evidence, then variant key.
        # This is especially important in the one-feature ablation, where every
        # top-zeta candidate otherwise has exactly one vote.
        tie_break_scores = {
            variant: sum(
                np.nan_to_num(feature_scores[feature].get(variant, 0.0), nan=0.0)
                for feature in voting_features
            )
            for variant in non_germline
        }
        voted_variants = sorted(
            [v for v in non_germline if vote_counts[v] >= 1],
            key=lambda v: (-vote_counts[v], -tie_break_scores[v], v),
        )
        somatic_set = set(voted_variants[:somatic_cap])

        for variant in non_germline:
            votes = vote_counts[variant]
            self.variant_scores[variant]['votes']   = votes
            self.variant_scores[variant]['somatic'] = votes / n_features  # normalised [0,1]
            if variant in somatic_set:
                self.somatic_variants.add(variant)
                self.somatic_denovo.add(variant)
            else:
                self.ambiguous_variants.add(variant)
                self.ambiguous_denovo.add(variant)

        logger.info(f"Classification results:")
        logger.info(f"  Germline : {len(self.germline_variants)} "
                    f"(defined: {len(self.germline_defined)}, denovo: {len(self.germline_denovo)})")
        logger.info(f"  Somatic  : {len(self.somatic_variants)} "
                    f"(voted: {len(voted_variants)}, cap: {somatic_cap})")
        logger.info(f"  Ambiguous: {len(self.ambiguous_variants)} (denovo only)")
        logger.info(f"  Somatic min-votes: {vote_counts.get(voted_variants[len(somatic_set)-1], 0) if somatic_set else 'N/A'}")

        # Distribute variants to spots
        self.distribute_variants_to_spots()
    
    def distribute_variants_to_spots(self):
        """Distribute classified variants back to individual spots."""
        for barcode, snvs in self.spot_snvs.items():
            for snv in snvs:
                if snv in self.germline_variants:
                    self.filtered_spot_snvs_germline[barcode].add(snv)
                elif snv in self.somatic_variants:
                    self.filtered_spot_snvs_somatic[barcode].add(snv)
                elif snv in self.ambiguous_variants:
                    self.filtered_spot_snvs_ambiguous[barcode].add(snv)
    
    def save_text_outputs(self):
        """Save variant lists to text files, split by classification and race."""
        # Define all variant sets with their directories and filenames
        variant_configs = [
            ("germline_defined", self.germline_defined, self.germline_defined_dir),
            ("germline_denovo", self.germline_denovo, self.germline_denovo_dir),
            ("somatic_defined", self.somatic_defined, self.somatic_defined_dir),
            ("somatic_denovo", self.somatic_denovo, self.somatic_denovo_dir),
            ("ambiguous_defined", self.ambiguous_defined, self.ambiguous_defined_dir),
            ("ambiguous_denovo", self.ambiguous_denovo, self.ambiguous_denovo_dir),
            # Also save combined files in parent directories
            ("germline_variants", self.germline_variants, self.germline_dir),
            ("somatic_variants", self.somatic_variants, self.somatic_dir),
            ("ambiguous_variants", self.ambiguous_variants, self.ambiguous_dir),
        ]
        
        # Save variant lists for each category
        for variant_name, variant_set, dir_path in variant_configs:
            if not variant_set:
                continue
            
            output_file = os.path.join(dir_path, f"{variant_name}.txt")
            with open(output_file, 'w') as f:
                f.write("chrom\tpos\tref\talt\tgermline_score\tsomatic_score\trace\ttheta\n")
                for variant in sorted(variant_set, key=variant_sort_key):
                    chrom, pos, ref, alt = parse_variant_key(variant)
                    scores = self.variant_scores.get(variant, {'germline': 0, 'somatic': 0})
                    race = self.snv_race.get(variant, "unknown")
                    theta = scores.get('theta', np.nan)
                    theta_text = f"{float(theta):.6g}" if np.isfinite(theta) else "NA"
                    f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t"
                           f"{scores['germline']:.4f}\t{scores['somatic']:.4f}\t{race}\t"
                           f"{theta_text}\n")
        
        # Save all scores — including per-feature breakdown for diagnostic analysis
        scores_file = os.path.join(self.output_dir, "all_variant_scores.txt")
        with open(scores_file, 'w') as f:
            # Header: composite scores + all individual feature components
            f.write("variant\tgermline_score\tsomatic_score\tclassification\trace"
                    "\tf_spatial_uniformity\tf_global_prevalence\tf_purity_independence"
                    "\tf_purity_correlation\tf_spatial_clustering\tf_clone_specific_proxy"
                    "\tf_clone_enrichment\tf_cnv_consistency"
                    "\tvaf_log_bf\tvaf_score\tvaf_information\tvaf_context_reliability"
                    "\tvaf_observed\tvaf_somatic_low\tvaf_somatic_high"
                    "\tvaf_germline_low\tvaf_germline_high\tvaf_cnv_scenario"
                    "\tvaf_covered_spots\tvaf_total_depth\tvaf_purity_fraction"
                    "\tvaf_cnv_fraction\n")
            for variant in sorted(self.variant_scores, key=variant_sort_key):
                scores = self.variant_scores[variant]
                if variant in self.germline_variants:
                    classification = "germline"
                elif variant in self.somatic_variants:
                    classification = "somatic"
                else:
                    classification = "ambiguous"
                race = self.snv_race.get(variant, "unknown")

                # Re-compute individual features (cheap; already cached in score functions
                # only when race==defined they are skipped and we write NA)
                if race == "defined":
                    feats = "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
                else:
                    # Reuse the exact scores already calculated during
                    # classification. This avoids a second O(variants × spots)
                    # pass, especially in the large P6 ablation.
                    f_unif = scores.get('alpha')
                    if f_unif is None:
                        f_unif = self.calculate_spatial_uniformity_score(variant)
                    f_prev = scores.get('beta')
                    if f_prev is None:
                        f_prev = self.calculate_global_prevalence_score(variant)
                    f_clust = scores.get('zeta')
                    if f_clust is None:
                        f_clust = self.calculate_spatial_clustering_score(variant)
                    if self.has_informative_purity:
                        f_indep = self.calculate_purity_independence_score(variant)
                        f_pcorr = scores.get('delta')
                        if f_pcorr is None:
                            f_pcorr = self.calculate_purity_correlation_score(variant)
                        f_csprx = scores.get('epsilon', f_clust * f_pcorr)
                    else:
                        f_indep = f_pcorr = f_csprx = float('nan')
                    f_cenr  = (self.calculate_clone_enrichment_score(variant)
                               if self.use_clone_cnv else float('nan'))
                    f_cnv   = (self.calculate_theta_score(variant)
                               if self.use_clone_cnv else float('nan'))
                    fmt = lambda value: ('NA' if np.isnan(value) else f'{value:.4f}')
                    feats = (f"\t{f_unif:.4f}\t{f_prev:.4f}\t{fmt(f_indep)}"
                             f"\t{fmt(f_pcorr)}\t{f_clust:.4f}\t{fmt(f_csprx)}"
                             f"\t{f_cenr if not (isinstance(f_cenr, float) and np.isnan(f_cenr)) else 'NA'}"
                             f"\t{f_cnv if not (isinstance(f_cnv, float) and np.isnan(f_cnv)) else 'NA'}")

                def score_text(key, digits=4):
                    value = scores.get(key, np.nan)
                    try:
                        return 'NA' if not np.isfinite(float(value)) else f'{float(value):.{digits}f}'
                    except (TypeError, ValueError):
                        return 'NA'

                vaf_fields = [
                    score_text('vaf_log_bf'), score_text('vaf_score'),
                    score_text('vaf_information'), score_text('vaf_context_reliability'),
                    score_text('vaf_observed'), score_text('vaf_somatic_low'),
                    score_text('vaf_somatic_high'), score_text('vaf_germline_low'),
                    score_text('vaf_germline_high'),
                    str(scores.get('vaf_cnv_scenario', 'NA')),
                    score_text('vaf_covered_spots', 0), score_text('vaf_total_depth', 0),
                    score_text('vaf_purity_fraction'), score_text('vaf_cnv_fraction'),
                ]
                f.write(f"{variant}\t{scores['germline']:.4f}\t"
                       f"{scores['somatic']:.4f}\t{classification}\t{race}{feats}\t"
                       + "\t".join(vaf_fields) + "\n")

        if self.mechanistic_vaf_mode != "off":
            evidence_file = os.path.join(self.output_dir, "mechanistic_vaf_scores.tsv")
            score_frame = pd.read_csv(scores_file, sep='\t')
            evidence_columns = [
                'variant', 'classification', 'race',
                *[column for column in score_frame.columns if column.startswith('vaf_')],
            ]
            score_frame[evidence_columns].to_csv(evidence_file, sep='\t', index=False)
        
        # Save per-barcode filtered SNVs
        for variant_type, spot_snvs in [("germline", self.filtered_spot_snvs_germline),
                                        ("somatic", self.filtered_spot_snvs_somatic)]:
            dir_path = self.germline_dir if variant_type == "germline" else self.somatic_dir
            
            for barcode, snvs in spot_snvs.items():
                if not snvs:
                    continue
                
                output_file = os.path.join(dir_path, f"{barcode}.txt")
                with open(output_file, 'w') as f:
                    f.write("chrom\tpos\tref\talt\trace\n")
                    for snv in sorted(snvs, key=variant_sort_key):
                        chrom, pos, ref, alt = parse_variant_key(snv)
                        race = self.snv_race.get(snv, "unknown")
                        f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{race}\n")
    
    def create_vcf_header(self) -> str:
        """Create VCF header."""
        header = []
        header.append("##fileformat=VCFv4.2")
        header.append(f"##fileDate={datetime.now().strftime('%Y%m%d')}")
        header.append(f"##source=EnhancedSpatialSNVFilter")
        header.append(f"##reference=GRCh38")
        header.append('##INFO=<ID=NS,Number=1,Type=Integer,Description="Number of spots with variant">')
        header.append('##INFO=<ID=NCOV,Number=1,Type=Integer,Description="Number of spots with biallelic AD coverage">')
        header.append('##INFO=<ID=AF,Number=1,Type=Float,Description="ALT-positive fraction among covered spots">')
        header.append('##INFO=<ID=SCORE,Number=1,Type=Float,Description="Classification score">')
        header.append('##INFO=<ID=PURITY_CORR,Number=1,Type=Float,Description="Correlation with tumor purity">')
        header.append('##INFO=<ID=THETA,Number=1,Type=Float,Description="Selected CNV feature score">')
        header.append('##INFO=<ID=VAFBF,Number=1,Type=Float,Description="Mechanistic somatic-vs-germline VAF log Bayes factor">')
        header.append('##INFO=<ID=VAFSCORE,Number=1,Type=Float,Description="Reliability-shrunk mechanistic VAF score">')
        header.append('##INFO=<ID=VAFINFO,Number=1,Type=Float,Description="Mechanistic VAF read-information factor">')
        header.append('##INFO=<ID=CNVSCEN,Number=1,Type=String,Description="Allele-specific CNV scenario">')
        header.append('##INFO=<ID=RACE,Number=1,Type=String,Description="SNV origin: defined (panel-concordant), panel_discordant, or denovo (panel-absent)">')
        header.append('##FILTER=<ID=PASS,Description="Passed all filters">')
        header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
        
        return '\n'.join(header)
    
    def save_vcf_outputs(self):
        """Save variants to VCF files, split by classification and race."""
        # Precompute spot prevalence once. Re-scanning every spot for every
        # variant made P6 VCF writing quadratic in practice.
        variant_spot_counts = defaultdict(int)
        for spot_variants in self.spot_snvs.values():
            for variant in spot_variants:
                variant_spot_counts[variant] += 1

        # Define all variant sets with their directories and filenames
        variant_configs = [
            ("germline_defined", self.germline_defined, self.germline_defined_dir),
            ("germline_denovo", self.germline_denovo, self.germline_denovo_dir),
            ("somatic_defined", self.somatic_defined, self.somatic_defined_dir),
            ("somatic_denovo", self.somatic_denovo, self.somatic_denovo_dir),
            ("ambiguous_defined", self.ambiguous_defined, self.ambiguous_defined_dir),
            ("ambiguous_denovo", self.ambiguous_denovo, self.ambiguous_denovo_dir),
            # Also save combined files in parent directories
            ("germline_variants", self.germline_variants, self.germline_dir),
            ("somatic_variants", self.somatic_variants, self.somatic_dir),
            ("ambiguous_variants", self.ambiguous_variants, self.ambiguous_dir),
        ]
        
        for variant_name, variant_set, dir_path in variant_configs:
            if not variant_set:
                logger.info(f"No variants for {variant_name}, skipping")
                continue
            
            vcf_file = os.path.join(dir_path, f"{variant_name}.vcf")
            
            with open(vcf_file, 'w') as f:
                # Write header
                f.write(self.create_vcf_header() + '\n')
                
                # Sort variants by chromosome and position
                sorted_variants = []
                for variant in variant_set:
                    sorted_variants.append((*variant_sort_key(variant), variant))
                
                sorted_variants.sort()
                
                # Write variants
                for *_, variant in sorted_variants:
                    chrom, pos, ref, alt = parse_variant_key(variant)
                    
                    # Calculate INFO fields
                    ns = variant_spot_counts.get(variant, 0)
                    covered_spots = sum(
                        1 for depth in self.snv_depth_counts.get(variant, {}).values()
                        if depth > 0
                    )
                    af = ns / covered_spots if covered_spots > 0 else 0
                    
                    scores = self.variant_scores.get(variant, {'germline': 0, 'somatic': 0})
                    # Determine which score to use
                    if "germline" in variant_name:
                        score = scores['germline']
                    elif "somatic" in variant_name:
                        score = scores['somatic']
                    else:  # ambiguous
                        score = max(scores['germline'], scores['somatic'])
                    
                    purity_corr = (f"{self.calculate_purity_correlation_score(variant):.4f}"
                                   if self.has_informative_purity else ".")
                    race = self.snv_race.get(variant, "unknown")
                    
                    def info_float(key):
                        value = scores.get(key, np.nan)
                        return f"{float(value):.4f}" if np.isfinite(float(value)) else "."

                    info = (
                        f"NS={ns};NCOV={covered_spots};AF={af:.4f};SCORE={score:.4f};"
                        f"PURITY_CORR={purity_corr};THETA={info_float('theta')};"
                        f"VAFBF={info_float('vaf_log_bf')};"
                        f"VAFSCORE={info_float('vaf_score')};VAFINFO={info_float('vaf_information')};"
                        f"CNVSCEN={scores.get('vaf_cnv_scenario', '.')};RACE={race}"
                    )
                    
                    # Ensure chr prefix
                    if chrom.startswith("chr"):
                        f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
                    else:
                        f.write(f"chr{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
            
            # Compression and indexing are part of the output contract. Any
            # failure leaves the run without RUN_COMPLETE and must propagate.
            subprocess.run(["bgzip", "-f", vcf_file], check=True)
            subprocess.run(["tabix", "-p", "vcf", vcf_file + ".gz"], check=True)
            logger.info(f"Created VCF: {vcf_file}.gz ({len(variant_set)} variants)")
    
    def save_summary(self):
        """Save comprehensive summary of the analysis."""
        summary_file = os.path.join(self.output_dir, "spatial_filter_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Enhanced Spatial SNV Filtering Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n")
            f.write(f"Evidence Mode: {self.evidence_mode}\n")
            f.write(f"Mechanistic VAF Mode: {self.mechanistic_vaf_mode}\n")
            f.write(f"Tumor Purity File: {self.tumor_purity_file}\n")
            f.write(f"CalicoST Gate: {self.calicost_gate_mode}\n")
            f.write(f"CNV Feature: {self.cnv_feature}\n")
            f.write(f"Presence Rule: {self.presence_rule}\n")
            f.write(f"Molecule Table: {Path(self.molecule_table_file).resolve()}\n")
            f.write(f"Molecule Table SHA256: {self._sha256_file(self.molecule_table_file)}\n")
            if self.calicost_gate_result is not None:
                gate = self.calicost_gate_result
                gate_path = os.path.join(self.output_dir, "calicost_gate.json")
                f.write(
                    f"Gate Result: CNV={'PASS' if gate['cnv_pass'] else 'FAIL'}, "
                    f"purity={'PASS' if gate['purity_pass'] else 'FAIL'}, "
                    f"reference={gate['reference_clone']}, tumors={gate['tumor_clones']}, "
                    f"dropped={gate['dropped_clones']}\n"
                )
                f.write(f"Gate JSON: {gate_path}\n")
                f.write(f"Gate JSON SHA256: {self._sha256_file(gate_path)}\n")
            f.write("\n")
            
            f.write("Parameters:\n")
            f.write(f"  Neighbor distance: {self.neighbor_distance}\n")
            f.write(
                "  Per-spot variant presence: "
                + {
                    "G0": "every in-tissue step-6 VCF record",
                    "G1": "G0 values at loci with >=1 ALT molecule in the section",
                    "G2": "only spot/locus records with >=1 ALT molecule",
                }[self.presence_rule]
                + "\n"
            )
            f.write(f"  Molecule table statistics: {self.molecule_table_stats}\n")
            f.write(
                "  Deprecated class-specific min-expression CLI values: "
                f"{self.deprecated_min_expression_germline}/"
                f"{self.deprecated_min_expression_somatic} (not classification gates)\n"
            )
            f.write(f"  Somatic voting features: {','.join(self.voting_features)}\n\n")
            
            f.write("Input Data:\n")
            f.write(f"  Total spots: {len(self.spot_positions)}\n")
            f.write(f"  Spots with SNVs: {len(self.spot_snvs)}\n")
            f.write(f"  Total SNV instances: {sum(len(snvs) for snvs in self.spot_snvs.values())}\n")
            f.write(f"  VCF records scanned: {self.vcf_evidence_stats['records_total']}\n")
            f.write(f"  ALT-positive AD records: {self.vcf_evidence_stats['records_alt_positive']}\n")
            f.write(f"  Reference-only AD records: {self.vcf_evidence_stats['records_ref_only']}\n")
            f.write(f"  Zero-depth AD records: {self.vcf_evidence_stats['records_zero_ad_depth']}\n")
            f.write(
                f"  Missing/non-biallelic AD records: "
                f"{self.vcf_evidence_stats['records_without_biallelic_ad']}\n"
            )
            
            all_variants = set()
            for snvs in self.spot_snvs.values():
                all_variants.update(snvs)
            f.write(f"  Unique variants: {len(all_variants)}\n\n")
            
            f.write("Tumor Purity Statistics:\n")
            purities = np.asarray(list(self.tumor_purity.values()), dtype=float)
            finite_purities = purities[np.isfinite(purities)]
            f.write(f"  Finite: {len(finite_purities)}/{len(purities)}\n")
            f.write(f"  Informative: {self.has_informative_purity}\n")
            if finite_purities.size:
                f.write(f"  Min: {np.min(finite_purities):.3f}\n")
                f.write(f"  Max: {np.max(finite_purities):.3f}\n")
                f.write(f"  Mean: {np.mean(finite_purities):.3f}\n")
                f.write(f"  Median: {np.median(finite_purities):.3f}\n\n")

            if self.mechanistic_vaf_mode != "off":
                evidence_scores = np.asarray([
                    scores.get('vaf_log_bf', np.nan)
                    for scores in self.variant_scores.values()
                ], dtype=float)
                evidence_scores = evidence_scores[np.isfinite(evidence_scores)]
                f.write("Mechanistic VAF Evidence:\n")
                f.write(f"  Scored variants: {len(evidence_scores)}\n")
                if evidence_scores.size:
                    f.write(f"  Median log BF: {np.median(evidence_scores):.3f}\n")
                    f.write(f"  Positive log BF: {np.sum(evidence_scores > 0)}\n")
                    f.write(f"  Strong log BF (>5): {np.sum(evidence_scores > 5)}\n\n")
            
            if self.exclude_snv_pool:
                f.write(f"Exclusion pool: {len(self.exclude_snv_pool)} variants\n")
            if self.include_snv_pool:
                f.write(f"Inclusion pool: {len(self.include_snv_pool)} variants\n")
            if self.kept_variants_pool:
                f.write(f"Kept variants pool: {len(self.kept_variants_pool)} variants\n")
            
            f.write("\nClassification Results:\n")
            f.write(f"  Germline variants: {len(self.germline_variants)}\n")
            f.write(f"    - Defined: {len(self.germline_defined)}\n")
            f.write(f"    - Denovo: {len(self.germline_denovo)}\n")
            f.write(f"  Somatic variants: {len(self.somatic_variants)}\n")
            f.write(f"    - Defined: {len(self.somatic_defined)}\n")
            f.write(f"    - Denovo: {len(self.somatic_denovo)}\n")
            f.write(f"  Ambiguous variants: {len(self.ambiguous_variants)}\n")
            f.write(f"    - Defined: {len(self.ambiguous_defined)}\n")
            f.write(f"    - Denovo: {len(self.ambiguous_denovo)}\n\n")
            
            f.write("Race Distribution:\n")
            race_counts = {"defined": 0, "denovo": 0, "unknown": 0}
            for race in self.snv_race.values():
                race_counts[race] = race_counts.get(race, 0) + 1
            f.write(f"  Defined (from Beagle): {race_counts.get('defined', 0)}\n")
            f.write(f"  Denovo (from Classifier): {race_counts.get('denovo', 0)}\n")
            f.write(f"  Unknown: {race_counts.get('unknown', 0)}\n\n")
            
            f.write("Per-Spot Distribution:\n")
            germline_spots = sum(1 for snvs in self.filtered_spot_snvs_germline.values() if snvs)
            somatic_spots = sum(1 for snvs in self.filtered_spot_snvs_somatic.values() if snvs)
            f.write(f"  Spots with germline variants: {germline_spots}\n")
            f.write(f"  Spots with somatic variants: {somatic_spots}\n")
            
            f.write("\nOutput Files:\n")
            f.write(f"  Combined variant files:\n")
            f.write(f"    - {self.germline_dir}/germline_variants.txt|.vcf.gz\n")
            f.write(f"    - {self.somatic_dir}/somatic_variants.txt|.vcf.gz\n")
            f.write(f"    - {self.ambiguous_dir}/ambiguous_variants.txt|.vcf.gz\n")
            f.write(f"  Race-specific variant files (defined/denovo):\n")
            f.write(f"    - Germline: {self.germline_defined_dir}/, {self.germline_denovo_dir}/\n")
            f.write(f"    - Somatic: {self.somatic_defined_dir}/, {self.somatic_denovo_dir}/\n")
            f.write(f"    - Ambiguous: {self.ambiguous_defined_dir}/, {self.ambiguous_denovo_dir}/\n")
            f.write(f"  Visualizations: {self.viz_dir}/*.png\n")

    def save_completion_manifest(self):
        """Validate the final partition and write the Stage 6 sentinel last."""
        class_sets = {
            "germline": set(self.germline_variants),
            "somatic": set(self.somatic_variants),
            "ambiguous": set(self.ambiguous_variants),
        }
        names = list(class_sets)
        for index, left in enumerate(names):
            for right in names[index + 1:]:
                overlap = class_sets[left] & class_sets[right]
                if overlap:
                    raise RuntimeError(
                        f"classification sets are not disjoint ({left}, {right})"
                    )
        classified = set().union(*class_sets.values())
        if classified != set(self.variant_scores):
            raise RuntimeError(
                "classification output does not conserve scored variants: "
                f"classified={len(classified)}, scored={len(self.variant_scores)}"
            )

        germline_files = sorted(
            os.path.basename(path) for path in glob.glob(
                os.path.join(self.germline_dir, "*.txt")
            ) if not os.path.basename(path).startswith("germline")
        )
        somatic_files = sorted(
            os.path.basename(path) for path in glob.glob(
                os.path.join(self.somatic_dir, "*.txt")
            ) if not os.path.basename(path).startswith("somatic")
        )
        expected_germline = sorted(
            f"{barcode}.txt" for barcode, variants
            in self.filtered_spot_snvs_germline.items() if variants
        )
        expected_somatic = sorted(
            f"{barcode}.txt" for barcode, variants
            in self.filtered_spot_snvs_somatic.items() if variants
        )
        if germline_files != expected_germline or somatic_files != expected_somatic:
            raise RuntimeError("per-barcode spatial-filter text inventory is incomplete")

        output_root = Path(self.output_dir)
        artifacts = sorted(
            str(path.relative_to(output_root))
            for path in output_root.rglob("*") if path.is_file()
        )
        manifest = {
            "schema": SPATIAL_FILTER_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "dataset": self.dataset,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "source_spotprofile_schema": self.spotprofile_manifest["schema"],
            "source_spotprofile_manifest": str(
                Path(self.spotprofile_manifest_path).resolve()
            ),
            "source_spotprofile_manifest_sha256": self._sha256_file(
                self.spotprofile_manifest_path
            ),
            "source_per_spot_vcf_count": self.spotprofile_manifest[
                "per_spot_vcf_count"
            ],
            "presence_rule": self.presence_rule,
            "per_spot_presence_rule": {
                "G0": "every in-tissue step-6 VCF record",
                "G1": "G0 values restricted to exact alleles with >=1 ALT molecule section-wide",
                "G2": "exact spot/allele has >=1 ALT molecule",
            }[self.presence_rule],
            "molecule_table": {
                "path": str(Path(self.molecule_table_file).resolve()),
                "sha256": self._sha256_file(self.molecule_table_file),
                **self.molecule_table_stats,
            },
            "deprecated_min_expression_arguments": {
                "germline": self.deprecated_min_expression_germline,
                "somatic": self.deprecated_min_expression_somatic,
                "affect_classification": False,
            },
            "classification_iteration_order": "chrom,pos,ref,alt",
            "stage2_tie_break_order": "votes desc, summed evidence desc, variant key asc",
            "variant_counts": {
                "germline": len(self.germline_variants),
                "somatic": len(self.somatic_variants),
                "ambiguous": len(self.ambiguous_variants),
                "germline_defined": len(self.germline_defined),
                "germline_denovo": len(self.germline_denovo),
                "somatic_defined": len(self.somatic_defined),
                "somatic_denovo": len(self.somatic_denovo),
                "ambiguous_defined": len(self.ambiguous_defined),
                "ambiguous_denovo": len(self.ambiguous_denovo),
                "total_classified": len(classified),
            },
            "per_barcode_files": {
                "germline": germline_files,
                "somatic": somatic_files,
            },
            "per_barcode_file_counts": {
                "germline": len(germline_files),
                "somatic": len(somatic_files),
            },
            "artifacts_before_manifest": artifacts,
        }
        if self.calicost_gate_result is not None:
            gate_path = output_root / "calicost_gate.json"
            manifest["calicost_gate"] = {
                "path": str(gate_path.resolve()),
                "sha256": self._sha256_file(gate_path),
                "cnv_pass": self.calicost_gate_result["cnv_pass"],
                "purity_pass": self.calicost_gate_result["purity_pass"],
            }
        manifest_path = output_root / "MANIFEST.json"
        complete_path = output_root / "RUN_COMPLETE"
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with complete_path.open("x") as handle:
            handle.write("complete\n")
        return manifest
    
    def run_analysis(self):
        """Run the complete enhanced spatial filtering pipeline."""
        logger.info("Starting enhanced spatial SNV filtering with tumor purity...")

        # Validate the required upstream production contract before expensive work.
        self.validate_spotprofile_bundle()

        # 1. Gate CalicoST inputs, then load only evidence that passed.
        if (self.evidence_mode == "full" and self.calicost_gate_mode == "on"
                and self.calicost_inputs_present):
            self.apply_calicost_gate()

        # 1a. Load tumor purity unless this is the explicit CalicoST ablation.
        if self.evidence_mode == "full":
            self.load_tumor_purity()
        else:
            logger.info("Skipping tumor-purity loading in spatial-only CalicoST ablation")
        
        # 1b. Load clone labels and CNV segments if available
        if self.evidence_mode == "full" and self.use_clone_cnv:
            self.load_clone_labels()
            self.load_cnv_segments()
            logger.info("Clone+CNV integration enabled")
        else:
            logger.info("Clone/CNV integration disabled")
        
        # 2. Load spatial positions
        self.load_spot_positions()
        
        # 3. Load the molecule inventory and build the spatial graph.
        self.load_molecule_table()
        self.build_spatial_graph()
        
        # 4. Load SNV data and apply the selected presence rule.
        self.load_snv_data()
        
        # 5. Create and apply pools
        self.create_exclusion_pool()
        self.create_inclusion_pool()
        self.create_kept_variants_pool()
        self.apply_pools()
        
        # 6. Classify variants
        self.classify_variants(self.germline_threshold, self.somatic_threshold)
        
        # 7. Atomically claim the fresh output root, then save outputs.
        self._initialize_output_dirs()
        if self.calicost_gate_result is not None:
            gate_path = os.path.join(self.output_dir, "calicost_gate.json")
            with open(gate_path, "x") as handle:
                json.dump(self.calicost_gate_result, handle, indent=2, sort_keys=True)
                handle.write("\n")
        self.save_text_outputs()
        self.save_vcf_outputs()

        # 8. Save summary and write the completion sentinel last.
        self.save_summary()
        self.save_completion_manifest()

        logger.info(f"Analysis complete! Results saved to {self.output_dir}")
        
        return self


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Spatial SNV Filter with Tumor Purity Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # P4 Tumor Section 1
  python %(prog)s --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 \\
    --tumor_purity_file /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv

  # DLPFC Section
  python %(prog)s --dataset dlpfc --section_id 151507 --quality_filter baseQ13mapQ20 \\
    --tumor_purity_file /path/to/tumor_purity.tsv \\
    --exclude_vcf known_artifacts.vcf.gz
        """
    )
    
    # Required arguments
    parser.add_argument('--dataset', required=True,
                       choices=['dlpfc', 'p4_tumor', 'p6_tumor', 'dcis', 'ovar_p5', 'ncce'],
                       help='Dataset name (case-insensitive)')
    parser.add_argument('--section_id', required=True,
                       help='Section ID (e.g., 1 for P4/P6, 151507 for DLPFC)')
    parser.add_argument('--quality_filter', required=True,
                       help='Quality filter applied (e.g., baseQ0mapQ0, baseQ13mapQ20)')
    parser.add_argument('--tumor_purity_file', required=False, default=None,
                       help='Path to CalicoST tumor purity file (loh_estimator_tumor_prop.tsv). '
                            'Optional; missing values remain unknown and are never coerced to 0')
    parser.add_argument('--molecule_table', required=True,
                       help='S6+ recount_spot_alleles.py TSV/TSV.GZ for this exact Stage-5 bundle')
    parser.add_argument('--presence_rule', default='G1', choices=PRESENCE_RULES,
                       help='G1 production default; G0/G2 reproduce the 2026-10-09 definitions')
    
    # NEW: Optional Clone and CNV arguments
    parser.add_argument('--clone_labels', default=None,
                       help='Path to CalicoST clone_labels.tsv (enables clone-based scoring)')
    parser.add_argument('--cnv_segments', default=None,
                       help='Path to CalicoST cnv_seglevel.tsv (enables CNV-based scoring)')
    parser.add_argument('--clone_posterior', default=None,
                       help='Path to posterior_clone_probability.npy')
    parser.add_argument('--spot_umi_tsv', default=None,
                       help='TSV with barcode and total_umi columns')
    parser.add_argument('--calicost_sample_id', default=None,
                       help='Joint CalicoST section suffix, e.g. P61')
    parser.add_argument('--calicost_gate', default='on', choices=['on', 'off'],
                       help='Gate CalicoST purity/CNV evidence (default: on)')
    parser.add_argument('--cnv_feature', default='clone_enrichment',
                       choices=['clone_enrichment', 'legacy'],
                       help='Theta feature: depth-stratified clone enrichment or legacy score')
    parser.add_argument('--evidence_mode', default='full',
                       choices=['full', 'spatial_only_no_calicost'],
                       help='full uses available CalicoST evidence; spatial_only_no_calicost '
                            'is the explicit ablation and uses zeta only')
    parser.add_argument('--mechanistic_vaf_mode', default='auto',
                       choices=['auto', 'off', 'shadow', 'vote'],
                       help='CNV/purity-aware beta-binomial VAF evidence. auto=shadow when '
                            'CalicoST inputs are supplied; shadow writes audit scores without '
                            'changing calls; vote adds the score to somatic rank voting')
    parser.add_argument('--mechanistic_kappa', type=float, default=8.0,
                       help='Beta-binomial concentration for VAF overdispersion (default: 8)')
    parser.add_argument('--mechanistic_score_scale', type=float, default=10.0,
                       help='Log-BF scale used only for bounded rank score (default: 10)')
    parser.add_argument('--mechanistic_min_information', type=float, default=0.20,
                       help='Minimum read-information factor for optional VAF voting')
    
    # Optional arguments
    parser.add_argument('--output_dir', default=None,
                       help='Custom output directory (default: auto-generated based on dataset)')
    parser.add_argument('--section-root', default=None,
                       help='Override both pipeline input and output section root for an isolated run')
    parser.add_argument('--neighbor_distance', type=float, default=2.0,
                       help='Distance threshold for defining neighbors (default: 2.0)')
    parser.add_argument(
        '--min_expression_germline', type=int, default=2,
        help='DEPRECATED compatibility argument; only legacy value 2 is accepted. '
             'Presence is explicitly ALT count >=1 per spot.'
    )
    parser.add_argument(
        '--min_expression_somatic', type=int, default=1,
        help='DEPRECATED compatibility argument; only legacy value 1 is accepted. '
             'Presence is explicitly ALT count >=1 per spot.'
    )
    
    # Pool arguments
    parser.add_argument('--exclude_vcf', dest='exclude_vcf',
                       help='VCF file with variants to exclude from analysis')
    parser.add_argument('--include_vcf', dest='include_vcf',
                       help='VCF file with variants to include only (restrict to these)')
    parser.add_argument('--kept_variants', dest='kept_variants',
                       help='VCF file with variants to force as germline')
    
    # Threshold arguments
    parser.add_argument('--germline_threshold', type=float, default=0.5,
                       help='Score threshold for classifying as germline (default: 0.5)')
    parser.add_argument('--somatic_threshold', type=float, default=0.4,
                       help='Score threshold for classifying as somatic (default: 0.4)')

    # Step 7c — optional UPV BAF-GMM sub-filter (post-processing of the UPV set)
    parser.add_argument('--run_baf_gmm_subfilter', action='store_true',
                       help='After filtering, run the UPV BAF-GMM sub-filter (step 7c) '
                            'to split germline_denovo (UPV) into germline-like vs '
                            'somatic-candidate. See upv_baf_gmm_subfilter.py.')

    args = parser.parse_args()
    
    # Validate dataset
    dataset_upper = args.dataset.upper()
    if dataset_upper not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {args.dataset}")
        logger.info(f"Available datasets: {', '.join(DATASET_CONFIGS.keys())}")
        sys.exit(1)
    
    # Create filter instance
    filter_instance = EnhancedSpatialSNVFilter(
        dataset=args.dataset,
        section_id=args.section_id,
        quality_filter=args.quality_filter,
        tumor_purity_file=args.tumor_purity_file,
        molecule_table_file=args.molecule_table,
        presence_rule=args.presence_rule,
        output_dir=args.output_dir,
        section_root=args.section_root,
        neighbor_distance=args.neighbor_distance,
        min_expression_germline=args.min_expression_germline,
        min_expression_somatic=args.min_expression_somatic,
        exclude_vcf_path=args.exclude_vcf,
        include_vcf_path=args.include_vcf,
        kept_variants_path=args.kept_variants,
        clone_labels_file=args.clone_labels,
        cnv_segments_file=args.cnv_segments,
        clone_posterior_file=args.clone_posterior,
        spot_umi_tsv=args.spot_umi_tsv,
        calicost_sample_id=args.calicost_sample_id,
        calicost_gate=args.calicost_gate,
        cnv_feature=args.cnv_feature,
        evidence_mode=args.evidence_mode,
        mechanistic_vaf_mode=args.mechanistic_vaf_mode,
        mechanistic_kappa=args.mechanistic_kappa,
        mechanistic_score_scale=args.mechanistic_score_scale,
        mechanistic_min_information=args.mechanistic_min_information,
        germline_threshold=args.germline_threshold,
        somatic_threshold=args.somatic_threshold
    )
    
    # Run analysis
    filter_instance.run_analysis()

    # Step 7c — optional UPV BAF-GMM sub-filter (post-processing)
    if args.run_baf_gmm_subfilter:
        sub = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'upv_baf_gmm_subfilter.py')
        cmd = [sys.executable, sub,
               '--dataset', dataset_upper,
               '--section_id', args.section_id,
               '--quality_filter', args.quality_filter]
        logger.info(f"Running UPV BAF-GMM sub-filter (7c): {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.warning(f"UPV BAF-GMM sub-filter failed (non-fatal): {e}")

    return filter_instance


if __name__ == "__main__":
    filter_instance = main()

# Example usage:
# P4 Tumor Section 1 with all options
# python run_spatial_snv_filter_enhanced.py \
#   --dataset p4_tumor \
#   --section_id 1 \
#   --quality_filter baseQ0mapQ0 \
#   --tumor_purity_file /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \
#   --exclude_vcf /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz \
#   --include_vcf /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz \
#   --kept_variants /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz \
#   --min_expression_germline 2 \
#   --min_expression_somatic 1 \
#   --germline_threshold 0.5 \
#   --somatic_threshold 0.4

# P6 Tumor Section 1
# python run_spatial_snv_filter_enhanced.py \
#   --dataset p6_tumor \
#   --section_id 1 \
#   --quality_filter baseQ0mapQ0 \
#   --tumor_purity_file /data/maiziezhou_lab/leiy4/CalicoST/P6_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv

# DLPFC Section 151507
# python run_spatial_snv_filter_enhanced.py \
#   --dataset dlpfc \
#   --section_id 151507 \
#   --quality_filter baseQ13mapQ20 \
#   --tumor_purity_file /path/to/dlpfc_tumor_purity.tsv \
#   --min_expression_germline 2 \
#   --min_expression_somatic 1

# P4 Tumor Section 1 with Clone+CNV Integration (RECOMMENDED)
# python run_spatial_snv_filter_enhanced.py \
#   --dataset p4_tumor \
#   --section_id 1 \
#   --quality_filter baseQ0mapQ0 \
#   --tumor_purity_file /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \
#   --clone_labels /data/maiziezhou_lab/leiy4/CalicoST/P4/CalicoST_output/best_result/clone_labels.tsv \
#   --cnv_segments /data/maiziezhou_lab/leiy4/CalicoST/P4/CalicoST_output/best_result/cnv_seglevel.tsv \
#   --min_expression_germline 2 \
#   --min_expression_somatic 1
