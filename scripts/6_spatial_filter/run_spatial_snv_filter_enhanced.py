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
import subprocess
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from datetime import datetime
import gzip
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Visualization settings
FIGURE_SIZE = (12, 10)
DPI = 300
COLORMAP_GERMLINE = ['white', 'lightblue', 'blue', 'darkblue']
COLORMAP_SOMATIC = ['white', 'lightyellow', 'orange', 'red']
COLORMAP_COMBINED = ['white', 'lightgray', 'gray', 'darkgray']

# Dataset configurations
# REPLACE the whole DATASET_CONFIGS dict with this:
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc",
        "spatial_subdir": "spatial",
        "position_file": "tissue_positions_list.csv",   # no header
        "scale_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "has_header": False,
        "in_tissue_column": 1
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_positions_list.csv",
            "2": "GSM4565824_P4_rep2_tissue_positions_list.csv"
        },
        "scale_file_patterns": {
            "1": "GSM4565823_P4_rep1_scalefactors_json.json",
            "2": "GSM4565824_P4_rep2_scalefactors_json.json"
        },
        "image_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_hires_image.png",
            "2": "GSM4565824_P4_rep2_tissue_hires_image.png"
        }
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_positions_list.csv",
            "2": "GSM4565826_P6_rep2_tissue_positions_list.csv"
        },
        "scale_file_patterns": {
            "1": "GSM4565825_P6_rep1_scalefactors_json.json",
            "2": "GSM4565826_P6_rep2_scalefactors_json.json"
        },
        "image_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_hires_image.png",
            "2": "GSM4565826_P6_rep2_tissue_hires_image.png"
        }
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
                 output_dir: str = None,
                 neighbor_distance: float = 2.0,
                 min_expression_germline: int = 2,
                 min_expression_somatic: int = 1,
                 exclude_vcf_path: str = None,
                 include_vcf_path: str = None,
                 kept_variants_path: str = None,
                 clone_labels_file: str = None,
                 cnv_segments_file: str = None,
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
        self.neighbor_distance = neighbor_distance
        self.min_expression_germline = min_expression_germline
        self.min_expression_somatic = min_expression_somatic
        self.exclude_vcf_path = exclude_vcf_path
        self.include_vcf_path = include_vcf_path
        self.kept_variants_path = kept_variants_path
        self.germline_threshold = germline_threshold
        self.somatic_threshold = somatic_threshold
        
        # NEW: CalicoST clone and CNV data (optional)
        self.clone_labels_file = clone_labels_file
        self.cnv_segments_file = cnv_segments_file
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
        self.spot_neighbors = defaultdict(list)  # barcode -> list of neighbors
        self.snv_ref_alt_map = {}  # snv_key -> (ref, alt)
        self.snv_race = {}  # snv_key -> "defined" or "denovo"
        
        # NEW: Clone and CNV data structures
        self.clone_labels = {}  # barcode -> clone_id (0=normal, 1,2,3...=cancer)
        self.cnv_segments = None  # DataFrame with CNV segments
        self.all_clones = []  # List of all clone IDs
        self.cancer_clones = []  # List of cancer clone IDs (excluding 0)
        
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
        
        if self.use_clone_cnv:
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

        # SNV per-barcode VCFs -- new path under spotprofiles/
        output_base = cfg.get('output_base')
        self.snv_pos_dir = os.path.join(
            output_base,
            self.section_id,
            f"output_VCFs/spotprofiles/{self.quality_filter}/vcf_by_spot"
        )

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
                cfg["output_base"], self.section_id, f"spatial_filter_purity/{self.quality_filter}"
            )
        os.makedirs(self.output_dir, exist_ok=True)

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
        
        for d in [self.germline_dir, self.somatic_dir, self.ambiguous_dir, self.viz_dir,
                  self.germline_defined_dir, self.germline_denovo_dir,
                  self.somatic_defined_dir, self.somatic_denovo_dir,
                  self.ambiguous_defined_dir, self.ambiguous_denovo_dir]:
            os.makedirs(d, exist_ok=True)

        # Clean all output directories before every run so stale files never persist
        self._clean_output_dirs()

        logger.info(f"SNV VCF dir: {self.snv_pos_dir}")
        logger.info(f"Positions: {self.positions_file}")
        logger.info(f"Scale:     {self.scale_factor_file}")
        logger.info(f"Image:     {self.image_file}")
        logger.info(f"Output:    {self.output_dir}")


    
    def _clean_output_dirs(self):
        """
        Remove all files in output subdirectories before every run.
        This prevents stale results from a previous run from persisting
        when the new run produces fewer or different variants.
        Directories themselves are preserved.
        """
        import shutil
        dirs_to_clean = [
            self.germline_defined_dir, self.germline_denovo_dir,
            self.somatic_defined_dir,  self.somatic_denovo_dir,
            self.ambiguous_defined_dir, self.ambiguous_denovo_dir,
            self.germline_dir, self.somatic_dir, self.ambiguous_dir,
            self.viz_dir,
        ]
        total_removed = 0
        for d in dirs_to_clean:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                fpath = os.path.join(d, fname)
                if os.path.isfile(fpath):
                    os.remove(fpath)
                    total_removed += 1
        # Also remove top-level output files (all_variant_scores.txt, spatial_filter_summary.txt)
        for fname in os.listdir(self.output_dir):
            fpath = os.path.join(self.output_dir, fname)
            if os.path.isfile(fpath):
                os.remove(fpath)
                total_removed += 1
        logger.info(f"Cleaned {total_removed} stale output files before run")

    def load_tumor_purity(self):
        """Load tumor purity data from CalicoST output."""
        if self.tumor_purity_file is None:
            logger.info("No tumor purity file provided — all spots default to purity 0.0 (normal tissue mode)")
            return

        logger.info(f"Loading tumor purity from: {self.tumor_purity_file}")

        try:
            purity_df = pd.read_csv(
                self.tumor_purity_file, 
                sep='\t', 
                index_col='BARCODES'
            )
            
            for barcode, row in purity_df.iterrows():
                clean_barcode = barcode.split('_')[0] if '_' in barcode else barcode
                purity = row['Tumor'] if pd.notna(row['Tumor']) else 0.0
                self.tumor_purity[clean_barcode] = float(purity)
            
            purities = list(self.tumor_purity.values())
            logger.info(f"Loaded tumor purity for {len(self.tumor_purity)} spots")
            logger.info(f"Purity stats: min={min(purities):.3f}, "
                       f"max={max(purities):.3f}, mean={np.mean(purities):.3f}")
            
        except Exception as e:
            logger.error(f"Error loading tumor purity: {e}")
            raise
    
    def load_clone_labels(self):
        """Load clone assignments from CalicoST."""
        if not self.clone_labels_file:
            return
        
        logger.info(f"Loading clone labels from: {self.clone_labels_file}")
        
        try:
            clone_df = pd.read_csv(self.clone_labels_file, sep='\t')
            
            for _, row in clone_df.iterrows():
                barcode = str(row['BARCODES'])
                # Clean barcode - remove suffix if present
                clean_barcode = barcode.split('_')[0] if '_' in barcode else barcode
                clone_id = int(row['clone_label'])
                self.clone_labels[clean_barcode] = clone_id
            
            # Identify all clones
            self.all_clones = sorted(set(self.clone_labels.values()))
            self.cancer_clones = [c for c in self.all_clones if c > 0]
            
            logger.info(f"Loaded {len(self.clone_labels)} clone assignments")
            logger.info(f"Clones detected: {self.all_clones}")
            for clone_id in self.all_clones:
                count = sum(1 for c in self.clone_labels.values() if c == clone_id)
                clone_type = "Normal (clone 0)" if clone_id == 0 else f"Cancer clone {clone_id}"
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
        elif self.dataset == "DCIS":
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
            # P4/P6: no header; cols: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
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

        logger.info(f"Loaded {len(self.spot_positions)} spot positions")

    
    def load_vcf_variants(self, vcf_path: str) -> Set[str]:
        """Load variants from a VCF file."""
        variants = set()
        
        if not os.path.exists(vcf_path):
            logger.warning(f"VCF file not found: {vcf_path}")
            return variants
        
        try:
            if vcf_path.endswith('.gz'):
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(vcf_path, mode) as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        chrom = parts[0].replace('chr', '')
                        pos = parts[1]
                        ref = parts[3]
                        alt = parts[4]
                        
                        # Handle multi-allelic sites
                        for alt_allele in alt.split(','):
                            snv_key = f"{chrom}_{pos}"
                            variants.add(snv_key)
                            self.snv_ref_alt_map[snv_key] = (ref, alt_allele)
            
            logger.info(f"Loaded {len(variants)} variants from {vcf_path}")
            
        except Exception as e:
            logger.error(f"Error loading VCF file {vcf_path}: {e}")
        
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
    
    def load_snv_data(self):
        vcf_files = glob.glob(os.path.join(self.snv_pos_dir, "*.vcf.gz"))
        logger.info(f"Looking for SNV VCFs in: {self.snv_pos_dir}")
        logger.info(f"Found {len(vcf_files)} VCFs")
        self.snv_expression_counts = defaultdict(lambda: defaultdict(int))

        for vcf in vcf_files:
            bc = os.path.basename(vcf).replace(".vcf.gz", "")
            if bc not in self.spot_positions:
                continue
            try:
                with gzip.open(vcf, "rt") as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.strip().split("\t")
                        if len(parts) < 8:  # Need at least INFO field
                            continue
                        chrom = parts[0].replace("chr", "")
                        pos   = parts[1]
                        ref   = parts[3]
                        alt   = parts[4]
                        info  = parts[7] if len(parts) > 7 else ""
                        
                        # Extract RACE from INFO field
                        race = "unknown"
                        if info:
                            for field in info.split(";"):
                                if field.startswith("RACE="):
                                    race = field.split("=")[1]
                                    break
                        
                        # Skip multi-allelic for consistency with Previous
                        if "," in alt:
                            continue
                        snv_key = f"{chrom}_{pos}"
                        self.spot_snvs[bc].add(snv_key)
                        self.snv_ref_alt_map[snv_key] = (ref, alt)
                        self.snv_race[snv_key] = race  # Track race
                        
                        # No explicit expression in these VCFs â€“ count presence as 1
                        self.snv_expression_counts[snv_key][bc] = max(
                            self.snv_expression_counts[snv_key].get(bc, 0), 1
                        )
            except Exception as e:
                logger.warning(f"Error reading {vcf}: {e}")

        total_snvs = sum(len(s) for s in self.spot_snvs.values())
        spots_with = sum(1 for s in self.spot_snvs.values() if s)
        logger.info(f"Loaded {total_snvs} total SNV instances across {spots_with} spots")
        
        # Log race distribution
        race_counts = {"defined": 0, "denovo": 0, "unknown": 0}
        for race in self.snv_race.values():
            race_counts[race] = race_counts.get(race, 0) + 1
        logger.info(f"Race distribution - Defined: {race_counts.get('defined', 0)}, "
                   f"Denovo: {race_counts.get('denovo', 0)}, Unknown: {race_counts.get('unknown', 0)}")
        logger.info(f"Loaded {total_snvs} SNVs across {spots_with} spots")

    
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
        spots_with_variant = [bc for bc, snvs in self.spot_snvs.items() 
                              if variant in snvs]
        
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
        """Calculate global prevalence score."""
        spots_with_variant = sum(1 for snvs in self.spot_snvs.values() 
                                 if variant in snvs)
        total_spots = len(self.spot_snvs)
        
        if total_spots == 0:
            return 0.0
        
        prevalence = spots_with_variant / total_spots
        score = 1.0 / (1.0 + np.exp(-20 * (prevalence - 0.1)))
        
        return score
    
    def calculate_purity_independence_score(self, variant: str) -> float:
        """Calculate independence from tumor purity."""
        purities_with = []
        purities_without = []
        
        for barcode, snvs in self.spot_snvs.items():
            purity = self.tumor_purity.get(barcode, 0.0)
            
            if variant in snvs:
                purities_with.append(purity)
            else:
                purities_without.append(purity)
        
        if len(purities_with) < 3 or len(purities_without) < 3:
            return 0.5
        
        statistic, pvalue = scipy.stats.ks_2samp(purities_with, purities_without)
        return pvalue
    
    def calculate_purity_correlation_score(self, variant: str) -> float:
        """Calculate correlation with tumor purity."""
        variant_presence = []
        purities = []
        
        for barcode in self.spot_positions:
            purity = self.tumor_purity.get(barcode, 0.0)
            has_variant = 1 if variant in self.spot_snvs.get(barcode, set()) else 0
            
            purities.append(purity)
            variant_presence.append(has_variant)
        
        if len(set(variant_presence)) < 2:
            return 0.0
        
        correlation, pvalue = scipy.stats.pointbiserialr(variant_presence, purities)
        score = max(0, correlation)
        
        return score
    
    def calculate_spatial_clustering_score(self, variant: str) -> float:
        """Calculate spatial clustering score."""
        spots_with_variant = [bc for bc, snvs in self.spot_snvs.items() 
                              if variant in snvs]
        
        if len(spots_with_variant) < 2:
            return 0.0
        
        clustering_scores = []
        
        for barcode in spots_with_variant:
            neighbors = self.spot_neighbors.get(barcode, [])
            if len(neighbors) > 0:
                neighbors_with_variant = sum(1 for n in neighbors 
                                            if variant in self.spot_snvs.get(n, set()))
                cluster_coef = neighbors_with_variant / len(neighbors)
                clustering_scores.append(cluster_coef)
        
        if clustering_scores:
            avg_clustering = np.mean(clustering_scores)
        else:
            avg_clustering = 0.0
        
        return avg_clustering
    
    def parse_variant_for_cnv(self, variant: str) -> Tuple[str, int]:
        """Parse variant to get chromosome and position for CNV lookup."""
        parts = variant.split('_')
        chrom = parts[0].replace('chr', '')  # Remove chr prefix for matching
        pos = int(parts[1]) if len(parts) > 1 else 0
        return chrom, pos
    
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
            spots_in_clone = [bc for bc, cid in self.clone_labels.items() if cid == clone_id]
            
            if len(spots_in_clone) == 0:
                prevalence_by_clone[clone_id] = 0.0
                continue
            
            spots_with_variant = sum(1 for bc in spots_in_clone 
                                     if variant in self.spot_snvs.get(bc, set()))
            
            prevalence_by_clone[clone_id] = spots_with_variant / len(spots_in_clone)
        
        # Get normal vs cancer prevalence
        prev_normal = prevalence_by_clone.get(0, 0.0)
        
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
        
        chrom, pos = self.parse_variant_for_cnv(variant)
        
        # Find segment containing this variant
        segment = self.cnv_segments[
            (self.cnv_segments['CHR'].astype(str) == str(chrom)) &
            (self.cnv_segments['START'] <= pos) &
            (self.cnv_segments['END'] >= pos)
        ]
        
        if segment.empty:
            return 0.5  # No CNV info — segment not found for this chromosome/position

        segment = segment.iloc[0]

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
            spots_in_clone = [bc for bc, cid in self.clone_labels.items() if cid == clone_id]
            if len(spots_in_clone) < 3:
                continue
            
            spots_with_variant = sum(1 for bc in spots_in_clone 
                                    if variant in self.spot_snvs.get(bc, set()))
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
        # Base components (always calculated)
        s_purity = self.calculate_purity_correlation_score(variant)
        s_cluster = self.calculate_spatial_clustering_score(variant)
        s_clone_specific = s_cluster * s_purity  # Simplified clone-specific score
        
        if self.use_clone_cnv:
            # Enhanced scoring with clone+CNV data
            s_clone_enrich = self.calculate_clone_enrichment_score(variant)
            s_cnv_consist = self.calculate_cnv_consistency_score(variant)
            
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
    
    def check_expression_threshold(self, variant: str, threshold: int) -> bool:
        """Check if variant meets expression threshold."""
        if variant not in self.snv_expression_counts:
            return False
        
        max_expression = max(self.snv_expression_counts[variant].values())
        return max_expression >= threshold
    
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
        for variant in all_variants:
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

        # ── Step 2: Somatic voting ──
        # Features used for voting (clone enrichment excluded by design)
        VOTE_TOP_FRACTION   = 0.20   # top 20% per feature → 1 vote
        SOMATIC_TOP_FRACTION = 0.10  # top 10% by votes → somatic

        # Determine which features participate in voting
        voting_features = ['delta', 'epsilon', 'zeta']
        if self.use_clone_cnv:
            voting_features.append('theta')
        n_features = len(voting_features)
        logger.info(f"Stage 2 — Somatic voting with {n_features} features: {voting_features}")

        # Calculate somatic feature scores for all non-germline denovo variants
        feature_scores = {f: {} for f in voting_features}
        for variant in non_germline:
            s_purity  = self.calculate_purity_correlation_score(variant)
            s_cluster = self.calculate_spatial_clustering_score(variant)
            s_clone_specific = s_cluster * s_purity
            feature_scores['delta'][variant]   = s_purity
            feature_scores['zeta'][variant]    = s_cluster
            feature_scores['epsilon'][variant] = s_clone_specific
            if self.use_clone_cnv:
                feature_scores['theta'][variant] = self.calculate_cnv_consistency_score(variant)
            # Store in variant_scores for downstream use / plotting
            self.variant_scores[variant]['delta']   = s_purity
            self.variant_scores[variant]['zeta']    = s_cluster
            self.variant_scores[variant]['epsilon'] = s_clone_specific
            if self.use_clone_cnv:
                self.variant_scores[variant]['theta'] = feature_scores['theta'][variant]

        # Each feature votes for its STRICTLY top-20% variants by rank
        # (sorted slice avoids tie inflation at the boundary)
        vote_counts = defaultdict(int)
        for feat in voting_features:
            scores = feature_scores[feat]
            if not scores:
                continue
            n_voters = max(1, int(len(scores) * VOTE_TOP_FRACTION))  # floor, not ceil
            top_variants = sorted(scores.keys(),
                                  key=lambda v: scores[v], reverse=True)[:n_voters]
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
        voted_variants = sorted(
            [v for v in non_germline if vote_counts[v] >= 1],
            key=lambda v: vote_counts[v], reverse=True
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
                f.write("chrom\tpos\tref\talt\tgermline_score\tsomatic_score\trace\n")
                for variant in sorted(variant_set):
                    chrom, pos = variant.split('_', 1)
                    ref, alt = self.snv_ref_alt_map.get(variant, ("N", "N"))
                    scores = self.variant_scores.get(variant, {'germline': 0, 'somatic': 0})
                    race = self.snv_race.get(variant, "unknown")
                    f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t"
                           f"{scores['germline']:.4f}\t{scores['somatic']:.4f}\t{race}\n")
        
        # Save all scores — including per-feature breakdown for diagnostic analysis
        scores_file = os.path.join(self.output_dir, "all_variant_scores.txt")
        with open(scores_file, 'w') as f:
            # Header: composite scores + all individual feature components
            f.write("variant\tgermline_score\tsomatic_score\tclassification\trace"
                    "\tf_spatial_uniformity\tf_global_prevalence\tf_purity_independence"
                    "\tf_purity_correlation\tf_spatial_clustering\tf_clone_specific_proxy"
                    "\tf_clone_enrichment\tf_cnv_consistency\n")
            for variant, scores in self.variant_scores.items():
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
                    f_unif  = self.calculate_spatial_uniformity_score(variant)
                    f_prev  = self.calculate_global_prevalence_score(variant)
                    f_indep = self.calculate_purity_independence_score(variant)
                    f_pcorr = self.calculate_purity_correlation_score(variant)
                    f_clust = self.calculate_spatial_clustering_score(variant)
                    f_csprx = f_clust * f_pcorr  # clone-specific proxy = ζ × δ product
                    f_cenr  = (self.calculate_clone_enrichment_score(variant)
                               if self.use_clone_cnv else float('nan'))
                    f_cnv   = (self.calculate_cnv_consistency_score(variant)
                               if self.use_clone_cnv else float('nan'))
                    feats = (f"\t{f_unif:.4f}\t{f_prev:.4f}\t{f_indep:.4f}"
                             f"\t{f_pcorr:.4f}\t{f_clust:.4f}\t{f_csprx:.4f}"
                             f"\t{f_cenr if not (isinstance(f_cenr, float) and np.isnan(f_cenr)) else 'NA'}"
                             f"\t{f_cnv if not (isinstance(f_cnv, float) and np.isnan(f_cnv)) else 'NA'}")

                f.write(f"{variant}\t{scores['germline']:.4f}\t"
                       f"{scores['somatic']:.4f}\t{classification}\t{race}{feats}\n")
        
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
                    for snv in sorted(snvs):
                        chrom, pos = snv.split('_', 1)
                        ref, alt = self.snv_ref_alt_map.get(snv, ("N", "N"))
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
        header.append('##INFO=<ID=AF,Number=1,Type=Float,Description="Allele frequency across spots">')
        header.append('##INFO=<ID=SCORE,Number=1,Type=Float,Description="Classification score">')
        header.append('##INFO=<ID=PURITY_CORR,Number=1,Type=Float,Description="Correlation with tumor purity">')
        header.append('##INFO=<ID=RACE,Number=1,Type=String,Description="SNV origin: defined (Beagle) or denovo (Classifier)">')
        header.append('##FILTER=<ID=PASS,Description="Passed all filters">')
        header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO")
        
        return '\n'.join(header)
    
    def save_vcf_outputs(self):
        """Save variants to VCF files, split by classification and race."""
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
                    chrom, pos = variant.split('_', 1)
                    try:
                        # Try to convert chrom to int for proper sorting
                        chrom_sort = int(chrom) if chrom.isdigit() else 100
                    except:
                        chrom_sort = 100
                    sorted_variants.append((chrom_sort, int(pos), variant))
                
                sorted_variants.sort()
                
                # Write variants
                for _, _, variant in sorted_variants:
                    chrom, pos = variant.split('_', 1)
                    ref, alt = self.snv_ref_alt_map.get(variant, ("N", "N"))
                    
                    # Calculate INFO fields
                    ns = sum(1 for snvs in self.spot_snvs.values() if variant in snvs)
                    total_spots = len(self.spot_positions)
                    af = ns / total_spots if total_spots > 0 else 0
                    
                    scores = self.variant_scores.get(variant, {'germline': 0, 'somatic': 0})
                    # Determine which score to use
                    if "germline" in variant_name:
                        score = scores['germline']
                    elif "somatic" in variant_name:
                        score = scores['somatic']
                    else:  # ambiguous
                        score = max(scores['germline'], scores['somatic'])
                    
                    purity_corr = self.calculate_purity_correlation_score(variant)
                    race = self.snv_race.get(variant, "unknown")
                    
                    info = f"NS={ns};AF={af:.4f};SCORE={score:.4f};PURITY_CORR={purity_corr:.4f};RACE={race}"
                    
                    # Ensure chr prefix
                    if chrom.startswith("chr"):
                        f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
                    else:
                        f.write(f"chr{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
            
            # Compress and index
            try:
                subprocess.run(f"bgzip -f {vcf_file}", shell=True, check=True)
                subprocess.run(f"tabix -p vcf {vcf_file}.gz", shell=True, check=True)
                logger.info(f"Created VCF: {vcf_file}.gz ({len(variant_set)} variants)")
            except:
                logger.warning(f"Could not compress/index VCF: {vcf_file}")
    
    def save_summary(self):
        """Save comprehensive summary of the analysis."""
        summary_file = os.path.join(self.output_dir, "spatial_filter_summary.txt")
        
        with open(summary_file, 'w') as f:
            f.write("Enhanced Spatial SNV Filtering Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n")
            f.write(f"Tumor Purity File: {self.tumor_purity_file}\n\n")
            
            f.write("Parameters:\n")
            f.write(f"  Neighbor distance: {self.neighbor_distance}\n")
            f.write(f"  Min expression (germline): {self.min_expression_germline}\n")
            f.write(f"  Min expression (somatic): {self.min_expression_somatic}\n\n")
            
            f.write("Input Data:\n")
            f.write(f"  Total spots: {len(self.spot_positions)}\n")
            f.write(f"  Spots with SNVs: {len(self.spot_snvs)}\n")
            f.write(f"  Total SNV instances: {sum(len(snvs) for snvs in self.spot_snvs.values())}\n")
            
            all_variants = set()
            for snvs in self.spot_snvs.values():
                all_variants.update(snvs)
            f.write(f"  Unique variants: {len(all_variants)}\n\n")
            
            f.write("Tumor Purity Statistics:\n")
            purities = list(self.tumor_purity.values())
            if purities:
                f.write(f"  Min: {min(purities):.3f}\n")
                f.write(f"  Max: {max(purities):.3f}\n")
                f.write(f"  Mean: {np.mean(purities):.3f}\n")
                f.write(f"  Median: {np.median(purities):.3f}\n\n")
            
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
    
    def run_analysis(self):
        """Run the complete enhanced spatial filtering pipeline."""
        logger.info("Starting enhanced spatial SNV filtering with tumor purity...")
        
        # 1. Load tumor purity
        self.load_tumor_purity()
        
        # 1b. Load clone labels and CNV segments if available
        if self.use_clone_cnv:
            self.load_clone_labels()
            self.load_cnv_segments()
            logger.info("Clone+CNV integration enabled")
        else:
            logger.info("Using purity-only mode (no clone/CNV data)")
        
        # 2. Load spatial positions
        self.load_spot_positions()
        
        # 3. Build spatial graph
        self.build_spatial_graph()
        
        # 4. Load SNV data
        self.load_snv_data()
        
        # 5. Create and apply pools
        self.create_exclusion_pool()
        self.create_inclusion_pool()
        self.create_kept_variants_pool()
        self.apply_pools()
        
        # 6. Classify variants
        self.classify_variants(self.germline_threshold, self.somatic_threshold)
        
        # 7. Save outputs
        self.save_text_outputs()
        self.save_vcf_outputs()

        # 8. Save summary
        self.save_summary()
        
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
                       choices=['dlpfc', 'p4_tumor', 'p6_tumor', 'dcis', 'ovar_p5'],
                       help='Dataset name (case-insensitive)')
    parser.add_argument('--section_id', required=True,
                       help='Section ID (e.g., 1 for P4/P6, 151507 for DLPFC)')
    parser.add_argument('--quality_filter', required=True,
                       help='Quality filter applied (e.g., baseQ0mapQ0, baseQ13mapQ20)')
    parser.add_argument('--tumor_purity_file', required=False, default=None,
                       help='Path to CalicoST tumor purity file (loh_estimator_tumor_prop.tsv). '
                            'Optional — omit for normal tissue (all spots default to purity 0.0)')
    
    # NEW: Optional Clone and CNV arguments
    parser.add_argument('--clone_labels', default=None,
                       help='Path to CalicoST clone_labels.tsv (enables clone-based scoring)')
    parser.add_argument('--cnv_segments', default=None,
                       help='Path to CalicoST cnv_seglevel.tsv (enables CNV-based scoring)')
    
    # Optional arguments
    parser.add_argument('--output_dir', default=None,
                       help='Custom output directory (default: auto-generated based on dataset)')
    parser.add_argument('--neighbor_distance', type=float, default=2.0,
                       help='Distance threshold for defining neighbors (default: 2.0)')
    parser.add_argument('--min_expression_germline', type=int, default=2,
                       help='Minimum expression level for germline variants (default: 2)')
    parser.add_argument('--min_expression_somatic', type=int, default=1,
                       help='Minimum expression level for somatic variants (default: 1)')
    
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
        output_dir=args.output_dir,
        neighbor_distance=args.neighbor_distance,
        min_expression_germline=args.min_expression_germline,
        min_expression_somatic=args.min_expression_somatic,
        exclude_vcf_path=args.exclude_vcf,
        include_vcf_path=args.include_vcf,
        kept_variants_path=args.kept_variants,
        clone_labels_file=args.clone_labels,
        cnv_segments_file=args.cnv_segments,
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
#   --min_expression_germline 3 \
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