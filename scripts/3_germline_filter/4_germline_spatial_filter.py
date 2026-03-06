import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from PIL import Image
from collections import defaultdict
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import glob
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.spatial import KDTree, Delaunay
import logging
from sklearn.neighbors import NearestNeighbors
import gzip

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================== CONSTANTS (non-path) ========================
DEFAULT_NEIGHBOR_DISTANCE = 1.5
MAX_NEIGHBORS = 12

# Visualization parameters
FIGURE_SIZE = (16, 16)
DPI = 300
COLORMAP = ['#4578b4', '#e6e6e6', 'red']
# ======================================================================

class SpatialSNVFilter:
    def __init__(self, cfg: PipelineConfig, 
                neighbor_distance: float = DEFAULT_NEIGHBOR_DISTANCE,
                exclude_vcf_path: Optional[str] = None,
                include_vcf_path: Optional[str] = None,
                kept_variants_path: Optional[str] = None,
                min_neighbours: int = 2,
                out_tissue_file: Optional[str] = None):
        """
        Initialize the spatial SNV filter.
        
        Args:
            cfg: PipelineConfig from config_loader
            neighbor_distance: Distance threshold for determining neighbors (in spot diameters)
            exclude_vcf_path: Optional path to VCF file with SNVs to exclude
            include_vcf_path: Optional path to VCF file with SNVs to include (keep only these)
            kept_variants_path: Optional path to VCF file with SNVs to keep directly (bypass spatial filtering)
            min_neighbours: Minimum number of neighboring spots required to keep an SNV
            out_tissue_file: Path to file containing list of out-tissue barcodes (will override auto-detection)
        """
        self.cfg = cfg
        self.dataset = cfg.dataset_name
        self.section_id = cfg.section_id
        self.neighbor_distance = neighbor_distance
        self.exclude_vcf_path = exclude_vcf_path
        self.include_vcf_path = include_vcf_path
        self.kept_variants_path = kept_variants_path
        self.min_neighbours = min_neighbours
        self.out_tissue_file = out_tissue_file
        self.out_tissue_barcodes = set()
        
        # Initialize data structures
        self.spot_positions = {}  # Dict mapping barcode to (x, y) position
        self.spot_neighbors = {}  # Dict mapping barcode to list of neighbor barcodes
        self.spot_snvs = defaultdict(set)  # Dict mapping barcode to set of SNVs
        self.filtered_spot_snvs = defaultdict(set)  # Dict after spatial filtering
        self.snv_ref_alt_map = {}

        
        # SNV pools for filtering
        self.filter_snv_pool = set()  # Set of SNVs to be filtered out
        self.include_snv_pool = set()  # Set of SNVs to keep (filter in)
        self.kept_variants_pool = set()  # Set of SNVs to keep directly (bypass spatial filtering)
        
        # Set up paths
        self.setup_paths()
        
        # Load out-tissue barcodes
        if self.out_tissue_file:
            self.load_out_tissue_barcodes()
        
    def setup_paths(self):
        """Set up file paths from PipelineConfig."""
        # Spatial data files — resolved from config
        spatial_dir = self.cfg.spatial_dir
        self.position_file = os.path.join(spatial_dir, self.cfg.position_file) if spatial_dir else None
        self.scale_factor_file = os.path.join(spatial_dir, self.cfg.scale_factor_file) if spatial_dir else None
        self.image_file = os.path.join(spatial_dir, self.cfg.image_file) if spatial_dir else None
        
        # Auto-detect out-tissue file from spatial config if not explicitly set
        if not self.out_tissue_file:
            missing_raw = self.cfg.raw.get("spatial", {}).get("missing_tissue_file", "")
            if missing_raw and self.section_id:
                candidate = missing_raw.replace("{section_id}", str(self.section_id))
                if not os.path.isabs(candidate):
                    candidate = os.path.join(self.cfg.bam_base_path, candidate)
                if os.path.exists(candidate):
                    self.out_tissue_file = candidate
                    logger.info(f"Auto-detected out-tissue file: {self.out_tissue_file}")
        
        # Output directories
        self.data_dir = self.cfg.output_dir
        
        # SNV positions directory (from BAM filtering step)
        self.snv_pos_dir = os.path.join(
            self.data_dir, "output_VCFs", "BAM_filtered", "snv_positions")
        
        # Output directory for spatial analysis
        self.output_dir = os.path.join(self.data_dir, "spatial_analysis")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"SNV positions directory: {self.snv_pos_dir}")
        logger.info(f"Tissue positions file: {self.position_file}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def create_exclusion_snv_pool_from_vcf(self, vcf_path: str) -> Set[str]:
        """
        Create an SNV pool from a VCF file.
        
        Args:
            vcf_path: Path to the VCF file (.vcf.gz)
            
        Returns:
            Set of SNV IDs in the format "chrom_pos"
        """
        if not os.path.exists(vcf_path):
            logger.error(f"VCF file not found: {vcf_path}")
            return set()
            
        logger.info(f"Creating SNV pool from VCF: {vcf_path}")
        
        snv_pool = set()
        
        try:
            # Open VCF file (handling gzip)
            with gzip.open(vcf_path, 'rt') if vcf_path.endswith('.gz') else open(vcf_path, 'r') as f:
                for line in f:
                    # Skip header lines
                    if line.startswith('#'):
                        continue
                    
                    # Parse variant line
                    fields = line.strip().split('\t')
                    
                    if len(fields) < 5:
                        continue
                        
                    chrom = fields[0].replace("chr", "")  # Remove 'chr' prefix if present

                    pos = fields[1]
                    ref = fields[3]
                    alt = fields[4]
                    
                    # Skip complex variants for simplicity
                    if ',' in alt:
                        continue
                    
                    # Generate SNV key (same format as in spot_snvs)
                    snv_key = f"{chrom}_{pos}"
                    snv_pool.add(snv_key)
            
            logger.info(f"Created SNV pool with {len(snv_pool)} variants")
            #print some from the pool
            logger.info(f"SNVs in pool: {list(snv_pool)[:5]}")

            self.filter_snv_pool = snv_pool
            return snv_pool
            
        except Exception as e:
            logger.error(f"Error creating SNV pool from VCF: {e}")
            return set()
    
    def create_inclusion_snv_pool(self, vcf_path: str) -> Set[str]:
        """
        Create an SNV inclusion pool from a VCF file.
        Only SNVs in this pool will be kept.
        
        Args:
            vcf_path: Path to the VCF file (.vcf.gz)
            
        Returns:
            Set of SNV IDs in the format "chrom_pos"
        """
        if not os.path.exists(vcf_path):
            logger.error(f"Include VCF file not found: {vcf_path}")
            return set()
            
        logger.info(f"Creating SNV inclusion pool from VCF: {vcf_path}")
        
        snv_pool = set()
        
        try:
            # Open VCF file (handling gzip)
            with gzip.open(vcf_path, 'rt') if vcf_path.endswith('.gz') else open(vcf_path, 'r') as f:
                for line in f:
                    # Skip header lines
                    if line.startswith('#'):
                        continue
                    
                    # Parse variant line
                    fields = line.strip().split('\t')
                    
                    if len(fields) < 5:
                        continue
                        
                    chrom = fields[0].replace("chr", "")  # Remove 'chr' prefix if present
                    pos = fields[1]
                    ref = fields[3]
                    alt = fields[4]
                    
                    # Skip complex variants for simplicity
                    if ',' in alt:
                        continue
                    
                    # Generate SNV key (same format as in spot_snvs)
                    snv_key = f"{chrom}_{pos}"
                    snv_pool.add(snv_key)
            
            logger.info(f"Created SNV inclusion pool with {len(snv_pool)} variants")
            #print some from the pool
            # logger.info(f"SNVs in pool: {list(snv_pool)[:5]}")
            self.include_snv_pool = snv_pool
            return snv_pool
            
        except Exception as e:
            logger.error(f"Error creating SNV inclusion pool from VCF: {e}")
            return set()

    def create_kept_variants_pool(self, vcf_path: str) -> Set[str]:
        """
        Create a pool of SNVs to keep directly (bypass spatial filtering).
        
        Args:
            vcf_path: Path to the VCF file (.vcf.gz)
            
        Returns:
            Set of SNV IDs in the format "chrom_pos"
        """
        if not os.path.exists(vcf_path):
            logger.error(f"Kept variants VCF file not found: {vcf_path}")
            return set()
            
        logger.info(f"Loading variants to keep directly from: {vcf_path}")
        
        kept_variants = set()
        
        try:
            # Open VCF file (handling gzip)
            with gzip.open(vcf_path, 'rt') if vcf_path.endswith('.gz') else open(vcf_path, 'r') as f:
                for line in f:
                    # Skip header lines
                    if line.startswith('#'):
                        continue
                    
                    # Parse variant line
                    fields = line.strip().split('\t')
                    
                    if len(fields) < 5:
                        continue
                        
                    chrom = fields[0].replace("chr", "")  # Remove 'chr' prefix if present
                    pos = fields[1]
                    
                    # Generate SNV key (same format as in spot_snvs)
                    snv_key = f"{chrom}_{pos}"
                    kept_variants.add(snv_key)
            
            logger.info(f"Loaded {len(kept_variants)} variants to keep directly")
            if kept_variants:
                logger.info(f"Sample kept variants: {list(kept_variants)[:5]}")
            self.kept_variants_pool = kept_variants
            return kept_variants
            
        except Exception as e:
            logger.error(f"Error loading kept variants from VCF: {e}")
            return set()
            
    def load_out_tissue_barcodes(self):
        """Load the list of out-tissue barcodes from file."""
        if not os.path.exists(self.out_tissue_file):
            logger.warning(f"Out-tissue file not found: {self.out_tissue_file}")
            return
            
        try:
            logger.info(f"Loading out-tissue barcodes from {self.out_tissue_file}")
            with open(self.out_tissue_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        # First element is the barcode
                        barcode = parts[0]
                        self.out_tissue_barcodes.add(barcode)
                        
            logger.info(f"Loaded {len(self.out_tissue_barcodes)} out-tissue barcodes")
        except Exception as e:
            logger.error(f"Error loading out-tissue barcodes: {e}")
            
    def filter_out_snv_pool(self):
        """
        Filter out SNVs that appear in the filter_snv_pool from all spots.
        This function modifies both spot_snvs and filtered_spot_snvs.
        """
        if not self.filter_snv_pool:
            logger.warning("SNV pool is empty. No filtering will be performed.")
            return
            
        logger.info(f"Filtering out {len(self.filter_snv_pool)} SNVs from the pool")
        
        # Count SNVs before filtering, and the union of all SNVs
        before_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        before_union = set.union(*self.spot_snvs.values())
        
        # Filter out from spot_snvs
        for barcode in self.spot_snvs:
            self.spot_snvs[barcode] = self.spot_snvs[barcode] - self.filter_snv_pool
        
            
        # Filter out from filtered_spot_snvs
        for barcode in self.filtered_spot_snvs:
            self.filtered_spot_snvs[barcode] = self.filtered_spot_snvs[barcode] - self.filter_snv_pool
            
        # Count SNVs after filtering
        after_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        if self.spot_snvs:
            after_union = set.union(*self.spot_snvs.values())
        
        logger.info(f"Removed {before_total - after_total} SNVs from all spots")
        logger.info(f"Remaining SNVs: {after_total}")
        logger.info(f"SNVs in `remove vcf` union before: {len(before_union)}")
        logger.info(f"SNVs in `remove vcf` union after: {len(after_union)}")
        logger.info(f"Unique SNVs removed: {len(before_union - after_union)}")

    def filter_keep_snv_pool(self):
        """
        Keep only SNVs that appear in the include_snv_pool from all spots.
        This function modifies both spot_snvs and filtered_spot_snvs.
        """
        if not self.include_snv_pool:
            logger.warning("SNV inclusion pool is empty. No inclusion filtering will be performed.")
            return
            
        logger.info(f"Keeping only {len(self.include_snv_pool)} SNVs from the inclusion pool")
        
        # Count SNVs before filtering, and the union of all SNVs
        before_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        before_union = set.union(*self.spot_snvs.values()) if self.spot_snvs else set()
        
        # Filter to keep only SNVs in the inclusion pool
        for barcode in self.spot_snvs:
            self.spot_snvs[barcode] = self.spot_snvs[barcode] & self.include_snv_pool
            
        # Also filter the already filtered SNVs
        for barcode in self.filtered_spot_snvs:
            self.filtered_spot_snvs[barcode] = self.filtered_spot_snvs[barcode] & self.include_snv_pool
            
        # Count SNVs after filtering
        after_total = sum(len(snvs) for snvs in self.spot_snvs.values())
        if self.spot_snvs:
            after_union = set.union(*self.spot_snvs.values()) if self.spot_snvs else set()
        
        logger.info(f"Kept {after_total} SNVs out of {before_total} total")
        logger.info(f"Remaining SNVs: {after_total}")
        logger.info(f"SNVs union before inclusion filter: {len(before_union)}")
        logger.info(f"SNVs union after inclusion filter: {len(after_union)}")
        logger.info(f"Unique SNVs filtered out: {len(before_union - after_union)}")
        
    def load_scale_factors(self) -> Dict:
        """Load scale factors from JSON file."""
        try:
            with open(self.scale_factor_file, 'r') as f:
                scale_factors = json.load(f)
            logger.info(f"Loaded scale factors: {scale_factors}")
            return scale_factors
        except Exception as e:
            logger.error(f"Error loading scale factors: {e}")
            raise
            
    def load_spot_positions(self):
        """Load spot positions from tissue positions list CSV file."""
        try:
            header = 0 if self.cfg.has_header else None
            df = pd.read_csv(self.position_file, header=header)
            
            in_tissue_col = self.cfg.in_tissue_column  # column index for in_tissue flag
            
            for _, row in df.iterrows():
                barcode = str(row.iloc[0])
                
                # Skip out-tissue barcodes
                if barcode in self.out_tissue_barcodes:
                    continue
                
                # Check in_tissue flag if available
                if in_tissue_col is not None and in_tissue_col < len(row):
                    if int(row.iloc[in_tissue_col]) != 1:
                        continue
                
                # Pixel coordinates are in columns 4 and 5
                array_x = float(row.iloc[4])
                array_y = float(row.iloc[5])
                self.spot_positions[barcode] = (array_x, array_y)
            
            logger.info(f"Loaded {len(self.spot_positions)} spot positions")
        except Exception as e:
            logger.error(f"Error loading spot positions: {e}")
            raise
    
    def build_spatial_graph(self):
        """
        Build spatial graph by connecting neighboring spots in the hexagonal grid.
        This implementation is optimized for the hexagonal arrangement of Visium spots.
        """
        if not self.spot_positions:
            logger.info("Loading spot positions first...")
            self.load_spot_positions()
        
        # Get spot diameter from scale factors
        scale_factors = self.load_scale_factors()
        spot_diameter = scale_factors.get("spot_diameter_fullres", 100.0) * 6 # Slightly increased for safety
        
        # Convert positions to numpy array for processing
        barcodes = list(self.spot_positions.keys())
        positions = np.array([self.spot_positions[b] for b in barcodes])
        
        # Use k-nearest neighbors to find potential neighbors
        # For hexagonal grids, we should typically have 6 neighbors, but use MAX_NEIGHBORS
        # to account for edge cases or grid irregularities
        k = min(MAX_NEIGHBORS + 1, len(positions))  # +1 because it includes the point itself
        
        # Use scikit-learn's NearestNeighbors to find k nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(positions)
        distances, indices = nbrs.kneighbors(positions)
        
        # Calculate distance threshold based on spot diameter
        neighbor_threshold = self.neighbor_distance * spot_diameter
        
        # For each spot, determine its neighbors
        for i, barcode in enumerate(barcodes):
            # Get indices of neighbors (skip first index which is the point itself)
            neighbor_indices = indices[i, 1:]
            neighbor_distances = distances[i, 1:]
            
            # Filter neighbors by distance threshold
            valid_neighbors = [
                barcodes[neighbor_indices[j]] 
                for j in range(len(neighbor_indices)) 
                if neighbor_distances[j] <= neighbor_threshold
            ]
            
            self.spot_neighbors[barcode] = valid_neighbors
        
        # Visualize the hexagonal grid and neighbors (for debugging/validation)
        self.visualize_hexagonal_grid()
        
        # Log neighbor statistics
        neighbor_counts = [len(neighbors) for neighbors in self.spot_neighbors.values()]
        logger.info(f"Built spatial graph with {len(self.spot_neighbors)} spots")
        logger.info(f"Average neighbors per spot: {np.mean(neighbor_counts):.2f}")
        logger.info(f"Min neighbors: {min(neighbor_counts)}, Max neighbors: {max(neighbor_counts)}")
    
    def visualize_hexagonal_grid(self):
        """
        Create a visualization of the hexagonal grid and neighbor connections to validate
        the spatial graph building process.
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Get scale factor
        scale_factors = self.load_scale_factors()
        
        # Plot all spot positions
        x_coords = []
        y_coords = []
        for barcode, (x, y) in self.spot_positions.items():
            x_coords.append(x)
            y_coords.append(y)
        
        # Plot spots
        ax.scatter(x_coords, y_coords, c='blue', s=30, alpha=0.6)
        
        # Plot neighbor connections for a sample of spots (to avoid cluttering)
        # Pick ~20 random spots for connection visualization
        if len(self.spot_neighbors) > 20:
            import random
            sample_barcodes = random.sample(list(self.spot_neighbors.keys()), 20)
        else:
            sample_barcodes = list(self.spot_neighbors.keys())
        
        # Draw connections
        for barcode in sample_barcodes:
            x1, y1 = self.spot_positions[barcode]
            
            # Draw connections to neighbors
            for neighbor in self.spot_neighbors[barcode]:
                x2, y2 = self.spot_positions[neighbor]
                ax.plot([x1, x2], [y1, y2], 'r-', alpha=0.3, linewidth=0.5)
        
        # Add title and save
        ax.set_title(f"Spatial Graph - Hexagonal Grid Structure\n(Showing connections for {len(sample_barcodes)} sample spots)")
        ax.set_aspect('equal')
        
        # Save figure
        grid_viz_path = os.path.join(self.output_dir, "hexagonal_grid_visualization.png")
        plt.savefig(grid_viz_path, dpi=DPI, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved hexagonal grid visualization to: {grid_viz_path}")
        
    def load_snv_positions(self):
        """Load SNV positions for each barcode from text files."""
        # Find all barcode.txt files in the SNV positions directory
        txt_files = glob.glob(os.path.join(self.snv_pos_dir, "*.txt"))
        print(f"txt_files path: {os.path.join(self.snv_pos_dir, '*.txt')}")
        if not txt_files:
            logger.warning(f"No SNV position files found in {self.snv_pos_dir}")
            return
        
        logger.info(f"Loading SNV positions from {len(txt_files)} files...")
        
        # Process each file
        for txt_file in txt_files:
            barcode = os.path.basename(txt_file).replace('.txt', '')
            
            # Skip if barcode is not in our spatial data
            if barcode not in self.spot_positions:
                continue
                    
            # Load the SNVs
            try:
                with open(txt_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            chrom, pos = parts[0], parts[1]
                            chrom = chrom.replace("chr", "")  # Remove 'chr' prefix if present
                            snv_key = f"{chrom}_{pos}"
                            self.spot_snvs[barcode].add(snv_key)
                            
                            # Store ref and alt if available
                            if len(parts) >= 4:
                                ref, alt = parts[2], parts[3]
                                self.snv_ref_alt_map[snv_key] = (ref, alt)
            except Exception as e:
                logger.warning(f"Error loading SNVs from {txt_file}: {e}")
        
        # Log statistics
        spot_with_snvs = sum(1 for snvs in self.spot_snvs.values() if snvs)
        total_snvs = sum(len(snvs) for snvs in self.spot_snvs.values())
        
        logger.info(f"Loaded SNVs for {spot_with_snvs} spots")
        logger.info(f"Total SNVs: {total_snvs}")
        logger.info(f"Loaded ref/alt information for {len(self.snv_ref_alt_map)} unique variants")
        logger.info(f"Average SNVs per spot with SNVs: {total_snvs / max(1, spot_with_snvs):.2f}")
    
    def apply_spatial_filter_n_neighbours(self, min_neighbours: int = 2):
        """
        Apply spatial filtering to SNVs requiring presence in at least n neighboring spots.
        
        Args:
            min_neighbours: Minimum number of neighboring spots that must share the SNV (default: 2)
        
        Returns:
            Dictionary mapping barcodes to sets of filtered SNVs
        """
        if not self.spot_neighbors:
            logger.info("Building spatial graph first...")
            self.build_spatial_graph()
            
        if not self.spot_snvs:
            logger.info("Loading SNV positions first...")
            self.load_snv_positions()
        
        logger.info(f"Applying spatial filter requiring {min_neighbours} neighboring spots...")
        
        # Track statistics for kept variants
        kept_variants_added = 0
        kept_variants_unique = set()
        kept_variants_by_barcode = defaultdict(int)
        
        # Process each spot
        for barcode, snvs in self.spot_snvs.items():
            if not snvs:
                continue
                
            # Get neighbors
            neighbors = self.spot_neighbors.get(barcode, [])
            
            # Skip if insufficient neighbors (unless we have kept variants)
            if len(neighbors) < min_neighbours and not self.kept_variants_pool:
                continue
                
            # Check each SNV
            for snv in snvs:
                # If SNV is in kept_variants_pool, keep it directly
                if snv in self.kept_variants_pool:
                    self.filtered_spot_snvs[barcode].add(snv)
                    kept_variants_added += 1
                    kept_variants_unique.add(snv)
                    kept_variants_by_barcode[barcode] += 1
                    continue
                    
                # Skip neighbor check if insufficient neighbors
                if len(neighbors) < min_neighbours:
                    continue
                    
                # Count how many neighbors have this SNV
                neighbor_count = 0
                for neighbor in neighbors:
                    if snv in self.spot_snvs.get(neighbor, set()):
                        neighbor_count += 1
                
                # Keep SNV if it appears in at least min_neighbours neighboring spots
                if neighbor_count >= min_neighbours:
                    self.filtered_spot_snvs[barcode].add(snv)
        
        # Log statistics
        before_count = sum(len(snvs) for snvs in self.spot_snvs.values())
        after_count = sum(len(snvs) for snvs in self.filtered_spot_snvs.values())
        before_union = set.union(*self.spot_snvs.values()) if self.spot_snvs else set()
        after_union = set.union(*self.filtered_spot_snvs.values()) if self.filtered_spot_snvs else set()
        
        spots_before = sum(1 for snvs in self.spot_snvs.values() if snvs)
        spots_after = sum(1 for snvs in self.filtered_spot_snvs.values() if snvs)
        
        # Log kept variants statistics
        if self.kept_variants_pool:
            logger.info("\nKept Variants Statistics:")
            logger.info(f"Total kept variants specified: {len(self.kept_variants_pool)}")
            logger.info(f"Unique kept variants found in data: {len(kept_variants_unique)}")
            logger.info(f"Total kept variant instances added: {kept_variants_added}")
            logger.info(f"Spots with kept variants: {len(kept_variants_by_barcode)}")
            
            # Calculate how many variants were added *only* because they were in kept_variants_pool
            spatial_filter_only = after_count - kept_variants_added
            logger.info(f"SNVs kept due to spatial filtering only: {spatial_filter_only}")
            logger.info(f"SNVs kept due to kept_variants specification: {kept_variants_added} ({kept_variants_added/max(1,after_count)*100:.2f}% of total)")
            
            # Show distribution of kept variants per barcode
            variant_counts = list(kept_variants_by_barcode.values())
            if variant_counts:
                logger.info(f"Average kept variants per spot: {np.mean(variant_counts):.2f}")
                logger.info(f"Max kept variants per spot: {max(variant_counts)}")
        
        logger.info(f"\nOverall Filtering Statistics:")
        logger.info(f"Before filtering: {before_count} SNVs across {spots_before} spots")
        logger.info(f"After filtering: {after_count} SNVs across {spots_after} spots")
        logger.info(f"Before filter unique variants: {len(before_union)}")
        logger.info(f"After filter unique variants: {len(after_union)}")
        logger.info(f"Filtered out: {before_count - after_count} SNVs ({(before_count - after_count) / max(1, before_count) * 100:.2f}%)")
        logger.info(f"Filtered out: {len(before_union) - len(after_union)} unique SNVs")
        
        return self.filtered_spot_snvs
        
    def generate_snv_count_maps(self):
        """
        Generate maps of SNV counts before and after filtering.
        
        Returns:
            Two dictionaries mapping barcodes to SNV counts before and after filtering.
        """
        # Before filtering
        before_counts = {barcode: len(snvs) for barcode, snvs in self.spot_snvs.items()}
        
        # After filtering
        after_counts = {barcode: len(snvs) for barcode, snvs in self.filtered_spot_snvs.items()}
        
        # Fill in zeros for spots without SNVs
        for barcode in self.spot_positions:
            if barcode not in before_counts:
                before_counts[barcode] = 0
            if barcode not in after_counts:
                after_counts[barcode] = 0
        
        return before_counts, after_counts
    
    def visualize_snv_counts(self, count_map: Dict[str, int], title: str, output_file: str):
        """
        Visualize SNV counts on the tissue image.
        
        Args:
            count_map: Dict mapping barcode to SNV count
            title: Title for the plot
            output_file: Output file path for the visualization
        """
        try:
            # Load the image
            img = plt.imread(self.image_file)
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=FIGURE_SIZE)
            
            # Display the image
            ax.imshow(img, origin='upper')
            
            # Get scale factor
            scale_factors = self.load_scale_factors()
            scale = scale_factors.get('tissue_hires_scalef', 1.0)
            
            # Get maximum count for color normalization
            max_count = max(count_map.values())
            
            if max_count == 0:
                logger.warning("All counts are zero. Skipping visualization.")
                return
            
            # Create colormap using configured colors
            cmap = LinearSegmentedColormap.from_list('', COLORMAP)
            norm = Normalize(vmin=0, vmax=max_count)
            
            # Prepare scatter plot data
            x_coords = []
            y_coords = []
            colors = []
            sizes = []
            
            for barcode, count in count_map.items():
                if barcode in self.spot_positions:
                    x, y = self.spot_positions[barcode]
                    
                    # Scale coordinates for visualization
                    # Note: Coordinate systems might need to be flipped here
                    if self.dataset == "DLPFC":
                        x_scaled = y * scale
                        y_scaled = x * scale
                    else:
                        # For P4/P6, adjust if needed based on actual coordinates
                        x_scaled = y * scale
                        y_scaled = x * scale
                    
                    x_coords.append(x_scaled)
                    y_coords.append(y_scaled)
                    colors.append(count)
                    
                    # Adjust size based on count (minimum size of 30)
                    size = 30 + (count * 20 / max(1, max_count))
                    sizes.append(size)
            
            # Plot all points
            scatter = ax.scatter(x_coords, y_coords, 
                              c=colors, 
                              cmap=cmap,
                              norm=norm,
                              s=sizes,
                              alpha=0.7)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6)
            cbar.set_label('SNV Count')
            
            # Add title
            ax.set_title(title, fontsize=16)
            
            # Remove axes
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add summary stats to plot
            non_zero_counts = [c for c in count_map.values() if c > 0]
            stats_text = (
                f"Total Spots: {len(count_map)}\n"
                f"Spots with SNVs: {len(non_zero_counts)}\n"
                f"Total SNVs: {sum(count_map.values())}\n"
                f"Max SNVs per spot: {max_count}\n"
                f"Mean SNVs per spot: {np.mean(list(count_map.values())):.2f}"
            )
            
            # Add text box with stats
            props = dict(boxstyle='round', facecolor='white', alpha=0.8)
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)
            
            # Save figure
            plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualization saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            raise
    
    def run_analysis(self):
        """
        Run the complete spatial SNV filtering pipeline.
        """
        exclude_vcf_path = self.exclude_vcf_path
        include_vcf_path = self.include_vcf_path
        kept_variants_path = self.kept_variants_path
        min_neighbours = self.min_neighbours

        # 1. Load spot positions
        self.load_spot_positions()
        
        # 2. Build spatial graph
        self.build_spatial_graph()
        
        # 3. Load SNV positions
        self.load_snv_positions()
        
        # 3.5. Create and apply SNV filter pool if provided
        if exclude_vcf_path and os.path.exists(exclude_vcf_path):
            logger.info(f"Excluding SNVs from VCF: {exclude_vcf_path}")
            self.create_exclusion_snv_pool_from_vcf(exclude_vcf_path)
            self.filter_out_snv_pool()
        elif exclude_vcf_path:
            logger.warning(f"Exclude VCF file not found: {exclude_vcf_path}. Continuing without excluding SNVs.")
        
        # 3.6. Apply inclusion filter if provided
        if include_vcf_path and os.path.exists(include_vcf_path):
            logger.info(f"Including only SNVs from VCF: {include_vcf_path}")
            self.create_inclusion_snv_pool(include_vcf_path)
            self.filter_keep_snv_pool()
        elif include_vcf_path:
            logger.warning(f"Include VCF file not found: {include_vcf_path}. Continuing without inclusion filtering.")
        
        # 3.7. Load kept variants if provided
        if kept_variants_path and os.path.exists(kept_variants_path):
            logger.info(f"Loading variants to keep directly: {kept_variants_path}")
            self.create_kept_variants_pool(kept_variants_path)
        elif kept_variants_path:
            logger.warning(f"Kept variants VCF file not found: {kept_variants_path}. Continuing without kept variants.")
        
        # 4. Apply spatial filter
        self.apply_spatial_filter_n_neighbours(min_neighbours=min_neighbours)

        
        # 5. Generate count maps
        before_counts, after_counts = self.generate_snv_count_maps()
        
        # 6. Create visualizations
        # Add SNV pool info to filenames if used
        pool_suffix = ""
        if exclude_vcf_path:
            vcf_basename = os.path.basename(exclude_vcf_path)
            pool_suffix = f"_no_{vcf_basename.replace('.vcf.gz', '')}"
        if include_vcf_path:
            vcf_basename = os.path.basename(include_vcf_path)
            include_suffix = f"_only_{vcf_basename.replace('.vcf.gz', '')}"
            pool_suffix = pool_suffix + include_suffix
        
        before_output = os.path.join(self.output_dir, f"snv_counts_before_filtering{pool_suffix}.png")
        after_output = os.path.join(self.output_dir, f"snv_counts_after_filtering{pool_suffix}.png")
        
        pool_info = ""
        if exclude_vcf_path:
            pool_size = len(self.filter_snv_pool)
            pool_info = f" (Excluded {pool_size} SNVs from: {os.path.basename(exclude_vcf_path)})"
        if include_vcf_path:
            include_size = len(self.include_snv_pool)
            include_info = f" (Kept only {include_size} SNVs from: {os.path.basename(include_vcf_path)})"
            pool_info = pool_info + include_info
        
        self.visualize_snv_counts(
            before_counts, 
            f"SNV Counts Before Spatial Filtering - {self.dataset} {self.section_id}{pool_info}", 
            before_output
        )
        
        self.visualize_snv_counts(
            after_counts, 
            f"SNV Counts After Spatial Filtering - {self.dataset} {self.section_id}{pool_info}",
            after_output
        )
        
        # 7. Save filtered SNVs to files
        # Determine subdirectory name for filtered SNVs
        filtered_subdir = "filtered_snvs"
        if pool_suffix:
            filtered_subdir += pool_suffix
            
        filtered_dir = os.path.join(self.output_dir, filtered_subdir)
        self.save_filtered_snvs(filtered_dir)
        
        return {
            "before_counts": before_counts,
            "after_counts": after_counts,
            "visualizations": {
                "before": before_output,
                "after": after_output
            },
            "filtered_dir": filtered_dir
        }
        
    def save_filtered_snvs(self, output_dir: str = None):
        """
        Save filtered SNVs to text files.
        
        Args:
            output_dir: Optional custom directory to save filtered SNVs
        """
        filtered_dir = output_dir if output_dir else os.path.join(self.output_dir, "filtered_snvs")
        os.makedirs(filtered_dir, exist_ok=True)
        
        logger.info(f"Saving filtered SNVs to {filtered_dir}")
        
        # Create a summary file with all filtered variants
        all_variants_file = os.path.join(filtered_dir, "all_filtered_variants.txt")
        all_variants = set()
        
        # Save for each barcode
        for barcode, snvs in self.filtered_spot_snvs.items():
            if not snvs:
                continue
                    
            output_file = os.path.join(filtered_dir, f"{barcode}.txt")
            with open(output_file, 'w') as f:
                for snv in sorted(snvs):
                    chrom, pos = snv.split('_', 1)
                    # Get ref and alt from our dictionary, default to "N" if not found
                    ref, alt = self.snv_ref_alt_map.get(snv, ("N", "N"))
                    f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")
                    all_variants.add((chrom, pos, ref, alt))
        
        # Write the summary file with all variants
        with open(all_variants_file, 'w') as f:
            f.write("Chrom\tPos\tRef\tAlt\n")
            for chrom, pos, ref, alt in sorted(all_variants, key=lambda x: (x[0], int(x[1]))):
                f.write(f"{chrom}\t{pos}\t{ref}\t{alt}\n")
        
        logger.info(f"Saved summary of all filtered variants to: {all_variants_file}")
        
        # Also save a summary file
        summary_file = os.path.join(self.output_dir, "spatial_filtering_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Spatial SNV Filtering Summary\n")
            f.write(f"============================\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            if self.section_id:
                f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n\n")
            
            spots_before = sum(1 for snvs in self.spot_snvs.values() if snvs)
            spots_after = sum(1 for snvs in self.filtered_spot_snvs.values() if snvs)
            
            snvs_before = sum(len(snvs) for snvs in self.spot_snvs.values())
            snvs_after = sum(len(snvs) for snvs in self.filtered_spot_snvs.values())
            
            f.write(f"Total spots: {len(self.spot_positions)}\n")
            f.write(f"Spots with SNVs before filtering: {spots_before}\n")
            f.write(f"Spots with SNVs after filtering: {spots_after}\n\n")
            
            f.write(f"Total SNVs before filtering: {snvs_before}\n")
            f.write(f"Total SNVs after filtering: {snvs_after}\n")
            f.write(f"SNVs filtered out: {snvs_before - snvs_after} ({(snvs_before - snvs_after) / max(1, snvs_before) * 100:.2f}%)\n")
            
            # Add information about the SNV pool if used
            if self.filter_snv_pool:
                f.write(f"\nSNV Pool Information:\n")
                f.write(f"SNVs in exclusion pool: {len(self.filter_snv_pool)}\n")
            
            if self.include_snv_pool:
                f.write(f"\nSNV Inclusion Pool Information:\n")
                f.write(f"SNVs in inclusion pool: {len(self.include_snv_pool)}\n")
                snvs_in_inclusion_pool = sum(1 for snvs in self.filtered_spot_snvs.values() 
                                            for snv in snvs if snv in self.include_snv_pool)
                f.write(f"SNVs in final result also in inclusion pool: {snvs_in_inclusion_pool}\n")
                
            # Add information about kept variants
            if self.kept_variants_pool:
                # Count how many kept variants made it to the final set
                kept_in_final = set()
                kept_count = 0
                for snvs in self.filtered_spot_snvs.values():
                    for snv in snvs:
                        if snv in self.kept_variants_pool:
                            kept_count += 1
                            kept_in_final.add(snv)
                            
                f.write(f"\nKept Variants Information:\n")
                f.write(f"SNVs in kept variants pool: {len(self.kept_variants_pool)}\n")
                f.write(f"Unique kept variants found in data: {len(kept_in_final)}\n")
                f.write(f"Total kept variant instances in final set: {kept_count}\n")
                
            f.write(f"\nNeighbor distance threshold: {self.neighbor_distance} spot diameters\n")
            f.write(f"Average neighbors per spot: {np.mean([len(n) for n in self.spot_neighbors.values()]):.2f}\n")
            f.write(f"\nAll filtered variants summary saved to: {all_variants_file}\n")
            f.write(f"Total unique filtered variants: {len(all_variants)}\n")

def main():
    """Main function to run the spatial SNV filtering pipeline."""
    parser = argparse.ArgumentParser(description="SPARCAL Step 6a: Spatial SNV Filtering Pipeline")
    
    parser.add_argument("--dataset", required=True, help="Dataset name (must match YAML config)")
    parser.add_argument("--section_id", help="Section ID")
    parser.add_argument("--neighbor_distance", type=float, default=DEFAULT_NEIGHBOR_DISTANCE,
                    help=f"Distance threshold for neighboring spots (default: {DEFAULT_NEIGHBOR_DISTANCE})")
    parser.add_argument("--output_dir", help="Custom output directory (overrides default)")
    parser.add_argument("--exclude_vcf", help="VCF with SNVs to exclude")
    parser.add_argument("--include_vcf", help="VCF with SNVs to include (keep only these)")
    parser.add_argument("--kept_variants", help="VCF with SNVs to keep directly (bypass spatial filter)")
    parser.add_argument("--min_neighbours", type=int, default=2,
                    help="Min neighboring spots sharing an SNV (default: 2)")
    parser.add_argument("--out_tissue_file", help="File with out-tissue barcodes to exclude")
    
    args = parser.parse_args()
    
    cfg = load_config(args.dataset, args.section_id)
    
    # Create spatial filter
    filter = SpatialSNVFilter(
        cfg=cfg,
        neighbor_distance=args.neighbor_distance,
        exclude_vcf_path=args.exclude_vcf,
        include_vcf_path=args.include_vcf,
        kept_variants_path=args.kept_variants,
        min_neighbours=args.min_neighbours,
        out_tissue_file=args.out_tissue_file
    )
    

    # Override output directory if specified
    if args.output_dir:
        filter.output_dir = args.output_dir
        os.makedirs(filter.output_dir, exist_ok=True)
        logger.info(f"Using custom output directory: {filter.output_dir}")
    
    # Run analysis
    results = filter.run_analysis()
    
    logger.info("Analysis complete!")
    logger.info(f"Output directory: {filter.output_dir}")

    # print out the set size of original, filtered, and excluded SNVs
    # logger.info(f"Original SNV pool size: {len(filter.spot_snvs)}")
    # logger.info(f"Filtered SNV pool size: {sum(len(snvs) for snvs in filter.filtered_spot_snvs.values())}")
    # logger.info(f"Excluded SNV pool size: {len(filter.spot_snv) - sum(len(snvs) for snvs in filter.filtered_spot_snvs.values())}")
    
    return filter, results


if __name__ == "__main__":
    filter, results = main()

# Usage for P4_TUMOR dataset, keep Beagle directly, section 1, calc inclusion with /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz
# python scripts/6_spatial_filter/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --exclude_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz --min_neighbours 1 --include_vcf /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz

# Usage for P4_TUMOR, section 1
# python scripts/6_spatial_filter/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --kept_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz --min_neighbours 1

# python scripts/6_spatial_filter/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --include_vcf /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz --min_neighbours 6
# python scripts/6_spatial/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --exclude_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz --include_vcf /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_exome_snps.vcf --min_neighbours 6
# python scripts/6_spatial/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --exclude_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz --include_vcf /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz --min_neighbours 0
# python scripts/6_spatial/run_spatial_snv_filter.py --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 --exclude_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz  

# /lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz

# Run on DLPFC 151507, include beagle SNVs, min_neighbours=2
# python scripts/6_spatial_filter/run_spatial_snv_filter.py --dataset dlpfc --section_id 151673 --quality_filter baseQ0mapQ0 --min_neighbours 2 --exclude_vcf /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/beagle/baseQ0mapQ0/all_filtered_out.vcf.gz
# Run on DLPFC 151507 baseQ13mapQ20, kept denovo variants, min_neighbours=3
# python scripts/6_spatial_filter/run_spatial_snv_filter.py --dataset dlpfc --section_id 151508 --quality_filter baseQ0mapQ0 --min_neighbours 1