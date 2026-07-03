#!/usr/bin/env python3
"""
Calculate and Visualize Germline/Somatic Scores for Mutect2 Overlap SNVs
==========================================================================
This script loads the 88 somatic SNVs from Mutect2 overlap, calculates their
germline and somatic scores using the spatial filter scoring system, and 
creates visualizations.
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, Set, Tuple, Optional
import scipy.stats
from scipy.spatial import distance_matrix
from sklearn.neighbors import NearestNeighbors
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Mutect2OverlapScoreCalculator:
    """
    Calculate germline and somatic scores for Mutect2 overlap variants.
    """
    
    def __init__(self,
                 mutect2_vcf: str,
                 dataset: str,
                 section_id: str,
                 quality_filter: str,
                 tumor_purity_file: str,
                 spatial_positions_file: str,
                 snv_vcf_dir: str,
                 neighbor_distance: float = 2.0):
        """
        Initialize the calculator.
        
        Parameters:
        -----------
        mutect2_vcf : str
            Path to the Mutect2 overlap VCF file (88 somatic SNVs)
        dataset : str
            Dataset name (e.g., 'P4_TUMOR')
        section_id : str
            Section ID (e.g., '1')
        quality_filter : str
            Quality filter used (e.g., 'baseQ0mapQ0')
        tumor_purity_file : str
            Path to CalicoST tumor purity file
        spatial_positions_file : str
            Path to spatial positions CSV
        snv_vcf_dir : str
            Directory containing per-spot SNV VCF files
        neighbor_distance : float
            Distance threshold for spatial neighbors (default: 2.0)
        """
        self.mutect2_vcf = mutect2_vcf
        self.dataset = dataset.upper()
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.tumor_purity_file = tumor_purity_file
        self.spatial_positions_file = spatial_positions_file
        self.snv_vcf_dir = snv_vcf_dir
        self.neighbor_distance = neighbor_distance
        
        # Data structures
        self.mutect2_variants = set()  # Set of SNV keys from Mutect2
        self.spot_positions = {}  # barcode -> (x, y)
        self.tumor_purity = {}  # barcode -> purity value
        self.spot_neighbors = defaultdict(list)  # barcode -> list of neighbor barcodes
        self.spot_snvs = defaultdict(set)  # barcode -> set of SNV keys
        self.variant_scores = {}  # variant -> {'germline': score, 'somatic': score}
        
        # Score weights (same as in run_spatial_snv_filter_enhanced.py)
        self.weights_germline = {
            'alpha': 0.4,   # spatial uniformity
            'beta': 0.3,    # global prevalence
            'gamma': 0.3    # purity independence
        }
        
        self.weights_somatic = {
            'delta': 0.5,   # purity correlation
            'epsilon': 0.2, # clone-specific
            'zeta': 0.3     # spatial clustering
        }
    
    def load_mutect2_variants(self):
        """Load the 88 somatic SNVs from Mutect2 overlap VCF."""
        logger.info(f"Loading Mutect2 variants from: {self.mutect2_vcf}")
        
        open_func = gzip.open if self.mutect2_vcf.endswith('.gz') else open
        
        with open_func(self.mutect2_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                chrom = fields[0]
                pos = fields[1]
                ref = fields[3]
                alt = fields[4]
                
                # Create SNV key: chr_pos_ref_alt
                snv_key = f"{chrom}_{pos}_{ref}_{alt}"
                self.mutect2_variants.add(snv_key)
        
        logger.info(f"Loaded {len(self.mutect2_variants)} Mutect2 variants")
    
    def load_tumor_purity(self):
        """Load tumor purity data from CalicoST output."""
        logger.info(f"Loading tumor purity from: {self.tumor_purity_file}")
        
        with open(self.tumor_purity_file, 'r') as f:
            header = f.readline().strip().split('\t')
            
            # Find the tumor proportion column (try multiple possible names)
            purity_col = None
            possible_names = ['Tumor', 'tumor_prop', 'tumor', 'purity', 'Tumor_Prop']
            for name in possible_names:
                if name in header:
                    purity_col = header.index(name)
                    break
            
            if purity_col is None:
                raise ValueError(f"Could not find tumor purity column. Header: {header}")
            
            # First column is usually barcodes
            barcode_col = 0
            
            skipped = 0
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) <= purity_col:
                    continue
                    
                barcode = fields[barcode_col]
                
                # Remove any suffix from barcode (e.g., _P41)
                if '_' in barcode:
                    barcode = barcode.split('_')[0]
                
                # Handle empty or missing values
                purity_str = fields[purity_col].strip()
                if not purity_str or purity_str == '' or purity_str == 'NA':
                    skipped += 1
                    continue
                
                try:
                    purity = float(purity_str)
                    self.tumor_purity[barcode] = purity
                except ValueError:
                    skipped += 1
                    continue
        
        logger.info(f"Loaded purity for {len(self.tumor_purity)} spots")
        if skipped > 0:
            logger.info(f"Skipped {skipped} spots with missing/invalid purity values")
    
    def load_spatial_positions(self):
        """Load spatial positions of spots."""
        logger.info(f"Loading spatial positions from: {self.spatial_positions_file}")
        
        with open(self.spatial_positions_file, 'r') as f:
            for line in f:
                fields = line.strip().split(',')
                barcode = fields[0]
                
                # Check if this spot is in tissue
                in_tissue = int(fields[1])
                if in_tissue == 0:
                    continue
                
                # Get array coordinates
                array_row = int(fields[2])
                array_col = int(fields[3])
                
                self.spot_positions[barcode] = (array_row, array_col)
        
        logger.info(f"Loaded positions for {len(self.spot_positions)} in-tissue spots")
    
    def build_spatial_graph(self):
        """Build spatial neighbor graph."""
        logger.info("Building spatial neighbor graph...")
        
        barcodes = list(self.spot_positions.keys())
        positions = np.array([self.spot_positions[bc] for bc in barcodes])
        
        # Use sklearn's NearestNeighbors for efficient neighbor finding
        nbrs = NearestNeighbors(radius=self.neighbor_distance, algorithm='ball_tree')
        nbrs.fit(positions)
        
        # Find neighbors for each spot
        for i, barcode in enumerate(barcodes):
            indices = nbrs.radius_neighbors([positions[i]], return_distance=False)[0]
            neighbors = [barcodes[j] for j in indices if j != i]
            self.spot_neighbors[barcode] = neighbors
        
        avg_neighbors = np.mean([len(neighs) for neighs in self.spot_neighbors.values()])
        logger.info(f"Built spatial graph. Average neighbors per spot: {avg_neighbors:.2f}")
    
    def load_spot_snvs(self):
        """Load SNV data from per-spot VCF files."""
        logger.info(f"Loading SNV data from: {self.snv_vcf_dir}")
        
        # Get all VCF files (both .vcf.gz and .vcf)
        vcf_files = [f for f in os.listdir(self.snv_vcf_dir) 
                    if f.endswith('.vcf.gz') or f.endswith('.vcf')]
        
        logger.info(f"Found {len(vcf_files)} VCF files to process")
        
        skipped_corrupted = 0
        skipped_no_position = 0
        
        for vcf_file in vcf_files:
            # Extract barcode from filename
            if vcf_file.endswith('.vcf.gz'):
                barcode = vcf_file.replace('.vcf.gz', '')
            else:
                barcode = vcf_file.replace('.vcf', '')
            
            # Only process barcodes that have positions
            if barcode not in self.spot_positions:
                skipped_no_position += 1
                continue
            
            vcf_path = os.path.join(self.snv_vcf_dir, vcf_file)
            
            try:
                # Try to open as gzipped first
                if vcf_file.endswith('.vcf.gz'):
                    try:
                        with gzip.open(vcf_path, 'rt') as f:
                            # Test if it's actually gzipped by reading first byte
                            f.read(1)
                            f.seek(0)
                            self._parse_vcf(f, barcode)
                    except gzip.BadGzipFile:
                        # Not actually gzipped, try as plain text
                        with open(vcf_path, 'r') as f:
                            self._parse_vcf(f, barcode)
                else:
                    # Plain VCF file
                    with open(vcf_path, 'r') as f:
                        self._parse_vcf(f, barcode)
            
            except Exception as e:
                logger.warning(f"Skipping corrupted/unreadable file {vcf_file}: {str(e)}")
                skipped_corrupted += 1
                continue
        
        logger.info(f"Loaded SNV data for {len(self.spot_snvs)} spots")
        if skipped_no_position > 0:
            logger.info(f"Skipped {skipped_no_position} files (barcode not in spatial positions)")
        if skipped_corrupted > 0:
            logger.warning(f"Skipped {skipped_corrupted} corrupted/unreadable files")
    
    def _parse_vcf(self, file_handle, barcode):
        """Parse VCF file and extract variants."""
        for line in file_handle:
            if line.startswith('#'):
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 5:
                continue
            
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            
            snv_key = f"{chrom}_{pos}_{ref}_{alt}"
            
            # Only include if it's one of the Mutect2 variants
            if snv_key in self.mutect2_variants:
                self.spot_snvs[barcode].add(snv_key)
    
    def calculate_scores(self):
        """Calculate germline and somatic scores for all Mutect2 variants."""
        logger.info("Calculating germline and somatic scores...")
        
        total_spots = len(self.spot_positions)
        
        for variant in self.mutect2_variants:
            # Find spots with this variant
            spots_with_variant = [bc for bc, snvs in self.spot_snvs.items() if variant in snvs]
            n_spots = len(spots_with_variant)
            
            if n_spots == 0:
                # Variant not observed in any spot
                self.variant_scores[variant] = {
                    'germline_score': 0.0,
                    'somatic_score': 0.0,
                    'n_spots': 0,
                    'global_prevalence': 0.0
                }
                continue
            
            # ============ GERMLINE SCORE ============
            
            # 1. Spatial uniformity (alpha)
            cv_spatial = self.calculate_spatial_cv(spots_with_variant)
            alpha_score = 1.0 - cv_spatial  # Lower CV = more uniform = higher score
            
            # 2. Global prevalence (beta)
            global_prevalence = n_spots / total_spots
            beta_score = global_prevalence
            
            # 3. Purity independence (gamma)
            gamma_score = self.calculate_purity_independence(spots_with_variant)
            
            # Combined germline score
            germline_score = (
                self.weights_germline['alpha'] * alpha_score +
                self.weights_germline['beta'] * beta_score +
                self.weights_germline['gamma'] * gamma_score
            )
            
            # ============ SOMATIC SCORE ============
            
            # 1. Purity correlation (delta)
            delta_score = self.calculate_purity_correlation(spots_with_variant)
            
            # 2. Clone-specific (epsilon)
            epsilon_score = self.calculate_clone_specificity(spots_with_variant)
            
            # 3. Spatial clustering (zeta)
            zeta_score = self.calculate_spatial_clustering(spots_with_variant)
            
            # Combined somatic score
            somatic_score = (
                self.weights_somatic['delta'] * delta_score +
                self.weights_somatic['epsilon'] * epsilon_score +
                self.weights_somatic['zeta'] * zeta_score
            )
            
            # Store scores
            self.variant_scores[variant] = {
                'germline_score': germline_score,
                'somatic_score': somatic_score,
                'n_spots': n_spots,
                'global_prevalence': global_prevalence,
                'alpha': alpha_score,
                'beta': beta_score,
                'gamma': gamma_score,
                'delta': delta_score,
                'epsilon': epsilon_score,
                'zeta': zeta_score
            }
        
        logger.info(f"Calculated scores for {len(self.variant_scores)} variants")
    
    def calculate_spatial_cv(self, spots_with_variant: list) -> float:
        """Calculate coefficient of variation for spatial distribution."""
        if len(spots_with_variant) < 2:
            return 1.0  # Maximum variability for single spot
        
        # Get positions
        positions = np.array([self.spot_positions[bc] for bc in spots_with_variant])
        
        # Calculate distances from centroid
        centroid = positions.mean(axis=0)
        distances = np.linalg.norm(positions - centroid, axis=1)
        
        if distances.mean() == 0:
            return 0.0
        
        cv = distances.std() / distances.mean()
        return min(cv, 1.0)  # Cap at 1.0
    
    def calculate_purity_independence(self, spots_with_variant: list) -> float:
        """Calculate independence from tumor purity (Kolmogorov-Smirnov test)."""
        if len(spots_with_variant) < 5:
            return 0.5  # Neutral score for low sample size
        
        # Get purity values for spots with and without variant (only spots that have purity)
        purity_with = [self.tumor_purity[bc] for bc in spots_with_variant 
                      if bc in self.tumor_purity]
        purity_without = [self.tumor_purity[bc] for bc in self.spot_positions.keys() 
                         if bc not in spots_with_variant and bc in self.tumor_purity]
        
        if len(purity_with) < 5 or len(purity_without) < 5:
            return 0.5
        
        # KS test: high p-value = distributions are similar = independent
        ks_stat, p_value = scipy.stats.ks_2samp(purity_with, purity_without)
        
        # Convert p-value to score (higher p-value = more independent = higher score)
        return p_value
    
    def calculate_purity_correlation(self, spots_with_variant: list) -> float:
        """Calculate correlation with tumor purity (point-biserial)."""
        if len(spots_with_variant) < 5:
            return 0.5  # Neutral score
        
        # Create binary variable for variant presence (only for spots with purity)
        all_barcodes = [bc for bc in self.spot_positions.keys() if bc in self.tumor_purity]
        
        if len(all_barcodes) < 10:
            return 0.5  # Not enough spots with purity data
        
        variant_presence = [1 if bc in spots_with_variant else 0 for bc in all_barcodes]
        purity_values = [self.tumor_purity[bc] for bc in all_barcodes]
        
        # Point-biserial correlation
        try:
            correlation, p_value = scipy.stats.pointbiserialr(variant_presence, purity_values)
            # Return absolute correlation (higher = more associated with purity)
            return abs(correlation)
        except:
            return 0.5  # Neutral if calculation fails
    
    def calculate_clone_specificity(self, spots_with_variant: list) -> float:
        """Calculate clone specificity based on purity distribution."""
        if len(spots_with_variant) < 3:
            return 0.5
        
        purity_with = [self.tumor_purity.get(bc, 0) for bc in spots_with_variant]
        
        # Clone-specific variants should be present in high-purity regions
        mean_purity = np.mean(purity_with)
        
        # Score based on mean purity (higher = more clone-specific)
        return mean_purity
    
    def calculate_spatial_clustering(self, spots_with_variant: list) -> float:
        """Calculate spatial clustering score."""
        if len(spots_with_variant) < 3:
            return 0.5
        
        # Count how many neighbors each spot has that also have the variant
        neighbor_counts = []
        for spot in spots_with_variant:
            neighbors_with_variant = sum(1 for neighbor in self.spot_neighbors[spot] 
                                        if neighbor in spots_with_variant)
            total_neighbors = len(self.spot_neighbors[spot])
            
            if total_neighbors > 0:
                neighbor_counts.append(neighbors_with_variant / total_neighbors)
        
        if not neighbor_counts:
            return 0.5
        
        # Higher mean = more clustered
        return np.mean(neighbor_counts)
    
    def save_scores(self, output_file: str):
        """Save scores to a TSV file."""
        logger.info(f"Saving scores to: {output_file}")
        
        # Convert to DataFrame
        rows = []
        for variant, scores in self.variant_scores.items():
            chrom, pos, ref, alt = variant.split('_')
            row = {
                'chrom': chrom,
                'pos': int(pos),
                'ref': ref,
                'alt': alt,
                'variant': variant,
                'germline_score': scores['germline_score'],
                'somatic_score': scores['somatic_score'],
                'n_spots': scores['n_spots'],
                'global_prevalence': scores['global_prevalence'],
                'alpha_spatial_uniformity': scores.get('alpha', 0),
                'beta_global_prevalence': scores.get('beta', 0),
                'gamma_purity_independence': scores.get('gamma', 0),
                'delta_purity_correlation': scores.get('delta', 0),
                'epsilon_clone_specific': scores.get('epsilon', 0),
                'zeta_spatial_clustering': scores.get('zeta', 0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['chrom', 'pos'])
        df.to_csv(output_file, sep='\t', index=False)
        
        logger.info(f"Saved {len(df)} variant scores")
        return df
    
    def plot_scatter(self, output_file: str, 
                    germline_threshold: float = 0.25,
                    somatic_threshold: float = 0.2):
        """
        Create scatter plot of germline vs somatic scores.
        """
        logger.info(f"Creating scatter plot: {output_file}")
        
        # Prepare data
        germline_scores = [v['germline_score'] for v in self.variant_scores.values()]
        somatic_scores = [v['somatic_score'] for v in self.variant_scores.values()]
        
        # Classify variants
        classifications = []
        for g_score, s_score in zip(germline_scores, somatic_scores):
            if g_score > germline_threshold and s_score < somatic_threshold:
                classifications.append('Germline')
            elif s_score > somatic_threshold and g_score < germline_threshold:
                classifications.append('Somatic')
            else:
                classifications.append('Ambiguous')
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color mapping
        color_map = {
            'Germline': '#3498db',    # Blue
            'Somatic': '#e74c3c',     # Red
            'Ambiguous': '#95a5a6'    # Gray
        }
        
        # Plot each classification
        for cls in ['Ambiguous', 'Germline', 'Somatic']:
            mask = [c == cls for c in classifications]
            g_subset = [g for g, m in zip(germline_scores, mask) if m]
            s_subset = [s for s, m in zip(somatic_scores, mask) if m]
            
            if len(g_subset) > 0:
                ax.scatter(s_subset, g_subset,
                          c=color_map[cls], label=cls,
                          alpha=0.6, s=100, edgecolors='white', linewidth=1.0)
        
        # Add classification regions
        germline_rect = Rectangle((0, germline_threshold), somatic_threshold,
                                  1-germline_threshold,
                                  facecolor='blue', alpha=0.05, edgecolor='blue',
                                  linewidth=2, linestyle='--')
        ax.add_patch(germline_rect)
        
        somatic_rect = Rectangle((somatic_threshold, 0), 1-somatic_threshold,
                                 somatic_threshold,
                                 facecolor='red', alpha=0.05, edgecolor='red',
                                 linewidth=2, linestyle='--')
        ax.add_patch(somatic_rect)
        
        # Add threshold lines
        ax.axhline(y=germline_threshold, color='blue', linestyle='--',
                  linewidth=1.5, alpha=0.5, label=f'Germline threshold ({germline_threshold})')
        ax.axvline(x=somatic_threshold, color='red', linestyle='--',
                  linewidth=1.5, alpha=0.5, label=f'Somatic threshold ({somatic_threshold})')
        
        # Add diagonal
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, label='Equal scores')
        
        # Labels
        ax.set_xlabel('Somatic Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Germline Score', fontsize=14, fontweight='bold')
        ax.set_title('Mutect2 Overlap SNVs: Germline vs Somatic Scores\n(88 Known Somatic Variants)',
                    fontsize=16, fontweight='bold', pad=20)
        
        # Limits
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        
        # Grid
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Legend
        ax.legend(loc='center right', fontsize=10, framealpha=0.9)
        
        # Statistics box
        n_germline = classifications.count('Germline')
        n_somatic = classifications.count('Somatic')
        n_ambiguous = classifications.count('Ambiguous')
        total = len(classifications)
        
        stats_text = f"""Classification Summary:
Germline: {n_germline} ({n_germline/total*100:.1f}%)
Somatic: {n_somatic} ({n_somatic/total*100:.1f}%)
Ambiguous: {n_ambiguous} ({n_ambiguous/total*100:.1f}%)
Total: {total}

Note: These are known somatic
variants from Mutect2"""
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props, family='monospace')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Scatter plot saved to: {output_file}")
        plt.close()
        
        # Print summary
        logger.info(f"\nClassification Summary:")
        logger.info(f"  Germline: {n_germline} ({n_germline/total*100:.1f}%)")
        logger.info(f"  Somatic: {n_somatic} ({n_somatic/total*100:.1f}%)")
        logger.info(f"  Ambiguous: {n_ambiguous} ({n_ambiguous/total*100:.1f}%)")
    
    def run_analysis(self, output_prefix: str):
        """Run the complete analysis pipeline."""
        logger.info("="*60)
        logger.info("Starting Mutect2 Overlap Score Analysis")
        logger.info("="*60)
        
        # Load all data
        self.load_mutect2_variants()
        self.load_tumor_purity()
        self.load_spatial_positions()
        self.build_spatial_graph()
        self.load_spot_snvs()
        
        # Calculate scores
        self.calculate_scores()
        
        # Save outputs
        scores_file = f"{output_prefix}_variant_scores.txt"
        df = self.save_scores(scores_file)
        
        # Create plot
        plot_file = f"{output_prefix}_score_scatter.png"
        self.plot_scatter(plot_file)
        
        logger.info("="*60)
        logger.info("Analysis complete!")
        logger.info(f"Output files:")
        logger.info(f"  - Scores: {scores_file}")
        logger.info(f"  - Plot: {plot_file}")
        logger.info("="*60)
        
        return df


def main():
    parser = argparse.ArgumentParser(
        description='Calculate and visualize germline/somatic scores for Mutect2 overlap SNVs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Usage:
  python %(prog)s \\
    --mutect2_vcf /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_MPILEUP/overlap_MPILEUP_P4_somatic_Mutect2_all/0000.vcf.gz \\
    --dataset P4_TUMOR \\
    --section_id 1 \\
    --quality_filter baseQ0mapQ0 \\
    --tumor_purity /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \\
    --spatial_positions /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/Meta_Data/GSM4565823_P4_rep1_tissue_positions_list.csv \\
    --snv_vcf_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/BAM_filtered/baseQ0mapQ0/snv_vcf \\
    --output_prefix mutect2_overlap_analysis
        """
    )
    
    # Required arguments
    parser.add_argument('--mutect2_vcf', required=True,
                       help='Path to Mutect2 overlap VCF file (88 somatic SNVs)')
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g., P4_TUMOR, P6_TUMOR)')
    parser.add_argument('--section_id', required=True,
                       help='Section ID (e.g., 1, 2)')
    parser.add_argument('--quality_filter', required=True,
                       help='Quality filter (e.g., baseQ0mapQ0)')
    parser.add_argument('--tumor_purity', required=True,
                       help='Path to CalicoST tumor purity file')
    parser.add_argument('--spatial_positions', required=True,
                       help='Path to spatial positions CSV file')
    parser.add_argument('--snv_vcf_dir', required=True,
                       help='Directory containing per-spot SNV VCF files')
    
    # Optional arguments
    parser.add_argument('--output_prefix', default='mutect2_overlap_analysis',
                       help='Output file prefix (default: mutect2_overlap_analysis)')
    parser.add_argument('--neighbor_distance', type=float, default=2.0,
                       help='Distance threshold for spatial neighbors (default: 2.0)')
    parser.add_argument('--germline_threshold', type=float, default=0.25,
                       help='Germline score threshold (default: 0.25)')
    parser.add_argument('--somatic_threshold', type=float, default=0.2,
                       help='Somatic score threshold (default: 0.2)')
    
    args = parser.parse_args()
    
    # Validate files exist
    for file_path, name in [(args.mutect2_vcf, 'Mutect2 VCF'),
                            (args.tumor_purity, 'Tumor purity'),
                            (args.spatial_positions, 'Spatial positions')]:
        if not os.path.exists(file_path):
            logger.error(f"{name} file not found: {file_path}")
            sys.exit(1)
    
    if not os.path.isdir(args.snv_vcf_dir):
        logger.error(f"SNV VCF directory not found: {args.snv_vcf_dir}")
        sys.exit(1)
    
    # Create calculator
    calculator = Mutect2OverlapScoreCalculator(
        mutect2_vcf=args.mutect2_vcf,
        dataset=args.dataset,
        section_id=args.section_id,
        quality_filter=args.quality_filter,
        tumor_purity_file=args.tumor_purity,
        spatial_positions_file=args.spatial_positions,
        snv_vcf_dir=args.snv_vcf_dir,
        neighbor_distance=args.neighbor_distance
    )
    
    # Run analysis
    df = calculator.run_analysis(args.output_prefix)
    
    return calculator, df


if __name__ == "__main__":
    calculator, df = main()

# Example command to run the script:
# python analyze_overlap.py \
#     --mutect2_vcf /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_MPILEUP/overlap_MPILEUP_P4_somatic_Mutect2_all/0000.vcf.gz \
#     --dataset P4_TUMOR \
#     --section_id 1 \
#     --quality_filter baseQ0mapQ0 \
#     --tumor_purity /data/maiziezhou_lab/leiy4/CalicoST/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv \
#     --spatial_positions /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/Meta_Data/GSM4565823_P4_rep1_tissue_positions_list.csv \
#     --snv_vcf_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/BAM_filtered/baseQ0mapQ0/snv_vcf \
#     --output_prefix mutect2_overlap_analysis