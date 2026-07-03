#!/usr/bin/env python3
"""
SPARCAL Modules - Core Classification Components
Feature Engineering, Likelihood Models, and Bayesian Classification
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, linregress
from collections import defaultdict
import logging
import warnings

# Suppress scipy warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SPARCAL')


# ============================================================================
# CLASS 1: CNA Profile Manager
# ============================================================================

class CNAProfile:
    """
    Manages copy number alteration (CNA) seg-level data from CalicoST
    """
    
    def __init__(self, seg_file):
        """
        Load CNA seg-level data
        
        Args:
            seg_file: Path to cnv_seglevel.tsv
        """
        logger.info(f"Loading CNA profile from {seg_file}")
        self.segments = pd.read_csv(seg_file, sep='\t')
        
        # Standardize chromosome names
        if 'CHR' in self.segments.columns:
            self.segments['CHR'] = self.segments['CHR'].astype(str)
            if not self.segments['CHR'].iloc[0].startswith('chr'):
                self.segments['CHR'] = 'chr' + self.segments['CHR'].astype(str)
        
        # Find clone columns (e.g., clone1 A, clone1 B)
        self.clone_cols = [col for col in self.segments.columns if 'clone' in col.lower()]
        logger.info(f"Found {len(self.clone_cols)} clone columns")
        
        # Use first clone pair by default (clone1)
        self.copy_a_col = [col for col in self.clone_cols if col.endswith(' A')][0]
        self.copy_b_col = [col for col in self.clone_cols if col.endswith(' B')][0]
        
        logger.info(f"Using columns: {self.copy_a_col}, {self.copy_b_col}")
        logger.info(f"Loaded {len(self.segments)} CNA segments")
    
    def get_state(self, chrom, pos):
        """
        Get copy number state at genomic position
        
        Args:
            chrom: Chromosome (e.g., 'chr17')
            pos: Position (int)
        
        Returns:
            (copy_A, copy_B): Tuple of copy numbers, or (None, None) if not found
        """
        # Ensure chromosome format matches
        if not chrom.startswith('chr'):
            chrom = 'chr' + chrom
        
        # Find overlapping segment
        mask = (self.segments['CHR'] == chrom) & \
               (self.segments['START'] <= pos) & \
               (self.segments['END'] >= pos)
        
        matching = self.segments[mask]
        
        if len(matching) == 0:
            logger.warning(f"No CNA segment found for {chrom}:{pos}")
            return None, None
        
        # Take first match (should only be one)
        seg = matching.iloc[0]
        copy_a = int(seg[self.copy_a_col])
        copy_b = int(seg[self.copy_b_col])
        
        return copy_a, copy_b
    
    def get_category(self, chrom, pos):
        """
        Categorize CNA state at position
        
        Returns:
            One of: 'LOH_amplified', 'LOH_neutral', 'LOH_deletion',
                    'balanced_diploid', 'balanced_amplified', 'imbalanced'
        """
        copy_a, copy_b = self.get_state(chrom, pos)
        
        if copy_a is None:
            return 'unknown'
        
        return self._categorize_cna(copy_a, copy_b)
    
    @staticmethod
    def _categorize_cna(copy_a, copy_b):
        """Categorize CNA state"""
        total = copy_a + copy_b
        
        if copy_a == 0 or copy_b == 0:
            if total > 2:
                return 'LOH_amplified'
            elif total == 2:
                return 'LOH_neutral'
            else:
                return 'LOH_deletion'
        elif copy_a == copy_b:
            if total == 2:
                return 'balanced_diploid'
            else:
                return 'balanced_amplified'
        else:
            return 'imbalanced'


# ============================================================================
# CLASS 2: Spot Metadata Manager
# ============================================================================

class SpotMetadata:
    """
    Manages spot-level metadata: purity, clone labels, spatial coordinates
    """
    
    def __init__(self, clone_file, spatial_dir):
        """
        Load spot metadata
        
        Args:
            clone_file: Path to clone_labels.tsv
            spatial_dir: Directory containing spatial coordinate files
        """
        logger.info(f"Loading spot metadata from {clone_file}")
        
        # Load clone labels and purity
        self.clone_data = pd.read_csv(clone_file, sep='\t')
        
        # Strip suffix from barcodes (e.g., AAACAAGTATCTCCCA-1_DCIS1 → AAACAAGTATCTCCCA-1)
        if 'BARCODES' in self.clone_data.columns:
            self.clone_data['barcode_clean'] = self.clone_data['BARCODES'].str.split('_').str[0]
        else:
            # Fallback: first column is barcodes
            self.clone_data['barcode_clean'] = self.clone_data.iloc[:, 0].str.split('_').str[0]
        
        # Create lookup dictionaries
        self.purity_dict = dict(zip(
            self.clone_data['barcode_clean'],
            self.clone_data['tumor_proportion']
        ))
        
        self.clone_dict = dict(zip(
            self.clone_data['barcode_clean'],
            self.clone_data['clone_label']
        ))
        
        logger.info(f"Loaded metadata for {len(self.purity_dict)} spots")
        
        # Load spatial coordinates (if available)
        self.coords_dict = {}
        try:
            # Try to find tissue_positions.csv or similar
            import os
            coord_file = os.path.join(spatial_dir, 'tissue_positions.csv')
            if not os.path.exists(coord_file):
                coord_file = os.path.join(spatial_dir, 'tissue_positions_list.csv')
            
            if os.path.exists(coord_file):
                coords = pd.read_csv(coord_file, header=None)
                # Format: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
                if len(coords.columns) >= 6:
                    self.coords_dict = dict(zip(
                        coords.iloc[:, 0],
                        zip(coords.iloc[:, 4], coords.iloc[:, 5])
                    ))
                    logger.info(f"Loaded spatial coordinates for {len(self.coords_dict)} spots")
        except Exception as e:
            logger.warning(f"Could not load spatial coordinates: {e}")
            logger.warning("Spatial statistics will be limited")
    
    def get_purity(self, barcode):
        """Get tumor purity for spot"""
        return self.purity_dict.get(barcode, None)
    
    def get_clone(self, barcode):
        """Get clone ID for spot"""
        return self.clone_dict.get(barcode, None)
    
    def get_coordinates(self, barcode):
        """Get (x, y) coordinates for spot"""
        return self.coords_dict.get(barcode, None)
    
    def has_coordinates(self):
        """Check if spatial coordinates are available"""
        return len(self.coords_dict) > 0


# ============================================================================
# CLASS 3: Feature Engineer
# ============================================================================

class SPARCALFeatureEngineer:
    """
    Comprehensive feature engineering for spatial variant calling
    Generates 40+ features across 6 categories
    """
    
    def __init__(self, config):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.min_coverage = config['coverage']['min_depth_per_spot']
        self.coverage_weight_min = config['coverage']['coverage_weight_min']
        self.coverage_weight_max = config['coverage']['coverage_weight_max']
        self.neighbor_threshold = config['spatial']['neighbor_distance_threshold']
        self.min_spots_morans = config['spatial']['morans_i_min_spots']
    
    def engineer_all_features(self, variant_data, cna_profile, spot_metadata):
        """
        Generate comprehensive feature set for a variant
        
        Args:
            variant_data: Dict with keys 'chrom', 'pos', 'ref', 'alt', 'spots'
                          spots is list of dicts: {'barcode', 'depth', 'vaf', 'ad_ref', 'ad_alt'}
            cna_profile: CNAProfile object
            spot_metadata: SpotMetadata object
        
        Returns:
            features (dict): Dictionary of 40+ features
        """
        features = {}
        chrom = variant_data['chrom']
        pos = variant_data['pos']
        spots = variant_data['spots']
        
        # ===== CATEGORY 1: CNA CONTEXT (8 features) =====
        copy_a, copy_b = cna_profile.get_state(chrom, pos)
        
        if copy_a is not None:
            features['copy_A'] = copy_a
            features['copy_B'] = copy_b
            features['total_copy'] = copy_a + copy_b
            features['copy_imbalance'] = abs(copy_a - copy_b) / (copy_a + copy_b + 1e-10)
            features['is_loh'] = int(copy_a == 0 or copy_b == 0)
            features['is_amplified'] = int(copy_a + copy_b > 2)
            features['max_copy'] = max(copy_a, copy_b)
            features['min_copy'] = min(copy_a, copy_b)
            features['cna_category'] = cna_profile._categorize_cna(copy_a, copy_b)
        else:
            # No CNA data - use diploid default
            features['copy_A'] = 1
            features['copy_B'] = 1
            features['total_copy'] = 2
            features['copy_imbalance'] = 0
            features['is_loh'] = 0
            features['is_amplified'] = 0
            features['max_copy'] = 1
            features['min_copy'] = 1
            features['cna_category'] = 'balanced_diploid'
        
        # Calculate expected germline VAF
        features['expected_germline_vaf'] = self._expected_germline_vaf(
            features['copy_A'], features['copy_B']
        )
        
        # ===== CATEGORY 2: COVERAGE STATISTICS (6 features) =====
        depths = [s['depth'] for s in spots]
        high_cov_spots = [s for s in spots if s['depth'] >= self.min_coverage]
        
        features['n_total_spots'] = len(spots)
        features['n_covered_spots'] = len(high_cov_spots)
        features['coverage_fraction'] = len(high_cov_spots) / max(len(spots), 1)
        features['mean_depth'] = np.mean(depths) if depths else 0
        features['median_depth'] = np.median(depths) if depths else 0
        features['cv_depth'] = np.std(depths) / (np.mean(depths) + 1e-10) if depths else 0
        
        # ===== CATEGORY 3: VAF STATISTICS (12 features) =====
        vafs = [s['vaf'] for s in high_cov_spots]
        
        if len(vafs) > 0:
            features['mean_vaf'] = np.mean(vafs)
            features['median_vaf'] = np.median(vafs)
            features['std_vaf'] = np.std(vafs)
            features['cv_vaf'] = np.std(vafs) / (np.mean(vafs) + 1e-10)
            features['min_vaf'] = np.min(vafs)
            features['max_vaf'] = np.max(vafs)
            features['vaf_range'] = np.max(vafs) - np.min(vafs)
            features['vaf_q25'] = np.percentile(vafs, 25)
            features['vaf_q75'] = np.percentile(vafs, 75)
            features['vaf_iqr'] = features['vaf_q75'] - features['vaf_q25']
            
            # Deviation from expected germline
            expected_germ = features['expected_germline_vaf']
            features['deviation_from_germline'] = abs(features['mean_vaf'] - expected_germ)
            
            # VAF entropy (uniformity measure)
            vaf_hist, _ = np.histogram(vafs, bins=10, range=(0, 1))
            vaf_probs = vaf_hist / (np.sum(vaf_hist) + 1e-10)
            vaf_probs = vaf_probs[vaf_probs > 0]
            features['vaf_entropy'] = -np.sum(vaf_probs * np.log2(vaf_probs + 1e-10))
        else:
            # No covered spots - set defaults
            for key in ['mean_vaf', 'median_vaf', 'std_vaf', 'cv_vaf', 
                       'min_vaf', 'max_vaf', 'vaf_range', 'vaf_q25', 'vaf_q75',
                       'vaf_iqr', 'deviation_from_germline', 'vaf_entropy']:
                features[key] = 0
        
        # ===== CATEGORY 4: SPATIAL PATTERNS (8 features) =====
        if spot_metadata.has_coordinates() and len(high_cov_spots) >= self.min_spots_morans:
            coords_with_vaf = []
            for spot in high_cov_spots:
                coord = spot_metadata.get_coordinates(spot['barcode'])
                if coord:
                    coords_with_vaf.append((coord[0], coord[1], spot['vaf']))
            
            if len(coords_with_vaf) >= self.min_spots_morans:
                positions = [(x, y) for x, y, _ in coords_with_vaf]
                vafs_spatial = [vaf for _, _, vaf in coords_with_vaf]
                
                # Moran's I
                features['morans_i'] = self._morans_i(vafs_spatial, positions)
                
                # Spatial clustering coefficient
                features['spatial_clustering'] = self._spatial_clustering_coefficient(
                    vafs_spatial, positions
                )
                
                # High-VAF spatial variance
                median_vaf = np.median(vafs_spatial)
                high_vaf_pos = [pos for pos, vaf in zip(positions, vafs_spatial) if vaf > median_vaf]
                if len(high_vaf_pos) > 2:
                    features['high_vaf_spatial_variance'] = np.var([p[0] for p in high_vaf_pos]) + \
                                                            np.var([p[1] for p in high_vaf_pos])
                else:
                    features['high_vaf_spatial_variance'] = 0
            else:
                features['morans_i'] = 0
                features['spatial_clustering'] = 0
                features['high_vaf_spatial_variance'] = 0
        else:
            features['morans_i'] = 0
            features['spatial_clustering'] = 0
            features['high_vaf_spatial_variance'] = 0
        
        # ===== CATEGORY 5: PURITY CORRELATION (5 features) =====
        purities = []
        vafs_with_purity = []
        for spot in high_cov_spots:
            purity = spot_metadata.get_purity(spot['barcode'])
            if purity is not None:
                purities.append(purity)
                vafs_with_purity.append(spot['vaf'])
        
        if len(purities) > 3:
            # Pearson correlation
            try:
                corr_p, pval_p = pearsonr(vafs_with_purity, purities)
                features['vaf_purity_pearson'] = corr_p
                features['vaf_purity_pearson_pval'] = pval_p
            except:
                features['vaf_purity_pearson'] = 0
                features['vaf_purity_pearson_pval'] = 1.0
            
            # Spearman correlation
            try:
                corr_s, pval_s = spearmanr(vafs_with_purity, purities)
                features['vaf_purity_spearman'] = corr_s
                features['vaf_purity_spearman_pval'] = pval_s
            except:
                features['vaf_purity_spearman'] = 0
                features['vaf_purity_spearman_pval'] = 1.0
            
            # Purity-adjusted VAF variance
            try:
                slope, intercept, r_val, p_val, std_err = linregress(purities, vafs_with_purity)
                predicted_vafs = [slope * p + intercept for p in purities]
                residuals = [actual - pred for actual, pred in zip(vafs_with_purity, predicted_vafs)]
                features['purity_adjusted_vaf_variance'] = np.var(residuals)
            except:
                features['purity_adjusted_vaf_variance'] = features['std_vaf']
        else:
            features['vaf_purity_pearson'] = 0
            features['vaf_purity_pearson_pval'] = 1.0
            features['vaf_purity_spearman'] = 0
            features['vaf_purity_spearman_pval'] = 1.0
            features['purity_adjusted_vaf_variance'] = features['std_vaf']
        
        # ===== CATEGORY 6: CLONE SPECIFICITY (4 features) =====
        clone_ids = []
        vafs_by_clone = defaultdict(list)
        
        for spot in high_cov_spots:
            clone = spot_metadata.get_clone(spot['barcode'])
            if clone is not None:
                clone_ids.append(clone)
                vafs_by_clone[clone].append(spot['vaf'])
        
        if len(clone_ids) > 0:
            # Tumor clone fraction (clone != 0)
            tumor_clones = [c for c in clone_ids if c != 0]
            features['tumor_clone_fraction'] = len(tumor_clones) / len(clone_ids)
            
            # Clone enrichment
            if 0 in vafs_by_clone and len(vafs_by_clone) > 1:
                normal_vaf = np.mean(vafs_by_clone[0])
                tumor_vafs = [np.mean(vafs_by_clone[c]) for c in vafs_by_clone if c != 0]
                max_tumor_vaf = max(tumor_vafs) if tumor_vafs else 0
                
                features['max_tumor_clone_vaf'] = max_tumor_vaf
                features['normal_clone_vaf'] = normal_vaf
                features['tumor_normal_vaf_ratio'] = max_tumor_vaf / (normal_vaf + 1e-10)
            else:
                features['max_tumor_clone_vaf'] = features['mean_vaf']
                features['normal_clone_vaf'] = 0
                features['tumor_normal_vaf_ratio'] = 1.0
        else:
            features['tumor_clone_fraction'] = 0.5
            features['max_tumor_clone_vaf'] = features['mean_vaf']
            features['normal_clone_vaf'] = 0
            features['tumor_normal_vaf_ratio'] = 1.0
        
        return features
    
    @staticmethod
    def _expected_germline_vaf(copy_a, copy_b):
        """Calculate expected VAF for heterozygous germline variant"""
        total = copy_a + copy_b
        if total == 0:
            return 0.5  # Default diploid
        return max(copy_a, copy_b) / total
    
    def _morans_i(self, values, positions):
        """
        Calculate Moran's I spatial autocorrelation statistic
        
        Args:
            values: List of VAF values
            positions: List of (x, y) tuples
        
        Returns:
            Moran's I (-1 to 1, where >0 indicates clustering)
        """
        n = len(values)
        if n < 3:
            return 0
        
        mean_val = np.mean(values)
        
        # Build spatial weights matrix (inverse distance)
        weights = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    weights[i, j] = 1.0 / (dist + 1)
        
        W = np.sum(weights)
        if W == 0:
            return 0
        
        numerator = 0
        denominator = 0
        
        for i in range(n):
            for j in range(n):
                numerator += weights[i, j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val)**2
        
        if denominator == 0:
            return 0
        
        morans_i = (n / W) * (numerator / denominator)
        return morans_i
    
    def _spatial_clustering_coefficient(self, values, positions):
        """
        Calculate fraction of high-value neighbors for high-value spots
        
        Returns:
            Clustering coefficient (0-1)
        """
        if len(values) < 3:
            return 0
        
        median_val = np.median(values)
        high_val_indices = [i for i, v in enumerate(values) if v > median_val]
        
        if len(high_val_indices) < 2:
            return 0
        
        clustering_scores = []
        for i in high_val_indices:
            # Find neighbors within threshold
            neighbors = []
            for j in range(len(positions)):
                if i != j:
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    if dist < self.neighbor_threshold:
                        neighbors.append(j)
            
            if len(neighbors) > 0:
                high_val_neighbors = sum(1 for j in neighbors if j in high_val_indices)
                clustering_scores.append(high_val_neighbors / len(neighbors))
        
        return np.mean(clustering_scores) if clustering_scores else 0


# ============================================================================
# CLASS 4: Germline Likelihood Model
# ============================================================================

class GermlineLikelihoodModel:
    """
    Calculate P(observations | variant is germline)
    """
    
    def __init__(self, config):
        """
        Initialize germline model
        
        Args:
            config: Configuration dictionary
        """
        self.weights = config['germline_weights']
        self.vaf_tolerance = config['likelihood']['vaf_tolerance']
        self.uniformity_penalty = config['likelihood']['vaf_uniformity_penalty']
        self.match_penalty = config['likelihood']['vaf_match_penalty']
    
    def calculate_likelihood(self, features):
        """
        Calculate germline likelihood score (0-1) and component breakdown
        
        Args:
            features: Feature dictionary from SPARCALFeatureEngineer
        
        Returns:
            (score, components): Overall score and dict of component scores
        """
        components = {}
        
        # ===== COMPONENT 1: VAF Uniformity =====
        cv_vaf = features.get('cv_vaf', 1.0)
        components['vaf_uniformity'] = np.exp(-self.uniformity_penalty * cv_vaf)
        
        # ===== COMPONENT 2: CNA-Adjusted VAF Match =====
        expected_germ = features.get('expected_germline_vaf', 0.5)
        actual_vaf = features.get('mean_vaf', 0.5)
        deviation = abs(actual_vaf - expected_germ)
        components['cna_adjusted_vaf_match'] = np.exp(-self.match_penalty * deviation)
        
        # ===== COMPONENT 3: Purity Independence =====
        purity_corr = abs(features.get('vaf_purity_pearson', 0))
        components['purity_independence'] = 1.0 / (1.0 + 5 * purity_corr)
        
        # ===== COMPONENT 4: Global Prevalence =====
        coverage_fraction = features.get('coverage_fraction', 0)
        components['global_prevalence'] = coverage_fraction
        
        # ===== COMPONENT 5: Spatial Randomness =====
        morans_i = features.get('morans_i', 0)
        if morans_i > 0:
            components['spatial_randomness'] = 1.0 / (1.0 + morans_i)
        else:
            components['spatial_randomness'] = 1.0
        
        # ===== WEIGHTED COMBINATION =====
        germline_score = sum(
            components.get(k, 0) * self.weights[k]
            for k in self.weights
        )
        
        return germline_score, components


# ============================================================================
# CLASS 5: Somatic Likelihood Model
# ============================================================================

class SomaticLikelihoodModel:
    """
    Calculate P(observations | variant is somatic)
    """
    
    def __init__(self, config):
        """
        Initialize somatic model
        
        Args:
            config: Configuration dictionary
        """
        self.weights = config['somatic_weights']
        self.cna_priors = config['cna_priors']
    
    def calculate_likelihood(self, features):
        """
        Calculate somatic likelihood score (0-1) and component breakdown
        
        Args:
            features: Feature dictionary from SPARCALFeatureEngineer
        
        Returns:
            (score, components): Overall score and dict of component scores
        """
        components = {}
        
        # ===== COMPONENT 1: Purity Correlation =====
        purity_corr = max(0, features.get('vaf_purity_pearson', 0))
        purity_pval = features.get('vaf_purity_pearson_pval', 1.0)
        components['purity_correlation'] = purity_corr * (1.0 - min(purity_pval, 1.0))
        
        # ===== COMPONENT 2: Spatial Clustering =====
        morans_i = max(0, features.get('morans_i', 0))
        spatial_coef = features.get('spatial_clustering', 0)
        components['spatial_clustering'] = (morans_i + spatial_coef) / 2
        
        # ===== COMPONENT 3: Clone Specificity =====
        tumor_fraction = features.get('tumor_clone_fraction', 0.5)
        tumor_normal_ratio = features.get('tumor_normal_vaf_ratio', 1.0)
        components['clone_specificity'] = (tumor_fraction + min(tumor_normal_ratio / 3, 1.0)) / 2
        
        # ===== COMPONENT 4: CNA Context Prior =====
        cna_category = features.get('cna_category', 'balanced_diploid')
        components['cna_context_prior'] = self.cna_priors.get(cna_category, 0.3) / 5.0  # Normalize
        
        # ===== COMPONENT 5: VAF Variability =====
        vaf_range = features.get('vaf_range', 0)
        if 0.15 <= vaf_range <= 0.50:
            components['vaf_variability'] = 1.0
        elif vaf_range < 0.15:
            components['vaf_variability'] = vaf_range / 0.15
        else:
            components['vaf_variability'] = max(0, 1.0 - (vaf_range - 0.50) / 0.50)
        
        # ===== COMPONENT 6: Germline Mismatch =====
        deviation = features.get('deviation_from_germline', 0)
        components['germline_mismatch'] = min(deviation / 0.3, 1.0)
        
        # ===== WEIGHTED COMBINATION =====
        somatic_score = sum(
            components.get(k, 0) * self.weights[k]
            for k in self.weights
        )
        
        return somatic_score, components


# ============================================================================
# CLASS 6: Bayesian Classifier
# ============================================================================

class BayesianClassifier:
    """
    Final Bayesian classification integrating germline and somatic likelihoods
    """
    
    def __init__(self, config):
        """
        Initialize classifier
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.prior_somatic = config['classification']['prior_somatic']
        self.max_adjusted_prior = config['classification']['max_adjusted_prior']
        self.prob_somatic_threshold = config['classification']['prob_somatic_threshold']
        self.prob_germline_threshold = config['classification']['prob_germline_threshold']
        self.lr_somatic_threshold = config['classification']['likelihood_ratio_somatic']
        self.lr_germline_threshold = config['classification']['likelihood_ratio_germline']
        
        self.germline_model = GermlineLikelihoodModel(config)
        self.somatic_model = SomaticLikelihoodModel(config)
        self.cna_priors = config['cna_priors']
    
    def classify(self, features, return_details=False):
        """
        Classify variant using Bayesian framework
        
        Args:
            features: Feature dictionary
            return_details: If True, return detailed component breakdowns
        
        Returns:
            result (dict): Classification results with keys:
                - classification: 'somatic', 'germline', or 'uncertain'
                - prob_somatic: P(somatic | observations)
                - prob_germline: P(germline | observations)
                - confidence: max(prob_somatic, prob_germline)
                - likelihood_ratio: L_somatic / L_germline
                - adjusted_prior_somatic: CNA-adjusted prior
        """
        # Calculate likelihoods
        L_germline, germ_components = self.germline_model.calculate_likelihood(features)
        L_somatic, som_components = self.somatic_model.calculate_likelihood(features)
        
        # Adjust prior based on CNA context
        adjusted_prior_somatic = self._adjust_prior(features)
        adjusted_prior_germline = 1 - adjusted_prior_somatic
        
        # Bayes' theorem
        posterior_somatic_unnorm = L_somatic * adjusted_prior_somatic
        posterior_germline_unnorm = L_germline * adjusted_prior_germline
        
        # Normalize
        total = posterior_somatic_unnorm + posterior_germline_unnorm + 1e-10
        prob_somatic = posterior_somatic_unnorm / total
        prob_germline = posterior_germline_unnorm / total
        
        # Likelihood ratio
        likelihood_ratio = L_somatic / (L_germline + 1e-10)
        
        # Classification decision
        if prob_somatic > self.prob_somatic_threshold:
            classification = 'somatic'
            confidence = prob_somatic
        elif prob_germline > self.prob_germline_threshold:
            classification = 'germline'
            confidence = prob_germline
        elif likelihood_ratio > self.lr_somatic_threshold:
            classification = 'somatic'
            confidence = min(prob_somatic + 0.1, 0.9)
        elif likelihood_ratio < self.lr_germline_threshold:
            classification = 'germline'
            confidence = min(prob_germline + 0.1, 0.9)
        else:
            classification = 'uncertain'
            confidence = max(prob_somatic, prob_germline)
        
        result = {
            'classification': classification,
            'prob_somatic': prob_somatic,
            'prob_germline': prob_germline,
            'confidence': confidence,
            'likelihood_ratio': likelihood_ratio,
            'adjusted_prior_somatic': adjusted_prior_somatic,
            'L_somatic': L_somatic,
            'L_germline': L_germline
        }
        
        if return_details:
            result['germline_components'] = germ_components
            result['somatic_components'] = som_components
        
        return result
    
    def _adjust_prior(self, features):
        """
        Adjust prior probability based on CNA context
        
        Args:
            features: Feature dictionary
        
        Returns:
            Adjusted somatic prior probability
        """
        base_prior = self.prior_somatic
        cna_category = features.get('cna_category', 'balanced_diploid')
        multiplier = self.cna_priors.get(cna_category, 1.0)
        
        adjusted_prior = min(base_prior * multiplier, self.max_adjusted_prior)
        return adjusted_prior