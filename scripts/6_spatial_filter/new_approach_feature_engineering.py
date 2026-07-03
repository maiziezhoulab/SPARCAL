class SPARCALFeatureEngineer:
    """
    Comprehensive feature engineering for spatial variant calling
    """
    
    def engineer_all_features(self, variant, cna_profile, min_coverage=5):
        """
        Generate all features for a variant
        
        Returns:
            features (dict): 40+ features across 6 categories
        """
        features = {}
        
        # ===== CATEGORY 1: CNA CONTEXT (8 features) =====
        cna_state = cna_profile.get_state(variant.chrom, variant.pos)
        copy_A, copy_B = cna_state
        
        features['copy_A'] = copy_A
        features['copy_B'] = copy_B
        features['total_copy'] = copy_A + copy_B
        features['copy_imbalance'] = abs(copy_A - copy_B) / (copy_A + copy_B + 1e-10)
        features['is_loh'] = int(copy_A == 0 or copy_B == 0)
        features['is_amplified'] = int(copy_A + copy_B > 2)
        features['max_copy'] = max(copy_A, copy_B)
        features['min_copy'] = min(copy_A, copy_B)
        
        # CNA category (for priors)
        features['cna_category'] = self._categorize_cna(copy_A, copy_B)
        
        # ===== CATEGORY 2: COVERAGE STATISTICS (6 features) =====
        depths = [spot.depth for spot in variant.spots]
        high_cov_spots = [s for s in variant.spots if s.depth >= min_coverage]
        
        features['n_total_spots'] = len(variant.spots)
        features['n_covered_spots'] = len(high_cov_spots)
        features['coverage_fraction'] = len(high_cov_spots) / len(variant.spots)
        features['mean_depth'] = np.mean(depths)
        features['median_depth'] = np.median(depths)
        features['cv_depth'] = np.std(depths) / (np.mean(depths) + 1e-10)
        
        # ===== CATEGORY 3: VAF STATISTICS (12 features) =====
        vafs = [s.vaf for s in high_cov_spots]
        
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
            
            # Deviation from expected germline VAF
            expected_germ_vaf = self._expected_germline_vaf(copy_A, copy_B)
            features['expected_germline_vaf'] = expected_germ_vaf
            features['deviation_from_germline'] = abs(features['mean_vaf'] - expected_germ_vaf)
        
        # ===== CATEGORY 4: SPATIAL PATTERNS (8 features) =====
        positions = [(s.x, s.y) for s in high_cov_spots]
        
        if len(positions) > 5:
            # Spatial autocorrelation (Moran's I)
            features['morans_i'] = self._morans_i(vafs, positions)
            
            # Spatial clustering coefficient
            features['spatial_clustering'] = self._spatial_clustering_coefficient(vafs, positions)
            
            # Spatial variance (how spread out are high-VAF spots?)
            high_vaf_positions = [pos for pos, vaf in zip(positions, vafs) if vaf > np.median(vafs)]
            if len(high_vaf_positions) > 2:
                features['high_vaf_spatial_variance'] = np.var([p[0] for p in high_vaf_positions]) + \
                                                        np.var([p[1] for p in high_vaf_positions])
        
        # ===== CATEGORY 5: PURITY CORRELATION (5 features) =====
        purities = [s.purity for s in high_cov_spots if hasattr(s, 'purity')]
        
        if len(purities) > 3 and len(vafs) == len(purities):
            from scipy.stats import pearsonr, spearmanr
            
            # Pearson correlation (linear relationship)
            corr_pearson, pval_pearson = pearsonr(vafs, purities)
            features['vaf_purity_pearson'] = corr_pearson
            features['vaf_purity_pearson_pval'] = pval_pearson
            
            # Spearman correlation (monotonic relationship)
            corr_spearman, pval_spearman = spearmanr(vafs, purities)
            features['vaf_purity_spearman'] = corr_spearman
            features['vaf_purity_spearman_pval'] = pval_spearman
            
            # Purity-adjusted VAF variance
            # Residual variance after accounting for purity
            from scipy.stats import linregress
            slope, intercept, r_value, p_value, std_err = linregress(purities, vafs)
            predicted_vafs = [slope * p + intercept for p in purities]
            residuals = [actual - pred for actual, pred in zip(vafs, predicted_vafs)]
            features['purity_adjusted_vaf_variance'] = np.var(residuals)
        
        # ===== CATEGORY 6: CLONE SPECIFICITY (4 features) =====
        clone_ids = [s.clone_id for s in high_cov_spots if hasattr(s, 'clone_id')]
        
        if len(clone_ids) > 0:
            # Fraction of variants in tumor clones (not clone 0)
            tumor_fraction = sum(c != 0 for c in clone_ids) / len(clone_ids)
            features['tumor_clone_fraction'] = tumor_fraction
            
            # Clone enrichment (max VAF in any tumor clone vs clone 0)
            vafs_by_clone = {}
            for spot in high_cov_spots:
                if hasattr(spot, 'clone_id'):
                    clone = spot.clone_id
                    if clone not in vafs_by_clone:
                        vafs_by_clone[clone] = []
                    vafs_by_clone[clone].append(spot.vaf)
            
            if 0 in vafs_by_clone and len(vafs_by_clone) > 1:
                normal_vaf = np.mean(vafs_by_clone[0])
                tumor_vafs = [np.mean(vafs_by_clone[c]) for c in vafs_by_clone if c != 0]
                max_tumor_vaf = max(tumor_vafs) if tumor_vafs else 0
                
                features['max_tumor_clone_vaf'] = max_tumor_vaf
                features['normal_clone_vaf'] = normal_vaf
                features['tumor_normal_vaf_ratio'] = max_tumor_vaf / (normal_vaf + 1e-10)
        
        return features
    
    def _categorize_cna(self, copy_A, copy_B):
        """Categorize CNA state"""
        if copy_A == 0 or copy_B == 0:
            if copy_A + copy_B > 2:
                return 'LOH_amplified'
            elif copy_A + copy_B == 2:
                return 'LOH_neutral'
            else:
                return 'LOH_deletion'
        elif copy_A == copy_B:
            if copy_A + copy_B == 2:
                return 'balanced_diploid'
            else:
                return 'balanced_amplified'
        else:
            return 'imbalanced'
    
    def _expected_germline_vaf(self, copy_A, copy_B):
        """Expected VAF for heterozygous germline variant"""
        total = copy_A + copy_B
        if total == 0:
            return None
        # Assume on more amplified allele
        return max(copy_A, copy_B) / total
    
    def _morans_i(self, values, positions):
        """Moran's I spatial autocorrelation"""
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
                    weights[i,j] = 1.0 / (dist + 1)
        
        W = np.sum(weights)
        
        numerator = 0
        denominator = 0
        for i in range(n):
            for j in range(n):
                numerator += weights[i,j] * (values[i] - mean_val) * (values[j] - mean_val)
            denominator += (values[i] - mean_val)**2
        
        if denominator == 0 or W == 0:
            return 0
        
        I = (n / W) * (numerator / denominator)
        return I
    
    def _spatial_clustering_coefficient(self, values, positions, threshold_distance=50):
        """
        Fraction of high-value neighbors for high-value spots
        """
        median_val = np.median(values)
        high_val_indices = [i for i, v in enumerate(values) if v > median_val]
        
        if len(high_val_indices) < 2:
            return 0
        
        clustering_scores = []
        for i in high_val_indices:
            # Find neighbors within threshold distance
            neighbors = []
            for j in range(len(positions)):
                if i != j:
                    dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                                 (positions[i][1] - positions[j][1])**2)
                    if dist < threshold_distance:
                        neighbors.append(j)
            
            if len(neighbors) > 0:
                # Fraction of neighbors that are also high-value
                high_val_neighbors = sum(1 for j in neighbors if j in high_val_indices)
                clustering_scores.append(high_val_neighbors / len(neighbors))
        
        return np.mean(clustering_scores) if clustering_scores else 0