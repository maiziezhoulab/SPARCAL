class GermlineScorer:
    """
    Score likelihood that variant is germline
    Uses CNA-adjusted expectations
    """
    
    def score(self, features):
        """
        Compute germline score (0-1)
        Higher = more likely germline
        """
        score_components = {}
        
        # 1. VAF uniformity (adjusted for CNA)
        if 'cv_vaf' in features:
            # Low CV = uniform = germline-like
            score_components['uniformity'] = 1.0 / (1.0 + features['cv_vaf'])
        
        # 2. Likelihood match to expected germline VAF
        if 'mean_germline_likelihood' in features:
            score_components['likelihood'] = features['mean_germline_likelihood']
        
        # 3. Purity independence
        if 'vaf_purity_correlation' in features:
            # Low/negative correlation = germline-like
            corr = features['vaf_purity_correlation']
            score_components['purity_independence'] = 1.0 / (1.0 + max(0, corr))
        
        # 4. Global prevalence (across all spots)
        if 'n_spots_with_coverage' in features:
            prevalence = features['n_spots_with_coverage'] / features.get('total_spots', 100)
            score_components['prevalence'] = prevalence
        
        # 5. CNA-aware adjustment
        # Germline variants should match expected VAF for CNA state
        if 'expected_germline_vaf' in features and 'mean_vaf' in features:
            expected = features['expected_germline_vaf']
            actual = features['mean_vaf']
            deviation = abs(expected - actual)
            score_components['cna_match'] = np.exp(-5 * deviation)  # Steep penalty
        
        # Weighted combination
        weights = {
            'uniformity': 0.25,
            'likelihood': 0.25,
            'purity_independence': 0.20,
            'prevalence': 0.15,
            'cna_match': 0.15
        }
        
        germline_score = sum(
            score_components.get(k, 0) * weights[k] 
            for k in weights
        )
        
        return germline_score, score_components


class SomaticScorer:
    """
    Score likelihood that variant is somatic
    Uses CNA-adjusted expectations and spatial patterns
    """
    
    def score(self, features):
        """
        Compute somatic score (0-1)
        Higher = more likely somatic
        """
        score_components = {}
        
        # 1. Purity correlation
        if 'vaf_purity_correlation' in features:
            corr = max(0, features['vaf_purity_correlation'])  # Only positive
            score_components['purity_correlation'] = corr
        
        # 2. Somatic likelihood
        if 'mean_somatic_likelihood' in features:
            score_components['likelihood'] = features['mean_somatic_likelihood']
        
        # 3. Spatial clustering
        if 'spatial_autocorrelation' in features:
            # Positive Moran's I = clustered = somatic-like
            morans_i = max(0, features['spatial_autocorrelation'])
            score_components['spatial_clustering'] = morans_i
        
        # 4. Clone specificity (if clone info available)
        if 'clone_enrichment' in features:
            score_components['clone_specificity'] = features['clone_enrichment']
        
        # 5. CNA context prior
        # LOH+amplification = higher somatic prior
        cna_priors = {
            'LOH_amplified': 0.8,
            'LOH_neutral': 0.6,
            'LOH_deletion': 0.5,
            'balanced_diploid': 0.3,
            'balanced_amplified': 0.4,
            'imbalanced': 0.5
        }
        cna_category = features.get('cna_category', 'balanced_diploid')
        score_components['cna_prior'] = cna_priors[cna_category]
        
        # 6. VAF range (somatic varies with purity)
        if 'vaf_range' in features:
            # Moderate range expected for somatic
            vaf_range = features['vaf_range']
            # Optimal range around 0.2-0.4
            if 0.2 <= vaf_range <= 0.5:
                score_components['vaf_variability'] = 1.0
            elif vaf_range < 0.2:
                score_components['vaf_variability'] = vaf_range / 0.2
            else:
                score_components['vaf_variability'] = 0.5 / vaf_range
        
        # Weighted combination
        weights = {
            'purity_correlation': 0.25,
            'likelihood': 0.20,
            'spatial_clustering': 0.20,
            'clone_specificity': 0.15,
            'cna_prior': 0.10,
            'vaf_variability': 0.10
        }
        
        somatic_score = sum(
            score_components.get(k, 0) * weights[k] 
            for k in weights
        )
        
        return somatic_score, score_components