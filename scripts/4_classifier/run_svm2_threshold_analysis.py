import os
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Set, Tuple, Optional
import pickle

class SVMFeatureExtractor:
    def extract_features(self, vcf_line: str) -> Optional[Dict]:
        try:
            fields = vcf_line.strip().split('\t')
            info = fields[7]
            format_str = fields[8]
            sample_str = fields[9]
            
            feature_dict = {}
            
            # Extract INFO fields
            for field in info.split(';'):
                if '=' in field:
                    key, value = field.split('=', 1)
                    if key in ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F']:
                        try:
                            feature_dict[key] = float(value.split(',')[0])
                        except ValueError:
                            feature_dict[key] = np.nan
            
            # Extract FORMAT fields
            format_fields = format_str.split(':')
            sample_values = sample_str.split(':')
            
            for field in ['BAF', 'GQ', 'PL']:
                if field in format_fields:
                    idx = format_fields.index(field)
                    try:
                        value = sample_values[idx]
                        if field == 'PL':
                            value = value.split(',')[0]
                        feature_dict[field] = float(value)
                    except (ValueError, IndexError):
                        feature_dict[field] = np.nan
            
            return feature_dict
            
        except Exception:
            return None

class SVM2ThresholdAnalyzer:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.feature_extractor = SVMFeatureExtractor()
        self.setup_paths()

    def setup_paths(self):
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        self.genotype_classify_dir = os.path.join(
            section_path, "output_VCFs/genotype_classify",
            self.quality_filter
        )
        
        self.model_dir = os.path.join(
            section_path, "output_VCFs/SVM2Model",
            self.quality_filter
        )
        
        self.output_dir = os.path.join(
            section_path, "metrics/SVM2Model",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_svm2_model(self):
        model_path = os.path.join(self.model_dir, 'svm2_model.pkl')
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
        
    def load_all_transition_variants(self) -> Dict[str, Tuple[List[str], int]]:
        """Load variants for all four transition types"""
        transitions = {
            '1/1->1/1': ('1_1_to_1_1', 0),
            '1/1->0/1': ('1_1_to_0_1', 1),
            '0/1->1/1': ('0_1_to_1_1', 2),
            '0/1->0/1': ('0_1_to_0_1', 3)
        }
        
        all_variants = {}
        
        for trans_name, (pattern, label) in transitions.items():
            variants = []
            vcf_count = variant_count = 0
            
            print(f"\nLoading {trans_name} transitions...")
            for base in ['A', 'C', 'G', 'T']:
                for alt in ['A', 'C', 'G', 'T']:
                    if base != alt:
                        vcf_path = os.path.join(
                            self.genotype_classify_dir,
                            f"{pattern}_{base}_{alt}variants.vcf.gz"
                        )
                        if os.path.exists(vcf_path):
                            vcf_count += 1
                            with gzip.open(vcf_path, 'rt') as f:
                                for line in f:
                                    if line.startswith('#'):
                                        continue
                                    variant_count += 1
                                    variants.append(line)
            
            if variants:
                all_variants[trans_name] = (variants, label)
                print(f"Found {vcf_count} VCFs with {variant_count:,} variants")
        
        return all_variants

    def extract_features_and_predict(self, variants: List[str], model_data: Dict) -> np.ndarray:
        svm_model = model_data['svm_model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_columns']
        
        print("\nExtracting features...")
        features = []
        for variant in variants:
            feat_dict = self.feature_extractor.extract_features(variant)
            if feat_dict is not None:
                features.append(feat_dict)
        
        features_df = pd.DataFrame(features)
        features_df = features_df.fillna(features_df.mean())
        
        X = features_df[feature_cols]
        X_scaled = scaler.transform(X)
        
        print("Getting SVM2 predictions...")
        probabilities = svm_model.predict_proba(X_scaled)[:, 1]
        
        return probabilities

    def plot_probability_distribution(self, all_probabilities: Dict[str, np.ndarray]):
        """Plot stacked distribution of SVM2 probabilities for all transition types"""
        plt.figure(figsize=(15, 8))
        
        bins = np.arange(0, 1.05, 0.05)
        colors = ['blue', 'red', 'green', 'purple']
        total_variants = sum(len(probs) for probs in all_probabilities.values())
        
        stats_text = f"Total variants: {total_variants:,}\n"
        
        # Calculate histogram data for each transition type
        hist_data = []
        for trans_name, probs in all_probabilities.items():
            counts, _ = np.histogram(probs, bins=bins)
            hist_data.append(counts)
            stats_text += f"{trans_name}: {len(probs):,}\n"
        
        # Create stacked histogram
        plt.hist([[] for _ in range(len(hist_data))], bins=bins, stacked=True, 
                label=list(all_probabilities.keys()), color=colors)
        
        bottom = np.zeros(len(bins)-1)
        for counts, color in zip(hist_data, colors):
            plt.bar(bins[:-1], counts, bottom=bottom, width=np.diff(bins),
                   align='edge', color=color, alpha=0.8)
        
        plt.xlabel('SVM2 Probability of Being Heterozygous')
        plt.ylabel('Count')
        plt.title('Distribution of SVM2 Probabilities by Genotype Transition')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'svm2_all_transitions_distribution.png'))
        plt.close()

    def evaluate_performance(self):
        print("Loading SVM2 model...")
        model_data = self.load_svm2_model()
        
        # Load variants for all transition types
        all_variants = self.load_all_transition_variants()
        
        # Get predictions for each transition type
        all_probabilities = {}
        
        for trans_name, (variants, _) in all_variants.items():
            print(f"\nProcessing {trans_name} transitions...")
            probabilities = self.extract_features_and_predict(variants, model_data)
            all_probabilities[trans_name] = probabilities
        
        # Plot combined probability distribution
        self.plot_probability_distribution(all_probabilities)
        
        # Save detailed statistics
        with open(os.path.join(self.output_dir, 'transition_analysis.txt'), 'w') as f:
            f.write("SVM2 Transition Analysis Results\n")
            f.write("=" * 50 + "\n\n")
            
            total_variants = sum(len(probs) for probs in all_probabilities.values())
            f.write(f"Total variants analyzed: {total_variants:,}\n\n")
            
            for trans_name, probs in all_probabilities.items():
                f.write(f"\n{trans_name} Statistics:\n")
                f.write(f"Count: {len(probs):,}\n")
                f.write(f"Mean probability: {np.mean(probs):.3f}\n")
                f.write(f"Median probability: {np.median(probs):.3f}\n")
                f.write(f"Std deviation: {np.std(probs):.3f}\n")
                
                # Calculate probability ranges
                ranges = [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 1.0  )]
                for low, high in ranges:
                    count = np.sum((probs >= low) & (probs < high))
                    f.write(f"Prob {low:.1f}-{high:.1f}: {count:,} ({count/len(probs)*100:.1f}%)\n")
                f.write("-" * 30 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze SVM2 probabilities for all transition types")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    analyzer = SVM2ThresholdAnalyzer(args.section_id, args.quality_filter)
    analyzer.evaluate_performance()
    
    print(f"\nAnalysis complete! Results saved to: {analyzer.output_dir}")

if __name__ == "__main__":
    main()