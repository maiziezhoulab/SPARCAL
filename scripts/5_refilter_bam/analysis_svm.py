import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from typing import Dict, List, Optional
import argparse
from pathlib import Path

class SVMPerformanceAnalyzer:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()
        
    def setup_paths(self):
        """Setup paths for input and output files"""
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        # Input VCF with SVM predictions
        self.svm_vcf = os.path.join(
            section_path, "output_VCFs/SVMModel", 
            self.quality_filter, "results/svm_predictions.vcf.gz"
        )
        
        # Output directory for plots
        self.output_dir = os.path.join(
            section_path, "metrics/SVMModel",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
    def extract_variant_info(self, info_str: str, field_name: str) -> Optional[str]:
        """Extract field from VCF INFO column"""
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None
        
    def extract_format_field(self, format_str: str, sample_str: str, field_name: str) -> Optional[str]:
        """Extract field from VCF FORMAT column"""
        try:
            format_fields = format_str.split(':')
            if field_name not in format_fields:
                return None
            
            field_idx = format_fields.index(field_name)
            value_fields = sample_str.split(':')
            
            if field_idx >= len(value_fields):
                return None
                
            return value_fields[field_idx]
        except (ValueError, IndexError):
            return None

    def load_variant_data(self) -> pd.DataFrame:
        """Load variant data from VCF file"""
        print(f"Loading variants from {self.svm_vcf}")
        variant_data = []
        
        with gzip.open(self.svm_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                info = fields[7]
                format_str = fields[8]
                sample_str = fields[9]
                
                # Extract required fields
                baf = float(self.extract_format_field(format_str, sample_str, "BAF"))
                depth = min(int(self.extract_variant_info(info, "DP")), 200)
                svm_pred = int(self.extract_variant_info(info, "SVM_PRED"))
                svm_prob = float(self.extract_variant_info(info, "SVM_PROB"))
                
                variant_data.append({
                    'CHROM': fields[0],
                    'POS': int(fields[1]),
                    'REF': fields[3],
                    'ALT': fields[4],
                    'BAF': baf,
                    'Depth': depth,
                    'SVM_Prediction': svm_pred,
                    'SVM_Probability': svm_prob
                })
                
                if len(variant_data) % 100000 == 0:
                    print(f"Processed {len(variant_data):,} variants...")
        
        df = pd.DataFrame(variant_data)
        print(f"Loaded {len(df):,} variants")
        return df

    def plot_probability_distribution(self, df: pd.DataFrame):
        """Plot distribution of SVM probabilities with actual counts"""
        plt.figure(figsize=(12, 6))
        
        # Create histogram with actual counts
        bins = np.arange(0, 1.05, 0.05)
        counts, bins, _ = plt.hist(df['SVM_Probability'], bins=bins, density=False, alpha=0.7)
        
        # Add labels and title
        plt.xlabel('SVM Probability')
        plt.ylabel('Count')
        plt.title('Distribution of SVM Probabilities')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add vertical line at 0.5
        plt.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
        
        # Add summary statistics
        stats_text = f"Total variants: {len(df):,}\n"
        stats_text += f"Predicted true variants: {(df['SVM_Prediction'] == 1).sum():,}\n"
        stats_text += f"Predicted errors: {(df['SVM_Prediction'] == 0).sum():,}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'svm_probability_distribution.png'), dpi=300)
        plt.close()

    def plot_feature_space(self, df: pd.DataFrame, sample_size=100000):
        """Plot BAF vs Read Depth with three probability ranges using random sampling"""
        plt.figure(figsize=(12, 8))
        
        # Random sampling if dataset is larger than sample_size
        if len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        # Create masks for different probability ranges
        low_prob = df['SVM_Probability'] < 0.1
        mid_prob = (df['SVM_Probability'] >= 0.1) & (df['SVM_Probability'] <= 0.9)
        high_prob = df['SVM_Probability'] > 0.9
        
        # Plot each range with different colors
        # plt.scatter(df.loc[low_prob, 'BAF'], df.loc[low_prob, 'Depth'], 
        #         color='red', alpha=0.5, s=20, label=f'Prob < 0.1 (n={sum(low_prob):,})')
        plt.scatter(df.loc[mid_prob, 'BAF'], df.loc[mid_prob, 'Depth'], 
                color='green', alpha=0.5, s=20, label=f'0.1 ≤ Prob ≤ 0.9 (n={sum(mid_prob):,})')
        # plt.scatter(df.loc[high_prob, 'BAF'], df.loc[high_prob, 'Depth'], 
        #         color='blue', alpha=0.5, s=20, label=f'Prob > 0.9 (n={sum(high_prob):,})')
        
        # Add labels and title
        plt.xlabel('B-Allele Frequency (BAF)')
        plt.ylabel('Read Depth (capped at 200)')
        plt.title(f'BAF vs Read Depth by SVM Probability Ranges\n(Random sample of {sample_size:,} variants)')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add legend with better positioning
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add total counts in the original dataset
        total_counts = len(df)
        stats_text = f"Total variants in sample: {total_counts:,}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot with extra space for legend
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'baf_depth_svm_scatter.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_performance(self):
        """Run all performance analyses"""
        print("\nStarting SVM performance analysis...")
        
        # Load data
        df = self.load_variant_data()
        
        print("\nGenerating probability distribution plot...")
        self.plot_probability_distribution(df)
        
        print("Generating feature space plot...")
        self.plot_feature_space(df)
        
        # Save summary statistics
        summary = {
            'total_variants': len(df),
            'predicted_true': (df['SVM_Prediction'] == 1).sum(),
            'predicted_error': (df['SVM_Prediction'] == 0).sum(),
            'mean_probability': df['SVM_Probability'].mean(),
            'median_probability': df['SVM_Probability'].median()
        }
        
        summary_file = os.path.join(self.output_dir, 'svm_performance_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("SVM Performance Summary\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key.replace('_', ' ').title()}: {value:,}\n")
        
        print(f"\nAnalysis complete. Results saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze SVM model performance")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    analyzer = SVMPerformanceAnalyzer(args.section_id, args.quality_filter)
    analyzer.analyze_performance()

if __name__ == "__main__":
    main()

# Usage
# python scripts/postprocess/analysis_svm.py --section_id 151507 --quality-filter baseQ0mapQ0