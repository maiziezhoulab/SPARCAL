import os
import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple

class SVMThresholdAnalyzer:
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
        
        # Output directory for analysis
        self.output_dir = os.path.join(
            section_path, "metrics/SVMThresholdAnalysis",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_genotype(self, format_str: str, sample_str: str) -> str:
        """Extract GT field from FORMAT column"""
        try:
            gt_idx = format_str.split(':').index('GT')
            return sample_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None
            
    def extract_svm_prob(self, info_str: str) -> float:
        """Extract SVM_PROB from INFO field"""
        for field in info_str.split(';'):
            if field.startswith('SVM_PROB='):
                return float(field.split('=')[1])
        return None

    def load_variants(self) -> pd.DataFrame:
        """Load variants with original GT and SVM probability"""
        print(f"Loading variants from {self.svm_vcf}")
        variants = []
        
        with gzip.open(self.svm_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                
                gt = self.extract_genotype(fields[8], fields[9])
                svm_prob = self.extract_svm_prob(fields[7])
                
                if gt is not None and svm_prob is not None:
                    variants.append({
                        'CHROM': fields[0],
                        'POS': int(fields[1]),
                        'REF': fields[3],
                        'ALT': fields[4],
                        'GT': gt,
                        'SVM_PROB': svm_prob
                    })
                
                if len(variants) % 100000 == 0:
                    print(f"Processed {len(variants):,} variants...")
        
        df = pd.DataFrame(variants)
        print(f"Loaded {len(df):,} variants")
        return df

    def evaluate_threshold(self, df: pd.DataFrame, threshold: float) -> Dict[str, float]:
        """Evaluate performance metrics for a given threshold"""
        # Convert GT to binary (0 for 0/0, 1 for others)
        y_true = (df['GT'] != '0/0').astype(int)
        
        # Predict based on threshold (treating middle range as uncertain)
        y_pred = np.zeros_like(y_true)
        y_pred[df['SVM_PROB'] >= (1 - threshold)] = 1
        
        # Calculate metrics
        metrics = {
            'threshold': threshold,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        return metrics

    def analyze_thresholds(self, thresholds: np.ndarray = np.arange(0.05, 0.55, 0.05)):
        """Analyze performance across different thresholds"""
        print("\nStarting threshold analysis...")
        
        # Load variant data
        df = self.load_variants()
        
        # Evaluate each threshold
        results = []
        for threshold in thresholds:
            print(f"Evaluating threshold: {threshold:.2f}")
            metrics = self.evaluate_threshold(df, threshold)
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Plot results
        self.plot_threshold_analysis(results_df)
        
        # Save results
        results_df.to_csv(os.path.join(self.output_dir, 'threshold_analysis.csv'), index=False)
        
        return results_df

    def plot_threshold_analysis(self, results_df: pd.DataFrame):
        """Plot performance metrics across thresholds"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['blue', 'green', 'red', 'purple']
        
        for metric, color in zip(metrics, colors):
            plt.plot(results_df['threshold'], results_df[metric], 
                    marker='o', label=metric.capitalize(), color=color)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Threshold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add value annotations
        for metric in metrics:
            for idx, row in results_df.iterrows():
                plt.annotate(f'{row[metric]:.3f}', 
                           (row['threshold'], row[metric]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis.png'), dpi=300)
        plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze SVM probability thresholds")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    analyzer = SVMThresholdAnalyzer(args.section_id, args.quality_filter)
    results = analyzer.analyze_thresholds()
    
    # Print best threshold for each metric
    print("\nBest thresholds:")
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        best_idx = results[metric].idxmax()
        best_threshold = results.loc[best_idx, 'threshold']
        best_score = results.loc[best_idx, metric]
        print(f"{metric.capitalize()}: {best_score:.3f} (threshold = {best_threshold:.2f})")

if __name__ == "__main__":
    main()

# Usage
# python scripts/postprocess/run_svm_threshold_analysis.py --section_id 151507 --quality-filter baseQ0mapQ0