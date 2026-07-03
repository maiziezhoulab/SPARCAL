import os
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import List, Dict, Set
import pickle
import tqdm

class SVMFeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_info_field(self, info_str: str, field: str) -> float:
        """Extract numerical value from INFO field"""
        for item in info_str.split(';'):
            if item.startswith(f"{field}="):
                try:
                    return float(item.split('=')[1].split(',')[0])  # Take first value if comma-separated
                except (ValueError, IndexError):
                    return np.nan
        return np.nan

    def extract_format_field(self, format_str: str, sample_str: str, field: str) -> float:
        """Extract numerical value from FORMAT field"""
        try:
            idx = format_str.split(':').index(field)
            value = sample_str.split(':')[idx]
            # print(f"idx and value: {idx}, {value}")
            if field == 'PL':
                return float(value.split(',')[0])
            if field == 'GT':
                return value
            return float(value)
        except (ValueError, IndexError):
            return np.nan

    def extract_i16_values(self, info_str: str) -> List[float]:
        """Extract I16 values from INFO field"""
        for item in info_str.split(';'):
            if item.startswith('I16='):
                try:
                    values = [float(x) for x in item.split('=')[1].split(',')]
                    if len(values) == 16:
                        return values
                except (ValueError, IndexError):
                    pass
        return [np.nan] * 16
    
    def extract_features_single_variant(self, fields) -> Dict:
        """Extract features from a single VCF line fields"""
        try:
            # Define fixed columns for features
            numeric_columns = [
                'POS', 'DP', 'BAF', 'GQ', 'PL', 'QS', 'VDB', 'RPB', 'BQB', 
                'MQSB', 'SGB', 'MQB', 'MQ0F'
            ] + [f'I16_{i}' for i in range(16)]
            
            feature_dict = {col: 0.0 for col in numeric_columns}
            feature_dict['POS'] = int(fields[1])
            
            # Extract INFO fields
            for field in fields[7].split(';'):
                if '=' in field:
                    key, value = field.split('=', 1)
                    if key in numeric_columns:
                        try:
                            feature_dict[key] = float(value.split(',')[0])
                        except ValueError:
                            continue
            
            # Extract FORMAT fields
            format_indices = {field: idx for idx, field in enumerate(fields[8].split(':'))}
            sample_values = fields[9].split(':')
            
            for field in ['BAF', 'GQ', 'PL']:
                if field in format_indices:
                    idx = format_indices[field]
                    if idx < len(sample_values):
                        try:
                            value = sample_values[idx]
                            if field == 'PL':
                                value = value.split(',')[0]
                            feature_dict[field] = float(value)
                        except ValueError:
                            continue
            
            # Extract I16 values
            i16_values = self.extract_i16_values(fields[7])
            for i, value in enumerate(i16_values):
                if not np.isnan(value):
                    feature_dict[f'I16_{i}'] = value
            
            return feature_dict
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None
    def extract_features(self, vcf_path: str) -> pd.DataFrame:
        """Extract features from VCF file"""
        features = []
        
        # Define fixed columns
        numeric_columns = [
            'POS', 'DP', 'BAF', 'GQ', 'PL', 'QS', 'VDB', 'RPB', 'BQB', 
            'MQSB', 'SGB', 'MQB', 'MQ0F'
        ] + [f'I16_{i}' for i in range(16)]
        
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                    
                try:
                    feature_dict = {col: 0.0 for col in numeric_columns}
                    feature_dict['POS'] = int(fields[1])
                    
                    # Extract INFO fields
                    for field in fields[7].split(';'):
                        if '=' in field:
                            key, value = field.split('=', 1)
                            if key in numeric_columns:
                                try:
                                    feature_dict[key] = float(value.split(',')[0])
                                except ValueError:
                                    continue
                    
                    # Extract FORMAT fields
                    format_indices = {field: idx for idx, field in enumerate(fields[8].split(':'))}
                    sample_values = fields[9].split(':')
                    
                    for field in ['BAF', 'GQ', 'PL']:
                        if field in format_indices:
                            idx = format_indices[field]
                            if idx < len(sample_values):
                                try:
                                    value = sample_values[idx]
                                    if field == 'PL':
                                        value = value.split(',')[0]
                                    feature_dict[field] = float(value)
                                except ValueError:
                                    continue
                    
                    # Extract I16 values
                    i16_values = self.extract_i16_values(fields[7])
                    for i, value in enumerate(i16_values):
                        if not np.isnan(value):
                            feature_dict[f'I16_{i}'] = value
                    
                    features.append(feature_dict)
                    
                except Exception:
                    continue
                    
        if not features:
            raise ValueError(f"No valid features extracted from {vcf_path}")
            
        return pd.DataFrame(features)

    def extract_i16_values(self, info_str: str) -> List[float]:
        """Extract I16 values from INFO field"""
        for item in info_str.split(';'):
            if item.startswith('I16='):
                try:
                    values = [float(x) for x in item.split('=')[1].split(',')]
                    if len(values) == 16:
                        return values
                except (ValueError, IndexError):
                    pass
        return [np.nan] * 16
    
class SVMEvaluator:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()
        self.feature_extractor = SVMFeatureExtractor()

    def setup_paths(self):
        """Setup paths for input and output files"""
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        # Input files
        self.mpileup_vcf = os.path.join(
            section_path, "output_VCFs/mpileup_multi_bam",
            self.quality_filter, "merged_sorted_gt.vcf.gz"
        )
        self.beagle_vcf = os.path.join(
            section_path, "output_VCFs/beagle",
            self.quality_filter, "all_filtered.vcf.gz"
        )
        
        # SVM model path
        self.model_path = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "model/svm_model.pkl"
        )

        # Output directory
        self.output_dir = os.path.join(
            section_path, "metrics/SVMComparison",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_beagle_variants(self) -> Dict:
        """Load variants from all Beagle chromosome VCFs"""
        print("Loading Beagle variants...")
        beagle_vars = {}
        
        # Get all chromosome VCFs
        beagle_base = os.path.dirname(self.beagle_vcf)
        for chrom in range(1, 23):
            chrom_vcf = os.path.join(beagle_base, f"chr{chrom}.vcf.gz")
            print(f"Processing {chrom_vcf}")
            
            with gzip.open(chrom_vcf, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    fields = line.strip().split('\t')
                    pos = f"{fields[0]}_{fields[1]}"
                    
                    gt = self.feature_extractor.extract_format_field(fields[8], fields[9], 'GT')
                    if gt is not None:
                        beagle_vars[pos] = {
                            'ref': fields[3],
                            'alt': fields[4],
                            'gt': gt
                        }
                    
                    if len(beagle_vars) % 100000 == 0:
                        print(f"Loaded {len(beagle_vars):,} Beagle variants")
        
        print(f"Total Beagle variants loaded: {len(beagle_vars):,}")
        return beagle_vars

    def load_mpileup_variants(self, positions: Set[str]) -> Dict:
        """Load only specified positions from mpileup VCF"""
        print("Loading matching mpileup variants...")
        mpileup_vars = {}
        
        with gzip.open(self.mpileup_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                pos = f"{fields[0]}_{fields[1]}"
                
                # Only process if position is in our target set
                if pos in positions:
                    features = self.feature_extractor.extract_features_single_variant(fields)
                    # print(f"Format, format string are: {fields[8]}, {fields[9]}")
                    gt = self.feature_extractor.extract_format_field(fields[8], fields[9], 'GT')
                    # print(f"GT is {gt}")
                    if features is not None and gt is not None:
                        mpileup_vars[pos] = {
                            'ref': fields[3],
                            'alt': fields[4],
                            'features': features,
                            'gt': gt
                        }
                    
                    if len(mpileup_vars) % 10000 == 0:
                        print(f"Loaded {len(mpileup_vars):,} mpileup variants")
        
        print(f"Total matching mpileup variants loaded: {len(mpileup_vars):,}")
        return mpileup_vars

    def load_svm_model(self):
        """Load trained SVM model and related data"""
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data

    def prepare_dataset(self):
        """Extract features and labels from VCFs with position matching"""
        print("Loading variant positions and features...")
        
        # Load variants from both VCFs
        # mpileup_vars = self.load_variants(self.mpileup_vcf)
        # beagle_vars = self.load_variants(self.beagle_vcf)
        beagle_vars = self.load_beagle_variants()
        mpileup_vars = self.load_mpileup_variants(set(beagle_vars.keys()))
        
        
        # Match variants
        matched_data = []
        for pos, mpileup_var in mpileup_vars.items():
            beagle_var = beagle_vars[pos]
            
            # Check ref/alt match
            if mpileup_var['ref'] == beagle_var['ref'] and mpileup_var['alt'] == beagle_var['alt']:
                matched_data.append({
                    'pos': pos,
                    'features': mpileup_var['features'],
                    'mpileup_gt': mpileup_var['gt'],
                    'beagle_gt': beagle_var['gt']
                })
        
        print(f"Found {len(matched_data)} matching variants")
        
        # Convert to appropriate format
        features = pd.DataFrame([d['features'] for d in matched_data])
        mpileup_labels = [(d['mpileup_gt'] in ['0/1', '1/1']) for d in matched_data]
        beagle_labels = [(d['beagle_gt'] in ['0/1', '1/1']) for d in matched_data]
        
        print("\nDetailed genotype distributions before conversion:")
        mpileup_gts = [d['mpileup_gt'] for d in matched_data]
        beagle_gts = [d['beagle_gt'] for d in matched_data]
        
        print("\nMpileup genotypes:")
        for gt in set(mpileup_gts):
            count = mpileup_gts.count(gt)
            print(f"{gt}: {count}")
        
        print("\nBeagle genotypes:")
        for gt in set(beagle_gts):
            count = beagle_gts.count(gt)
            print(f"{gt}: {count}")

        # Convert genotypes to binary labels with debug prints
        print("\nConverting to binary labels...")
        features = pd.DataFrame([d['features'] for d in matched_data])
        mpileup_labels = []
        beagle_labels = []
        
        for d in matched_data:
            mpileup_gt = d['mpileup_gt']
            beagle_gt = d['beagle_gt']
            
            # Debug problematic conversions
            if mpileup_gt not in ['0/0', '0/1', '1/1']:
                print(f"Unusual mpileup genotype: {mpileup_gt}")
            if beagle_gt not in ['0/0', '0/1', '1/1']:
                print(f"Unusual beagle genotype: {beagle_gt}")
                
            mpileup_labels.append(1 if mpileup_gt in ['0/1', '1/1'] else 0)
            beagle_labels.append(1 if beagle_gt in ['0/1', '1/1'] else 0)
        
        return features, mpileup_labels, beagle_labels


    def load_variants(self, vcf_path: str) -> Dict:
        """Load variants from VCF with position as key"""
        variants = {}
        with gzip.open(vcf_path, 'rt') as f:
            # use tqdm to load variants
            for line in tqdm.tqdm(f):
            # for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                pos = f"{fields[0]}_{fields[1]}"
                
                # Extract features and genotype
                features = self.feature_extractor.extract_features_single_variant(fields)
                gt = self.feature_extractor.extract_format_field(fields[8], fields[9], 'GT')
                
                if features is not None and gt is not None:
                    variants[pos] = {
                        'ref': fields[3],
                        'alt': fields[4],
                        'features': features,
                        'gt': gt
                    }
        
        return variants
        
    def analyze_thresholds(self, svm_probs, mpileup_labels):
        """Analyze how different thresholds affect metrics"""
        thresholds = np.arange(0.05, 0.5, 0.05)
        results = []
        
        for threshold in thresholds:
            svm_pred = (svm_probs >= (1 - threshold)).astype(int)
            metrics = self.calculate_metrics(mpileup_labels, svm_pred, zero_division=0)
            metrics['threshold'] = threshold
            results.append(metrics)
        
        return pd.DataFrame(results)

    def plot_threshold_analysis(self, results_df):
        """Plot how metrics change with different thresholds"""
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
            best_idx = results_df[metric].idxmax()
            best_threshold = results_df.loc[best_idx, 'threshold']
            best_score = results_df.loc[best_idx, metric]
            plt.annotate(f'Best {metric}: {best_score:.3f}\nThreshold: {best_threshold:.2f}',
                        xy=(best_threshold, best_score),
                        xytext=(10, 10), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis.png'))
        plt.close()

    def evaluate_performance(self):
        """Compare SVM and Beagle performance with both reference points"""
        # Load trained model and prepare dataset
        model_data = self.load_svm_model()
        svm_model = model_data['svm_model']
        scaler = model_data['scaler']
        features, mpileup_labels, beagle_labels = self.prepare_dataset()
        
        # Print distributions
        print("\nLabel distributions:")
        print("Mpileup labels:")
        print(f"0/0: {sum(not x for x in mpileup_labels)}")
        print(f"0/1 or 1/1: {sum(mpileup_labels)}")
        
        print("\nBeagle labels:")
        print(f"0/0: {sum(not x for x in beagle_labels)}")
        print(f"0/1 or 1/1: {sum(beagle_labels)}")
        
        # Get SVM predictions
        features_scaled = scaler.transform(features)
        svm_probs = svm_model.predict_proba(features_scaled)[:, 1]
        
        # Analyze thresholds using both reference points
        print("\nAnalyzing with mpileup as ground truth...")
        mpileup_results = self.analyze_thresholds(svm_probs, mpileup_labels)
        
        print("\nAnalyzing with Beagle as ground truth...")
        beagle_results = self.analyze_thresholds(svm_probs, beagle_labels)
        
        # Plot both analyses
        self.plot_threshold_analysis(mpileup_results)
        self.plot_threshold_analysis(beagle_results)
        
        # Find best thresholds for both cases
        best_mpileup_thresh = mpileup_results.loc[mpileup_results['f1'].idxmax(), 'threshold']
        best_beagle_thresh = beagle_results.loc[beagle_results['f1'].idxmax(), 'threshold']
        
        # Final predictions using both thresholds
        svm_pred_mpileup = (svm_probs >= (1 - best_mpileup_thresh)).astype(int)
        svm_pred_beagle = (svm_probs >= (1 - best_beagle_thresh)).astype(int)
        
        # Calculate metrics for both perspectives
        metrics = {
            'VS_MPILEUP': {
                'SVM': self.calculate_metrics(mpileup_labels, svm_pred_mpileup, zero_division=0),
                'Beagle': self.calculate_metrics(mpileup_labels, beagle_labels, zero_division=0)
            },
            'VS_BEAGLE': {
                'SVM': self.calculate_metrics(beagle_labels, svm_pred_beagle, zero_division=0),
            }
        }
        
        # Print results
        print("\nResults using mpileup as ground truth:")
        print(f"Best threshold: {best_mpileup_thresh:.3f}")
        for method, results in metrics['VS_MPILEUP'].items():
            print(f"\n{method} Results:")
            for metric, value in results.items():
                print(f"{metric}: {value:.3f}")
                
        print("\nResults using Beagle as ground truth:")
        print(f"Best threshold: {best_beagle_thresh:.3f}")
        for metric, value in metrics['VS_BEAGLE']['SVM'].items():
            print(f"{metric}: {value:.3f}")
        
        return metrics, best_beagle_thresh

    def calculate_metrics(self, y_true, y_pred, zero_division=0):
        """Calculate performance metrics with handling for edge cases"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=zero_division),
            'recall': recall_score(y_true, y_pred, zero_division=zero_division),
            'f1': f1_score(y_true, y_pred, zero_division=zero_division)
        }

    def plot_comparison(self, metrics):
        """Plot comparison of SVM vs Beagle performance"""
        plt.figure(figsize=(10, 6))
        
        metric_names = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metric_names))
        width = 0.35
        
        plt.bar(x - width/2, [metrics['SVM'][m] for m in metric_names], width, label='SVM')
        plt.bar(x + width/2, [metrics['Beagle'][m] for m in metric_names], width, label='Beagle')
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('SVM vs Beagle Performance Comparison')
        plt.xticks(x, metric_names)
        plt.legend()
        
        # Add value labels on bars
        for i in x:
            plt.text(i - width/2, metrics['SVM'][metric_names[i]], 
                    f'{metrics["SVM"][metric_names[i]]:.3f}', 
                    ha='center', va='bottom')
            plt.text(i + width/2, metrics['Beagle'][metric_names[i]], 
                    f'{metrics["Beagle"][metric_names[i]]:.3f}', 
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'performance_comparison.png'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compare SVM and Beagle performance")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    evaluator = SVMEvaluator(args.section_id, args.quality_filter)
    metrics, best_threshold = evaluator.evaluate_performance()
    
    print("\nPerformance Comparison:")
    print("-" * 50)
    print(f"Best threshold: {best_threshold:.3f}")
    # for method in ['SVM', 'Beagle']:
    #     print(f"\n{method} Results:")
    #     for metric, value in metrics[method].items():
    #         print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()

# Run the script with the following command:
# python scripts/postprocess/run_svm_compare_to_beagle.py --section_id 151507 --quality-filter baseQ0mapQ0