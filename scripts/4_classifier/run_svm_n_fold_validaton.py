import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Union
import gzip


class SVMTrainer:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()
        
    def setup_paths(self):
        """Setup paths for input and output files"""
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        # Training data paths
        self.positive_vcf = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "positive_training.vcf.gz"
        )
        self.negative_vcf = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "negative_training.vcf.gz"
        )
        
        # Output directory
        self.output_dir = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "model"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """
        Find optimal threshold using validation data
        """
        thresholds = np.arange(0.05, 0.55, 0.05)
        results = []
        
        for threshold in thresholds:
            # Predict using current threshold
            y_pred = (y_prob >= (1 - threshold)).astype(int)
            
            # Calculate metrics
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred)
            }
            results.append(metrics)
            
        results_df = pd.DataFrame(results)
        
        # Find best threshold based on F1 score
        best_idx = results_df['f1'].idxmax()
        best_results = results_df.loc[best_idx].to_dict()
        
        # Plot threshold analysis
        self.plot_threshold_analysis(results_df)
        
        return best_results

    def train_svm_with_threshold(self):
        """Train SVM model and find optimal threshold"""
        # Load and prepare data
        print("\nLoading training data...")
        feature_extractor = SVMFeatureExtractor()
        
        print("Processing positive examples...")
        positive_features = feature_extractor.extract_features(self.positive_vcf)
        positive_features['label'] = 1
        
        print("Processing negative examples...")
        negative_features = feature_extractor.extract_features(self.negative_vcf)
        negative_features['label'] = 0
        
        # Combine datasets
        features = pd.concat([positive_features, negative_features])
        features = features.fillna(features.mean())
        
        # Split features and labels
        # First get list of numeric columns (excluding 'label' and any string columns)
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols.remove('label')  # Remove label from feature columns
        
        X = features[feature_cols]
        y = features['label']
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train SVM
        print("\nTraining SVM model...")
        svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
        svm.fit(X_train_scaled, y_train)
        
        # Get probabilities for validation set
        val_probs = svm.predict_proba(X_val_scaled)[:, 1]
        
        # Find optimal threshold
        print("\nFinding optimal threshold...")
        threshold_results = self.find_optimal_threshold(y_val, val_probs)
        
        print("\nOptimal threshold results:")
        for metric, value in threshold_results.items():
            print(f"{metric}: {value:.3f}")
        
        # Save model, scaler, and threshold
        model_data = {
            'svm_model': svm,
            'scaler': scaler,
            'feature_columns': feature_cols,  # Save feature columns for later use
            'threshold': threshold_results['threshold'],
            'performance_metrics': threshold_results
        }
        
        with open(os.path.join(self.output_dir, 'svm_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
            
        return model_data

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
        plt.title('Performance Metrics vs. Threshold on Validation Set')
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
            if field == 'PL':
                return float(value.split(',')[0])
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
    
def main():
    parser = argparse.ArgumentParser(description="Train SVM model with threshold optimization")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    trainer = SVMTrainer(args.section_id, args.quality_filter)
    model_data = trainer.train_svm_with_threshold()
    
    print(f"\nModel and threshold saved to: {trainer.output_dir}")
    print(f"Optimal threshold: {model_data['threshold']:.3f}")
    print("\nValidation set performance:")
    for metric, value in model_data['performance_metrics'].items():
        print(f"{metric}: {value:.3f}")

if __name__ == "__main__":
    main()

# Usage
# python scripts/postprocess/run_svm_n_fold_validaton.py --section_id 151507 --quality-filter baseQ0mapQ0