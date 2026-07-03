import os
import gzip
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import argparse
from typing import Dict, List, Optional

class SVM2HomozygousClassifier:
    def __init__(self, section_id: str, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()
        
    def setup_paths(self):
        """Setup paths for input and output files"""
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        # Input paths: Using Beagle output directly
        self.beagle_dir = os.path.join(
            section_path, "output_VCFs/beagle",
            self.quality_filter
        )
        
        # Will read high confidence variants from SVMModel output
        self.svm1_dir = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter
        )
        
        # Output directory for SVM2 model
        self.output_dir = os.path.join(
            section_path, "output_VCFs/SVM2Model",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_features(self, vcf_path: str) -> pd.DataFrame:
        """Extract features for training"""
        print(f"\nExtracting features from {vcf_path}")
        features = []
        labels = []
        
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                info = fields[7]
                format_str = fields[8]
                sample_str = fields[9]
                
                try:
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
                    format_indices = {field: idx for idx, field in enumerate(format_str.split(':'))}
                    sample_values = sample_str.split(':')
                    
                    for field in ['BAF', 'GQ', 'PL']:
                        if field in format_indices:
                            idx = format_indices[field]
                            try:
                                value = sample_values[idx]
                                if field == 'PL':
                                    value = value.split(',')[0]
                                feature_dict[field] = float(value)
                            except (ValueError, IndexError):
                                feature_dict[field] = np.nan
                    
                    # Extract GT for label
                    gt_idx = format_indices['GT']
                    gt = sample_values[gt_idx]
                    
                    if gt in ['0/1', '1/1']:  # Only keep heterozygous and homozygous alt
                        features.append(feature_dict)
                        labels.append(1 if gt == '0/1' else 0)  # 1 for heterozygous, 0 for homozygous alt
                        
                except (ValueError, IndexError):
                    continue
        
        if not features:
            raise ValueError(f"No valid features extracted from {vcf_path}")
            
        features_df = pd.DataFrame(features)
        features_df['label'] = labels
        
        print(f"Extracted {len(features_df)} features")
        print(f"Label distribution:")
        print(f"Heterozygous (0/1): {sum(labels)}")
        print(f"Homozygous alt (1/1): {len(labels) - sum(labels)}")
        
        return features_df

    def train_model(self):
        """Train SVM model using Beagle output"""
        # Load Beagle output for all chromosomes
        all_features = []
        for chrom in range(1, 23):
            vcf_path = os.path.join(self.beagle_dir, f"chr{chrom}.vcf.gz")
            if os.path.exists(vcf_path):
                features = self.extract_features(vcf_path)
                all_features.append(features)
        
        features = pd.concat(all_features, ignore_index=True)
        features = features.fillna(features.mean())
        
        # Split features and labels
        X = features.drop('label', axis=1)
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
        print("\nTraining homozygous classification model...")
        svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
        svm.fit(X_train_scaled, y_train)
        
        # Evaluate model
        val_score = svm.score(X_val_scaled, y_val)
        y_pred = svm.predict(X_val_scaled)
        
        # Save performance metrics
        with open(os.path.join(self.output_dir, 'performance_metrics.txt'), 'w') as f:
            f.write("SVM2 Classification Model Performance:\n")
            f.write(f"Validation accuracy: {val_score:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_val, y_pred, 
                                       target_names=['Homozygous alt (1/1)', 'Heterozygous (0/1)']))
        
        # Save model and scaler
        model_data = {
            'svm_model': svm,
            'scaler': scaler,
            'feature_columns': X.columns.tolist()
        }
        
        with open(os.path.join(self.output_dir, 'svm2_model.pkl'), 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"\nModel and metrics saved to: {self.output_dir}")
        
        return model_data

    def apply_model(self, model_data: Dict):
        """Apply trained model to 1/1 variants from SVM1 high confidence output"""
        print("\nApplying model to high confidence variants...")
        
        svm = model_data['svm_model']
        scaler = model_data['scaler']
        feature_cols = model_data['feature_columns']
        
        # Load high confidence VCF and filter for 1/1 genotypes
        high_conf_vcf = os.path.join(self.svm1_dir, "results/high_confidence.vcf.gz")
        features = self.extract_features(high_conf_vcf)
        
        # Keep only 1/1 variants
        homozygous_mask = (features['label'] == 0)  # 0 was our label for 1/1
        features = features[homozygous_mask]
        
        if len(features) == 0:
            raise ValueError("No homozygous alt variants found in high confidence VCF")
        
        print(f"Found {len(features):,} homozygous alt variants to evaluate")
        
        # Prepare features
        features = features.fillna(features.mean())
        X_scaled = scaler.transform(features[feature_cols])
        
        # Get predictions and probabilities
        predictions = svm.predict(X_scaled)
        probabilities = svm.predict_proba(X_scaled)
        
        # Save predictions to new VCF
        output_vcf = os.path.join(self.output_dir, "svm2_predictions.vcf.gz")
        
        with gzip.open(high_conf_vcf, 'rt') as f_in, \
             gzip.open(output_vcf, 'wt') as f_out:
            
            # Copy header and add new INFO fields
            for line in f_in:
                if line.startswith('#'):
                    if line.startswith('#CHROM'):
                        f_out.write('##INFO=<ID=SVM2_PRED,Number=1,Type=Integer,'
                                'Description="SVM2 prediction (1=should be 0/1, 0=should stay 1/1)">\n')
                        f_out.write('##INFO=<ID=SVM2_PROB,Number=1,Type=Float,'
                                'Description="SVM2 probability of being heterozygous">\n')
                    f_out.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            pred_idx = 0
            processed = skipped = 0
            
            for line in f_in:
                fields = line.strip().split('\t')
                info = fields[7]
                format_str = fields[8]
                sample_str = fields[9]
                
                try:
                    gt_idx = format_str.split(':').index('GT')
                    gt = sample_str.split(':')[gt_idx]
                    
                    if gt == '1/1' and pred_idx < len(predictions):
                        pred_idx += 1
                        processed += 1
                    else:
                        skipped += 1
                    info += f";SVM2_PRED={predictions[pred_idx]}"
                    info += f";SVM2_PROB={probabilities[pred_idx][1]:.4f}"
                    fields[7] = info
                    f_out.write('\t'.join(fields) + '\n')
                    
                except (ValueError, IndexError):
                    skipped += 1
                    f_out.write(line)
                
                if (processed + skipped) % 100000 == 0:
                    print(f"Processed {processed:,} variants ({skipped:,} skipped)...")
        
        print(f"\nFinal statistics:")
        print(f"Processed 1/1 variants: {processed:,}")
        print(f"Skipped variants: {skipped:,}")
        print(f"Recommended for change to 0/1: {sum(predictions == 1):,}")
        print(f"Recommended to stay 1/1: {sum(predictions == 0):,}")

def main():
    parser = argparse.ArgumentParser(
        description="Train SVM2 model for homozygous/heterozygous classification")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    parser.add_argument("--use-existing-model", action="store_true",
                      help="Use existing model instead of training new one")
    args = parser.parse_args()
    
    classifier = SVM2HomozygousClassifier(args.section_id, args.quality_filter)
    
    if args.use_existing_model:
        model_path = os.path.join(classifier.output_dir, 'svm2_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No existing model found at {model_path}")
            
        print(f"Loading existing model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
    else:
        print("Training new model...")
        model_data = classifier.train_model()
    
    classifier.apply_model(model_data)

if __name__ == "__main__":
    main()

# # Train new model
# python scripts/postprocess/run_svm2_hetero_finding.py --section_id 151507 --quality-filter baseQ0mapQ0

# # Use existing model
# python scripts/postprocess/run_svm2_hetero_finding.py --section_id 151507 --quality-filter baseQ0mapQ0 --use-existing-model