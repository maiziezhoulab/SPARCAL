#!/usr/bin/env python3
"""
SPARCAL-Net: Neural Network Classifier for Spatial Transcriptomics Variant Classification
Streamlined script for training and applying neural network to classify variants as:
- homozygous (1/1)
- heterozygous (0/1)
- no_variance (0/0)

Input: Variants from two sources:
    1. Beagle output (all_filtered_in.vcf.gz) - SOURCE=BEAGLE
    2. Sequence error model output (sequence_no_error.vcf.gz) - SOURCE=seq_no_err

Output: Three classified VCF files with all original headers/fields preserved
"""

import os
import gzip
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import argparse
from dataclasses import dataclass
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import subprocess

# Dataset and reference configurations
REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",
        "regions": [str(i) for i in range(1, 23)]
    },
    "CHR_PREFIX": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",
        "regions": [f"chr{i}" for i in range(1, 23)]
    }
}

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC"
    }
}

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "BCFTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix",
    "SAMTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools",
}


class FeatureExtractor:
    """Extract features from VCF files for neural network training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def extract_info_field(self, info_str: str, field: str) -> Optional[float]:
        """Extract numerical value from INFO field"""
        for item in info_str.split(';'):
            if item.startswith(f"{field}="):
                try:
                    return float(item.split('=')[1].split(',')[0])
                except (ValueError, IndexError):
                    return None
        return None

    def extract_format_field(self, format_str: str, sample_str: str, field: str) -> Optional[str]:
        """Extract field from FORMAT column"""
        try:
            idx = format_str.split(':').index(field)
            value = sample_str.split(':')[idx]
            if field == 'PL':
                return float(value.split(',')[0])
            return value if field == 'GT' else float(value)
        except (ValueError, IndexError):
            return None

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

    def extract_features(self, vcf_path: str, source: str) -> pd.DataFrame:
        """
        Extract features from VCF file
        
        Args:
            vcf_path: Path to VCF file
            source: 'BEAGLE' or 'seq_no_err' to track variant origin
        """
        features = []
        numeric_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F', 'BAF', 'GQ']
        info_fields = ['DP', 'VDB', 'RPB', 'MQB', 'BQB', 'SGB', 'BAF']
        custom_fields = ['BAF', 'GQ']
        
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                    
                try:
                    feature_dict = {field: 0.0 for field in numeric_fields}
                    feature_dict['POS'] = int(fields[1])
                    feature_dict['SOURCE'] = source  # Track variant source
                    
                    # Extract INFO fields
                    for field in info_fields:
                        value = self.extract_info_field(fields[7], field)
                        if value is not None:
                            feature_dict[field] = value
                    
                    # Extract FORMAT fields
                    for field in custom_fields:
                        value = self.extract_format_field(fields[8], fields[9], field)
                        if value is not None:
                            feature_dict[field] = value
                    
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


class TrainingSetBuilder:
    """Build training sets from Beagle and SeqErrModel outputs"""
    
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", 
                 section_id: str = None, max_training_samples: int = 90000):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.max_training_samples = max_training_samples
        self.base_dir = PATH_CONFIG["PROJECT_DIR"]
        
        self.validate_dataset_config()
        self.setup_paths()

    def validate_dataset_config(self):
        """Validate dataset configuration and section ID"""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        if dataset_config["has_sections"]:
            if not self.section_id:
                raise ValueError(f"Dataset {self.dataset_name} requires a section_id")
            if "section_ids" in dataset_config:
                if self.section_id not in dataset_config["section_ids"]:
                    raise ValueError(f"Invalid section_id for {self.dataset_name}. "
                                   f"Valid section IDs are: {dataset_config['section_ids']}")

    def setup_paths(self):
        """Setup all necessary file paths"""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        if dataset_config["has_sections"]:
            output_base = os.path.join(
                self.base_dir,
                dataset_config["output_dir"].format(section_id=self.section_id)
            )
        else:
            output_base = os.path.join(
                self.base_dir,
                dataset_config["output_dir"]
            )
        
        # Input VCF files
        self.beagle_vcf = os.path.join(output_base, "output_VCFs/beagle", 
                                       self.quality_filter, "all_filtered_in.vcf.gz")
        self.seq_no_error_vcf = os.path.join(output_base, "output_VCFs/SeqErrModel", 
                                            self.quality_filter, "sequence_no_error.vcf.gz")
        
        # Output directory for neural network
        self.nn_dir = os.path.join(output_base, "output_VCFs/SPARCALNet", self.quality_filter)
        os.makedirs(self.nn_dir, exist_ok=True)
        
        # Model and metrics files
        self.model_file = os.path.join(self.nn_dir, "neural_network_model.pkl")
        self.scaler_file = os.path.join(self.nn_dir, "neural_network_scaler.pkl")
        self.label_encoder_file = os.path.join(self.nn_dir, "neural_network_label_encoder.pkl")
        
        print(f"\nInput VCF files:")
        print(f"  Beagle output: {self.beagle_vcf}")
        print(f"  SeqErr output: {self.seq_no_error_vcf}")
        print(f"  Output directory: {self.nn_dir}")

    def build_training_set(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Build training sets from both Beagle and SeqErrModel outputs
        Returns: (X_train, y_train, feature_names)
        """
        print("\n" + "="*80)
        print("Building Training Set from Two Sources")
        print("="*80)
        
        extractor = FeatureExtractor()
        
        # Extract features from both sources
        print("\n1. Extracting features from Beagle output (SOURCE=BEAGLE)...")
        beagle_features = extractor.extract_features(self.beagle_vcf, source='BEAGLE')
        print(f"   Extracted {len(beagle_features)} variants from Beagle")
        
        print("\n2. Extracting features from SeqErrModel output (SOURCE=seq_no_err)...")
        seq_features = extractor.extract_features(self.seq_no_error_vcf, source='seq_no_err')
        print(f"   Extracted {len(seq_features)} variants from SeqErrModel")
        
        # Combine features
        all_features = pd.concat([beagle_features, seq_features], ignore_index=True)
        print(f"\n3. Total variants: {len(all_features)}")
        print(f"   - From BEAGLE: {len(beagle_features)}")
        print(f"   - From seq_no_err: {len(seq_features)}")
        
        # Extract labels from ground truth genotypes
        print("\n4. Extracting ground truth labels from genotypes...")
        labels = self._extract_labels_from_vcf()
        
        # Ensure labels match features
        if len(labels) != len(all_features):
            raise ValueError(f"Label count ({len(labels)}) doesn't match feature count ({len(all_features)})")
        
        all_features['label'] = labels
        
        # Remove rows with missing labels
        all_features = all_features.dropna(subset=['label'])
        print(f"   Valid labeled variants: {len(all_features)}")
        
        # Separate features and labels
        feature_cols = [col for col in all_features.columns if col not in ['label', 'SOURCE', 'POS']]
        X = all_features[feature_cols]
        y = all_features['label']
        
        # Sample if needed
        if len(X) > self.max_training_samples:
            print(f"\n5. Sampling {self.max_training_samples} variants for training...")
            sample_indices = np.random.choice(len(X), self.max_training_samples, replace=False)
            X = X.iloc[sample_indices]
            y = y.iloc[sample_indices]
        
        print(f"\nFinal training set: {len(X)} variants")
        print(f"Label distribution:")
        print(y.value_counts())
        
        return X, y, feature_cols

    def _extract_labels_from_vcf(self) -> List[str]:
        """Extract ground truth genotype labels from VCF files"""
        labels = []
        
        # Read Beagle VCF
        with gzip.open(self.beagle_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                
                # Extract genotype
                format_fields = fields[8].split(':')
                sample_fields = fields[9].split(':')
                
                try:
                    gt_idx = format_fields.index('GT')
                    gt = sample_fields[gt_idx]
                    
                    # Map genotype to label
                    if gt == '0/0':
                        labels.append('no_variance')
                    elif gt == '0/1' or gt == '1/0':
                        labels.append('heterozygous')
                    elif gt == '1/1':
                        labels.append('homozygous')
                    else:
                        labels.append(None)
                except (ValueError, IndexError):
                    labels.append(None)
        
        # Read SeqErrModel VCF
        with gzip.open(self.seq_no_error_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                
                # Extract genotype
                format_fields = fields[8].split(':')
                sample_fields = fields[9].split(':')
                
                try:
                    gt_idx = format_fields.index('GT')
                    gt = sample_fields[gt_idx]
                    
                    # Map genotype to label
                    if gt == '0/0':
                        labels.append('no_variance')
                    elif gt == '0/1' or gt == '1/0':
                        labels.append('heterozygous')
                    elif gt == '1/1':
                        labels.append('homozygous')
                    else:
                        labels.append(None)
                except (ValueError, IndexError):
                    labels.append(None)
        
        return labels


class NeuralNetworkClassifier:
    """Neural Network Classifier for variant genotypes"""
    
    def __init__(self, builder: TrainingSetBuilder):
        self.builder = builder
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def train(self):
        """Train the neural network model"""
        print("\n" + "="*80)
        print("Training Neural Network Classifier")
        print("="*80)
        
        # Build training set
        X, y, feature_names = self.builder.build_training_set()
        self.feature_names = feature_names
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\nTraining set: {len(X_train)} variants")
        print(f"Test set: {len(X_test)} variants")
        
        # Train neural network
        print("\nTraining neural network...")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            verbose=True
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n" + "="*80)
        print(f"Model Performance")
        print("="*80)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=self.label_encoder.classes_))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        self._save_model()
        
    def _save_model(self):
        """Save trained model and associated objects"""
        print(f"\nSaving model to {self.builder.nn_dir}...")
        
        with open(self.builder.model_file, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.builder.scaler_file, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(self.builder.label_encoder_file, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("Model saved successfully!")
    
    def load_model(self):
        """Load pre-trained model"""
        print(f"\nLoading model from {self.builder.nn_dir}...")
        
        if not os.path.exists(self.builder.model_file):
            raise FileNotFoundError(f"Model file not found: {self.builder.model_file}")
        
        with open(self.builder.model_file, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(self.builder.scaler_file, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(self.builder.label_encoder_file, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Model loaded successfully!")

    def apply_to_vcf(self, conf_threshold: float = 0.5):
        """
        Apply trained model to VCF files and create classified outputs
        
        Args:
            conf_threshold: Confidence threshold for classification
        """
        print("\n" + "="*80)
        print("Applying Neural Network to VCF Files")
        print("="*80)
        print(f"Confidence threshold: {conf_threshold}")
        
        # Process both VCF files
        self._process_vcf_file(self.builder.beagle_vcf, 'BEAGLE', conf_threshold)
        self._process_vcf_file(self.builder.seq_no_error_vcf, 'seq_no_err', conf_threshold)
        
        # Merge and finalize outputs
        self._merge_and_compress_outputs()

    def _process_vcf_file(self, vcf_path: str, source: str, conf_threshold: float):
        """Process a single VCF file and collect classified variants"""
        print(f"\nProcessing {source} variants from: {vcf_path}")
        
        # Extract features
        extractor = FeatureExtractor()
        features_df = extractor.extract_features(vcf_path, source=source)
        
        # Prepare features for prediction
        feature_cols = [col for col in features_df.columns if col not in ['SOURCE', 'POS']]
        X = features_df[feature_cols].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Store results for VCF writing
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            pred_class = self.label_encoder.inverse_transform([pred])[0]
            pred_conf = probs[pred]
            
            if pred_conf >= conf_threshold:
                results.append({
                    'index': i,
                    'class': pred_class,
                    'prob_homo': probs[self.label_encoder.transform(['homozygous'])[0]],
                    'prob_hetero': probs[self.label_encoder.transform(['heterozygous'])[0]],
                    'prob_novar': probs[self.label_encoder.transform(['no_variance'])[0]],
                    'source': source
                })
        
        # Write classified variants to VCF
        self._write_classified_vcf(vcf_path, results)
        
        print(f"Processed {len(results)}/{len(features_df)} variants (threshold={conf_threshold})")

    def _write_classified_vcf(self, input_vcf: str, results: List[Dict]):
        """Write classified variants to separate VCF files"""
        # Open temporary output files
        homo_vcf = os.path.join(self.builder.nn_dir, "temp_homozygous.vcf")
        hetero_vcf = os.path.join(self.builder.nn_dir, "temp_heterozygous.vcf")
        novar_vcf = os.path.join(self.builder.nn_dir, "temp_no_variance.vcf")
        
        # Create file handles
        homo_handle = open(homo_vcf, 'a')
        hetero_handle = open(hetero_vcf, 'a')
        novar_handle = open(novar_vcf, 'a')
        
        # Check if headers have been written
        write_header = not os.path.exists(homo_vcf) or os.path.getsize(homo_vcf) == 0
        
        # Read and process VCF
        with gzip.open(input_vcf, 'rt') as f:
            variant_idx = 0
            
            for line in f:
                # Write headers to all output files
                if line.startswith('#'):
                    if write_header:
                        # Add custom INFO lines before #CHROM
                        if line.startswith('#CHROM'):
                            for handle in [homo_handle, hetero_handle, novar_handle]:
                                handle.write('##INFO=<ID=SOURCE,Number=1,Type=String,Description="Variant source: BEAGLE or seq_no_err">\n')
                                handle.write('##INFO=<ID=NN_PROB_HOMO,Number=1,Type=Float,Description="Neural network probability for homozygous genotype">\n')
                                handle.write('##INFO=<ID=NN_PROB_HETERO,Number=1,Type=Float,Description="Neural network probability for heterozygous genotype">\n')
                                handle.write('##INFO=<ID=NN_PROB_NOVAR,Number=1,Type=Float,Description="Neural network probability for no variance genotype">\n')
                        
                        homo_handle.write(line)
                        hetero_handle.write(line)
                        novar_handle.write(line)
                    continue
                
                # Check if this variant should be written
                if variant_idx < len(results):
                    result = results[variant_idx]
                    
                    # Add NN probabilities and SOURCE to INFO field
                    fields = line.strip().split('\t')
                    info_field = fields[7]
                    info_field += f";SOURCE={result['source']}"
                    info_field += f";NN_PROB_HOMO={result['prob_homo']:.4f}"
                    info_field += f";NN_PROB_HETERO={result['prob_hetero']:.4f}"
                    info_field += f";NN_PROB_NOVAR={result['prob_novar']:.4f}"
                    fields[7] = info_field
                    
                    modified_line = '\t'.join(fields) + '\n'
                    
                    # Write to appropriate file
                    if result['class'] == 'homozygous':
                        homo_handle.write(modified_line)
                    elif result['class'] == 'heterozygous':
                        hetero_handle.write(modified_line)
                    elif result['class'] == 'no_variance':
                        novar_handle.write(modified_line)
                
                variant_idx += 1
        
        # Close file handles
        homo_handle.close()
        hetero_handle.close()
        novar_handle.close()

    def _merge_and_compress_outputs(self):
        """Merge temporary VCF files and compress final outputs"""
        print("\nFinalizing output VCF files...")
        
        # Define final output paths
        homo_vcf_gz = os.path.join(self.builder.nn_dir, "neural_network_homozygous.vcf.gz")
        hetero_vcf_gz = os.path.join(self.builder.nn_dir, "neural_network_heterozygous.vcf.gz")
        novar_vcf_gz = os.path.join(self.builder.nn_dir, "neural_network_no_variance.vcf.gz")
        
        # Compress and index each file
        for temp_name, final_name in [
            ("temp_homozygous.vcf", homo_vcf_gz),
            ("temp_heterozygous.vcf", hetero_vcf_gz),
            ("temp_no_variance.vcf", novar_vcf_gz)
        ]:
            temp_path = os.path.join(self.builder.nn_dir, temp_name)
            
            if os.path.exists(temp_path):
                # Compress
                subprocess.run([PATH_CONFIG['BGZIP'], '-f', temp_path], check=True)
                temp_gz = temp_path + '.gz'
                
                # Move to final location
                subprocess.run(['mv', temp_gz, final_name], check=True)
                
                # Index
                subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', final_name], check=True)
        
        print(f"\nOutput VCF files created:")
        print(f"  Homozygous: {homo_vcf_gz}")
        print(f"  Heterozygous: {hetero_vcf_gz}")
        print(f"  No variance: {novar_vcf_gz}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="SPARCAL-Net: Neural Network Classifier for Variant Classification"
    )
    parser.add_argument("--dataset", required=True, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset to process")
    parser.add_argument("--section_id", 
                       help="Section ID (required for datasets with sections)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                       help="Quality filter to use (default: baseQ0mapQ0)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for classification (default: 0.5)")
    parser.add_argument("--max-training-samples", type=int, default=90000,
                       help="Maximum number of training samples (default: 90000)")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training and use existing model")
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. "
                        f"Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
    
    # Print configuration
    print("\n" + "="*80)
    print("SPARCAL-Net Configuration")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Max Training Samples: {args.max_training_samples}")
    print(f"Skip Training: {args.skip_training}")
    
    # Initialize
    builder = TrainingSetBuilder(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        max_training_samples=args.max_training_samples
    )
    
    classifier = NeuralNetworkClassifier(builder)
    
    # Train or load model
    if not args.skip_training:
        classifier.train()
    else:
        classifier.load_model()
    
    # Apply to VCF files
    classifier.apply_to_vcf(conf_threshold=args.confidence_threshold)
    
    print("\n" + "="*80)
    print("SPARCAL-Net Complete!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    exit(main())


# Usage examples:
# Train and apply for DLPFC:
# python run_sparcal_net.py --dataset DLPFC --section_id 151669 --quality-filter baseQ0mapQ0

# Train and apply for P4_TUMOR:
# python run_sparcal_net.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0

# Apply pre-trained model only:
# python run_sparcal_net.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0 --skip-training --confidence-threshold 0.5

# Train and apply for DCIS:
# python run_sparcal_net.py --dataset DCIS --section_id 1 --quality-filter baseQ0mapQ0