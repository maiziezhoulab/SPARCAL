import os
import gzip
import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

# Import dataset configurations
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
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "CHR_PREFIX"
    }
}

@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str

class SVMFeatureExtractor:
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

    def extract_features(self, vcf_path: str, dataset_name: str) -> pd.DataFrame:
        """Extract features from VCF file with dataset-specific handling"""
        features = []
        numeric_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F',
                         'BAF', 'GQ']
        info_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F']
        custom_fields = ['BAF', 'GQ']
        
        # In Monopogen paper, SNVs calling quality metrics including quality score for calling, variant distance bias for filtering splice-site artifacts, Mann–Whitney U
        # test of read position bias, Mann–Whitney U test of base quality bias,
        # Mann–Whitney U test of ratio of mapping quality and strand bias,
        # segregation-based metric and BAF are selected as features. 
        # And those are: 'DP', 'VDB', 'RPB', 'MQB', 'BQB', 'SGB', 'BAF'
        
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
                    
                    features.append(feature_dict)
                    
                except Exception:
                    continue
                    
        if not features:
            raise ValueError(f"No valid features extracted from {vcf_path}")
            
        return pd.DataFrame(features)

class TrainingSetBuilder:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.validate_dataset_config()
        self.setup_paths()
        self.positive_variants = []
        self.negative_variants = []

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
                    raise ValueError(f"Invalid section_id {self.section_id}")

    def setup_paths(self):
        """Setup paths for input and output files"""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        if dataset_config["has_sections"]:
            section_path = os.path.join(
                self.base_dir,
                dataset_config["output_dir"].format(section_id=self.section_id)
            )
        else:
            section_path = os.path.join(
                self.base_dir,
                dataset_config["output_dir"]
            )
            
        # Results paths
        self.shifted_results = os.path.join(
            section_path, "metrics/beagle",
            self.quality_filter,
            f"{'_'.join(filter(None, [self.dataset_name, self.section_id]))}_shifted_results.pkl"
        )
        
        self.stable_results = os.path.join(
            section_path, "metrics/beagle",
            self.quality_filter,
            f"{'_'.join(filter(None, [self.dataset_name, self.section_id]))}_stable_results.pkl"
        )
        
        # Input VCF paths
        self.input_vcf = os.path.join(
            section_path, "output_VCFs/mpileup_multi_bam",
            self.quality_filter, "merged_sorted_gt.vcf.gz"
        )
        
        self.seq_error_vcf = os.path.join(
            section_path, "output_VCFs/SeqErrModel",
            self.quality_filter, "sequence_error.vcf.gz"
        )
        
        # Output directory
        self.output_dir = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_pickle_results(self, file_path: str) -> Dict:
        """Load results from pickle file"""
        with open(file_path, 'rb') as f:
            results = pickle.load(f)
            return results.get('metrics_by_transition', {})

    def collect_variants_from_metrics(self, metrics_dict: Dict, target_transition: Tuple[str, str]) -> List[Dict]:
        """Collect variants from metrics dictionary for specific transition"""
        orig_gt, new_gt = target_transition
        key = f"{orig_gt}->{new_gt}"
        variants = []
        
        for trans_key, metrics in metrics_dict.items():
            if trans_key.startswith(key):
                ref, alt = trans_key.split('_')[1:]
                for metric in metrics:
                    if 'line' in metric:  # Check if original VCF line is available
                        variants.append({
                            'line': metric['line'],
                            'original_gt': orig_gt,
                            'new_gt': new_gt
                        })
        
        return variants

    def collect_seq_error_variants(self) -> List[Dict]:
        """Collect sequence error variants from VCF"""
        variants = []
        with gzip.open(self.seq_error_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                variants.append({'line': line})
        return variants

    def build_training_sets(self):
        """Build positive and negative training sets"""
        print("\nLoading transition metrics...")
        shifted_metrics = self.load_pickle_results(self.shifted_results)
        stable_metrics = self.load_pickle_results(self.stable_results)
        
        print("\nCollecting training examples...")
        # Collect positive examples
        positive_variants = []
        positive_variants.extend(self.collect_variants_from_metrics(
            stable_metrics, ("0/1", "0/1")
        ))
        positive_variants.extend(self.collect_variants_from_metrics(
            stable_metrics, ("1/1", "1/1")
        ))
        
        # Collect negative examples
        negative_variants = []
        negative_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("0/1", "0/0")
        ))
        negative_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("1/1", "0/0")
        ))
        negative_variants.extend(self.collect_seq_error_variants())

        # Save training sets
        self.save_variants(positive_variants, "positive_training.vcf.gz")
        self.save_variants(negative_variants, "negative_training.vcf.gz")
        self.positive_variants = positive_variants
        self.negative_variants = negative_variants
        
        print(f"\nTraining Set Statistics:")
        print(f"Positive examples: {len(positive_variants):,}")
        print(f"  - 0/1 -> 0/1: {len([v for v in positive_variants if v.get('original_gt')=='0/1']):,}")
        print(f"  - 1/1 -> 1/1: {len([v for v in positive_variants if v.get('original_gt')=='1/1']):,}")
        print(f"\nNegative examples: {len(negative_variants):,}")
        print(f"  - 0/1 -> 0/0: {len([v for v in negative_variants if v.get('original_gt')=='0/1']):,}")
        print(f"  - 1/1 -> 0/0: {len([v for v in negative_variants if v.get('original_gt')=='1/1']):,}")
        print(f"  - Sequence errors: {len([v for v in negative_variants if not v.get('original_gt')]):,}")

    def save_variants(self, variants: List[Dict], filename: str):
        """Save variants to VCF file"""
        output_path = os.path.join(self.output_dir, filename)
        with gzip.open(output_path, 'wt') as f:
            # Write header from first variant that has header lines
            header_written = False
            for variant in variants:
                if 'header_lines' in variant:
                    for header in variant['header_lines']:
                        f.write(header)
                    header_written = True
                    break
            
            # If no header found in variants, write minimal header
            if not header_written:
                f.write("##fileformat=VCFv4.2\n")
                f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            
            # Write variants
            for variant in variants:
                f.write(variant['line'])

def main():
    parser = argparse.ArgumentParser(description="Build SVM training sets")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    args = parser.parse_args()
    
    builder = TrainingSetBuilder(args.dataset, args.quality_filter, args.section_id)
    builder.build_training_sets()

    # Get positive and negative training sets
    positive_variants = builder.positive_variants
    negative_variants = builder.negative_variants
    # Extract features
    extractor = SVMFeatureExtractor()
    positive_features = extractor.extract_features(builder.input_vcf, args.dataset)
    negative_features = extractor.extract_features(builder.seq_error_vcf, args.dataset)
    # Combine features
    features = pd.concat([positive_features, negative_features], ignore_index=True)
    # Prepare labels
    labels = np.array([1] * len(positive_features) + [0] * len(negative_features))
    # Train SVM with progress bar
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm = SVC(
        kernel='linear',
        verbose=True,
        # max_iter=1000,  # Limit maximum iterations
        cache_size=2000  # Increase cache size for faster training
    )
    svm.fit(X_train, y_train)
    # Evaluate SVM
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    # Save model
    model_path = os.path.join(builder.output_dir, "svm_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(svm, f)
    # Save scaler
    scaler_path = os.path.join(builder.output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(extractor.scaler, f)
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(svm.coef_[0])), svm.coef_[0])
    plt.xlabel('Feature Index')
    plt.ylabel('Coefficient Value')
    plt.title('Feature Importance')
    plt.savefig(os.path.join(builder.output_dir, "feature_importance.png"))
    plt.close() 
    # Plot distribution of positive and negative examples on first two principal components
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(features)
    plt.figure(figsize=(10, 6))
    plt.scatter(X_pca[y_train == 1, 0], X_pca[y_train == 1, 1], label='Positive', alpha=0.5)
    plt.scatter(X_pca[y_train == 0, 0], X_pca[y_train == 0, 1], label='Negative', alpha=0.5)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Training Set')
    plt.legend()
    plt.savefig(os.path.join(builder.output_dir, "pca_distribution.png"))
    plt.close()
    # Plot distribution of positive and negative examples on DP and BAF
    plt.figure(figsize=(10, 6))
    plt.scatter(positive_features['DP'], positive_features['BAF'], label='Positive', alpha=0.5)
    plt.scatter(negative_features['DP'], negative_features['BAF'], label='Negative', alpha=0.5)
    plt.xlabel('DP')
    plt.ylabel('BAF')
    plt.title('Distribution of Positive and Negative Examples')
    plt.legend()
    plt.savefig(os.path.join(builder.output_dir, "dp_baf_distribution.png"))
    plt.close()

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC with section:
# python scripts/postprocess/run_svm_hetero_finding.py --dataset DLPFC --section_id 151507 --quality-filter baseQ0mapQ0

# For 10X_BC_6.5MM (no section):
# python scripts/postprocess/run_svm_hetero_finding.py --dataset 10X_BC_6.5MM --quality-filter baseQ0mapQ0

# For P4_TUMOR with section:
# python scripts/postprocess/run_svm_hetero_finding.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0