import os
import gzip
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Set, Tuple, Optional
import pickle

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
    "10X_BC_6.5MM": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE",
        "output_dir": "data/10X_BC_6.5mm",
        "has_sections": False,
        "reference": "CHR_PREFIX"
    },
    "10X_BC_FFPE": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_Ductal_Carcinoma_In_Situ_Invasive_Carcinoma_FFPE",
        "output_dir": "data/10X_BC_FFPE",
        "has_sections": False,
        "reference": "CHR_PREFIX"
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

class SVMFeatureExtractor:
    def __init__(self):
        pass

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

    def extract_features_single_variant(self, fields: List[str], dataset_name: str) -> Optional[Dict]:
        """Extract features from a single VCF line with dataset-specific handling"""
        try:
            feature_dict = {}
            
            # Extract INFO fields
            info_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F']
            for field in info_fields:
                value = self.extract_info_field(fields[7], field)
                if value is not None:
                    feature_dict[field] = value

            # Extract FORMAT fields based on dataset
            format_fields = ['BAF', 'GQ', 'PL']
            for field in format_fields:
                value = self.extract_format_field(fields[8], fields[9], field)
                if value is not None:
                    feature_dict[field] = value

            return feature_dict if feature_dict else None

        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            return None

class SVMEvaluator:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.validate_dataset_config()
        self.setup_paths()
        self.feature_extractor = SVMFeatureExtractor()

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
        """Load variants from Beagle VCFs with dataset-specific handling"""
        print("Loading Beagle variants...")
        beagle_vars = {}
        
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
        
        # Process each chromosome
        for chrom in reference_config['regions']:
            chrom_vcf = os.path.join(os.path.dirname(self.beagle_vcf), f"{chrom}.vcf.gz")
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

    def evaluate_performance(self):
        """Compare SVM and Beagle performance"""
        # Load model data
        print("Loading SVM model...")
        model_data = self.load_svm_model()
        
        # Load variants and extract features
        print("\nPreparing dataset...")
        features, mpileup_labels, beagle_labels = self.prepare_dataset()
        
        # Scale features and get predictions
        X_scaled = model_data['scaler'].transform(features)
        svm_probs = model_data['svm_model'].predict_proba(X_scaled)[:, 1]
        
        # Analyze thresholds
        print("\nAnalyzing thresholds...")
        mpileup_results = self.analyze_thresholds(svm_probs, mpileup_labels)
        beagle_results = self.analyze_thresholds(svm_probs, beagle_labels)
        
        # Plot results
        self.plot_threshold_analysis(mpileup_results, "mpileup")
        self.plot_threshold_analysis(beagle_results, "beagle")
        
        # Save results
        self.save_results(mpileup_results, beagle_results)
        
        return mpileup_results, beagle_results

    def analyze_thresholds(self, probabilities: np.ndarray, true_labels: List[int]) -> pd.DataFrame:
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.05, 0.95, 0.05)
        results = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            metrics = {
                'threshold': threshold,
                'accuracy': accuracy_score(true_labels, predictions),
                'precision': precision_score(true_labels, predictions, zero_division=0),
                'recall': recall_score(true_labels, predictions, zero_division=0),
                'f1': f1_score(true_labels, predictions, zero_division=0)
            }
            results.append(metrics)
        
        return pd.DataFrame(results)

    def plot_threshold_analysis(self, results_df: pd.DataFrame, reference: str):
        """Plot performance metrics across thresholds"""
        plt.figure(figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        colors = ['blue', 'green', 'red', 'purple']
        
        for metric, color in zip(metrics, colors):
            plt.plot(results_df['threshold'], results_df[metric], 
                    marker='o', label=metric.capitalize(), color=color)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'SVM Performance vs {reference.title()}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add best threshold line
        best_idx = results_df['f1'].idxmax()
        best_threshold = results_df.loc[best_idx, 'threshold']
        plt.axvline(x=best_threshold, color='black', linestyle='--',
                   label=f'Best threshold: {best_threshold:.2f}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'threshold_analysis_{reference}.png'))
        plt.close()

    def save_results(self, mpileup_results: pd.DataFrame, beagle_results: pd.DataFrame):
        """Save analysis results"""
        results_file = os.path.join(self.output_dir, 'comparison_results.txt')
        
        with open(results_file, 'w') as f:
            f.write(f"SVM Performance Analysis\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            if self.section_id:
                f.write(f"Section: {self.section_id}\n")
            f.write("\nBest Thresholds:\n")
            
            # Mpileup comparison
            best_mpileup = mpileup_results.loc[mpileup_results['f1'].idxmax()]
            f.write("\nVs Mpileup:\n")
            for metric in ['threshold', 'accuracy', 'precision', 'recall', 'f1']:
                f.write(f"{metric}: {best_mpileup[metric]:.3f}\n")
            
            # Beagle comparison
            best_beagle = beagle_results.loc[beagle_results['f1'].idxmax()]
            f.write("\nVs Beagle:\n")
            for metric in ['threshold', 'accuracy', 'precision', 'recall', 'f1']:
                f.write(f"{metric}: {best_beagle[metric]:.3f}\n")

def main():
    parser = argparse.ArgumentParser(description="Compare SVM and Beagle performance")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    args = parser.parse_args()
    
    evaluator = SVMEvaluator(args.dataset, args.quality_filter, args.section_id)
    mpileup_results, beagle_results = evaluator.evaluate_performance()
    
    # Print summary results
    print("\nComparison Results Summary:")
    print("\nBest Threshold vs Mpileup:")
    best_mpileup = mpileup_results.loc[mpileup_results['f1'].idxmax()]
    for metric in ['threshold', 'accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric}: {best_mpileup[metric]:.3f}")
    
    print("\nBest Threshold vs Beagle:")
    best_beagle = beagle_results.loc[beagle_results['f1'].idxmax()]
    for metric in ['threshold', 'accuracy', 'precision', 'recall', 'f1']:
        print(f"{metric}: {best_beagle[metric]:.3f}")

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC with section:
# python scripts/postprocess/run_svm_compare_to_beagle.py --dataset DLPFC --section_id 151507 --quality-filter baseQ0mapQ0

# For 10X_BC_6.5MM (no section):
# python scripts/postprocess/run_svm_compare_to_beagle.py --dataset 10X_BC_6.5MM --quality-filter baseQ0mapQ0

# For P4_TUMOR with section:
# python scripts/postprocess/run_svm_compare_to_beagle.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0