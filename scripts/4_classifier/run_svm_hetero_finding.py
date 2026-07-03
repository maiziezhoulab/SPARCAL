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
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from tqdm import tqdm
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Import dataset configurations from your existing code
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

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "BEAGLE_JAR": "beagle.27Jul16.86a.jar",
    "JAVA": "src/jdk-11.0.2/bin/java",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/yuqi/snv_calling/apps/tabix",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "THOUSAND_GENOME_DIR": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/"
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

    def extract_features(self, vcf_path: str, dataset_name: str) -> pd.DataFrame:
        """Extract features from VCF file with dataset-specific handling"""
        features = []
        # Important fields in the Monopogen paper: 'DP', 'VDB', 'RPB', 'MQB', 'BQB', 'SGB', 'BAF'
        # All numeric fields we want to extract
        numeric_fields = ['DP', 'QS', 'VDB', 'RPB', 'MQB', 'BQB', 'MQSB', 'SGB', 'MQ0F',
                         'BAF', 'GQ']
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
                    
                    # Extract I16 values if needed (can be useful for advanced models)
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
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None,
                max_training_samples: int = 90000):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.max_training_samples = max_training_samples
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.validate_dataset_config()
        self.setup_paths()
        self.setup_environment()

    def setup_environment(self):
        """Setup environment variables for the pipeline."""
        apps_dir = PATH_CONFIG['APPS_DIR']
        os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
        os.environ['LD_LIBRARY_PATH'] = new_ld_path
        
        print(f"Environment set up with PATH including: {apps_dir}")
        print(f"LD_LIBRARY_PATH includes: {apps_dir}")
        
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
        self.consecutive_denovo_vcf = os.path.join(
            section_path,
            "output_VCFs/SeqErrModel",
            self.quality_filter,
            "consecutive_denovo.vcf.gz"
        )
        
        # Default testing VCF (sequence_no_error.vcf.gz)
        self.seq_no_error_vcf = os.path.join(
            section_path, "output_VCFs/SeqErrModel",
            self.quality_filter, "sequence_no_error.vcf.gz"
        )
        
        # Output directory
        self.output_dir = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Model results directory
        self.model_dir = os.path.join(self.output_dir, "results")
        os.makedirs(self.model_dir, exist_ok=True)

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
        print(f"Collected: {key}: {len(variants)}")
        return variants

    def collect_seq_error_variants(self) -> List[Dict]:
        """Collect sequence error variants from VCF"""
        variants = []
        with gzip.open(self.seq_error_vcf, 'rt') as f:
            header_lines = []
            for line in f:
                if line.startswith('#'):
                    header_lines.append(line)
                    continue
                variants.append({
                    'line': line,
                    'header_lines': header_lines
                })
        return variants

    def collect_consecutive_denovo_variants(self, vcf_path: str) -> List[Dict]:
        """Collect consecutive de novo variants from VCF file for negative training set."""
        variants = []
        with gzip.open(vcf_path, 'rt') as f:
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
        # extend 0/1 to 1/1 and 1/1 to 0/1
        positive_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("0/1", "1/1")
        ))
        positive_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("1/1", "0/1")
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
        # negative_variants.extend(self.collect_consecutive_denovo_variants(self.consecutive_denovo_vcf))

        # Sample down large datasets if needed
        total_variants = len(positive_variants) + len(negative_variants)
        if total_variants > self.max_training_samples:
            print(f"\nTotal variants ({total_variants}) exceeds maximum training size ({self.max_training_samples})")
            print("Sampling down the training set...")
            
            # Calculate sample sizes to maintain class balance ratio
            total_ratio = total_variants / self.max_training_samples
            pos_sample_size = min(len(positive_variants), int(len(positive_variants) / total_ratio))
            neg_sample_size = min(len(negative_variants), self.max_training_samples - pos_sample_size)
            
            # Re-adjust positive sample size if needed
            if pos_sample_size + neg_sample_size < self.max_training_samples:
                pos_sample_size = min(len(positive_variants), self.max_training_samples - neg_sample_size)
            
            # Sample variants
            if len(positive_variants) > pos_sample_size:
                positive_variants = random_sample(positive_variants, pos_sample_size)
            if len(negative_variants) > neg_sample_size:
                negative_variants = random_sample(negative_variants, neg_sample_size)
                
            print(f"Sampled training set: {len(positive_variants)} positive, {len(negative_variants)} negative")

        # Save training sets
        self.save_variants(positive_variants, "positive_training.vcf.gz")
        self.save_variants(negative_variants, "negative_training.vcf.gz")
        
        # Store variants for later processing
        self.positive_variants = positive_variants
        self.negative_variants = negative_variants
        
        print(f"\nTraining Set Statistics:")
        print(f"Positive examples: {len(positive_variants):,}")
        print(f"  - 0/1 -> 0/1: {len([v for v in positive_variants if v.get('original_gt')=='0/1' and v.get('new_gt')=='0/1']):,}")
        print(f"  - 1/1 -> 1/1: {len([v for v in positive_variants if v.get('original_gt')=='1/1'and v.get('new_gt')=='1/1']):,}")
        print(f"  - 0/1 -> 1/1: {len([v for v in positive_variants if v.get('original_gt')=='0/1'and v.get('new_gt')=='1/1']):,}")
        print(f"  - 1/1 -> 0/1: {len([v for v in positive_variants if v.get('original_gt')=='1/1'and v.get('new_gt')=='0/1']):,}")
        print(f"\nNegative examples: {len(negative_variants):,}")
        print(f"  - 0/1 -> 0/0: {len([v for v in negative_variants if v.get('original_gt')=='0/1']):,}")
        print(f"  - 1/1 -> 0/0: {len([v for v in negative_variants if v.get('original_gt')=='1/1']):,}")
        print(f"  - Sequence errors: {len([v for v in negative_variants if not v.get('original_gt')]):,}")

    # def save_variants(self, variants: List[Dict], filename: str):
    #     """Save variants to VCF file"""
    #     output_path = os.path.join(self.output_dir, filename)
    #     with gzip.open(output_path, 'wt') as f:
    #         # Write header from first variant that has header lines
    #         header_written = False
    #         for variant in variants:
    #             if 'header_lines' in variant:
    #                 for header in variant['header_lines']:
    #                     f.write(header)
    #                 header_written = True
    #                 break
            
    #         # If no header found in variants, write minimal header
    #         if not header_written:
    #             f.write("##fileformat=VCFv4.2\n")
    #             f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            
    #         # Write variants
    #         for variant in variants:
    #             f.write(variant['line'])
    def save_variants(self, variants: List[Dict], filename: str): # TODO: this is too complicated.
        """
        Save variants to a properly formatted, sorted, indexed, and compressed VCF file.
        
        Args:
            variants: List of variant dictionaries
            filename: Name of the output file (will be created in the output directory)
        """
        # Create temporary uncompressed VCF for initial writing
        output_path = os.path.join(self.output_dir, filename)
        temp_vcf = output_path.replace('.vcf.gz', '.temp.vcf')
        
        # Ensure any parent directories exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            print(f"Creating VCF file with {len(variants)} variants...")
            
            # Check if variants is empty
            if not variants:
                print(f"Warning: No variants to save for {filename}")
                # Create an empty VCF with just headers
                with open(temp_vcf, 'w') as f:
                    f.write("##fileformat=VCFv4.2\n")
                    reference_path = REFERENCE_CONFIGS.get(
                        DATASET_CONFIGS.get(self.dataset_name, {}).get('reference', ''), 
                        {}
                    ).get('path', '')
                    f.write(f"##reference={reference_path}\n")
                    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                
                # Compress directly with bgzip
                bgzip_cmd = f"{PATH_CONFIG['BGZIP']} -c {temp_vcf} > {output_path}"
                subprocess.run(bgzip_cmd, shell=True, check=True)
                # Index with tabix
                tabix_cmd = f"tabix -p vcf {output_path}"
                subprocess.run(tabix_cmd, shell=True, check=True)
                
                # Cleanup and return
                if os.path.exists(temp_vcf):
                    os.remove(temp_vcf)
                return
            
            # Write variants to temporary file
            with open(temp_vcf, 'w') as f:
                # Write header from first variant that has header lines
                header_written = False
                for variant in variants:
                    if 'header_lines' in variant and variant['header_lines']:
                        for header in variant['header_lines']:
                            f.write(header)
                        header_written = True
                        break
                
                # If no header found in variants, write minimal header
                if not header_written:
                    f.write("##fileformat=VCFv4.2\n")
                    reference_path = REFERENCE_CONFIGS.get(
                        DATASET_CONFIGS.get(self.dataset_name, {}).get('reference', ''), 
                        {}
                    ).get('path', '')
                    f.write(f"##reference={reference_path}\n")
                    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                
                # Write variants
                for variant in variants:
                    if 'line' in variant:
                        f.write(variant['line'])
            
            # First validate that the VCF is properly formatted
            print(f"Validating VCF file...")
            try:
                # Use grep to check if the file has valid header
                check_header_cmd = f"grep -c '^#CHROM' {temp_vcf}"
                header_result = subprocess.run(check_header_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if header_result.returncode != 0 or int(header_result.stdout.decode().strip()) == 0:
                    print(f"Warning: VCF file does not have a valid header. Adding minimal header.")
                    # Create a new temp file with proper header
                    temp_vcf_with_header = temp_vcf + ".header"
                    with open(temp_vcf_with_header, 'w') as f_out:
                        f_out.write("##fileformat=VCFv4.2\n")
                        reference_path = REFERENCE_CONFIGS.get(
                            DATASET_CONFIGS.get(self.dataset_name, {}).get('reference', ''), 
                            {}
                        ).get('path', '')
                        f_out.write(f"##reference={reference_path}\n")
                        f_out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                        
                        # Copy content from original temp file
                        with open(temp_vcf, 'r') as f_in:
                            for line in f_in:
                                if not line.startswith('#'):  # Skip any malformed headers
                                    f_out.write(line)
                    
                    # Replace original temp file
                    os.rename(temp_vcf_with_header, temp_vcf)
            except Exception as e:
                print(f"Warning: Error during VCF validation: {str(e)}")
            
            # Try using bcftools directly to compress and index instead of sort first
            print(f"Compressing and indexing VCF file...")
            try:
                # Compress with bgzip
                bgzip_cmd = f"{PATH_CONFIG['BGZIP']} -c {temp_vcf} > {output_path}"
                subprocess.run(bgzip_cmd, shell=True, check=True)
                
                # Index with tabix
                tabix_cmd = f"tabix -p vcf {output_path}"
                subprocess.run(tabix_cmd, shell=True, check=True)
                
                print(f"Successfully created, compressed, and indexed: {output_path}")
            except Exception as e:
                print(f"Warning: Error during compression and indexing: {str(e)}")
                print(f"Trying alternative approach...")
                
                # Alternative approach: use VCF fix tools
                try:
                    # Create a fixed VCF with vcf-validator (if available)
                    vcf_fix_cmd = f"vcf-validator -f {temp_vcf} > {temp_vcf}.fixed"
                    try:
                        subprocess.run(vcf_fix_cmd, shell=True, check=False)
                        if os.path.exists(f"{temp_vcf}.fixed") and os.path.getsize(f"{temp_vcf}.fixed") > 0:
                            temp_vcf = f"{temp_vcf}.fixed"
                    except:
                        print("Warning: vcf-validator not installed or failed. Continuing without fixing.")
                    
                    # Sort the VCF by position
                    print("Manually sorting VCF...")
                    sorted_vcf = temp_vcf + ".sorted"
                    
                    # Extract header
                    header_cmd = f"grep '^#' {temp_vcf} > {sorted_vcf}"
                    subprocess.run(header_cmd, shell=True, check=True)
                    
                    # Sort non-header lines
                    sort_cmd = f"grep -v '^#' {temp_vcf} | sort -k1,1 -k2,2n >> {sorted_vcf}"
                    subprocess.run(sort_cmd, shell=True, check=True)
                    
                    # Compress with bgzip
                    print(f"Compressing VCF file with bgzip...")
                    bgzip_cmd = f"{PATH_CONFIG['BGZIP']} -c {sorted_vcf} > {output_path}"
                    subprocess.run(bgzip_cmd, shell=True, check=True)
                    
                    # Index with tabix
                    print(f"Indexing VCF file with tabix...")
                    tabix_cmd = f"tabix -p vcf {output_path}"
                    subprocess.run(tabix_cmd, shell=True, check=True)
                    
                    print(f"Successfully created, sorted, compressed, and indexed using alternative method: {output_path}")
                except Exception as e2:
                    print(f"Error with alternative approach: {str(e2)}")
                    print("Saving uncompressed VCF as fallback...")
                    
                    # Save uncompressed as fallback
                    uncompressed_output = output_path.replace('.vcf.gz', '.vcf')
                    with open(uncompressed_output, 'w') as f_out:
                        with open(temp_vcf, 'r') as f_in:
                            f_out.write(f_in.read())
                    print(f"Saved uncompressed VCF as fallback: {uncompressed_output}")
                    raise
            
            # Clean up temporary files
            for temp_file in [temp_vcf, f"{temp_vcf}.fixed", temp_vcf + ".sorted"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            print(f"Error saving variants to VCF: {str(e)}")
            # Try to cleanup any temp files
            for temp_file in [temp_vcf, f"{temp_vcf}.fixed", temp_vcf + ".sorted"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            raise

# Add a helper function for random sampling
def random_sample(variants: List[Dict], sample_size: int) -> List[Dict]:
    """
    Randomly sample a subset of variants without replacement
    """
    if len(variants) <= sample_size:
        return variants
    
    # Use numpy for efficient random sampling
    indices = np.random.choice(len(variants), size=sample_size, replace=False)
    return [variants[i] for i in indices]

class SVMWithPCA:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", 
                section_id: str = None, max_training_samples: int = 90000):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.max_training_samples = max_training_samples
        self.feature_extractor = SVMFeatureExtractor()
        self.builder = TrainingSetBuilder(dataset_name, quality_filter, section_id, max_training_samples)
        self.pca = None
        self.svm = None
        self.scaler = StandardScaler()
        self.explained_variance_ratios = []
        self.n_components = None
        self.model_loaded = False
        
    def build_training_sets(self):
        """Build training sets using the builder"""
        self.builder.build_training_sets()
        
    def extract_and_preprocess_features(self):
        """Extract features from positive and negative training sets"""
        print("\nExtracting features from training data...")
        
        # Extract features from positive examples
        print("Processing positive examples...")
        positive_features = self.feature_extractor.extract_features(
            os.path.join(self.builder.output_dir, "positive_training.vcf.gz"),
            self.dataset_name
        )
        positive_features['label'] = 1
        
        # Extract features from negative examples
        print("Processing negative examples...")
        negative_features = self.feature_extractor.extract_features(
            os.path.join(self.builder.output_dir, "negative_training.vcf.gz"),
            self.dataset_name
        )
        negative_features['label'] = 0
        
        # Combine datasets
        features = pd.concat([positive_features, negative_features])
        
        # Handle missing values
        features = features.fillna(features.mean())
        
        # Split features and labels
        X = features.drop('label', axis=1)
        y = features['label']
        
        # Store feature column names for later reference
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_val, y_train, y_val, X, y
    
    def find_optimal_pca_components(self, X_train, variance_threshold=0.95):
        """Find optimal number of PCA components based on explained variance"""
        # Initialize PCA with maximum number of components
        max_components = min(X_train.shape[0], X_train.shape[1])
        pca_full = PCA(n_components=max_components)
        pca_full.fit(X_train)
        
        # Store explained variance ratios
        self.explained_variance_ratios = pca_full.explained_variance_ratio_
        
        # Calculate cumulative explained variance
        cumulative_variance = np.cumsum(self.explained_variance_ratios)
        
        # Find number of components needed to explain threshold variance
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        print(f"Number of components needed to explain {variance_threshold*100}% of variance: {n_components}")
        
        # Store and return number of components
        self.n_components = n_components
        return n_components
    
    def plot_explained_variance(self, save_path=None):
        """Plot explained variance ratio and cumulative explained variance"""
        plt.figure(figsize=(12, 5))
        
        # Plot explained variance ratio
        plt.subplot(1, 2, 1)
        plt.bar(range(1, len(self.explained_variance_ratios) + 1), 
                self.explained_variance_ratios, alpha=0.7)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance by Component')
        plt.tight_layout()
        
        # Plot cumulative explained variance
        plt.subplot(1, 2, 2)
        cumulative_variance = np.cumsum(self.explained_variance_ratios)
        plt.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'o-', color='green')
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
        plt.axvline(x=self.n_components, color='gray', linestyle='--', 
                   label=f'{self.n_components} Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Cumulative Explained Variance')
        plt.legend()
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def plot_confidence_distribution(self, X_train, X_val, y_train, y_val, save_path=None):
        """
        Plot confidence score distribution for training and validation sets
        """
        if not self.model_loaded and self.svm is None:
            print("Model not loaded or trained yet.")
            return
            
        print("Analyzing confidence distribution...")
            
        # Apply PCA transformation
        X_train_pca = self.pca.transform(X_train)
        X_val_pca = self.pca.transform(X_val)
        
        # Get probability estimates
        train_probs = self.svm.predict_proba(X_train_pca)[:, 1]
        val_probs = self.svm.predict_proba(X_val_pca)[:, 1]
        
        # Separate by true class
        train_pos_probs = train_probs[y_train == 1]
        train_neg_probs = train_probs[y_train == 0]
        val_pos_probs = val_probs[y_val == 1]
        val_neg_probs = val_probs[y_val == 0]
        
        # Create figure with 4 subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot training set distributions
        bins = np.linspace(0, 1, 50)
        axs[0, 0].hist(train_pos_probs, bins=bins, alpha=0.7, color='green', 
                      label=f'Positive (n={len(train_pos_probs)})')
        axs[0, 0].hist(train_neg_probs, bins=bins, alpha=0.7, color='red', 
                      label=f'Negative (n={len(train_neg_probs)})')
        axs[0, 0].set_title('Training Set Confidence Distribution')
        axs[0, 0].set_xlabel('SVM Confidence Score')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].legend()
        axs[0, 0].grid(alpha=0.3)
        
        # Plot validation set distributions
        axs[0, 1].hist(val_pos_probs, bins=bins, alpha=0.7, color='green', 
                      label=f'Positive (n={len(val_pos_probs)})')
        axs[0, 1].hist(val_neg_probs, bins=bins, alpha=0.7, color='red', 
                      label=f'Negative (n={len(val_neg_probs)})')
        axs[0, 1].set_title('Validation Set Confidence Distribution')
        axs[0, 1].set_xlabel('SVM Confidence Score')
        axs[0, 1].set_ylabel('Count')
        axs[0, 1].legend()
        axs[0, 1].grid(alpha=0.3)
        
        # Plot density estimates for training set
        from scipy.stats import gaussian_kde
        if len(train_pos_probs) > 1:
            kde_pos = gaussian_kde(train_pos_probs)
            kde_neg = gaussian_kde(train_neg_probs)
            x_vals = np.linspace(0, 1, 1000)
            axs[1, 0].plot(x_vals, kde_pos(x_vals), color='green', label='Positive')
            axs[1, 0].plot(x_vals, kde_neg(x_vals), color='red', label='Negative')
            axs[1, 0].set_title('Training Set Confidence Density')
            axs[1, 0].set_xlabel('SVM Confidence Score')
            axs[1, 0].set_ylabel('Density')
            axs[1, 0].legend()
            axs[1, 0].grid(alpha=0.3)
        
        # Plot ROC curve with confidence thresholds
        if len(val_pos_probs) > 0 and len(val_neg_probs) > 0:
            fpr, tpr, thresholds = roc_curve(y_val, val_probs)
            roc_auc = roc_auc_score(y_val, val_probs)
            
            # Plot ROC curve
            axs[1, 1].plot(fpr, tpr, color='blue', lw=2, 
                          label=f'ROC curve (AUC = {roc_auc:.3f})')
            axs[1, 1].plot([0, 1], [0, 1], color='gray', linestyle='--')
            
            # Add threshold markers
            threshold_markers = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for threshold in threshold_markers:
                # Find closest threshold value
                idx = (np.abs(thresholds - threshold)).argmin()
                if idx < len(fpr):
                    axs[1, 1].plot(fpr[idx], tpr[idx], 'o', markersize=5, 
                                  label=f'Threshold = {threshold:.1f}')
            
            axs[1, 1].set_xlim([0.0, 1.0])
            axs[1, 1].set_ylim([0.0, 1.05])
            axs[1, 1].set_xlabel('False Positive Rate')
            axs[1, 1].set_ylabel('True Positive Rate')
            axs[1, 1].set_title('ROC Curve with Confidence Thresholds')
            axs[1, 1].legend(loc="lower right", fontsize=8)
            axs[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
        
        # Print summary statistics
        print("\nConfidence Distribution Summary:")
        print("Training Set:")
        print(f"  Positive samples: mean = {np.mean(train_pos_probs):.3f}, std = {np.std(train_pos_probs):.3f}")
        print(f"  Negative samples: mean = {np.mean(train_neg_probs):.3f}, std = {np.std(train_neg_probs):.3f}")
        print("Validation Set:")
        print(f"  Positive samples: mean = {np.mean(val_pos_probs):.3f}, std = {np.std(val_pos_probs):.3f}")
        print(f"  Negative samples: mean = {np.mean(val_neg_probs):.3f}, std = {np.std(val_neg_probs):.3f}")
    
    def plot_pca_projections(self, X_train, X_val, y_train, y_val, sample_size=500, save_path=None):
        """
        Project a random sample of data points onto the first two PCA components
        """
        if self.pca is None:
            print("PCA model not trained yet.")
            return
            
        print("Creating PCA projection visualization...")
        
        # Function to randomly sample with stratification
        def stratified_sample(X, y, n):
            # Make sure we're working with numpy arrays
            X = np.asarray(X)
            y = np.asarray(y)
            
            # Find positive and negative sample indices
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            # Handle case where we don't have enough samples
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                print(f"Warning: One class has no samples. Positive: {len(pos_idx)}, Negative: {len(neg_idx)}")
                # Return all samples if we have fewer than requested
                if len(X) <= n:
                    return X, y
                else:
                    # Random sample without stratification
                    all_idx = np.random.choice(len(X), size=min(n, len(X)), replace=False)
                    return X[all_idx], y[all_idx]
            
            # Calculate how many samples to take from each class
            n_pos = min(len(pos_idx), n // 2)
            n_neg = min(len(neg_idx), n - n_pos)  # Adjust neg to fill remaining spots
            
            # If one class doesn't have enough samples, take more from the other
            if n_pos < n // 2 and len(neg_idx) > n_neg:
                n_neg = min(len(neg_idx), n - n_pos)
            elif n_neg < (n - n // 2) and len(pos_idx) > n_pos:
                n_pos = min(len(pos_idx), n - n_neg)
                
            # Check if we need to sample at all
            total_samples = n_pos + n_neg
            if total_samples == 0:
                return np.empty((0, X.shape[1])), np.empty(0)
                
            # Sample with replacement if we don't have enough samples
            replace_pos = len(pos_idx) < n_pos
            replace_neg = len(neg_idx) < n_neg
            
            # Sample indices with replacement if necessary
            if n_pos > 0:
                sampled_pos = np.random.choice(pos_idx, size=n_pos, replace=replace_pos)
            else:
                sampled_pos = np.array([], dtype=int)
                
            if n_neg > 0:
                sampled_neg = np.random.choice(neg_idx, size=n_neg, replace=replace_neg)
            else:
                sampled_neg = np.array([], dtype=int)
            
            # Combine indices
            sampled_idx = np.concatenate([sampled_pos, sampled_neg])
            
            # Check if indices are valid
            if np.max(sampled_idx) >= len(X) or np.min(sampled_idx) < 0:
                print(f"Warning: Invalid indices generated. Max index: {np.max(sampled_idx)}, Array length: {len(X)}")
                valid_idx = sampled_idx[(sampled_idx >= 0) & (sampled_idx < len(X))]
                if len(valid_idx) == 0:
                    return np.empty((0, X.shape[1])), np.empty(0)
                return X[valid_idx], y[valid_idx]
                
            return X[sampled_idx], y[sampled_idx]
        
        # Get PCA projections
        try:
            X_train_pca = self.pca.transform(X_train)
            X_val_pca = self.pca.transform(X_val)
            
            # Sample data points
            X_train_sample, y_train_sample = stratified_sample(X_train_pca, y_train, sample_size)
            X_val_sample, y_val_sample = stratified_sample(X_val_pca, y_val, sample_size)
            
            # Check if we have samples to plot
            if len(X_train_sample) == 0 and len(X_val_sample) == 0:
                print("No samples available for PCA projection. Skipping plot.")
                return
            
            # Create figure with subplots based on available data
            if len(X_train_sample) > 0 and len(X_val_sample) > 0:
                fig, axs = plt.subplots(1, 2, figsize=(16, 7))
            else:
                fig, axs = plt.subplots(1, 1, figsize=(8, 7))
                axs = [axs]
            
            # Plot training set projections if available
            plot_idx = 0
            if len(X_train_sample) > 0:
                pos_mask = y_train_sample == 1
                neg_mask = y_train_sample == 0
                
                # Check if we have both positive and negative samples
                if np.any(pos_mask):
                    axs[plot_idx].scatter(X_train_sample[pos_mask, 0], X_train_sample[pos_mask, 1], 
                                        c='green', marker='o', alpha=0.6, label='Positive')
                if np.any(neg_mask):
                    axs[plot_idx].scatter(X_train_sample[neg_mask, 0], X_train_sample[neg_mask, 1], 
                                        c='red', marker='x', alpha=0.6, label='Negative')
                
                var1 = self.explained_variance_ratios[0] * 100
                var2 = self.explained_variance_ratios[1] * 100
                axs[plot_idx].set_xlabel(f'PC1 ({var1:.1f}% variance)')
                axs[plot_idx].set_ylabel(f'PC2 ({var2:.1f}% variance)')
                axs[plot_idx].set_title(f'PCA Projection - Training Set (n={len(X_train_sample)})')
                axs[plot_idx].legend()
                axs[plot_idx].grid(alpha=0.3)
                
                # Plot decision boundary if we have an SVM model and enough samples
                if self.svm is not None and len(X_train_sample) > 1:
                    try:
                        # Create a mesh grid for the first two PCA components
                        x_min, x_max = X_train_sample[:, 0].min() - 1, X_train_sample[:, 0].max() + 1
                        y_min, y_max = X_train_sample[:, 1].min() - 1, X_train_sample[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                                          np.arange(y_min, y_max, (y_max - y_min) / 100))
                        
                        # Create feature vectors for the grid points
                        grid = np.c_[xx.ravel(), yy.ravel()]
                        
                        # Pad with zeros for remaining PCA components if needed
                        if self.n_components > 2:
                            padding = np.zeros((grid.shape[0], self.n_components - 2))
                            grid = np.hstack([grid, padding])
                        
                        # Get predictions for the grid points
                        Z = self.svm.predict_proba(grid)[:, 1].reshape(xx.shape)
                        
                        # Plot decision boundary contour
                        contour = axs[plot_idx].contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
                        plt.colorbar(contour, ax=axs[plot_idx], label='Probability')
                    except Exception as e:
                        print(f"Warning: Could not plot decision boundary: {str(e)}")
                
                plot_idx += 1
            
            # Plot validation set projections if available
            if plot_idx < len(axs) and len(X_val_sample) > 0:
                pos_mask = y_val_sample == 1
                neg_mask = y_val_sample == 0
                
                # Check if we have both positive and negative samples
                if np.any(pos_mask):
                    axs[plot_idx].scatter(X_val_sample[pos_mask, 0], X_val_sample[pos_mask, 1], 
                                        c='green', marker='o', alpha=0.6, label='Positive')
                if np.any(neg_mask):
                    axs[plot_idx].scatter(X_val_sample[neg_mask, 0], X_val_sample[neg_mask, 1], 
                                        c='red', marker='x', alpha=0.6, label='Negative')
                
                var1 = self.explained_variance_ratios[0] * 100
                var2 = self.explained_variance_ratios[1] * 100
                axs[plot_idx].set_xlabel(f'PC1 ({var1:.1f}% variance)')
                axs[plot_idx].set_ylabel(f'PC2 ({var2:.1f}% variance)')
                axs[plot_idx].set_title(f'PCA Projection - Validation Set (n={len(X_val_sample)})')
                axs[plot_idx].legend()
                axs[plot_idx].grid(alpha=0.3)
                
                # Plot decision boundary if we have an SVM model and enough samples
                if self.svm is not None and len(X_val_sample) > 1:
                    try:
                        # Create a mesh grid for the first two PCA components
                        x_min, x_max = X_val_sample[:, 0].min() - 1, X_val_sample[:, 0].max() + 1
                        y_min, y_max = X_val_sample[:, 1].min() - 1, X_val_sample[:, 1].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100),
                                          np.arange(y_min, y_max, (y_max - y_min) / 100))
                        
                        # Create feature vectors for the grid points
                        grid = np.c_[xx.ravel(), yy.ravel()]
                        
                        # Pad with zeros for remaining PCA components if needed
                        if self.n_components > 2:
                            padding = np.zeros((grid.shape[0], self.n_components - 2))
                            grid = np.hstack([grid, padding])
                        
                        # Get predictions for the grid points
                        Z = self.svm.predict_proba(grid)[:, 1].reshape(xx.shape)
                        
                        # Plot decision boundary contour
                        contour = axs[plot_idx].contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.coolwarm)
                        plt.colorbar(contour, ax=axs[plot_idx], label='Probability')
                    except Exception as e:
                        print(f"Warning: Could not plot decision boundary: {str(e)}")
            
            plt.tight_layout()
            
            # Save or show plot
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()
            plt.close()
            
        except Exception as e:
            print(f"Error generating PCA projection: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def train_model(self, variance_threshold=0.95, grid_search=False):
        """Train SVM model with PCA dimensionality reduction"""
        # Extract and preprocess features
        X_train, X_val, y_train, y_val, X_full, y_full = self.extract_and_preprocess_features()
        
        # Find optimal number of PCA components
        n_components = self.find_optimal_pca_components(X_train, variance_threshold)
        
        # Initialize and fit PCA with optimal number of components
        self.pca = PCA(n_components=n_components)
        X_train_pca = self.pca.fit_transform(X_train)
        X_val_pca = self.pca.transform(X_val)
        
        print(f"\nTraining SVM on {n_components} principal components...")
        
        # Grid search for optimal hyperparameters (optional)
        if grid_search:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf']
            }
            
            svm = GridSearchCV(
                SVC(probability=True, class_weight='balanced'),
                param_grid,
                cv=5,
                scoring='f1',
                verbose=1,
                n_jobs=-1
            )
            
            svm.fit(X_train_pca, y_train)
            self.svm = svm.best_estimator_
            print(f"Best parameters: {svm.best_params_}")
        else:
            # Train SVM with default parameters
            self.svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
            self.svm.fit(X_train_pca, y_train)
        
        # Evaluate model
        train_score = self.svm.score(X_train_pca, y_train)
        val_score = self.svm.score(X_val_pca, y_val)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        
        # Generate classification report
        y_pred = self.svm.predict(X_val_pca)
        class_report = classification_report(y_val, y_pred, target_names=['Negative', 'Positive'])
        print("\nClassification Report:")
        print(class_report)
        
        # Generate ROC curve
        y_proba = self.svm.predict_proba(X_val_pca)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_proba)
        roc_auc = roc_auc_score(y_val, y_proba)
        
        # Generate PR curve
        precision, recall, _ = precision_recall_curve(y_val, y_proba)
        pr_auc = auc(recall, precision)
        
        # Create and save additional visualizations
        # 1. Confidence distribution analysis
        self.plot_confidence_distribution(
            X_train, X_val, y_train, y_val, 
            save_path=os.path.join(self.builder.model_dir, "confidence_distribution.png")
        )
        
        # 2. PCA projection visualization
        self.plot_pca_projections(
            X_train, X_val, y_train, y_val, sample_size=500,
            save_path=os.path.join(self.builder.model_dir, "pca_projection.png")
        )
        
        # Save model, PCA, scaler, and other metadata
        model_data = {
            'svm_model': self.svm,
            'pca_model': self.pca,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_components': n_components,
            'training_accuracy': train_score,
            'validation_accuracy': val_score,
            'explained_variance_ratios': self.explained_variance_ratios,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        model_path = os.path.join(self.builder.model_dir, "svm_pca_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metrics
        metrics_path = os.path.join(self.builder.model_dir, "svm_pca_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"SVM with PCA Model Performance:\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            if self.section_id:
                f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n\n")
            f.write(f"PCA Components: {n_components}\n")
            f.write(f"Training Accuracy: {train_score:.3f}\n")
            f.write(f"Validation Accuracy: {val_score:.3f}\n")
            f.write(f"ROC AUC: {roc_auc:.3f}\n")
            f.write(f"PR AUC: {pr_auc:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(class_report)
        
        # Plot and save explained variance
        self.plot_explained_variance(save_path=os.path.join(self.builder.model_dir, "pca_explained_variance.png"))
        
        # Plot and save ROC and PR curves
        self.plot_curves(fpr, tpr, roc_auc, precision, recall, pr_auc, 
                        save_path=os.path.join(self.builder.model_dir, "svm_pca_curves.png"))
        
        print(f"\nModel and metrics saved to: {self.builder.model_dir}")
        return model_data
    
    def plot_curves(self, fpr, tpr, roc_auc, precision, recall, pr_auc, save_path=None):
        """Plot ROC and Precision-Recall curves"""
        plt.figure(figsize=(12, 5))
        
        # Plot ROC curve
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Plot Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def load_model(self):
        """Load saved model from disk"""
        model_path = os.path.join(self.builder.model_dir, "svm_pca_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.svm = model_data['svm_model']
        self.pca = model_data['pca_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.n_components = model_data['n_components']
        self.explained_variance_ratios = model_data['explained_variance_ratios']
        self.model_loaded = True
        
        print(f"Loaded model with {self.n_components} PCA components")
        print(f"Training accuracy: {model_data['training_accuracy']:.3f}")
        print(f"Validation accuracy: {model_data['validation_accuracy']:.3f}")
        
        return model_data
    
    def apply_model_to_vcf(self, input_vcf, output_vcf=None):
        """Apply trained model to a VCF file"""
        if not output_vcf:
            output_vcf = os.path.join(self.builder.model_dir, "svm_predictions.vcf.gz")
        
        # Load model if not already loaded
        if not self.model_loaded:
            try:
                self.load_model()
            except FileNotFoundError:
                print("No saved model found. Please train a model first.")
                return
        
        print(f"\nApplying SVM model to {input_vcf}...")
        
        # Extract features from input VCF
        features = self.feature_extractor.extract_features(input_vcf, self.dataset_name)
        features = features.fillna(features.mean())
        
        # Ensure features have the same columns as training data
        for col in self.feature_columns:
            if col not in features.columns:
                features[col] = 0.0
        
        # Select only columns used in training
        X = features[self.feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Apply PCA transformation
        X_pca = self.pca.transform(X_scaled)
        
        # Get predictions and probabilities
        predictions = self.svm.predict(X_pca)
        probabilities = self.svm.predict_proba(X_pca)[:, 1]
        
        # Read and modify VCF
        with gzip.open(input_vcf, 'rt') as f_in, gzip.open(output_vcf, 'wt') as f_out:
            # Copy header and add SVM fields
            header_written = False
            for line in f_in:
                if line.startswith('#'):
                    if line.startswith('#CHROM'):
                        # Add new INFO fields before the header line
                        f_out.write('##INFO=<ID=SVM_PRED,Number=1,Type=Integer,'
                                'Description="SVM prediction (1=true variant, 0=error)">\n')
                        f_out.write('##INFO=<ID=SVM_PROB,Number=1,Type=Float,'
                                'Description="SVM probability of being a true variant">\n')
                    f_out.write(line)
                    if line.startswith('#CHROM'):
                        header_written = True
                        break
            
            # Process variants
            idx = 0
            for line in f_in:
                if idx < len(predictions):
                    fields = line.strip().split('\t')
                    info = fields[7]
                    
                    # Add SVM predictions to INFO field
                    info += f";SVM_PRED={predictions[idx]};SVM_PROB={probabilities[idx]:.4f}"
                    fields[7] = info
                    
                    # Write modified line
                    f_out.write('\t'.join(fields) + '\n')
                    idx += 1
                else:
                    # If we've used all predictions but there are more variants, just copy as is
                    f_out.write(line)
        
        print(f"Applied model to {idx} variants")
        print(f"Output saved to: {output_vcf}")
        
        # Create filtered high-confidence VCF
        self.create_filtered_vcf(output_vcf, probabilities)
    
    def create_filtered_vcf(self, input_vcf, probabilities, threshold=0.5):
        """Create filtered VCF with only high-confidence variants using bgzip compression"""
        # Define output paths
        base_filename = os.path.basename(input_vcf).replace('.vcf.gz', '')
        high_conf_vcf = os.path.join(self.builder.model_dir, "high_confidence.vcf")
        low_conf_vcf = os.path.join(self.builder.model_dir, "low_confidence.vcf")
        high_conf_vcf_gz = high_conf_vcf + ".gz"
        low_conf_vcf_gz = low_conf_vcf + ".gz"
        
        print(f"\nFiltering variants with confidence threshold: {threshold}")
        
        # First create uncompressed VCF files
        with gzip.open(input_vcf, 'rt') as f_in, \
            open(high_conf_vcf, 'wt') as f_high, \
            open(low_conf_vcf, 'wt') as f_low:
            
            # Copy header
            header_lines = []
            for line in f_in:
                if line.startswith('#'):
                    header_lines.append(line)
                    f_high.write(line)
                    f_low.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            # Process variants
            high_count = low_count = 0
            idx = 0
            
            for line in f_in:
                if idx < len(probabilities):
                    prob = probabilities[idx]
                    if prob >= threshold:
                        f_high.write(line)
                        high_count += 1
                    else:
                        f_low.write(line)
                        low_count += 1
                    idx += 1
                else:
                    # If we've used all probabilities but there are more variants, put in low confidence
                    f_low.write(line)
                    low_count += 1
        
        # Use bgzip to compress the files
        try:
            # Compress high confidence VCF
            subprocess.run(['/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip', '-f', high_conf_vcf], check=True)
            # Index with tabix
            subprocess.run(['tabix', '-p', 'vcf', high_conf_vcf_gz], check=True)
            
            # Compress low confidence VCF
            subprocess.run(['/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip', '-f', low_conf_vcf], check=True)
            # Index with tabix
            subprocess.run(['tabix', '-p', 'vcf', low_conf_vcf_gz], check=True)
            
            print(f"High-confidence variants: {high_count}")
            print(f"Low-confidence variants: {low_count}")
            print(f"High-confidence VCF: {high_conf_vcf_gz}")
            print(f"Low-confidence VCF: {low_conf_vcf_gz}")
            
            return high_conf_vcf_gz, low_conf_vcf_gz
        
        except subprocess.CalledProcessError as e:
            print(f"Error compressing or indexing VCF files: {e}")
            return None, None


def main():
    """Main function to parse arguments and run the workflow"""
    parser = argparse.ArgumentParser(description="SVM with PCA for variant classification")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", 
                      help="Section ID (required for some datasets)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    parser.add_argument("--variance-threshold", type=float, default=0.95,
                      help="Variance threshold for PCA component selection (default: 0.95)")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                      help="Confidence threshold for high-confidence variants (default: 0.5)")
    parser.add_argument("--grid-search", action="store_true",
                      help="Perform grid search for SVM hyperparameters")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip training and use existing model")
    parser.add_argument("--input-vcf",
                      help="Input VCF file to classify (if skipping training)")
    parser.add_argument("--max-training-samples", type=int, default=90000,
                      help="Maximum number of training samples to use (default: 90000)")
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
    
    # Initialize SVM with PCA
    svm_pca = SVMWithPCA(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        max_training_samples=args.max_training_samples
    )
    
    # Print configuration
    print("\nSVM with PCA Configuration:")
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Variance Threshold: {args.variance_threshold}")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Max Training Samples: {args.max_training_samples}")
    if args.grid_search:
        print("Grid Search: Enabled")
    
    if not args.skip_training:
        # Build training sets
        svm_pca.build_training_sets()
        
        # Train model
        svm_pca.train_model(
            variance_threshold=args.variance_threshold,
            grid_search=args.grid_search
        )
        
        # Apply model to input VCF (or to the sequence_no_error.vcf.gz file if no input specified)
        input_vcf = args.input_vcf if args.input_vcf else svm_pca.builder.seq_no_error_vcf
        print(f"Applying trained model to: {input_vcf}")
        svm_pca.apply_model_to_vcf(input_vcf)
    else:
        # Load existing model
        try:
            svm_pca.load_model()
            
            # Apply model to input VCF (or to the sequence_no_error.vcf.gz file if no input specified)
            input_vcf = args.input_vcf if args.input_vcf else svm_pca.builder.seq_no_error_vcf
            print(f"Applying saved model to: {input_vcf}")
            svm_pca.apply_model_to_vcf(input_vcf)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please train a model first or provide the correct path to an existing model.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

# Run the script with the following command:
# On DLPFC dataset with section ID 151507
# python scripts/postprocess/run_svm_hetero_finding.py --dataset DLPFC --section_id 151507 --variance-threshold 0.9 --quality-filter baseQ0mapQ0

# On P4/P6 section 1/2, apply model without training
# python scripts/3_classifier_prep/run_svm_hetero_finding.py --dataset P4_TUMOR --section_id 1 --max-training-samples 900000 
# python scripts/3_classifier_prep/run_svm_hetero_finding.py --dataset P4_TUMOR --section_id 2 --max-training-samples 90000
# python scripts/postprocess/run_svm_hetero_finding.py --dataset P6_TUMOR --section_id 1 --skip-training
# python scripts/postprocess/run_svm_hetero_finding.py --dataset P6_TUMOR --section_id 2 --skip-training
