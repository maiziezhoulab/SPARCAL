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
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve, auc, roc_curve, roc_auc_score, confusion_matrix
from sklearn.metrics import f1_score, make_scorer
from sklearn.decomposition import PCA
from tqdm import tqdm
import subprocess
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available, XGBoost model type will not be available")

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
}       # BUGS HERE!!!

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
        "reference": "CHR_PREFIX"
    },
    "OVAR_P5": {
        # GRCh38, chr prefix — merged VCF is chr-prefixed, so use CHR_PREFIX.
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "CHR_PREFIX"
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

@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str

class FeatureExtractor:
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
        self.base_dir = "/data/maiziezhou_lab/leiy4/snv_calling"
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
        
        # Default testing VCF (sequence_no_error.vcf.gz)
        self.seq_no_error_vcf = os.path.join(
            section_path, "output_VCFs/SeqErrModel",
            self.quality_filter, "sequence_no_error.vcf.gz"
        )
        
        # Output directory
        self.output_dir = os.path.join(
            section_path, "output_VCFs/Classifier",
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

    def build_training_sets(self):
        """Build positive and negative training sets"""
        print("\nLoading transition metrics...")
        shifted_metrics = self.load_pickle_results(self.shifted_results)
        stable_metrics = self.load_pickle_results(self.stable_results)
        
        print("\nCollecting training examples...")
        
        # Initialize 3-class training sets
        homozygous_variants = []
        heterozygous_variants = []
        novar_variants = []
        
        # Collect homozygous examples (1/1)
        # Stable 1/1
        homozygous_variants.extend(self.collect_variants_from_metrics(
            stable_metrics, ("1/1", "1/1")
        ))
        # Transition 0/1 to 1/1
        homozygous_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("0/1", "1/1")
        ))
        
        # Collect heterozygous examples (0/1)
        # Stable 0/1
        heterozygous_variants.extend(self.collect_variants_from_metrics(
            stable_metrics, ("0/1", "0/1")
        ))
        # Transition 1/1 to 0/1
        heterozygous_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("1/1", "0/1")
        ))
        
        # Collect no-variance examples (0/0)
        # Transition 0/1 to 0/0
        novar_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("0/1", "0/0")
        ))
        # Transition 1/1 to 0/0
        novar_variants.extend(self.collect_variants_from_metrics(
            shifted_metrics, ("1/1", "0/0")
        ))
        # Sequence errors
        novar_variants.extend(self.collect_seq_error_variants())
        
        # Label the variants with their class
        for variant in homozygous_variants:
            variant['class'] = 'homozygous'  # Class 2
        for variant in heterozygous_variants:
            variant['class'] = 'heterozygous'  # Class 1
        for variant in novar_variants:
            variant['class'] = 'no_variance'  # Class 0
            
        # Sample down large datasets if needed
        total_variants = len(homozygous_variants) + len(heterozygous_variants) + len(novar_variants)
        if total_variants > self.max_training_samples:
            print(f"\nTotal variants ({total_variants}) exceeds maximum training size ({self.max_training_samples})")
            print("Sampling down the training set...")
            
            # Calculate sample sizes to maintain class balance ratio
            total_ratio = total_variants / self.max_training_samples
            homo_sample_size = min(len(homozygous_variants), int(len(homozygous_variants) / total_ratio))
            hetero_sample_size = min(len(heterozygous_variants), int(len(heterozygous_variants) / total_ratio))
            novar_sample_size = min(len(novar_variants), self.max_training_samples - homo_sample_size - hetero_sample_size)
            
            # Re-adjust sample sizes if needed
            if homo_sample_size + hetero_sample_size + novar_sample_size < self.max_training_samples:
                remaining = self.max_training_samples - homo_sample_size - hetero_sample_size - novar_sample_size
                # Distribute remaining capacity proportionally
                total_remaining = len(homozygous_variants) - homo_sample_size + len(heterozygous_variants) - hetero_sample_size + len(novar_variants) - novar_sample_size
                if total_remaining > 0:
                    homo_extra = int(remaining * (len(homozygous_variants) - homo_sample_size) / total_remaining)
                    hetero_extra = int(remaining * (len(heterozygous_variants) - hetero_sample_size) / total_remaining)
                    novar_extra = remaining - homo_extra - hetero_extra
                    
                    homo_sample_size += homo_extra
                    hetero_sample_size += hetero_extra
                    novar_sample_size += novar_extra
            
            # Sample variants
            if len(homozygous_variants) > homo_sample_size:
                homozygous_variants = random_sample(homozygous_variants, homo_sample_size)
            if len(heterozygous_variants) > hetero_sample_size:
                heterozygous_variants = random_sample(heterozygous_variants, hetero_sample_size)
            if len(novar_variants) > novar_sample_size:
                novar_variants = random_sample(novar_variants, novar_sample_size)
                
            print(f"Sampled training set: {len(homozygous_variants)} homozygous, {len(heterozygous_variants)} heterozygous, {len(novar_variants)} no variance")

        # Save training sets by class
        self.save_variants(homozygous_variants, "homozygous_training.vcf.gz")
        self.save_variants(heterozygous_variants, "heterozygous_training.vcf.gz")
        self.save_variants(novar_variants, "no_variance_training.vcf.gz")
        
        # Also save combined training set with all variants
        all_variants = homozygous_variants + heterozygous_variants + novar_variants
        self.save_variants(all_variants, "all_classes_training.vcf.gz")
        
        # Store variants for later processing
        self.homozygous_variants = homozygous_variants
        self.heterozygous_variants = heterozygous_variants
        self.novar_variants = novar_variants
        
        print(f"\nTraining Set Statistics (3-Class):")
        print(f"Homozygous examples (1/1): {len(homozygous_variants):,}")
        print(f"  - 1/1 -> 1/1 (stable): {len([v for v in homozygous_variants if v.get('original_gt')=='1/1' and v.get('new_gt')=='1/1']):,}")
        print(f"  - 0/1 -> 1/1 (transition): {len([v for v in homozygous_variants if v.get('original_gt')=='0/1' and v.get('new_gt')=='1/1']):,}")
        
        print(f"\nHeterozygous examples (0/1): {len(heterozygous_variants):,}")
        print(f"  - 0/1 -> 0/1 (stable): {len([v for v in heterozygous_variants if v.get('original_gt')=='0/1' and v.get('new_gt')=='0/1']):,}")
        print(f"  - 1/1 -> 0/1 (transition): {len([v for v in heterozygous_variants if v.get('original_gt')=='1/1' and v.get('new_gt')=='0/1']):,}")
        
        print(f"\nNo variance examples (0/0): {len(novar_variants):,}")
        print(f"  - 0/1 -> 0/0: {len([v for v in novar_variants if v.get('original_gt')=='0/1' and v.get('new_gt')=='0/0']):,}")
        print(f"  - 1/1 -> 0/0: {len([v for v in novar_variants if v.get('original_gt')=='1/1' and v.get('new_gt')=='0/0']):,}")
        print(f"  - Sequence errors: {len([v for v in novar_variants if not v.get('original_gt')]):,}")


    # def save_variants(self, variants: List[Dict], filename: str):
    #     """
    #     Save variants to a properly formatted, sorted, indexed, and compressed VCF file.
        
    #     Args:
    #         variants: List of variant dictionaries
    #         filename: Name of the output file (will be created in the output directory)
    #     """
    #     output_path = os.path.join(self.output_dir, filename)
    #     temp_vcf = output_path.replace('.vcf.gz', '.temp.vcf')
        
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
    #     try:
    #         print(f"Creating VCF file with {len(variants)} variants...")
            
    #         # Handle empty variant list
    #         if not variants:
    #             print(f"Warning: No variants to save for {filename}")
    #             with open(temp_vcf, 'w') as f:
    #                 f.write("##fileformat=VCFv4.2\n")
    #                 reference_path = REFERENCE_CONFIGS.get(
    #                     DATASET_CONFIGS.get(self.dataset_name, {}).get('reference', ''), 
    #                     {}
    #                 ).get('path', '')
    #                 f.write(f"##reference={reference_path}\n")
    #                 f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                
    #             # Sort, compress, and index even empty file
    #             sorted_vcf = temp_vcf + ".sorted.vcf.gz"
    #             subprocess.run([PATH_CONFIG['BCFTOOLS'], 'sort', '-Oz', '-o', sorted_vcf, temp_vcf], check=True)
    #             subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', sorted_vcf], check=True)
    #             subprocess.run(['mv', sorted_vcf, output_path], check=True)
    #             subprocess.run(['mv', f"{sorted_vcf}.tbi", f"{output_path}.tbi"], check=True)
                
    #             if os.path.exists(temp_vcf):
    #                 os.remove(temp_vcf)
    #             return
            
    #         # Build comprehensive header from ALL sources
    #         print(f"Building comprehensive header from all variant sources...")
    #         header_lines = []
    #         header_set = set()  # Track unique header lines
            
    #         # Get base header from input VCF
    #         try:
    #             with gzip.open(self.input_vcf, 'rt') as f:
    #                 for line in f:
    #                     if line.startswith('#'):
    #                         if line not in header_set:
    #                             header_lines.append(line)
    #                             header_set.add(line)
    #                     else:
    #                         break
    #         except Exception as e:
    #             print(f"Warning: Could not extract header from {self.input_vcf}: {e}")
            
    #         # Get header from sequence error VCF (if it exists and is being used)
    #         if os.path.exists(self.seq_error_vcf):
    #             try:
    #                 with gzip.open(self.seq_error_vcf, 'rt') as f:
    #                     for line in f:
    #                         if line.startswith('#'):
    #                             # Skip duplicate headers
    #                             if line.startswith('##') and line not in header_set:
    #                                 # Insert before the #CHROM line
    #                                 if header_lines and header_lines[-1].startswith('#CHROM'):
    #                                     header_lines.insert(-1, line)
    #                                 else:
    #                                     header_lines.append(line)
    #                                 header_set.add(line)
    #                         else:
    #                             break
    #             except Exception as e:
    #                 print(f"Warning: Could not extract header from {self.seq_error_vcf}: {e}")
            
    #         # Add any missing essential field definitions
    #         essential_headers = [
    #             "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Raw read depth\">\n",
    #             "##INFO=<ID=I16,Number=16,Type=Float,Description=\"Auxiliary tag used for calling\">\n",
    #             "##INFO=<ID=QS,Number=R,Type=Float,Description=\"Auxiliary tag used for calling\">\n",
    #             "##INFO=<ID=VDB,Number=1,Type=Float,Description=\"Variant Distance Bias\">\n",
    #             "##INFO=<ID=RPB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Read Position Bias\">\n",
    #             "##INFO=<ID=MQB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Mapping Quality Bias\">\n",
    #             "##INFO=<ID=BQB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Base Quality Bias\">\n",
    #             "##INFO=<ID=MQSB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Mapping Quality vs Strand Bias\">\n",
    #             "##INFO=<ID=SGB,Number=1,Type=Float,Description=\"Segregation based metric\">\n",
    #             "##INFO=<ID=MQ0F,Number=1,Type=Float,Description=\"Fraction of MQ0 reads\">\n",
    #             "##FILTER=<ID=LowGQ,Description=\"Low genotype quality\">\n",
    #             "##FILTER=<ID=DiscordantBAF,Description=\"Discordant B-allele frequency\">\n",
    #             "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
    #             "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n",
    #             "##FORMAT=<ID=BAF,Number=1,Type=Float,Description=\"B-allele frequency\">\n",
    #             "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods\">\n",
    #         ]
            
    #         # Find where to insert (before #CHROM line)
    #         chrom_idx = None
    #         for i, line in enumerate(header_lines):
    #             if line.startswith('#CHROM'):
    #                 chrom_idx = i
    #                 break
            
    #         # Add missing essential headers
    #         for essential in essential_headers:
    #             if essential not in header_set:
    #                 if chrom_idx is not None:
    #                     header_lines.insert(chrom_idx, essential)
    #                     chrom_idx += 1
    #                 else:
    #                     header_lines.append(essential)
    #                 header_set.add(essential)
            
    #         # Ensure we have a #CHROM line at the end
    #         if not any(line.startswith('#CHROM') for line in header_lines):
    #             header_lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
            
    #         # Write to temporary VCF
    #         with open(temp_vcf, 'w') as f:
    #             # Write comprehensive header
    #             for header in header_lines:
    #                 f.write(header)
                
    #             # Write variants (may be unsorted)
    #             for variant in variants:
    #                 if 'line' in variant:
    #                     f.write(variant['line'])
            
    #         # ALWAYS sort with bcftools before indexing
    #         print(f"Sorting VCF file with bcftools...")
    #         sorted_vcf = temp_vcf + ".sorted.vcf.gz"
            
    #         # Sort
    #         sort_cmd = [PATH_CONFIG['BCFTOOLS'], 'sort', '-Oz', '-o', sorted_vcf, temp_vcf]
    #         result = subprocess.run(sort_cmd, capture_output=True, text=True)
    #         if result.returncode != 0:
    #             print(f"bcftools sort stderr: {result.stderr}")
    #             raise subprocess.CalledProcessError(result.returncode, sort_cmd, result.stdout, result.stderr)
            
    #         # Index
    #         print(f"Indexing sorted VCF file with tabix...")
    #         tabix_cmd = [PATH_CONFIG['TABIX'], '-p', 'vcf', sorted_vcf]
    #         subprocess.run(tabix_cmd, check=True)
            
    #         # Move to final location
    #         subprocess.run(['mv', sorted_vcf, output_path], check=True)
    #         subprocess.run(['mv', f"{sorted_vcf}.tbi", f"{output_path}.tbi"], check=True)
            
    #         print(f"Successfully created, sorted, and indexed: {output_path}")
            
    #     except Exception as e:
    #         print(f"Error saving variants to VCF: {str(e)}")
    #         raise
    #     finally:
    #         # Clean up temporary files
    #         for temp_file in [temp_vcf, temp_vcf + ".sorted.vcf.gz"]:
    #             if os.path.exists(temp_file):
    #                 os.remove(temp_file)

    def save_variants(self, variants: List[Dict], filename: str):
        """
        Save variants to a properly formatted, sorted, indexed, and compressed VCF file.

        Args:
            variants: List of variant dictionaries
            filename: Name of the output file (will be created in the output directory)
        """
        output_path = os.path.join(self.output_dir, filename)
        temp_vcf = output_path.replace('.vcf.gz', '.temp.vcf')
        sorted_vcf = temp_vcf + ".sorted.vcf.gz"

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            print(f"Creating VCF file with {len(variants)} variants...")

            # --- Build header from self.input_vcf ---
            print("Building comprehensive header from all variant sources...")
            header_lines = []
            header_set = set()

            try:
                with gzip.open(self.input_vcf, 'rt') as f:
                    for line in f:
                        if line.startswith('#'):
                            if line not in header_set:
                                header_lines.append(line)
                                header_set.add(line)
                        else:
                            break
            except Exception as e:
                print(f"Warning: Could not extract header from {self.input_vcf}: {e}")

            # Merge in any extra ##meta lines from seq_error_vcf (e.g. SEQ_ERROR_MODEL INFO)
            if os.path.exists(self.seq_error_vcf):
                try:
                    with gzip.open(self.seq_error_vcf, 'rt') as f:
                        for line in f:
                            if not line.startswith('#'):
                                break
                            if line.startswith('##') and line not in header_set:
                                chrom_idx = next(
                                    (i for i, l in enumerate(header_lines) if l.startswith('#CHROM')),
                                    len(header_lines)
                                )
                                header_lines.insert(chrom_idx, line)
                                header_set.add(line)
                except Exception as e:
                    print(f"Warning: Could not extract header from {self.seq_error_vcf}: {e}")

            # Add missing essential FORMAT/INFO/FILTER definitions
            essential_headers = [
                "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Raw read depth\">\n",
                "##INFO=<ID=I16,Number=16,Type=Float,Description=\"Auxiliary tag used for calling\">\n",
                "##INFO=<ID=QS,Number=R,Type=Float,Description=\"Auxiliary tag used for calling\">\n",
                "##INFO=<ID=VDB,Number=1,Type=Float,Description=\"Variant Distance Bias\">\n",
                "##INFO=<ID=RPB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Read Position Bias\">\n",
                "##INFO=<ID=MQB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Mapping Quality Bias\">\n",
                "##INFO=<ID=BQB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Base Quality Bias\">\n",
                "##INFO=<ID=MQSB,Number=1,Type=Float,Description=\"Mann-Whitney U test of Mapping Quality vs Strand Bias\">\n",
                "##INFO=<ID=SGB,Number=1,Type=Float,Description=\"Segregation based metric\">\n",
                "##INFO=<ID=MQ0F,Number=1,Type=Float,Description=\"Fraction of MQ0 reads\">\n",
                "##FILTER=<ID=LowGQ,Description=\"Low genotype quality\">\n",
                "##FILTER=<ID=DiscordantBAF,Description=\"Discordant B-allele frequency\">\n",
                "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n",
                "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n",
                "##FORMAT=<ID=BAF,Number=1,Type=Float,Description=\"B-allele frequency\">\n",
                "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods\">\n",
            ]
            for essential in essential_headers:
                if essential not in header_set:
                    chrom_idx = next(
                        (i for i, l in enumerate(header_lines) if l.startswith('#CHROM')),
                        len(header_lines)
                    )
                    header_lines.insert(chrom_idx, essential)
                    header_set.add(essential)

            # Ensure #CHROM line exists
            if not any(l.startswith('#CHROM') for l in header_lines):
                header_lines.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")

            # --- Chr prefix mismatch correction ---
            # Detect whether variants use chr prefix by peeking at the first data line
            uses_chr_prefix = False
            for v in variants:
                line = v.get('line', '')
                if line and not line.startswith('#'):
                    uses_chr_prefix = line.startswith('chr')
                    break

            header_has_chr_contigs = any(
                l.startswith('##contig') and '<ID=chr' in l for l in header_lines
            )

            if uses_chr_prefix and not header_has_chr_contigs:
                print("Chr prefix mismatch detected: adding 'chr' prefix to contig IDs in header.")
                header_lines = [
                    l.replace('##contig=<ID=', '##contig=<ID=chr', 1)
                    if l.startswith('##contig=<ID=') and not l.startswith('##contig=<ID=chr')
                    else l
                    for l in header_lines
                ]
            elif not uses_chr_prefix and header_has_chr_contigs:
                print("Chr prefix mismatch detected: removing 'chr' prefix from contig IDs in header.")
                header_lines = [
                    l.replace('##contig=<ID=chr', '##contig=<ID=', 1)
                    if l.startswith('##contig=<ID=chr')
                    else l
                    for l in header_lines
                ]

            # --- Handle empty variant list ---
            if not variants:
                print(f"Warning: No variants to save for {filename}")
                with open(temp_vcf, 'w') as f:
                    for line in header_lines:
                        f.write(line)
                sort_result = subprocess.run(
                    [PATH_CONFIG['BCFTOOLS'], 'sort', '-Oz', '-o', sorted_vcf, temp_vcf],
                    capture_output=True, text=True
                )
                if sort_result.returncode != 0:
                    print(f"bcftools sort stderr: {sort_result.stderr}")
                    raise subprocess.CalledProcessError(sort_result.returncode, 'bcftools sort')
                subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', sorted_vcf], check=True)
                subprocess.run(['mv', sorted_vcf, output_path], check=True)
                subprocess.run(['mv', f"{sorted_vcf}.tbi", f"{output_path}.tbi"], check=True)
                return

            # --- Write temp VCF ---
            with open(temp_vcf, 'w') as f:
                for line in header_lines:
                    f.write(line)
                for variant in variants:
                    if 'line' in variant:
                        f.write(variant['line'])

            # --- Sort, compress, index ---
            print("Sorting VCF file with bcftools...")
            sort_result = subprocess.run(
                [PATH_CONFIG['BCFTOOLS'], 'sort', '-Oz', '-o', sorted_vcf, temp_vcf],
                capture_output=True, text=True
            )
            if sort_result.returncode != 0:
                print(f"bcftools sort stderr: {sort_result.stderr}")
                raise subprocess.CalledProcessError(sort_result.returncode, 'bcftools sort',
                                                    sort_result.stdout, sort_result.stderr)

            print("Indexing sorted VCF file with tabix...")
            subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', sorted_vcf], check=True)
            subprocess.run(['mv', sorted_vcf, output_path], check=True)
            subprocess.run(['mv', f"{sorted_vcf}.tbi", f"{output_path}.tbi"], check=True)

            print(f"Successfully created, sorted, and indexed: {output_path}")

        except Exception as e:
            print(f"Error saving variants to VCF: {str(e)}")
            raise
        finally:
            for temp_file in [temp_vcf, sorted_vcf]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
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



# Custom scoring function that only considers heterozygous and homozygous classes
def variant_only_f1(y_true, y_pred, label_encoder=None):
    """
    Calculate F1 score considering only heterozygous and homozygous classes
    
    Args:
        y_true: True labels (encoded integers or class names)
        y_pred: Predicted labels (encoded integers or class names)
        label_encoder: Optional label encoder to convert class names to indices
    
    Returns:
        F1 score averaged over heterozygous and homozygous classes only
    """
    # Get class indices for heterozygous and homozygous
    if label_encoder is not None:
        hetero_idx = label_encoder.transform(['heterozygous'])[0]
        homo_idx = label_encoder.transform(['homozygous'])[0]
        # Convert string labels if provided
        if isinstance(y_true[0], str):
            y_true = label_encoder.transform(y_true)
        if isinstance(y_pred[0], str):
            y_pred = label_encoder.transform(y_pred)
    else:
        # Default indices based on our class order ['no_variance', 'heterozygous', 'homozygous']
        hetero_idx = 1
        homo_idx = 2
    
    # Create a mask for the relevant classes (only heterozygous and homozygous)
    relevant_indices = (y_true == hetero_idx) | (y_true == homo_idx)
    
    # If no relevant samples, return 0
    if not np.any(relevant_indices):
        return 0.0
    
    # Calculate F1 score only for the relevant classes
    return f1_score(
        y_true[relevant_indices], 
        y_pred[relevant_indices], 
        labels=[hetero_idx, homo_idx],
        average='macro'
    )

# Create scikit-learn compatible scorer
variant_f1_scorer = make_scorer(variant_only_f1)

class ModelTrainer:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", 
                section_id: str = None, max_training_samples: int = 90000):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.max_training_samples = max_training_samples
        self.feature_extractor = FeatureExtractor()
        self.builder = TrainingSetBuilder(dataset_name, quality_filter, section_id, max_training_samples)
        self.use_pca = False
        self.pca = None
        self.model = None
        self.scaler = StandardScaler()
        self.explained_variance_ratios = []
        self.n_components = None
        self.model_loaded = False
        self.model_type = "svm"  # Default model type
        self.class_labels = ['no_variance', 'heterozygous', 'homozygous']
        self.label_encoder = LabelEncoder()
        self.multiclass = True  # Set multiclass classification as default
        
    def build_training_sets(self):
        """Build training sets using the builder"""
        self.builder.build_training_sets()
        
    def extract_and_preprocess_features(self):
        """Extract features from all three class training sets"""
        print("\nExtracting features from training data...")
        
        try:
            # Extract features from homozygous examples
            print("Processing homozygous examples...")
            homozygous_features = self.feature_extractor.extract_features(
                os.path.join(self.builder.output_dir, "homozygous_training.vcf.gz"),
                self.dataset_name
            )
            homozygous_features['class'] = 'homozygous'
            
            # Extract features from heterozygous examples
            print("Processing heterozygous examples...")
            heterozygous_features = self.feature_extractor.extract_features(
                os.path.join(self.builder.output_dir, "heterozygous_training.vcf.gz"),
                self.dataset_name
            )
            heterozygous_features['class'] = 'heterozygous'
            
            # Extract features from no variance examples
            print("Processing no variance examples...")
            novar_features = self.feature_extractor.extract_features(
                os.path.join(self.builder.output_dir, "no_variance_training.vcf.gz"),
                self.dataset_name
            )
            novar_features['class'] = 'no_variance'
            
            # Combine datasets
            features = pd.concat([homozygous_features, heterozygous_features, novar_features])
            
            # Separate class labels before handling missing values
            y_raw = features['class']
            X = features.drop('class', axis=1)
            
            # Handle missing values (only on numeric features)
            X = X.fillna(X.mean())
            
            # Encode class labels
            self.label_encoder.fit(self.class_labels)
            y = self.label_encoder.transform(y_raw)
            
            # Store feature column names for later reference
            self.feature_columns = X.columns.tolist()
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            return X_train, X_val, y_train, y_val, X, y
            
        except Exception as e:
            print(f"Error extracting features: {str(e)}")
            raise
    
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
        if not self.model_loaded and self.model is None:
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
    
    def calculate_variant_f1(self, y_true, y_pred):
        """Calculate F1 score for variant classes only (heterozygous and homozygous)"""
        return variant_only_f1(y_true, y_pred, self.label_encoder)
            
    def train_model(self, model_type='svm', variance_threshold=0.95, use_pca=False, grid_search=False):
        """
        Train machine learning model for 3-class classification
        
        Args:
            model_type: Type of model to train ('svm', 'random_forest', 'xgboost', 'neural_network')
            variance_threshold: Threshold for explained variance when using PCA
            use_pca: Whether to use PCA dimensionality reduction
            grid_search: Whether to perform grid search for hyperparameter optimization
        """
        # Store model type
        self.model_type = model_type
        self.use_pca = use_pca
        
        # Extract and preprocess features
        X_train, X_val, y_train, y_val, X_full, y_full = self.extract_and_preprocess_features()
        
        # Apply PCA if requested
        if use_pca:
            # Find optimal number of PCA components
            n_components = self.find_optimal_pca_components(X_train, variance_threshold)
            
            # Initialize and fit PCA with optimal number of components
            self.pca = PCA(n_components=n_components)
            X_train_transformed = self.pca.fit_transform(X_train)
            X_val_transformed = self.pca.transform(X_val)
            
            print(f"\nTraining {model_type.upper()} model on {n_components} principal components...")
        else:
            X_train_transformed = X_train
            X_val_transformed = X_val
            print(f"\nTraining {model_type.upper()} model directly on features...")
            
        # Initialize model based on type, using multi-class configuration
        if model_type == 'svm':
            self.model = SVC(kernel='rbf', probability=True, decision_function_shape='ovr', random_state=42)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost is not available. Please install xgboost.")
            self.model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, random_state=42)
        elif model_type == 'neural_network':
            self.model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Adjust param_grid for multiclass classification
        if grid_search:
            param_grid = {}
            if model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                    'kernel': ['rbf']
                }
            elif model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            elif model_type == 'xgboost' and XGB_AVAILABLE:
                param_grid = {
                    'learning_rate': [0.01, 0.1, 0.2],
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7]
                }
            elif model_type == 'neural_network':
                param_grid = {
                    'hidden_layer_sizes': [(32, 16), (64, 32), (128, 64)],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate_init': [0.001, 0.01, 0.1]
                }
                
            if param_grid:
                print(f"Performing grid search for {model_type} hyperparameters...")
                # Use custom scoring function that only considers variant classes
                grid = GridSearchCV(
                    self.model, param_grid, cv=5, 
                    scoring=variant_f1_scorer,  # Use custom scorer instead of 'f1_macro'
                    verbose=1, n_jobs=-1
                )
                grid.fit(X_train_transformed, y_train)
                self.model = grid.best_estimator_
                print(f"Best parameters: {grid.best_params_}")
        
        # Train model
        self.model.fit(X_train_transformed, y_train)
            
        # Evaluate model
        train_score = self.model.score(X_train_transformed, y_train)
        val_score = self.model.score(X_val_transformed, y_val)
        
        # Generate predictions
        y_train_pred = self.model.predict(X_train_transformed)
        y_val_pred = self.model.predict(X_val_transformed)
        
        # Calculate F1 score for variant classes only (heterozygous and homozygous)
        variant_train_f1 = self.calculate_variant_f1(y_train, y_train_pred)
        variant_val_f1 = self.calculate_variant_f1(y_val, y_val_pred)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Validation accuracy: {val_score:.3f}")
        print(f"Training variant-only F1: {variant_train_f1:.3f}")
        print(f"Validation variant-only F1: {variant_val_f1:.3f}")
        
        # Print feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            if use_pca:
                print("\nFeature ranking (on PCA components):")
                for i, idx in enumerate(indices[:10]):  # Show top 10 components
                    print(f"{i+1}. PCA component {idx+1} ({importances[idx]:.4f})")
            else:
                print("\nFeature ranking:")
                for i, idx in enumerate(indices[:10]):  # Show top 10 features
                    print(f"{i+1}. {self.feature_columns[idx]} ({importances[idx]:.4f})")
        
        # Generate classification report
        y_pred = self.model.predict(X_val_transformed)
        class_report = classification_report(y_val, y_pred, 
                                           target_names=self.class_labels)
        print("\nClassification Report:")
        print(class_report)
        
        # Calculate and print class-specific metrics
        # Get indices for each class
        hetero_idx = self.label_encoder.transform(['heterozygous'])[0]
        homo_idx = self.label_encoder.transform(['homozygous'])[0]
        
        # Calculate class-specific metrics
        hetero_precision = precision_score(y_val, y_pred, labels=[hetero_idx], average='macro')
        hetero_recall = recall_score(y_val, y_pred, labels=[hetero_idx], average='macro')
        hetero_f1 = f1_score(y_val, y_pred, labels=[hetero_idx], average='macro')
        
        homo_precision = precision_score(y_val, y_pred, labels=[homo_idx], average='macro')
        homo_recall = recall_score(y_val, y_pred, labels=[homo_idx], average='macro')
        homo_f1 = f1_score(y_val, y_pred, labels=[homo_idx], average='macro')
        
        print("\nClass-specific metrics:")
        print(f"Heterozygous - Precision: {hetero_precision:.3f}, Recall: {hetero_recall:.3f}, F1: {hetero_f1:.3f}")
        print(f"Homozygous - Precision: {homo_precision:.3f}, Recall: {homo_recall:.3f}, F1: {homo_f1:.3f}")
        
        # Calculate and plot confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        self.plot_confusion_matrix(conf_matrix, save_path=os.path.join(self.builder.model_dir, f"{model_type}_confusion_matrix.png"))
        
        # Save model, PCA, scaler, and other metadata
        model_data = {
            'model': self.model,
            'model_type': model_type,
            'pca_model': self.pca if use_pca else None,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_components': self.n_components if use_pca else None,
            'training_accuracy': train_score,
            'validation_accuracy': val_score,
            'variant_train_f1': variant_train_f1,
            'variant_val_f1': variant_val_f1,
            'explained_variance_ratios': self.explained_variance_ratios if use_pca else None,
            'label_encoder': self.label_encoder,
            'class_labels': self.class_labels,
            'use_pca': use_pca,
            'multiclass': True
        }
        
        # Save model to disk
        model_path = os.path.join(self.builder.model_dir, f"{model_type}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save metrics
        metrics_path = os.path.join(self.builder.model_dir, f"{model_type}_metrics.txt")
        with open(metrics_path, 'w') as f:
            f.write(f"{model_type.upper()} Model Performance (Multi-class):\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            if self.section_id:
                f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n\n")
            if use_pca:
                f.write(f"PCA Components: {self.n_components}\n")
            f.write(f"Training Accuracy: {train_score:.3f}\n")
            f.write(f"Validation Accuracy: {val_score:.3f}\n")
            f.write(f"Training variant-only F1: {variant_train_f1:.3f}\n")
            f.write(f"Validation variant-only F1: {variant_val_f1:.3f}\n\n")
            f.write("Class-specific metrics:\n")
            f.write(f"Heterozygous - Precision: {hetero_precision:.3f}, Recall: {hetero_recall:.3f}, F1: {hetero_f1:.3f}\n")
            f.write(f"Homozygous - Precision: {homo_precision:.3f}, Recall: {homo_recall:.3f}, F1: {homo_f1:.3f}\n\n")
            f.write("Classification Report:\n")
            f.write(class_report)
            f.write("\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
        
        print(f"\nModel and metrics saved to: {self.builder.model_dir}")
        return model_data

    def plot_confusion_matrix(self, conf_matrix, save_path=None):
        """Plot confusion matrix for multi-class classification"""
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add labels and values
        classes = self.class_labels
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], fmt),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        
        # Save or show plot
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    def load_model(self, model_type='svm'):
        """Load saved model from disk"""
        model_path = os.path.join(self.builder.model_dir, f"{model_type}_model.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading {model_type} model from {model_path}")
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data.get('model_type', model_type)
        self.pca = model_data['pca_model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.use_pca = model_data.get('use_pca', self.pca is not None)
        self.n_components = model_data['n_components'] if self.use_pca else None
        self.explained_variance_ratios = model_data.get('explained_variance_ratios', [])
        
        # Load multi-class specific data
        self.multiclass = model_data.get('multiclass', True)  # Default to multi-class
        self.label_encoder = model_data.get('label_encoder', self.label_encoder)
        self.class_labels = model_data.get('class_labels', self.class_labels)
        
        self.model_loaded = True
        
        print(f"Loaded {self.model_type} model")
        if self.multiclass:
            print(f"Model type: Multi-class classification ({len(self.class_labels)} classes)")
        if self.use_pca:
            print(f"Model uses PCA with {self.n_components} components")
        print(f"Training accuracy: {model_data['training_accuracy']:.3f}")
        print(f"Validation accuracy: {model_data['validation_accuracy']:.3f}")
        
        # Print variant-only F1 if available
        if 'variant_val_f1' in model_data:
            print(f"Validation variant-only F1: {model_data['variant_val_f1']:.3f}")
        
        return model_data
    
    def apply_model_to_vcf(self, input_vcf, output_vcf=None, model_type=None, conf_threshold=0.5):
        """Apply trained model to a VCF file"""
        if model_type:
            self.model_type = model_type
            
        if not output_vcf:
            output_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_predictions.vcf.gz")
        
        # Load model if not already loaded
        if not self.model_loaded:
            try:
                self.load_model(self.model_type)
            except FileNotFoundError:
                print(f"No saved {self.model_type} model found. Please train a model first.")
                return
        
        print(f"\nApplying {self.model_type.upper()} model to {input_vcf}...")
        
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
        
        # Apply PCA transformation if model uses it
        if self.use_pca and self.pca is not None:
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled
        
        # Get predictions and probabilities
        predictions = self.model.predict(X_transformed)
        probabilities = self.model.predict_proba(X_transformed)
        
        # Convert numeric predictions back to class names
        class_predictions = self.label_encoder.inverse_transform(predictions)
        
        # Read and modify VCF
        with gzip.open(input_vcf, 'rt') as f_in, gzip.open(output_vcf, 'wt') as f_out:
            # Copy header and add model prediction fields
            header_written = False
            for line in f_in:
                if line.startswith('#'):
                    if line.startswith('#CHROM'):
                        # Add new INFO fields before the header line
                        f_out.write(f'##INFO=<ID={self.model_type.upper()}_CLASS,Number=1,Type=String,'
                                'Description="{self.model_type.upper()} predicted class (homozygous, heterozygous, no_variance)">\n')
                        f_out.write(f'##INFO=<ID={self.model_type.upper()}_HOMO,Number=1,Type=Float,'
                                'Description="{self.model_type.upper()} probability of being homozygous (1/1)">\n')
                        f_out.write(f'##INFO=<ID={self.model_type.upper()}_HETERO,Number=1,Type=Float,'
                                'Description="{self.model_type.upper()} probability of being heterozygous (0/1)">\n')
                        f_out.write(f'##INFO=<ID={self.model_type.upper()}_NOVAR,Number=1,Type=Float,'
                                'Description="{self.model_type.upper()} probability of being no variance (0/0)">\n')
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
                    
                    # Get class prediction and probabilities for this variant
                    class_pred = class_predictions[idx]
                    probs = probabilities[idx]
                    
                    # Find probability for each class (order matters here)
                    # The order should match self.class_labels = ['no_variance', 'heterozygous', 'homozygous']
                    novar_prob = probs[0]  # probability of class 0 (no_variance)
                    hetero_prob = probs[1]  # probability of class 1 (heterozygous)
                    homo_prob = probs[2]    # probability of class 2 (homozygous)
                    
                    # Add model predictions to INFO field
                    info += f";{self.model_type.upper()}_CLASS={class_pred}"
                    info += f";{self.model_type.upper()}_HOMO={homo_prob:.4f}"
                    info += f";{self.model_type.upper()}_HETERO={hetero_prob:.4f}"
                    info += f";{self.model_type.upper()}_NOVAR={novar_prob:.4f}"
                    
                    fields[7] = info
                    
                    # Write modified line
                    f_out.write('\t'.join(fields) + '\n')
                    idx += 1
                else:
                    # If we've used all predictions but there are more variants, just copy as is
                    f_out.write(line)
        
        print(f"Applied {self.model_type} model to {idx} variants")
        print(f"Output saved to: {output_vcf}")
        
        # Create filtered VCFs by class
        self.create_filtered_vcfs_by_class(output_vcf, class_predictions, probabilities, conf_threshold)
        
        return output_vcf
    
    def create_filtered_vcfs_by_class(self, input_vcf, class_predictions, probabilities, threshold=0.5):
        """Create filtered VCF files for each class"""
        # Define output paths
        homo_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_homozygous.vcf")
        hetero_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_heterozygous.vcf")
        novar_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_no_variance.vcf")
        
        homo_vcf_gz = homo_vcf + ".gz"
        hetero_vcf_gz = hetero_vcf + ".gz"
        novar_vcf_gz = novar_vcf + ".gz"
        
        print(f"\nFiltering variants with confidence threshold: {threshold}")
        
        # First create uncompressed VCF files
        with gzip.open(input_vcf, 'rt') as f_in, \
            open(homo_vcf, 'wt') as f_homo, \
            open(hetero_vcf, 'wt') as f_hetero, \
            open(novar_vcf, 'wt') as f_novar:
            
            # Copy header to all files
            header_lines = []
            for line in f_in:
                if line.startswith('#'):
                    header_lines.append(line)
                    f_homo.write(line)
                    f_hetero.write(line)
                    f_novar.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            # Process variants
            homo_count = hetero_count = novar_count = 0
            idx = 0
            
            for line in f_in:
                if idx < len(class_predictions):
                    class_pred = class_predictions[idx]
                    prob = probabilities[idx][self.label_encoder.transform([class_pred])[0]]
                    
                    # Only include variants with confidence above threshold
                    if prob >= threshold:
                        if class_pred == 'homozygous':
                            f_homo.write(line)
                            homo_count += 1
                        elif class_pred == 'heterozygous':
                            f_hetero.write(line)
                            hetero_count += 1
                        elif class_pred == 'no_variance':
                            f_novar.write(line)
                            novar_count += 1
                    
                    idx += 1
                else:
                    # If we've used all predictions but there are more variants, skip them
                    pass
        
        # Use bgzip to compress the files
        try:
            # Compress homozygous VCF
            subprocess.run([PATH_CONFIG['BGZIP'], '-f', homo_vcf], check=True)
            subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', homo_vcf_gz], check=True)
            
            # Compress heterozygous VCF
            subprocess.run([PATH_CONFIG['BGZIP'], '-f', hetero_vcf], check=True)
            subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', hetero_vcf_gz], check=True)
            
            # Compress no variance VCF
            subprocess.run([PATH_CONFIG['BGZIP'], '-f', novar_vcf], check=True)
            subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', novar_vcf_gz], check=True)
            
            print(f"\nFiltered variant counts (threshold={threshold}):")
            print(f"Homozygous variants (1/1): {homo_count}")
            print(f"Heterozygous variants (0/1): {hetero_count}")
            print(f"No variance (0/0): {novar_count}")
            
            return homo_vcf_gz, hetero_vcf_gz, novar_vcf_gz
        
        except subprocess.CalledProcessError as e:
            print(f"Error compressing or indexing VCF files: {e}")
            return None, None, None

    # Remove the old create_filtered_vcf method or keep it with a warning that it's deprecated
    def create_filtered_vcf(self, input_vcf, probabilities, threshold=0.5):
        """
        Old method for binary classification - kept for compatibility.
        Use create_filtered_vcfs_by_class instead.
        """
        print("Warning: Using binary classification filtering method with multi-class model.")
        print("Consider using create_filtered_vcfs_by_class instead.")
        
        # Define output paths
        high_conf_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_high_confidence.vcf")
        low_conf_vcf = os.path.join(self.builder.model_dir, f"{self.model_type}_low_confidence.vcf")
        high_conf_vcf_gz = high_conf_vcf + ".gz"
        low_conf_vcf_gz = low_conf_vcf + ".gz"
        
        # Implementation continues as before...
        # ...existing code...


def main():
    """Main function to parse arguments and run the workflow"""
    parser = argparse.ArgumentParser(description="Machine Learning Model for Variant Classification")
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
                      help="Perform grid search for model hyperparameters")
    parser.add_argument("--skip-training", action="store_true",
                      help="Skip training and use existing model")
    parser.add_argument("--input-vcf",
                      help="Input VCF file to classify (if skipping training)")
    parser.add_argument("--max-training-samples", type=int, default=90000,
                      help="Maximum number of training samples to use (default: 90000)")
    parser.add_argument("--model-type", default="neural_network", 
                      choices=["svm", "random_forest", "xgboost", "neural_network"],
                      help="Type of model to train (default: svm)")
    parser.add_argument("--use-pca", action="store_true",
                      help="Use PCA dimensionality reduction before training")
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
    
    # Check if XGBoost is available if requested
    if args.model_type == "xgboost" and not XGB_AVAILABLE:
        parser.error("XGBoost is not available. Please install xgboost or select a different model type.")
    
    # Initialize model trainer
    model_trainer = ModelTrainer(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        max_training_samples=args.max_training_samples
    )
    
    # Print configuration
    print("\nModel Training Configuration:")
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Model Type: {args.model_type}")
    if args.use_pca:
        print(f"Using PCA with variance threshold: {args.variance_threshold}")
    else:
        print("PCA: Disabled")
    print(f"Confidence Threshold: {args.confidence_threshold}")
    print(f"Max Training Samples: {args.max_training_samples}")
    if args.grid_search:
        print("Grid Search: Enabled")
    
    if not args.skip_training:
        # Build training sets
        model_trainer.build_training_sets()
        
        # Train model
        model_trainer.train_model(
            model_type=args.model_type,
            variance_threshold=args.variance_threshold,
            use_pca=args.use_pca,
            grid_search=args.grid_search
        )
        
        # Apply model to input VCF (or to the sequence_no_error.vcf.gz file if no input specified)
        input_vcf = args.input_vcf if args.input_vcf else model_trainer.builder.seq_no_error_vcf
        print(f"Applying trained {args.model_type} model to: {input_vcf}")
        model_trainer.apply_model_to_vcf(input_vcf, model_type=args.model_type, conf_threshold=args.confidence_threshold)
    else:
        # Load existing model
        try:
            model_trainer.load_model(args.model_type)
            
            # Apply model to input VCF (or to the sequence_no_error.vcf.gz file if no input specified)
            input_vcf = args.input_vcf if args.input_vcf else model_trainer.builder.seq_no_error_vcf
            print(f"Applying saved {args.model_type} model to: {input_vcf}")
            model_trainer.apply_model_to_vcf(input_vcf, model_type=args.model_type, conf_threshold=args.confidence_threshold)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Please train a {args.model_type} model first or provide the correct path to an existing model.")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


# Run: DLPFC 151669, nn, dont use pca, baseq0mapq0
# python scripts/4_classifier/run_supplimentary_models.py --dataset DLPFC  --section_id 151669 --model-type neural_network --quality-filter baseQ0mapQ0

# Run: P4 sec1, nn, dont use pca, baseq0mapq0
# python scripts/4_classifier/run_supplimentary_models.py --dataset P4_TUMOR  --section_id 1 --model-type neural_network --quality-filter baseQ0mapQ0 --confidence-threshold 0.5 --skip-training