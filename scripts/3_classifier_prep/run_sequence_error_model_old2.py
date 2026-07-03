import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse

REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [str(i) for i in range(1, 23)]  # 1, 2, 3, ..., 22
    },
    "FFPE_VISIUM": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    },
    "TUMOR": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
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
        "reference": "FFPE_VISIUM"
    },
    "10X_BC_FFPE": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_Ductal_Carcinoma_In_Situ_Invasive_Carcinoma_FFPE",
        "output_dir": "data/10X_BC_FFPE",
        "has_sections": False,
        "reference": "FFPE_VISIUM"
    },
    "P4_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    }
}

THOUSAND_GENOME_CONFIGS = {
    "GRCh38": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "pattern": "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    "hg19": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19/",
        "pattern": "hg19_chr{chrom}.vcf.gz"
    }
}


@dataclass
class GenotypeMetrics:
    baf: float
    depth: int
    ref: str
    alt: str

@dataclass
class TransitionMetrics:
    baf_threshold: float
    depth_threshold: float
    transition_type: str

class SequenceErrorModel:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.transition_metrics = {}
        self.validate_dataset_config()
        self.setup_directories()
        
    def validate_dataset_config(self):
        """Validate dataset configuration and section ID if required."""
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
        
    def setup_directories(self): 
        """Setup necessary directories for output files"""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Determine base path based on dataset configuration
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
        
        # Setup paths with quality filter
        self.filtered_vcf = os.path.join(
            output_base, 
            "output_VCFs/beagle", 
            self.quality_filter, 
            "all_filtered_out.vcf.gz"  # Now using filtered-out variants
        )
        self.output_dir = os.path.join(
            output_base, 
            "output_VCFs/SeqErrModel",
            self.quality_filter
        )
        self.metrics_dir = os.path.join(
            output_base, 
            "metrics/beagle",
            self.quality_filter
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
    def extract_format_field(self, format_str: str, value_str: str, field_name: str) -> Optional[str]:
        """Extract a specific field from VCF FORMAT column with dataset-specific handling."""
        try:
            format_fields = format_str.split(':')
            if field_name not in format_fields:
                return None
            
            field_idx = format_fields.index(field_name)
            value_fields = value_str.split(':')
            
            if field_idx >= len(value_fields):
                return None
                
            return value_fields[field_idx]
        except (ValueError, IndexError):
            return None

    def extract_info_field(self, info_str: str, field_name: str) -> Optional[str]:
        """Extract a specific field from VCF INFO column with dataset-specific handling."""
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None
            
    def _extract_metrics(self, info_str: str, format_str: str, value_str: str) -> Tuple[float, int]:
        """Extract BAF and depth with dataset-specific handling"""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Handle different VCF formats based on dataset
        if self.dataset_name in ["DLPFC", "10X_BC_6.5MM", "10X_BC_FFPE"]:
            # Extract BAF from FORMAT column
            baf_str = self.extract_format_field(format_str, value_str, "BAF")
            baf = float(baf_str) if baf_str is not None else None
            
            # Extract depth from INFO column
            depth_str = self.extract_info_field(info_str, "DP")
            depth = int(depth_str) if depth_str is not None else None
            
        elif self.dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
            # These datasets might have different field names or locations
            # Adjust extraction based on actual format
            baf_str = self.extract_format_field(format_str, value_str, "AF")  # Example: using AF instead of BAF
            baf = float(baf_str) if baf_str is not None else None
            
            depth_str = self.extract_info_field(info_str, "DP")
            depth = int(depth_str) if depth_str is not None else None
            
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_name}")
            
        return baf, depth
        
    def _extract_genotype(self, format_str: str, value_str: str) -> str:
        """Extract GT field from VCF FORMAT/VALUE strings"""
        try:
            gt_idx = format_str.split(':').index('GT')
            return value_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None
            
    def _load_transition_metrics(self) -> Dict[str, List[GenotypeMetrics]]:
        """Load transition metrics from CSV files"""
        metrics_by_transition = defaultdict(list)
        
        # Load all CSV files in the metrics directory
        for file_path in Path(self.metrics_dir).glob("*.csv"):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                transition = row['transition']
                baf = row['BAF']
                depth = row['Depth']
                ref = row['Ref']
                alt = row['Alt']
                
                metrics_by_transition[transition].append(GenotypeMetrics(baf, depth, ref, alt))
                
        return metrics_by_transition


    def calculate_transition_thresholds(self):
        """Calculate BAF and depth thresholds for 0/1->0/0 transitions by ref/alt pairs"""
        metrics_by_transition = self._load_transition_metrics()
        
        # Only process 0/1->0/0 transitions
        print("\nCalculating thresholds for 0/1->0/0 transitions by ref/alt pairs:")
        print("-" * 80)
        print(f"{'Ref->Alt':<10} {'BAF Threshold':<15} {'Depth Threshold':<15} {'Number of Variants':<20}")
        print("-" * 80)
        
        for transition, metrics in metrics_by_transition.items():
            if not metrics:
                continue
                
            # Check if this is a 0/1->0/0 transition
            trans_parts = transition.split('_')[0].split('->')
            orig_gt, new_gt = trans_parts[0], trans_parts[1]
            
            if orig_gt == '0/1' and new_gt == '0/0':
                # Get ref and alt alleles from first metric (they're the same for all metrics in this transition)
                ref, alt = metrics[0].ref, metrics[0].alt
                key = f"{ref}->{alt}"
                
                bafs = [m.baf for m in metrics]
                depths = [m.depth for m in metrics]
                
                # Calculate median values as thresholds
                baf_threshold = np.median(bafs)
                depth_threshold = np.median(depths)
                
                self.transition_metrics[key] = TransitionMetrics(
                    baf_threshold=baf_threshold,
                    depth_threshold=depth_threshold,
                    transition_type=key
                )
                
                print(f"{key:<10} {baf_threshold:>13.3f}  {depth_threshold:>13.1f}  {len(metrics):>18,}")
        
        print("-" * 80)
        print(f"Total transition types: {len(self.transition_metrics)}")

    def apply_model(self):
        """Apply the sequence error model to filtered-out variants"""
        if not self.transition_metrics:
            self.calculate_transition_thresholds()
            
        seq_error_vcf = os.path.join(self.output_dir, "sequence_error.vcf.gz")
        no_seq_error_vcf = os.path.join(self.output_dir, "sequence_no_error.vcf.gz")
        
        # Counter for tracking classifications
        counters = {
            'total': 0,
            'seq_error': 0,
            'no_error': 0,
            'missing_info': 0
        }

        # First count total variants for progress bar
        print("Counting total variants...")
        total_variants = 0
        with gzip.open(self.filtered_vcf, 'rt') as f_in:
            for line in f_in:
                if not line.startswith('#'):
                    total_variants += 1
        print(f"Found {total_variants:,} variants to process")
        
        with gzip.open(self.filtered_vcf, 'rt') as f_in, \
             gzip.open(seq_error_vcf, 'wt') as f_error, \
             gzip.open(no_seq_error_vcf, 'wt') as f_no_error:
            
            # Copy header
            for line in f_in:
                if line.startswith('#'):
                    f_error.write(line)
                    f_no_error.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            # Process variants with progress bar
            from tqdm import tqdm
            pbar = tqdm(total=total_variants, desc="Processing variants", unit="variants")
            
            for line in f_in:
                if line.startswith('#'):
                    continue
                    
                counters['total'] += 1
                fields = line.strip().split('\t')
                ref, alt = fields[3], fields[4]
                baf, depth = self._extract_metrics(fields[7], fields[8], fields[9])
                gt = self._extract_genotype(fields[8], fields[9])
                
                # Skip if missing required information
                if None in (baf, depth, gt):
                    f_no_error.write(line)
                    counters['missing_info'] += 1
                    pbar.update(1)
                    continue
                
                # Check if this is a heterozygous variant
                is_seq_error = False
                if gt == '0/1':
                    # Look up the appropriate threshold for this ref/alt combination
                    transition_key = f"{ref}->{alt}"
                    if transition_key in self.transition_metrics:
                        metrics = self.transition_metrics[transition_key]
                        if (baf <= metrics.baf_threshold and 
                            depth <= metrics.depth_threshold):
                            is_seq_error = True
                
                # Write to appropriate output file
                if is_seq_error:
                    f_error.write(line)
                    counters['seq_error'] += 1
                else:
                    f_no_error.write(line)
                    counters['no_error'] += 1
                
                pbar.update(1)
                
            pbar.close()
            
        # Print final summary
        print("\nSequence Error Model Summary:")
        print(f"Dataset: {self.dataset_name}")
        if self.section_id:
            print(f"Section ID: {self.section_id}")
        print(f"Total variants processed: {counters['total']:,}")
        print(f"Sequence errors identified: {counters['seq_error']:,} "
              f"({(counters['seq_error']/counters['total']*100):.2f}%)")
        print(f"Non-sequence errors: {counters['no_error']:,} "
              f"({(counters['no_error']/counters['total']*100):.2f}%)")
        print(f"Variants with missing information: {counters['missing_info']:,} "
              f"({(counters['missing_info']/counters['total']*100):.2f}%)")
        print(f"\nOutput files:")
        print(f"  Sequence errors: {seq_error_vcf}")
        print(f"  Non-sequence errors: {no_seq_error_vcf}")

def main():
    parser = argparse.ArgumentParser(description="Build and apply sequence error model")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    args = parser.parse_args()
    
    model = SequenceErrorModel(args.dataset, args.quality_filter, args.section_id)
    model.calculate_transition_thresholds()
    model.apply_model()

if __name__ == "__main__":
    main()

# Usage on 10X_BC_6.5MM baseQ0mapQ0:
# python scripts/postprocess/run_sequence_error_model.py --dataset P4_TUMOR --quality_filter baseQ0mapQ0 --section_id 1