import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse
from tqdm import tqdm

# Import dataset configurations from beagle script
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
    genotype: str  # Added genotype to distinguish het vs hom models

class SequenceErrorModel:
    def __init__(self, dataset_name: str, quality_filter: str, section_id: str = None, 
                 het_baf_threshold: float = None, het_depth_threshold: float = None,
                 hom_baf_threshold: float = 0.9, hom_depth_threshold: float = None):
        self.dataset_name = dataset_name
        self.section_id = section_id
        self.quality_filter = quality_filter
        
        # Store manual thresholds (None means calculate from data)
        self.manual_het_baf_threshold = het_baf_threshold
        self.manual_het_depth_threshold = het_depth_threshold
        self.manual_hom_baf_threshold = hom_baf_threshold
        self.manual_hom_depth_threshold = hom_depth_threshold
        
        self.base_dir = "/data/maiziezhou_lab/leiy4/snv_calling"
        # Separate dictionaries for heterozygous and homozygous transition metrics
        self.het_transition_metrics = {}
        self.hom_transition_metrics = {}
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
            
        self.filtered_vcf = os.path.join(output_base, 
                                       "output_VCFs/beagle", 
                                       self.quality_filter, 
                                       "all_filtered_out.vcf.gz")
        self.output_dir = os.path.join(output_base, 
                                     "output_VCFs/SeqErrModel", 
                                     self.quality_filter)
        self.metrics_dir = os.path.join(output_base, 
                                      "metrics/beagle", 
                                      self.quality_filter)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_transition_thresholds(self):
        """Calculate BAF and depth thresholds for both 0/1->0/0 and 1/1->0/0 transitions by ref/alt pairs"""
        metrics_by_transition = self._load_transition_metrics()
        
        # Process both 0/1->0/0 and 1/1->0/0 transitions
        print("\nCalculating thresholds for HET (0/1->0/0) transitions by ref/alt pairs:")
        print("-" * 80)
        print(f"{'Ref->Alt':<10} {'BAF Threshold':<15} {'Depth Threshold':<15} {'Number of Variants':<20}")
        print("-" * 80)
        
        # Track counts for summary
        het_count = 0
        hom_count = 0
        
        for transition, metrics in metrics_by_transition.items():
            if not metrics:
                continue
                
            # Parse transition parts
            trans_parts = transition.split('_')[0].split('->')
            orig_gt, new_gt = trans_parts[0], trans_parts[1]
            ref, alt = metrics[0]['metrics'].ref, metrics[0]['metrics'].alt
            key = f"{ref}->{alt}"
            
            # Process heterozygous transitions (0/1->0/0)
            if orig_gt == '0/1' and new_gt == '0/0':
                bafs = [m['metrics'].baf for m in metrics]
                depths = [m['metrics'].depth for m in metrics]
                
                # Use manual thresholds if provided, otherwise calculate from data
                if self.manual_het_baf_threshold is not None:
                    baf_threshold = self.manual_het_baf_threshold
                    baf_source = "manual"
                else:
                    baf_threshold = np.median(bafs)
                    baf_source = "median"
                    
                if self.manual_het_depth_threshold is not None:
                    depth_threshold = self.manual_het_depth_threshold
                    depth_source = "manual"
                else:
                    depth_threshold = np.median(depths)
                    depth_source = "median"
                
                self.het_transition_metrics[key] = TransitionMetrics(
                    baf_threshold=baf_threshold,
                    depth_threshold=depth_threshold,
                    transition_type=key,
                    genotype='0/1'
                )
                
                print(f"{key:<10} {baf_threshold:>13.3f} ({baf_source})  {depth_threshold:>13.1f} ({depth_source})  {len(metrics):>18,}")
                het_count += 1
        
        # Print summary for heterozygous transitions
        print("-" * 80)
        print(f"Total heterozygous transition types: {het_count}")
        
        # Process homozygous transitions (1/1->0/0) with manually set BAF threshold
        print("\nCalculating thresholds for HOM (1/1->0/0) transitions by ref/alt pairs:")
        print("-" * 80)
        print(f"{'Ref->Alt':<10} {'BAF Threshold':<15} {'Depth Threshold':<15} {'Number of Variants':<20}")
        print("-" * 80)
        
        for transition, metrics in metrics_by_transition.items():
            if not metrics:
                continue
                
            # Parse transition parts
            trans_parts = transition.split('_')[0].split('->')
            orig_gt, new_gt = trans_parts[0], trans_parts[1]
            
            # Process homozygous transitions (1/1->0/0)
            if orig_gt == '1/1' and new_gt == '0/0':
                ref, alt = metrics[0]['metrics'].ref, metrics[0]['metrics'].alt
                key = f"{ref}->{alt}"
                
                depths = [m['metrics'].depth for m in metrics]
                
                # Always use manual BAF threshold for homozygous variants
                baf_threshold = self.manual_hom_baf_threshold
                baf_source = "manual"
                
                # Use manual depth threshold if provided, otherwise calculate
                if self.manual_hom_depth_threshold is not None:
                    depth_threshold = self.manual_hom_depth_threshold
                    depth_source = "manual"
                else:
                    depth_threshold = np.median(depths)
                    depth_source = "median"
                
                self.hom_transition_metrics[key] = TransitionMetrics(
                    baf_threshold=baf_threshold,
                    depth_threshold=depth_threshold,
                    transition_type=key,
                    genotype='1/1'
                )
                
                print(f"{key:<10} {baf_threshold:>13.3f} ({baf_source})  {depth_threshold:>13.1f} ({depth_source})  {len(metrics):>18,}")
                hom_count += 1
        
        # Print summary for homozygous transitions
        print("-" * 80)
        print(f"Total homozygous transition types: {hom_count}")
            
    def _load_transition_metrics(self):
        """Load transition metrics from cached results"""
        if self.section_id:
            cache_file = os.path.join(self.metrics_dir, 
                                    f'{self.dataset_name}_{self.section_id}_shifted_results.pkl')
        else:
            cache_file = os.path.join(self.metrics_dir,
                                    f'{self.dataset_name}_shifted_results.pkl')
            
        if not os.path.exists(cache_file):
            raise FileNotFoundError(f"Shifted results cache not found: {cache_file}")
            
        import pickle
        with open(cache_file, 'rb') as f:
            results = pickle.load(f)
            return results['metrics_by_transition']
            
    def _get_transition_key(self, orig_gt: str, new_gt: str, ref: str, alt: str) -> str:
        """Generate transition key in same format as variant_quality_model"""
        return f"{orig_gt}->{new_gt}_{ref}_{alt}"

    def extract_format_field(self, format_str: str, value_str: str, field_name: str) -> Optional[str]:
        """Extract a specific field from VCF FORMAT column."""
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
        """Extract a specific field from VCF INFO column."""
        for field in info_str.split(';'):
            if field.startswith(f"{field_name}="):
                return field.split('=')[1]
        return None

    def _extract_metrics(self, info_str: str, format_str: str, value_str: str) -> Tuple[float, int]:
        """Extract BAF and depth based on dataset format"""
        if self.dataset_name in ["DLPFC", "10X_BC_6.5MM", "10X_BC_FFPE", "DCIS", "OVAR_P5"]:
            # Extract BAF from FORMAT column
            baf_str = self.extract_format_field(format_str, value_str, "BAF")
            baf = float(baf_str) if baf_str is not None else None
            
            # Extract depth from INFO column
            depth_str = self.extract_info_field(info_str, "DP")
            depth = int(depth_str) if depth_str is not None else None
            
        elif self.dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
            # These datasets might have different field names
            baf_str = self.extract_format_field(format_str, value_str, "BAF")
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
            
    def apply_model(self):
        """Apply both sequence error models (het and hom) to filtered variants"""
        if not self.het_transition_metrics and not self.hom_transition_metrics:
            self.calculate_transition_thresholds()
            
        seq_error_vcf = os.path.join(self.output_dir, "sequence_error.vcf.gz")
        no_seq_error_vcf = os.path.join(self.output_dir, "sequence_no_error.vcf.gz")
        
        # Counter for tracking classifications
        counters = {
            'total': 0,
            'het_seq_error': 0,
            'hom_seq_error': 0,
            'total_het': 0,  # Track total heterozygous variants
            'total_hom': 0,  # Track total homozygous variants
            'no_error': 0,
            'filtered_0_0': 0,  # Track 0/0 genotypes filtered out
            'missing_info': 0
        }

        # First count total variants for progress bar
        print("\nCounting total variants...")
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
                        # Add new INFO fields for model
                        f_error.write('##INFO=<ID=SEQ_ERROR_MODEL,Number=1,Type=String,Description="Sequence error model applied (HET or HOM)">\n')
                        break
            
            # Process variants
            pbar = tqdm(total=total_variants, desc="Processing variants", unit="variants")
            for line in f_in:
                if line.startswith('#'):
                    continue
                    
                counters['total'] += 1
                fields = line.strip().split('\t')
                ref, alt = fields[3], fields[4]
                baf, depth = self._extract_metrics(fields[7], fields[8], fields[9])
                gt = self._extract_genotype(fields[8], fields[9])
                
                if counters['total'] <= 5:  # Print first few variants for debugging
                    print(f"\nDebug - First few variants:")
                    print(f"REF: {ref}, ALT: {alt}")
                    print(f"INFO: {fields[7]}")
                    print(f"FORMAT: {fields[8]}")
                    print(f"VALUES: {fields[9]}")
                    print(f"Extracted - BAF: {baf}, Depth: {depth}, GT: {gt}")
                
                # Skip if missing required information
                if None in (baf, depth, gt):
                    f_no_error.write(line)
                    counters['missing_info'] += 1
                    pbar.update(1)
                    continue
                
                # Skip 0/0 genotypes - they are not variants
                if gt == '0/0':
                    counters['filtered_0_0'] += 1
                    pbar.update(1)
                    continue
                
                # Create transition key for lookup
                transition_key = f"{ref}->{alt}"
                is_seq_error = False
                error_model = None
                
                # Check if this is a heterozygous variant
                if gt == '0/1':
                    counters['total_het'] += 1
                    if transition_key in self.het_transition_metrics:
                        # Apply heterozygous sequence error model
                        metrics = self.het_transition_metrics[transition_key]
                        if (baf <= metrics.baf_threshold and 
                            depth <= metrics.depth_threshold):
                            is_seq_error = True
                            error_model = "HET"
                            counters['het_seq_error'] += 1
                
                # Check if this is a homozygous variant
                elif gt == '1/1':
                    counters['total_hom'] += 1
                    if transition_key in self.hom_transition_metrics:
                        # Apply homozygous sequence error model
                        metrics = self.hom_transition_metrics[transition_key]
                        if (baf <= metrics.baf_threshold and 
                            depth <= metrics.depth_threshold):
                            is_seq_error = True
                            error_model = "HOM"
                            counters['hom_seq_error'] += 1
                
                # Write to appropriate output file
                if is_seq_error:
                    # Add model info to sequence error variants
                    info_field = fields[7]
                    info_field += f";SEQ_ERROR_MODEL={error_model}"
                    fields[7] = info_field
                    f_error.write('\t'.join(fields) + '\n')
                else:
                    f_no_error.write(line)
                    counters['no_error'] += 1
                
                # Update progress bar
                pbar.update(1)
                if counters['total'] % 100000 == 0:
                    # Update progress bar description with current stats
                    pbar.set_postfix({
                        'het_err': f"{counters['het_seq_error']:,}",
                        'hom_err': f"{counters['hom_seq_error']:,}",
                        'no_err': f"{counters['no_error']:,}"
                    })
        
            # Close progress bar
            pbar.close()
            
        # Calculate ratios for heterozygous and homozygous errors
        het_ratio = (counters['het_seq_error'] / counters['total_het']) * 100 if counters['total_het'] > 0 else 0
        hom_ratio = (counters['hom_seq_error'] / counters['total_hom']) * 100 if counters['total_hom'] > 0 else 0
        
        # Calculate ratios excluding 0/0 genotypes
        total_variants_no_ref = counters['total'] - counters['filtered_0_0']
            
        # Print final summary
        print("\nSequence Error Model Summary:")
        print(f"Total variants processed: {counters['total']:,}")
        print(f"Reference genotypes (0/0) filtered out: {counters['filtered_0_0']:,}")
        print(f"Total non-reference variants: {total_variants_no_ref:,}")
        print(f"Heterozygous variants (0/1): {counters['total_het']:,}")
        print(f"Homozygous variants (1/1): {counters['total_hom']:,}")
        print("-" * 60)
        print(f"Heterozygous sequence errors: {counters['het_seq_error']:,} ({(counters['het_seq_error']/total_variants_no_ref*100):.2f}% of non-ref)")
        print(f"  Ratio within heterozygous variants: {het_ratio:.2f}% of all 0/1 variants")
        print(f"Homozygous sequence errors: {counters['hom_seq_error']:,} ({(counters['hom_seq_error']/total_variants_no_ref*100):.2f}% of non-ref)")
        print(f"  Ratio within homozygous variants: {hom_ratio:.2f}% of all 1/1 variants")
        print("-" * 60)
        print(f"Total sequence errors: {counters['het_seq_error'] + counters['hom_seq_error']:,} ({((counters['het_seq_error'] + counters['hom_seq_error'])/total_variants_no_ref*100):.2f}%)")
        print(f"Non-sequence errors: {counters['no_error']:,} ({(counters['no_error']/total_variants_no_ref*100):.2f}%)")
        print(f"Variants with missing information: {counters['missing_info']:,} ({(counters['missing_info']/total_variants_no_ref*100):.2f}%)")
        
        # Save summary to file
        summary_file = os.path.join(self.output_dir, "sequence_error_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("Sequence Error Model Summary\n")
            f.write("==========================\n\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            if self.section_id:
                f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality filter: {self.quality_filter}\n\n")
            f.write(f"Threshold Settings:\n")
            f.write(f"  Heterozygous BAF threshold: {self.manual_het_baf_threshold if self.manual_het_baf_threshold is not None else 'auto (median)'}\n")
            f.write(f"  Heterozygous depth threshold: {self.manual_het_depth_threshold if self.manual_het_depth_threshold is not None else 'auto (median)'}\n")
            f.write(f"  Homozygous BAF threshold: {self.manual_hom_baf_threshold}\n")
            f.write(f"  Homozygous depth threshold: {self.manual_hom_depth_threshold if self.manual_hom_depth_threshold is not None else 'auto (median)'}\n\n")
            f.write(f"Total variants processed: {counters['total']:,}\n")
            f.write(f"Reference genotypes (0/0) filtered out: {counters['filtered_0_0']:,}\n")
            f.write(f"Total non-reference variants: {total_variants_no_ref:,}\n")
            f.write(f"Heterozygous variants (0/1): {counters['total_het']:,}\n")
            f.write(f"Homozygous variants (1/1): {counters['total_hom']:,}\n")
            f.write("-" * 60 + "\n")
            f.write(f"Heterozygous sequence errors: {counters['het_seq_error']:,} ({(counters['het_seq_error']/total_variants_no_ref*100):.2f}% of non-ref)\n")
            f.write(f"  Ratio within heterozygous variants: {het_ratio:.2f}% of all 0/1 variants\n")
            f.write(f"Homozygous sequence errors: {counters['hom_seq_error']:,} ({(counters['hom_seq_error']/total_variants_no_ref*100):.2f}% of non-ref)\n")
            f.write(f"  Ratio within homozygous variants: {hom_ratio:.2f}% of all 1/1 variants\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total sequence errors: {counters['het_seq_error'] + counters['hom_seq_error']:,} ({((counters['het_seq_error'] + counters['hom_seq_error'])/total_variants_no_ref*100):.2f}%)\n")
            f.write(f"Non-sequence errors: {counters['no_error']:,} ({(counters['no_error']/total_variants_no_ref*100):.2f}%)\n")
            f.write(f"Variants with missing information: {counters['missing_info']:,} ({(counters['missing_info']/total_variants_no_ref*100):.2f}%)\n")
        
        print(f"\nOutput files:")
        print(f"  Sequence errors: {seq_error_vcf}")
        print(f"  Non-sequence errors: {no_seq_error_vcf}")
        print(f"  Summary: {summary_file}")
        print("\nNote: 0/0 genotypes were filtered out and not included in the output files")

def main():
    parser = argparse.ArgumentParser(description="Build and apply sequence error model for both heterozygous and homozygous variants")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0",
                      help="Quality filter to use")
    
    # Add threshold parameters
    parser.add_argument("--het_baf_threshold", type=float, default=None,
                      help="BAF threshold for heterozygous variants (default: auto-calculate)")
    parser.add_argument("--het_depth_threshold", type=float, default=None,
                      help="Depth threshold for heterozygous variants (default: auto-calculate)")
    parser.add_argument("--hom_baf_threshold", type=float, default=0.99,
                      help="BAF threshold for homozygous variants (default: 0.99)")
    parser.add_argument("--hom_depth_threshold", type=float, default=None,
                      help="Depth threshold for homozygous variants (default: auto-calculate)")
    args = parser.parse_args()
    
    model = SequenceErrorModel(
        args.dataset, 
        args.quality_filter, 
        args.section_id, 
        args.het_baf_threshold,
        args.het_depth_threshold,
        args.hom_baf_threshold,
        args.hom_depth_threshold
    )
    model.calculate_transition_thresholds()
    model.apply_model()

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC with section:
# python scripts/postprocess/run_sequence_error_model.py --dataset DLPFC --section_id 151507 --quality_filter baseQ0mapQ0 --hom_depth_threshold 4

# For P4_TUMOR with section:
# python scripts/3_classifier_prep/run_sequence_error_model.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ0mapQ0

# For P6_TUMOR with custom thresholds:
# python scripts/postprocess/run_sequence_error_model.py --dataset P6_TUMOR --section_id 2 --quality_filter baseQ0mapQ0 --het_baf_threshold 0.35 --het_depth_threshold 30 --hom_baf_threshold 0.85 --hom_depth_threshold 25