import os
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional
import argparse

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
    def __init__(self, section_id: str, quality_filter: str):
        self.section_id = section_id
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.transition_metrics = {}
        self.quality_filter = quality_filter
        self.section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        self.setup_directories(quality_filter)
        self.output_base = os.path.join(self.section_path, 
            "output_VCFs/genotype_classify", quality_filter)
        
    def setup_directories(self, quality_filter = "baseQ0mapQ0"): 
        """Setup necessary directories for output files"""
        # section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        self.filtered_vcf = os.path.join(self.section_path, 
                                    "output_VCFs/beagle", quality_filter, "all_filtered.vcf.gz")
        self.output_dir = os.path.join(self.section_path, "output_VCFs/SeqErrModel", quality_filter)
        self.metrics_dir = os.path.join(self.section_path, "metrics/beagle", quality_filter)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
    def calculate_transition_thresholds(self):
        """Calculate BAF and depth thresholds for 0/1->0/0 transitions by ref/alt pairs"""
        metrics_by_transition = self._load_transition_metrics()
        
        # Only process 0/1->0/0 transitions
        heterozygous_to_ref = {}
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
            
    def _load_transition_metrics(self):
        """Load transition metrics from cached results"""
        # We want to load the shifted results since we're looking at 0/1->0/0 transitions
        cache_file = os.path.join(self.metrics_dir, f'{self.section_id}_shifted_results.pkl')
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
        """Extract BAF and depth from VCF fields"""
        # Extract BAF from FORMAT column
        baf_str = self.extract_format_field(format_str, value_str, "BAF")
        baf = float(baf_str) if baf_str is not None else None
        
        # Extract depth from INFO column
        depth_str = self.extract_info_field(info_str, "DP")
        depth = int(depth_str) if depth_str is not None else None
            
        return baf, depth
        
    def _extract_genotype(self, format_str: str, value_str: str) -> str:
        """Extract GT field from VCF FORMAT/VALUE strings"""
        try:
            gt_idx = format_str.split(':').index('GT')
            return value_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None
            
    def apply_model(self):
        """Apply the sequence error model to filtered variants"""
        from tqdm import tqdm

        if not self.transition_metrics:
            self.calculate_transition_thresholds()
            
        seq_error_vcf = os.path.join(self.output_dir, "sequence_error.vcf.gz")
        no_seq_error_vcf = os.path.join(self.output_dir, "sequence_no_error.vcf.gz")
        
        # Counter for tracking classifications
        counters = {
            'total': 0,
            'total_het': 0,  # Total heterozygous variants
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
                if gt == '0/1':
                    counters['total_het'] += 1
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
                
                # Update progress bar
                pbar.update(1)
                if counters['total'] % 100000 == 0:
                    # Update progress bar description with current stats
                    pbar.set_postfix({
                        'seq_error': f"{counters['seq_error']:,}",
                        'no_error': f"{counters['no_error']:,}"
                    })
        
            # Close progress bar
            pbar.close()
            
        # Print final summary
        # Print final summary
        print("\nSequence Error Model Summary:")
        print(f"Total variants processed: {counters['total']:,}")
        print(f"\nHeterozygous (0/1) variants:")
        print(f"Total heterozygous variants: {counters['total_het']:,} ({(counters['total_het']/counters['total']*100):.2f}% of all variants)")
        print(f"Sequence errors identified: {counters['seq_error']:,} ({(counters['seq_error']/counters['total_het']*100):.2f}% of heterozygous variants)")
        print(f"Non-sequence errors: {counters['no_error']:,} ({(counters['no_error']/counters['total']*100):.2f}% of total variants)")
        print(f"  Of which heterozygous: {counters['total_het'] - counters['seq_error']:,} ({((counters['total_het'] - counters['seq_error'])/counters['total_het']*100):.2f}% of heterozygous variants)")
        print(f"Variants with missing information: {counters['missing_info']:,} ({(counters['missing_info']/counters['total']*100):.2f}% of total variants)")
        print(f"\nOutput files:")
        print(f"  Sequence errors: {seq_error_vcf}")
        print(f"  Non-sequence errors: {no_seq_error_vcf}")

def main():
    parser = argparse.ArgumentParser(description="Build and apply sequence error model")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0", help="Quality filter to use")
    parser.add_argument("--section_id", required=True, help="Section ID")
    args = parser.parse_args()
    
    model = SequenceErrorModel(args.section_id, args.quality_filter)
    model.calculate_transition_thresholds()
    model.apply_model()

if __name__ == "__main__":
    main()

# Usage
# python scripts/postprocess/run_sequence_error_model.py --section_id 151507 --quality_filter baseQ0mapQ0