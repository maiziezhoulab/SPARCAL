import os
import gzip
import argparse
from typing import Dict, List, Optional
from pathlib import Path

class SVMResultDivider:
    def __init__(self, section_id: str, threshold: float, quality_filter: str = "baseQ0mapQ0"):
        self.section_id = section_id
        self.threshold = threshold
        self.quality_filter = quality_filter
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        self.setup_paths()
        
    def setup_paths(self):
        """Setup paths for input and output files"""
        section_path = os.path.join(self.base_dir, "data/dlpfc", self.section_id)
        
        # Input VCF with SVM predictions
        self.input_vcf = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "results/svm_predictions.vcf.gz"
        )
        
        # Output directory
        self.output_dir = os.path.join(
            section_path, "output_VCFs/SVMModel",
            self.quality_filter, "results"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def extract_svm_prob(self, info_str: str) -> Optional[float]:
        """Extract SVM probability from INFO field"""
        for field in info_str.split(';'):
            if field.startswith('SVM_PROB='):
                try:
                    return float(field.split('=')[1])
                except ValueError:
                    return None
        return None

    def extract_genotype(self, format_str: str, sample_str: str) -> Optional[str]:
        """Extract GT field from FORMAT column"""
        try:
            gt_idx = format_str.split(':').index('GT')
            return sample_str.split(':')[gt_idx]
        except (ValueError, IndexError):
            return None

    def divide_variants(self):
        """Divide variants based on SVM probability threshold"""
        # Open output files
        high_prob_vcf = os.path.join(self.output_dir, "high_confidence.vcf.gz")
        low_prob_vcf = os.path.join(self.output_dir, "low_confidence.vcf.gz")
        
        counters = {
            'total': 0,
            'high_prob': 0,
            'low_prob': 0,
            'high_prob_het': 0,
            'high_prob_hom': 0,
            'low_prob_het': 0,
            'low_prob_hom': 0
        }

        with gzip.open(self.input_vcf, 'rt') as f_in, \
             gzip.open(high_prob_vcf, 'wt') as f_high, \
             gzip.open(low_prob_vcf, 'wt') as f_low:

            # Copy header
            for line in f_in:
                if line.startswith('#'):
                    f_high.write(line)
                    f_low.write(line)
                    if line.startswith('#CHROM'):
                        break

            # Process variants
            for line in f_in:
                counters['total'] += 1
                fields = line.strip().split('\t')
                
                svm_prob = self.extract_svm_prob(fields[7])
                gt = self.extract_genotype(fields[8], fields[9])
                
                if svm_prob is None or gt is None:
                    continue

                # Determine high/low confidence based on threshold
                if svm_prob >= (1 - self.threshold):
                    f_high.write(line)
                    counters['high_prob'] += 1
                    if gt == '0/1':
                        counters['high_prob_het'] += 1
                    elif gt == '1/1':
                        counters['high_prob_hom'] += 1
                else:
                    f_low.write(line)
                    counters['low_prob'] += 1
                    if gt == '0/1':
                        counters['low_prob_het'] += 1
                    elif gt == '1/1':
                        counters['low_prob_hom'] += 1

                if counters['total'] % 100000 == 0:
                    print(f"Processed {counters['total']:,} variants...")

        # Print summary
        print("\nVariant Division Summary:")
        print(f"Total variants processed: {counters['total']:,}")
        print(f"\nHigh confidence variants (>= {1-self.threshold:.3f}):")
        print(f"Total: {counters['high_prob']:,}")
        print(f"Heterozygous (0/1): {counters['high_prob_het']:,}")
        print(f"Homozygous (1/1): {counters['high_prob_hom']:,}")
        print(f"\nLow confidence variants (< {1-self.threshold:.3f}):")
        print(f"Total: {counters['low_prob']:,}")
        print(f"Heterozygous (0/1): {counters['low_prob_het']:,}")
        print(f"Homozygous (1/1): {counters['low_prob_hom']:,}")

        print(f"\nOutput files:")
        print(f"High confidence variants: {high_prob_vcf}")
        print(f"Low confidence variants: {low_prob_vcf}")

def main():
    parser = argparse.ArgumentParser(description="Divide SVM predictions based on probability threshold")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--threshold", type=float, required=True,
                      help="Threshold for dividing variants (e.g., 0.2)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0")
    args = parser.parse_args()
    
    divider = SVMResultDivider(args.section_id, args.threshold, args.quality_filter)
    divider.divide_variants()

if __name__ == "__main__":
    main()

# Usage:
# python scripts/postprocess/run_svm_divide.py --section_id 151507 --threshold 0.2 --quality-filter baseQ0mapQ0