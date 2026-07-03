import os
import subprocess
import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import pysam
import gzip
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import median

# File Paths
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "BGZIP": "bgzip",
    "TABIX": "tabix",
}

def setup_environment() -> dict:
    apps_dir = PATH_CONFIG['APPS_DIR']
    os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {'PATH': os.environ['PATH'], 'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']}

class GenotypeTransitionAnalyzer:
    def __init__(self):
        self.transitions = defaultdict(list)  # [(baf, depth), ...] for each transition
        self.baf_bin_edges = np.arange(0, 1.05, 0.05)  # BAF bins
        self.depth_bin_edges = None  # Will be set dynamically based on data
        self.original_gt = "1/1"
        self.target_gt = "0/1"
        
    def parse_metrics(self, info_str: str) -> Tuple[float, int]:
        """Extract BAF and depth values from INFO field"""
        baf = None
        depth = None
        for field in info_str.split(';'):
            if field.startswith('BAF='):
                try:
                    baf = float(field.split('=')[1])
                except:
                    baf = None
            elif field.startswith('DP='):
                try:
                    depth = int(field.split('=')[1])
                except:
                    depth = None
        return baf, depth

    def analyze_transitions(self, original_vcf: str, beagle_vcf: str, target_chrom: str) -> Dict:
        """Analyze genotype transitions and collect BAF values"""
        variant_count = 0
        match_count = 0
        
        print(f"\nProcessing VCF files:")
        print(f"Original: {original_vcf}")
        print(f"Beagle: {beagle_vcf}")
        
        with gzip.open(original_vcf, 'rt') as orig, gzip.open(beagle_vcf, 'rt') as beagle:
            # Skip headers
            for line in orig:
                if not line.startswith('#'):
                    break
            for line in beagle:
                if not line.startswith('#'):
                    break
            
            # Create dictionary of beagle variants
            beagle_variants = {}
            for line in beagle:
                fields = line.strip().split('\t')
                chrom, pos, _, ref, alt, _, _, info, format_str, value_str = fields
                if chrom == target_chrom:
                    gt = self.extract_gt_from_format(format_str, value_str)
                    # print(f" format and value str, gt: {format_str}, {value_str}, {gt}")
                    baf, depth = self.parse_metrics(info)
                    # print(f"gt and baf: {gt} {info} {baf}")
                    if gt and baf is not None and depth is not None:
                        beagle_variants[int(pos)] = (ref, alt, gt, baf, depth)
            
            print(f"Loaded {len(beagle_variants)} variants from beagle VCF")
            
            # Process original VCF
            for line in orig:
                fields = line.strip().split('\t')
                chrom, pos, _, ref, alt, _, _, _, format_str, value_str = fields
                if chrom == target_chrom:
                    variant_count += 1
                    pos = int(pos)
                    
                    if pos in beagle_variants:
                        match_count += 1
                        beagle_ref, beagle_alt, beagle_gt, baf, depth = beagle_variants[pos]
                        
                        if ref != beagle_ref or alt != beagle_alt:
                            continue
                        
                        orig_gt = self.extract_gt_from_format(format_str, value_str)
                        # print(f" format and value str, gt: {format_str}, {value_str}, {gt}")
                        # print(f"oring gt and beagle gt: {orig_gt} {beagle_gt}")

                        if orig_gt == self.original_gt and beagle_gt == self.target_gt :     # change to plot 0/1 -> 0/0 (12 variance) or all of them
                        # if orig_gt and beagle_gt:
                            orig_alleles = self.parse_genotype(orig_gt, ref, alt)
                            beagle_alleles = self.parse_genotype(beagle_gt, beagle_ref, beagle_alt)
                            
                            if orig_alleles and beagle_alleles and orig_alleles != beagle_alleles:
                                shift_key = f"{orig_alleles}->{beagle_alleles}"
                                # print(f"shift_key: {shift_key}")
                                self.transitions[shift_key].append((baf, depth))

                    if variant_count % 1000 == 0:
                        print(f"Processed {variant_count} variants")

        all_depths = [depth for trans in self.transitions.values() for _, depth in trans]
        self.depth_bin_edges = np.logspace(np.log10(min(all_depths) + 1), 
                                         np.log10(max(all_depths) + 1), 
                                         20)
        
        return {
            'variant_count': variant_count,
            'match_count': match_count,
            'transitions': dict(self.transitions)
        }

    def plot_baf_distributions(self, output_dir: str):
        """Plot BAF distribution for each transition type"""
        if not self.transitions:
            print("No transitions to plot")
            return

        # Set up the plot grid
        n_transitions = len(self.transitions)
        n_cols = 3
        n_rows = (n_transitions + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, (transition, values) in enumerate(sorted(self.transitions.items())):
            if not values:  # Skip if no values for this transition
                continue
                
            # Extract BAF values from the tuples
            bafs = [baf for baf, _ in values]
            
            plt.subplot(n_rows, n_cols, idx + 1)
            
            # Create histogram
            counts, edges, _ = plt.hist(bafs, bins=self.baf_bin_edges, density=True, 
                                      alpha=0.7, color='skyblue', edgecolor='black')
            
            # Add median line
            med = median(bafs)
            plt.axvline(x=med, color='red', linestyle='--', label=f'Median: {med:.3f}')
            
            plt.title(f'BAF Distribution: {transition}\n(n={len(bafs)})')
            plt.xlabel('BAF Value')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add value annotations on top of bars
            for i in range(len(counts)):
                if counts[i] > 0:  # Only annotate non-zero bars
                    plt.text(edges[i] + 0.025, counts[i], 
                           f'{counts[i]:.2f}', 
                           rotation=90, 
                           va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baf_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_depth_distributions(self, output_dir: str):
        """Plot depth distribution for each transition type"""
        if not self.transitions:
            print("No transitions to plot")
            return

        n_transitions = len(self.transitions)
        n_cols = 3
        n_rows = (n_transitions + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, (transition, values) in enumerate(sorted(self.transitions.items())):
            if not values:  # Skip if no values for this transition
                continue
                
            # Extract depth values from the tuples
            depths = [depth for _, depth in values]
            
            plt.subplot(n_rows, n_cols, idx + 1)
            
            # Create histogram with log-scale x-axis
            counts, edges, _ = plt.hist(depths, bins=self.depth_bin_edges, 
                                      density=True, alpha=0.7,
                                      color='lightgreen', edgecolor='black')
            
            # Add median line
            med = median(depths)
            plt.axvline(x=med, color='red', linestyle='--', 
                       label=f'Median: {med:.0f}')
            
            plt.title(f'Depth Distribution: {transition}\n(n={len(depths)})')
            plt.xlabel('Read Depth (log scale)')
            plt.ylabel('Frequency')
            plt.xscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Add value annotations
            for i in range(len(counts)):
                if counts[i] > 0:
                    plt.text(edges[i] * 1.1, counts[i], 
                           f'{counts[i]:.2f}', 
                           rotation=45,
                           va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'depth_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_baf_depth_scatter(self, output_dir: str):
        """Create scatter plots of BAF vs Depth for each transition type"""
        if not self.transitions:
            print("No transitions to plot")
            return

        n_transitions = len(self.transitions)
        n_cols = 3
        n_rows = (n_transitions + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for idx, (transition, values) in enumerate(sorted(self.transitions.items())):
            if not values:  # Skip if no values for this transition
                continue
                
            # Extract BAF and depth values from the tuples
            bafs = [baf for baf, _ in values]
            depths = [depth for _, depth in values]
            
            plt.subplot(n_rows, n_cols, idx + 1)
            
            # Create scatter plot
            plt.scatter(bafs, depths, alpha=0.5, s=20, c='blue')
            
            # Add median lines
            baf_med = median(bafs)
            depth_med = median(depths)
            plt.axvline(x=baf_med, color='red', linestyle='--', 
                       label=f'Median BAF: {baf_med:.3f}')
            plt.axhline(y=depth_med, color='green', linestyle='--', 
                       label=f'Median Depth: {depth_med:.0f}')
            
            plt.title(f'BAF vs Depth: {transition}\n(n={len(bafs)})')
            plt.xlabel('BAF Value')
            plt.ylabel('Read Depth (log scale)')
            plt.yscale('log')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'baf_depth_scatter.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_all_distributions(self, output_dir: str):
        """Create all three types of plots"""
        # Calculate depth bins from all depth values
        all_depths = [depth for trans in self.transitions.values() for _, depth in trans]
        if all_depths:
            self.depth_bin_edges = np.logspace(np.log10(min(all_depths) + 1), 
                                             np.log10(max(all_depths) + 1), 
                                             20)
            
            self.plot_baf_distributions(output_dir)
            self.plot_depth_distributions(output_dir)
            self.plot_baf_depth_scatter(output_dir)
        else:
            print("No data to plot")
    
    @staticmethod
    def parse_genotype(gt_str: str, ref_allele: str, alt_allele: str) -> str:
        """Parse genotype into allele representation"""
        if gt_str in ('0/0', '0|0'):
            return f"{ref_allele}{ref_allele}"
        elif gt_str in ('0/1', '1/0', '0|1', '1|0'):
            return f"{ref_allele}{alt_allele}"
        elif gt_str in ('1/1', '1|1'):
            return f"{alt_allele}{alt_allele}"
        return None

    @staticmethod
    def extract_gt_from_format(format_str: str, value_str: str) -> str:
        """Extract GT value from FORMAT field"""
        try:
            gt_index = format_str.split(':').index('GT')
            return value_str.split(':')[gt_index]
        except (ValueError, IndexError):
            return None
# def parse_genotype(gt_str: str, ref_allele: str, alt_allele: str) -> str:
#     """Parse genotype into allele representation"""
#     if gt_str in ('0/0', '0|0'):
#         return f"{ref_allele}{ref_allele}"
#     elif gt_str in ('0/1', '1/0', '0|1', '1|0'):
#         return f"{ref_allele}{alt_allele}"
#     elif gt_str in ('1/1', '1|1'):
#         return f"{alt_allele}{alt_allele}"
#     return None

# def extract_gt_from_format(format_str: str, value_str: str) -> str:
#     """Extract GT value from FORMAT field"""
#     try:
#         gt_index = format_str.split(':').index('GT')
#         return value_str.split(':')[gt_index]
#     except (ValueError, IndexError):
#         return None
        
# def count_genotype_shifts(original_vcf: str, beagle_vcf: str, target_chrom: str) -> Counter:
#     shifts = Counter()
#     gt_shifts = Counter()
#     gt_shift_count = 0
#     variant_count = 0
#     shift_count = 0
#     match_count = 0
#     mismatch_count = 0
    
#     print(f"\nProcessing VCF files:")
#     print(f"Original: {original_vcf}")
#     print(f"Beagle: {beagle_vcf}")
    
#     with gzip.open(original_vcf, 'rt') as orig, gzip.open(beagle_vcf, 'rt') as beagle:
#         # Skip headers
#         for line in orig:
#             if not line.startswith('#'):
#                 break
#         for line in beagle:
#             if not line.startswith('#'):
#                 break
        
#         # Create dictionary of beagle variants
#         beagle_variants = {}
#         for line in beagle:
#             fields = line.strip().split('\t')
#             chrom, pos, _, ref, alt, _, _, info, format_str, value_str = fields
#             if chrom == target_chrom:
#                 gt = extract_gt_from_format(format_str, value_str)
#                 if gt:
#                     beagle_variants[int(pos)] = (ref, alt, gt)
        
#         print(f"Loaded {len(beagle_variants)} variants from beagle VCF")
        
#         # Process original VCF
#         for line in orig:
#             fields = line.strip().split('\t')
#             # format: CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	151507
#             # chrom, pos, _, ref, alt, _, _, _, format_str, value_str = fields
#             chrom, pos, _, ref, alt, _, _, info, format_str, value_str = fields
#             # print(f"format and value: {format_str} {value_str}")
#             if chrom == target_chrom:
#                 variant_count += 1
#                 pos = int(pos)
                
#                 if pos in beagle_variants:
#                     match_count += 1
#                     beagle_ref, beagle_alt, beagle_gt = beagle_variants[pos]
                    
#                     # Verify ref/alt alleles match
#                     if ref != beagle_ref:
#                         mismatch_count += 1
#                         continue
                    
#                     # Get original GT
#                     orig_gt = extract_gt_from_format(format_str, value_str)
#                     # print(f"Original GT: {orig_gt}")
                    
#                     if orig_gt and beagle_gt:
#                         # Parse into actual alleles
#                         orig_alleles = parse_genotype(orig_gt, ref, alt)
#                         beagle_alleles = parse_genotype(beagle_gt, beagle_ref, beagle_alt)
#                         # skip beagle_gt != 0 or orig_gt == 0
#                         # For Monopogen's approach (only 0/1 -> 0/0)
#                         if not (beagle_gt == '0/0' and orig_gt in ('0/1', '1/0')):
#                             continue
#                         if orig_alleles and beagle_alleles and orig_alleles != beagle_alleles:
#                             shift_key = f"{orig_alleles}->{beagle_alleles}"
#                             gt_shifts_key = f"{orig_gt}->{beagle_gt}"
#                             shifts[shift_key] += 1
#                             gt_shifts[gt_shifts_key] += 1
#                             shift_count += 1
#                             gt_shift_count += 1

                            
#                             # Debug print for first few shifts
#                             # if shift_count <= 5:
#                             #     print(f"\nShift example #{shift_count}:")
#                             #     print(f"Position: {pos}")
#                             #     print(f"Original GT: {orig_gt} -> {orig_alleles}")
#                             #     print(f"Beagle GT: {beagle_gt} -> {beagle_alleles}")
                
#                 if variant_count % 100000 == 0:
#                     print(f"Processed {variant_count} variants, found {shift_count} shifts")
    
#     print(f"\nSummary for {target_chrom}:")
#     print(f"Total variants processed: {variant_count}")
#     print(f"Matching positions found: {match_count}")
#     print(f"Total shifts found: {shift_count}")
#     print(f"Allele mismatch count: {mismatch_count}")
#     print(f"Genotype shifts: {gt_shift_count}")
#     return shifts, gt_shifts


def main():
    parser = argparse.ArgumentParser(description="Count genotype shifts after Beagle processing")
    parser.add_argument("--section_id", required=True, help="Section ID")
    parser.add_argument("--chromosomes", nargs="+", help="Specific chromosomes to process")
    args = parser.parse_args()
    
    setup_environment()
    
    # original count shifting code:

    # chromosomes = args.chromosomes if args.chromosomes else [f"chr{i}" for i in range(1, 23)]
    # all_shifts = Counter()
    # all_gt_shifts = Counter()
    
    # for chrom in chromosomes:
    #     print(f"\nProcessing {chrom}...")
        
    #     orig_vcf = f"{PATH_CONFIG['PROJECT_DIR']}/data/dlpfc/{args.section_id}/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr_gt.vcf.gz"
    #     beagle_vcf = f"{PATH_CONFIG['PROJECT_DIR']}/data/dlpfc/{args.section_id}/output_VCFs/beagle_noimput/{chrom}.vcf.gz"
        
    #     shifts, gt_shifts = count_genotype_shifts(orig_vcf, beagle_vcf, chrom)
    #     all_shifts.update(shifts)
    #     all_gt_shifts.update(gt_shifts)
        
    #     print(f"\nChromosome {chrom} shifts:")
    #     for shift_type, count in shifts.most_common():
    #         print(f"{shift_type}: {count}")

    #     print(f"\n GenoType shifting:")
    #     for shift_type, count in gt_shifts.most_common():
    #         print(f"{shift_type}: {count}")

    # print("\nTotal shifts across all chromosomes:")
    # for shift_type, count in all_shifts.most_common():
    #     print(f"{shift_type}: {count}")
    
    # for shift_type, count in all_gt_shifts.most_common():
    #     print(f"{shift_type}: {count}")



#     # New code for BAF analysis
    project_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
    output_dir = os.path.join(project_dir, "data/dlpfc", args.section_id, "metrics/beagle")
    os.makedirs(output_dir, exist_ok=True)
    
    chromosomes = args.chromosomes if args.chromosomes else [f"chr{i}" for i in range(1, 23)]
    analyzer = GenotypeTransitionAnalyzer()
    
    for chrom in chromosomes:
        print(f"\nProcessing {chrom}...")
        
        orig_vcf = f"{project_dir}/data/dlpfc/{args.section_id}/output_VCFs/mpileup_multi_bam/merged_multi_bam.chr_gt.vcf.gz"
        beagle_vcf = f"{project_dir}/data/dlpfc/{args.section_id}/output_VCFs/beagle/{chrom}.with_baf.vcf.gz"
        
        results = analyzer.analyze_transitions(orig_vcf, beagle_vcf, chrom)
        
        print(f"\nChromosome {chrom} summary:")
        print(f"Total variants processed: {results['variant_count']}")
        print(f"Matching positions found: {results['match_count']}")
        
    # Plot BAF distributions
    analyzer.plot_all_distributions(output_dir)


if __name__ == "__main__":
    main()

    # usage
    # python scripts/postprocess/count_genotype_shifting.py --section_id 151507 --chromosomes chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22
    # ctrl + F "change to plot 0/1 -> 0/0 (12 variance) or all of them" to change the plot type