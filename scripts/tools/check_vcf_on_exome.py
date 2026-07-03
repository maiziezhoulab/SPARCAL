#!/usr/bin/env python3
"""
Check if variants in a VCF file fall within exome regions defined in a BED file.
Currently only supports hg19 genome build, focusing on TUMOR datasets.
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import matplotlib.pyplot as plt

# Constants
DEFAULT_EXOME_BED = "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed"
OUTPUT_DIR = "/data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis"

# Dataset configurations
DATASET_CONFIGS = {
    "P4_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "hg19"
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "hg19"
    }
}


def load_exome_regions(bed_file: str, verbose: bool = False) -> Dict[str, List[Tuple[int, int]]]:
    """
    Load exome regions from a BED file into a dictionary by chromosome.
    
    Args:
        bed_file: Path to the BED file defining exome regions
        verbose: Print detailed information
        
    Returns:
        Dictionary mapping chromosome names to lists of (start, end) position tuples
    """
    if verbose:
        print(f"Loading exome regions from {bed_file}...")
    
    regions = defaultdict(list)
    region_count = 0
    
    with open(bed_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            
            fields = line.strip().split('\t')
            if len(fields) < 3:
                continue
                
            chrom = fields[0]
            # Standardize chromosome name (ensure it has "chr" prefix)
            if not chrom.startswith('chr'):
                chrom = f"chr{chrom}"
                
            try:
                start = int(fields[1])
                end = int(fields[2])
                regions[chrom].append((start, end))
                region_count += 1
            except ValueError:
                if verbose:
                    print(f"Warning: Skipping invalid region: {line.strip()}")
    
    # Sort the regions for each chromosome
    for chrom in regions:
        regions[chrom].sort()
    
    if verbose:
        print(f"Loaded {region_count:,} exome regions across {len(regions)} chromosomes")
        total_bp = sum(end - start for chrom_regions in regions.values() for start, end in chrom_regions)
        print(f"Total exome size: {total_bp:,} bp")
        
        # Print some example regions
        print("\nExample regions:")
        example_count = 0
        for chrom in sorted(regions.keys()):
            if example_count >= 5:
                break
            if regions[chrom]:
                for start, end in regions[chrom][:2]:  # Show up to 2 regions per chromosome
                    if example_count >= 5:
                        break
                    print(f"  {chrom}:{start}-{end} ({end - start} bp)")
                    example_count += 1
    
    return regions


def is_in_exome(chrom: str, pos: int, exome_regions: Dict[str, List[Tuple[int, int]]]) -> bool:
    """
    Check if a genomic position falls within exome regions.
    
    Args:
        chrom: Chromosome name
        pos: Position to check
        exome_regions: Dictionary of exome regions by chromosome
        
    Returns:
        True if position is in exome, False otherwise
    """
    # Standardize chromosome name
    if not chrom.startswith('chr'):
        chrom = f"chr{chrom}"
        
    if chrom not in exome_regions:
        return False
    
    # Binary search to find the region
    regions = exome_regions[chrom]
    low = 0
    high = len(regions) - 1
    
    while low <= high:
        mid = (low + high) // 2
        start, end = regions[mid]
        
        if pos < start:
            high = mid - 1
        elif pos >= end:
            low = mid + 1
        else:
            return True  # Position is within this region
    
    return False


def analyze_vcf_exome_coverage(vcf_file: str, exome_regions: Dict[str, List[Tuple[int, int]]], 
                              verbose: bool = False) -> Dict:
    """
    Analyze what proportion of variants in a VCF file fall within exome regions.
    
    Args:
        vcf_file: Path to the VCF file to analyze
        exome_regions: Dictionary of exome regions by chromosome
        verbose: Print detailed information
        
    Returns:
        Dictionary with analysis results
    """
    if not os.path.exists(vcf_file):
        print(f"Error: VCF file not found: {vcf_file}")
        return {
            'total_variants': 0,
            'exome_variants': 0,
            'exome_percentage': 0,
            'variants_by_chrom': {},
            'exome_variants_by_chrom': {}
        }
    
    # Check if file is gzipped
    is_gzipped = vcf_file.endswith('.gz')
    opener = gzip.open if is_gzipped else open
    mode = 'rt' if is_gzipped else 'r'
    
    if verbose:
        print(f"Analyzing VCF file: {vcf_file}")
    
    # Counters
    total_variants = 0
    exome_variants = 0
    variants_by_chrom = defaultdict(int)
    exome_variants_by_chrom = defaultdict(int)
    
    with opener(vcf_file, mode) as f:
        for i, line in enumerate(f):
            # Progress reporting for large files
            if verbose and i > 0 and i % 100000 == 0:
                print(f"  Processed {i:,} lines...")
            
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            if len(fields) < 2:
                continue
                
            try:
                chrom = fields[0]
                pos = int(fields[1])
                
                # Standardize chromosome name
                if not chrom.startswith('chr'):
                    chrom = f"chr{chrom}"
                
                # Increment counters
                total_variants += 1
                variants_by_chrom[chrom] += 1
                
                # Check if variant is in exome
                if is_in_exome(chrom, pos, exome_regions):
                    exome_variants += 1
                    exome_variants_by_chrom[chrom] += 1
            except ValueError as e:
                if verbose:
                    print(f"Warning: Error parsing line {i+1}: {str(e)}")
                continue
    
    # Calculate statistics
    exome_percentage = (exome_variants / total_variants * 100) if total_variants > 0 else 0
    
    if verbose:
        print(f"\nAnalysis complete:")
        print(f"  Total variants: {total_variants:,}")
        print(f"  Variants in exome: {exome_variants:,} ({exome_percentage:.2f}%)")
        
        # Print per-chromosome statistics
        print("\nVariants by chromosome:")
        for chrom in sorted(variants_by_chrom.keys()):
            chrom_total = variants_by_chrom[chrom]
            chrom_exome = exome_variants_by_chrom[chrom]
            chrom_percentage = (chrom_exome / chrom_total * 100) if chrom_total > 0 else 0
            print(f"  {chrom}: {chrom_exome:,}/{chrom_total:,} in exome ({chrom_percentage:.2f}%)")
    
    return {
        'total_variants': total_variants,
        'exome_variants': exome_variants,
        'exome_percentage': exome_percentage,
        'variants_by_chrom': dict(variants_by_chrom),
        'exome_variants_by_chrom': dict(exome_variants_by_chrom)
    }


def get_default_vcf_path(dataset_name: str, section_id: str, quality_filter: str = "baseQ0mapQ0") -> str:
    """
    Get default path to the beagle filtered in VCF file.
    
    Args:
        dataset_name: Name of the dataset
        section_id: Section ID
        quality_filter: Quality filter used
        
    Returns:
        Path to the default VCF file
    """
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    if dataset_config["has_sections"] and not section_id:
        raise ValueError(f"Dataset {dataset_name} requires a section_id")
        
    base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
    
    if dataset_config["has_sections"]:
        data_path = dataset_config["output_dir"].format(section_id=section_id)
    else:
        data_path = dataset_config["output_dir"]
        
    return os.path.join(
        base_dir,
        data_path,
        "output_VCFs/beagle",
        quality_filter,
        "all_filtered_in.vcf.gz"
    )


def plot_exome_coverage(results: Dict, output_file: str, dataset_name: str, section_id: Optional[str] = None):
    """
    Create a bar plot showing exome coverage by chromosome.
    
    Args:
        results: Results dictionary from analyze_vcf_exome_coverage
        output_file: Path to save the plot
        dataset_name: Name of the dataset
        section_id: Section ID (if applicable)
    """
    # Extract data for plotting
    chromosomes = sorted(results['variants_by_chrom'].keys())
    
    # Calculate percentages by chromosome
    percentages = []
    for chrom in chromosomes:
        total = results['variants_by_chrom'][chrom]
        exome = results['exome_variants_by_chrom'].get(chrom, 0)
        percentage = (exome / total * 100) if total > 0 else 0
        percentages.append(percentage)
    
    # Create the plot
    plt.figure(figsize=(15, 8))
    
    # Bar plot for percentages
    plt.bar(chromosomes, percentages, color='steelblue')
    
    # Add horizontal line for overall percentage
    plt.axhline(y=results['exome_percentage'], color='red', linestyle='--', 
                label=f'Overall: {results["exome_percentage"]:.2f}%')
    
    # Add labels and title
    plt.xlabel('Chromosome')
    plt.ylabel('Variants in Exome (%)')
    
    title = f'Exome Coverage - {dataset_name}'
    if section_id:
        title += f' (Section {section_id})'
    plt.title(title)
    
    # Add text with summary statistics
    summary = (
        f"Total variants: {results['total_variants']:,}\n"
        f"Variants in exome: {results['exome_variants']:,}\n"
        f"Percentage in exome: {results['exome_percentage']:.2f}%"
    )
    plt.annotate(summary, xy=(0.02, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', ha='left')
    
    plt.legend()
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_file, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Check if variants in a VCF file fall within exome regions.')
    
    parser.add_argument('--vcf', help='Path to the VCF file to analyze')
    parser.add_argument('--dataset', choices=list(DATASET_CONFIGS.keys()),
                      help='Dataset name (for default VCF path)')
    parser.add_argument('--section-id', help='Section ID (for default VCF path)')
    parser.add_argument('--bed', default=DEFAULT_EXOME_BED,
                      help=f'Path to the BED file defining exome regions (default: {DEFAULT_EXOME_BED})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                      help=f'Directory to save output files (default: {OUTPUT_DIR})')
    parser.add_argument('--quality-filter', default='baseQ0mapQ0',
                      help='Quality filter to use for default VCF path (default: baseQ0mapQ0)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed information')
    
    args = parser.parse_args()
    
    # Check if either VCF path or dataset info is provided
    if not args.vcf and not args.dataset:
        parser.error("Either --vcf or --dataset must be specified")
    
    # Get VCF file path
    vcf_path = args.vcf
    if not vcf_path and args.dataset:
        if args.dataset not in DATASET_CONFIGS:
            parser.error(f"Unknown dataset: {args.dataset}")
        
        dataset_config = DATASET_CONFIGS[args.dataset]
        if dataset_config["has_sections"] and not args.section_id:
            if "section_ids" in dataset_config:
                valid_sections = dataset_config["section_ids"]
                parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
            else:
                parser.error(f"Dataset {args.dataset} requires --section-id")
        
        try:
            vcf_path = get_default_vcf_path(args.dataset, args.section_id, args.quality_filter)
        except ValueError as e:
            parser.error(str(e))
    
    # Check if VCF file exists
    if not os.path.exists(vcf_path):
        parser.error(f"VCF file not found: {vcf_path}")
    
    # Check if BED file exists
    if not os.path.exists(args.bed):
        parser.error(f"BED file not found: {args.bed}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load exome regions
    exome_regions = load_exome_regions(args.bed, args.verbose)
    
    # Analyze VCF file
    results = analyze_vcf_exome_coverage(vcf_path, exome_regions, args.verbose)
    
    # Create output file base name
    if args.dataset:
        output_base = os.path.join(args.output_dir, f"{args.dataset}")
        if args.section_id:
            output_base += f"_section{args.section_id}"
    else:
        vcf_basename = os.path.basename(vcf_path)
        output_base = os.path.join(args.output_dir, os.path.splitext(vcf_basename)[0])
        if output_base.endswith('.vcf'):
            output_base = output_base[:-4]
    
    # Save results to text file
    output_text = f"{output_base}_exome_coverage.txt"
    with open(output_text, 'w') as f:
        f.write(f"Exome Coverage Analysis\n")
        f.write(f"======================\n\n")
        if args.dataset:
            f.write(f"Dataset: {args.dataset}\n")
            if args.section_id:
                f.write(f"Section: {args.section_id}\n")
        f.write(f"VCF file: {vcf_path}\n")
        f.write(f"BED file: {args.bed}\n\n")
        
        f.write(f"Total variants: {results['total_variants']:,}\n")
        f.write(f"Variants in exome: {results['exome_variants']:,}\n")
        f.write(f"Percentage in exome: {results['exome_percentage']:.2f}%\n\n")
        
        f.write(f"Variants by chromosome:\n")
        f.write(f"----------------------\n")
        for chrom in sorted(results['variants_by_chrom'].keys()):
            total = results['variants_by_chrom'][chrom]
            exome = results['exome_variants_by_chrom'].get(chrom, 0)
            percentage = (exome / total * 100) if total > 0 else 0
            f.write(f"{chrom}: {exome:,}/{total:,} in exome ({percentage:.2f}%)\n")
    
    print(f"Results saved to {output_text}")
    
    # Create plot
    output_plot = f"{output_base}_exome_coverage.png"
    plot_exome_coverage(results, output_plot, args.dataset or os.path.basename(vcf_path), args.section_id)
    print(f"Plot saved to {output_plot}")


if __name__ == "__main__":
    main()

# check this vcf: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_without00.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_without00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check vcf: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/negative_training_without00.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/negative_training_without00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome

# check vcf (hg19): /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/positive_training.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/positive_training.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check vcf (hg19): /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/results/high_confidence_no00.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/results/high_confidence_no00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check vcf (hg19): /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_wo00.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P6_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_wo00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check vcf (hg19): /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_wo00.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in_wo00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# For dlpfc, bed file: /data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed
# DLPFC check:  
# vcf: /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/SVMModel/baseQ0mapQ0/positive_training.vcf.gz
# python scripts/tools/check_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/SVMModel/baseQ0mapQ0/positive_training.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check this vcf: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt_exclude00.vcf.gz
# python scripts/tools/check_vcf_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt_exclude00.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose

# check this vcf:  /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/SeqErrModel/baseQ0mapQ0/consecutive_denovo.vcf.gz
# python scripts/tools/check_vcf_on_exome.py --vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/SeqErrModel/baseQ0mapQ0/consecutive_denovo.vcf.gz --bed /data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/exome_analysis --verbose