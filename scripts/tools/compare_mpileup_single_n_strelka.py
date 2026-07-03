#!/usr/bin/env python3
"""
Compare variant calls between mpileup single-BAM and Strelka approaches
for spatial transcriptomics data.

This script compares the variant calls from mpileup_single_bam and Strelka pipelines
for the same BAM files and visualizes the overlaps using Venn diagrams.
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Set the number of threads for OpenBLAS to 1
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
from collections import defaultdict, Counter
import subprocess
from typing import Dict, List, Set, Tuple, Optional


def parse_vcf(vcf_path: str, min_qual: float = 0) -> Set[str]:
    """
    Parse variants from a VCF file and return a set of variant identifiers.
    
    Args:
        vcf_path: Path to the VCF file
        min_qual: Minimum QUAL value to include variant
        
    Returns:
        Set of variant identifiers in format "CHROM_POS_REF_ALT"
    """
    variants = set()
    
    # Check if file exists
    if not os.path.exists(vcf_path):
        print(f"WARNING: VCF file not found: {vcf_path}")
        return variants
    
    # Open file with gzip if it's compressed
    opener = gzip.open if vcf_path.endswith('.gz') else open
    mode = 'rt' if vcf_path.endswith('.gz') else 'r'
    
    with opener(vcf_path, mode) as f:
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            if len(fields) < 8:
                continue
                
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            qual = float(fields[5]) if fields[5] != '.' else 0
            
            # Skip if below quality threshold
            if qual < min_qual:
                continue
                
            # Skip complex variants for simplicity
            if ',' in alt:
                continue
                
            # Create variant identifier
            variant_id = f"{chrom}_{pos}_{ref}_{alt}"
            variants.add(variant_id)
    
    return variants


def extract_genotype(format_str: str, sample_str: str) -> Optional[str]:
    """Extract genotype (GT) from FORMAT and sample fields."""
    try:
        format_fields = format_str.split(':')
        gt_idx = format_fields.index('GT')
        sample_fields = sample_str.split(':')
        return sample_fields[gt_idx]
    except (ValueError, IndexError):
        return None


def extract_genotype_from_pl(format_str: str, sample_str: str) -> Optional[str]:
    """Extract genotype (GT) from FORMAT and sample fields using PL values."""
    try:
        format_fields = format_str.split(':')
        pl_idx = format_fields.index('PL')
        sample_fields = sample_str.split(':')
        pl_values = list(map(int, sample_fields[pl_idx].split(',')))
        
        # Determine genotype based on the lowest PL value
        if len(pl_values) == 3:
            min_pl_idx = pl_values.index(min(pl_values))
            if min_pl_idx == 0:
                return '0/0'
            elif min_pl_idx == 1:
                return '0/1'
            elif min_pl_idx == 2:
                return '1/1'
    except (ValueError, IndexError):
        return None

    return None


def parse_vcf_with_genotypes(vcf_path: str, min_qual: float = 0) -> Dict[str, str]:
    """
    Parse variants from a VCF file and return a dict of variant identifiers to genotypes.
    
    Args:
        vcf_path: Path to the VCF file
        min_qual: Minimum QUAL value to include variant
        
    Returns:
        Dict mapping variant identifiers to genotypes
    """
    variants = {}
    
    # Check if file exists
    if not os.path.exists(vcf_path):
        print(f"WARNING: VCF file not found: {vcf_path}")
        return variants
    
    # Open file with gzip if it's compressed
    opener = gzip.open if vcf_path.endswith('.gz') else open
    mode = 'rt' if vcf_path.endswith('.gz') else 'r'
    
    with opener(vcf_path, mode) as f:
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            if len(fields) < 10:  # Need FORMAT and sample fields
                continue
                
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            qual = float(fields[5]) if fields[5] != '.' else 0
            
            # Skip if below quality threshold
            if qual < min_qual:
                continue
                
            # Skip complex variants for simplicity
            if ',' in alt:
                continue
            
            # Extract genotype
            gt = extract_genotype(fields[8], fields[9])
            if not gt:
                # Infer genotype from PL values if GT is not available
                gt = extract_genotype_from_pl(fields[8], fields[9])
                if not gt:
                    gt = './.'
            if gt == '0/0':
                continue
            # Create variant identifier
            variant_id = f"{chrom}_{pos}_{ref}_{alt}"
            variants[variant_id] = gt
    
    return variants


def find_vcf_files(barcode: str, dataset: str = "dlpfc/151507") -> Tuple[str, str]:
    """
    Find the mpileup and Strelka VCF files for a given barcode.
    
    Args:
        barcode: Cell barcode
        dataset: Dataset name (dlpfc/151507, P4, or P6)
        
    Returns:
        Tuple of (mpileup_vcf_path, strelka_vcf_path)
    """
    if dataset in ("P4", "P6"):
        base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling/data/{dataset}"
    else:
        base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling/data/{dataset}"
    
    # Mpileup VCF path
    mpileup_vcf = os.path.join(
        base_dir, 
        "output_VCFs/mpileup_single_bam/baseQ0mapQ0", 
        f"{barcode}.vcf.gz"
    )
    
    # Strelka VCF path
    if dataset in ("P4", "P6"):
        strelka_vcf = os.path.join(
            base_dir, 
            "output_VCFs/strelka/strelkaQ0", 
            f"{barcode}/results/variants/somatic.snvs.vcf.gz"
        )
    else:
        strelka_vcf = os.path.join(
            base_dir, 
            "output_VCFs/strelka/strelkaQ0", 
            f"{barcode}/results/variants/variants.vcf.gz"
        )
    
    return mpileup_vcf, strelka_vcf


def analyze_genotype_concordance(mpileup_genotypes: Dict[str, str], 
                                strelka_genotypes: Dict[str, str]) -> pd.DataFrame:
    """
    Analyze genotype concordance between mpileup and Strelka calls.
    
    Args:
        mpileup_genotypes: Dict mapping variant IDs to genotypes from mpileup
        strelka_genotypes: Dict mapping variant IDs to genotypes from Strelka
        
    Returns:
        DataFrame with genotype concordance statistics
    """
    # Find common variants
    common_variants = set(mpileup_genotypes.keys()) & set(strelka_genotypes.keys())
    
    # Collect genotype pairs
    genotype_pairs = []
    for variant in common_variants:
        mpileup_gt = mpileup_genotypes[variant]
        strelka_gt = strelka_genotypes[variant]
        
        # Only compare 0/1 and 1/1 genotypes
        if mpileup_gt in {'0/1', '1/1'} and strelka_gt in {'0/1', '1/1'}:
            genotype_pairs.append((mpileup_gt, strelka_gt))
    
    # Count occurrences
    genotype_counts = Counter(genotype_pairs)
    
    # Convert to DataFrame
    df_rows = []
    for (mpileup_gt, strelka_gt), count in genotype_counts.items():
        concordant = mpileup_gt == strelka_gt
        df_rows.append({
            'Mpileup_GT': mpileup_gt,
            'Strelka_GT': strelka_gt,
            'Count': count,
            'Concordant': concordant
        })
    
    df = pd.DataFrame(df_rows)
    if not df.empty:
        df = df.sort_values(['Concordant', 'Count'], ascending=[False, False])
    
    return df


def analyze_variant_types(variants: Set[str]) -> Dict[str, int]:
    """
    Analyze the types of variants in a set.
    
    Args:
        variants: Set of variant identifiers in format "CHROM_POS_REF_ALT"
        
    Returns:
        Dict mapping variant types to counts
    """
    variant_types = defaultdict(int)
    
    for variant in variants:
        parts = variant.split('_')
        if len(parts) < 4:
            continue
            
        ref = parts[2]
        alt = parts[3]
        
        if len(ref) == 1 and len(alt) == 1:
            # SNV
            variant_types['SNV'] += 1
        elif len(ref) > len(alt):
            # Deletion
            variant_types['Deletion'] += 1
        elif len(ref) < len(alt):
            # Insertion
            variant_types['Insertion'] += 1
        else:
            # MNV
            variant_types['MNV'] += 1
    
    return dict(variant_types)


def create_pipeline_comparison(barcodes: List[str], min_qual: float = 0, 
                             output_dir: str = "comparison_results", dataset: str = "dlpfc/151507"):
    """
    Create comparison between mpileup and Strelka results for specified barcodes.
    
    Args:
        barcodes: List of cell barcodes
        min_qual: Minimum QUAL value to include variant
        output_dir: Directory to save output files
        dataset: Dataset name (dlpfc/151507, P4, or P6)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Per-barcode results
    all_mpileup_variants = set()
    all_strelka_variants = set()
    all_mpileup_genotypes = {}
    all_strelka_genotypes = {}
    per_barcode_results = []
    
    # Process each barcode
    for barcode in barcodes:
        print(f"Processing barcode: {barcode}")
        
        # Find VCF files
        mpileup_vcf, strelka_vcf = find_vcf_files(barcode, dataset)
        
        # Check if files exist
        if not os.path.exists(mpileup_vcf):
            print(f"  WARNING: Mpileup VCF not found: {mpileup_vcf}")
            continue
            
        if not os.path.exists(strelka_vcf):
            print(f"  WARNING: Strelka VCF not found: {strelka_vcf}")
            continue
        
        # Parse VCFs with genotypes
        mpileup_genotypes = parse_vcf_with_genotypes(mpileup_vcf, min_qual)
        strelka_genotypes = parse_vcf_with_genotypes(strelka_vcf, min_qual)
        
        # Extract variant sets
        mpileup_variants = set(mpileup_genotypes.keys())
        strelka_variants = set(strelka_genotypes.keys())
        
        # Update all variant sets and genotypes
        all_mpileup_variants.update(mpileup_variants)
        all_strelka_variants.update(strelka_variants)
        all_mpileup_genotypes.update(mpileup_genotypes)
        all_strelka_genotypes.update(strelka_genotypes)
        
        # Find common and unique variants
        common_variants = mpileup_variants & strelka_variants
        mpileup_only = mpileup_variants - strelka_variants
        strelka_only = strelka_variants - mpileup_variants
        
        # Analyze variant types
        mpileup_types = analyze_variant_types(mpileup_variants)
        strelka_types = analyze_variant_types(strelka_variants)
        common_types = analyze_variant_types(common_variants)
        
        # Analyze genotype concordance
        genotype_concordance = analyze_genotype_concordance(mpileup_genotypes, strelka_genotypes)
        
        # Save genotype concordance
        if not genotype_concordance.empty:
            concordance_file = os.path.join(output_dir, f"{barcode}_genotype_concordance.csv")
            genotype_concordance.to_csv(concordance_file, index=False)
        
        # Plot Venn diagrams for 0/1, 1/1, and both genotypes together
        for gt in ['0/1', '1/1', 'both']:
            if gt == 'both':
                mpileup_set = {k for k, v in mpileup_genotypes.items() if v in {'0/1', '1/1'}}
                strelka_set = {k for k, v in strelka_genotypes.items() if v in {'0/1', '1/1'}}
                venn_title = f"Variant Call Comparison for {barcode} (0/1 and 1/1)"
                venn_path = os.path.join(output_dir, f"{barcode}_venn_both.png")
            else:
                mpileup_set = {k for k, v in mpileup_genotypes.items() if v == gt}
                strelka_set = {k for k, v in strelka_genotypes.items() if v == gt}
                venn_title = f"Variant Call Comparison for {barcode} ({gt})"
                venn_path = os.path.join(output_dir, f"{barcode}_venn_{gt.replace('/', '')}.png")
            
            plot_venn_diagram(mpileup_set, strelka_set, venn_title, venn_path)
        
        # Store results
        per_barcode_results.append({
            'Barcode': barcode,
            'Mpileup_Variants': len(mpileup_variants),
            'Strelka_Variants': len(strelka_variants),
            'Common_Variants': len(common_variants),
            'Mpileup_Only': len(mpileup_only),
            'Strelka_Only': len(strelka_only),
            'Mpileup_SNVs': mpileup_types.get('SNV', 0),
            'Mpileup_Insertions': mpileup_types.get('Insertion', 0),
            'Mpileup_Deletions': mpileup_types.get('Deletion', 0),
            'Strelka_SNVs': strelka_types.get('SNV', 0),
            'Strelka_Insertions': strelka_types.get('Insertion', 0),
            'Strelka_Deletions': strelka_types.get('Deletion', 0),
            'Concordant_Genotypes': genotype_concordance['Concordant'].sum() if not genotype_concordance.empty else 0
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(per_barcode_results)
    
    # Save summary
    summary_path = os.path.join(output_dir, "comparison_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Plot aggregate Venn diagrams
    if barcodes:
        for gt in ['0/1', '1/1', 'both']:
            if gt == 'both':
                aggregate_sets = {
                    'Mpileup': {k for k, v in all_mpileup_genotypes.items() if v in {'0/1', '1/1'}},
                    'Strelka': {k for k, v in all_strelka_genotypes.items() if v in {'0/1', '1/1'}}
                }
                aggregate_title = f"Aggregate Variant Call Comparison ({len(barcodes)} barcodes) (0/1 and 1/1)"
                aggregate_path = os.path.join(output_dir, "aggregate_venn_both.png")
            else:
                aggregate_sets = {
                    'Mpileup': {k for k, v in all_mpileup_genotypes.items() if v == gt},
                    'Strelka': {k for k, v in all_strelka_genotypes.items() if v == gt}
                }
                aggregate_title = f"Aggregate Variant Call Comparison ({len(barcodes)} barcodes) ({gt})"
                aggregate_path = os.path.join(output_dir, f"aggregate_venn_{gt.replace('/', '')}.png")
            
            plot_venn_diagram(aggregate_sets['Mpileup'], aggregate_sets['Strelka'], aggregate_title, aggregate_path)
    
    print(f"\nComparison complete. Results saved to: {output_dir}")
    print(f"Summary file: {summary_path}")
    
    # Print brief summary
    if not summary_df.empty:
        print("\nBrief Summary:")
        print(f"Total barcodes analyzed: {len(summary_df)}")
        print(f"Average Mpileup variants per barcode: {summary_df['Mpileup_Variants'].mean():.1f}")
        print(f"Average Strelka variants per barcode: {summary_df['Strelka_Variants'].mean():.1f}")
        print(f"Average common variants per barcode: {summary_df['Common_Variants'].mean():.1f}")
        
        # Calculate overall concordance percentage
        if summary_df['Common_Variants'].sum() > 0:
            concordance_pct = (summary_df['Concordant_Genotypes'].sum() / summary_df['Common_Variants'].sum()) * 100
            print(f"Overall genotype concordance: {concordance_pct:.1f}%")


def plot_venn_diagram(mpileup_set: Set[str], strelka_set: Set[str], title: str, output_path: str):
    """
    Plot a Venn diagram for two sets (mpileup and strelka) and print the set sizes.
    
    Args:
        mpileup_set: Set of variants from mpileup
        strelka_set: Set of variants from strelka
        title: Title for the plot
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Print set sizes
    print(f"Mpileup set size: {len(mpileup_set)}")
    print(f"Strelka set size: {len(strelka_set)}")
    print(f"Common variants: {len(mpileup_set & strelka_set)}")
    
    # Plot Venn diagram
    venn = venn2([mpileup_set, strelka_set], set_labels=('Mpileup', 'Strelka'))
    
    # Calculate set sizes for labels
    mpileup_only = len(mpileup_set - strelka_set)
    strelka_only = len(strelka_set - mpileup_set)
    intersection = len(mpileup_set & strelka_set)
    
    # Add set sizes to the Venn diagram
    if venn.get_patch_by_id('10'):
        venn.get_patch_by_id('10').set_alpha(0.7)
        venn.get_label_by_id('10').set_text(f'Mpileup only\n({mpileup_only})')
    
    if venn.get_patch_by_id('01'):
        venn.get_patch_by_id('01').set_alpha(0.7)
        venn.get_label_by_id('01').set_text(f'Strelka only\n({strelka_only})')
    
    if venn.get_patch_by_id('11'):
        venn.get_patch_by_id('11').set_alpha(0.7)
        venn.get_label_by_id('11').set_text(f'Common\n({intersection})')
    
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def get_all_barcodes_from_directory(directory: str, pattern: str = "*.vcf.gz") -> List[str]:
    """
    Get all barcodes from a directory by looking at VCF files.
    
    Args:
        directory: Directory containing VCF files
        pattern: File pattern to match
        
    Returns:
        List of barcodes extracted from filenames
    """
    vcf_files = glob.glob(os.path.join(directory, pattern))
    barcodes = [os.path.basename(f).replace('.vcf.gz', '') for f in vcf_files]
    return barcodes


def process_all_barcodes(dataset: str, output_dir: str, min_qual: float = 0,
                       workers: int = 4, limit: int = None) -> None:
    """
    Process all barcodes found in the mpileup output directory.
    
    Args:
        dataset: Dataset name (dlpfc/151507, P4, or P6)
        output_dir: Directory to save output files
        min_qual: Minimum QUAL value to include variant
        workers: Number of parallel workers
        limit: Maximum number of barcodes to process (None for all)
    """
    print(f"Finding barcodes in dataset: {dataset}")
    
    # Determine the directory based on dataset name
    if dataset in ("P4", "P6"):
        base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling/data/{dataset}/output_VCFs/mpileup_single_bam/baseQ0mapQ0"
    else:
        base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling/data/{dataset}/output_VCFs/mpileup_single_bam/baseQ0mapQ0"
    
    # Get all barcodes from the directory
    barcodes = get_all_barcodes_from_directory(base_dir)
    
    if not barcodes:
        print(f"No barcodes found in {base_dir}")
        return
    
    # Limit the number of barcodes if specified
    if limit is not None and limit > 0:
        barcodes = barcodes[:limit]
    
    print(f"Found {len(barcodes)} barcodes")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the list of barcodes to process
    with open(os.path.join(output_dir, "processed_barcodes.txt"), 'w') as f:
        for barcode in barcodes:
            f.write(f"{barcode}\n")
    
    # Process barcodes in parallel
    if workers > 1:
        process_barcodes_parallel(barcodes, min_qual, output_dir, dataset, workers)
    else:
        create_pipeline_comparison(barcodes, min_qual, output_dir, dataset)


def process_barcodes_parallel(barcodes: List[str], min_qual: float, output_dir: str,
                            dataset: str, workers: int) -> None:
    """
    Process barcodes in parallel using ThreadPoolExecutor.
    
    Args:
        barcodes: List of barcodes to process
        min_qual: Minimum QUAL value to include variant
        output_dir: Directory to save output files
        dataset: Dataset name
        workers: Number of parallel workers
    """
    # Create a directory for each barcode to avoid file conflicts
    barcode_results = []
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        
        # Submit jobs
        for i, barcode_batch in enumerate(chunks(barcodes, 10)):  # Process in batches of 10
            batch_dir = os.path.join(output_dir, f"batch_{i}")
            futures[executor.submit(create_pipeline_comparison, barcode_batch, min_qual, batch_dir, dataset)] = i
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batches"):
            batch_idx = futures[future]
            try:
                future.result()
                print(f"Completed batch {batch_idx}")
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
    
    # Generate aggregate summary
    generate_aggregate_summary(output_dir)


def chunks(lst, n):
    """Split a list into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def generate_aggregate_summary(output_dir: str) -> None:
    """
    Generate an aggregate summary from all batch results.
    
    Args:
        output_dir: Base output directory containing batch directories
    """
    # Find all summary files
    summary_files = glob.glob(os.path.join(output_dir, "batch_*/comparison_summary.csv"))
    
    if not summary_files:
        print("No summary files found to aggregate")
        return
    
    # Concatenate all summaries
    all_summaries = []
    for file in summary_files:
        try:
            df = pd.read_csv(file)
            all_summaries.append(df)
        except Exception as e:
            print(f"Error reading {file}: {str(e)}")
    
    if not all_summaries:
        print("Failed to read any summary files")
        return
    
    # Combine all summaries
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    
    # Save combined summary
    combined_summary_path = os.path.join(output_dir, "combined_comparison_summary.csv")
    combined_summary.to_csv(combined_summary_path, index=False)
    
    # Generate overall statistics
    print("\nOverall Summary:")
    print(f"Total barcodes analyzed: {len(combined_summary)}")
    print(f"Average Mpileup variants per barcode: {combined_summary['Mpileup_Variants'].mean():.1f}")
    print(f"Average Strelka variants per barcode: {combined_summary['Strelka_Variants'].mean():.1f}")
    print(f"Average common variants per barcode: {combined_summary['Common_Variants'].mean():.1f}")
    
    # Calculate overall concordance percentage
    if combined_summary['Common_Variants'].sum() > 0:
        concordance_pct = (combined_summary['Concordant_Genotypes'].sum() / combined_summary['Common_Variants'].sum()) * 100
        print(f"Overall genotype concordance: {concordance_pct:.1f}%")
    
    print(f"Combined summary saved to: {combined_summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare mpileup single BAM and Strelka variant calls')
    parser.add_argument('--barcodes', nargs='+', 
                        help='List of cell barcodes to analyze (if not using --all)')
    parser.add_argument('--min-qual', type=float, default=0, 
                        help='Minimum QUAL value to include variant')
    parser.add_argument('--output-dir', default='comparison_results', 
                        help='Directory to save output files')
    parser.add_argument('--dataset', default='dlpfc/151507', 
                        help='Dataset name (dlpfc/151507, P4, or P6)')
    parser.add_argument('--all', action='store_true',
                        help='Process all barcodes found in the dataset')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of parallel workers (default: 1)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of barcodes to process')
    
    args = parser.parse_args()
    
    if args.all:
        print(f"Comparing mpileup and Strelka variant calls for all barcodes in {args.dataset}")
        process_all_barcodes(args.dataset, args.output_dir, args.min_qual, args.workers, args.limit)
    else:
        if not args.barcodes:
            # Default barcodes if none provided and not using --all
            args.barcodes = [
                "AAACAACGAATAGTTC-1",
                "AAACAAGTATCTCCCA-1",
                "AAACAATCTACTAGCA-1", 
                "AAACACCAATAACTGC-1",
                "AAACAGAGCGACTCCT-1"
            ]
        print(f"Comparing mpileup and Strelka variant calls for {len(args.barcodes)} barcodes")
        create_pipeline_comparison(args.barcodes, args.min_qual, args.output_dir, args.dataset)


if __name__ == "__main__":
    main()

# Run all bams
# python scripts/tools/compare_mpileup_single_n_strelka.py --dataset dlpfc/151507 --output-dir comparison_results