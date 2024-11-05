import pandas as pd
import numpy as np
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import argparse
import os
import gzip
from pathlib import Path

def read_vcf_file(file_path: str) -> Set[str]:
    """
    Read a VCF file (gzipped or plain text) and extract SNV IDs (CHROM_POS)
    
    Args:
        file_path: Path to the VCF file (can be .vcf or .vcf.gz)
        
    Returns:
        Set of SNV IDs in format CHROM_POS
    """
    snvs = set()
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return snvs
        
    # Check if file is gzipped
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'  # text mode for gzipped files
    else:
        open_func = open
        mode = 'r'
    
    try:
        with open_func(file_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split('\t')
                chrom, pos = parts[0], parts[1]
                snv_id = f"{chrom}_{pos}"
                snvs.add(snv_id)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return set()
        
    return snvs

def calculate_similarity_metrics(set1: Set[str], set2: Set[str]) -> Dict[str, float]:
    """
    Calculate various similarity metrics between two sets
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    jaccard = intersection / union if union > 0 else 0
    min_size = min(len(set1), len(set2))
    overlap = intersection / min_size if min_size > 0 else 0
    dice = (2 * intersection) / (len(set1) + len(set2)) if (len(set1) + len(set2)) > 0 else 0
    
    return {
        'jaccard': jaccard,
        'overlap': overlap,
        'dice': dice,
        'intersection_size': intersection,
        'union_size': union
    }

def create_venn_diagram(sets: List[Set[str]], labels: List[str], output_path: str, bam_name: str):
    """
    Create and save a Venn diagram showing the intersection of SNV sets
    
    Args:
        sets: List of sets containing SNV IDs
        labels: List of labels for each set
        output_path: Path to save the output image
        bam_name: Name of the BAM file for the title
    """
    plt.figure(figsize=(12, 10))
    venn_plot = venn3(sets, labels)
    
    # Customize the appearance
    for text in venn_plot.set_labels:
        if text is not None:
            text.set_fontsize(12)
    
    for text in venn_plot.subset_labels:
        if text is not None:
            text.set_fontsize(10)
    
    venn3_circles(sets)
    plt.title(f'SNV Caller Comparison - {bam_name}', fontsize=14, pad=20)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def process_single_comparison(vcf_name: str, vcf_folder: str, labels: List[str]) -> Dict:
    """
    Process a single VCF comparison and return the results
    
    Args:
        vcf_name: Base name of the VCF file
        vcf_folder: Root folder containing the caller directories
        labels: Labels for the three callers
    
    Returns:
        Dictionary containing comparison results
    """
    # Construct file paths
    new_caller_path = os.path.join(vcf_folder, 'new_caller', vcf_name)
    old_caller_path = os.path.join(vcf_folder, 'old_caller', vcf_name)
    mpileup_path = os.path.join(vcf_folder, 'mpileup', vcf_name)

    # Read VCF files
    print(f"\nProcessing BAM: {vcf_name}")
    print("Reading VCF files...")
    snvs_new = read_vcf_file(new_caller_path)
    snvs_old = read_vcf_file(old_caller_path)
    snvs_mpileup = read_vcf_file(mpileup_path)

    # Calculate pairwise similarities
    pairs = [
        (snvs_new, snvs_old, f"{labels[0]} vs {labels[1]}"),
        (snvs_old, snvs_mpileup, f"{labels[1]} vs {labels[2]}"),
        (snvs_new, snvs_mpileup, f"{labels[0]} vs {labels[2]}")
    ]

    similarity_results = {}
    for set1, set2, label in pairs:
        metrics = calculate_similarity_metrics(set1, set2)
        similarity_results[label] = metrics

    # Calculate additional statistics
    all_intersection = snvs_new.intersection(snvs_old).intersection(snvs_mpileup)
    unique_new = len(snvs_new - (snvs_old | snvs_mpileup))
    unique_old = len(snvs_old - (snvs_new | snvs_mpileup))
    unique_mpileup = len(snvs_mpileup - (snvs_new | snvs_old))

    return {
        'sets': [snvs_new, snvs_old, snvs_mpileup],
        'set_sizes': [len(snvs_new), len(snvs_old), len(snvs_mpileup)],
        'similarities': similarity_results,
        'all_intersection': len(all_intersection),
        'unique_counts': [unique_new, unique_old, unique_mpileup]
    }

def write_summary_report(results: Dict, bam_name: str, output_dir: str, labels: List[str]):
    """
    Write a detailed summary report for a single BAM comparison
    """
    report_path = os.path.join(output_dir, f'{bam_name}_summary.txt')
    
    with open(report_path, 'w') as f:
        f.write(f"SNV Caller Comparison Summary - {bam_name}\n")
        f.write("="* 50 + "\n\n")
        
        f.write("Set Sizes:\n")
        f.write("---------\n")
        for label, size in zip(labels, results['set_sizes']):
            f.write(f"{label}: {size} SNVs\n")
        f.write("\n")
        
        f.write("Pairwise Similarities:\n")
        f.write("--------------------\n")
        for pair, metrics in results['similarities'].items():
            f.write(f"\n{pair}:\n")
            f.write(f"  Jaccard similarity: {metrics['jaccard']:.3f}\n")
            f.write(f"  Overlap coefficient: {metrics['overlap']:.3f}\n")
            f.write(f"  Dice coefficient: {metrics['dice']:.3f}\n")
            f.write(f"  Intersection size: {metrics['intersection_size']}\n")
            f.write(f"  Union size: {metrics['union_size']}\n")
        
        f.write("\nUnique SNVs per caller:\n")
        f.write("----------------------\n")
        for label, count in zip(labels, results['unique_counts']):
            f.write(f"{label}: {count}\n")
        
        f.write(f"\nSNVs called by all three callers: {results['all_intersection']}\n")

def main():
    parser = argparse.ArgumentParser(description='Compare SNV calls from different callers')
    parser.add_argument('--vcffolder', required=True, 
                       help='Path to folder containing VCF files', 
                       default='/data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/performance_test')
    parser.add_argument('--labels', nargs=3, 
                       default=['New Caller', 'Old Caller', 'MPileup'],
                       help='Labels for the three callers')
    parser.add_argument('--output_dir', 
                       help='Output directory for results',
                       default='./comparison_results')
    args = parser.parse_args()

    # Set up output directory
    # if args.output_dir is None:
    #     args.output_dir = os.path.join(args.vcffolder, 'comparison_results')
    # os.makedirs(args.output_dir, exist_ok=True)

    # Get list of VCF files from new_caller directory (use as reference)
    new_caller_dir = os.path.join(args.vcffolder, 'new_caller')
    vcf_files = [f for f in os.listdir(new_caller_dir) if f.endswith('.vcf')]

    # Process each VCF file
    summary_data = []
    for vcf_name in vcf_files:
        bam_name = vcf_name.replace('.vcf', '')
        
        # Process the comparison
        results = process_single_comparison(vcf_name, args.vcffolder, args.labels)
        
        # Create Venn diagram
        output_path = os.path.join(args.output_dir, f'{bam_name}_venn.png')
        create_venn_diagram(results['sets'], args.labels, output_path, bam_name)
        
        # Write detailed summary report
        write_summary_report(results, bam_name, args.output_dir, args.labels)
        
        # Store summary data for final report
        summary_data.append({
            'bam_name': bam_name,
            'results': results
        })

    # Write overall summary report
    overall_summary_path = os.path.join(args.output_dir, 'overall_summary.txt')
    with open(overall_summary_path, 'w') as f:
        f.write("Overall SNV Caller Comparison Summary\n")
        f.write("===================================\n\n")
        
        for data in summary_data:
            f.write(f"\nBAM: {data['bam_name']}\n")
            f.write("-" * (len(data['bam_name']) + 5) + "\n")
            f.write(f"Total SNVs: {sum(data['results']['set_sizes'])}\n")
            f.write(f"Common SNVs: {data['results']['all_intersection']}\n")
            f.write("Unique SNVs per caller: ")
            f.write(", ".join(f"{label}: {count}" 
                            for label, count in zip(args.labels, data['results']['unique_counts'])))
            f.write("\n")

    print(f"\nProcessing complete. Results saved in: {args.output_dir}")

if __name__ == "__main__":
    main()

# jjjki
# Usage:
# python compare_caller_output.py \
#     --vcffolder /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/performance_test \
#     --labels "New Caller" "Old Caller" "MPileup" \
#     --output_dir ./comparison_results