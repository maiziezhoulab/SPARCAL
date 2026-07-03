#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple

def get_snv_directory(experiment: str, section_id: str) -> str:
    """Construct the path to the SNV positions directory."""
    base_dir = "/data/maiziezhou_lab/yuqi/snv_calling/data"
    
    if experiment.lower() == "dlpfc":
        return os.path.join(base_dir, f"dlpfc/{section_id}/output_VCFs/BAM_filtered/snv_positions")
    elif experiment.lower() in ["p4", "p4_tumor"]:
        return os.path.join(base_dir, f"P4_tumor/{section_id}/output_VCFs/BAM_filtered/snv_positions")
    elif experiment.lower() in ["p6", "p6_tumor"]:
        return os.path.join(base_dir, f"P6_tumor/{section_id}/output_VCFs/BAM_filtered/snv_positions")
    else:
        raise ValueError(f"Unknown experiment: {experiment}. Supported values: dlpfc, p4, p6")

def count_snvs_in_file(file_path: str) -> int:
    """Count the number of SNVs in a text file."""
    try:
        with open(file_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0

def analyze_snv_files(snv_dir: str) -> Dict:
    """Analyze SNV files and return statistics."""
    if not os.path.exists(snv_dir):
        raise FileNotFoundError(f"SNV directory not found: {snv_dir}")
    
    # Get all txt files
    txt_files = glob.glob(os.path.join(snv_dir, "*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No SNV text files found in {snv_dir}")
    
    print(f"Found {len(txt_files)} SNV files. Analyzing...")
    
    # Count SNVs in each file
    snv_counts = []
    spot_to_count = {}
    for file_path in txt_files:
        barcode = os.path.basename(file_path).replace(".txt", "")
        count = count_snvs_in_file(file_path)
        snv_counts.append(count)
        spot_to_count[barcode] = count
    
    # Calculate statistics
    np_counts = np.array(snv_counts)
    stats = {
        "total_files": len(txt_files),
        "min_snvs": np.min(np_counts),
        "max_snvs": np.max(np_counts),
        "mean_snvs": np.mean(np_counts),
        "median_snvs": np.median(np_counts),
        "std_snvs": np.std(np_counts),
        "total_snvs": np.sum(np_counts),
        "spots_with_0_snvs": sum(1 for count in snv_counts if count == 0),
        "spots_with_1_to_5_snvs": sum(1 for count in snv_counts if 1 <= count <= 5),
        "spots_with_6_to_10_snvs": sum(1 for count in snv_counts if 6 <= count <= 10),
        "spots_with_11_to_20_snvs": sum(1 for count in snv_counts if 11 <= count <= 20),
        "spots_with_21_to_50_snvs": sum(1 for count in snv_counts if 21 <= count <= 50),
        "spots_with_more_than_50_snvs": sum(1 for count in snv_counts if count > 50),
        "raw_counts": snv_counts,
        "spot_to_count": spot_to_count
    }
    
    return stats

def generate_histogram(stats: Dict, output_dir: str, experiment: str, section_id: str):
    """Generate a histogram of SNV counts."""
    plt.figure(figsize=(12, 8))
    
    # Main histogram with log scale
    plt.subplot(2, 1, 1)
    plt.hist(stats["raw_counts"], bins=50, edgecolor='black')
    plt.title(f"_ of SNV Counts - {experiment.upper()} Section {section_id}")
    plt.xlabel("Number of SNVs per spot")
    plt.ylabel("Number of spots")
    plt.grid(True, alpha=0.3)
    
    # Zoom in on the lower counts
    plt.subplot(2, 1, 2)
    plt.hist(stats["raw_counts"], bins=50, range=(0, min(50, stats["max_snvs"])), edgecolor='black')
    plt.title("Distribution of SNV Counts (0-50 range)")
    plt.xlabel("Number of SNVs per spot")
    plt.ylabel("Number of spots")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment}_section{section_id}_snv_histogram.png")
    plt.savefig(output_file)
    print(f"Histogram saved to {output_file}")

def export_data(stats: Dict, output_dir: str, experiment: str, section_id: str):
    """Export SNV count data to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{experiment}_section{section_id}_snv_counts.csv")
    
    # Create DataFrame
    df = pd.DataFrame({
        "spot": list(stats["spot_to_count"].keys()),
        "snv_count": list(stats["spot_to_count"].values())
    })
    
    # Sort by SNV count descending
    df = df.sort_values("snv_count", ascending=False)
    
    # Export to CSV
    df.to_csv(output_file, index=False)
    print(f"SNV counts exported to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze statistics of SNV positions across spots")
    
    # Required arguments
    parser.add_argument("--experiment", required=True, help="Experiment name (dlpfc, p4, or p6)")
    parser.add_argument("--section-id", required=True, help="Section ID")
    
    # Optional arguments
    parser.add_argument("--output-dir", default=None, 
                        help="Directory to save output files (default: experiment's data directory)")
    
    args = parser.parse_args()
    
    # Get SNV directory
    try:
        snv_dir = get_snv_directory(args.experiment, args.section_id)
        print(f"Analyzing SNV files in: {snv_dir}")
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(snv_dir), "analysis")
    
    # Analyze SNV files
    try:
        stats = analyze_snv_files(snv_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Print statistics
    print("\n===== SNV Statistics =====")
    print(f"Total spots analyzed: {stats['total_files']}")
    print(f"Total SNVs detected: {stats['total_snvs']}")
    print(f"SNVs per spot:")
    print(f"  Minimum: {stats['min_snvs']}")
    print(f"  Maximum: {stats['max_snvs']}")
    print(f"  Mean: {stats['mean_snvs']:.2f}")
    print(f"  Median: {stats['median_snvs']}")
    print(f"  Standard deviation: {stats['std_snvs']:.2f}")
    print("\nSpot distribution:")
    print(f"  Spots with 0 SNVs: {stats['spots_with_0_snvs']} ({stats['spots_with_0_snvs']/stats['total_files']*100:.2f}%)")
    print(f"  Spots with 1-5 SNVs: {stats['spots_with_1_to_5_snvs']} ({stats['spots_with_1_to_5_snvs']/stats['total_files']*100:.2f}%)")
    print(f"  Spots with 6-10 SNVs: {stats['spots_with_6_to_10_snvs']} ({stats['spots_with_6_to_10_snvs']/stats['total_files']*100:.2f}%)")
    print(f"  Spots with 11-20 SNVs: {stats['spots_with_11_to_20_snvs']} ({stats['spots_with_11_to_20_snvs']/stats['total_files']*100:.2f}%)")
    print(f"  Spots with 21-50 SNVs: {stats['spots_with_21_to_50_snvs']} ({stats['spots_with_21_to_50_snvs']/stats['total_files']*100:.2f}%)")
    print(f"  Spots with >50 SNVs: {stats['spots_with_more_than_50_snvs']} ({stats['spots_with_more_than_50_snvs']/stats['total_files']*100:.2f}%)")
    
    # Generate histogram
    generate_histogram(stats, output_dir, args.experiment, args.section_id)
    
    # Export data to CSV
    export_data(stats, output_dir, args.experiment, args.section_id)
    
    # Save summary to a text file
    summary_file = os.path.join(output_dir, f"{args.experiment}_section{args.section_id}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"SNV Statistics for {args.experiment} Section {args.section_id}\n")
        f.write("========================================\n\n")
        f.write(f"Data source: {snv_dir}\n\n")
        f.write(f"Total spots analyzed: {stats['total_files']}\n")
        f.write(f"Total SNVs detected: {stats['total_snvs']}\n")
        f.write(f"SNVs per spot:\n")
        f.write(f"  Minimum: {stats['min_snvs']}\n")
        f.write(f"  Maximum: {stats['max_snvs']}\n")
        f.write(f"  Mean: {stats['mean_snvs']:.2f}\n")
        f.write(f"  Median: {stats['median_snvs']}\n")
        f.write(f"  Standard deviation: {stats['std_snvs']:.2f}\n\n")
        f.write("Spot distribution:\n")
        f.write(f"  Spots with 0 SNVs: {stats['spots_with_0_snvs']} ({stats['spots_with_0_snvs']/stats['total_files']*100:.2f}%)\n")
        f.write(f"  Spots with 1-5 SNVs: {stats['spots_with_1_to_5_snvs']} ({stats['spots_with_1_to_5_snvs']/stats['total_files']*100:.2f}%)\n")
        f.write(f"  Spots with 6-10 SNVs: {stats['spots_with_6_to_10_snvs']} ({stats['spots_with_6_to_10_snvs']/stats['total_files']*100:.2f}%)\n")
        f.write(f"  Spots with 11-20 SNVs: {stats['spots_with_11_to_20_snvs']} ({stats['spots_with_11_to_20_snvs']/stats['total_files']*100:.2f}%)\n")
        f.write(f"  Spots with 21-50 SNVs: {stats['spots_with_21_to_50_snvs']} ({stats['spots_with_21_to_50_snvs']/stats['total_files']*100:.2f}%)\n")
        f.write(f"  Spots with >50 SNVs: {stats['spots_with_more_than_50_snvs']} ({stats['spots_with_more_than_50_snvs']/stats['total_files']*100:.2f}%)\n")
    
    print(f"\nSummary saved to {summary_file}")
    print(f"Analysis complete. Results saved to {output_dir}")
    
    return 0

if __name__ == "__main__":
    exit(main())


# # For DLPFC section 151507:
# python /data/maiziezhou_lab/yuqi/snv_calling/scripts/tools/analyze_snv_stats_per_bam.py --experiment dlpfc --section-id 151507

# # For P4_TUMOR section 1:
# python /data/maiziezhou_lab/yuqi/snv_calling/scripts/tools/analyze_snv_stats_per_bam.py --experiment p4 --section-id 1

# # For P6_TUMOR section 2:
# python /data/maiziezhou_lab/yuqi/snv_calling/scripts/tools/analyze_snv_stats_per_bam.py --experiment p6 --section-id 2