#!/usr/bin/env python3
"""
Generate SNV matrix from filtered SNV positions for spatial analysis.

This script processes the output of spatial_snv_filter.py and creates a matrix
where rows are spots/barcodes and columns are SNVs, with binary values indicating
the presence (1) or absence (0) of each SNV in each spot.

Usage:
    python generate_snv_matrix.py --dataset DLPFC --section_id 151507 --quality-filter baseQ0mapQ0 --filter-name filtered_by_neighbor_2
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pickle

# Base directories
PROJECT_BASE_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
OUTPUT_BASE_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"

# Dataset-specific configurations (simplified version from spatial_snv_filter.py)
DATASET_CONFIG = {
    "DLPFC": {
        "output_subdir": "data/dlpfc"
    },
    "P4_TUMOR": {
        "output_subdir": "data/P4_tumor"
    },
    "P6_TUMOR": {
        "output_subdir": "data/P6_tumor"
    },
    "DCIS": {
        "output_subdir": "data/DCIS"
    },
    "OVAR_P5": {
        "output_subdir": "data/ovar_p5"
    }
}

def setup_paths(dataset, section_id, quality_filter, filter_subdir="filtered_snvs"):
    """Set up input and output paths."""
    # Get dataset config
    if dataset.upper() not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset: {dataset}")
        
    config = DATASET_CONFIG[dataset.upper()]
    
    # Data directory
    data_dir = os.path.join(
        PROJECT_BASE_DIR, 
        config["output_subdir"], 
        section_id
    )
    
    # Input directory with filtered SNVs
    input_dir = os.path.join(
        data_dir, 
        "spatial_analysis", 
        quality_filter,
        filter_subdir
    )
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    return {
        "input_dir": input_dir,
        "data_dir": data_dir
    }

def load_filtered_snvs(input_dir):
    """
    Load filtered SNV data from text files in the input directory.
    
    Args:
        input_dir: Directory containing barcode.txt files with filtered SNVs
        
    Returns:
        Dictionary mapping barcodes to sets of SNV keys
    """
    print(f"Loading filtered SNVs from {input_dir}")
    
    # Find all barcode files
    barcode_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not barcode_files:
        raise FileNotFoundError(f"No SNV files found in {input_dir}")
    
    print(f"Found {len(barcode_files)} barcode files")
    
    # Load SNVs for each barcode
    spot_snvs = {}
    for txt_file in tqdm(barcode_files, desc="Loading barcode files"):
        barcode = os.path.basename(txt_file).replace('.txt', '')
        
        # Initialize empty set for this barcode
        spot_snvs[barcode] = set()
        
        # Read SNVs from file
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    chrom, pos = parts[0], parts[1]
                    if not pos.isdigit():  # skip header lines
                        continue
                    snv_key = f"{chrom}_{pos}"
                    spot_snvs[barcode].add(snv_key)
    
    # Count SNVs and barcodes
    total_snvs = sum(len(snvs) for snvs in spot_snvs.values())
    print(f"Loaded {total_snvs} SNVs across {len(spot_snvs)} barcodes")
    
    return spot_snvs

def create_snv_matrix(spot_snvs):
    """
    Create a binary matrix of spots (rows) vs SNVs (columns).
    
    Args:
        spot_snvs: Dictionary mapping barcodes to sets of SNV keys
        
    Returns:
        pandas DataFrame with spots as rows and SNVs as columns
    """
    print("Creating SNV matrix...")
    
    # Collect all unique SNVs
    all_snvs = set()
    for snvs in spot_snvs.values():
        all_snvs.update(snvs)
    
    print(f"Found {len(all_snvs)} unique SNVs")
    
    # Sort SNVs for consistent column order
    snv_list = sorted(all_snvs)
    # O(1) column lookup; snv_list.index() is O(N) and becomes a bottleneck for
    # large column counts (e.g. unfiltered GATK ~560k SNVs).
    snv_index = {snv: j for j, snv in enumerate(snv_list)}

    # Create an empty matrix
    matrix = np.zeros((len(spot_snvs), len(snv_list)), dtype=np.int8)

    # Fill the matrix
    for i, (barcode, snvs) in enumerate(tqdm(spot_snvs.items(), desc="Building matrix")):
        for snv in snvs:
            j = snv_index.get(snv)
            if j is not None:  # This check should always pass
                matrix[i, j] = 1
    
    # Create DataFrame
    df = pd.DataFrame(
        matrix,
        index=list(spot_snvs.keys()),
        columns=snv_list
    )
    
    print(f"Created matrix with shape {df.shape}")
    return df

def save_snv_matrix(df, dataset, section_id, caller="self", output_name="filtered", grouping="6"):
    """
    Save the SNV matrix to a pickle file.
    
    Args:
        df: pandas DataFrame with the SNV matrix
        dataset: Dataset name (e.g., 'DLPFC')
        section_id: Section ID
        caller: Caller used for SNV data
        filter_name: Filter method used for SNV data
        grouping: Grouping strategy used for SNV data
    """
    # Create output file name following the expected format
    output_file = f"{dataset.upper()}_{section_id}_{caller}_{output_name}_{grouping}_matrix.pkl"
    # Use the dataset's real output_subdir (case-sensitive FS: data/P6_tumor != data/p6_tumor);
    # falling back to the raw dataset arg only for unknown datasets.
    output_subdir = DATASET_CONFIG.get(dataset.upper(), {}).get("output_subdir",
                                                                os.path.join("data", dataset))
    output_path = os.path.join(OUTPUT_BASE_DIR, output_subdir, section_id, "matrix", output_file)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the DataFrame
    print(f"Saving SNV matrix to {output_path}")
    df.to_pickle(output_path)
    print("Done!")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Generate SNV matrix from filtered SNVs")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, 
                      choices=[k.lower() for k in DATASET_CONFIG.keys()],
                      help="Dataset name (e.g., 'dlpfc', 'p4_tumor', 'p6_tumor')")
    parser.add_argument("--section_id", required=True,
                      help="Section ID (e.g., '151507' for DLPFC)")
    
    # Optional arguments
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter used for SNV data (default: baseQ0mapQ0)")
    parser.add_argument("--filter-subdir", default="filtered_snvs",
                      help="Subdirectory containing filtered SNV data (default: filtered_snvs)")
    parser.add_argument("--output-name", default="filtered",
                      help="Name to use for the filter in the output file (default: filtered)")
    parser.add_argument("--caller", default="bcftools",
                      help="Name of the variant caller (default: self)")
    parser.add_argument("--grouping", default="6",
                      help="Grouping strategy (default: 6)")
    parser.add_argument("--output-dir",
                      help="Custom output directory (overrides default)")
    parser.add_argument("--input-dir",
                      help="Direct path to directory containing per-barcode .txt files "
                           "(overrides dataset/section/quality-filter path construction)")

    args = parser.parse_args()

    # Override output directory if specified
    if args.output_dir:
        global OUTPUT_BASE_DIR
        OUTPUT_BASE_DIR = args.output_dir
        print(f"Using custom output directory: {OUTPUT_BASE_DIR}")

    # Resolve input directory
    if args.input_dir:
        input_dir = args.input_dir
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
    else:
        paths = setup_paths(
            args.dataset,
            args.section_id,
            args.quality_filter,
            args.filter_subdir
        )
        input_dir = paths["input_dir"]

    # Load filtered SNVs
    spot_snvs = load_filtered_snvs(input_dir)
    
    # Create SNV matrix
    snv_matrix = create_snv_matrix(spot_snvs)
    
    # Save SNV matrix
    output_path = save_snv_matrix(
        snv_matrix,
        args.dataset,
        args.section_id,
        args.caller,
        args.output_name,
        args.grouping
    )
    
    print(f"SNV matrix saved to: {output_path}")
    print(f"Matrix shape: {snv_matrix.shape[0]} spots x {snv_matrix.shape[1]} SNVs")

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC section 151507 with neighbor filtering:
# python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151508 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs --output-name normal
# python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151507 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs_no_all_filtered_out --output-name beagle_only 
# python scripts/6_spatial_filter/run_generate_matrix.py --dataset dlpfc --section_id 151507 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs_no_all_filtered_in --output-name denovo_only 


# For P4_TUMOR section 1 with filter: filtered_snvs_no_all_filtered_out_only_P4_somatic_snp_chr1_22
# python scripts/6_spatial_filter/run_generate_matrix.py --dataset p4_tumor --section_id 1 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs_no_all_filtered_out_only_P4_somatic_snp_chr1_22 --output-name filtered_by_somatic_Beagle

# python generate_snv_matrix.py --dataset p4_tumor --section_id 1 --quality-filter baseQ0mapQ0 --filter-subdir filtered_snvs_only_P4_Normal_WES_gatk_snp_chr1_22 --filter-name filtered_by_Normal_WES