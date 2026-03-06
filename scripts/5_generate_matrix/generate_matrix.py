#!/usr/bin/env python3
"""
SPARCAL Step 7: Generate SNV Matrix
====================================
Reads per-barcode filtered SNV text files from the spatial analysis step
and produces a binary spot×SNV matrix saved as .pkl and optionally .h5ad.

Usage:
    python run_generate_matrix.py --dataset DLPFC --section_id 151507 --filter-subdir filtered_snvs
    python run_generate_matrix.py --dataset P4_TUMOR --section_id 1 --filter-subdir filtered_snvs
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig


def setup_paths(cfg: PipelineConfig, filter_subdir: str = "filtered_snvs"):
    """Set up input directory from config."""
    input_dir = os.path.join(cfg.output_dir, "spatial_analysis", filter_subdir)
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    return {"input_dir": input_dir, "data_dir": cfg.output_dir}


def load_filtered_snvs(input_dir: str):
    """Load filtered SNV data from barcode.txt files."""
    barcode_files = glob.glob(os.path.join(input_dir, "*.txt"))
    if not barcode_files:
        raise FileNotFoundError(f"No SNV files found in {input_dir}")

    print(f"Found {len(barcode_files)} barcode files")
    spot_snvs = {}
    for txt_file in tqdm(barcode_files, desc="Loading barcode files"):
        barcode = os.path.basename(txt_file).replace('.txt', '')
        spot_snvs[barcode] = set()
        with open(txt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    spot_snvs[barcode].add(f"{parts[0]}_{parts[1]}")

    total = sum(len(s) for s in spot_snvs.values())
    print(f"Loaded {total} SNVs across {len(spot_snvs)} barcodes")
    return spot_snvs


def create_snv_matrix(spot_snvs):
    """Create binary spot×SNV matrix."""
    all_snvs = set()
    for snvs in spot_snvs.values():
        all_snvs.update(snvs)

    snv_list = sorted(all_snvs)
    snv_idx = {s: i for i, s in enumerate(snv_list)}
    matrix = np.zeros((len(spot_snvs), len(snv_list)), dtype=np.int8)

    for i, (barcode, snvs) in enumerate(tqdm(spot_snvs.items(), desc="Building matrix")):
        for snv in snvs:
            matrix[i, snv_idx[snv]] = 1

    df = pd.DataFrame(matrix, index=list(spot_snvs.keys()), columns=snv_list)
    print(f"Matrix shape: {df.shape}")
    return df


def save_snv_matrix(df: pd.DataFrame, cfg: PipelineConfig,
                    caller: str = "bcftools", output_name: str = "filtered",
                    grouping: str = "6"):
    """Save SNV matrix to pickle."""
    fname = f"{cfg.dataset_name}_{cfg.section_id}_{caller}_{output_name}_{grouping}_matrix.pkl"
    out_dir = os.path.join(cfg.output_dir, "matrix")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)

    print(f"Saving to {out_path}")
    df.to_pickle(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 7: Generate SNV Matrix")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match YAML config)")
    parser.add_argument("--section_id", required=True)
    parser.add_argument("--filter-subdir", default="filtered_snvs",
                        help="Subdirectory with filtered SNVs (default: filtered_snvs)")
    parser.add_argument("--output-name", default="filtered")
    parser.add_argument("--caller", default="bcftools")
    parser.add_argument("--grouping", default="6")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.section_id)
    paths = setup_paths(cfg, args.filter_subdir)

    spot_snvs = load_filtered_snvs(paths["input_dir"])
    matrix = create_snv_matrix(spot_snvs)
    out = save_snv_matrix(matrix, cfg, args.caller, args.output_name, args.grouping)

    print(f"Done! {matrix.shape[0]} spots × {matrix.shape[1]} SNVs → {out}")


if __name__ == "__main__":
    main()