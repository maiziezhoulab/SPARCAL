#!/usr/bin/env python3
"""
filter_matrix_exome.py

Post-hoc exome filter for the SPARCAL spot x SNV matrices produced by
generate_sparcal_matrices.py (scripts/6_spatial_filter/). Non-destructive:
reads each existing `{STUDY}_{section}_SPARCAL_{class}_matrix.pkl`, drops
columns (SNVs) whose position does not fall inside the exome-capture BED for
that dataset's genome build, and writes a sibling
`{STUDY}_{section}_SPARCAL_{class}_exome_matrix.pkl` next to it. Originals are
never modified.

Genome build -> BED (hard-coded, matches the datasets' reference genomes):
    hg19 (P4_TUMOR, P6_TUMOR): TruSeq_Exome_TargetedRegions_v1.2_hg19.bed
    hg38 (DCIS, OVAR_P5, DLPFC): Twist_Exome_Core_Covered_Targets_hg38.bed

Matrix columns are keyed "chrom_pos" with bare chromosome names (no "chr"
prefix, e.g. "17_57450461"); BED files use "chr"-prefixed names. BED
coordinates are treated as standard 0-based half-open intervals, so a
1-based matrix position `pos` is inside a BED row (start, end) iff
`start < pos <= end`.

Usage:
    python filter_matrix_exome.py --dataset P4_TUMOR --section_id 1
    python filter_matrix_exome.py --dataset DCIS --section_id dcis1
    python filter_matrix_exome.py --dataset OVAR_P5 --section_id P5_sr13
    python filter_matrix_exome.py --dataset P4_TUMOR --section_id 1 --classes germline somatic
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

PROJECT_BASE_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"

# Kept in sync with scripts/6_spatial_filter/generate_sparcal_matrices.py
DATASET_OUTPUT_BASE = {
    "DLPFC":    "data/dlpfc",
    "P4_TUMOR": "data/P4_tumor",
    "P6_TUMOR": "data/P6_tumor",
    "DCIS":     "data",        # section_id already carries the "dcis1"/"dcis2" prefix
    "OVAR_P5":  "data/ovar_p5",
}

HG19_EXOME_BED = "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed"
HG38_EXOME_BED = "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed"

DATASET_EXOME_BED = {
    "DLPFC":    HG38_EXOME_BED,
    "P4_TUMOR": HG19_EXOME_BED,
    "P6_TUMOR": HG19_EXOME_BED,
    "DCIS":     HG38_EXOME_BED,
    "OVAR_P5":  HG38_EXOME_BED,
}


def load_bed_intervals(bed_path):
    """Return {bare_chrom: (sorted_starts, sorted_ends)}, overlaps merged."""
    df = pd.read_csv(bed_path, sep="\t", header=None, usecols=[0, 1, 2],
                      names=["chrom", "start", "end"], dtype={"chrom": str})
    df["chrom"] = df["chrom"].str.replace("^chr", "", regex=True)

    intervals = {}
    for chrom, grp in df.groupby("chrom"):
        arr = grp[["start", "end"]].sort_values("start").to_numpy()
        merged = [list(arr[0])]
        for s, e in arr[1:]:
            if s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        merged = np.array(merged)
        intervals[chrom] = (merged[:, 0], merged[:, 1])
    return intervals


def positions_in_bed(chrom, positions, intervals):
    """Vectorized containment check for 1-based `positions` on `chrom`."""
    if chrom not in intervals:
        return np.zeros(len(positions), dtype=bool)
    starts, ends = intervals[chrom]
    idx = np.searchsorted(starts, positions, side="left") - 1
    result = np.zeros(len(positions), dtype=bool)
    valid = idx >= 0
    result[valid] = positions[valid] <= ends[idx[valid]]
    return result


def exome_keep_mask(columns, intervals):
    """Boolean keep-mask for matrix columns keyed 'chrom_pos'."""
    chrom_arr = np.empty(len(columns), dtype=object)
    pos_arr = np.empty(len(columns), dtype=np.int64)
    for i, col in enumerate(columns):
        c, p = col.split("_", 1)
        chrom_arr[i] = c
        pos_arr[i] = int(p)

    keep = np.zeros(len(columns), dtype=bool)
    for chrom in np.unique(chrom_arr):
        mask = chrom_arr == chrom
        keep[mask] = positions_in_bed(chrom, pos_arr[mask], intervals)
    return keep


def find_class_matrices(matrix_dir, dataset, section_id):
    """{class_token: path} for every existing (non-exome) SPARCAL matrix."""
    prefix = f"{dataset}_{section_id}_SPARCAL_"
    pattern = os.path.join(matrix_dir, f"{prefix}*_matrix.pkl")
    found = {}
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        cls = name[len(prefix):-len("_matrix.pkl")]
        if cls.endswith("_exome"):
            continue  # already-filtered output from a prior run
        found[cls] = path
    return found


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASET_OUTPUT_BASE))
    ap.add_argument("--section_id", required=True, help="e.g. 1, dcis1, P5_sr13")
    ap.add_argument("--classes", nargs="+", default=None,
                     help="subset of class tokens to filter (default: all found, "
                          "e.g. 1000G germline somatic merged UPV)")
    ap.add_argument("--bed", default=None,
                     help="override the auto-selected exome BED file")
    args = ap.parse_args()

    dataset = args.dataset.upper()
    section_root = os.path.join(PROJECT_BASE_DIR, DATASET_OUTPUT_BASE[dataset],
                                 str(args.section_id))
    matrix_dir = os.path.join(section_root, "matrix")
    if not os.path.isdir(matrix_dir):
        print(f"ERROR: matrix dir not found: {matrix_dir}", file=sys.stderr)
        sys.exit(1)

    bed_path = args.bed or DATASET_EXOME_BED[dataset]
    print(f"Loading exome BED: {bed_path}")
    intervals = load_bed_intervals(bed_path)
    n_regions = sum(len(s) for s, _ in intervals.values())
    print(f"  {n_regions:,} merged regions across {len(intervals)} chromosomes")

    class_paths = find_class_matrices(matrix_dir, dataset, args.section_id)
    if args.classes is not None:
        missing = set(args.classes) - set(class_paths)
        if missing:
            print(f"ERROR: requested classes not found in {matrix_dir}: {sorted(missing)}",
                  file=sys.stderr)
            sys.exit(1)
        class_paths = {c: class_paths[c] for c in args.classes}

    if not class_paths:
        print(f"ERROR: no SPARCAL class matrices found in {matrix_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{dataset} {args.section_id}: filtering {len(class_paths)} matrix file(s)")
    for cls, path in class_paths.items():
        df = pd.read_pickle(path)
        keep = exome_keep_mask(df.columns.to_numpy(), intervals)
        filtered = df.loc[:, keep]

        out_path = path[:-len("_matrix.pkl")] + "_exome_matrix.pkl"
        filtered.to_pickle(out_path)

        n_before, n_after = df.shape[1], filtered.shape[1]
        pct = (n_after / n_before * 100) if n_before else 0.0
        print(f"  [{cls:10s}] {df.shape[0]:>6,} spots x {n_before:>8,} -> {n_after:>8,} SNVs "
              f"({pct:5.1f}% in exome)  ->  {os.path.basename(out_path)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
