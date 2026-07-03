#!/usr/bin/env python3
"""
generate_sparcal_matrices.py

Canonical SPARCAL spot x SNV matrix builder (replaces the fragile
run_generate_matrix.py / final_snv_mat.py path juggling that failed repeatedly).

It reads the per-barcode SNV lists produced by the spatial filter (step 7)
directly from

    {output_base}/{section_id}/spatial_filter_purity/{quality_filter}/
        germline/{barcode}.txt   (cols: chrom pos ref alt race[defined|denovo])
        somatic/{barcode}.txt     (cols: chrom pos ref alt race[denovo])

and emits FOUR binary (int8) spot x SNV matrices, one per class:

    1000G     germline rows with race == defined   (present in the 1000G panel)
    germline  all germline rows                    (1000G + UPV; == old "normal")
    somatic   all somatic rows
    merged    germline U somatic

All four share the SAME row index (the union of every barcode that has a
germline or somatic file), so they are directly comparable. Columns are the
per-class unique SNVs, keyed `chrom_pos` (same convention as the benchmark
matrices).

Output (matrix dir is created if missing):
    {output_base}/{section_id}/matrix/{STUDY}_{section_id}_SPARCAL_{class}_matrix.pkl

Usage:
    python generate_sparcal_matrices.py --dataset P6_TUMOR --section_id 1
    python generate_sparcal_matrices.py --dataset DCIS     --section_id dcis1
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# Per-dataset pipeline output roots (case-sensitive: data/P6_tumor != data/p6_tumor).
# Kept in sync with run_spatial_snv_filter_enhanced.py "output_base".
PROJECT_BASE_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
DATASET_OUTPUT_BASE = {
    "DLPFC":    "data/dlpfc",
    "P4_TUMOR": "data/P4_tumor",
    "P6_TUMOR": "data/P6_tumor",
    "DCIS":     "data",        # section_id already carries the "dcis1"/"dcis2" prefix
    "OVAR_P5":  "data/ovar_p5",
}

# class token -> (source dir, race filter or None for "all rows in that dir")
# merged is handled specially (union of germline + somatic).
# "normal" is an alias of "germline" (all germline rows) for normal-tissue
# datasets like DLPFC — it is the direct successor of the old bcftools "normal".
CLASS_SPECS = {
    "1000G":    ("germline", "defined"),
    "germline": ("germline", None),
    "normal":   ("germline", None),
    "somatic":  ("somatic",  None),
}


def _barcode_txt_files(cat_dir):
    """Per-barcode .txt files in a category dir, excluding the aggregate
    germline_*/somatic_* category lists that live alongside them."""
    files = []
    for path in glob.glob(os.path.join(cat_dir, "*.txt")):
        name = os.path.basename(path)
        if name.startswith("germline") or name.startswith("somatic"):
            continue
        files.append(path)
    return sorted(files)


def load_category(cat_dir, race_filter):
    """
    Return {barcode: set('chrom_pos')} for every per-barcode file in cat_dir,
    keeping only rows matching race_filter (or all rows if race_filter is None).
    Barcodes with no matching rows still appear (empty set) so the row index is
    complete.
    """
    spot_snvs = {}
    if not os.path.isdir(cat_dir):
        return spot_snvs
    for path in _barcode_txt_files(cat_dir):
        barcode = os.path.basename(path)[:-4]
        try:
            df = pd.read_csv(path, sep="\t", dtype={"chrom": str})
        except pd.errors.EmptyDataError:
            spot_snvs[barcode] = set()
            continue
        except Exception as e:  # pragma: no cover - defensive
            print(f"  WARNING: could not read {path}: {e}", file=sys.stderr)
            spot_snvs[barcode] = set()
            continue
        if df.empty or "pos" not in df.columns:
            spot_snvs[barcode] = set()
            continue
        if race_filter is not None and "race" in df.columns:
            df = df[df["race"] == race_filter]
        spot_snvs[barcode] = {f"{c}_{p}" for c, p in zip(df["chrom"], df["pos"])}
    return spot_snvs


def build_matrix(spot_snvs, all_barcodes):
    """Binary int8 DataFrame: rows=all_barcodes, cols=sorted unique SNVs."""
    all_snvs = set()
    for s in spot_snvs.values():
        all_snvs.update(s)
    snv_list = sorted(all_snvs)
    col_idx = {snv: j for j, snv in enumerate(snv_list)}

    mat = np.zeros((len(all_barcodes), len(snv_list)), dtype=np.int8)
    for i, bc in enumerate(all_barcodes):
        for snv in spot_snvs.get(bc, ()):  # bc may only exist in the other class
            j = col_idx.get(snv)
            if j is not None:
                mat[i, j] = 1
    return pd.DataFrame(mat, index=all_barcodes, columns=snv_list)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="P4_TUMOR, P6_TUMOR, DCIS, DLPFC, OVAR_P5")
    ap.add_argument("--section_id", required=True, help="e.g. 1, dcis1, 151507")
    ap.add_argument("--quality_filter", default="baseQ0mapQ0")
    ap.add_argument("--model", default="SPARCAL",
                    help="model token in the output name (default: SPARCAL)")
    ap.add_argument("--classes", nargs="+",
                    default=["1000G", "germline", "somatic", "merged"],
                    help="subset of classes to build (default: all four)")
    args = ap.parse_args()

    dataset = args.dataset.upper()
    if dataset not in DATASET_OUTPUT_BASE:
        print(f"ERROR: unknown dataset '{args.dataset}'", file=sys.stderr)
        sys.exit(1)

    section_root = os.path.join(PROJECT_BASE_DIR, DATASET_OUTPUT_BASE[dataset],
                                str(args.section_id))
    sf_root = os.path.join(section_root, "spatial_filter_purity", args.quality_filter)
    if not os.path.isdir(sf_root):
        print(f"ERROR: spatial filter output not found: {sf_root}\n"
              f"       run step 7 (spatial filter) first.", file=sys.stderr)
        sys.exit(1)

    matrix_dir = os.path.join(section_root, "matrix")
    os.makedirs(matrix_dir, exist_ok=True)

    # Load the two source dirs once (germline carries the race split; somatic is all denovo).
    print(f"Reading per-barcode SNVs from {sf_root}")
    germline_defined = load_category(os.path.join(sf_root, "germline"), "defined")
    germline_all     = load_category(os.path.join(sf_root, "germline"), None)
    somatic_all      = load_category(os.path.join(sf_root, "somatic"),  None)

    # Shared row index = every barcode seen in either category.
    all_barcodes = sorted(set(germline_all) | set(somatic_all))
    if not all_barcodes:
        print("ERROR: no per-barcode SNV files found — nothing to build.", file=sys.stderr)
        sys.exit(1)
    print(f"  germline barcodes: {len(germline_all):,}  "
          f"somatic barcodes: {len(somatic_all):,}  "
          f"union (rows): {len(all_barcodes):,}")

    sources = {
        "1000G":    germline_defined,
        "germline": germline_all,
        "normal":   germline_all,   # alias for normal-tissue datasets (DLPFC)
        "somatic":  somatic_all,
    }

    outputs = []
    for cls in args.classes:
        if cls == "merged":
            merged = {bc: (germline_all.get(bc, set()) | somatic_all.get(bc, set()))
                      for bc in all_barcodes}
            spot_snvs = merged
        elif cls in sources:
            spot_snvs = sources[cls]
        else:
            print(f"  WARNING: unknown class '{cls}' — skipping", file=sys.stderr)
            continue

        df = build_matrix(spot_snvs, all_barcodes)
        out_name = f"{dataset}_{args.section_id}_{args.model}_{cls}_matrix.pkl"
        out_path = os.path.join(matrix_dir, out_name)
        df.to_pickle(out_path)
        outputs.append((cls, df.shape, out_path))
        print(f"  [{cls:8s}] {df.shape[0]:>6,} spots x {df.shape[1]:>8,} SNVs  ->  {out_name}")

    print("\nDone. Matrices written to:", matrix_dir)
    for cls, shape, path in outputs:
        print(f"  {cls:8s} {shape}  {path}")


if __name__ == "__main__":
    main()
