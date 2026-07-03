#!/usr/bin/env python3
"""
01_summarize_germline_matrix.py

Summarize a germline spot×SNV binary matrix pkl produced by the SPARCAL pipeline
(final_matrices/{quality_filter}/germline_*_spot_snv_matrix.pkl).

Prints a structured report to stdout and saves:
  - {out_dir}/summary.txt         — the same text report
  - {out_dir}/snv_counts_per_spot.csv   — per-barcode SNV counts
  - {out_dir}/spot_counts_per_snv.csv   — per-SNV spot counts
  - {out_dir}/chromosome_summary.csv    — per-chromosome SNV counts

Usage:
    python 01_summarize_germline_matrix.py \\
        --pkl /data/maiziezhou_lab/leiy4/snv_calling/data/dcis1/final_matrices/baseQ0mapQ0/germline_denovo_spot_snv_matrix.pkl \\
        [--out_dir <dir>]   # default: same directory as pkl
"""

import argparse
import os
import sys
import pickle
import io
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_matrix(pkl_path: str) -> pd.DataFrame:
    """Load pkl and normalise to a pandas DataFrame (spots × SNVs)."""
    with open(pkl_path, "rb") as fh:
        obj = pickle.load(fh)

    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        # dict of {barcode: set-of-snv-keys}  →  build DataFrame
        all_snvs = sorted({s for snvs in obj.values() for s in snvs})
        barcodes = sorted(obj.keys())
        mat = pd.DataFrame(0, index=barcodes, columns=all_snvs, dtype=np.int8)
        for bc, snvs in obj.items():
            for s in snvs:
                mat.at[bc, s] = 1
        return mat
    raise TypeError(f"Unrecognised pkl content type: {type(obj)}")


def parse_chrom(col) -> str:
    """Extract chromosome string from a column label (tuple or str)."""
    if isinstance(col, tuple):
        chrom = str(col[0])
    else:
        chrom = str(col).split("_")[0]
    # Normalise: strip 'chr' for sorting, then re-add for display
    return "chr" + chrom.lstrip("chr")


def chrom_sort_key(chrom: str) -> tuple:
    c = chrom.lstrip("chr")
    try:
        return (0, int(c))
    except ValueError:
        order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
        return (1, order.get(c, 99))


def percentile_str(arr: np.ndarray) -> str:
    p = np.percentile(arr, [0, 25, 50, 75, 100])
    return (f"min={p[0]:.0f}  Q1={p[1]:.1f}  median={p[2]:.1f}"
            f"  Q3={p[3]:.1f}  max={p[4]:.0f}")


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(df: pd.DataFrame) -> dict:
    n_spots, n_snvs = df.shape
    mat = df.values.astype(np.int8)

    total_ones      = int(mat.sum())
    sparsity        = 1.0 - total_ones / (n_spots * n_snvs)

    snv_per_spot    = mat.sum(axis=1)   # per-row
    spot_per_snv    = mat.sum(axis=0)   # per-column

    # Chromosome distribution
    chroms          = [parse_chrom(c) for c in df.columns]
    chrom_counts    = Counter(chroms)
    chrom_order     = sorted(chrom_counts.keys(), key=chrom_sort_key)

    # Spots with zero SNVs
    zero_snv_spots  = int((snv_per_spot == 0).sum())

    # SNVs present in only 1 spot
    singleton_snvs  = int((spot_per_snv == 1).sum())
    # SNVs present in >50 % of spots
    ubiquitous_snvs = int((spot_per_snv > 0.5 * n_spots).sum())

    return {
        "n_spots": n_spots,
        "n_snvs": n_snvs,
        "total_ones": total_ones,
        "sparsity": sparsity,
        "snv_per_spot": snv_per_spot,
        "spot_per_snv": spot_per_snv,
        "zero_snv_spots": zero_snv_spots,
        "singleton_snvs": singleton_snvs,
        "ubiquitous_snvs": ubiquitous_snvs,
        "chrom_counts": chrom_counts,
        "chrom_order": chrom_order,
    }


def format_report(pkl_path: str, df: pd.DataFrame, res: dict) -> str:
    buf = io.StringIO()

    def p(line=""):
        buf.write(line + "\n")

    p("=" * 70)
    p("  GERMLINE SPOT×SNV MATRIX — SUMMARY REPORT")
    p("=" * 70)
    p(f"  File   : {pkl_path}")
    p(f"  Shape  : {res['n_spots']:,} spots  ×  {res['n_snvs']:,} SNVs")
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    p(f"  Memory : {mem_mb:.1f} MB")
    p()

    p("── MATRIX DENSITY ─────────────────────────────────────────────────────")
    p(f"  Present (1) entries : {res['total_ones']:,}")
    p(f"  Absent  (0) entries : {res['n_spots']*res['n_snvs'] - res['total_ones']:,}")
    p(f"  Density (fill rate) : {1 - res['sparsity']:.4%}")
    p(f"  Sparsity            : {res['sparsity']:.4%}")
    p()

    p("── SNVs PER SPOT ───────────────────────────────────────────────────────")
    sps = res["snv_per_spot"]
    p(f"  {percentile_str(sps)}")
    p(f"  Mean ± SD           : {sps.mean():.1f} ± {sps.std():.1f}")
    p(f"  Spots with 0 SNVs   : {res['zero_snv_spots']:,}"
      f"  ({res['zero_snv_spots']/res['n_spots']:.1%})")
    p()

    p("── SPOTS PER SNV ───────────────────────────────────────────────────────")
    sps2 = res["spot_per_snv"]
    p(f"  {percentile_str(sps2)}")
    p(f"  Mean ± SD           : {sps2.mean():.1f} ± {sps2.std():.1f}")
    p(f"  Singleton SNVs (1 spot)  : {res['singleton_snvs']:,}"
      f"  ({res['singleton_snvs']/res['n_snvs']:.1%})")
    p(f"  Ubiquitous SNVs (>50%)   : {res['ubiquitous_snvs']:,}"
      f"  ({res['ubiquitous_snvs']/res['n_snvs']:.1%})")
    p()

    p("── CHROMOSOMAL DISTRIBUTION OF SNVs ────────────────────────────────────")
    p(f"  {'Chromosome':<12}  {'SNV count':>10}  {'% of total':>10}")
    p(f"  {'-'*12}  {'-'*10}  {'-'*10}")
    for chrom in res["chrom_order"]:
        cnt = res["chrom_counts"][chrom]
        p(f"  {chrom:<12}  {cnt:>10,}  {cnt/res['n_snvs']:>9.1%}")
    p()

    p("── BARCODE SAMPLE ──────────────────────────────────────────────────────")
    n_show = min(5, res["n_spots"])
    p(f"  First {n_show} barcodes (spot index, barcode, SNV count):")
    for i in range(n_show):
        bc  = df.index[i]
        cnt = int(res["snv_per_spot"][i])
        p(f"    [{i:>4}] {bc}  →  {cnt:,} SNVs")
    p()

    p("=" * 70)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pkl", required=True,
                    help="Path to germline_*_spot_snv_matrix.pkl")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: same as pkl)")
    args = ap.parse_args()

    if not os.path.exists(args.pkl):
        print(f"ERROR: file not found: {args.pkl}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.pkl))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {args.pkl} …")
    df = load_matrix(args.pkl)

    print(f"Matrix shape: {df.shape[0]:,} spots × {df.shape[1]:,} SNVs")
    res = analyse(df)

    report = format_report(args.pkl, df, res)
    print(report)

    # ── Save text report ──────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "germline_denovo_summary.txt")
    with open(report_path, "w") as fh:
        fh.write(report)
    print(f"Report saved → {report_path}")

    # ── Save per-spot SNV counts ──────────────────────────────────────────
    snv_per_spot_df = pd.DataFrame({
        "barcode":   df.index,
        "snv_count": res["snv_per_spot"],
    })
    spot_csv = os.path.join(out_dir, "snv_counts_per_spot.csv")
    snv_per_spot_df.to_csv(spot_csv, index=False)
    print(f"Per-spot counts → {spot_csv}")

    # ── Save per-SNV spot counts ──────────────────────────────────────────
    spot_per_snv_df = pd.DataFrame({
        "snv":         [str(c) for c in df.columns],
        "spot_count":  res["spot_per_snv"],
    })
    snv_csv = os.path.join(out_dir, "spot_counts_per_snv.csv")
    spot_per_snv_df.to_csv(snv_csv, index=False)
    print(f"Per-SNV counts  → {snv_csv}")

    # ── Save chromosome summary ───────────────────────────────────────────
    chrom_rows = [
        {"chromosome": ch, "snv_count": res["chrom_counts"][ch],
         "pct_of_total": res["chrom_counts"][ch] / res["n_snvs"]}
        for ch in res["chrom_order"]
    ]
    chrom_df = pd.DataFrame(chrom_rows)
    chrom_csv = os.path.join(out_dir, "chromosome_summary.csv")
    chrom_df.to_csv(chrom_csv, index=False)
    print(f"Chromosome dist → {chrom_csv}")


if __name__ == "__main__":
    main()
