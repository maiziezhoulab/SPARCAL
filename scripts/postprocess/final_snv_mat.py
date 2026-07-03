#!/usr/bin/env python3
"""
final_snv_mat.py

Generates two final outputs merging all 3 spatial-filter categories
(germline_defined, germline_denovo, somatic_denovo):

  a) merged_snv_spot_matrix.vcf.gz
       Giant SNV × Spot VCF. One row per unique SNV, one sample column per
       barcode. INFO carries all pipeline features + CATEGORY tag (which of
       the 3 categories the SNV belongs to). FORMAT = GT only (1/1 present,
       0/0 absent).

  b) per_barcode/{barcode}.vcf.gz
       One VCF per spot listing every SNV found in that barcode (union of all
       3 categories). Same INFO fields as (a).

Output root: {data_root}/{section_id}/final_matrices/{quality_filter}/

Usage:
  python final_snv_mat.py --dataset P4_TUMOR --section_id 1
  python final_snv_mat.py --dataset DCIS     --section_id dcis1
"""

import argparse
import gzip
import os
import sys

import pandas as pd

DATASET_CONFIGS = {
    "P4_TUMOR": {"base_path": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor"},
    "P6_TUMOR": {"base_path": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor"},
    "DCIS":     {"base_path": "/data/maiziezhou_lab/leiy4/snv_calling/data"},
}

# (variant_class, subtype, category_name) — priority order for deduplication
CATEGORIES = [
    ("germline", "defined", "germline_defined"),
    ("germline", "denovo",  "germline_denovo"),
    ("somatic",  "denovo",  "somatic_denovo"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def strip_chr(chrom: str) -> str:
    return chrom[3:] if chrom.startswith("chr") else chrom


def snv_key(chrom, pos, ref, alt):
    return (strip_chr(str(chrom)), int(pos), str(ref), str(alt))


def load_passing_snvs(txt_path):
    """Return set of snv_key tuples from a category txt file."""
    if not os.path.exists(txt_path):
        return set()
    df = pd.read_csv(txt_path, sep="\t", dtype={"chrom": str})
    if df.empty:
        return set()
    return {snv_key(r["chrom"], r["pos"], r["ref"], r["alt"]) for _, r in df.iterrows()}


def load_vcf_info(vcf_path):
    """
    Read a VCF and return:
      info_map:   {snv_key: info_str}
      meta_lines: [##INFO header strings]
    """
    info_map, meta_lines = {}, []
    if not os.path.exists(vcf_path):
        return info_map, meta_lines
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("##INFO"):
                meta_lines.append(line.rstrip())
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 8:
                continue
            chrom, pos, _, ref, alt, _, _, info = parts[:8]
            info_map[snv_key(chrom, pos, ref, alt)] = info
    return info_map, meta_lines


def load_barcode_snvs(barcode_dir, race_filter, passing_set):
    """
    Read per-barcode txt files in barcode_dir.
    Returns {barcode: set of snv_key} filtered to race_filter ∩ passing_set.
    """
    spot_snvs = {}
    for fname in sorted(os.listdir(barcode_dir)):
        if not fname.endswith(".txt") or fname.startswith("germline") or fname.startswith("somatic"):
            continue
        barcode = fname[:-4]
        fpath = os.path.join(barcode_dir, fname)
        try:
            df = pd.read_csv(fpath, sep="\t", dtype={"chrom": str})
        except Exception as e:
            print(f"  WARNING: could not read {fpath}: {e}", file=sys.stderr)
            spot_snvs[barcode] = set()
            continue
        if race_filter is not None:
            df = df[df["race"] == race_filter]
        spot_snvs[barcode] = {
            snv_key(r["chrom"], r["pos"], r["ref"], r["alt"])
            for _, r in df.iterrows()
            if snv_key(r["chrom"], r["pos"], r["ref"], r["alt"]) in passing_set
        }
    return spot_snvs


def chrom_sort_key(k):
    c = k[0]
    try:
        return (int(c), k[1])
    except ValueError:
        return (999, k[1])


def build_merged_meta(category_vcf_metas, summary_meta):
    """Deduplicate and merge ##INFO lines from all sources."""
    seen, merged = set(), []
    for meta_lines in category_vcf_metas + [summary_meta]:
        for line in meta_lines:
            tag = line.split("ID=")[1].split(",")[0] if "ID=" in line else line
            if tag not in seen:
                seen.add(tag)
                merged.append(line)
    return merged


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_merged_snv_spot_vcf(out_path, all_barcodes, snv_list, snv_category,
                               spot_snvs_by_cat, info_map, meta_lines):
    """
    Write one giant VCF: SNVs × all barcodes.
    snv_category: {snv_key: category_name}
    spot_snvs_by_cat: {category_name: {barcode: set of snv_key}}
    info_map: {snv_key: info_str}  (merged from spatial + summary VCFs)
    """
    # Build a fast barcode-index lookup
    bc_idx = {bc: i for i, bc in enumerate(all_barcodes)}

    # Per-SNV presence vector
    def presence_vec(k):
        cat = snv_category[k]
        cat_spot_snvs = spot_snvs_by_cat[cat]
        return ["1/1" if k in cat_spot_snvs.get(bc, set()) else "0/0"
                for bc in all_barcodes]

    with gzip.open(out_path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("##source=final_snv_mat.py\n")
        for line in meta_lines:
            fh.write(line + "\n")
        fh.write('##INFO=<ID=CATEGORY,Number=1,Type=String,Description='
                 '"SNV category: germline_defined, germline_denovo, or somatic_denovo">\n')
        fh.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')

        header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
        fh.write("\t".join(header + list(all_barcodes)) + "\n")

        for k in snv_list:
            chrom, pos, ref, alt = k
            base_info = info_map.get(k, ".")
            cat = snv_category[k]
            info = f"{base_info};CATEGORY={cat}" if base_info != "." else f"CATEGORY={cat}"
            row = [f"chr{chrom}", str(pos), ".", ref, alt, ".", "PASS", info, "GT"]
            row += presence_vec(k)
            fh.write("\t".join(row) + "\n")


def write_per_barcode_vcfs(out_dir, all_barcodes, snv_list, snv_category,
                            spot_snvs_by_cat, info_map, meta_lines):
    """
    Write one VCF per barcode listing every SNV present in that spot.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build per-barcode SNV presence: {barcode: list of snv_keys present}
    bc_snvs = {bc: [] for bc in all_barcodes}
    for k in snv_list:
        cat = snv_category[k]
        for bc, snv_set in spot_snvs_by_cat[cat].items():
            if k in snv_set:
                bc_snvs[bc].append(k)

    vcf_header_lines = (
        ["##fileformat=VCFv4.2", "##source=final_snv_mat.py"]
        + meta_lines
        + ['##INFO=<ID=CATEGORY,Number=1,Type=String,Description='
           '"SNV category: germline_defined, germline_denovo, or somatic_denovo">',
           '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">']
    )

    n_written = 0
    for bc in all_barcodes:
        snvs = bc_snvs[bc]
        if not snvs:
            continue  # skip barcodes with no SNVs in any category
        out_path = os.path.join(out_dir, f"{bc}.vcf.gz")
        with gzip.open(out_path, "wt") as fh:
            for h in vcf_header_lines:
                fh.write(h + "\n")
            fh.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{bc}\n")
            for k in snvs:
                chrom, pos, ref, alt = k
                base_info = info_map.get(k, ".")
                cat = snv_category[k]
                info = f"{base_info};CATEGORY={cat}" if base_info != "." else f"CATEGORY={cat}"
                fh.write(f"chr{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\tGT\t1/1\n")
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",        required=True,
                        help="P4_TUMOR, P6_TUMOR, or DCIS")
    parser.add_argument("--section_id",     required=True,
                        help="e.g. 1, dcis1, dcis2")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0")
    args = parser.parse_args()

    dataset = args.dataset.upper()
    if dataset not in DATASET_CONFIGS:
        print(f"ERROR: unknown dataset '{args.dataset}'", file=sys.stderr)
        sys.exit(1)

    base_path    = DATASET_CONFIGS[dataset]["base_path"]
    section_root = os.path.join(base_path, str(args.section_id))
    qf           = args.quality_filter
    sf_root      = os.path.join(section_root, "spatial_filter_purity", qf)
    spotprof_dir = os.path.join(section_root, "output_VCFs", "spotprofiles", qf)
    out_dir      = os.path.join(section_root, "final_matrices", qf)
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Load summary VCF (rich INFO fallback)
    # ------------------------------------------------------------------
    summary_vcf = os.path.join(spotprof_dir, "all_detected_variants_summary.vcf.gz")
    print(f"Loading summary VCF: {summary_vcf}")
    summary_info, summary_meta = load_vcf_info(summary_vcf)
    print(f"  {len(summary_info):,} variants")

    # ------------------------------------------------------------------
    # Load each category
    # ------------------------------------------------------------------
    snv_category     = {}   # snv_key -> category_name  (first category wins on overlap)
    spot_snvs_by_cat = {}   # category_name -> {barcode: set of snv_key}
    all_barcodes_set = set()
    category_metas   = []

    for variant_class, subtype, cat_name in CATEGORIES:
        subdir   = os.path.join(sf_root, variant_class, subtype)
        txt_path = os.path.join(subdir, f"{cat_name}.txt")
        vcf_path = os.path.join(subdir, f"{cat_name}.vcf.gz")

        passing = load_passing_snvs(txt_path)
        if not passing:
            print(f"[{cat_name}] No passing SNVs — skipping")
            spot_snvs_by_cat[cat_name] = {}
            continue
        print(f"[{cat_name}] {len(passing):,} passing SNVs")

        # Assign category (first category wins for overlapping positions)
        for k in passing:
            if k not in snv_category:
                snv_category[k] = cat_name

        # Load spatial filter VCF INFO for this category
        sp_info, sp_meta = load_vcf_info(vcf_path)
        category_metas.append(sp_meta)

        # Merge INFO: spatial filter takes priority over summary
        for k, info in sp_info.items():
            if k not in summary_info:
                summary_info[k] = info

        # Load per-barcode SNV presence
        barcode_dir = os.path.join(sf_root, variant_class)
        cat_spot_snvs = load_barcode_snvs(barcode_dir, subtype, passing)
        spot_snvs_by_cat[cat_name] = cat_spot_snvs

        for bc in cat_spot_snvs:
            all_barcodes_set.add(bc)

        print(f"  {len(cat_spot_snvs):,} barcodes loaded")

    if not snv_category:
        print("No SNVs found across all categories. Exiting.")
        sys.exit(0)

    all_barcodes = sorted(all_barcodes_set)
    snv_list     = sorted(snv_category.keys(), key=chrom_sort_key)
    merged_meta  = build_merged_meta(category_metas, summary_meta)

    print(f"\nTotal unique SNVs: {len(snv_list):,}")
    print(f"Total barcodes:    {len(all_barcodes):,}")

    # ------------------------------------------------------------------
    # Output a: merged SNV × Spot VCF
    # ------------------------------------------------------------------
    vcf_out = os.path.join(out_dir, "merged_snv_spot_matrix.vcf.gz")
    print(f"\nWriting merged SNV×spot VCF: {vcf_out}")
    write_merged_snv_spot_vcf(vcf_out, all_barcodes, snv_list, snv_category,
                               spot_snvs_by_cat, summary_info, merged_meta)
    print(f"  Done: {len(snv_list):,} SNVs × {len(all_barcodes):,} spots")

    # ------------------------------------------------------------------
    # Output b: per-barcode VCF profiles
    # ------------------------------------------------------------------
    bc_vcf_dir = os.path.join(out_dir, "per_barcode")
    print(f"\nWriting per-barcode VCFs: {bc_vcf_dir}/")
    n = write_per_barcode_vcfs(bc_vcf_dir, all_barcodes, snv_list, snv_category,
                                spot_snvs_by_cat, summary_info, merged_meta)
    print(f"  Done: {n:,} barcode VCFs written (barcodes with ≥1 SNV)")

    print(f"\nOutputs in: {out_dir}")


if __name__ == "__main__":
    main()
