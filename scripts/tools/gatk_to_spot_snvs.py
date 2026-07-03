#!/usr/bin/env python3
"""
gatk_to_spot_snvs.py
--------------------
Turn per-barcode GATK VCFs into the per-spot .txt format that
`run_generate_matrix.py` consumes, so a GATK spot×SNV matrix can be built and
compared against the strelka2 and in-house pipeline matrices.

Unlike strelka2 (single merged-BAM VCF needing a CB-tag BAM scan), GATK already
produces ONE VCF PER BARCODE, e.g.
    {GATK_BASE}/{section}/gatk/output_VCFs/unfiltered/0/{barcode}.vcf
so no BAM scan is needed — we just parse each VCF for SNV positions.

Selection (matches strelka2 matrix conventions):
  - in-tissue spots only (from DLPFC12 tissue_positions_list.csv)
  - SNVs only: single-base REF and single-base ALT (multiallelic / indels skipped)
  - GT=0/0 and missing genotypes skipped (GATK here emits variant sites only, but guarded)

Output: one `<barcode>.txt` (lines "chrom<TAB>pos") per in-tissue spot under --out-dir.
Spots with no SNV get an empty file so the matrix has a consistent row set.

Then build the matrix with the canonical builder:
    python scripts/6_spatial_filter/run_generate_matrix.py \
        --dataset dlpfc --section_id 151507 --quality-filter baseQ0mapQ0 \
        --input-dir data/dlpfc/151507/gatk/spot_snvs \
        --caller gatk --output-name germline

Usage:
    python scripts/tools/gatk_to_spot_snvs.py --section_id 151507
"""

import os
import glob
import argparse
import logging
from typing import Set

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("gatk_to_spot_snvs")

PROJECT_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
GATK_BASE = "/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC"
DLPFC_SPATIAL_BASE = "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12"  # {section}/spatial/tissue_positions_list.csv

_ACGT = {"A", "C", "G", "T"}


def load_in_tissue_barcodes(positions_file: str) -> Set[str]:
    """DLPFC tissue_positions_list.csv (no header): barcode,in_tissue,arow,acol,prow,pcol."""
    barcodes: Set[str] = set()
    with open(positions_file) as fh:
        for line in fh:
            row = line.rstrip("\n").split(",")
            if len(row) < 2:
                continue
            try:
                if int(row[1]) == 1:
                    barcodes.add(row[0])
            except ValueError:
                continue
    log.info("Loaded %d in-tissue spots from %s", len(barcodes), positions_file)
    return barcodes


def extract_snvs(vcf_path: str):
    """Yield (chrom, pos_str) for SNV records (single-base REF & ALT, GT != 0/0)."""
    with open(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, pos, _id, ref, alt = parts[:5]
            if len(ref) != 1 or ref.upper() not in _ACGT:
                continue
            if len(alt) != 1 or alt.upper() not in _ACGT:   # SNV only; skips indels & multiallelic
                continue
            if not pos.isdigit():
                continue
            # guard: drop ref/missing genotypes if a sample column is present
            if len(parts) >= 10:
                gt = parts[9].split(":", 1)[0].replace("|", "/")
                if gt in ("0/0", "./.", "."):
                    continue
            yield chrom, pos


def main():
    ap = argparse.ArgumentParser(description="Convert per-barcode GATK VCFs to per-spot SNV .txt (DLPFC).")
    ap.add_argument("--section_id", required=True)
    ap.add_argument("--gatk-subdir", default="unfiltered/0",
                    help="Subdir under {section}/gatk/output_VCFs/ holding per-barcode VCFs (default: unfiltered/0).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: data/dlpfc/{section}/gatk/spot_snvs")
    args = ap.parse_args()

    sec = args.section_id
    vcf_dir = os.path.join(GATK_BASE, sec, "gatk", "output_VCFs", args.gatk_subdir)
    pos_file = os.path.join(DLPFC_SPATIAL_BASE, sec, "spatial", "tissue_positions_list.csv")
    out_dir = args.out_dir or os.path.join(PROJECT_DIR, "data/dlpfc", sec, "gatk", "spot_snvs")

    if not os.path.isdir(vcf_dir):
        raise FileNotFoundError(f"GATK per-barcode VCF dir not found: {vcf_dir}")
    if not os.path.exists(pos_file):
        raise FileNotFoundError(f"tissue positions not found: {pos_file}")

    in_tissue = load_in_tissue_barcodes(pos_file)
    os.makedirs(out_dir, exist_ok=True)
    # Clear any stale per-spot files (e.g. from a previous filter-level run) so the
    # matrix reflects only the current --gatk-subdir.
    for old in glob.glob(os.path.join(out_dir, "*.txt")):
        os.remove(old)

    all_vcfs = glob.glob(os.path.join(vcf_dir, "*.vcf"))
    log.info("Section %s | %d per-barcode VCFs in %s", sec, len(all_vcfs), vcf_dir)

    n_written = n_with_snv = total_snv = n_offtissue = 0
    seen_barcodes: Set[str] = set()

    for vcf in all_vcfs:
        barcode = os.path.basename(vcf)[:-4]  # strip ".vcf"
        if barcode not in in_tissue:
            n_offtissue += 1
            continue
        seen_barcodes.add(barcode)
        snvs = sorted({(c, p) for c, p in extract_snvs(vcf)},
                      key=lambda t: (t[0], int(t[1])))
        with open(os.path.join(out_dir, f"{barcode}.txt"), "w") as f:
            for chrom, pos in snvs:
                f.write(f"{chrom}\t{pos}\n")
        n_written += 1
        if snvs:
            n_with_snv += 1
            total_snv += len(snvs)

    # in-tissue spots with no GATK VCF at all -> empty file (consistent matrix rows)
    n_empty_missing = 0
    for barcode in in_tissue - seen_barcodes:
        open(os.path.join(out_dir, f"{barcode}.txt"), "w").close()
        n_written += 1
        n_empty_missing += 1

    log.info("Wrote %d per-spot files to %s", n_written, out_dir)
    log.info("  in-tissue spots: %d | with >=1 SNV: %d | empty: %d (incl. %d with no GATK VCF)",
             len(in_tissue), n_with_snv, n_written - n_with_snv, n_empty_missing)
    log.info("  off-tissue VCFs skipped: %d | total (spot×SNV) presences: %s",
             n_offtissue, f"{total_snv:,}")
    log.info("Next: run_generate_matrix.py --dataset dlpfc --section_id %s "
             "--input-dir %s --caller gatk --output-name germline", sec, out_dir)


if __name__ == "__main__":
    main()
