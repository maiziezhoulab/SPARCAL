#!/usr/bin/env python3
"""
strelka2_to_spot_snvs.py
------------------------
Project section-level strelka2 SNV calls onto spatial spots, so the result can
be turned into a spot×SNV matrix comparable to the in-house pipeline's matrices.

WHY THIS EXISTS
    The 2026-06 strelka2 runs call variants on the MERGED per-section BAM, so the
    output (`.../strelka2/results/variants/variants.vcf.gz`) is a single-sample
    VCF with NO per-spot resolution. `run_generate_matrix.py` needs one .txt file
    per barcode. This script bridges the gap exactly like
    `generate_original_snp_profile.py`: it scans the merged BAM with pysam, reads
    the CB (cell-barcode) tag at each strelka2 SNV position, and records, per
    in-tissue spot, the set of strelka2 SNVs that have >= --min-alt-reads
    ALT-supporting reads in that spot.

OUTPUT
    One `<barcode>.txt` per in-tissue spot under --out-dir, each line "chrom<TAB>pos"
    (the format run_generate_matrix.py parses). Spots with no supported SNV get an
    empty file so the matrix has a consistent row set.

THEN build the matrix with the canonical builder (same one used for the pipeline):
    python scripts/6_spatial_filter/run_generate_matrix.py \
        --dataset dlpfc --section_id 151507 --quality-filter baseQ0mapQ0 \
        --input-dir data/dlpfc/151507/strelka2/spot_snvs \
        --caller strelka2 --output-name germline

Usage:
    python scripts/tools/strelka2_to_spot_snvs.py --section_id 151507 [--min-alt-reads 1] [--max-workers 22]
"""

import os
import gzip
import argparse
import logging
from collections import defaultdict
from typing import Dict, FrozenSet, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:
    import pysam
except ImportError:
    pysam = None

# ---[ Config ]----------------------------------------------------------------

PROJECT_DIR = "/data/maiziezhou_lab/leiy4/snv_calling"
DLPFC_SPATIAL_BASE = "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12"  # {section}/spatial/tissue_positions_list.csv (no header)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("strelka2_to_spot_snvs")


# ---[ Inputs ]----------------------------------------------------------------

def load_pass_snv_positions(vcf_path: str, pass_only: bool = True
                            ) -> Dict[str, Dict[int, FrozenSet[str]]]:
    """
    Read PASS single-nucleotide variants from a strelka2 VCF.
    Returns {chrom: {pos_1based: frozenset(alt_bases)}}.
    Indels, multi-base and (optionally) non-PASS records are skipped.
    """
    positions: Dict[str, Dict[int, FrozenSet[str]]] = {}
    n_snv = n_skip_indel = n_skip_filter = 0
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 7)              # CHROM POS ID REF ALT QUAL FILTER ...
            if len(parts) < 7:
                continue
            chrom, pos_str, _id, ref, alt_field, _qual, filt = parts[:7]
            if pass_only and filt not in ("PASS", "."):
                n_skip_filter += 1
                continue
            if len(ref) != 1:                        # SNVs only
                n_skip_indel += 1
                continue
            alts = [a for a in alt_field.split(",")
                    if len(a) == 1 and a not in (".", "*", "N")]
            if not alts:
                n_skip_indel += 1
                continue
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            positions.setdefault(chrom, {})[pos] = frozenset(alts)
            n_snv += 1
    log.info("Loaded %s PASS SNV positions across %d chroms "
             "(%s indels/multibase, %s non-PASS skipped)",
             f"{n_snv:,}", len(positions), f"{n_skip_indel:,}", f"{n_skip_filter:,}")
    return positions


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


def build_chrom_map(bam_path: str, vcf_chroms: List[str]) -> Dict[str, str]:
    """Map VCF chrom names to BAM contig names, handling chr-prefix mismatch."""
    bam = pysam.AlignmentFile(bam_path, "rb")
    refs = set(bam.references)
    bam.close()
    mapping: Dict[str, str] = {}
    for c in vcf_chroms:
        if c in refs:
            mapping[c] = c
        elif c.startswith("chr") and c[3:] in refs:
            mapping[c] = c[3:]
        elif not c.startswith("chr") and ("chr" + c) in refs:
            mapping[c] = "chr" + c
    missing = [c for c in vcf_chroms if c not in mapping]
    if missing:
        log.warning("VCF chroms with no BAM contig (skipped): %s", missing)
    return mapping


# ---[ BAM scan ]--------------------------------------------------------------

def scan_one_chrom(bam_path: str, bam_chrom: str,
                   variant_pos: Dict[int, FrozenSet[str]],
                   in_tissue: Set[str], min_alt_reads: int
                   ) -> Dict[str, Set[str]]:
    """
    Pileup one chromosome. For each variant position, count ALT-supporting reads
    per CB barcode; emit snv_key 'chrom_pos' for barcodes meeting min_alt_reads.
    Returns {barcode: set(snv_key)}. Each thread opens its own BAM handle.
    """
    if not variant_pos:
        return {}
    ps = sorted(variant_pos)
    hits: Dict = defaultdict(int)   # (barcode, snv_key) -> alt read count
    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
    except Exception as exc:
        log.warning("open BAM failed for %s: %s", bam_chrom, exc)
        return {}
    try:
        for pcol in bam.pileup(contig=bam_chrom, start=ps[0] - 1, stop=ps[-1],
                               max_depth=2_000_000, min_base_quality=0,
                               stepper="all", truncate=True, ignore_overlaps=False):
            pos1 = pcol.reference_pos + 1
            alts = variant_pos.get(pos1)
            if alts is None:
                continue
            snv_key = f"{bam_chrom}_{pos1}"
            for pr in pcol.pileups:
                if pr.is_del or pr.is_refskip or pr.query_position is None:
                    continue
                if pr.alignment.query_sequence[pr.query_position] not in alts:
                    continue
                try:
                    cb = pr.alignment.get_tag("CB")
                except KeyError:
                    continue
                if cb not in in_tissue:
                    continue
                hits[(cb, snv_key)] += 1
    except Exception as exc:
        log.warning("pileup error on %s: %s", bam_chrom, exc)
    finally:
        bam.close()

    out: Dict[str, Set[str]] = defaultdict(set)
    for (cb, key), c in hits.items():
        if c >= min_alt_reads:
            out[cb].add(key)
    return out


def scan_bam(bam_path: str, variant_positions: Dict[str, Dict[int, FrozenSet[str]]],
             in_tissue: Set[str], min_alt_reads: int, max_workers: int
             ) -> Dict[str, Set[str]]:
    if pysam is None:
        raise RuntimeError("pysam not installed. `conda activate snv_caller` (has pysam).")
    vcf_chroms = sorted(variant_positions)
    cmap = build_chrom_map(bam_path, vcf_chroms)
    to_scan = [c for c in vcf_chroms if c in cmap]
    n_pos = sum(len(variant_positions[c]) for c in to_scan)
    log.info("BAM scan: %d chroms, %s SNV positions, %d in-tissue spots, %d threads",
             len(to_scan), f"{n_pos:,}", len(in_tissue), max_workers)

    spot_snvs: Dict[str, Set[str]] = defaultdict(set)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(scan_one_chrom, bam_path, cmap[c],
                          variant_positions[c], in_tissue, min_alt_reads): c
                for c in to_scan}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scanning chroms"):
            try:
                part = fut.result()
            except Exception as exc:
                log.warning("scan %s raised: %s", futs[fut], exc)
                continue
            for cb, keys in part.items():
                spot_snvs[cb] |= keys
    total = sum(len(v) for v in spot_snvs.values())
    log.info("Scan done: %d spots carry >=1 strelka2 SNV; %s (spot×SNV) presences",
             len(spot_snvs), f"{total:,}")
    return spot_snvs


# ---[ Output ]----------------------------------------------------------------

def write_spot_txt(spot_snvs: Dict[str, Set[str]], in_tissue: Set[str], out_dir: str) -> None:
    """One <barcode>.txt per in-tissue spot, lines 'chrom<TAB>pos'. Empty spots get empty files."""
    os.makedirs(out_dir, exist_ok=True)
    for bc in in_tissue:
        keys = spot_snvs.get(bc, set())
        # snv_key is 'chrom_pos'; chrom may itself contain no '_' for DLPFC (1..22,X,Y)
        rows = []
        for k in keys:
            chrom, pos = k.rsplit("_", 1)
            rows.append((chrom, int(pos)))
        rows.sort(key=lambda t: (t[0], t[1]))
        with open(os.path.join(out_dir, f"{bc}.txt"), "w") as f:
            for chrom, pos in rows:
                f.write(f"{chrom}\t{pos}\n")
    log.info("Wrote %d per-spot files to %s", len(in_tissue), out_dir)


# ---[ Main ]------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Project strelka2 SNVs onto spatial spots (DLPFC).")
    ap.add_argument("--section_id", required=True)
    ap.add_argument("--min-alt-reads", type=int, default=1,
                    help="Min ALT-supporting reads in a spot to call the SNV present (default 1).")
    ap.add_argument("--max-workers", type=int, default=22)
    ap.add_argument("--include-nonpass", action="store_true",
                    help="Include non-PASS strelka2 records (default: PASS only).")
    ap.add_argument("--vcf-override", default=None)
    ap.add_argument("--bam-override", default=None)
    ap.add_argument("--out-dir", default=None,
                    help="Default: data/dlpfc/{section}/strelka2/spot_snvs")
    args = ap.parse_args()

    sec = args.section_id
    vcf = args.vcf_override or os.path.join(
        PROJECT_DIR, "data/dlpfc", sec, "strelka2/results/variants/variants.vcf.gz")
    bam = args.bam_override or os.path.join(
        PROJECT_DIR, "data/dlpfc", sec, f"{sec}_merged.bam")
    pos_file = os.path.join(DLPFC_SPATIAL_BASE, sec, "spatial", "tissue_positions_list.csv")
    out_dir = args.out_dir or os.path.join(PROJECT_DIR, "data/dlpfc", sec, "strelka2/spot_snvs")

    for label, p in (("strelka2 VCF", vcf), ("merged BAM", bam),
                     ("BAM index", bam + ".bai"), ("tissue positions", pos_file)):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing {label}: {p}")

    log.info("Section %s | min_alt_reads=%d | PASS_only=%s",
             sec, args.min_alt_reads, not args.include_nonpass)
    log.info("  VCF: %s", vcf)
    log.info("  BAM: %s", bam)

    variant_positions = load_pass_snv_positions(vcf, pass_only=not args.include_nonpass)
    in_tissue = load_in_tissue_barcodes(pos_file)
    spot_snvs = scan_bam(bam, variant_positions, in_tissue,
                         args.min_alt_reads, args.max_workers)
    write_spot_txt(spot_snvs, in_tissue, out_dir)

    n_cols = len({k for v in spot_snvs.values() for k in v})
    log.info("Done. spot_snvs dir: %s", out_dir)
    log.info("Matrix preview: %d in-tissue spots × %d distinct strelka2 SNVs (observed in >=1 spot)",
             len(in_tissue), n_cols)
    log.info("Next: run_generate_matrix.py --dataset dlpfc --section_id %s "
             "--input-dir %s --caller strelka2 --output-name germline", sec, out_dir)


if __name__ == "__main__":
    main()
