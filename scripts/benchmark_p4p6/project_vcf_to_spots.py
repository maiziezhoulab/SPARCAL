#!/usr/bin/env python3
"""Project PASS biallelic SNVs in a VCF onto CB-tagged Visium reads.

This path-generalizes scripts/tools/strelka2_to_spot_snvs.py and preserves its
pileup/read behavior: stepper="all", min_base_quality=0, max_depth=2,000,000,
ignore_overlaps=False; deletions, reference skips, missing query positions,
missing CB tags, and out-of-tissue reads are excluded. Presence requires the
exact ALT in >= --min-alt-reads reads in a spot.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import logging
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pysam
from tqdm import tqdm

LOG = logging.getLogger("project_vcf_to_spots")


def norm_chrom(value: str) -> str:
    value = str(value).strip()
    return value[3:] if value.lower().startswith("chr") else value


def chrom_sort_key(chrom: str) -> tuple:
    chrom = norm_chrom(chrom)
    return (0, int(chrom)) if chrom.isdigit() else (1, chrom)


@dataclass(frozen=True)
class Variant:
    chrom: str
    pos: int
    ref: str
    alt: str

    @property
    def key(self) -> str:
        return f"{norm_chrom(self.chrom)}_{self.pos}_{self.ref}_{self.alt}"


def open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_pass_biallelic_snvs(vcf: str) -> Dict[str, Dict[int, List[Variant]]]:
    result: Dict[str, Dict[int, List[Variant]]] = defaultdict(lambda: defaultdict(list))
    n_all = n_pass = n_snv = 0
    with open_text(vcf) as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            n_all += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 7 or fields[6] != "PASS":
                continue
            n_pass += 1
            chrom, pos_s, ref, alt = fields[0], fields[1], fields[3].upper(), fields[4].upper()
            if "," in alt or len(ref) != 1 or len(alt) != 1 or alt in {".", "*", "N"}:
                continue
            variant = Variant(chrom, int(pos_s), ref, alt)
            result[chrom][variant.pos].append(variant)
            n_snv += 1
    LOG.info("VCF records=%d PASS=%d PASS_biallelic_SNV=%d", n_all, n_pass, n_snv)
    return {chrom: dict(by_pos) for chrom, by_pos in result.items()}


def load_in_tissue_barcodes(path: str) -> List[str]:
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Empty tissue positions file: {path}")
    if rows[0] and rows[0][0].strip().lower() == "barcode":
        header = {name.strip(): i for i, name in enumerate(rows[0])}
        barcode_i, tissue_i, data = header["barcode"], header["in_tissue"], rows[1:]
    else:
        barcode_i, tissue_i, data = 0, 1, rows
    seen: Set[str] = set()
    out: List[str] = []
    for row in data:
        if len(row) <= max(barcode_i, tissue_i) or row[tissue_i].strip() != "1":
            continue
        barcode = row[barcode_i].strip()
        if barcode and barcode not in seen:
            seen.add(barcode)
            out.append(barcode)
    LOG.info("Loaded %d ordered in-tissue barcodes", len(out))
    return out


def build_chrom_map(bam_path: str, vcf_chroms: Iterable[str]) -> Dict[str, str]:
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        refs = set(bam.references)
    mapping: Dict[str, str] = {}
    for chrom in vcf_chroms:
        candidates = (chrom, chrom[3:] if chrom.startswith("chr") else "chr" + chrom)
        for candidate in candidates:
            if candidate in refs:
                mapping[chrom] = candidate
                break
    missing = sorted(set(vcf_chroms) - set(mapping), key=chrom_sort_key)
    if missing:
        LOG.warning("VCF contigs absent from BAM and skipped: %s", ",".join(missing))
    return mapping


def scan_one_chrom(
    bam_path: str,
    bam_chrom: str,
    variants_by_pos: Mapping[int, Sequence[Variant]],
    in_tissue: Set[str],
    min_alt_reads: int,
) -> Dict[str, Set[str]]:
    if not variants_by_pos:
        return {}
    positions = sorted(variants_by_pos)
    hits: MutableMapping[Tuple[str, str], int] = defaultdict(int)
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for pcol in bam.pileup(
            contig=bam_chrom, start=positions[0] - 1, stop=positions[-1],
            max_depth=2_000_000, min_base_quality=0, stepper="all",
            truncate=True, ignore_overlaps=False,
        ):
            pos1 = pcol.reference_pos + 1
            variants = variants_by_pos.get(pos1)
            if variants is None:
                continue
            by_alt: Dict[str, List[str]] = defaultdict(list)
            for variant in variants:
                by_alt[variant.alt].append(variant.key)
            for pileup_read in pcol.pileups:
                if pileup_read.is_del or pileup_read.is_refskip or pileup_read.query_position is None:
                    continue
                alignment = pileup_read.alignment
                base = alignment.query_sequence[pileup_read.query_position].upper()
                keys = by_alt.get(base)
                if not keys:
                    continue
                try:
                    barcode = alignment.get_tag("CB")
                except KeyError:
                    continue
                if barcode not in in_tissue:
                    continue
                for key in keys:
                    hits[(barcode, key)] += 1
    out: Dict[str, Set[str]] = defaultdict(set)
    for (barcode, key), count in hits.items():
        if count >= min_alt_reads:
            out[barcode].add(key)
    return out


def scan_bam(
    bam_path: str,
    variants: Mapping[str, Mapping[int, Sequence[Variant]]],
    barcodes: Sequence[str],
    min_alt_reads: int,
    workers: int,
) -> Dict[str, Set[str]]:
    chrom_map = build_chrom_map(bam_path, variants)
    barcode_set = set(barcodes)
    result: Dict[str, Set[str]] = defaultdict(set)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(scan_one_chrom, bam_path, chrom_map[chrom], variants[chrom],
                            barcode_set, min_alt_reads): chrom
            for chrom in variants if chrom in chrom_map
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning contigs"):
            chrom = futures[future]
            try:
                partial = future.result()
            except Exception:
                LOG.exception("Pileup failed on %s", chrom)
                raise
            for barcode, keys in partial.items():
                result[barcode].update(keys)
    return result


def write_matrix(spot_snvs: Mapping[str, Set[str]], barcodes: Sequence[str], output: str) -> pd.DataFrame:
    columns = sorted(
        {key for keys in spot_snvs.values() for key in keys},
        key=lambda key: (chrom_sort_key(key.rsplit("_", 3)[0]), int(key.rsplit("_", 3)[1]), key),
    )
    column_index = {key: i for i, key in enumerate(columns)}
    matrix = np.zeros((len(barcodes), len(columns)), dtype=np.uint8)
    for row_i, barcode in enumerate(barcodes):
        for key in spot_snvs.get(barcode, ()):
            matrix[row_i, column_index[key]] = 1
    frame = pd.DataFrame(matrix, index=list(barcodes), columns=columns, dtype=np.uint8)
    frame.to_pickle(output)
    LOG.info("Wrote %s: shape=%s presences=%d", output, frame.shape, int(matrix.sum()))
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vcf", required=True)
    parser.add_argument("--bam", required=True)
    parser.add_argument("--tissue-positions", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--section", required=True)
    parser.add_argument("--caller", required=True, choices=("strelka2", "strelka2_rna", "gatk"))
    parser.add_argument("--min-alt-reads", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for path in (args.vcf, args.bam, args.tissue_positions):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
    os.makedirs(args.outdir, exist_ok=True)
    output = os.path.join(args.outdir, f"{args.section}_{args.caller}_presence_matrix.pkl")
    variants = load_pass_biallelic_snvs(args.vcf)
    barcodes = load_in_tissue_barcodes(args.tissue_positions)
    spot_snvs = scan_bam(args.bam, variants, barcodes, args.min_alt_reads, args.max_workers)
    write_matrix(spot_snvs, barcodes, output)


if __name__ == "__main__":
    main()
