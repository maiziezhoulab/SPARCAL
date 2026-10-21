#!/usr/bin/env python3
"""Recount per-spot REF/ALT support at every step-6 per-spot VCF record, in reads and in molecules.

Why: step 6 (scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py, filter_bam_one_chrom) writes a per-spot
record for every read-covered candidate position, including REF-only coverage, and its AD counts every mapped
read (secondary, supplementary, QC-fail, duplicate-flagged) with no UMI collapse. The step-7 code that built the
P4/P6 2026-09-07 outputs treats every record as presence. This tool provides the counts needed to rebuild
presence as "ALT molecules >= 1" without rerunning steps 1-5.

Inputs (read-only): <spotprofile_dir>/vcf_by_spot/{bc}.vcf.gz and, by default,
<spotprofile_dir>/bam_by_spot/{bc}.bam. ``--bam-dir`` can instead select intact source BAMs; this was added
2026-10-09 because corrupt step-6 per-spot BAMs were found in nine DLPFC sections.

Per record it reports
  ad_ref, ad_alt       the AD already in the VCF
  ref_read, alt_read   recounted with the step-6 rule (every mapped read covering the base); must equal AD
  ref_mol, alt_mol     molecule rule: primary, non-QC-fail alignments with a UB tag; reads of one UB vote, the
                       majority base is that molecule's base; a UB whose top bases tie contributes nothing
  n_tie                UBs dropped for a tie at this locus
Duplicate-flagged reads (0x400) are kept: the per-spot BAMs come from a position-based umi_tools dedup, so
Space Ranger's flags mark extra fragments of molecules, which the UB collapse handles.

Output: one gzipped TSV with a row per (spot, record).
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import pysam

COLUMNS = ["barcode", "chrom", "pos", "ref", "alt", "ad_ref", "ad_alt",
           "ref_read", "alt_read", "ref_mol", "alt_mol", "n_tie"]


def read_records(vcf_path):
    """{chrom: {pos: [(ref, alt, ad_ref, ad_alt), ...]}} from one per-spot VCF."""
    recs = defaultdict(lambda: defaultdict(list))
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            fmt = f[8].split(":")
            val = f[9].split(":")
            ad = val[fmt.index("AD")].split(",") if "AD" in fmt else ["0", "0"]
            recs[f[0]][int(f[1])].append((f[3].upper(), f[4].upper(), int(ad[0]), int(ad[1])))
    return recs


def count_spot(args):
    bc, vcf_path, bam_path, keep_keys = args
    recs = read_records(vcf_path)
    rows = []
    if not recs:
        return bc, rows
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        refs = set(bam.references)
        for chrom, by_pos in recs.items():
            if keep_keys is not None:
                c = chrom[3:] if chrom.startswith("chr") else chrom
                by_pos = {p: v for p, v in by_pos.items() if f"{c}_{p}" in keep_keys}
                if not by_pos:
                    continue
            positions = sorted(by_pos)
            read_base = defaultdict(Counter)            # pos -> base -> reads (step-6 rule)
            umi_votes = defaultdict(lambda: defaultdict(Counter))  # pos -> UB -> base -> reads
            # VCFs say chr1, some per-spot BAMs (DLPFC) say 1: map to whichever the BAM has.
            bam_chrom = next((c for c in (chrom, chrom[3:] if chrom.startswith("chr") else "chr" + chrom)
                              if c in refs), None)
            if bam_chrom is not None:
                for read in bam.fetch(bam_chrom, positions[0] - 1, positions[-1]):
                    if read.is_unmapped or read.query_sequence is None:
                        continue
                    start, end = read.reference_start + 1, read.reference_end
                    if end is None:
                        continue
                    lo = bisect.bisect_left(positions, start)
                    hi = bisect.bisect_right(positions, end)
                    if lo >= hi:
                        continue
                    wanted = set(positions[lo:hi])
                    seq = read.query_sequence
                    molecule_ok = (not read.is_secondary and not read.is_supplementary
                                   and not read.is_qcfail and read.has_tag("UB"))
                    ub = read.get_tag("UB") if molecule_ok else None
                    for qpos, rpos in read.get_aligned_pairs(matches_only=True):
                        p = rpos + 1
                        if p in wanted:
                            base = seq[qpos].upper()
                            read_base[p][base] += 1
                            if molecule_ok:
                                umi_votes[p][ub][base] += 1
            for p in positions:
                mol = Counter()
                ties = 0
                for votes in umi_votes[p].values():
                    top = votes.most_common(2)
                    if len(top) == 2 and top[0][1] == top[1][1]:
                        ties += 1
                    else:
                        mol[top[0][0]] += 1
                for ref, alt, ad_ref, ad_alt in by_pos[p]:
                    rows.append((bc, chrom, p, ref, alt, ad_ref, ad_alt,
                                 read_base[p][ref], read_base[p][alt], mol[ref], mol[alt], ties))
    return bc, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spotprofile-dir", required=True)
    ap.add_argument("--bam-dir", help="optional BAM directory; default: <spotprofile-dir>/bam_by_spot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--barcodes", help="optional file, one barcode per line (e.g. in-tissue spots)")
    ap.add_argument("--keys", help="optional file of chrom_pos keys (no chr) to restrict recounting")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    vdir = os.path.join(a.spotprofile_dir, "vcf_by_spot")
    bdir = a.bam_dir if a.bam_dir is not None else os.path.join(a.spotprofile_dir, "bam_by_spot")
    bcs = sorted(f[:-7] for f in os.listdir(vdir) if f.endswith(".vcf.gz"))
    if a.barcodes:
        keep = {l.strip() for l in open(a.barcodes) if l.strip()}
        bcs = [b for b in bcs if b in keep]
    keys = {l.strip() for l in open(a.keys) if l.strip()} if a.keys else None
    jobs = []
    for bc in bcs:
        bam = os.path.join(bdir, bc + ".bam")
        if not os.path.exists(bam):
            sys.exit(f"missing BAM for {bc}: {bam}")
        jobs.append((bc, os.path.join(vdir, bc + ".vcf.gz"), bam, keys))
    print(f"spots={len(jobs)} keys={'all' if keys is None else len(keys)}", flush=True)

    n = mism = 0
    tmp = a.out + ".partial"
    with gzip.open(tmp, "wt") as out, ProcessPoolExecutor(a.workers) as ex:
        out.write("\t".join(COLUMNS) + "\n")
        for i, (bc, rows) in enumerate(ex.map(count_spot, jobs, chunksize=8), 1):
            for r in rows:
                out.write("\t".join(map(str, r)) + "\n")
                n += 1
                mism += (r[5], r[6]) != (r[7], r[8])
            if i % 500 == 0:
                print(f"  {i}/{len(jobs)} spots, {n:,} records, read-rule AD mismatches {mism:,}", flush=True)
    os.replace(tmp, a.out)
    print(f"done: {n:,} records; read-rule recount vs VCF AD mismatches: {mism:,}")
    if mism:
        print("WARNING: read-level recount does not reproduce step-6 AD for some records", file=sys.stderr)


if __name__ == "__main__":
    main()
