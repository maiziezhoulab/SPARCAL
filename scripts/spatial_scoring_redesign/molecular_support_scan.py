#!/usr/bin/env python
"""
molecular_support_scan.py
==========================
Endpoint 1 (molecular support) input: for every candidate variant that
appears in ANY method's selected set (union, across all sections' selected
sets already written by build_selected_sets.py), count distinct ALT-supporting
UMI families (CB,UB pairs) -- NOT just ALT read count, since "RT errors live
in one family by construction" (task spec).

This is a READ-ONLY re-implementation of the ALT-read / UMI-family counting
logic in scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py
(scan_chromosome's som_alt_reads/som_ref_count/som_other_count accumulators),
stripped down to just that piece (the het-site read-backed-phasing logic in
that script is not needed here and is not reused). Same BAM, same read
filter, same convention, by design -- so numbers from this script and that
one are directly comparable.

Read filter: exclude unmapped/secondary/supplementary/QC-fail (flag 0x904).
Duplicate-flag reads are KEPT (matches project-wide convention: these BAMs
are already umi_tools-deduplicated; CellRanger's dup flag reflects gene-level
marking, not redundancy in this file -- see measure_readbacked_feasibility.py
module docstring for the full justification).

Output: data/spatial_scoring_redesign_2026-08-31/_cache/{section}/molecular_support.tsv
  columns: variant, n_alt_reads, n_ref_reads, n_other_reads, n_distinct_alt_umi_families
"""
import argparse
import bisect
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict

import pandas as pd
import pysam

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

FLAG_EXCLUDE = 0x904


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def union_candidate_variants(section):
    sel_dir = os.path.join(common.OUT_ROOT, "selected_sets")
    manifest_path = os.path.join(sel_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest["section"] == section]
    variants = set()
    for _, row in manifest.iterrows():
        df = pd.read_csv(row["path"], sep="\t")
        variants.update(df["variant"].tolist())
    log(f"{section}: union of all selected sets = {len(variants)} distinct variants "
        f"(from {len(manifest)} selected-set files)")
    return variants


def scan_chromosome(args):
    section, bam_path, bam_contig, chrom_bare, positions_sorted, allele_map = args
    t0 = time.time()
    bf = pysam.AlignmentFile(bam_path, "rb")

    alt_reads = defaultdict(list)   # pos0 -> list of (CB,UB)
    ref_count = defaultdict(int)
    other_count = defaultdict(int)
    n_reads = 0
    n_pass = 0

    pos_arr = positions_sorted
    for read in bf.fetch(bam_contig):
        n_reads += 1
        if read.flag & FLAG_EXCLUDE:
            continue
        n_pass += 1
        rs, re = read.reference_start, read.reference_end
        if rs is None or re is None:
            continue
        lo = bisect.bisect_left(pos_arr, rs)
        hi = bisect.bisect_left(pos_arr, re)
        if hi <= lo:
            continue
        cands = pos_arr[lo:hi]
        if read.query_sequence is None:
            continue
        aligned = {ref_p: query_p for query_p, ref_p in read.get_aligned_pairs(matches_only=True)}
        seq = read.query_sequence
        cb = ub = None
        got_tags = False
        for p in cands:
            qpos = aligned.get(p)
            if qpos is None:
                continue
            base = seq[qpos].upper()
            ref, alt = allele_map[(chrom_bare, p)]
            if base == alt.upper():
                if not got_tags:
                    try:
                        cb = read.get_tag("CB")
                    except KeyError:
                        cb = None
                    try:
                        ub = read.get_tag("UB")
                    except KeyError:
                        ub = None
                    got_tags = True
                alt_reads[p].append((cb, ub))
            elif base == ref.upper():
                ref_count[p] += 1
            else:
                other_count[p] += 1
    bf.close()
    log(f"  [{section}:{chrom_bare}] {n_reads} reads, {n_pass} pass-filter, "
        f"{len(pos_arr)} candidate sites, {time.time() - t0:.1f}s")
    return chrom_bare, alt_reads, ref_count, other_count


def run_section(section, threads=10):
    cfg = common.SECTIONS[section]
    outdir = common.cache_dir(section)
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t")

    union_variants = union_candidate_variants(section)
    sub = candidates[candidates["variant"].isin(union_variants)].copy()
    sub = sub.dropna(subset=["ref", "alt"])
    log(f"{section}: {len(sub)}/{len(union_variants)} union variants have known ref/alt "
        f"(the rest never appeared in ANY spot's vcf_by_spot row -- zero local coverage anywhere, "
        f"cannot be scanned; they will show as n_alt_reads=NA)")

    by_chrom_pos = defaultdict(set)
    allele_map = {}
    for _, row in sub.iterrows():
        chrom_bare = str(row["chrom"])
        pos0 = int(row["pos"]) - 1
        by_chrom_pos[chrom_bare].add(pos0)
        allele_map[(chrom_bare, pos0)] = (row["ref"], row["alt"])
    by_chrom_pos = {c: sorted(s) for c, s in by_chrom_pos.items()}

    tasks = []
    for c, positions_sorted in by_chrom_pos.items():
        bam_contig = f"chr{c}" if cfg["chr_prefix_in_bam"] else c
        tasks.append((section, cfg["bam"], bam_contig, c, positions_sorted, allele_map))

    log(f"{section}: scanning {len(tasks)} chromosomes with {threads} workers (BAM={cfg['bam']}) ...")
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(threads) as pool:
        results = pool.map(scan_chromosome, tasks)
    log(f"{section}: BAM scan done in {time.time() - t0:.1f}s")

    rows = []
    for chrom_bare, alt_reads, ref_count, other_count in results:
        for pos0, entries in alt_reads.items():
            fams = set((cb, ub) for cb, ub in entries if cb and ub)
            rows.append(dict(
                variant=f"{chrom_bare}_{pos0 + 1}",
                n_alt_reads=len(entries),
                n_ref_reads=ref_count.get(pos0, 0),
                n_other_reads=other_count.get(pos0, 0),
                n_distinct_alt_umi_families=len(fams),
            ))
        # positions with zero alt reads at all still need a row (n_alt=0)
        for pos0 in by_chrom_pos[chrom_bare]:
            if pos0 not in alt_reads:
                rows.append(dict(
                    variant=f"{chrom_bare}_{pos0 + 1}",
                    n_alt_reads=0,
                    n_ref_reads=ref_count.get(pos0, 0),
                    n_other_reads=other_count.get(pos0, 0),
                    n_distinct_alt_umi_families=0,
                ))

    out = pd.DataFrame(rows).drop_duplicates(subset=["variant"])
    out_path = os.path.join(outdir, "molecular_support.tsv")
    out.to_csv(out_path, sep="\t", index=False)
    log(f"{section}: wrote {out_path} ({len(out)} variants)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()
    run_section(args.section, threads=args.threads)
    log("DONE")
