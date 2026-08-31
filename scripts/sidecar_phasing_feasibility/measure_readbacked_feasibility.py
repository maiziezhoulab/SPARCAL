#!/usr/bin/env python
"""
measure_readbacked_feasibility.py
==================================

PURE MEASUREMENT script for the SPARCAL "read-backed phasing" feasibility
question (P-1 / P-2 gate). Does NOT modify the shipped pipeline, does NOT
write into any existing data/<sample>/output_VCFs/ tree, does NOT touch
scripts/1_calling .. scripts/6_spatial_filter, and does NOT touch the other
agent's scripts/sidecar_phasing/ or data/sidecar_phasing_probe_2026-08-23/.

Inputs (all read-only):
  - the per-section deduplicated whole-BAM (possorted_genome_bam.dedup.bam),
    same file the shipped pipeline's split_BAM/{barcode}.bam are split from
    (see run_slurm/{P4_tumor,P6_tumor,DCIS}/0_umidedup_split_*.sh)
  - the SPARCAL final germline call set restricted to race=="defined"
    (1000-Genomes-resolved), spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt
  - the raw per-section pseudobulk genotype VCF (merged_sorted_gt.vcf.gz,
    output of samtools mpileup+call on the multi-BAM list) used ONLY to
    determine zygosity (het vs hom-alt) at each retained germline position
  - the SPARCAL final somatic call set,
    spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt

Outputs (this script only writes here):
  data/readbacked_feasibility_2026-08-23/
    01_het_pair_distance.csv         one row per het site: nearest-neighbour distance
    01b_het_pair_distance_summary.csv per-section threshold summary
    02_read_level_pairs.csv          one row per distinct het-pair co-observed by >=1 read
    02b_read_level_summary.csv       per-section read-level co-observation summary
    03_umi_family_detail.csv         one row per UMI family covering >=2 het sites
    03b_umi_family_summary.csv       per-section UMI-family co-observation summary
    04_somatic_gate.csv              one row per retained somatic candidate (the P-2 funnel)
    05_alt_umi_family_structure.csv  per-candidate distinct-ALT-UMI-family distribution
    summary.json                     all headline numbers, machine-readable, per section

Read filtering convention (documented, see RESULTS.md "dup flag" caveat):
  exclude unmapped / secondary / supplementary / QC-fail reads.
  Duplicate-flag (SAM 0x400) reads are KEPT (not excluded) -- empirically this
  flag is inherited unchanged from the ORIGINAL possorted_genome_bam.bam
  (confirmed: ~67% of reads are dup-flagged even in the PRE-dedup file), i.e.
  it reflects CellRanger's own gene+UMI duplicate marking, not redundancy
  within the umi_tools-deduplicated file. Excluding it would throw away ~2/3
  of the molecules umi_tools already certified as unique, and would
  specifically cripple UMI-family multi-position linkage (the reads that
  extend range beyond one read are disproportionately the ones CellRanger
  flagged dup at the gene level). The shipped calling pipeline's `samtools
  mpileup` DOES exclude dup-flagged reads by default (no --ff override) --
  this is a real, previously-undocumented discrepancy, reported as a finding.

Re-runnable: `python measure_readbacked_feasibility.py --sections P4 P6 DCIS1 DCIS2`
"""
import argparse
import bisect
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from itertools import combinations

import pysam

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"

# ---------------------------------------------------------------------------
# Section configuration -- traced from run_slurm/{P4_tumor,P6_tumor,DCIS}/0_umidedup_split_*.sh
# (the dedup BAM that split_BAM/{barcode}.bam, and hence every downstream
# pipeline step, is derived from) and from data/{P4_tumor,P6_tumor,dcis1,dcis2}/.
# ---------------------------------------------------------------------------
SECTIONS = {
    "P4": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        germline_txt=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt",
        somatic_txt=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        ambiguous_txt=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.txt",
        raw_gt_vcf=f"{REPO}/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz",
        chr_prefix_in_bam=True,   # BAM contigs: chr1, chr2, ...
    ),
    "P6": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        germline_txt=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt",
        somatic_txt=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        ambiguous_txt=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.txt",
        raw_gt_vcf=f"{REPO}/data/P6_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz",
        chr_prefix_in_bam=True,
    ),
    "DCIS1": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.dedup.bam",
        germline_txt=f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt",
        somatic_txt=f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        ambiguous_txt=f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.txt",
        raw_gt_vcf=f"{REPO}/data/dcis1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz",
        chr_prefix_in_bam=False,  # BAM contigs: 1, 2, ... (bare; VCF/txt calls are chr-prefixed -- known mismatch)
    ),
    "DCIS2": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.dedup.bam",
        germline_txt=f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt",
        somatic_txt=f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        ambiguous_txt=f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.txt",
        raw_gt_vcf=f"{REPO}/data/dcis2/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz",
        chr_prefix_in_bam=False,
    ),
}

FLAG_EXCLUDE = 0x904  # UNMAP(0x4) + SECONDARY(0x100) + SUPPLEMENTARY(0x800); DUP intentionally kept (see module docstring)
HET_RANGE_BP_DEFAULT = 1000  # "in range" positional pre-filter for the somatic gate (step 3 of measurement 4)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Loading & zygosity join
# ---------------------------------------------------------------------------
def load_raw_gt(raw_gt_vcf_path):
    """Load the pseudobulk mpileup+call VCF into a dict (chrom,pos,ref,alt) -> GT string.
    chrom here keeps whatever prefix convention the VCF itself uses (chr-prefixed for all 4 sections)."""
    gt = {}
    vf = pysam.VariantFile(raw_gt_vcf_path)
    for rec in vf.fetch():
        sample = rec.samples[0]
        gt_tuple = sample.get("GT")
        if gt_tuple is None:
            continue
        for alt in rec.alts or ():
            gt[(rec.chrom, rec.pos, rec.ref, alt)] = gt_tuple
    vf.close()
    return gt


def is_het(gt_tuple):
    if gt_tuple is None:
        return False
    alleles = set(a for a in gt_tuple if a is not None)
    return alleles == {0, 1}


def load_het_sites(section_cfg):
    """Return dict: chrom_bare(str) -> sorted list of 0-based positions (deduped),
    plus a dict (chrom_bare,pos0) -> (ref,alt), plus match/zygosity stats."""
    raw_gt = load_raw_gt(section_cfg["raw_gt_vcf"])
    n_defined = 0
    n_matched = 0
    n_het = 0
    n_hom = 0
    n_other_gt = 0
    per_chrom = defaultdict(set)
    allele_map = {}
    with open(section_cfg["germline_txt"]) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["race"] != "defined":
                continue
            n_defined += 1
            chrom_bare = row["chrom"]
            pos1 = int(row["pos"])
            ref, alt = row["ref"], row["alt"]
            key = (f"chr{chrom_bare}", pos1, ref, alt)
            gt_tuple = raw_gt.get(key)
            if gt_tuple is None:
                continue
            n_matched += 1
            if is_het(gt_tuple):
                n_het += 1
                pos0 = pos1 - 1
                per_chrom[chrom_bare].add(pos0)
                allele_map[(chrom_bare, pos0)] = (ref, alt)
            else:
                alleles = set(a for a in gt_tuple if a is not None)
                if alleles in ({0}, {1}):
                    n_hom += 1
                else:
                    n_other_gt += 1
    sorted_per_chrom = {c: sorted(s) for c, s in per_chrom.items()}
    stats = dict(n_defined_1kgp=n_defined, n_matched_to_raw_gt=n_matched,
                 n_heterozygous=n_het, n_homozygous=n_hom, n_other_gt=n_other_gt)
    return sorted_per_chrom, allele_map, stats


def load_somatic_sites(section_cfg, query_class="somatic"):
    """query_class: 'somatic' (default, the retained somatic_denovo class) or
    'ambiguous' (the spatial-filter's unresolved third class, same file schema:
    chrom,pos,ref,alt,germline_score,somatic_score,race) -- used for the P-2
    comparison-class funnel (coordinator request 2026-08-24)."""
    txt_key = "somatic_txt" if query_class == "somatic" else "ambiguous_txt"
    per_chrom = defaultdict(set)
    allele_map = {}
    n_total = 0
    with open(section_cfg[txt_key]) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            n_total += 1
            chrom_bare = row["chrom"]
            pos1 = int(row["pos"])
            pos0 = pos1 - 1
            ref, alt = row["ref"], row["alt"]
            per_chrom[chrom_bare].add(pos0)
            allele_map[(chrom_bare, pos0)] = (ref, alt)
    sorted_per_chrom = {c: sorted(s) for c, s in per_chrom.items()}
    return sorted_per_chrom, allele_map, n_total


# ---------------------------------------------------------------------------
# Measurement 1: het-pair distance (pure position arithmetic, no BAM needed)
# ---------------------------------------------------------------------------
def measure_het_pair_distance(section, het_by_chrom):
    rows = []
    for chrom, positions in het_by_chrom.items():
        n = len(positions)
        if n == 1:
            rows.append((section, chrom, positions[0] + 1, None))
            continue
        for i, p in enumerate(positions):
            dists = []
            if i > 0:
                dists.append(p - positions[i - 1])
            if i < n - 1:
                dists.append(positions[i + 1] - p)
            nearest = min(dists) if dists else None
            rows.append((section, chrom, p + 1, nearest))
    return rows


# ---------------------------------------------------------------------------
# Per-chromosome BAM worker (runs in a subprocess)
# ---------------------------------------------------------------------------
def scan_chromosome(args):
    (section, bam_path, bam_contig, chrom_bare, het_pos_sorted, som_pos_sorted,
     som_allele) = args
    t0 = time.time()
    bf = pysam.AlignmentFile(bam_path, "rb")

    n_reads = 0
    n_pass = 0
    n_dup_flagged = 0
    n_cover_ge2_het = 0
    pair_counter = Counter()             # (pos0_i, pos0_j) -> n reads
    umi_family_hets = defaultdict(set)   # (CB,UB) -> set(het pos0)
    umi_family_nreads = Counter()        # (CB,UB) -> n reads in family (that touch >=1 het OR som candidate; cheap proxy)
    ref_span_sample = []

    # somatic accumulators
    som_alt_reads = defaultdict(list)    # som pos0 -> list of (CB,UB, covered_het_set_from_this_read)
    som_ref_count = Counter()
    som_other_count = Counter()

    has_het = bool(het_pos_sorted)
    has_som = bool(som_pos_sorted)

    for read in bf.fetch(bam_contig):
        n_reads += 1
        if read.flag & FLAG_EXCLUDE:
            continue
        if read.is_duplicate:
            n_dup_flagged += 1
        n_pass += 1
        rs, re = read.reference_start, read.reference_end
        if re is None or rs is None:
            continue
        if n_pass % 20000 == 0:
            ref_span_sample.append(re - rs)

        het_lo = bisect.bisect_left(het_pos_sorted, rs) if has_het else 0
        het_hi = bisect.bisect_left(het_pos_sorted, re) if has_het else 0
        som_lo = bisect.bisect_left(som_pos_sorted, rs) if has_som else 0
        som_hi = bisect.bisect_left(som_pos_sorted, re) if has_som else 0

        het_cands = het_pos_sorted[het_lo:het_hi] if het_hi > het_lo else ()
        som_cands = som_pos_sorted[som_lo:som_hi] if som_hi > som_lo else ()
        if not het_cands and not som_cands:
            continue

        # pysam returns (query_pos, ref_pos) tuples; we need ref_pos(0based) -> query_pos for lookups
        aligned = {ref_p: query_p for query_p, ref_p in read.get_aligned_pairs(matches_only=True)}
        covered_hets = [p for p in het_cands if p in aligned]

        if len(covered_hets) >= 2:
            n_cover_ge2_het += 1
            if len(covered_hets) <= 50:
                for a, b in combinations(covered_hets, 2):
                    pair_counter[(a, b)] += 1

        cb = ub = None
        if covered_hets or som_cands:
            try:
                cb = read.get_tag("CB")
            except KeyError:
                cb = None
            try:
                ub = read.get_tag("UB")
            except KeyError:
                ub = None

        if covered_hets and cb and ub:
            fam = (cb, ub)
            umi_family_hets[fam].update(covered_hets)
            umi_family_nreads[fam] += 1

        if som_cands and read.query_sequence is not None:
            seq = read.query_sequence
            for p in som_cands:
                qpos = aligned.get(p)
                if qpos is None:
                    continue
                base = seq[qpos].upper()
                ref, alt = som_allele[(chrom_bare, p)]
                if base == alt.upper():
                    som_alt_reads[p].append((cb, ub, frozenset(covered_hets)))
                elif base == ref.upper():
                    som_ref_count[p] += 1
                else:
                    som_other_count[p] += 1

    bf.close()
    elapsed = time.time() - t0
    log(f"  [{section}:{chrom_bare}] done: {n_reads} reads, {n_pass} pass-filter "
        f"({100.0*n_dup_flagged/max(n_pass,1):.1f}% dup-flagged-but-kept), "
        f"{n_cover_ge2_het} reads cover>=2 het, {elapsed:.1f}s")

    return dict(
        section=section, chrom=chrom_bare,
        n_reads=n_reads, n_pass=n_pass, n_dup_flagged=n_dup_flagged,
        n_cover_ge2_het=n_cover_ge2_het, pair_counter=pair_counter,
        umi_family_hets=umi_family_hets, umi_family_nreads=umi_family_nreads,
        ref_span_sample=ref_span_sample,
        som_alt_reads=som_alt_reads, som_ref_count=som_ref_count, som_other_count=som_other_count,
        elapsed=elapsed,
    )


# ---------------------------------------------------------------------------
# Orchestration for one section
# ---------------------------------------------------------------------------
def run_section(section, out_dir, threads, het_range_bp, chroms_limit=None, query_class="somatic"):
    cfg = SECTIONS[section]
    log(f"=== {section} === loading het/{query_class} candidate sites")
    het_by_chrom, het_allele, het_stats = load_het_sites(cfg)
    som_by_chrom, som_allele, n_som_total = load_somatic_sites(cfg, query_class=query_class)
    log(f"{section}: {het_stats} ; n_somatic_candidates={n_som_total}")

    chroms = sorted(set(het_by_chrom) | set(som_by_chrom))
    if chroms_limit:
        chroms = [c for c in chroms if c in chroms_limit]

    # measurement 1 (no BAM needed)
    dist_rows = measure_het_pair_distance(section, het_by_chrom)

    # build per-chromosome worker args
    tasks = []
    for c in chroms:
        bam_contig = c if not cfg["chr_prefix_in_bam"] else f"chr{c}"
        tasks.append((section, cfg["bam"], bam_contig, c,
                      het_by_chrom.get(c, []), som_by_chrom.get(c, []), som_allele))

    log(f"{section}: scanning {len(chroms)} chromosomes with {threads} workers "
        f"(BAM={cfg['bam']})")
    results = []
    if threads <= 1:
        for t in tasks:
            results.append(scan_chromosome(t))
    else:
        with mp.Pool(threads) as pool:
            for r in pool.imap_unordered(scan_chromosome, tasks):
                results.append(r)

    # ---- aggregate measurement 2 (read-level) ----
    total_reads = sum(r["n_reads"] for r in results)
    total_pass = sum(r["n_pass"] for r in results)
    total_dup_flagged = sum(r["n_dup_flagged"] for r in results)
    total_cover_ge2 = sum(r["n_cover_ge2_het"] for r in results)
    all_pairs = Counter()
    for r in results:
        all_pairs.update(r["pair_counter"])

    read_level_rows = [(section, c, p0 + 1, p1 + 1, n)
                        for r in results for (p0, p1), n in r["pair_counter"].items()
                        for c in [r["chrom"]]]

    ref_span_all = []
    for r in results:
        ref_span_all.extend(r["ref_span_sample"])
    ref_span_all.sort()

    # ---- aggregate measurement 3 (UMI family) ----
    umi_detail_rows = []
    n_fam_ge1 = 0
    n_fam_ge2 = 0
    fam_size_dist = Counter()  # n_hets_covered -> n_families (only for families touching >=1 het)
    for r in results:
        chrom = r["chrom"]
        for fam, hets in r["umi_family_hets"].items():
            n_fam_ge1 += 1
            fam_size_dist[len(hets)] += 1
            if len(hets) >= 2:
                n_fam_ge2 += 1
                cb, ub = fam
                sorted_hets = sorted(hets)
                span = sorted_hets[-1] - sorted_hets[0]
                umi_detail_rows.append((section, chrom, cb, ub, len(hets),
                                         r["umi_family_nreads"][fam], span,
                                         ";".join(str(p + 1) for p in sorted_hets)))

    # ---- aggregate measurement 4 + 5 (somatic gate) ----
    het_pos_by_chrom = het_by_chrom  # sorted 0-based lists, already built

    som_alt_reads_combined = defaultdict(list)
    som_ref_combined = Counter()
    som_other_combined = Counter()
    for r in results:
        chrom = r["chrom"]
        for p, entries in r["som_alt_reads"].items():
            som_alt_reads_combined[(chrom, p)].extend(entries)
        for p, n in r["som_ref_count"].items():
            som_ref_combined[(chrom, p)] += n
        for p, n in r["som_other_count"].items():
            som_other_combined[(chrom, p)] += n

    # global umi_family_hets across the section (chrom-scoped keys folded in)
    umi_family_hets_global = {}
    for r in results:
        chrom = r["chrom"]
        for fam, hets in r["umi_family_hets"].items():
            umi_family_hets_global[(chrom, fam)] = hets

    gate_rows = []
    alt_umi_rows = []
    scanned_chroms = set(chroms)  # only chromosomes actually BAM-scanned this run
    for chrom in chroms:
        positions = som_by_chrom.get(chrom, [])
        het_list = het_pos_by_chrom.get(chrom, [])
        for p0 in positions:
            ref, alt = som_allele[(chrom, p0)]
            alt_entries = som_alt_reads_combined.get((chrom, p0), [])
            n_alt = len(alt_entries)
            n_ref = som_ref_combined.get((chrom, p0), 0)
            n_other = som_other_combined.get((chrom, p0), 0)
            ge2_alt = n_alt >= 2

            # step 3: positional "het in range" (independent of actual reads)
            if het_list:
                idx = bisect.bisect_left(het_list, p0)
                cand_dists = []
                if idx > 0:
                    cand_dists.append(p0 - het_list[idx - 1])
                if idx < len(het_list):
                    cand_dists.append(het_list[idx] - p0)
                nearest_het_dist = min(cand_dists) if cand_dists else None
            else:
                nearest_het_dist = None
            het_in_range = nearest_het_dist is not None and nearest_het_dist <= het_range_bp

            # step 4: actual co-observation among the ALT-supporting reads
            read_level_coobs = any(len(covered) > 0 for (_cb, _ub, covered) in alt_entries)
            alt_umi_families = set((cb, ub) for (cb, ub, _c) in alt_entries if cb and ub)
            umi_level_coobs = any(
                len(umi_family_hets_global.get((chrom, fam), ())) > 0
                for fam in alt_umi_families
            )
            testable = ge2_alt and het_in_range and (read_level_coobs or umi_level_coobs)

            gate_rows.append((
                section, chrom, p0 + 1, ref, alt, n_alt, n_ref, n_other, ge2_alt,
                nearest_het_dist, het_in_range, read_level_coobs, umi_level_coobs,
                testable, len(alt_umi_families),
            ))
            alt_umi_rows.append((section, chrom, p0 + 1, ref, alt, n_alt, len(alt_umi_families)))

    section_summary = dict(
        section=section,
        n_chroms_scanned=len(chroms),
        het=het_stats,
        n_somatic_candidates=n_som_total,
        bam_total_reads_scanned=total_reads,
        bam_reads_pass_flag_filter=total_pass,
        bam_reads_dup_flagged_but_kept=total_dup_flagged,
        dup_flagged_pct_of_pass=round(100.0 * total_dup_flagged / max(total_pass, 1), 2),
        n_reads_covering_ge2_het=total_cover_ge2,
        n_distinct_het_pairs_covered_by_reads=len(all_pairs),
        ref_span_median=(ref_span_all[len(ref_span_all)//2] if ref_span_all else None),
        ref_span_p90=(ref_span_all[int(0.9*len(ref_span_all))] if ref_span_all else None),
        ref_span_n_sampled=len(ref_span_all),
        n_umi_families_touching_ge1_het=n_fam_ge1,
        n_umi_families_touching_ge2_het=n_fam_ge2,
        gate_n_candidates=len(gate_rows),
        gate_n_ge2_alt=sum(1 for r in gate_rows if r[8]),
        gate_n_het_in_range=sum(1 for r in gate_rows if r[8] and r[10]),
        gate_n_testable=sum(1 for r in gate_rows if r[13]),
        gate_n_testable_via_read_level_only=sum(1 for r in gate_rows if r[13] and r[11]),
        gate_n_testable_via_umi_level_only=sum(1 for r in gate_rows if r[13] and r[12] and not r[11]),
        alt_umi_ge2_n=sum(1 for r in alt_umi_rows if r[5] >= 2 and r[6] >= 2),
        alt_umi_denominator_ge2_alt=sum(1 for r in alt_umi_rows if r[5] >= 2),
    )

    return dict(
        dist_rows=dist_rows,
        read_level_rows=read_level_rows,
        umi_detail_rows=umi_detail_rows,
        gate_rows=gate_rows,
        alt_umi_rows=alt_umi_rows,
        summary=section_summary,
    )


# ---------------------------------------------------------------------------
# CSV writers (append across sections)
# ---------------------------------------------------------------------------
def write_csv(path, header, rows, mode="w"):
    write_header = mode == "w" or not os.path.exists(path)
    with open(path, mode, newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS.keys()), choices=list(SECTIONS.keys()))
    ap.add_argument("--out-dir", default=f"{REPO}/data/readbacked_feasibility_2026-08-23")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--het-range-bp", type=int, default=HET_RANGE_BP_DEFAULT)
    ap.add_argument("--chroms", nargs="+", default=None, help="restrict to these bare chrom names (testing)")
    ap.add_argument("--fresh", action="store_true", help="overwrite CSVs instead of appending")
    ap.add_argument("--query-class", choices=["somatic", "ambiguous"], default="somatic",
                     help="'ambiguous' reuses the same het-site/BAM-scan machinery against the "
                          "spatial-filter's unresolved third class, for the P-2 comparison-class "
                          "funnel. Writes to separate 04/05_ambiguous_*.csv and does NOT re-write "
                          "01/02/03 (those are query-class-independent -- identical to the somatic run).")
    args = ap.parse_args()
    qc = args.query_class
    gate_name = "04_somatic_gate.csv" if qc == "somatic" else "04_ambiguous_gate.csv"
    altfam_name = ("05_alt_umi_family_structure.csv" if qc == "somatic"
                   else "05_alt_umi_family_structure_ambiguous.csv")

    os.makedirs(args.out_dir, exist_ok=True)

    all_summaries = {}
    for i, section in enumerate(args.sections):
        mode_dist = "w" if (args.fresh and i == 0) else "a"
        res = run_section(section, args.out_dir, args.threads, args.het_range_bp, args.chroms, query_class=qc)

        if qc == "somatic":
            write_csv(f"{args.out_dir}/01_het_pair_distance.csv",
                      ["section", "chrom", "pos_1based", "nearest_het_dist_bp"],
                      res["dist_rows"], mode_dist)
            write_csv(f"{args.out_dir}/02_read_level_pairs.csv",
                      ["section", "chrom", "pos1_1based", "pos2_1based", "n_reads_supporting_pair"],
                      res["read_level_rows"], mode_dist)
            write_csv(f"{args.out_dir}/03_umi_family_detail.csv",
                      ["section", "chrom", "CB", "UB", "n_het_sites_covered", "n_reads_in_family",
                       "span_bp", "het_positions_1based"],
                      res["umi_detail_rows"], mode_dist)
        write_csv(f"{args.out_dir}/{gate_name}",
                  ["section", "chrom", "pos_1based", "ref", "alt", "n_alt_reads", "n_ref_reads",
                   "n_other_reads", "ge2_alt_reads", "nearest_het_dist_bp", "het_in_range",
                   "read_level_coobs", "umi_level_coobs", "testable_P2", "n_distinct_alt_umi_families"],
                  res["gate_rows"], mode_dist)
        write_csv(f"{args.out_dir}/{altfam_name}",
                  ["section", "chrom", "pos_1based", "ref", "alt", "n_alt_reads", "n_distinct_alt_umi_families"],
                  res["alt_umi_rows"], mode_dist)

        summary_key = f"{section}_somatic" if qc == "somatic" else f"{section}_ambiguous"
        all_summaries[summary_key] = res["summary"]
        # write/refresh summary.json after every section so partial progress is never lost.
        # ALWAYS merge-on-write: load existing JSON (if any), update only this run's keys,
        # write back. This is INDEPENDENT of --fresh -- --fresh controls whether the CSVs for
        # *this invocation's query-class* are overwritten vs appended; it must never cause
        # summary.json (shared across every past and future invocation, any query-class) to
        # drop keys written by a DIFFERENT invocation.
        # (bug history 2026-08-24: v1 checked `not args.fresh` on every section -> reset on
        # every section, kept only the last. v2 fixed that to fire only on i==0 of a --fresh
        # run, but a --fresh ambiguous-class run at i==0 still wiped the somatic-class keys
        # from an EARLIER invocation of this same file, because the reset was still keyed to
        # this run's --fresh flag. v3 (this version): never reset from here; merge only.)
        summary_path = f"{args.out_dir}/summary.json"
        merged = {}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                merged = json.load(f)
        merged[summary_key] = res["summary"]
        with open(summary_path, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        log(f"=== {section} ({qc}) done === {json.dumps(res['summary'], default=str)}")

    log("ALL SECTIONS DONE")


if __name__ == "__main__":
    main()
