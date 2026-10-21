#!/usr/bin/env python3
"""
xmhc_artifact_mechanism.py
===========================

Tests whether SPARCAL's "somatic" (= spatially-clustered rank class -- NOT a
somatic-origin claim; see CLAUDE.md variant-category table) calls in the
extended MHC (chr6:28-34 Mb, "xMHC") are REFERENCE-DIVERGENCE ARTIFACTS
(hypothesis A: an individual's true HLA alleles differ substantially from the
reference, so reads misalign/mismatch and those mismatches get called as
variants) or GENUINE VARIANTS that happen to concentrate there (hypothesis B),
using read-level evidence from the BAM alone -- no HLA typing tool.

Five tests, run per section, xMHC vs a DEPTH-MATCHED non-xMHC control:
  1. Edit distance per ALT-supporting read (STARsolo `nM` tag -- see NOTE below;
     standard `NM`/`MD` tags are absent from these CellRanger/STARsolo BAMs).
  2. Variants-per-read: for each ALT-supporting read, how many OTHER
     class-listed positions (same section, same class, genome-wide) does that
     SAME PHYSICAL READ also carry the ALT allele for, using the read's true
     covered reference positions (CIGAR M-ops only, via get_aligned_pairs;
     spliced N-gaps are NOT treated as "covered"). THE DECISIVE PANEL.
  3. MAPQ of ALT-supporting reads.
  4. Soft-clip fraction and mean clipped length of ALT-supporting reads.
  5. Panel polymorphism: fraction of variant POSITIONS (position-only, any
     allele) that coincide with a 1000 Genomes panel record, and the AF of
     those that do.

CONTROL MATCHING: non-xMHC control sites are 1:1 nearest-neighbour matched to
xMHC sites on read depth (computed by an exact per-base pileup that correctly
excludes spliced N/D gaps -- NOT a naive BAM region-overlap count, which would
over-count spliced reads). Sampled from a depth-computed pool of up to
POOL_CAP non-xMHC sites (seeded, reproducible). Gene-expression matching was
NOT attempted (would need a per-gene UMI table cross-referenced to each
variant's host gene; out of scope here, noted as a limitation).

Read filtering: exclude unmapped/secondary/supplementary/QC-fail (flag 0x904);
duplicate-flag reads KEPT, matching the documented convention in
scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py (BAMs
are already umi_tools-deduplicated; the CellRanger dup flag reflects gene-level
marking, not redundancy within this file).

NON-DESTRUCTIVE: read-only on all BAMs, all shipped-pipeline outputs, all
frozen data/figs trees. Writes ONLY under --out-dir.

Run (env snv_caller):
    python scripts/postanalyze/xmhc_artifact_mechanism.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31 \
        --sections P4 P6 DCIS1 DCIS2 DLPFC_151507 DLPFC_151669
"""
from __future__ import annotations

import argparse
import bisect
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pysam

REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")

MHC_CHROM = "6"
MHC_LO, MHC_HI = 28_000_000, 34_000_000  # 1-based inclusive, matches cosmic_mhc_partition.py

FLAG_EXCLUDE = 0x904  # UNMAP + SECONDARY + SUPPLEMENTARY; DUP intentionally kept (see docstring)

POOL_MULT = 5      # non-xMHC candidate-pool size = min(POOL_MULT * n_xmhc, POOL_CAP, n_available)
POOL_CAP = 6000
MAX_ALT_READS_PER_SITE = 50  # cap per-site ALT-read examination (seeded subsample beyond this)
SEED = 20260831

PANELS = {
    # build -> (directory, filename template with {chrom} as a bare number)
    "hg19": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19",
        "hg19_chr{chrom}.vcf.gz",
    ),
    "GRCh38": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}"
        ".filtered.shapeit2-duohmm-phased.vcf.gz",
    ),
}

# Section configs -- BAM paths and class-VCF-adjacent txt paths copied read-only from
# scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py (P4/P6/DCIS1/DCIS2)
# and the DLPFC per-section tree (same schema, see CLAUDE.md "Output Directory Structure"
# and scripts/postanalyze/read_level_features.py for the DLPFC merged-BAM precedent).
SECTIONS = {
    "P4": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        somatic_txt=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=True,
        build="hg19",
    ),
    "P6": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        somatic_txt=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=True,
        build="hg19",
    ),
    "DCIS1": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.dedup.bam",
        somatic_txt=f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=False,
        build="GRCh38",
    ),
    "DCIS2": dict(
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.dedup.bam",
        somatic_txt=f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=False,
        build="GRCh38",
    ),
    "DLPFC_151507": dict(
        bam=f"{REPO}/data/dlpfc/151507/151507_merged.bam",
        somatic_txt=f"{REPO}/data/dlpfc/151507/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=False,
        build="GRCh38",
    ),
    "DLPFC_151669": dict(
        bam=f"{REPO}/data/dlpfc/151669/151669_merged.bam",
        somatic_txt=f"{REPO}/data/dlpfc/151669/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
        chr_prefix_in_bam=False,
        build="GRCh38",
    ),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- loading
def load_class_sites(somatic_txt):
    """Return list of dict(chrom, pos1, ref, alt) for the section's somatic
    (spatially-clustered) class, plus per-chrom sorted pos0 arrays + allele map
    for the genome-wide co-occurrence lookup."""
    rows = []
    with open(somatic_txt) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            chrom = parts[idx["chrom"]]
            pos1 = int(parts[idx["pos"]])
            ref = parts[idx["ref"]]
            alt = parts[idx["alt"]]
            rows.append(dict(chrom=chrom, pos1=pos1, ref=ref, alt=alt))
    by_chrom = defaultdict(list)
    allele_map = {}
    for r in rows:
        pos0 = r["pos1"] - 1
        by_chrom[r["chrom"]].append(pos0)
        allele_map[(r["chrom"], pos0)] = (r["ref"], r["alt"])
    sorted_by_chrom = {c: sorted(v) for c, v in by_chrom.items()}
    return rows, sorted_by_chrom, allele_map


def split_xmhc(rows):
    xmhc, other = [], []
    for r in rows:
        if r["chrom"] == MHC_CHROM and MHC_LO <= r["pos1"] <= MHC_HI:
            xmhc.append(r)
        else:
            other.append(r)
    return xmhc, other


# --------------------------------------------------------------------------- BAM site inspection
def inspect_site(bam, bam_chrom, pos1, ref, alt, rng):
    """One exact per-base pileup at (bam_chrom, pos1). Returns
    (depth, n_ref, alt_reads) where alt_reads is a (possibly subsampled) list
    of pysam.AlignedSegment carrying the ALT base at this exact position
    (is_del/is_refskip pileup entries -- i.e. spliced-over or deleted bases --
    are excluded from depth entirely, not just from ALT support)."""
    pos0 = pos1 - 1
    depth = 0
    n_ref = 0
    alt_reads = []
    ref_up = ref.upper()
    alt_up = alt.upper()
    try:
        cols = bam.pileup(bam_chrom, pos0, pos0 + 1, truncate=True,
                           min_base_quality=0, flag_filter=FLAG_EXCLUDE,
                           max_depth=200_000, ignore_overlaps=False)
        for col in cols:
            for pu in col.pileups:
                if pu.is_del or pu.is_refskip:
                    continue
                qpos = pu.query_position
                if qpos is None:
                    continue
                aln = pu.alignment
                seq = aln.query_sequence
                if not seq:
                    continue
                depth += 1
                base = seq[qpos].upper()
                if base == alt_up:
                    alt_reads.append(aln)
                elif base == ref_up:
                    n_ref += 1
    except (ValueError, OSError) as e:
        log(f"  pileup failed at {bam_chrom}:{pos1}: {e}")
        return 0, 0, []
    if len(alt_reads) > MAX_ALT_READS_PER_SITE:
        alt_reads = rng.sample(alt_reads, MAX_ALT_READS_PER_SITE)
    return depth, n_ref, alt_reads


def read_qc_features(aln):
    """nM (STARsolo edit-distance tag; NM/MD absent from these BAMs -- see
    module docstring), MAPQ, soft-clip total bases, soft-clip boolean."""
    nm = None
    if aln.has_tag("nM"):
        nm = aln.get_tag("nM")
    elif aln.has_tag("NM"):
        nm = aln.get_tag("NM")
    mapq = aln.mapping_quality
    cigar = aln.cigartuples or []
    soft_clip = sum(length for op, length in cigar if op == 4)  # BAM_CSOFT_CLIP
    read_len = len(aln.query_sequence) if aln.query_sequence else sum(
        length for op, length in cigar if op in (0, 1, 4, 7, 8))
    return nm, mapq, soft_clip, read_len


def count_cooccurring_alt(aln, chrom, seed_pos0, sorted_by_chrom, allele_map):
    """How many OTHER class-listed positions (this section, genome-wide) does
    this read's TRUE covered reference span (CIGAR-M aligned pairs only, no
    spliced N-gaps) carry the ALT allele for."""
    arr = sorted_by_chrom.get(chrom)
    if not arr:
        return 0, 0
    lo = bisect.bisect_left(arr, aln.reference_start)
    hi = bisect.bisect_right(arr, aln.reference_end - 1 if aln.reference_end else aln.reference_start)
    candidates = arr[lo:hi]
    if len(candidates) <= 1:
        return 0, 0
    pairs = aln.get_aligned_pairs(matches_only=True)  # [(qpos, rpos), ...]
    rpos_to_qpos = {rpos: qpos for qpos, rpos in pairs}
    seq = aln.query_sequence
    if not seq:
        return 0, 0
    n_alt_other = 0
    n_covered_other = 0
    for cand_pos0 in candidates:
        if cand_pos0 == seed_pos0:
            continue
        qpos = rpos_to_qpos.get(cand_pos0)
        if qpos is None:
            continue
        n_covered_other += 1
        cand_ref, cand_alt = allele_map[(chrom, cand_pos0)]
        base = seq[qpos].upper()
        if base == cand_alt.upper():
            n_alt_other += 1
    return n_alt_other, n_covered_other


# --------------------------------------------------------------------------- matching
def match_by_depth(xmhc_depths, pool_depths):
    """1:1 nearest-neighbour greedy matching without replacement.
    xmhc_depths, pool_depths: list of (site_idx, depth). Returns list of
    (xmhc_idx, pool_idx) pairs, processed in ASCENDING xmhc-depth order."""
    pool_sorted = sorted(pool_depths, key=lambda t: t[1])
    pool_vals = [d for _, d in pool_sorted]
    pool_idx = [i for i, _ in pool_sorted]
    pairs = []
    for xi, xd in sorted(xmhc_depths, key=lambda t: t[1]):
        if not pool_vals:
            break
        j = bisect.bisect_left(pool_vals, xd)
        # compare neighbours at j-1 and j for nearest
        best = None
        for cand in (j - 1, j):
            if 0 <= cand < len(pool_vals):
                if best is None or abs(pool_vals[cand] - xd) < abs(pool_vals[best] - xd):
                    best = cand
        if best is None:
            break
        pairs.append((xi, pool_idx[best]))
        del pool_vals[best]
        del pool_idx[best]
    return pairs


# --------------------------------------------------------------------------- panel polymorphism
_panel_cache = {}


def get_panel(build):
    if build not in _panel_cache:
        directory, template = PANELS[build]
        _panel_cache[build] = {}
    return _panel_cache[build]


def panel_chrom_file(build, chrom):
    directory, template = PANELS[build]
    return f"{directory}/{template.format(chrom=chrom)}"


def query_panel(build, chrom, pos1, vf_cache):
    """Position-only (any allele) 1000 Genomes panel lookup. Returns
    (is_polymorphic: bool, best_af: float|None) -- best_af is the AF of an
    exact REF/ALT-agnostic position match (first record found; if multiple,
    the max AF)."""
    key = (build, chrom)
    vf = vf_cache.get(key)
    if vf is None:
        path = panel_chrom_file(build, chrom)
        try:
            vf = pysam.VariantFile(path)
        except (ValueError, OSError):
            vf = False
        vf_cache[key] = vf
    if vf is False:
        return None, None
    pos0 = pos1 - 1
    hit = False
    best_af = None
    try:
        for rec in vf.fetch(f"chr{chrom}", pos0, pos1):
            if rec.pos == pos1:
                hit = True
                af = rec.info.get("AF")
                if af is not None:
                    if isinstance(af, (tuple, list)):
                        af = af[0]
                    if best_af is None or af > best_af:
                        best_af = af
    except (ValueError, OSError):
        return None, None
    return hit, best_af


# --------------------------------------------------------------------------- per-section driver
def run_section(sec_name, cfg, out_dir, rng):
    log(f"=== {sec_name} ===")
    rows, sorted_by_chrom, allele_map = load_class_sites(cfg["somatic_txt"])
    xmhc, other = split_xmhc(rows)
    n_total = len(rows)
    log(f"{sec_name}: {n_total} total class sites, {len(xmhc)} in xMHC, {len(other)} outside")

    if not xmhc:
        log(f"{sec_name}: no xMHC sites, skipping")
        return None

    bam = pysam.AlignmentFile(cfg["bam"], "rb")
    prefix = "chr" if cfg["chr_prefix_in_bam"] else ""

    def bam_chrom(c):
        return f"{prefix}{c}"

    # ---- depth for ALL xmhc sites -------------------------------------------------
    xmhc_info = []
    for r in xmhc:
        depth, n_ref, alt_reads = inspect_site(bam, bam_chrom(r["chrom"]), r["pos1"], r["ref"], r["alt"], rng)
        xmhc_info.append(dict(r, depth=depth, n_ref=n_ref, alt_reads=alt_reads, group="xMHC"))
    log(f"{sec_name}: computed depth for {len(xmhc_info)} xMHC sites")

    # ---- depth for a seeded pool of non-xMHC candidate sites -----------------------
    pool_size = min(POOL_MULT * len(xmhc), POOL_CAP, len(other))
    pool_sites = rng.sample(other, pool_size) if pool_size < len(other) else list(other)
    pool_info = []
    for r in pool_sites:
        depth, n_ref, alt_reads = inspect_site(bam, bam_chrom(r["chrom"]), r["pos1"], r["ref"], r["alt"], rng)
        pool_info.append(dict(r, depth=depth, n_ref=n_ref, alt_reads=alt_reads, group="control_pool"))
    log(f"{sec_name}: computed depth for {len(pool_info)} non-xMHC pool sites")

    # ---- 1:1 depth matching (require >=1 alt read on both sides to be usable) ------
    xmhc_usable = [(i, s["depth"]) for i, s in enumerate(xmhc_info) if s["alt_reads"]]
    pool_usable = [(i, s["depth"]) for i, s in enumerate(pool_info) if s["alt_reads"]]
    pairs = match_by_depth(xmhc_usable, pool_usable)
    log(f"{sec_name}: {len(xmhc_usable)}/{len(xmhc_info)} xMHC sites usable (>=1 ALT read), "
        f"{len(pool_usable)}/{len(pool_info)} pool sites usable, matched {len(pairs)} pairs")

    matched_xmhc = [xmhc_info[i] for i, _ in pairs]
    matched_control = [pool_info[j] for _, j in pairs]

    # ---- per-read feature rows + co-occurrence -------------------------------------
    read_rows = []
    for group_name, sites in (("xMHC", matched_xmhc), ("matched_control", matched_control)):
        for s in sites:
            chrom = s["chrom"]
            seed_pos0 = s["pos1"] - 1
            for aln in s["alt_reads"]:
                nm, mapq, soft_clip, read_len = read_qc_features(aln)
                n_alt_other, n_covered_other = count_cooccurring_alt(
                    aln, chrom, seed_pos0, sorted_by_chrom, allele_map)
                read_rows.append(dict(
                    section=sec_name, group=group_name, chrom=chrom, pos=s["pos1"],
                    ref=s["ref"], alt=s["alt"], site_depth=s["depth"],
                    read_name=aln.query_name, nm=nm, mapq=mapq,
                    soft_clip_bases=soft_clip, read_len=read_len,
                    soft_clip_frac=soft_clip / read_len if read_len else None,
                    n_cooccurring_other_positions_covered=n_covered_other,
                    n_cooccurring_other_alt=n_alt_other,
                ))
    read_df = pd.DataFrame(read_rows)
    log(f"{sec_name}: {len(read_df)} (site,read) rows "
        f"({(read_df.group=='xMHC').sum()} xMHC, {(read_df.group=='matched_control').sum()} control)")

    # ---- site-level summary ---------------------------------------------------------
    site_rows = []
    for group_name, sites in (("xMHC", matched_xmhc), ("matched_control", matched_control)):
        for s in sites:
            site_rows.append(dict(
                section=sec_name, group=group_name, chrom=s["chrom"], pos=s["pos1"],
                ref=s["ref"], alt=s["alt"], depth=s["depth"], n_ref=s["n_ref"],
                n_alt_reads_examined=len(s["alt_reads"])))
    site_df = pd.DataFrame(site_rows)

    # ---- panel polymorphism (position-only, matched sites) --------------------------
    vf_cache = {}
    panel_rows = []
    for group_name, sites in (("xMHC", matched_xmhc), ("matched_control", matched_control)):
        for s in sites:
            hit, af = query_panel(cfg["build"], s["chrom"], s["pos1"], vf_cache)
            panel_rows.append(dict(
                section=sec_name, group=group_name, chrom=s["chrom"], pos=s["pos1"],
                ref=s["ref"], alt=s["alt"], depth=s["depth"],
                in_1kgp_panel=hit, panel_af=af))
    panel_df = pd.DataFrame(panel_rows)
    log(f"{sec_name}: panel query done, "
        f"{panel_df[panel_df.group=='xMHC'].in_1kgp_panel.mean():.3f} xMHC hit-rate, "
        f"{panel_df[panel_df.group=='matched_control'].in_1kgp_panel.mean():.3f} control hit-rate")

    bam.close()
    return dict(read_df=read_df, site_df=site_df, panel_df=panel_df,
                n_total=n_total, n_xmhc=len(xmhc), n_other=len(other),
                n_matched_pairs=len(pairs))


def summarize(all_results, out_dir):
    """Build the cross-section summary tables reported in RESULTS.md."""
    read_all = pd.concat([r["read_df"] for r in all_results.values() if r], ignore_index=True)
    site_all = pd.concat([r["site_df"] for r in all_results.values() if r], ignore_index=True)
    panel_all = pd.concat([r["panel_df"] for r in all_results.values() if r], ignore_index=True)

    read_all.to_csv(out_dir / "01_read_level_features.csv", index=False)
    site_all.to_csv(out_dir / "02_site_depth_matching.csv", index=False)
    panel_all.to_csv(out_dir / "05_panel_polymorphism.csv", index=False)

    # per-section, per-group summary of the four read-level panels
    rows = []
    for (sec, grp), g in read_all.groupby(["section", "group"]):
        nm = g["nm"].dropna()
        mapq = g["mapq"].dropna()
        sc_frac = g["soft_clip_frac"].dropna()
        cooc = g["n_cooccurring_other_alt"].dropna()
        rows.append(dict(
            section=sec, group=grp, n_reads=len(g),
            n_sites=g[["chrom", "pos"]].drop_duplicates().shape[0],
            mean_nm=nm.mean(), median_nm=nm.median(),
            mean_mapq=mapq.mean(), median_mapq=mapq.median(),
            frac_mapq_lt30=(mapq < 30).mean() if len(mapq) else float("nan"),
            mean_soft_clip_frac=sc_frac.mean(),
            frac_any_soft_clip=(g["soft_clip_bases"] > 0).mean(),
            mean_cooccurring_other_alt=cooc.mean(),
            median_cooccurring_other_alt=cooc.median(),
            frac_reads_with_ge1_cooccurring_alt=(cooc >= 1).mean() if len(cooc) else float("nan"),
        ))
    summary_df = pd.DataFrame(rows).sort_values(["section", "group"])
    summary_df.to_csv(out_dir / "00_summary_by_section_group.csv", index=False)

    # panel polymorphism summary
    panel_rows = []
    for (sec, grp), g in panel_all.groupby(["section", "group"]):
        hits = g["in_1kgp_panel"].dropna()
        af = g.loc[g["in_1kgp_panel"] == True, "panel_af"].dropna()
        panel_rows.append(dict(
            section=sec, group=grp, n_sites=len(g),
            frac_in_1kgp_panel=hits.mean() if len(hits) else float("nan"),
            n_af_observed=len(af), mean_af_of_hits=af.mean() if len(af) else float("nan"),
            median_af_of_hits=af.median() if len(af) else float("nan"),
        ))
    panel_summary_df = pd.DataFrame(panel_rows).sort_values(["section", "group"])
    panel_summary_df.to_csv(out_dir / "05b_panel_polymorphism_summary.csv", index=False)

    return summary_df, panel_summary_df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS.keys()))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for sec_name in args.sections:
        if sec_name not in SECTIONS:
            log(f"unknown section {sec_name}, skipping")
            continue
        rng = random.Random(hash((SEED, sec_name)) & 0xFFFFFFFF)
        cfg = SECTIONS[sec_name]
        all_results[sec_name] = run_section(sec_name, cfg, out_dir, rng)

    summary_df, panel_summary_df = summarize(all_results, out_dir)
    print("\n=== 00_summary_by_section_group.csv ===")
    print(summary_df.to_string(index=False))
    print("\n=== 05b_panel_polymorphism_summary.csv ===")
    print(panel_summary_df.to_string(index=False))
    print(f"\nWROTE outputs under {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
