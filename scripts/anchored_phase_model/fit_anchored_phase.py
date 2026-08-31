#!/usr/bin/env python
"""
fit_anchored_phase.py
======================

THE DECISIVE EXPERIMENT for SPARCAL's phasing rework: can a model predict the
RELATIVE haplotype phase (same vs opposite) between a non-panel candidate variant
v and a nearby phased germline het h, WITHOUT using read-backed linkage as a
feature? If yes, the ~92-98% of somatic candidates that are not read-backed
testable (see data/readbacked_feasibility_2026-08-23/RESULTS.md gate, section 4)
could in principle be phased by the model instead of by direct molecule
co-observation. If no, phasing stays bounded to the small read-backed-observable
subset.

NON-DESTRUCTIVE. Never edits / re-runs scripts/1_calling .. scripts/6_spatial_filter,
never writes into any data/<sample>/output_VCFs/ tree, never writes into any of the
frozen dirs listed in the task (figs/v5_2026-08-23, figs/v6_2026-08-24,
data/sparcalnet_eval_2026-08-23, data/somatic_calibration_2026-08-23,
data/readbacked_feasibility_2026-08-23, data/sidecar_phasing_probe_2026-08-23,
data_sidecar_phased/, scripts/sidecar_phasing/, scripts/sidecar_phasing_feasibility/).
Those are read from (BAM paths, het definitions, the somatic gate) but nothing is
written back into them. All output goes to data/anchored_phase_model_2026-08-24/.

GENOME BUILDS. P4/P6 = GRCh37/hg19, chr-prefixed BAM contigs. DCIS1/DCIS2 =
GRCh38, bare BAM contigs. Section config (BAM paths, chrom-naming, het/somatic
source files) is reused UNMODIFIED (imported, not re-typed) from
scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py so label
extraction uses exactly the same het definitions that produced testable_P2.

IMPORTANT BUILD-SAFETY FINDING (verified directly, not assumed): every CalicoST
run for P4 and P6 on disk (P4_sec1, P4_sec2, P4_all, P4_12, P4_r1_hg38_fei,
P6_sec1, P6_sec2, P6_r1_hg38_fei -- all of them, checked by grepping each run's
configuration_cna* for geneticmap_file) was configured against GRCh38_resources,
never hg19_resources -- i.e. there is no hg19 CalicoST output for P4/P6 anywhere
on disk, while the task's own build table fixes P4/P6 SNV calls in GRCh37/hg19.
Using those CalicoST segment coordinates against hg19 v/h positions would silently
misalign CNV/LOH state by tens of megabases. This mirrors a documented, already
-established precedent in this exact codebase: scripts/postanalyze/loh_allelic_test.py
refuses P4/P6 for the identical reason with copyKAT ("BUILD RESTRICTION ... Refuses
other samples"). This script follows the same precedent: CalicoST CNV/LOH features
are computed ONLY for DCIS1/DCIS2 (native GRCh38, matching their own SNV build);
for P4/P6 those columns are left NaN with calicost_build_mismatch=True, never
silently populated from mismatched coordinates.

STAGE 1 (labels, needs BAM re-scan -- the expensive step): for every testable_P2
candidate in the somatic gate, re-scan the same dedup BAM in a narrow window
around v and its nearby het(s), reconstruct per-molecule evidence of which ALLELE
of h travels with v's ALT allele, and write labels_relative_phase.csv. Two
evidence tiers per molecule, most-reliable-first:
  read_level      -- a SINGLE READ directly spans both v and h (unambiguous,
                      one molecule, one fragment, no aggregation needed).
  umi_level_only  -- no single read spans both, but a UMI family (CB,UB) has an
                      ALT-supporting read at v AND a (different) read covering h,
                      and ALL of that family's h-covering reads agree on one base
                      (family used only when internally consistent -- if a family
                      shows conflicting bases at h, or conflicting ALT/REF calls
                      at v that make attribution ambiguous, it contributes no vote
                      rather than guessing).
This was validated by hand on a real testable candidate (P4 chr1:3702430 A>T,
het at chr1:3702425) before being scaled to all 4,017 -- see RESULTS.md "Stage 1
methodology / worked example" for the full trace.

STAGE 2 (features, no BAM re-scan needed): builds features.csv from sources that
do NOT touch the specific v-h co-observation events used to build the label --
spatial ASE coherence (per-spot ALT-fraction correlation, from
output_VCFs/spotprofiles/<qf>/vcf_by_spot/), CalicoST CNV/LOH + clone info (DCIS
only, see build note above), genomic distance + local het density (pure position
arithmetic), per-spot CO-DETECTION rate (binary presence/absence matrices, NOT
co-observation), and depth/BAF for v (from the already-computed somatic gate) and
h (from the pseudobulk mpileup+call VCF's I16, an aggregate pileup statistic).
See RESULTS.md "Leakage audit" for the precise, argued definition of what counts
as leakage here and why each feature does or doesn't cross that line.

STAGE 3 (fit): logistic regression + HistGradientBoostingClassifier (NOT an MLP --
~4,000 rows / ~15 features do not justify a neural net, and SparcalNet's own
6-feature MLP evaluation on this project (data/sparcalnet_eval_2026-08-23/)
already showed that failure mode). StratifiedGroupKFold(n_splits=5, group=
candidate_id) so a candidate can never span folds. Baselines: fixed 50% chance
and DummyClassifier(strategy='stratified'). Reports AUROC/accuracy/Brier per
fold (never a bare mean), then slices the out-of-fold predictions by depth,
distance-to-anchor, LOH-vs-copy-neutral, and section -- the covariate-shift
question (does performance hold where the untestable 92% actually live) is
answered from that slicing plus a direct testable-vs-untestable covariate
comparison pulled from the full somatic gate (covariate_shift_summary.csv).

Usage:
  python fit_anchored_phase.py --stage all
  python fit_anchored_phase.py --stage labels --sections P4 --limit 50   # quick test
  python fit_anchored_phase.py --stage labels                            # full, all 4 sections
  python fit_anchored_phase.py --stage features
  python fit_anchored_phase.py --stage fit
"""
from __future__ import annotations

import argparse
import bisect
import csv
import os
import pickle
import sys
import time
import traceback
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pysam
from scipy import stats as sstats

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
sys.path.insert(0, os.path.join(REPO, "scripts/sidecar_phasing_feasibility"))
# Reused UNMODIFIED from the frozen, read-only feasibility script -- guarantees the
# exact same het definitions / BAM paths / chrom-naming handling that produced
# testable_P2 in 04_somatic_gate.csv. This script only ever imports from it.
from measure_readbacked_feasibility import SECTIONS, load_het_sites, FLAG_EXCLUDE  # noqa: E402

OUT_DIR = os.path.join(REPO, "data/anchored_phase_model_2026-08-24")
GATE_CSV = os.path.join(REPO, "data/readbacked_feasibility_2026-08-23/04_somatic_gate.csv")
SECTION_LIST = ["P4", "P6", "DCIS1", "DCIS2"]

RADIUS_STEPS = [5000, 20000]   # expanding het-search radius around v, bp
PAD = 250                      # fetch-window padding beyond the farthest site of interest, bp
CONFIDENT_MIN_N = 2
CONFIDENT_MIN_FRAC = 0.8

CALICOST_ROOT = "/data/maiziezhou_lab/leiy4/CalicoST"
SECTION_PATHS = {
    "P4": dict(
        matrix_dir=f"{REPO}/data/P4_tumor/1/matrix", matrix_prefix="P4_TUMOR_1_SPARCAL",
        vcf_by_spot_dir=f"{REPO}/data/P4_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot",
        calicost_dir=f"{CALICOST_ROOT}/P4_sec1/calicost/clone3_rectangle0_w1.0",
        calicost_build_ok=False,   # verified GRCh38 CalicoST vs GRCh37/hg19 SNV calls -- see module docstring
    ),
    "P6": dict(
        matrix_dir=f"{REPO}/data/P6_tumor/1/matrix", matrix_prefix="P6_TUMOR_1_SPARCAL",
        vcf_by_spot_dir=f"{REPO}/data/P6_tumor/1/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot",
        calicost_dir=f"{CALICOST_ROOT}/P6_sec1/calicost/clone3_rectangle0_w1.0",
        calicost_build_ok=False,
    ),
    "DCIS1": dict(
        matrix_dir=f"{REPO}/data/dcis1/matrix", matrix_prefix="DCIS_dcis1_SPARCAL",
        vcf_by_spot_dir=f"{REPO}/data/dcis1/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot",
        calicost_dir=f"{CALICOST_ROOT}/DCIS1/calicost/clone3_rectangle0_w1.0",
        calicost_build_ok=True,    # native GRCh38, matches DCIS1 SNV build
    ),
    "DCIS2": dict(
        matrix_dir=f"{REPO}/data/dcis2/matrix", matrix_prefix="DCIS_dcis2_SPARCAL",
        vcf_by_spot_dir=f"{REPO}/data/dcis2/output_VCFs/spotprofiles/baseQ0mapQ0/vcf_by_spot",
        calicost_dir=f"{CALICOST_ROOT}/DCIS2/calicost/clone3_rectangle0_w1.0",
        calicost_build_ok=True,
    ),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# =============================================================================
# STAGE 1 -- extract relative-phase labels from the BAM (molecule-level, tiered)
# =============================================================================
def load_testable_candidates(section):
    rows = []
    with open(GATE_CSV) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["section"] != section or row["testable_P2"] != "True":
                continue
            rows.append(dict(
                chrom=row["chrom"], pos_1based=int(row["pos_1based"]),
                ref=row["ref"], alt=row["alt"],
                n_alt_reads=int(row["n_alt_reads"]), n_ref_reads=int(row["n_ref_reads"]),
                n_other_reads=int(row["n_other_reads"]),
                gate_nearest_het_dist_bp=(int(row["nearest_het_dist_bp"]) if row["nearest_het_dist_bp"] else None),
                n_distinct_alt_umi_families=int(row["n_distinct_alt_umi_families"]),
            ))
    return rows


def find_near_positions(sorted_pos, pos0, radius):
    lo = bisect.bisect_left(sorted_pos, pos0 - radius)
    hi = bisect.bisect_right(sorted_pos, pos0 + radius)
    return sorted_pos[lo:hi]


def scan_one_candidate(bf, bam_contig, chrom_bare, pos0_v, ref_v, alt_v, het_sorted, het_allele):
    """Tiered, per-molecule phase-vote scan for one candidate against every het found
    nearby (expanding radius). Returns (radius_used_or_None, votes, tiers, barcodes,
    mids_used), all dicts keyed by het pos0. barcodes[h] is the set of spot barcodes (CB)
    whose reads contributed a same/opposite vote for that het; mids_used[h] is the set of
    exact molecule keys (("umi",cb,ub) or ("read",query_name)) that did. Both are needed so
    Stage 2 can exclude this EXACT evidence -- spots for spatial-ASE/co-detection, molecules
    for the h pileup depth/BAF recompute -- otherwise those features leak (see RESULTS.md
    leakage audit: this was caught empirically, not anticipated up front, at AUROC ~0.99)."""
    alt_v = alt_v.upper()
    for radius in RADIUS_STEPS:
        near = find_near_positions(het_sorted, pos0_v, radius)
        if not near:
            continue
        lo = max(0, min([pos0_v] + near) - PAD)
        hi = max([pos0_v] + near) + PAD + 1

        # Per molecule (CB,UB UMI family, or a singleton per untagged read), for each
        # nearby het h we keep THREE separate base-pools, gated on what that SAME READ
        # showed at v (not on the molecule as a whole -- a family can be a UMI collision
        # of >1 original fragment, so pooling across the whole family before checking
        # what each read showed at v silently conflates ALT-side and REF-side evidence;
        # this was a real bug caught by hand-validating P4 chr1:3702430, see RESULTS.md):
        #   h_from_alt[h]   bases at h from reads whose OWN base at v == alt_v
        #                   (these reads directly span both v and h -- Tier 1 pool)
        #   h_from_unseen[h] bases at h from reads that do NOT cover v at all
        #                   (extends reach via the family -- Tier-2-only contribution)
        # Reads whose own base at v is REF (or a third allele) are EXCLUDED from both
        # pools for this het -- they are direct evidence for a DIFFERENT read/molecule
        # state and must not contaminate the ALT-side pool.
        molecules = defaultdict(lambda: {"has_alt": False, "cb": None,
                                          "h_from_alt": defaultdict(Counter),
                                          "h_from_unseen": defaultdict(Counter)})
        for read in bf.fetch(bam_contig, lo, hi):
            if read.flag & FLAG_EXCLUDE:
                continue
            rs, re_ = read.reference_start, read.reference_end
            if rs is None or re_ is None:
                continue
            v_in_span = rs <= pos0_v < re_
            h_in_span = [h for h in near if rs <= h < re_]
            if not v_in_span and not h_in_span:
                continue
            seq = read.query_sequence
            if seq is None:
                continue
            aligned = {rp: qp for qp, rp in read.get_aligned_pairs(matches_only=True)}
            v_base = seq[aligned[pos0_v]].upper() if (v_in_span and pos0_v in aligned) else None
            h_here = {}
            for h in h_in_span:
                if h in aligned:
                    h_here[h] = seq[aligned[h]].upper()
            if v_base is None and not h_here:
                continue
            try:
                cb = read.get_tag("CB")
            except KeyError:
                cb = None
            try:
                ub = read.get_tag("UB")
            except KeyError:
                ub = None
            if cb and ub:
                mid = ("umi", cb, ub)
            else:
                # keyed by query_name (stable/reproducible), not an arbitrary counter, so
                # Stage 2 can re-identify and exclude this EXACT read later when recomputing
                # depth_h/baf_h with the label-source evidence held out (see leakage audit:
                # baf_h from the pooled pseudobulk VCF leaks the same way spatial_ase_corr
                # did, via the same shared-reads mechanism, when depth_h is small).
                mid = ("read", read.query_name)
            rec = molecules[mid]
            if cb is not None:
                rec["cb"] = cb
            if v_base == alt_v:
                rec["has_alt"] = True
                for h, hb in h_here.items():
                    rec["h_from_alt"][h][hb] += 1
            elif v_base is None:
                for h, hb in h_here.items():
                    rec["h_from_unseen"][h][hb] += 1
            # v_base == ref_v (or a third allele): this read's h-observation is dropped,
            # not pooled -- see rationale above.

        if not molecules:
            continue

        votes = {h: Counter() for h in near}
        tiers = {h: Counter() for h in near}
        barcodes = {h: set() for h in near}
        mids_used = {h: set() for h in near}
        for mid, rec in molecules.items():
            if not rec["has_alt"]:
                continue  # this molecule shows no ALT-at-v read at all -- irrelevant to the phase label
            for h in near:
                ref_h, alt_h = het_allele[(chrom_bare, h)]
                ref_h, alt_h = ref_h.upper(), alt_h.upper()
                call, tier = None, None
                direct = rec["h_from_alt"].get(h)
                if direct and len(direct) == 1:
                    call, tier = next(iter(direct)), "read_level"
                if call is None:
                    combined = Counter()
                    combined.update(rec["h_from_alt"].get(h, {}))
                    combined.update(rec["h_from_unseen"].get(h, {}))
                    if combined and len(combined) == 1:
                        call, tier = next(iter(combined)), "umi_level_only"
                if call is None:
                    continue
                if call == alt_h:
                    votes[h]["same"] += 1
                elif call == ref_h:
                    votes[h]["opposite"] += 1
                else:
                    votes[h]["other"] += 1
                tiers[h][tier] += 1
                if rec["cb"] is not None:
                    barcodes[h].add(rec["cb"])
                mids_used[h].add(mid)

        if any((v["same"] + v["opposite"]) > 0 for v in votes.values()):
            return radius, votes, tiers, barcodes, mids_used
    return None, {}, {}, {}, {}


def stage1_extract_labels(sections, limit=None):
    os.makedirs(OUT_DIR, exist_ok=True)
    all_rows, fail_rows = [], []
    for section in sections:
        cfg = SECTIONS[section]
        log(f"[labels] {section}: loading het sites (reused from measure_readbacked_feasibility.load_het_sites)")
        het_by_chrom, het_allele, het_stats = load_het_sites(cfg)
        candidates = load_testable_candidates(section)
        if limit:
            candidates = candidates[:limit]
        log(f"[labels] {section}: {len(candidates)} testable_P2 candidates to scan; het_stats={het_stats}")
        bf = pysam.AlignmentFile(cfg["bam"], "rb")
        t0 = time.time()
        n_labeled_section = 0
        for i, c in enumerate(candidates):
            chrom = c["chrom"]
            pos0_v = c["pos_1based"] - 1
            bam_contig = f"chr{chrom}" if cfg["chr_prefix_in_bam"] else chrom
            het_sorted = het_by_chrom.get(chrom, [])
            radius, votes, tiers, barcodes, mids_used = scan_one_candidate(
                bf, bam_contig, chrom, pos0_v, c["ref"], c["alt"], het_sorted, het_allele)
            candidate_id = f"{section}:{chrom}:{c['pos_1based']}:{c['ref']}>{c['alt']}"
            if radius is None:
                fail_rows.append(dict(candidate_id=candidate_id, section=section, chrom=chrom,
                                       pos_1based=c["pos_1based"], ref=c["ref"], alt=c["alt"],
                                       gate_nearest_het_dist_bp=c["gate_nearest_het_dist_bp"],
                                       n_alt_reads=c["n_alt_reads"], n_ref_reads=c["n_ref_reads"]))
                continue
            supported = [h for h, v in votes.items() if (v["same"] + v["opposite"]) > 0]
            supported.sort(key=lambda h: abs(h - pos0_v))
            anchor = supported[0]
            v = votes[anchor]
            n_same, n_opposite, n_other = v["same"], v["opposite"], v["other"]
            n_total = n_same + n_opposite
            if n_same > n_opposite:
                label = "same"
            elif n_opposite > n_same:
                label = "opposite"
            else:
                label = "tie"
            majority_frac = (max(n_same, n_opposite) / n_total) if n_total else None
            pval = (sstats.binomtest(max(n_same, n_opposite), n_total, 0.5, alternative="two-sided").pvalue
                    if n_total > 0 else None)
            confident = bool(label != "tie" and n_total >= CONFIDENT_MIN_N
                              and majority_frac is not None and majority_frac >= CONFIDENT_MIN_FRAC)
            ref_h, alt_h = het_allele[(chrom, anchor)]
            t = tiers[anchor]
            bcs = sorted(barcodes[anchor])
            mid_strs = sorted(
                (f"umi:{m[1]}:{m[2]}" if m[0] == "umi" else f"read:{m[1]}") for m in mids_used[anchor])
            all_rows.append(dict(
                candidate_id=candidate_id, section=section, chrom=chrom,
                pos_1based=c["pos_1based"], ref=c["ref"], alt=c["alt"],
                n_alt_reads=c["n_alt_reads"], n_ref_reads=c["n_ref_reads"],
                anchor_pos_1based=anchor + 1, anchor_ref=ref_h, anchor_alt=alt_h,
                dist_to_anchor_bp=abs(anchor - pos0_v),
                gate_nearest_het_dist_bp=c["gate_nearest_het_dist_bp"],
                anchor_matches_nearest_het=bool(c["gate_nearest_het_dist_bp"] is not None
                                                 and abs(anchor - pos0_v) == c["gate_nearest_het_dist_bp"]),
                n_same=n_same, n_opposite=n_opposite, n_other=n_other, n_total=n_total,
                n_tier_read_level=t.get("read_level", 0), n_tier_umi_level_only=t.get("umi_level_only", 0),
                n_hets_with_any_support=len(supported),
                label=label, majority_frac=majority_frac, binom_pvalue=pval,
                confident=confident, radius_used_bp=radius,
                n_label_source_spots=len(bcs), label_source_barcodes=";".join(bcs),
                label_source_molecule_keys=";".join(mid_strs),
            ))
            n_labeled_section += 1
            if (i + 1) % 200 == 0:
                log(f"[labels] {section}: {i + 1}/{len(candidates)} scanned ({time.time() - t0:.1f}s elapsed)")
        bf.close()
        log(f"[labels] {section}: DONE -- {n_labeled_section} labeled, "
            f"{len(candidates) - n_labeled_section} failed-to-reproduce-evidence, {time.time() - t0:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT_DIR, "labels_relative_phase.csv"), index=False)
    if fail_rows:
        pd.DataFrame(fail_rows).to_csv(os.path.join(OUT_DIR, "label_extraction_failures.csv"), index=False)
    else:
        # still write an empty-but-headered file so downstream code / a reader never has to guess
        pd.DataFrame(columns=["candidate_id", "section", "chrom", "pos_1based", "ref", "alt",
                               "gate_nearest_het_dist_bp", "n_alt_reads", "n_ref_reads"]
                      ).to_csv(os.path.join(OUT_DIR, "label_extraction_failures.csv"), index=False)
    log(f"[labels] TOTAL: {len(df)} labeled, {len(fail_rows)} failures, "
        f"written to {os.path.join(OUT_DIR, 'labels_relative_phase.csv')}")
    return df


# =============================================================================
# STAGE 2 -- build features that EXCLUDE read-backed linkage
# =============================================================================
def load_pseudobulk_depth_baf(section, positions_needed):
    """positions_needed: set of (chrom_bare, pos0). Returns {(chrom,pos0): (dp, baf)}.
    Source: the pseudobulk samtools-mpileup+call VCF (merged_sorted_gt.vcf.gz).
    NAIVE / LEAKY: this pools reads across the WHOLE section, but it is the SAME dedup
    BAM Stage 1 scans for co-observation, and h's total depth is often small (median 10
    among confident-subset candidates) -- so the label-generating molecule(s) can be a
    large share of this pool. Confirmed empirically: point-biserial r(baf_h, label) = 0.64
    at depth_h<=10, falling to 0.17 (n.s.) at depth_h>100 -- a leak, not a real effect
    (a real ASE/haplotype-expression effect would not fade with pooled depth like that;
    it would strengthen with depth as noise shrinks). See recompute_h_baf_excl_label()
    for the leakage-safe replacement, and RESULTS.md's leakage audit for the full trace."""
    cfg = SECTIONS[section]
    out = {}
    vf = pysam.VariantFile(cfg["raw_gt_vcf"])
    for chrom, pos0 in positions_needed:
        contig = f"chr{chrom}"
        pos1 = pos0 + 1
        try:
            for rec in vf.fetch(contig, pos0, pos1):
                if rec.pos != pos1:
                    continue
                dp = rec.info.get("DP")
                i16 = rec.info.get("I16")
                baf = None
                if i16 is not None and len(i16) >= 4:
                    ref_c = float(i16[0]) + float(i16[1])
                    alt_c = float(i16[2]) + float(i16[3])
                    if (ref_c + alt_c) > 0:
                        baf = alt_c / (ref_c + alt_c)
                out[(chrom, pos0)] = (dp, baf)
        except ValueError:
            continue
    vf.close()
    return out


def recompute_h_baf_excl_label(bf, bam_contig, h_pos0, ref_h, alt_h, exclude_mid_strs):
    """Direct single-locus pileup at h (NOT the whole-window co-observation scan --
    just one position, so this is cheap even though it re-touches the BAM). Returns
    ((depth_all, baf_all), (depth_excl, baf_excl)) computed from ONE pass so the two are
    a true apples-to-apples ablation -- SAME flag convention (FLAG_EXCLUDE, dup-flagged
    reads kept, matching Stage 1 and this whole project's documented convention) for
    both, differing ONLY in whether the label-source molecules are counted.
    depth_all/baf_all is deliberately NOT the same number as load_pseudobulk_depth_baf's
    depth_h/baf_h -- that one comes from the pre-built pseudobulk VCF, whose upstream
    samtools-mpileup call excludes dup-flagged reads by default (a real, documented
    discrepancy in this project, see readbacked_feasibility RESULTS.md section 6), so
    comparing it directly against a dup-kept recompute would confound two different
    axes (dup-inclusion AND label-exclusion) in one number. Using depth_all/baf_all as
    the "naive" partner to depth_excl/baf_excl isolates the SINGLE axis this audit is
    about (see RESULTS.md leakage audit for the full trace, including why the very
    first attempt at this ablation -- comparing against the pseudobulk I16 stat -- gave
    a confusing sign flip that turned out to be exactly this confound, not a new leak)."""
    ref_h, alt_h = ref_h.upper(), alt_h.upper()
    n_ref = n_alt = n_other = 0
    n_ref_x = n_alt_x = n_other_x = 0
    for read in bf.fetch(bam_contig, h_pos0, h_pos0 + 1):
        if read.flag & FLAG_EXCLUDE:
            continue
        rs, re_ = read.reference_start, read.reference_end
        if rs is None or re_ is None or not (rs <= h_pos0 < re_):
            continue
        seq = read.query_sequence
        if seq is None:
            continue
        aligned = {rp: qp for qp, rp in read.get_aligned_pairs(matches_only=True)}
        if h_pos0 not in aligned:
            continue
        base = seq[aligned[h_pos0]].upper()
        try:
            cb = read.get_tag("CB")
        except KeyError:
            cb = None
        try:
            ub = read.get_tag("UB")
        except KeyError:
            ub = None
        mid_str = f"umi:{cb}:{ub}" if (cb and ub) else f"read:{read.query_name}"
        is_excluded = mid_str in exclude_mid_strs
        if base == alt_h:
            n_alt += 1
            if not is_excluded:
                n_alt_x += 1
        elif base == ref_h:
            n_ref += 1
            if not is_excluded:
                n_ref_x += 1
        else:
            n_other += 1
            if not is_excluded:
                n_other_x += 1
    total = n_ref + n_alt + n_other
    baf = (n_alt / (n_ref + n_alt)) if (n_ref + n_alt) > 0 else np.nan
    total_x = n_ref_x + n_alt_x + n_other_x
    baf_x = (n_alt_x / (n_ref_x + n_alt_x)) if (n_ref_x + n_alt_x) > 0 else np.nan
    return (total, baf), (total_x, baf_x)


def load_presence_matrix(section, cls):
    cfg = SECTION_PATHS[section]
    path = os.path.join(cfg["matrix_dir"], f"{cfg['matrix_prefix']}_{cls}_matrix.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_calicost(section):
    cfg = SECTION_PATHS[section]
    if not cfg["calicost_build_ok"]:
        return None
    d = cfg["calicost_dir"]
    clone_labels = pd.read_csv(os.path.join(d, "clone_labels.tsv"), sep="\t")
    clone_labels["barcode"] = clone_labels["BARCODES"].str.split("_").str[0]
    dominant_clone = clone_labels["clone_label"].value_counts().idxmax()
    clones_present = sorted(clone_labels["clone_label"].dropna().unique().tolist())
    seg = pd.read_csv(os.path.join(d, "cnv_seglevel.tsv"), sep="\t")
    seg["CHR"] = seg["CHR"].astype(str)
    return dict(clone_labels=clone_labels, dominant_clone=dominant_clone,
                clones_present=clones_present, seg=seg)


def calicost_lookup(calico, chrom, pos_1based):
    if calico is None:
        return dict(loh_dominant_clone=np.nan, loh_any_clone=np.nan, n_clones_section=np.nan,
                     seg_length_bp=np.nan, calicost_build_mismatch=True)
    seg = calico["seg"]
    hit = seg[(seg["CHR"] == str(chrom)) & (seg["START"] <= pos_1based) & (seg["END"] >= pos_1based)]
    if hit.empty:
        return dict(loh_dominant_clone=np.nan, loh_any_clone=np.nan,
                     n_clones_section=len(calico["clones_present"]), seg_length_bp=np.nan,
                     calicost_build_mismatch=False)
    row = hit.iloc[0]
    any_loh, dom_loh = False, np.nan
    for k in calico["clones_present"]:
        acol, bcol = f"clone{k} A", f"clone{k} B"
        if acol in row.index and bcol in row.index and pd.notna(row[acol]) and pd.notna(row[bcol]):
            is_loh = (row[acol] == 0) or (row[bcol] == 0)
            any_loh = any_loh or is_loh
            if k == calico["dominant_clone"]:
                dom_loh = is_loh
    return dict(loh_dominant_clone=dom_loh, loh_any_clone=any_loh,
                n_clones_section=len(calico["clones_present"]),
                seg_length_bp=int(row["END"]) - int(row["START"]),
                calicost_build_mismatch=False)


def het_density(het_sorted, pos0, window):
    lo = bisect.bisect_left(het_sorted, pos0 - window)
    hi = bisect.bisect_right(het_sorted, pos0 + window)
    return hi - lo


def scan_vcf_by_spot(section, positions_needed, barcodes):
    """positions_needed: set of (chrom_bare, pos_1based). Single linear pass per spot
    file (not per-position tabix fetch -- far fewer total ops given thousands of spot
    files but only tens-hundreds of positions of interest per section).
    Returns {(chrom,pos_1based): {barcode: (ref_count, alt_count)}}."""
    cfg = SECTION_PATHS[section]
    want = set(positions_needed)
    out = defaultdict(dict)
    vdir = cfg["vcf_by_spot_dir"]
    t0 = time.time()
    n_files = 0
    for bc in barcodes:
        fp = os.path.join(vdir, f"{bc}.vcf.gz")
        if not os.path.exists(fp):
            continue
        n_files += 1
        vf = pysam.VariantFile(fp)
        sample_name = list(vf.header.samples)[0]
        for rec in vf.fetch():
            chrom = rec.chrom[3:] if rec.chrom.startswith("chr") else rec.chrom
            key = (chrom, rec.pos)
            if key not in want:
                continue
            ad = rec.samples[sample_name].get("AD")
            if ad is None or len(ad) < 2 or ad[0] is None or ad[1] is None:
                continue
            out[key][bc] = (ad[0], ad[1])
        vf.close()
    log(f"[features] {section}: scanned {n_files} vcf_by_spot files for {len(want)} positions "
        f"in {time.time() - t0:.1f}s")
    return out


def _pearson_spearman(vaf, haf):
    n = len(vaf)
    pear = spear = np.nan
    if n >= 3 and np.std(vaf) > 0 and np.std(haf) > 0:
        pear = sstats.pearsonr(vaf, haf)[0]
        spear = sstats.spearmanr(vaf, haf)[0]
    return pear, spear, n


def compute_spatial_and_codetect(sub, ad_by_pos, presence_v, presence_h, n_total_spots):
    """Builds BOTH a naive version of the spatial-ASE / co-detection features (uses every
    spot with data) and a leakage-safe version that EXCLUDES the exact spot barcode(s)
    whose molecules produced this candidate's phase label (label_source_barcodes, from
    Stage 1). This split exists because the naive version can leak: the co-observing
    molecule(s) that produced the label live in a specific spot, and when spatial_ase_n_spots
    is small (median ~5 among confident candidates) that spot can dominate the correlation
    -- confirmed empirically (see RESULTS.md leakage audit) to inflate AUROC to ~0.99 on the
    naive features. The _excl_label_spots columns are the ones actually used to answer the
    task's question; the naive columns are kept only to quantify/report the leak."""
    rows = []
    for _, r in sub.iterrows():
        chrom = r["chrom"]
        v_key_col = f"{chrom}_{r['pos_1based']}"
        h_key_col = f"{chrom}_{r['anchor_pos_1based']}"
        v_ad = ad_by_pos.get((chrom, r["pos_1based"]), {})
        h_ad = ad_by_pos.get((chrom, r["anchor_pos_1based"]), {})
        common_bc = set(v_ad) & set(h_ad)
        label_bcs = set(str(r.get("label_source_barcodes") or "").split(";")) - {""}

        vaf, haf, vaf_x, haf_x = [], [], [], []
        for bc in common_bc:
            vr, va = v_ad[bc]
            hr, ha = h_ad[bc]
            if (vr + va) > 0 and (hr + ha) > 0:
                vaf.append(va / (vr + va))
                haf.append(ha / (hr + ha))
                if bc not in label_bcs:
                    vaf_x.append(va / (vr + va))
                    haf_x.append(ha / (hr + ha))
        pear, spear, n_both = _pearson_spearman(vaf, haf)
        pear_x, spear_x, n_both_x = _pearson_spearman(vaf_x, haf_x)

        nv = nh = both = np.nan
        co_detect_rate = np.nan
        nv_x = nh_x = both_x = np.nan
        co_detect_rate_x = np.nan
        if v_key_col in presence_v.columns and h_key_col in presence_h.columns:
            vcol = presence_v[v_key_col]
            hcol = presence_h[h_key_col]
            v_detect = vcol > 0
            h_detect = hcol > 0
            nv = int(v_detect.sum())
            nh = int(h_detect.sum())
            both = int((v_detect & h_detect).sum())
            co_detect_rate = (both / nv) if nv > 0 else np.nan

            keep_mask = ~pd.Series(vcol.index, index=vcol.index).isin(label_bcs)
            v_detect_x = v_detect[keep_mask]
            h_detect_x = h_detect[keep_mask]
            nv_x = int(v_detect_x.sum())
            nh_x = int(h_detect_x.sum())
            both_x = int((v_detect_x & h_detect_x).sum())
            co_detect_rate_x = (both_x / nv_x) if nv_x > 0 else np.nan

        rows.append(dict(
            candidate_id=r["candidate_id"], n_label_source_spots=len(label_bcs),
            spatial_ase_corr=pear, spatial_ase_corr_spearman=spear, spatial_ase_n_spots=n_both,
            n_spots_v_detected=nv, n_spots_h_detected=nh, n_spots_both_detected=both,
            co_detect_rate=co_detect_rate, n_spots_total=n_total_spots,
            spatial_ase_corr_excl_label_spots=pear_x,
            spatial_ase_corr_spearman_excl_label_spots=spear_x,
            spatial_ase_n_spots_excl_label_spots=n_both_x,
            n_spots_v_detected_excl_label_spots=nv_x, n_spots_h_detected_excl_label_spots=nh_x,
            n_spots_both_detected_excl_label_spots=both_x,
            co_detect_rate_excl_label_spots=co_detect_rate_x,
        ))
    return pd.DataFrame(rows)


def stage2_build_features(labels_df, sections):
    os.makedirs(OUT_DIR, exist_ok=True)
    feat_parts, spat_parts = [], []
    for section in sections:
        sub = labels_df[labels_df["section"] == section].copy()
        if sub.empty:
            continue
        cfg = SECTIONS[section]
        het_by_chrom, het_allele, _ = load_het_sites(cfg)

        pos_needed_pb = {(r["chrom"], r["anchor_pos_1based"] - 1) for _, r in sub.iterrows()}
        pb = load_pseudobulk_depth_baf(section, pos_needed_pb)
        calico = load_calicost(section)
        presence_v = load_presence_matrix(section, "somatic")
        presence_h = load_presence_matrix(section, "1000G")
        in_tissue_barcodes = presence_v.index.tolist()

        positions_needed = set()
        for _, r in sub.iterrows():
            positions_needed.add((r["chrom"], r["pos_1based"]))
            positions_needed.add((r["chrom"], r["anchor_pos_1based"]))
        ad_by_pos = scan_vcf_by_spot(section, positions_needed, in_tissue_barcodes)
        spat_parts.append(compute_spatial_and_codetect(sub, ad_by_pos, presence_v, presence_h,
                                                         len(in_tissue_barcodes)))

        bf = pysam.AlignmentFile(cfg["bam"], "rb")
        rows = []
        for _, r in sub.iterrows():
            chrom = r["chrom"]
            pos0_v = r["pos_1based"] - 1
            het_sorted = het_by_chrom.get(chrom, [])
            depth_v = r["n_alt_reads"] + r["n_ref_reads"]
            baf_v = (r["n_alt_reads"] / depth_v) if depth_v > 0 else np.nan
            dp_h_pooled, baf_h_pooled = pb.get((chrom, r["anchor_pos_1based"] - 1), (np.nan, np.nan))
            # leakage audit pair: (depth_h, baf_h) = fresh dup-kept pileup at h using
            # EVERY covering read; (depth_h_excl_label_spots, baf_h_excl_label_spots) =
            # the identical pileup with the label-source molecules held out. Same flag
            # convention for both (see recompute_h_baf_excl_label docstring for why this,
            # not the pooled pseudobulk VCF stat, is the correct "naive" comparison
            # partner). depth_h_pooled/baf_h_pooled (the pseudobulk-VCF I16 stat) is kept
            # too, purely as an independent sanity cross-check, not fed to any model.
            bam_contig = f"chr{chrom}" if cfg["chr_prefix_in_bam"] else chrom
            h_pos0 = r["anchor_pos_1based"] - 1
            exclude_mids = set(str(r.get("label_source_molecule_keys") or "").split(";")) - {""}
            (dp_h, baf_h), (dp_h_x, baf_h_x) = recompute_h_baf_excl_label(
                bf, bam_contig, h_pos0, r["anchor_ref"], r["anchor_alt"], exclude_mids)
            calico_feat = calicost_lookup(calico, chrom, r["pos_1based"])
            rows.append(dict(
                candidate_id=r["candidate_id"], section=section, chrom=chrom, pos_1based=r["pos_1based"],
                dist_to_anchor_bp=r["dist_to_anchor_bp"],
                het_density_10kb=het_density(het_sorted, pos0_v, 10_000),
                het_density_100kb=het_density(het_sorted, pos0_v, 100_000),
                depth_v=depth_v, baf_v=baf_v, depth_h=dp_h, baf_h=baf_h,
                depth_h_excl_label_spots=dp_h_x, baf_h_excl_label_spots=baf_h_x,
                depth_h_pooled_vcf=dp_h_pooled, baf_h_pooled_vcf=baf_h_pooled,
                **calico_feat,
            ))
        bf.close()
        feat_parts.append(pd.DataFrame(rows))
        log(f"[features] {section}: {len(sub)} candidates featurized")

    feat_df = pd.concat(feat_parts, ignore_index=True)
    spat_df = pd.concat(spat_parts, ignore_index=True)
    out = feat_df.merge(spat_df, on="candidate_id", how="left")
    out.to_csv(os.path.join(OUT_DIR, "features.csv"), index=False)
    log(f"[features] TOTAL: {len(out)} rows written to {os.path.join(OUT_DIR, 'features.csv')}")
    return out


def covariate_shift_summary(sections):
    """Where does the UNTESTABLE 92% actually live, in depth/distance terms? Pulled
    straight from the full somatic gate (all rows, not just testable_P2) -- no BAM
    re-scan needed, this is pure re-aggregation of an already-computed table."""
    df = pd.read_csv(GATE_CSV)
    df = df[df["section"].isin(sections)].copy()
    for c in ["testable_P2", "het_in_range", "ge2_alt_reads"]:
        df[c] = df[c].astype(str) == "True"
    df["depth"] = df["n_alt_reads"] + df["n_ref_reads"]
    rows = []
    for section, g in df.groupby("section"):
        testable = g[g["testable_P2"]]
        untestable = g[~g["testable_P2"]]
        rows.append(dict(
            section=section, n_testable=len(testable), n_untestable=len(untestable),
            testable_median_depth=testable["depth"].median(),
            untestable_median_depth=untestable["depth"].median(),
            testable_median_nearest_het_dist_bp=testable["nearest_het_dist_bp"].median(),
            untestable_median_nearest_het_dist_bp=untestable["nearest_het_dist_bp"].median(),
            untestable_frac_no_het_within_1000bp=float((~untestable["het_in_range"]).mean()),
            untestable_frac_lt2_alt_reads=float((~untestable["ge2_alt_reads"]).mean()),
            untestable_frac_het_in_range_and_ge2_alt_but_not_coobs=float(
                (untestable["het_in_range"] & untestable["ge2_alt_reads"]).mean()),
        ))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, "covariate_shift_summary.csv"), index=False)
    log(f"[covariate] written {os.path.join(OUT_DIR, 'covariate_shift_summary.csv')}")
    return out


# =============================================================================
# STAGE 3 -- fit + evaluate (LR + gradient boosting, NOT an MLP)
# =============================================================================
FEATURE_COLS_SHARED = ["dist_to_anchor_bp", "het_density_10kb", "het_density_100kb",
                        "depth_v", "baf_v", "seg_length_bp"]
# depth_v/baf_v ARE shared (empirically checked: point-biserial r(baf_v,label) is
# non-significant, p>0.17, at every depth stratum -- no mechanistic reason for it to
# leak the phase direction, since it only describes v's own read support, not h's
# allele).
#
# depth_h/baf_h are DROPPED FROM THE CLEAN SET ENTIRELY (not just given an
# excl_label_spots variant) after a two-stage investigation, both stages empirical,
# not assumed -- full trace in RESULTS.md leakage audit, summary here:
#  1. Naive baf_h (pooled pseudobulk-VCF I16, or an equivalent fresh full pileup) has
#     point-biserial r(baf_h,label) = 0.65 at depth_h<=10, decaying to 0.17 (n.s.) at
#     depth_h>100 -- the label-generating molecules are a large share of a small pool.
#  2. The "obvious" fix -- recompute h's pileup EXCLUDING those exact molecules -- does
#     NOT clean this up. It produces an even STRONGER correlation with the OPPOSITE
#     sign (r=-0.70 at low depth), because removing the reads that agree with a label
#     from a small finite pool mechanically enriches what's left for the other allele
#     -- a leave-k-out arithmetic artifact, confirmed by stratifying on excluded
#     fraction: r is -0.75 when >50% of the pool was excluded, falls to -0.19 (n.s.,
#     p=0.22) when <5% was excluded. There is no read-count-based way to build an
#     h-BAF feature here that is both informative and demonstrably leakage-free at
#     the depths this dataset actually has, so it is left out of the clean set rather
#     than shipped with an unresolved bias. depth_h_excl_label_spots/baf_h_excl_label_spots
#     (and the naive/pooled variants) are still written to features.csv for inspection.
#
# "clean" = the leakage-safe primary feature set (spatial-ASE / co-detection computed
# with the exact label-source spot(s) excluded; no h-BAF feature at all).
# "naive" = spatial-ASE / co-detection WITHOUT that exclusion, plus naive baf_h, kept
# only to quantify the leak (see RESULTS.md leakage audit -- naive spatial_ase_corr +
# baf_h together drove AUROC to ~0.99 on the confident subset).
FEATURE_VARIANTS = {
    "clean_excl_label_spots": FEATURE_COLS_SHARED + [
        "spatial_ase_corr_excl_label_spots", "spatial_ase_corr_spearman_excl_label_spots",
        "spatial_ase_n_spots_excl_label_spots", "co_detect_rate_excl_label_spots",
        "n_spots_v_detected_excl_label_spots", "n_spots_h_detected_excl_label_spots",
        "n_spots_both_detected_excl_label_spots",
        # NOTE: no depth_h/baf_h feature here -- see the comment above FEATURE_COLS_SHARED.
    ],
    "naive_leaky": FEATURE_COLS_SHARED + [
        "spatial_ase_corr", "spatial_ase_corr_spearman", "spatial_ase_n_spots",
        "co_detect_rate", "n_spots_v_detected", "n_spots_h_detected", "n_spots_both_detected",
        "depth_h", "baf_h",
    ],
}
FEATURE_COLS_BOOL = ["loh_dominant_clone", "loh_any_clone", "calicost_build_mismatch"]
FEATURE_COLS_CATEG = ["section"]


def _bin_depth(d):
    if d <= 2:
        return "2 (gate minimum)"
    if d <= 3:
        return "3"
    if d <= 5:
        return "4-5"
    return "6+"


def _bin_dist(d):
    if d <= 50:
        return "0-50bp"
    if d <= 200:
        return "51-200bp"
    if d <= 1000:
        return "201-1000bp"
    return ">1000bp"


def stage3_fit(labels_df, features_df):
    from sklearn.base import clone as sk_clone
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    os.makedirs(OUT_DIR, exist_ok=True)
    merged = labels_df.merge(features_df, on="candidate_id", how="inner", suffixes=("", "_feat"))
    log(f"[fit] merged labels x features: {len(merged)} rows "
        f"(labels={len(labels_df)}, features={len(features_df)})")

    all_cv_rows, all_strat_rows, all_fi_rows = [], [], []

    for feature_variant, feature_cols_numeric in FEATURE_VARIANTS.items():
        all_feature_cols = feature_cols_numeric + FEATURE_COLS_BOOL + FEATURE_COLS_CATEG
        for subset_name in ["confident", "all_labeled"]:
            if subset_name == "confident":
                df = merged[merged["confident"] & (merged["label"] != "tie")].copy()
            else:
                df = merged[merged["label"] != "tie"].copy()
            n_ties = int((merged["label"] == "tie").sum())
            log(f"[fit] variant={feature_variant} subset={subset_name}: {len(df)} candidates "
                f"(ties excluded from all subsets: {n_ties})")
            if len(df) < 30:
                log(f"[fit] variant={feature_variant} subset={subset_name}: too few rows ({len(df)}), skipping")
                continue

            df["y"] = (df["label"] == "same").astype(int)
            for c in FEATURE_COLS_BOOL:
                df[c] = df[c].astype("float")
            X = df[all_feature_cols].copy()
            y = df["y"].values
            groups = df["candidate_id"].values
            n_same, n_opp = int(y.sum()), int((1 - y).sum())
            log(f"[fit] variant={feature_variant} subset={subset_name}: class balance same={n_same} "
                f"({100 * y.mean():.1f}%) opposite={n_opp} ({100 * (1 - y.mean()):.1f}%)")

            numeric_all = feature_cols_numeric + FEATURE_COLS_BOOL
            pre_lr = ColumnTransformer([
                ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                                   ("scale", StandardScaler())]), numeric_all),
                ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_CATEG),
            ])
            lr_pipe = Pipeline([("pre", pre_lr),
                                 ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])

            pre_gbm = ColumnTransformer([
                ("num", "passthrough", numeric_all),
                ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLS_CATEG),
            ])
            # HistGradientBoostingClassifier only turns on early stopping automatically
            # above 10,000 training samples (sklearn default early_stopping='auto'); this
            # dataset has 700-3300 per fold, so the untuned default silently runs the full
            # 100 boosting rounds with no validation-based stopping -- a known overfitting
            # risk on tabular data this size. Caught empirically, not just by reputation:
            # every individual "clean" feature has near-chance univariate AUROC (0.50-0.57,
            # see RESULTS.md leakage audit), yet default-hyperparameter GBM reached 0.74 --
            # exactly the gap-between-univariate-and-multivariate signature of overfitting.
            # A depth/iteration/L2 sweep showed AUROC falls monotonically with regularization
            # strength (0.74 default -> 0.70 early-stopped -> 0.66 shallow+L2 -> 0.63 very
            # conservative) WITHOUT collapsing to 0.5 -- i.e. some real signal survives even
            # heavy regularization, so this isn't pure noise-fitting either. "gradient_boosting"
            # (early_stopping=True, the standard scikit-learn-recommended fix for exactly this
            # dataset-size regime) is reported as primary; "gradient_boosting_unregularized"
            # (bare defaults) is kept alongside, not to cherry-pick the lower number but so the
            # sensitivity is fully visible rather than hidden behind one hyperparameter choice.
            gbm_pipe = Pipeline([("pre", pre_gbm),
                                  ("clf", HistGradientBoostingClassifier(
                                      random_state=0, early_stopping=True,
                                      n_iter_no_change=10, validation_fraction=0.2))])
            gbm_unreg_pipe = Pipeline([("pre", pre_gbm),
                                        ("clf", HistGradientBoostingClassifier(random_state=0))])

            models = {"logistic_regression": lr_pipe, "gradient_boosting": gbm_pipe,
                      "gradient_boosting_unregularized": gbm_unreg_pipe,
                      "dummy_stratified": DummyClassifier(strategy="stratified", random_state=0)}

            n_splits = 5
            sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
            fold_assign = list(sgkf.split(X, y, groups))

            oof = {name: np.full(len(df), np.nan) for name in models}
            for name, model in models.items():
                for fold, (tr, te) in enumerate(fold_assign):
                    m = sk_clone(model)
                    m.fit(X.iloc[tr], y[tr])
                    proba = m.predict_proba(X.iloc[te])[:, 1]
                    oof[name][te] = proba
                    pred = (proba >= 0.5).astype(int)
                    yte = y[te]
                    auc = roc_auc_score(yte, proba) if len(set(yte)) > 1 else np.nan
                    all_cv_rows.append(dict(feature_variant=feature_variant, label_subset=subset_name,
                                             model=name, fold=fold, n_test=len(te), n_train=len(tr),
                                             auroc=auc, accuracy=accuracy_score(yte, pred),
                                             brier=brier_score_loss(yte, proba)))
            for fold, (tr, te) in enumerate(fold_assign):
                yte = y[te]
                proba50 = np.full(len(te), 0.5)
                all_cv_rows.append(dict(feature_variant=feature_variant, label_subset=subset_name,
                                         model="chance_50pct", fold=fold, n_test=len(te), n_train=len(tr),
                                         auroc=0.5, accuracy=0.5, brier=brier_score_loss(yte, proba50)))

            model_names = list(models.keys())
            for name in model_names:
                df[f"oof_{name}"] = oof[name]
            df["depth_v_bin"] = df["depth_v"].apply(_bin_depth)
            df["dist_bin"] = df["dist_to_anchor_bp"].apply(_bin_dist)
            df["loh_stratum"] = df["loh_any_clone"].map(
                {1.0: "LOH", 0.0: "copy-neutral"}).fillna("unknown/build-mismatch")

            for stratify_col in ["depth_v_bin", "dist_bin", "loh_stratum", "section"]:
                for level, g in df.groupby(stratify_col, observed=True):
                    row = dict(feature_variant=feature_variant, label_subset=subset_name,
                               stratify_by=stratify_col, level=level,
                               n=len(g), n_same=int(g["y"].sum()), n_opposite=int((1 - g["y"]).sum()))
                    for name in model_names:
                        col = f"oof_{name}"
                        yv, pv = g["y"].values, g[col].values
                        mask = ~np.isnan(pv)
                        row[f"{name}_auroc"] = (roc_auc_score(yv[mask], pv[mask])
                                                 if mask.sum() >= 2 and len(set(yv[mask])) > 1 else np.nan)
                        row[f"{name}_accuracy"] = (accuracy_score(yv[mask], (pv[mask] >= 0.5).astype(int))
                                                    if mask.sum() > 0 else np.nan)
                    all_strat_rows.append(row)

            # full-data fits (not CV) purely to read off feature importance / coefficients
            lr_full = sk_clone(lr_pipe)
            lr_full.fit(X, y)
            gbm_full = sk_clone(gbm_pipe)
            gbm_full.fit(X, y)
            lr_names = lr_full.named_steps["pre"].get_feature_names_out()
            for name, coef in zip(lr_names, lr_full.named_steps["clf"].coef_[0]):
                all_fi_rows.append(dict(feature_variant=feature_variant, label_subset=subset_name,
                                         model="logistic_regression", feature=name, importance=coef,
                                         importance_type="standardized_coef"))
            # HistGradientBoostingClassifier exposes no impurity-based feature_importances_
            # (unlike GradientBoostingClassifier/RandomForest) -- permutation importance
            # (below) is the recommended, and here the only, importance measure for it.
            perm = permutation_importance(gbm_full, X, y, n_repeats=20, random_state=0, scoring="roc_auc")
            for name, imp, sd in zip(X.columns, perm.importances_mean, perm.importances_std):
                all_fi_rows.append(dict(feature_variant=feature_variant, label_subset=subset_name,
                                         model="gradient_boosting", feature=name, importance=imp,
                                         importance_type="permutation_auc", importance_std=sd))

            # persist the OOF table itself (useful for ad hoc follow-up, not in the required list but
            # cheap and directly supports the stratified numbers above being independently checkable)
            oof_path = os.path.join(OUT_DIR, f"oof_predictions_{feature_variant}_{subset_name}.csv")
            keep_cols = (["candidate_id", "section", "label", "y", "depth_v", "dist_to_anchor_bp",
                          "loh_stratum", "depth_v_bin", "dist_bin"]
                         + [f"oof_{name}" for name in model_names])
            df[keep_cols].to_csv(oof_path, index=False)

    pd.DataFrame(all_cv_rows).to_csv(os.path.join(OUT_DIR, "cv_performance.csv"), index=False)
    pd.DataFrame(all_strat_rows).to_csv(os.path.join(OUT_DIR, "stratified_performance.csv"), index=False)
    pd.DataFrame(all_fi_rows).to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)
    log(f"[fit] wrote cv_performance.csv, stratified_performance.csv, feature_importance.csv to {OUT_DIR}")


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["labels", "features", "fit", "covariate", "all"], default="all")
    ap.add_argument("--sections", nargs="+", default=SECTION_LIST, choices=SECTION_LIST)
    ap.add_argument("--limit", type=int, default=None, help="cap candidates per section (labels stage, testing)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    labels_path = os.path.join(OUT_DIR, "labels_relative_phase.csv")
    features_path = os.path.join(OUT_DIR, "features.csv")

    if args.stage in ("labels", "all"):
        stage1_extract_labels(args.sections, limit=args.limit)

    # NOTE: "chrom" must stay a STRING dtype. All autosomes here are numeric-looking
    # ("1".."22"), so pandas' default type inference silently reads the column as
    # int64 -- which then breaks every (chrom, pos) dict-key lookup against
    # het_by_chrom / ad_by_pos (both keyed by the STRING chrom that load_het_sites()
    # and the vcf_by_spot scan use). Caught empirically: with chrom as int64,
    # het_density_10kb/100kb were silently always 0 and spatial_ase_corr was always
    # NaN for all 3,787 rows on the first run -- both fixed by this dtype pin.
    if args.stage in ("features", "fit", "all"):
        if not os.path.exists(labels_path):
            raise SystemExit(f"labels_relative_phase.csv not found at {labels_path} -- run --stage labels first")
        labels_df = pd.read_csv(labels_path, dtype={"chrom": str})

    if args.stage in ("features", "all"):
        stage2_build_features(labels_df, args.sections)
        covariate_shift_summary(args.sections)

    if args.stage in ("fit", "all"):
        if not os.path.exists(features_path):
            raise SystemExit(f"features.csv not found at {features_path} -- run --stage features first")
        features_df = pd.read_csv(features_path, dtype={"chrom": str})
        stage3_fit(labels_df, features_df)

    if args.stage == "covariate":
        covariate_shift_summary(args.sections)

    log("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
