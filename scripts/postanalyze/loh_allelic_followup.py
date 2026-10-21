#!/usr/bin/env python
"""Step 4c -- rebuild the LOH test after its original positive control failed.

WHAT WENT WRONG IN loh_allelic_test.py, and why it had to be rebuilt.

  (1) THE CONTROL SET WAS NOT HETEROZYGOUS.  That script assumed SPARCAL's
      1KGP-defined germline class is a set of known heterozygotes.  It is not:
      it contains every panel-matching germline variant regardless of genotype,
      and measured in diploid spots in copy-balanced regions, 71.1% of those
      sites sit at BAF ~ 1 (homozygous alt) with only 14.4% in the heterozygous
      band.  A set that is three-quarters homozygous cannot detect LOSS of
      heterozygosity, so the gate reported median BAF = 1.0 in every cell.

  (2) copyKAT "LOSS" SEGMENTS ARE LOW-EXPRESSION SEGMENTS, so region and depth
      are confounded.  Pooled depth in loss vs balanced regions: 10.7 vs 19.7
      (DCIS2 aneuploid), 11.7 vs 19.3 (DCIS1 aneuploid).  Critically the same
      drop appears in DIPLOID spots (4.4 vs 7.0; 6.9 vs 9.1), which cannot be a
      copy-number effect because those spots carry no loss -- it is expression.
      Low depth pushes BAF toward 0 and 1, which mimics the LOH signature.
      This is the coverage confound that already sank the region-detection story
      and the COSMIC enrichment, appearing a third time.

THE REBUILT DESIGN, which fixes both:

  * HETEROZYGOSITY IS DEFINED EMPIRICALLY AND IN THE OTHER SPOT GROUP.  A site
    counts as heterozygous if its BAF measured in DIPLOID spots lies in
    [0.35, 0.65].  The readout is then the allelic shift in ANEUPLOID spots.
    Defining the set in one group and measuring it in the other removes the
    circularity, and it needs no assumption about panel genotypes.
  * DEPTH IS REQUIRED IN BOTH GROUPS (>= --min-dp each) AND THEN STRATIFIED.
    A single depth threshold is not enough when the region contrast is itself a
    depth contrast, so the comparison is repeated inside fixed depth bands and
    is only believed if it holds in all of them.
  * THE METHOD COMPARISON CARRIES A NEGATIVE CONTROL.  The same contrast is run
    in copy-BALANCED regions, where the two methods should not differ if the
    metric measures consistency with copy state rather than callset composition.

RESULT (see RESULT.md next to the outputs): the segment validation passes for
DCIS1 in every depth band and fails for DCIS2; the method comparison fails its
negative control and is not interpretable.  Read RESULT.md before reusing any
number from here.

Reads   data/loh_allelic_test_2026-08-06/loh_baf_raw.csv   (no new pileup needed)
Writes  data/loh_allelic_test_2026-08-06/followup_*.csv

Run:  python scripts/postanalyze/loh_allelic_followup.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
INDIR = os.path.join(REPO, "data", "loh_allelic_test_2026-08-06")

HET_LO, HET_HI = 0.35, 0.65
DP_BANDS = [(20, 50, "20-50"), (50, 100, "50-100"), (100, 200, "100-200"), (200, 1e9, ">200")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-dp", type=int, default=20,
                    help="pooled depth required in BOTH spot groups")
    ap.add_argument("--indir", default=INDIR)
    args = ap.parse_args()

    d = pd.read_csv(os.path.join(args.indir, "loh_baf_raw.csv"))

    # ---- 0. document the two failures, so they cannot be quietly forgotten ----
    diag = []
    for s in sorted(d["sample"].unique()):
        for reg in ("loss", "balanced"):
            for tag in ("aneuploid", "diploid"):
                dp = d[(d["sample"] == s) & (d.region == reg)][f"dp_{tag}"].dropna()
                diag.append(dict(sample=s, region=reg, spot_group=tag, n=len(dp),
                                 median_DP=float(dp.median()), mean_DP=round(float(dp.mean()), 2),
                                 pct_DP_ge_10=round(100 * float((dp >= 10).mean()), 2)))
    diag = pd.DataFrame(diag)

    gen = []
    g = d[d.group == "germline_1kG"]
    for s in sorted(d["sample"].unique()):
        b = g[(g["sample"] == s) & (g.region == "balanced")]["baf_diploid"].dropna()
        if not len(b):
            continue
        gen.append(dict(sample=s, n=len(b),
                        pct_hom_ref=round(100 * float((b < 0.05).mean()), 2),
                        pct_het_band=round(100 * float(((b >= HET_LO) & (b <= HET_HI)).mean()), 2),
                        pct_hom_alt=round(100 * float((b > 0.95).mean()), 2)))
    gen = pd.DataFrame(gen)

    # ---- 1. segment validation on an empirical, cross-group het set ----------
    val = []
    for s in sorted(d["sample"].unique()):
        sub = d[(d["sample"] == s) & (d.dp_diploid >= args.min_dp)
                & (d.dp_aneuploid >= args.min_dp)]
        het = sub[(sub.baf_diploid >= HET_LO) & (sub.baf_diploid <= HET_HI)].copy()
        het["shift"] = (het.baf_aneuploid - 0.5).abs()
        for lo, hi, lab in DP_BANDS + [(args.min_dp, 1e9, "ALL")]:
            h = het[(het.dp_aneuploid >= lo) & (het.dp_aneuploid < hi)]
            L = h[h.region == "loss"]["shift"].dropna().values
            B = h[h.region == "balanced"]["shift"].dropna().values
            row = dict(sample=s, dp_band=lab, n_loss=len(L), n_balanced=len(B))
            if len(L) >= 5 and len(B) >= 5:
                row.update(median_shift_loss=round(float(np.median(L)), 4),
                           median_shift_balanced=round(float(np.median(B)), 4),
                           p_loss_gt_balanced=float(
                               mannwhitneyu(L, B, alternative="greater").pvalue))
            val.append(row)
    val = pd.DataFrame(val)

    # ---- 2. method comparison, WITH its negative control ---------------------
    d["method_S"] = d.group.isin(["concordant", "sparcal_private"])
    d["method_T"] = d.group.isin(["concordant", "spatialsnv_private"])
    mth = []
    for s in sorted(d["sample"].unique()):
        for reg in ("loss", "balanced"):
            sub = d[(d["sample"] == s) & (d.region == reg)
                    & (d.dp_aneuploid >= args.min_dp)].copy()
            sub["het"] = (sub.baf_aneuploid >= HET_LO) & (sub.baf_aneuploid <= HET_HI)
            S, T = sub[sub.method_S], sub[sub.method_T]
            if len(S) < 5 or len(T) < 5:
                mth.append(dict(sample=s, region=reg, n_SPARCAL=len(S),
                                n_SpatialSNV=len(T), note="too few"))
                continue
            sh, th = int(S.het.sum()), int(T.het.sum())
            mth.append(dict(
                sample=s, region=reg, n_SPARCAL=len(S), het_pct_SPARCAL=round(100 * sh / len(S), 2),
                n_SpatialSNV=len(T), het_pct_SpatialSNV=round(100 * th / len(T), 2),
                fisher_p=float(fisher_exact([[sh, len(S) - sh], [th, len(T) - th]])[1])))
    mth = pd.DataFrame(mth)

    for name, df in [("followup_depth_diagnostic.csv", diag),
                     ("followup_germline_genotype_mix.csv", gen),
                     ("followup_segment_validation.csv", val),
                     ("followup_method_comparison.csv", mth)]:
        df.to_csv(os.path.join(args.indir, name), index=False)
        print(f"wrote {name}")

    print("\n=== why the original control failed: 1KGP-defined genotype mix "
          "(diploid spots, balanced regions) ===")
    print(gen.to_string(index=False))
    print("\n=== why region and depth are confounded ===")
    print(diag[diag.spot_group == "diploid"].to_string(index=False))
    print("\n=== segment validation (empirical het set, depth-stratified) ===")
    print(val.to_string(index=False))
    print("\n=== method comparison + negative control ===")
    print(mth.to_string(index=False))


if __name__ == "__main__":
    main()
