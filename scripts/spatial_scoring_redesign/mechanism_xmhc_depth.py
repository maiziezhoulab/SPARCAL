#!/usr/bin/env python
"""
mechanism_xmhc_depth.py
=========================
Coordinator request 2026-09-03, point 3: the size-matched Method A set is
BELOW the genome-length xMHC expectation (0.33x in the DCIS2 smoke test) --
striking, and needs a mechanism, not just a number.

Hypothesis to test: the depth-conditioned null explicitly controls for
coverage; xMHC is a high-coverage region (highly expressed HLA genes); so
conditioning on depth removes exactly the variants the UNCONTROLLED shipped
quota (Method C's zeta = raw local-clustering-coefficient, no depth
adjustment) over-selects there. Two direct, data-driven checks, no BAM
rescan (everything from candidates.tsv, already cached):

  (a) Depth excess: is per-candidate depth (info_dp) systematically higher
      in xMHC than the rest of the genome, in the FULL Stage-2 candidate pool?
  (b) Depth-bin-stratified selection rates: bin the full candidate pool into
      depth deciles; compute P(selected | depth bin) separately for Method C
      (shipped quota) and Method A (matched). If Method C's selection rate
      rises with depth while Method A's stays roughly flat, AND xMHC sits in
      the high-depth tail, that is the mechanism, shown directly rather than
      asserted -- and the A/C selection-RATIO across bins should be roughly
      constant (not xMHC-specific) if the effect is a general depth
      correction rather than a special-cased fix.

Output: data/spatial_scoring_redesign_2026-08-31/mechanism_xmhc_depth_{section}.csv
        (one row per depth decile) + a printed summary.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def run_section(section):
    outdir = common.cache_dir(section)
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t", dtype={"chrom": str})
    sel_dir = os.path.join(common.OUT_ROOT, "selected_sets")
    c_sel = set(pd.read_csv(os.path.join(sel_dir, f"{section}_methodC_shipped.tsv"), sep="\t")["variant"])
    a_sel = set(pd.read_csv(os.path.join(sel_dir, f"{section}_methodA_matched.tsv"), sep="\t")["variant"])

    in_xmhc = (candidates["chrom"].astype(str) == common.XMHC_CHROM) & \
              (candidates["pos"] >= common.XMHC_START) & (candidates["pos"] <= common.XMHC_END)
    depth = candidates["info_dp"].values
    valid = ~np.isnan(depth)

    # (a) depth excess
    med_xmhc = float(np.median(depth[in_xmhc.values & valid]))
    med_rest = float(np.median(depth[(~in_xmhc.values) & valid]))
    mean_xmhc = float(np.mean(depth[in_xmhc.values & valid]))
    mean_rest = float(np.mean(depth[(~in_xmhc.values) & valid]))
    pct_xmhc_in_rest_dist = float((depth[(~in_xmhc.values) & valid] < med_xmhc).mean() * 100)
    log(f"{section}: candidate-pool depth -- xMHC median={med_xmhc:.1f} mean={mean_xmhc:.2f}  "
        f"rest-of-genome median={med_rest:.1f} mean={mean_rest:.2f}  "
        f"(xMHC's median depth sits at the {pct_xmhc_in_rest_dist:.1f}th percentile of the rest-of-genome depth distribution)")

    # (b) depth-decile-stratified selection rates
    dvalid = candidates[valid].copy()
    dvalid["decile"] = pd.qcut(dvalid["info_dp"].rank(method="first"), 10, labels=False, duplicates="drop")
    dvalid["in_xmhc"] = in_xmhc[valid].values
    dvalid["sel_c"] = dvalid["variant"].isin(c_sel)
    dvalid["sel_a"] = dvalid["variant"].isin(a_sel)

    rows = []
    for d in sorted(dvalid["decile"].unique()):
        sub = dvalid[dvalid["decile"] == d]
        rate_c = sub["sel_c"].mean()
        rate_a = sub["sel_a"].mean()
        rows.append(dict(
            section=section, depth_decile=int(d), n_candidates=len(sub),
            depth_median=float(sub["info_dp"].median()),
            frac_xmhc=float(sub["in_xmhc"].mean()),
            selection_rate_methodC=float(rate_c),
            selection_rate_methodA_matched=float(rate_a),
            ratio_A_over_C=float(rate_a / rate_c) if rate_c > 0 else np.nan,
        ))
    decile_df = pd.DataFrame(rows)

    out_path = os.path.join(common.OUT_ROOT, f"mechanism_xmhc_depth_{section}.csv")
    decile_df.to_csv(out_path, index=False)
    log(f"{section}: wrote {out_path}")
    log(decile_df.to_string(index=False))

    # is xMHC concentrated in the high-depth deciles?
    xmhc_by_decile = dvalid[dvalid["in_xmhc"]]["decile"].value_counts(normalize=True).sort_index()
    log(f"{section}: xMHC candidates' distribution across depth deciles (0=shallowest, 9=deepest):")
    log(xmhc_by_decile.to_string())

    summary = dict(
        section=section,
        depth_median_xmhc=med_xmhc, depth_median_rest=med_rest,
        depth_mean_xmhc=mean_xmhc, depth_mean_rest=mean_rest,
        xmhc_median_depth_percentile_in_rest=pct_xmhc_in_rest_dist,
        overall_selection_rate_C=float(dvalid["sel_c"].mean()),
        overall_selection_rate_A=float(dvalid["sel_a"].mean()),
        corr_decile_rate_C=float(decile_df["depth_decile"].corr(decile_df["selection_rate_methodC"], method="spearman")),
        corr_decile_rate_A=float(decile_df["depth_decile"].corr(decile_df["selection_rate_methodA_matched"], method="spearman")),
        ratio_A_over_C_cv=float(decile_df["ratio_A_over_C"].std() / decile_df["ratio_A_over_C"].mean())
        if decile_df["ratio_A_over_C"].mean() else np.nan,
    )
    return decile_df, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    args = ap.parse_args()
    _, summary = run_section(args.section)
    log("SUMMARY: " + str(summary))
