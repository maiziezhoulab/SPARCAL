#!/usr/bin/env python
"""
build_selected_sets.py
=======================
Combines Method A / B / C outputs into the selected-set files used by the
four evaluation endpoints, matched on size where possible (task spec: "For
the set each method selects (matched on selected-set SIZE where possible, so
the comparison is fair)").

For every section, writes (under selected_sets/):
  {section}_methodC_shipped.tsv        the shipped 10% quota, size S_C (already
                                        written by method_c_baseline.py; re-written
                                        here too so this script is idempotent/self-contained)
  {section}_methodA_natural_q05.tsv    Method A's OWN selection at q<0.05 (may be empty)
  {section}_methodA_natural_q10.tsv    ... q<0.10
  {section}_methodA_natural_q20.tsv    ... q<0.20
  {section}_methodA_matched.tsv        top S_C candidates by ascending p_joincount
                                        (tie-break: descending join_count_obs, then variant)
  {section}_methodA_diagnostic_plt1.tsv  DIAGNOSTIC STRATUM, NOT A CALLSET: every candidate
                                        that beat at least one of the 999 permutations
                                        (p_joincount < 1). This is NOT a significance
                                        threshold -- p<1 only requires beating 1/999 draws,
                                        expected to happen by chance for many candidates.
                                        Reported so its molecular-support number can be
                                        checked against the random-multispot baseline below
                                        for circularity (coordinator request 2026-09-03).
  {section}_random_multispot_baseline.tsv  seeded random sample, size-matched to the
                                        diagnostic stratum above, drawn ONLY from candidates
                                        with n_pos_spots>=2 (same structural precondition a
                                        join-count/Moran's I statistic requires to be
                                        non-degenerate) -- the circularity control: if this
                                        baseline's molecular support is ALSO ~100%, the
                                        diagnostic stratum's 100% is a property of "having
                                        >=2 spots" and not of Method A's statistic.
  {section}_methodB_matched.tsv        top S_C candidates by descending gae_score
                                        (tie-break: variant)
  manifest.csv                         one row per (section, method_variant, n_selected, path)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

RANDOM_BASELINE_SEED = 0


def _write(df, section, name, sel_dir, manifest_rows):
    path = os.path.join(sel_dir, f"{section}_{name}.tsv")
    df[["variant", "chrom", "pos"]].to_csv(path, sep="\t", index=False)
    manifest_rows.append(dict(section=section, method_variant=name, n_selected=len(df), path=path))
    print(f"{section}: {name} -> {len(df)} variants -> {path}")


def run_section(section):
    outdir = common.cache_dir(section)
    sel_dir = os.path.join(common.OUT_ROOT, "selected_sets")
    os.makedirs(sel_dir, exist_ok=True)
    manifest_rows = []

    a = pd.read_csv(os.path.join(outdir, f"methodA_{section}_scores.tsv"), sep="\t")
    b_path = os.path.join(outdir, f"methodB_{section}_scores.tsv")
    c = a[a["shipped_classification"] == "somatic"].copy()
    S_C = len(c)
    _write(c.sort_values("shipped_vote_fraction", ascending=False), section, "methodC_shipped", sel_dir, manifest_rows)

    for q in (0.05, 0.10, 0.20):
        sub = a[a["q_joincount"] < q].sort_values(["q_joincount", "join_count_obs"], ascending=[True, False])
        tag = f"methodA_natural_q{int(q * 100):02d}"
        _write(sub, section, tag, sel_dir, manifest_rows)

    # BUG FOUND AND FIXED 2026-09-03 (coordinator's mechanism-check request surfaced
    # it): the original tie-break used ascending VARIANT-NAME STRING for the ~87%
    # of methodA_matched that is p==1 padding. Chromosome tokens sort as STRINGS
    # ("1","10","11",...,"19","2","20",...), so alphabetically-first padding was
    # chromosomes 10/11 almost exclusively -- verified directly: the first 21,865
    # (of 248,251) p==1 DCIS2 candidates sorted by variant name were 100% chrom
    # 10/11, chrom 6 (xMHC) entirely absent. This manufactured a fake "Method A
    # avoids xMHC" signal in the padded matched-size set that had nothing to do
    # with the calibrated statistic. Fixed with a seeded RANDOM tie-break instead
    # (deterministic, reproducible, not biased by chromosome-string sort order).
    rng_tiebreak = np.random.default_rng(RANDOM_BASELINE_SEED)
    a = a.copy()
    a["_tiebreak"] = rng_tiebreak.random(len(a))
    a_sorted = a.sort_values(["p_joincount", "join_count_obs", "_tiebreak"], ascending=[True, False, True])
    a_matched = a_sorted.head(S_C).drop(columns="_tiebreak")
    _write(a_matched, section, "methodA_matched", sel_dir, manifest_rows)

    # DIAGNOSTIC STRATUM, NOT A CALLSET, NOT A SIGNIFICANCE THRESHOLD (see module
    # docstring and RESULTS.md): methodA_matched mixes this stratum with arbitrary
    # same-p==1 tie-break padding once it's exhausted. p<1 only means "beat at
    # least one of 999 permutations" -- for a low-count variant that is a weak bar.
    a_diag = a[a["p_joincount"] < 1].sort_values(["p_joincount", "join_count_obs"], ascending=[True, False])
    _write(a_diag, section, "methodA_diagnostic_plt1", sel_dir, manifest_rows)

    # Circularity control (coordinator request 2026-09-03): a size-matched RANDOM
    # sample drawn only from the same structural precondition (n_pos_spots>=2) that
    # made a_diag possible in the first place. If this baseline's molecular support
    # is also ~100%, the diagnostic stratum's number is a spot-count artifact, not
    # a Method A finding -- see endpoint reporting.
    testable = a[a["n_pos_spots"] >= 2]
    rng = np.random.default_rng(RANDOM_BASELINE_SEED)
    n_draw = min(len(a_diag), len(testable))
    baseline_idx = rng.choice(testable.index.values, size=n_draw, replace=False)
    random_baseline = testable.loc[baseline_idx].sort_values("variant")
    _write(random_baseline, section, "random_multispot_baseline", sel_dir, manifest_rows)

    if os.path.exists(b_path):
        b = pd.read_csv(b_path, sep="\t")
        b_sorted = b.sort_values(["gae_score", "variant"], ascending=[False, True])
        b_matched = b_sorted.head(S_C)
        _write(b_matched, section, "methodB_matched", sel_dir, manifest_rows)
    else:
        print(f"{section}: WARNING methodB scores not found at {b_path} -- skipping methodB_matched "
              f"(run method_b_gae.py under conda env snv_clustering first)")

    manifest_path = os.path.join(sel_dir, "manifest.csv")
    new_df = pd.DataFrame(manifest_rows)
    if os.path.exists(manifest_path):
        prev = pd.read_csv(manifest_path)
        prev = prev[~((prev["section"] == section) & (prev["method_variant"].isin(new_df["method_variant"])))]
        new_df = pd.concat([prev, new_df], ignore_index=True)
    new_df.to_csv(manifest_path, index=False)
    print(f"{section}: manifest updated -> {manifest_path}")
    return new_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--section", required=True, choices=list(common.SECTIONS.keys()))
    args = ap.parse_args()
    run_section(args.section)
