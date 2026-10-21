#!/usr/bin/env python
"""Do xMHC somatic-class calls recur across UNRELATED patients more than chance?

Independent somatic mutations in different individuals should almost never land
on the identical chrom:pos:ref>alt.  Systematic reference/alignment error does,
because it is a property of the reference and the assay, not the patient.  So a
high cross-patient recurrence rate is an artifact signature.

The obvious confound is call DENSITY: xMHC carries far more calls per Mb than
the genome average, so it gets more coincidental overlap.  This script controls
for that by binning the genome into fixed windows, computing each window's
recurrence rate, and locating the xMHC windows within the distribution of
windows of COMPARABLE call density -- not against the genome-wide average.

Pairs compared (same build, different patients):
    P4 vs P6      (hg19, two cSCC patients)
    DCIS1 vs DCIS2 (hg38, two sections; NOTE these may be the same patient --
                    treated as a separate, weaker line and labelled as such)

Read-only; writes only under --out-dir.

Run (env snv_caller):
    python scripts/postanalyze/xmhc_recurrence_check.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31
"""
from __future__ import annotations

import argparse
import collections
import csv
import os
import statistics as st

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
LO, HI = 28_000_000, 34_000_000
WIN = 6_000_000          # window size == xMHC width, so density is comparable
CALLS = {
    "P4":    f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
    "P6":    f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
    "DCIS1": f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
    "DCIS2": f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt",
}
PAIRS = [("P4", "P6", "different patients (cSCC), hg19"),
         ("DCIS1", "DCIS2", "two DCIS sections, hg38 -- may share a patient, weaker evidence")]


def load(path):
    out = set()
    with open(path) as fh:
        next(fh)
        for line in fh:
            q = line.rstrip("\n").split("\t")
            if len(q) >= 4:
                out.add((q[0], int(q[1]), q[2].upper(), q[3].upper()))
    return out


def win_of(c, p):
    return (c, p // WIN)


def is_xmhc_win(w):
    return w[0] == "6" and LO // WIN <= w[1] <= HI // WIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-window-calls", type=int, default=20,
                    help="windows with fewer calls than this in either sample are "
                         "too sparse for a stable recurrence rate")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rows = []

    for s1, s2, note in PAIRS:
        if not (os.path.exists(CALLS[s1]) and os.path.exists(CALLS[s2])):
            print(f"{s1} vs {s2}: SKIPPED (missing call file)")
            continue
        A, B = load(CALLS[s1]), load(CALLS[s2])
        print(f"\n===== {s1} vs {s2} =====  ({note})")
        print(f"  {s1}: {len(A)} somatic calls   {s2}: {len(B)} somatic calls")

        # allele-exact recurrence, whole callset, xMHC vs rest
        def strat(S, xm):
            return {v for v in S if (v[0] == "6" and LO <= v[1] <= HI) == xm}
        for label, xm in (("xMHC", True), ("non-xMHC", False)):
            a1, b1 = strat(A, xm), strat(B, xm)
            sh = len(a1 & b1)
            r1 = 100 * sh / len(a1) if a1 else 0
            r2 = 100 * sh / len(b1) if b1 else 0
            print(f"  {label:9s} {s1} n={len(a1):>6}  {s2} n={len(b1):>6}  "
                  f"shared={sh:>5}  ({r1:5.2f}% of {s1}, {r2:5.2f}% of {s2})")
            rows.append(dict(pair=f"{s1}_vs_{s2}", stratum=label, n_a=len(a1),
                             n_b=len(b1), n_shared=sh, pct_of_a=round(r1, 3),
                             pct_of_b=round(r2, 3), window="", note=note))

        # density-controlled: per-window recurrence, xMHC windows vs comparable ones
        wa = collections.Counter(win_of(c, p) for c, p, _, _ in A)
        wb = collections.Counter(win_of(c, p) for c, p, _, _ in B)
        shared_w = collections.Counter(win_of(c, p) for c, p, _, _ in (A & B))
        recs, xrecs = [], []
        for w in set(wa) & set(wb):
            na, nb = wa[w], wb[w]
            if min(na, nb) < a.min_window_calls:
                continue
            r = 100 * shared_w.get(w, 0) / min(na, nb)
            (xrecs if is_xmhc_win(w) else recs).append((w, na, nb, r))
        if recs:
            allr = [r for *_, r in recs]
            med, p90 = st.median(allr), sorted(allr)[int(0.9 * len(allr))]
            print(f"\n  density-controlled ({WIN//10**6} Mb windows, "
                  f">={a.min_window_calls} calls in both):")
            print(f"    non-xMHC windows: n={len(recs)}  median recurrence={med:5.2f}%  "
                  f"90th pct={p90:5.2f}%  max={max(allr):5.2f}%")
            for w, na, nb, r in sorted(xrecs):
                pctile = 100 * sum(1 for x in allr if x < r) / len(allr)
                print(f"    xMHC window {w[0]}:{w[1]*WIN//10**6}-{(w[1]+1)*WIN//10**6} Mb  "
                      f"{s1} n={na:>5} {s2} n={nb:>5}  recurrence={r:5.2f}%  "
                      f"-> {pctile:.1f}th percentile of all windows")
                rows.append(dict(pair=f"{s1}_vs_{s2}", stratum="xMHC_window",
                                 n_a=na, n_b=nb, n_shared=shared_w.get(w, 0),
                                 pct_of_a=round(r, 3), pct_of_b="",
                                 window=f"{w[0]}:{w[1]*WIN}-{(w[1]+1)*WIN}",
                                 note=f"percentile_vs_other_windows={pctile:.1f}; "
                                      f"nonxMHC_median={med:.3f}"))
            rows.append(dict(pair=f"{s1}_vs_{s2}", stratum="nonxMHC_windows_summary",
                             n_a=len(recs), n_b="", n_shared="",
                             pct_of_a=round(med, 3), pct_of_b=round(p90, 3),
                             window=f"{WIN}bp", note="pct_of_a=median, pct_of_b=90th pct"))

    p = os.path.join(a.out_dir, "09_xmhc_cross_patient_recurrence.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
