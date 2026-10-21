#!/usr/bin/env python
"""Cross-caller xMHC concentration, matched on spot support and set size.

Supersedes the raw fold_vs_genome column in
data/xmhc_artifact_mechanism_2026-08-31/cross_caller_xmhc.csv, which divides
every caller's xMHC share by the genome-length expectation (0.194%).  That
denominator is wrong for RNA callers in general and badly wrong for SpaceTracer
in particular, which only interrogates a fixed 24.9 Mb "visium callable region"
panel -- xMHC is 0.736% of that panel, ~4x its genome-length share.

Three tables:

  A  Per caller x sample, xMHC share of calls carried by >= MIN_SPOTS in-tissue
     spots -- one identical filter for every tool.  Spot support = column sum of
     the tool's binary presence matrix; for SpaceTracer, the NUM_MUT_SPOTS INFO
     field of its PASS VCF.
  B  The same filter's EFFECT: xMHC share before vs after, per caller x sample.
     Tests whether a spot-support floor purges xMHC (artifact hypothesis
     predicts it does not) or concentrates it.
  C  SPARCAL somatic restricted to SpaceTracer's own callable panel and
     subsetted, by descending spot support, to SpaceTracer's set sizes -- the
     size-matched comparison.  Fold is vs the panel share (0.7364%), not genome
     length.

CAVEATS (do not quote a number from here without them):
  * P4/P6 are GRCh37, DCIS1/DCIS2 GRCh38.  chr6:28-34 Mb is NOT the same
    interval in both builds (referee item m5, open).  Within-sample comparisons
    are sound; cross-build comparisons are approximate.
  * fold_vs_genome is retained ONLY for continuity with the older table.  Each
    caller interrogates a different space, so folds are not comparable ACROSS
    callers; the ranking WITHIN a sample is what carries meaning.
  * Table C's panel is hg38.  The hg19 "callable" bed shipped with SpaceTracer
    v2.2.0 is a different, much larger (156 Mb) construction, so C is DCIS-only.
  * These are artifact-rate proxies, not accuracy.  Higher xMHC share means more
    reference-divergence-like calls; see RESULTS.md sections 4-10 for the
    read-level evidence establishing that direction.

Read-only; writes only under --out-dir.

Run (env snv_caller):
    python scripts/postanalyze/xmhc_cross_caller_matched.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import os
import pickle
import re
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
BENCH = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/monopogen"
LO, HI = 28_000_000, 34_000_000
GENOME_PCT = 100 * (HI - LO) / 3.1e9
PANEL = f"{REPO}/SpaceTracer/resources-v220/hg38-nochr/visium_callable_region.bed"
PANEL_ALT = f"{REPO}/SpaceTracer/resources-v220/hg38/visium_callable_region.bed"
MIN_SPOTS = 30

MATRICES = {
    ("SPARCAL somatic", "P4"):    f"{REPO}/data/P4_tumor/1/matrix/P4_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    ("SPARCAL somatic", "P6"):    f"{REPO}/data/P6_tumor/1/matrix/P6_TUMOR_1_SPARCAL_somatic_matrix.pkl",
    ("SPARCAL somatic", "DCIS1"): f"{REPO}/data/dcis1/matrix/DCIS_dcis1_SPARCAL_somatic_matrix.pkl",
    ("SPARCAL somatic", "DCIS2"): f"{REPO}/data/dcis2/matrix/DCIS_dcis2_SPARCAL_somatic_matrix.pkl",
    ("SpatialSNV", "P4"):    f"{REPO}/SpatialSNV/results/p4/matrix/p4_spatialsnv_presence_matrix.pkl",
    ("SpatialSNV", "P6"):    f"{REPO}/SpatialSNV/results/p6/matrix/p6_spatialsnv_presence_matrix.pkl",
    ("SpatialSNV", "DCIS1"): f"{REPO}/SpatialSNV/results/dcis1/matrix/dcis1_spatialsnv_presence_matrix.pkl",
    ("SpatialSNV", "DCIS2"): f"{REPO}/SpatialSNV/results/dcis2/matrix/dcis2_spatialsnv_presence_matrix.pkl",
    ("Monopogen somatic", "P6"):    f"{BENCH}/P6_rep1_som/P6_rep1_som_monopogen_somatic_presence_matrix.pkl",
    ("Monopogen somatic", "DCIS1"): f"{BENCH}/dcis1/dcis1_monopogen_somatic_presence_matrix.pkl",
    ("Monopogen somatic", "DCIS2"): f"{BENCH}/dcis2/dcis2_monopogen_somatic_presence_matrix.pkl",
}
SPACETRACER = {
    s: (f"{REPO}/SpaceTracer/results-v220/{s}/mutation_prediction/results/"
        f"{s}_spatial_free_model_total_pred_truesites_PASS.vcf")
    for s in ("DCIS1", "DCIS2")
}
UNAVAILABLE = {
    ("Monopogen somatic", "P4"): "cellScan cell_snv.mat.gz not retained when the run tree was copied",
    ("SpaceTracer PASS", "P4"): "v2.2.0 run stopped at umi_combine",
    ("SpaceTracer PASS", "P6"): "v2.2.0 run stopped at umi_combine",
}
TOOLS = ["SPARCAL somatic", "SpatialSNV", "Monopogen somatic", "SpaceTracer PASS"]
SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]


def load_panel():
    path = PANEL if os.path.exists(PANEL) else PANEL_ALT
    iv = collections.defaultdict(list)
    total = xm = 0
    with open(path) as fh:
        for line in fh:
            q = line.split()
            if len(q) < 3:
                continue
            c = q[0][3:] if q[0].startswith("chr") else q[0]
            s, e = int(q[1]), int(q[2])
            iv[c].append((s, e))
            total += e - s
            if c == "6" and s >= LO and e <= HI:
                xm += e - s
    for c in iv:
        iv[c].sort()
    starts = {c: [s for s, _ in v] for c, v in iv.items()}
    return iv, starts, 100 * xm / total, path


def in_panel(iv, starts, c, p):
    v = iv.get(c)
    if not v:
        return False
    i = bisect.bisect_right(starts[c], p) - 1
    return i >= 0 and v[i][0] <= p < v[i][1]


def sites_from_matrix(path):
    M = pickle.load(open(path, "rb"))
    ns = np.asarray(M.values).sum(axis=0)
    out = []
    for j, col in enumerate(M.columns):
        m = re.match(r"^(?:chr)?(\w+)[_:](\d+)", str(col))
        if m:
            out.append((int(ns[j]), m.group(1), int(m.group(2))))
    return out


def sites_from_spacetracer(path):
    out = []
    for line in open(path):
        if line.startswith("#"):
            continue
        q = line.rstrip("\n").split("\t")
        d = dict(kv.split("=") for kv in q[7].split(";"))
        out.append((int(d["NUM_MUT_SPOTS"]), q[0].lstrip("chr"), int(q[1])))
    return out


def tally(sites):
    n = len(sites)
    x = sum(1 for _, c, p in sites if c == "6" and LO <= p <= HI)
    pct = 100 * x / n if n else 0.0
    return n, x, pct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-spots", type=int, default=MIN_SPOTS)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    iv, starts, panel_xmhc_pct, panel_path = load_panel()

    print(f"xMHC = chr6:{LO:,}-{HI:,}   genome-length expectation {GENOME_PCT:.4f}%")
    print(f"callable panel: {os.path.relpath(panel_path, REPO)}  "
          f"xMHC = {panel_xmhc_pct:.4f}% of it\n")

    A, B = [], []
    cache = {}
    print("TABLE A/B -- identical >=%d-spot filter, all tools" % a.min_spots)
    hdr = (f"{'tool':19}{'sample':7}{'build':7}{'all':>9}{'>=%d' % a.min_spots:>8}"
           f"{'xMHC':>6}{'xMHC%':>8}{'fold/gen':>10}{'before%':>9}{'shift':>7}")
    print(hdr); print("-" * len(hdr))
    for tool in TOOLS:
        for samp in SAMPLES:
            build = "GRCh37" if samp in ("P4", "P6") else "GRCh38"
            if (tool, samp) in UNAVAILABLE:
                print(f"{tool:19}{samp:7}{build:7}{'--':>9}{'--':>8}{'--':>6}"
                      f"{'--':>8}{'--':>10}{'--':>9}{'--':>7}   {UNAVAILABLE[(tool, samp)]}")
                A.append(dict(tool=tool, sample=samp, build=build, n_all="", n_kept="",
                              n_xmhc="", pct_xmhc="", fold_vs_genome="",
                              pct_xmhc_before="", concentration_shift="",
                              note=UNAVAILABLE[(tool, samp)]))
                continue
            if tool == "SpaceTracer PASS":
                sites = sites_from_spacetracer(SPACETRACER[samp])
            else:
                p = MATRICES[(tool, samp)]
                if not os.path.exists(p):
                    print(f"{tool:19}{samp:7}{build:7}  MISSING {p}")
                    continue
                sites = sites_from_matrix(p)
            cache[(tool, samp)] = sites
            n0, x0, pct0 = tally(sites)
            kept = [s for s in sites if s[0] >= a.min_spots]
            n1, x1, pct1 = tally(kept)
            fold = pct1 / GENOME_PCT if n1 else 0.0
            shift = pct1 / pct0 if pct0 else float("nan")
            print(f"{tool:19}{samp:7}{build:7}{n0:>9}{n1:>8}{x1:>6}{pct1:>7.3f}%"
                  f"{fold:>10.2f}{pct0:>8.3f}%{shift:>6.2f}x")
            A.append(dict(tool=tool, sample=samp, build=build, n_all=n0, n_kept=n1,
                          n_xmhc=x1, pct_xmhc=round(pct1, 4),
                          fold_vs_genome=round(fold, 3),
                          pct_xmhc_before=round(pct0, 4),
                          concentration_shift=round(shift, 3), note=""))
            B.append(dict(tool=tool, sample=samp, pct_before=round(pct0, 4),
                          pct_after=round(pct1, 4), shift=round(shift, 3)))

    print("\nTABLE C -- SPARCAL somatic inside SpaceTracer's callable panel, "
          "size-matched (fold vs panel share %.4f%%)" % panel_xmhc_pct)
    C = []
    for samp in ("DCIS1", "DCIS2"):
        st_sites = cache.get(("SpaceTracer PASS", samp), [])
        st_kept = [s for s in st_sites if s[0] >= a.min_spots]
        sp = [s for s in cache[("SPARCAL somatic", samp)]
              if in_panel(iv, starts, s[1], s[2])]
        sp.sort(key=lambda t: -t[0])
        rows = [("SpaceTracer PASS, all", st_sites),
                (f"SpaceTracer PASS, >={a.min_spots} spots", st_kept),
                ("SPARCAL, all in-panel", sp),
                (f"SPARCAL, >={a.min_spots} spots", [s for s in sp if s[0] >= a.min_spots]),
                (f"SPARCAL, top {len(st_sites)} by spots", sp[:len(st_sites)])]
        print(f"  --- {samp} ---")
        for label, ss in rows:
            n, x, pct = tally(ss)
            if not n:
                continue
            print(f"    {label:38s} n={n:>6}  xMHC={x:>4}  {pct:6.3f}%  "
                  f"fold={pct / panel_xmhc_pct:6.2f}")
            C.append(dict(sample=samp, subset=label, n=n, n_xmhc=x,
                          pct_xmhc=round(pct, 4),
                          fold_vs_panel=round(pct / panel_xmhc_pct, 3)))

    for name, rows in (("08_xmhc_spotmatched_cross_caller.csv", A),
                       ("08b_xmhc_filter_effect.csv", B),
                       ("08c_xmhc_size_matched_panel.csv", C)):
        p = os.path.join(a.out_dir, name)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
