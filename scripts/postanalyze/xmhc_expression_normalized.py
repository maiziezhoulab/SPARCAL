#!/usr/bin/env python
"""Does the xMHC call excess survive normalising for gene expression?

The standing alternative explanation for xMHC enrichment is plain coverage:
HLA genes are among the most highly expressed loci in any tissue, more reads
means more chances to call a variant, so an excess of calls there could be
entirely a depth effect with no MHC-specific mechanism.

This script tests that directly.  Every SPARCAL somatic-class call is assigned
to a gene footprint; every gene's total UMI count is read from the SpaceRanger
filtered matrix.  The endpoint is CALLS PER 1000 UMI per gene.  Genes are then
binned by expression decile and xMHC genes are compared to non-xMHC genes IN
THE SAME DECILE -- so the comparison is between loci of similar depth.

Reading the result:
  * xMHC rate ~= matched non-xMHC rate  -> the excess is a pure expression /
    coverage effect; nothing MHC-specific, and it says nothing about somatic
    biology either way.
  * xMHC rate >> matched                -> something MHC-SPECIFIC is happening
    (either genuine clustering of variation, or mismapping peculiar to the MHC);
    expression alone does not explain it.
  * xMHC rate << matched                -> calls are depleted there once depth
    is accounted for.

This is a confound test, NOT an artifact test: an MHC-specific excess is
consistent with BOTH real HLA variation and reference-divergence mismapping.
Discriminating those needs the read-level panels (RESULTS.md sec 4-10) or HLA
typing.

Read-only; writes only under --out-dir.

Run (env snv_caller):
    python scripts/postanalyze/xmhc_expression_normalized.py \
        --out-dir data/xmhc_artifact_mechanism_2026-08-31
"""
from __future__ import annotations

import argparse
import bisect
import collections
import csv
import os
import statistics as st

import h5py
import numpy as np

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
STD = "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets"
LO, HI = 28_000_000, 34_000_000
RES = f"{REPO}/SpaceTracer/resources-v220"

SAMPLES = {
    "P4": dict(build="hg19",
               h5=f"{STD}/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/filtered_feature_bc_matrix.h5",
               calls=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt"),
    "P6": dict(build="hg19",
               h5=f"{STD}/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix.h5",
               calls=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt"),
    "DCIS1": dict(build="hg38",
                  h5=f"{STD}/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/filtered_feature_bc_matrix.h5",
                  calls=f"{REPO}/data/dcis1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt"),
    "DCIS2": dict(build="hg38",
                  h5=f"{STD}/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/filtered_feature_bc_matrix.h5",
                  calls=f"{REPO}/data/dcis2/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt"),
}
BED = {"hg19": f"{RES}/hg19/gene_region.bed", "hg38": f"{RES}/hg38-nochr/gene_region.bed"}
N_DECILES = 10


def load_bed(path):
    """chrom -> (sorted starts, [(start,end,name)])"""
    iv = collections.defaultdict(list)
    for line in open(path):
        q = line.split()
        if len(q) < 4:
            continue
        c = q[0][3:] if q[0].startswith("chr") else q[0]
        try:
            iv[c].append((int(q[1]), int(q[2]), q[3]))
        except ValueError:
            continue
    for c in iv:
        iv[c].sort()
    return {c: ([s for s, _, _ in v], v) for c, v in iv.items()}


def genes_at(bed, c, p):
    e = bed.get(c)
    if not e:
        return []
    starts, v = e
    out = []
    i = bisect.bisect_right(starts, p) - 1
    while i >= 0 and i > bisect.bisect_right(starts, p) - 60:
        if i < len(v) and v[i][0] <= p < v[i][1]:
            out.append(v[i][2])
        i -= 1
    return out


def gene_is_xmhc(bed, name):
    for c, (starts, v) in bed.items():
        if c != "6":
            continue
        for s, e, n in v:
            if n == name:
                return LO <= s <= HI or LO <= e <= HI
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    beds = {b: load_bed(p) for b, p in BED.items() if os.path.exists(p)}
    rows, per_gene_rows = [], []

    for samp, cfg in SAMPLES.items():
        if not (os.path.exists(cfg["h5"]) and os.path.exists(cfg["calls"])):
            print(f"{samp}: SKIPPED (missing input)"); continue
        bed = beds[cfg["build"]]

        # 10x .h5 is CSC: shape=(n_features, n_barcodes), indices index features.
        # Total UMI per feature = sum of data grouped by feature index.
        with h5py.File(cfg["h5"], "r") as h:
            g = h["matrix"]
            n_feat = int(g["shape"][0])
            data = g["data"][:]
            idx = g["indices"][:]
            names = [n.decode() for n in g["features"]["name"][:]]
        tot = np.bincount(idx, weights=data.astype(float), minlength=n_feat)
        expr = collections.defaultdict(float)
        for gname, u in zip(names, tot):
            expr[gname] += float(u)

        ncalls = collections.Counter()
        n_tot = n_nogene = 0
        with open(cfg["calls"]) as fh:
            next(fh)
            for line in fh:
                q = line.rstrip("\n").split("\t")
                if len(q) < 4:
                    continue
                n_tot += 1
                gs = genes_at(bed, q[0], int(q[1]))
                if not gs:
                    n_nogene += 1
                    continue
                for g in set(gs):
                    ncalls[g] += 1

        # genes with any expression AND present in the bed footprint universe
        universe = [g for g in expr if expr[g] > 0]
        universe = [g for g in universe if ncalls.get(g, 0) > 0 or expr[g] > 0]
        xset = {g for g in universe if gene_is_xmhc(bed, g)}
        if not xset:
            print(f"{samp}: no xMHC genes matched between bed and expression matrix"); continue

        vals = sorted(expr[g] for g in universe)
        cuts = [vals[int(len(vals) * k / N_DECILES)] for k in range(1, N_DECILES)]

        def dec(g):
            return bisect.bisect_left(cuts, expr[g])

        agg = collections.defaultdict(lambda: dict(x_c=0, x_u=0.0, x_n=0,
                                                   o_c=0, o_u=0.0, o_n=0))
        for g in universe:
            d = agg[dec(g)]
            if g in xset:
                d["x_c"] += ncalls.get(g, 0); d["x_u"] += expr[g]; d["x_n"] += 1
            else:
                d["o_c"] += ncalls.get(g, 0); d["o_u"] += expr[g]; d["o_n"] += 1

        print(f"\n===== {samp} ({cfg['build']}) =====")
        print(f"  {n_tot} somatic calls, {n_nogene} ({100*n_nogene/n_tot:.1f}%) outside any gene footprint")
        print(f"  {len(xset)} xMHC genes vs {len(universe)-len(xset)} other expressed genes")
        print(f"  {'decile':7}{'xMHC genes':>12}{'calls/1k UMI':>14}{'other genes':>13}{'calls/1k UMI':>14}{'ratio':>8}")
        ratios = []
        for d in sorted(agg):
            v = agg[d]
            if v["x_n"] == 0 or v["o_u"] == 0:
                continue
            xr = 1000 * v["x_c"] / v["x_u"] if v["x_u"] else 0
            orr = 1000 * v["o_c"] / v["o_u"]
            r = xr / orr if orr else float("nan")
            if v["x_n"] >= 2 and v["x_c"] + v["o_c"] > 0:
                ratios.append(r)
            print(f"  {d+1:<7}{v['x_n']:>12}{xr:>14.4f}{v['o_n']:>13}{orr:>14.4f}{r:>8.2f}")
            per_gene_rows.append(dict(sample=samp, decile=d+1, n_xmhc_genes=v["x_n"],
                                      xmhc_calls=v["x_c"], xmhc_umi=round(v["x_u"], 1),
                                      xmhc_rate=round(xr, 5), n_other_genes=v["o_n"],
                                      other_calls=v["o_c"], other_umi=round(v["o_u"], 1),
                                      other_rate=round(orr, 5), ratio=round(r, 4)))
        # pooled, expression-matched
        tx = sum(agg[d]["x_c"] for d in agg); ux = sum(agg[d]["x_u"] for d in agg)
        to = sum(agg[d]["o_c"] for d in agg); uo = sum(agg[d]["o_u"] for d in agg)
        rx = 1000 * tx / ux if ux else 0
        ro = 1000 * to / uo if uo else 0
        med = st.median(ratios) if ratios else float("nan")
        print(f"  POOLED  xMHC {rx:.4f} calls/1k UMI   other {ro:.4f}   "
              f"raw ratio {rx/ro if ro else 0:.2f}   median within-decile ratio {med:.2f}")
        rows.append(dict(sample=samp, build=cfg["build"], n_calls=n_tot,
                         pct_outside_genes=round(100*n_nogene/n_tot, 2),
                         n_xmhc_genes=len(xset), xmhc_calls=tx,
                         xmhc_rate_per_1k_umi=round(rx, 5),
                         other_rate_per_1k_umi=round(ro, 5),
                         raw_ratio=round(rx/ro, 4) if ro else "",
                         median_within_decile_ratio=round(med, 4)))

    for name, rr in (("11_xmhc_expression_normalized.csv", rows),
                     ("11b_xmhc_expression_by_decile.csv", per_gene_rows)):
        p = os.path.join(a.out_dir, name)
        with open(p, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0].keys()))
            w.writeheader(); w.writerows(rr)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
