#!/usr/bin/env python
"""Step 4a -- SCOPE the clone-specific LOH vignette before building it.

The point of this script is to decide WHERE the test can actually be run, using
depth, rather than picking a pretty CNV region first and discovering afterwards
that nothing in it is measurable.  Two independent constraints have to be
satisfied simultaneously by the chosen segment:

  (1) IT MUST REALLY BE A CLONE-SPECIFIC LOSS.  copyKAT gives a per-spot,
      per-bin relative expression log-ratio, plus an aneuploid/diploid call per
      spot.  A candidate segment is one where the aneuploid spots sit clearly
      below the diploid spots -- i.e. the loss is clone-specific, not global.
      (Verified in axis2_cnv_consensus.py: inferCNV and siCNV are 100% identical
      to copyKAT here because Weiman's pipeline seeds them with copyKAT labels,
      so "5-tool consensus" is copyKAT and nothing is gained by re-deriving it.)

  (2) IT MUST CONTAIN ENOUGH TESTABLE VARIANTS.  This is the binding constraint,
      not (1).  61-82% of somatic candidates are a single alt UMI in a single
      spot, where BAF is 1.0 BY CONSTRUCTION and carries no information.  The
      allelic test only runs on positions with enough aggregate depth to make
      BAF non-degenerate, so segments are ranked by the count of variants at
      DP >= --min-dp, broken down by method group.

WHY copyKAT LOSS IS USED AS THE LOH PROXY, and what that costs.  copyKAT is
expression-based and returns TOTAL relative copy number, not allele-specific
copy number, so it cannot state min(A,B)==0 directly.  A clone-specific loss
segment is the available proxy for LOH.  The BAF test in step 4b is what
actually supplies allelic evidence, and it is measured independently of copyKAT
-- so copyKAT selects the region, and the allelic readout is not circular with
respect to it.  Germline heterozygous sites (1KGP-defined, which SPARCAL routes
out of the somatic cascade entirely) provide the internal positive control:
inside a genuine LOH segment their BAF must shift away from 0.5 in aneuploid
spots while staying near 0.5 in diploid spots.  If that control fails, the
segment is not LOH regardless of what copyKAT says, and step 4b must not run
on it.

BUILD RESTRICTION -- READ BEFORE ADDING P4/P6.  Only DCIS1/DCIS2 are scoped by
default.  Weiman's copyKAT run for P4/P6 lives in `P4_rep1_hg38` / `P6_rep1_hg38`
and is on GRCh38 coordinates, while every P4/P6 variant callset in this project
is hg19.  Joining those bins to hg19 positions would silently mis-assign
variants to segments.  P4/P6 need a liftover before they can be scoped, and the
script refuses them rather than producing a plausible-looking wrong answer.

Outputs (under data/loh_vignette_scoping_<date>/):
  loh_bins.csv          per copyKAT bin: aneuploid vs diploid log-ratio + delta
  loh_segments.csv      merged candidate loss segments, ranked by testable n
  loh_segment_variants.csv  per segment x group: variant counts by DP threshold

Run (env snv_caller, ~2 min):
  python scripts/postanalyze/loh_vignette_scoping.py --min-dp 10
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
CK = "/data/maiziezhou_lab/Weiman/ST_CNV/copyKAT"

# sample -> (copyKAT dir, CNA filename).  The CNA filename is given explicitly
# because DCIS_2/ also contains a stray P6_Visium_rep1_copykat_CNA_results.txt;
# globbing for "*_CNA_results.txt" in that directory picks up the wrong sample.
SAMPLES = {
    "DCIS1": ("DCIS_1", "DCIS1_copykat_CNA_results.txt"),
    "DCIS2": ("DCIS_2", "DCIS2_copykat_CNA_results.txt"),
}

CONCORDANCE_DIR = os.path.join(REPO, "data", "spatialsnv_concordance_2026-08-06")
GROUPS = ["concordant", "sparcal_private", "spatialsnv_private"]


def load_copykat(sample: str):
    ckdir, cna_name = SAMPLES[sample]
    pred = {}
    with open(os.path.join(CK, ckdir, "prediction_aligned.txt")) as fh:
        next(fh)
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1] in ("aneuploid", "diploid"):
                pred[p[0]] = p[1]

    cna = pd.read_csv(os.path.join(CK, ckdir, cna_name), sep="\t")
    # copyKAT writes barcodes with '.' where the matrix had '-'
    colmap = {c: c.replace(".", "-") for c in cna.columns[3:]}
    cna = cna.rename(columns=colmap)

    aneu = [c for c in cna.columns[3:] if pred.get(c) == "aneuploid"]
    dip = [c for c in cna.columns[3:] if pred.get(c) == "diploid"]
    return cna, aneu, dip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="DCIS1,DCIS2")
    ap.add_argument("--min-dp", type=float, default=10.0,
                    help="aggregate DP at which BAF stops being degenerate")
    ap.add_argument("--delta-quantile", type=float, default=0.10,
                    help="bins below this quantile of (aneuploid-diploid) are loss candidates")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    names = [s.strip() for s in args.samples.split(",") if s.strip()]
    for n in names:
        if n not in SAMPLES:
            raise SystemExit(
                f"{n} is not scopeable: copyKAT for P4/P6 is on GRCh38 while their "
                f"variants are hg19. Lift over before scoping -- see module docstring.")

    outdir = args.outdir or os.path.join(
        REPO, f"data/loh_vignette_scoping_{date.today().isoformat()}")
    os.makedirs(outdir, exist_ok=True)

    variants = pd.read_csv(os.path.join(CONCORDANCE_DIR, "concordance_variant_level.csv.gz"))
    variants["chrom"] = variants["chrom"].astype(str)

    all_bins, all_segs, all_segvars = [], [], []

    for sample in names:
        cna, aneu, dip = load_copykat(sample)
        print(f"[{sample}] copyKAT bins={len(cna):,}  aneuploid={len(aneu)}  diploid={len(dip)}")

        b = cna[["chrom", "chrompos"]].copy()
        b["chrom"] = b["chrom"].astype(str)
        b["sample"] = sample
        b["mean_aneuploid"] = cna[aneu].mean(axis=1)
        b["mean_diploid"] = cna[dip].mean(axis=1)
        b["delta"] = b["mean_aneuploid"] - b["mean_diploid"]
        # Welch t on the two spot groups, per bin -- is the clone difference real?
        va, vd = cna[aneu].values, cna[dip].values
        se = np.sqrt(va.var(axis=1, ddof=1) / len(aneu) + vd.var(axis=1, ddof=1) / len(dip))
        b["t_stat"] = b["delta"].values / np.where(se > 0, se, np.nan)
        all_bins.append(b)

        # ---- candidate loss bins -> merged segments -----------------------
        cut = b["delta"].quantile(args.delta_quantile)
        b["is_loss"] = (b["delta"] <= cut) & (b["t_stat"] <= -2)
        print(f"[{sample}] loss-bin threshold delta<={cut:.4f}  n_loss={int(b['is_loss'].sum()):,}")

        seg_id, segs = 0, []
        for chrom, grp in b.groupby("chrom", sort=False):
            grp = grp.sort_values("chrompos")
            run = None
            for _, row in grp.iterrows():
                if row["is_loss"]:
                    if run is None:
                        run = dict(sample=sample, chrom=chrom, start=row["chrompos"],
                                   end=row["chrompos"], n_bins=1,
                                   delta_sum=row["delta"], t_sum=row["t_stat"])
                    else:
                        run["end"] = row["chrompos"]
                        run["n_bins"] += 1
                        run["delta_sum"] += row["delta"]
                        run["t_sum"] += row["t_stat"]
                else:
                    if run is not None and run["n_bins"] >= 3:
                        segs.append(run)
                    run = None
            if run is not None and run["n_bins"] >= 3:
                segs.append(run)

        for s in segs:
            seg_id += 1
            s["segment"] = f"{sample}_seg{seg_id:03d}"
            s["mean_delta"] = s.pop("delta_sum") / s["n_bins"]
            s["mean_t"] = s.pop("t_sum") / s["n_bins"]
            s["span_mb"] = (s["end"] - s["start"]) / 1e6

        # ---- how many testable variants does each segment hold? -----------
        sv = variants[variants["sample"] == sample]
        for s in segs:
            sub = sv[(sv["chrom"] == s["chrom"]) &
                     (sv["pos"] >= s["start"]) & (sv["pos"] <= s["end"])]
            deep = sub[sub["DP"] >= args.min_dp]
            s["n_variants"] = len(sub)
            s["n_testable"] = len(deep)
            for g in GROUPS:
                s[f"n_testable_{g}"] = int((deep["group"] == g).sum())
                all_segvars.append(dict(
                    sample=sample, segment=s["segment"], group=g,
                    n_all=int((sub["group"] == g).sum()),
                    n_dp_ge_min=int((deep["group"] == g).sum()),
                    median_DP=float(sub[sub["group"] == g]["DP"].median())
                    if (sub["group"] == g).any() else "",
                ))
        all_segs.extend(segs)

    bins_df = pd.concat(all_bins, ignore_index=True)
    segs_df = pd.DataFrame(all_segs).sort_values("n_testable", ascending=False)
    segvars_df = pd.DataFrame(all_segvars)

    cols = ["segment", "sample", "chrom", "start", "end", "span_mb", "n_bins",
            "mean_delta", "mean_t", "n_variants", "n_testable"] + \
           [f"n_testable_{g}" for g in GROUPS]
    segs_df = segs_df[cols]

    bins_df.to_csv(os.path.join(outdir, "loh_bins.csv"), index=False)
    segs_df.to_csv(os.path.join(outdir, "loh_segments.csv"), index=False)
    segvars_df.to_csv(os.path.join(outdir, "loh_segment_variants.csv"), index=False)

    print(f"\nTop candidate segments by testable variant count (DP >= {args.min_dp:g}):")
    print(segs_df.head(12).to_string(index=False))
    print(f"\nWrote 3 CSVs to {outdir}")


if __name__ == "__main__":
    main()
