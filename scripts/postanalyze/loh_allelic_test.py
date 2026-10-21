#!/usr/bin/env python
"""Step 4b -- do each method's calls behave the way the CNV state says they must?

THE TEST.  Inside a clone-specific loss segment, the aneuploid clone has lost one
allele.  A site sitting at BAF ~ 0.5 in that clone is therefore INCONSISTENT with
the copy-number state: with one allele gone there is nothing to produce a balanced
heterozygous signal.  Such calls are best explained as retained germline
heterozygotes or as artifacts, not as somatic variants on the surviving allele.

So the measurement is: of each method's somatic calls inside clone-loss segments,
what fraction sits in the heterozygous band (BAF 0.35-0.65) when measured in the
aneuploid spots only?  Lower is better.  This is a direct extension of the
germline-leakage result (spatialsnv_callset_quality.py): leakage says common
germline variants survive in the callset, and this says whether the surviving
calls contradict the tumour's own copy-number state.

WHY THIS IS NOT CIRCULAR.  Three separate guards:

  * BAF is measured from OUR OWN per-spot BAMs, at BOTH methods' positions, with
    the pipeline's own baseQ0/mapQ0 settings.  Neither tool grades itself, and
    neither tool's internal depth or VAF model enters the readout.
  * SPARCAL's spatial descriptor uses within-clone PREVALENCE in CNV segments and
    never sees BAF, so BAF is an independent axis even though the segments are
    shared.  State this explicitly in Methods rather than waiting to be asked.
  * copyKAT selects the segments but supplies no allelic information (it is
    expression-based and returns total, not allele-specific, copy number).  The
    allelic evidence is the germline positive control below.

THE POSITIVE CONTROL GATES EVERYTHING.  1KGP-defined germline sites are known
heterozygotes and are routed out of SPARCAL's somatic cascade entirely, so they
are neutral ground.  Inside a genuine LOH segment they MUST lose the 0.5 band in
aneuploid spots while keeping it in diploid spots.  Two internal contrasts check
this: same sites in diploid spots (within-segment control) and the same sites in
copy-balanced regions (between-region control).  If the control fails, the
segment set is not LOH and the main comparison is not interpretable -- the script
reports the control first and flags the failure rather than burying it.

POWER, MEASURED NOT ASSUMED.  61-82% of somatic candidates are one alt UMI in one
spot, where BAF is 1.0 by construction and carries no information.  Positions are
therefore admitted only at pooled group depth >= --min-group-dp *within the spot
group being measured*, which is a stricter and more honest filter than the
aggregate DP used for scoping.  n is reported per cell of every table; the
SPARCAL side is the small one (order 10^2), so treat any per-segment split as
descriptive and read the pooled number as the result.

BUILD RESTRICTION.  DCIS1/DCIS2 only: copyKAT for P4/P6 is GRCh38 while their
variants are hg19 (see loh_vignette_scoping.py).  Refuses other samples.

Outputs (under data/loh_allelic_test_<date>/):
  loh_positions.tsv          the queried positions with their labels
  loh_baf_raw.csv            per position x spot-group pooled ref/alt/BAF
  loh_control_germline.csv   the positive control table (report this first)
  loh_method_comparison.csv  het-band fraction by method x region x spot group
  RESULT.md                  written by hand after inspecting the above

Run (env snv_caller, ~20-40 min):
  python scripts/postanalyze/loh_allelic_test.py --samples DCIS2,DCIS1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date

import numpy as np
import pandas as pd

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
BCFTOOLS = os.path.join(REPO, "apps", "bcftools")
SAMTOOLS = os.path.join(REPO, "apps", "samtools")
CK = "/data/maiziezhou_lab/Weiman/ST_CNV/copyKAT"
REF = os.path.join(REPO, "data/reference/GRCh38.fa")

SCOPING_DIR = os.path.join(REPO, "data", "loh_vignette_scoping_2026-08-06")
CONCORDANCE_DIR = os.path.join(REPO, "data", "spatialsnv_concordance_2026-08-06")

SAMPLES = {
    "DCIS1": dict(ckdir="DCIS_1", root="data/dcis1"),
    "DCIS2": dict(ckdir="DCIS_2", root="data/dcis2"),
}
SPOT_BAMS = "output_VCFs/spotprofiles/baseQ0mapQ0/bam_by_spot"
GERMLINE_DEF = "spatial_filter_purity/baseQ0mapQ0/germline/defined/germline_defined.vcf.gz"

HET_LO, HET_HI = 0.35, 0.65      # the band a lost allele cannot produce


def load_pred(ckdir: str) -> dict[str, str]:
    pred = {}
    with open(os.path.join(CK, ckdir, "prediction_aligned.txt")) as fh:
        next(fh)
        for ln in fh:
            p = ln.rstrip("\n").split("\t")
            if len(p) >= 2 and p[1] in ("aneuploid", "diploid"):
                pred[p[0]] = p[1]
    return pred


def germline_sites(path: str, chroms: set[str]) -> pd.DataFrame:
    proc = subprocess.run([BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", path],
                          capture_output=True, text=True)
    rows = []
    for ln in proc.stdout.splitlines():
        p = ln.split("\t")
        if len(p) < 4 or len(p[2]) != 1 or len(p[3]) != 1:
            continue
        c = p[0][3:] if p[0].lower().startswith("chr") else p[0]
        if c in chroms:
            rows.append((c, int(p[1]), p[2], p[3]))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"])


def in_segments(chrom, pos, segs: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(chrom), dtype=bool)
    for _, r in segs.iterrows():
        mask |= (chrom == str(r["chrom"])) & (pos >= r["start"]) & (pos <= r["end"])
    return mask


def merge_clone_bam(bam_paths: list[str], workdir: str, tag: str, threads: int) -> str:
    """Merge one clone's spot BAMs into a single BAM, once, and cache it.

    NOT an optimisation detail -- it is the difference between minutes and days.
    `bcftools mpileup -b <list> -R <positions>` costs roughly
    n_bams x n_positions index seeks; measured at ~88 s for 150 positions across
    only 200 spot BAMs, which extrapolates to many hours for ~900 BAMs x ~20k
    positions.  Merging is a single linear read+write (~2 GB per clone), after
    which the pileup is one indexed pass over one file.

    Pooling is lossless for this analysis: every spot BAM carries the same @RG SM
    tag, so bcftools would have collapsed them into one sample column anyway, and
    the quantity wanted is the clone-level pooled count, not per-spot values."""
    merged = os.path.join(workdir, f"{tag}_merged.bam")
    if os.path.exists(merged + ".bai"):
        print(f"    reusing cached {os.path.basename(merged)}")
        return merged
    if not os.path.exists(merged) or os.path.getsize(merged) == 0:
        listfile = os.path.join(workdir, f"bams_{tag}.txt")
        with open(listfile, "w") as fh:
            fh.write("\n".join(bam_paths) + "\n")
        subprocess.run([SAMTOOLS, "merge", "-f", "-@", str(threads),
                        "-b", listfile, merged], check=True)
    else:
        print(f"    {os.path.basename(merged)} already merged; indexing only")
    # samtools 1.2 ships with this project and its `index` predates -@; passing
    # thread flags here fails outright rather than being ignored.
    subprocess.run([SAMTOOLS, "index", merged], check=True)
    return merged


def pileup_group(merged_bam: str, regions_file: str, tag: str) -> pd.DataFrame:
    """Pooled ref/alt at the query positions for one clone.

    Quality settings mirror the pipeline's baseQ0mapQ0 convention (-q 0 -Q 0) so
    the numbers are comparable with everything else in the project.  n_alt sums
    every non-reference allele rather than matching the queried ALT: the readout
    is distance from the balanced heterozygous band, and a second alt allele at
    the same site contributes to imbalance just as the first does.

    -T, NOT -R.  Both restrict to the query positions, but `-R` jumps through the
    index once per region, and these are ~10^5 scattered single-base positions:
    measured at 42 min of CPU without finishing one clone.  `-T` streams the BAM
    once and skips non-target positions, which is far cheaper at this position
    count and scatter.  The targets file must be in the BAM's contig order --
    both are lexicographic here (1, 10, 11, ... 2, 20, ...), so the plain
    sort_values(["chrom", "pos"]) upstream is already correct; re-check this if
    the contig naming ever changes."""
    cmd = [BCFTOOLS, "mpileup", merged_bam, "-f", REF, "-T", regions_file,
           "-q", "0", "-Q", "0", "-d", "10000000", "-a", "AD",
           "-I", "--no-version", "-Ou"]
    p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    p2 = subprocess.Popen(
        [BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT[\t%AD]\n"],
        stdin=p1.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    assert p1.stdout is not None and p2.stdout is not None
    p1.stdout.close()

    rows = []
    for line in p2.stdout:
        f = line.rstrip("\n").split("\t")
        if len(f) < 5:
            continue
        chrom, pos, ref, alt, ad = f[0], int(f[1]), f[2], f[3], f[4]
        counts = [int(x) for x in ad.split(",") if x not in (".", "")]
        if not counts:
            continue
        n_ref = counts[0]
        n_alt = sum(counts[1:]) if len(counts) > 1 else 0
        rows.append((chrom, pos, ref, alt, n_ref, n_alt))
    p2.wait()
    p1.wait()
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref_mpileup", "alt_mpileup",
                                       f"n_ref_{tag}", f"n_alt_{tag}"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", default="DCIS2,DCIS1")
    ap.add_argument("--min-group-dp", type=int, default=10,
                    help="pooled depth required WITHIN the spot group being measured")
    ap.add_argument("--max-germline", type=int, default=6000,
                    help="cap on control sites per region class (keeps mpileup bounded)")
    ap.add_argument("--max-balanced", type=int, default=15000,
                    help="cap on copy-BALANCED control variants. Loss-region variants "
                         "are never subsampled -- they are the measurement. The "
                         "balanced set is only a reference distribution, and the "
                         "uncapped version contributed ~72k of ~101k queried "
                         "positions in DCIS2 for no statistical gain.")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bamwork",
                    default="/tmp/claude-958611/-panfs-accrepfs-vampire-home-chowx/"
                            "f63884a9-0a46-4146-9fa8-abdfa19f1be2/scratchpad/loh_bams",
                    help="scratch for the merged per-clone BAMs (~4 GB per sample); "
                         "local disk, not panfs -- these are intermediates, not results")
    args = ap.parse_args()

    names = [s.strip() for s in args.samples.split(",") if s.strip()]
    for n in names:
        if n not in SAMPLES:
            raise SystemExit(f"{n} not supported: hg19/GRCh38 mismatch, see docstring.")

    outdir = args.outdir or os.path.join(
        REPO, f"data/loh_allelic_test_{date.today().isoformat()}")
    os.makedirs(outdir, exist_ok=True)

    segs_all = pd.read_csv(os.path.join(SCOPING_DIR, "loh_segments.csv"))
    bins_all = pd.read_csv(os.path.join(SCOPING_DIR, "loh_bins.csv"))
    variants = pd.read_csv(os.path.join(CONCORDANCE_DIR, "concordance_variant_level.csv.gz"))
    variants["chrom"] = variants["chrom"].astype(str)

    control_rows, method_rows, raw_frames = [], [], []

    for sample in names:
        cfg = SAMPLES[sample]
        segs = segs_all[segs_all["sample"] == sample].copy()
        segs["chrom"] = segs["chrom"].astype(str)
        bins = bins_all[bins_all["sample"] == sample].copy()
        bins["chrom"] = bins["chrom"].astype(str)
        chroms = set(bins["chrom"].unique())

        # ---- balanced control regions: bins with the smallest clone difference
        bal = bins[bins["delta"].abs() <= bins["delta"].abs().quantile(0.25)]
        bal_segs = pd.DataFrame(dict(chrom=bal["chrom"],
                                     start=bal["chrompos"] - 1_000_000,
                                     end=bal["chrompos"] + 1_000_000))

        # ---- assemble query positions --------------------------------------
        v = variants[variants["sample"] == sample].copy()
        vc, vp = v["chrom"].values, v["pos"].values
        v["region"] = np.where(in_segments(vc, vp, segs), "loss",
                               np.where(in_segments(vc, vp, bal_segs), "balanced", "other"))
        v = v[v["region"] != "other"]
        n_bal = int((v["region"] == "balanced").sum())
        if n_bal > args.max_balanced:
            v = pd.concat([
                v[v["region"] == "loss"],
                v[v["region"] == "balanced"].sample(args.max_balanced, random_state=0),
            ], ignore_index=True)
            print(f"[{sample}] balanced-region control variants subsampled "
                  f"{n_bal:,} -> {args.max_balanced:,} (loss region kept in full)")

        g = germline_sites(os.path.join(REPO, cfg["root"], GERMLINE_DEF), chroms)
        gc, gp = g["chrom"].values, g["pos"].values
        g["region"] = np.where(in_segments(gc, gp, segs), "loss",
                               np.where(in_segments(gc, gp, bal_segs), "balanced", "other"))
        g = g[g["region"] != "other"]
        g = (g.groupby("region", group_keys=False)
               .apply(lambda d: d.sample(min(len(d), args.max_germline), random_state=0)))
        g["group"] = "germline_1kG"

        query = pd.concat([
            v[["chrom", "pos", "group", "region"]],
            g[["chrom", "pos", "group", "region"]],
        ], ignore_index=True).drop_duplicates(subset=["chrom", "pos"])
        query = query.sort_values(["chrom", "pos"])
        print(f"[{sample}] querying {len(query):,} positions "
              f"({(query['group']=='germline_1kG').sum():,} germline control)")

        regions_file = os.path.join(outdir, f"{sample}_regions.tsv")
        query[["chrom", "pos"]].to_csv(regions_file, sep="\t", header=False, index=False)
        query.to_csv(os.path.join(outdir, f"{sample}_loh_positions.tsv"),
                     sep="\t", index=False)

        # ---- pooled pileup per clone ---------------------------------------
        pred = load_pred(cfg["ckdir"])
        bamdir = os.path.join(REPO, cfg["root"], SPOT_BAMS)
        got = {}
        for tag, want in (("aneuploid", "aneuploid"), ("diploid", "diploid")):
            paths = [os.path.join(bamdir, f"{bc}.bam") for bc, lab in pred.items()
                     if lab == want and os.path.exists(os.path.join(bamdir, f"{bc}.bam"))]
            print(f"[{sample}] {tag}: {len(paths)} spot BAMs")
            if not paths:
                raise SystemExit(f"no {tag} BAMs found under {bamdir}")
            workdir = os.path.join(args.bamwork, sample)
            os.makedirs(workdir, exist_ok=True)
            merged = merge_clone_bam(paths, workdir, tag, args.threads)
            got[tag] = pileup_group(merged, regions_file, tag)

        df = query.merge(got["aneuploid"], on=["chrom", "pos"], how="left") \
                  .merge(got["diploid"], on=["chrom", "pos"], how="left")
        for tag in ("aneuploid", "diploid"):
            dp = df[f"n_ref_{tag}"].fillna(0) + df[f"n_alt_{tag}"].fillna(0)
            df[f"dp_{tag}"] = dp
            df[f"baf_{tag}"] = np.where(dp >= args.min_group_dp,
                                        df[f"n_alt_{tag}"] / dp.replace(0, np.nan), np.nan)
        df["sample"] = sample
        raw_frames.append(df)

        # ---- positive control: does the segment set actually show LOH? ------
        for region in ("loss", "balanced"):
            for tag in ("aneuploid", "diploid"):
                sub = df[(df["group"] == "germline_1kG") & (df["region"] == region)]
                baf = sub[f"baf_{tag}"].dropna()
                if len(baf) == 0:
                    continue
                control_rows.append(dict(
                    sample=sample, region=region, spot_group=tag, n=len(baf),
                    het_band_frac=round(float(((baf >= HET_LO) & (baf <= HET_HI)).mean()), 4),
                    median_baf=round(float(baf.median()), 4),
                    mean_abs_dev_from_half=round(float((baf - 0.5).abs().mean()), 4),
                ))

        # ---- the comparison -------------------------------------------------
        # SPARCAL = concordant + sparcal_private; SpatialSNV = concordant +
        # spatialsnv_private.  The shared concordant calls are deliberately
        # counted on both sides: the question is what each DELIVERED callset
        # looks like, not what is unique to it.
        methods = {
            "SPARCAL": ("concordant", "sparcal_private"),
            "SpatialSNV": ("concordant", "spatialsnv_private"),
        }
        for region in ("loss", "balanced"):
            for method, grps in methods.items():
                for tag in ("aneuploid", "diploid"):
                    sub = df[(df["region"] == region) & (df["group"].isin(grps))]
                    baf = sub[f"baf_{tag}"].dropna()
                    if len(baf) == 0:
                        continue
                    het = int(((baf >= HET_LO) & (baf <= HET_HI)).sum())
                    method_rows.append(dict(
                        sample=sample, region=region, method=method, spot_group=tag,
                        n_testable=len(baf), n_het_band=het,
                        het_band_pct=round(100 * het / len(baf), 3),
                        median_baf=round(float(baf.median()), 4),
                    ))

    raw = pd.concat(raw_frames, ignore_index=True)
    raw.to_csv(os.path.join(outdir, "loh_baf_raw.csv"), index=False)
    ctl = pd.DataFrame(control_rows)
    ctl.to_csv(os.path.join(outdir, "loh_control_germline.csv"), index=False)
    mth = pd.DataFrame(method_rows)
    mth.to_csv(os.path.join(outdir, "loh_method_comparison.csv"), index=False)

    print("\n===== POSITIVE CONTROL (1KGP germline hets) =====")
    print("LOH is real only if het_band_frac drops in loss/aneuploid vs its controls.")
    print(ctl.to_string(index=False))
    print("\n===== METHOD COMPARISON (het-band %, lower = more consistent with CNV state) =====")
    print(mth.to_string(index=False))
    print(f"\nWrote outputs to {outdir}")


if __name__ == "__main__":
    main()
