#!/usr/bin/env python3
"""P1-4: depth-matched COSMIC contrast as the primary somatic-vs-unresolved analysis.

Background (PAPER_PLAN_DEPRECATED.md sec 3.1, PAPER_PLAN.md sec 4.8/M5): a 2026-07-13
logistic control on P4/P6 (`cosmic_hit ~ is_somatic + log10(DP)`, xMHC excluded)
already found the enrichment null: OR 1.04 p=0.56 (P4), OR 1.00 p=0.95 (P6). That
result never became the primary analysis and DCIS1/DCIS2 were never run through
it. This script:

  1. Recomputes the depth control for all four sections (P4, P6, DCIS1, DCIS2),
     not just P4/P6.
  2. Makes it the PRIMARY analysis: reported first, before the raw contrast.
  3. Replaces the one-sided Fisher exact test on the raw variant-count table
     with a permutation null over variants (variants are not independent
     observations -- they cluster genomically, e.g. COSMIC hotspot genes and
     multiple SNVs called at the same locus/gene, so the binomial-independence
     assumption behind Fisher's exact test is not credible here).
  4. Reports xMHC (chr6:28-34 Mb, both builds, same interval as fig4/5) both
     excluded (default/primary) and included (sensitivity).

DEPTH-MATCHED DESIGN
    Per-variant depth = INFO/DP from the pooled pre-classification mpileup VCF
    (`output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz`), the
    same source used by the original 2026-07-13 control and by
    somatic_evidence_package.py's "our RNA depth" column.
    (a) Logistic regression `cosmic_hit ~ is_somatic + log10(DP+1)` -- Wald test
        on the is_somatic coefficient, reported as an odds ratio (continuity
        with the 2026-07-13 numbers).
    (b) A depth-STRATIFIED permutation null: bin variants into the project's
        standard RNA-depth strata (1-3, 4-9, 10-29, 30+ reads), and within each
        stratum permute the somatic/unresolved label 10,000 times (marginal
        counts per stratum held fixed), recomputing a depth-standardized
        (Mantel-Haenszel) rate ratio each time. The empirical p-value is the
        fraction of permuted ratios at least as extreme as observed. This
        controls for depth by construction (matching) rather than by a
        parametric depth term, and treats within-stratum shuffling as the
        exchangeability unit, which is the closest defensible approximation to
        "not independent observations" without a full LD/genomic-block model.

RAW (UNADJUSTED) DESIGN
    Global permutation null: shuffle the somatic/unresolved label across all
    variants (10,000 permutations, group sizes held fixed), recompute the raw
    rate ratio each time, empirical two-sided p-value from the null
    distribution. Reported for direct contrast with the depth-matched design,
    never as the headline.

Run (env snv_caller): python scripts/postanalyze/p1_4_cosmic_depthmatched.py --outdir <dir>
"""
from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")

COSMIC = {
    "hg19": Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz"),
    "GRCh38": Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz"),
}

SAMPLES = {
    "P4": dict(build="hg19", root=PROJECT / "data/P4_tumor/1"),
    "P6": dict(build="hg19", root=PROJECT / "data/P6_tumor/1"),
    "DCIS1": dict(build="GRCh38", root=PROJECT / "data/dcis1"),
    "DCIS2": dict(build="GRCh38", root=PROJECT / "data/dcis2"),
}
QF = "baseQ0mapQ0"
CLASS_REL = {
    "somatic": "spatial_filter_purity/{qf}/somatic/denovo/somatic_denovo.vcf.gz",
    "unresolved": "spatial_filter_purity/{qf}/ambiguous/denovo/ambiguous_denovo.vcf.gz",
}
DP_SOURCE_REL = "output_VCFs/mpileup_multi_bam/{qf}/merged_sorted_gt.vcf.gz"

MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000
DEPTH_BINS = [(1, 3), (4, 9), (10, 29), (30, 10**9)]
N_PERM = 10000
RNG_SEED = 20260827


def strip_chr(c: str) -> str:
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def depth_bin_label(dp: int) -> str | None:
    for lo, hi in DEPTH_BINS:
        if lo <= dp <= hi:
            return f"{lo}-{hi}" if hi < 10**8 else f"{lo}+"
    return None


def load_class_variants(root: Path, cls: str) -> dict[str, tuple[str, int, str, str]]:
    path = root / CLASS_REL[cls].format(qf=QF)
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = f[0], int(f[1]), f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            key = f"{strip_chr(chrom)}_{pos}_{ref.upper()}_{alt.upper()}"
            out[key] = (chrom, pos, ref.upper(), alt.upper())
    return out


def load_dp_map(root: Path) -> dict[str, int]:
    """key(no chr,pos,ref,alt) -> INFO/DP from the pooled pre-classification VCF."""
    path = root / DP_SOURCE_REL.format(qf=QF)
    out = {}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt, info = f[0], int(f[1]), f[3], f[4], f[7]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            dp = None
            for item in info.split(";"):
                if item.startswith("DP="):
                    try:
                        dp = int(item[3:])
                    except ValueError:
                        dp = None
                    break
            if dp is None:
                continue
            key = f"{strip_chr(chrom)}_{pos}_{ref.upper()}_{alt.upper()}"
            out[key] = dp
    return out


def cosmic_hit_keys(build: str, keys: set[str]) -> set[str]:
    import subprocess
    path = COSMIC[build]
    hits = set()
    proc = subprocess.Popen(["zcat", str(path)], stdout=subprocess.PIPE, text=True,
                            stderr=subprocess.DEVNULL, bufsize=1 << 20)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line[0] == "#":
            continue
        f = line.split("\t", 8)
        if len(f) < 8:
            continue
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1:
            continue
        k = f"{strip_chr(f[0])}_{f[1]}_{ref.upper()}_{alt.upper()}"
        if k in keys:
            hits.add(k)
    proc.wait()
    return hits


def build_table(sample: str, cfg: dict) -> pd.DataFrame:
    root = cfg["root"]
    som = load_class_variants(root, "somatic")
    unr = load_class_variants(root, "unresolved")
    dp_map = load_dp_map(root)
    rows = []
    for cls, d in [("somatic", som), ("unresolved", unr)]:
        for key, (chrom, pos, ref, alt) in d.items():
            rows.append({
                "sample": sample, "class": cls, "key": key,
                "chrom": strip_chr(chrom), "pos": pos, "ref": ref, "alt": alt,
                "dp": dp_map.get(key, np.nan),
            })
    df = pd.DataFrame(rows)
    all_keys = set(df["key"])
    hits = cosmic_hit_keys(cfg["build"], all_keys)
    df["cosmic_hit"] = df["key"].isin(hits).astype(int)
    df["in_mhc"] = (df["chrom"] == MHC_CHR) & (df["pos"].between(MHC_LO, MHC_HI))
    df["depth_bin"] = df["dp"].apply(lambda x: depth_bin_label(int(x)) if pd.notna(x) else None)
    return df


def raw_permutation_test(df: pd.DataFrame, rng: np.random.Generator):
    """Global label-permutation null, no depth control.

    Drawing n_som labels at random (without replacement) from the pooled
    somatic+unresolved variant set and counting how many are COSMIC hits is
    exactly a draw from Hypergeometric(ngood=n_hit_total, nbad=n_total-n_hit,
    nsample=n_som) -- the exact distribution explicit label-shuffling would
    produce, sampled directly rather than by shuffling ~1e6 elements 10,000x.
    """
    is_som = (df["class"] == "somatic").values
    hit = df["cosmic_hit"].values
    n_som, n_unr = int(is_som.sum()), int((~is_som).sum())
    h_som, h_unr = int(hit[is_som].sum()), int(hit[~is_som].sum())
    if n_som == 0 or n_unr == 0:
        return dict(n_somatic=n_som, n_unresolved=n_unr, somatic_hits=h_som,
                    unresolved_hits=h_unr, somatic_rate_pct=np.nan,
                    unresolved_rate_pct=np.nan, ratio=np.nan, perm_p=np.nan, n_perm=N_PERM)
    rate_som, rate_unr = h_som / n_som, h_unr / n_unr
    ratio = rate_som / rate_unr if rate_unr > 0 else np.inf
    n_total = len(df)
    n_hit_total = int(hit.sum())
    perm_hit_som = rng.hypergeometric(ngood=n_hit_total, nbad=n_total - n_hit_total,
                                       nsample=n_som, size=N_PERM)
    perm_hit_unr = n_hit_total - perm_hit_som
    with np.errstate(divide="ignore", invalid="ignore"):
        r_som = perm_hit_som / n_som
        r_unr = perm_hit_unr / n_unr
        null_ratios = r_som / r_unr
    valid = null_ratios[np.isfinite(null_ratios)]
    # two-sided empirical p on log-ratio distance from 0 (ratio 1)
    obs_dist = abs(np.log(ratio)) if np.isfinite(ratio) and ratio > 0 else np.inf
    null_dist = np.abs(np.log(valid[valid > 0]))
    perm_p = float((null_dist >= obs_dist).sum() + 1) / (len(null_dist) + 1)
    return dict(n_somatic=n_som, n_unresolved=n_unr, somatic_hits=h_som,
                unresolved_hits=h_unr, somatic_rate_pct=100 * rate_som,
                unresolved_rate_pct=100 * rate_unr, ratio=ratio, perm_p=perm_p, n_perm=N_PERM)


def stratified_mh_ratio(df: pd.DataFrame) -> float:
    """Mantel-Haenszel depth-standardized somatic/unresolved rate ratio."""
    num = den = 0.0
    for b, g in df.groupby("depth_bin", observed=True):
        n_som = int((g["class"] == "somatic").sum())
        n_unr = int((g["class"] == "unresolved").sum())
        h_som = int(g.loc[g["class"] == "somatic", "cosmic_hit"].sum())
        h_unr = int(g.loc[g["class"] == "unresolved", "cosmic_hit"].sum())
        n = n_som + n_unr
        if n == 0 or n_unr == 0 or n_som == 0:
            continue
        num += h_som * n_unr / n
        den += h_unr * n_som / n
    return num / den if den > 0 else np.nan


def depth_matched_permutation_test(df: pd.DataFrame, rng: np.random.Generator):
    """Depth-stratified label-permutation null for the Mantel-Haenszel ratio.

    Within each depth stratum, the number of COSMIC hits among a random
    n_som_b-sized draw (without replacement, labels shuffled) is exactly
    Hypergeometric(ngood=h_total_b, nbad=n_b-h_total_b, nsample=n_som_b);
    drawing N_PERM samples per stratum and recombining with the MH formula
    gives the depth-matched permutation null directly, without materializing
    10,000 explicit shuffles of up to ~6e5 variants.
    """
    df = df.dropna(subset=["depth_bin"]).copy()
    if df.empty or df["class"].nunique() < 2:
        return dict(n_somatic=0, n_unresolved=0, mh_ratio=np.nan, perm_p=np.nan, n_perm=N_PERM)
    obs_ratio = stratified_mh_ratio(df)
    n_som = int((df["class"] == "somatic").sum())
    n_unr = int((df["class"] == "unresolved").sum())

    num_total = np.zeros(N_PERM)
    den_total = np.zeros(N_PERM)
    for b, g in df.groupby("depth_bin", observed=True):
        n_som_b = int((g["class"] == "somatic").sum())
        n_unr_b = int((g["class"] == "unresolved").sum())
        n_b = n_som_b + n_unr_b
        if n_b == 0 or n_som_b == 0 or n_unr_b == 0:
            continue
        h_total_b = int(g["cosmic_hit"].sum())
        h_som_b = rng.hypergeometric(ngood=h_total_b, nbad=n_b - h_total_b,
                                     nsample=n_som_b, size=N_PERM)
        h_unr_b = h_total_b - h_som_b
        num_total += h_som_b * n_unr_b / n_b
        den_total += h_unr_b * n_som_b / n_b

    with np.errstate(divide="ignore", invalid="ignore"):
        null_ratios = num_total / den_total
    valid = null_ratios[np.isfinite(null_ratios)]
    obs_dist = abs(np.log(obs_ratio)) if np.isfinite(obs_ratio) and obs_ratio > 0 else np.inf
    null_dist = np.abs(np.log(valid[valid > 0]))
    perm_p = float((null_dist >= obs_dist).sum() + 1) / (len(null_dist) + 1) if len(null_dist) else np.nan
    return dict(n_somatic=n_som, n_unresolved=n_unr, mh_ratio=obs_ratio, perm_p=perm_p, n_perm=N_PERM)


def logistic_depth_control(df: pd.DataFrame):
    """cosmic_hit ~ is_somatic + log10(DP+1); returns OR, CI, Wald p for is_somatic."""
    sub = df.dropna(subset=["dp"]).copy()
    if sub.empty or sub["class"].nunique() < 2:
        return dict(n=0, odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, wald_p=np.nan)
    try:
        import statsmodels.api as sm
    except ImportError:
        return dict(n=len(sub), odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, wald_p=np.nan,
                    note="statsmodels unavailable")
    sub["is_somatic"] = (sub["class"] == "somatic").astype(int)
    sub["log10dp"] = np.log10(sub["dp"].astype(float) + 1)
    X = sm.add_constant(sub[["is_somatic", "log10dp"]])
    y = sub["cosmic_hit"].astype(int)
    try:
        model = sm.Logit(y, X).fit(disp=0)
    except Exception as exc:  # perfect separation etc.
        return dict(n=len(sub), odds_ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, wald_p=np.nan,
                    note=f"fit failed: {exc}")
    coef = model.params["is_somatic"]
    se = model.bse["is_somatic"]
    p = model.pvalues["is_somatic"]
    return dict(n=len(sub), odds_ratio=float(np.exp(coef)),
                ci_lo=float(np.exp(coef - 1.96 * se)), ci_hi=float(np.exp(coef + 1.96 * se)),
                wald_p=float(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    depth_strata_rows, matched_rows, perm_rows = [], [], []
    all_detail = []
    for sample in args.samples:
        cfg = SAMPLES[sample]
        print(f"[{sample}] building variant table ...", flush=True)
        df = build_table(sample, cfg)
        all_detail.append(df)
        print(f"[{sample}] n_somatic={int((df['class']=='somatic').sum())} "
              f"n_unresolved={int((df['class']=='unresolved').sum())}", flush=True)

        for xmhc_label, sub in [("xmhc_excluded", df[~df["in_mhc"]]),
                                 ("xmhc_included", df)]:
            # -- raw / unadjusted --
            raw = raw_permutation_test(sub, rng)
            perm_rows.append({"sample": sample, "analysis": "raw", "xmhc": xmhc_label, **raw})

            # -- depth-matched: stratified permutation --
            matched_perm = depth_matched_permutation_test(sub, rng)
            perm_rows.append({"sample": sample, "analysis": "depth_matched_stratified_permutation",
                              "xmhc": xmhc_label, **matched_perm})

            # -- depth-matched: logistic control (continuity with 2026-07-13 numbers) --
            logit = logistic_depth_control(sub)
            matched_rows.append({"sample": sample, "xmhc": xmhc_label, **logit})

            # -- depth strata table --
            for b, g in sub.dropna(subset=["depth_bin"]).groupby("depth_bin", observed=True):
                for cls in ["somatic", "unresolved"]:
                    gc = g[g["class"] == cls]
                    n = len(gc)
                    h = int(gc["cosmic_hit"].sum())
                    depth_strata_rows.append({
                        "sample": sample, "xmhc": xmhc_label, "depth_bin": b, "class": cls,
                        "n": n, "cosmic_hits": h,
                        "cosmic_rate_pct": 100 * h / n if n else np.nan,
                    })

    depth_strata = pd.DataFrame(depth_strata_rows)
    matched = pd.DataFrame(matched_rows)
    perms = pd.DataFrame(perm_rows)
    detail = pd.concat(all_detail, ignore_index=True)

    depth_strata.to_csv(args.outdir / "depth_strata.csv", index=False)
    matched.to_csv(args.outdir / "matched_contrast.csv", index=False)
    perms.to_csv(args.outdir / "permutation_null.csv", index=False)
    detail.to_csv(args.outdir / "cosmic_depthmatched_detail.csv.gz", index=False, compression="gzip")
    # Combined single table as requested output name.
    combo = perms.merge(
        matched.rename(columns={"n": "n_logit", "odds_ratio": "logit_odds_ratio",
                                "ci_lo": "logit_ci_lo", "ci_hi": "logit_ci_hi",
                                "wald_p": "logit_wald_p"}),
        on=["sample", "xmhc"], how="left")
    combo.to_csv(args.outdir / "cosmic_depthmatched.csv", index=False)

    pd.set_option("display.width", 220, "display.max_columns", 50)
    print("\n=== permutation_null.csv ===")
    print(perms.round(4).to_string(index=False))
    print("\n=== matched_contrast.csv (logistic) ===")
    print(matched.round(4).to_string(index=False))
    print(f"\nWrote P1-4 package to {args.outdir}")


if __name__ == "__main__":
    main()
