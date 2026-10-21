#!/usr/bin/env python
"""Cross-method concordance and COSMIC catalogue membership.

THE QUESTION.  If two independently-designed callers agree on a variant, is that
agreement worth anything?  If yes, cross-method concordance is a usable confidence
axis for the field -- the practical answer to "which calls should I trust".

CURRENT USE. The paper reports allele-exact COSMIC membership for every caller
on the same basis as descriptive catalogue corroboration, not as a truth set or
as validation of somatic identity. This older script also retains stratified
and regression sensitivity outputs for reproducibility:

    logistic   cosmic_hit ~ group + log10(DP)          (group = 3-level)
    plus non-parametric depth-stratified rates within DP bins.

Those sensitivity outputs are not required for the head-to-head catalogue table.

DESIGN CHOICES, and why:

  * DP is measured METHOD-NEUTRALLY, from OUR OWN mpileup pool
    (merged_sorted_gt.vcf.gz), at every position regardless of which method
    called it.  Using each method's self-reported depth would let each tool grade
    itself on its own coverage model.

  * Restricting to positions present in our pool is itself a selection, and it is
    NOT missing-at-random: ~77% of SpatialSNV's calls appear in our pool (verified
    on P6 chr1: 10,061/13,032 allele-exact).  The retained/dropped counts per
    group are written to concordance_coverage_availability.csv and MUST be
    reported alongside any effect size -- the SpatialSNV-private group is the one
    that loses the most positions, and it loses them non-randomly.

    THE ASYMMETRY IS STRUCTURAL, so read the two contrasts differently.  DP is
    available for 100% of `concordant` and 100% of `sparcal_private` by
    construction (both are drawn from our pool), but only ~50-73% of
    `spatialsnv_private`.  Therefore:
      - concordant vs sparcal_private  is a CLEAN contrast: no differential
        selection, both groups fully observed.  This is the headline test, and
        it is also the question that matters ("does agreement add anything on
        top of SPARCAL's own calls?").
      - anything vs spatialsnv_private inherits the ~30% non-random dropout and
        is reported as secondary, with the availability column attached.

  * ALLELE-EXACT matching throughout, never position-only.  A span/region match
    against a catalogue silently counts indel/SV records whose SPAN covers the
    query point; that artifact inflated an earlier WES-leakage figure ~3x
    (28% -> 8.3%).  See spatialsnv_callset_quality.py and PAPER_PLAN.md Sec 3 C2.

  * xMHC (chr6:28-34 Mb) is excluded in a parallel set of fits, never only
    reported raw: HLA drove the DCIS COSMIC effect entirely (DCIS2 1.17x -> 0.99x,
    p=0.55).  Framing must stay tumor-type-specific.

  * SPARCAL side is the somatic denovo set (somatic/denovo/somatic_denovo.vcf.gz),
    matching spatialsnv_callset_quality.py.  1KGP-panel variants never enter it.

WHAT THIS SCRIPT DOES NOT CLAIM. COSMIC catalogue membership is not truth, and
no result here validates driver status or somatic identity.

Outputs (under data/spatialsnv_concordance_<date>/):
  concordance_group_sizes.csv            per sample: set sizes + overlap (Jaccard)
  concordance_coverage_availability.csv  per sample x group: DP-observed fraction
  concordance_cosmic_raw.csv             per sample x group: unadjusted hit rates
  concordance_depth_stratified.csv       per sample x group x DP bin: rates
  concordance_depth_logistic.csv         per sample x subset: adjusted ORs + p
  concordance_variant_level.csv.gz       one row per variant (group, DP, hit)

Run (env snv_caller, ~1h wall with 11 workers):
  python scripts/postanalyze/spatialsnv_concordance_cosmic.py --workers 11
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date

import numpy as np
from scipy.stats import fisher_exact, norm

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
BCFTOOLS = os.path.join(REPO, "apps", "bcftools")

COSMIC = {
    "hg19": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz",
    "GRCh38": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz",
}

SPARCAL_SOMATIC = "spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz"
MERGED_GT = "output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz"

SAMPLES = {
    "P4": dict(build="hg19", ssnv_dir="SpatialSNV/results/p4", ssnv_tag="p4",
               sparcal_root="data/P4_tumor/1"),
    "P6": dict(build="hg19", ssnv_dir="SpatialSNV/results/p6", ssnv_tag="p6",
               sparcal_root="data/P6_tumor/1"),
    "DCIS1": dict(build="GRCh38", ssnv_dir="SpatialSNV/results/dcis1", ssnv_tag="dcis1",
                  sparcal_root="data/dcis1"),
    "DCIS2": dict(build="GRCh38", ssnv_dir="SpatialSNV/results/dcis2", ssnv_tag="dcis2",
                  sparcal_root="data/dcis2"),
}

AUTOSOMES = [str(c) for c in range(1, 23)]
MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000

GROUPS = ["concordant", "sparcal_private", "spatialsnv_private"]
DP_BINS = [0, 1, 2, 5, 15, 1e9]
DP_LABELS = ["DP=1", "DP=2", "DP3-5", "DP6-15", "DP>15"]


# ---------------------------------------------------------------- keys
VALID_CHROMS = set(str(c) for c in range(1, 23))


def norm_chrom(c: str) -> str:
    c = c.strip()
    return c[3:] if c.lower().startswith("chr") else c


def is_autosome(c: str) -> bool:
    """Scaffolds must be rejected on the WHOLE normalised name, not its first
    underscore-delimited token: `chr1_KI270706v1_random` -> `1_KI270706v1_random`,
    whose leading token is `1` and would otherwise pass an autosome filter and
    then produce a 6-field key that breaks parsing downstream."""
    return norm_chrom(c) in VALID_CHROMS


def key(chrom, pos, ref, alt) -> str:
    return f"{norm_chrom(chrom)}_{pos}_{ref.upper()}_{alt.upper()}"


def split_key(k: str):
    chrom, pos, ref, alt = k.split("_")
    return chrom, int(pos), ref, alt


# ---------------------------------------------------------------- readers
def read_spatialsnv_features(path: str) -> set[str]:
    """Their final callset: `chr11_74456788:G>A` (P4/P6) or `12_54858388:T>A` (DCIS)."""
    out: set[str] = set()
    with gzip.open(path, "rt") as fh:
        for line in fh:
            fid = line.split("\t", 1)[0]
            try:
                locus, alleles = fid.split(":", 1)
                chrom, pos = locus.rsplit("_", 1)
                ref, alt = alleles.split(">", 1)
            except ValueError:
                continue
            if len(ref) == 1 and len(alt) == 1 and is_autosome(chrom):
                out.add(key(chrom, pos, ref, alt))
    return out


def read_vcf_keys(path: str) -> set[str]:
    proc = subprocess.Popen(
        [BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    out: set[str] = set()
    assert proc.stdout is not None
    for line in proc.stdout:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4 or len(p[2]) != 1 or len(p[3]) != 1 or not is_autosome(p[0]):
            continue
        out.add(key(p[0], p[1], p[2], p[3]))
    proc.wait()
    return out


# ---------------------------------------------------------------- scans
def scan_cosmic_chrom(args) -> tuple[str, set[str]]:
    """Allele-exact COSMIC hits on one chromosome, via the tabix index."""
    build, chrom, query_keys = args
    proc = subprocess.Popen(
        [BCFTOOLS, "query", "-r", chrom, "-f", "%CHROM\t%POS\t%REF\t%ALT\n", COSMIC[build]],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
    hits: set[str] = set()
    assert proc.stdout is not None
    for line in proc.stdout:
        p = line.rstrip("\n").split("\t")
        if len(p) < 4 or len(p[2]) != 1 or len(p[3]) != 1:
            continue
        k = f"{norm_chrom(p[0])}_{p[1]}_{p[2].upper()}_{p[3].upper()}"
        if k in query_keys:
            hits.add(k)
    proc.wait()
    return chrom, hits


def scan_depth_chrom(args) -> tuple[str, dict[tuple[str, int], float]]:
    """Position-level DP from our own pileup pool. Method-neutral by construction."""
    merged_path, chrom, query_pos = args
    dp: dict[tuple[str, int], float] = {}
    for prefix in (f"chr{chrom}", chrom):          # P4/P6 and DCIS differ on `chr`
        proc = subprocess.Popen(
            [BCFTOOLS, "query", "-r", prefix, "-f", "%CHROM\t%POS\t%INFO/DP\n", merged_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1 << 20)
        assert proc.stdout is not None
        n = 0
        for line in proc.stdout:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            try:
                pos = int(p[1])
            except ValueError:
                continue
            if pos in query_pos:
                try:
                    dp[(chrom, pos)] = float(p[2])
                except ValueError:
                    pass
            n += 1
        proc.wait()
        if n:                                       # this naming worked; don't retry
            break
    return chrom, dp


# ---------------------------------------------------------------- statistics
def logit_irls(X, y, max_iter=100, tol=1e-8):
    """Logistic regression by IRLS. X includes the intercept column.
    Returns (beta, se, wald_p). Same routine as the Fig. 5 depth control."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-eta))
        W = np.clip(mu * (1 - mu), 1e-9, None)
        z = eta + (y - mu) / W
        XtW = X.T * W
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    eta = X @ beta
    mu = 1.0 / (1.0 + np.exp(-eta))
    W = np.clip(mu * (1 - mu), 1e-9, None)
    cov = np.linalg.inv((X.T * W) @ X)
    se = np.sqrt(np.diag(cov))
    p = 2 * (1 - norm.cdf(np.abs(beta / se)))
    return beta, se, p


def dp_bin(dp: float) -> str:
    for lo, hi, lab in zip(DP_BINS[:-1], DP_BINS[1:], DP_LABELS):
        if lo < dp <= hi:
            return lab
    return DP_LABELS[-1]


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--chroms", default=",".join(AUTOSOMES))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples", default=",".join(SAMPLES))
    args = ap.parse_args()

    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    chrom_set = set(chroms)
    names = [s.strip() for s in args.samples.split(",") if s.strip()]
    outdir = args.outdir or os.path.join(
        REPO, f"data/spatialsnv_concordance_{date.today().isoformat()}")
    os.makedirs(outdir, exist_ok=True)

    # ---------------- assemble the variant table --------------------------
    loaded: dict[str, dict] = {}
    size_rows = []
    for name in names:
        cfg = SAMPLES[name]
        feat = os.path.join(REPO, cfg["ssnv_dir"], "matrix",
                            f"{cfg['ssnv_tag']}_alt", "features.tsv.gz")
        som = os.path.join(REPO, cfg["sparcal_root"], SPARCAL_SOMATIC)

        sys.stderr.write(f"[{name}] loading callsets\n")
        ssnv = {k for k in read_spatialsnv_features(feat) if k.split("_", 1)[0] in chrom_set}
        sparcal = {k for k in read_vcf_keys(som) if k.split("_", 1)[0] in chrom_set}

        conc = sparcal & ssnv
        group_of: dict[str, str] = {}
        for k in conc:
            group_of[k] = "concordant"
        for k in sparcal - ssnv:
            group_of[k] = "sparcal_private"
        for k in ssnv - sparcal:
            group_of[k] = "spatialsnv_private"

        loaded[name] = dict(build=cfg["build"], group_of=group_of,
                            merged=os.path.join(REPO, cfg["sparcal_root"], MERGED_GT))
        union = len(sparcal | ssnv)
        size_rows.append(dict(
            sample=name, build=cfg["build"],
            n_sparcal_somatic=len(sparcal), n_spatialsnv=len(ssnv),
            n_concordant=len(conc), n_sparcal_private=len(sparcal - ssnv),
            n_spatialsnv_private=len(ssnv - sparcal), n_union=union,
            jaccard=round(len(conc) / union, 5) if union else "",
            pct_of_sparcal_concordant=round(100 * len(conc) / len(sparcal), 3) if sparcal else "",
            pct_of_spatialsnv_concordant=round(100 * len(conc) / len(ssnv), 3) if ssnv else "",
        ))
        sys.stderr.write(
            f"[{name}] SPARCAL={len(sparcal):,} SpatialSNV={len(ssnv):,} "
            f"concordant={len(conc):,} (J={len(conc)/union:.4f})\n")

    # ---------------- COSMIC, allele-exact, pooled per build ---------------
    per_build: dict[str, set[str]] = defaultdict(set)
    for name in names:
        per_build[loaded[name]["build"]] |= loaded[name]["group_of"].keys()

    cosmic_hits: dict[str, set[str]] = {}
    for build, qkeys in per_build.items():
        by_chrom: dict[str, set[str]] = defaultdict(set)
        for k in qkeys:
            by_chrom[k.split("_", 1)[0]].add(k)
        jobs = [(build, c, by_chrom[c]) for c in chroms if by_chrom.get(c)]
        sys.stderr.write(f"[cosmic:{build}] {len(jobs)} chromosomes, "
                         f"{len(qkeys):,} query alleles\n")
        merged_hits: set[str] = set()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chrom, hits in ex.map(scan_cosmic_chrom, jobs):
                sys.stderr.write(f"[cosmic:{build}] chr{chrom}: {len(hits):,} hits\n")
                merged_hits |= hits
        cosmic_hits[build] = merged_hits

    # ---------------- DP from our own pool, per sample ---------------------
    for name in names:
        d = loaded[name]
        by_chrom: dict[str, set[int]] = defaultdict(set)
        for k in d["group_of"]:
            c, p, _, _ = split_key(k)
            by_chrom[c].add(p)
        jobs = [(d["merged"], c, by_chrom[c]) for c in chroms if by_chrom.get(c)]
        sys.stderr.write(f"[depth:{name}] scanning {len(jobs)} chromosomes\n")
        dpmap: dict[tuple[str, int], float] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chrom, dp in ex.map(scan_depth_chrom, jobs):
                dpmap.update(dp)
        d["dp"] = dpmap
        sys.stderr.write(f"[depth:{name}] DP observed at {len(dpmap):,} positions\n")

    # ---------------- variant-level table ----------------------------------
    var_rows = []
    for name in names:
        d = loaded[name]
        hits = cosmic_hits[d["build"]]
        for k, grp in d["group_of"].items():
            c, p, ref, alt = split_key(k)
            dp = d["dp"].get((c, p))
            var_rows.append(dict(
                sample=name, chrom=c, pos=p, ref=ref, alt=alt, group=grp,
                cosmic_hit=int(k in hits),
                DP="" if dp is None else dp,
                in_pileup_pool=int(dp is not None),
                mhc=int(c == MHC_CHR and MHC_LO <= p <= MHC_HI),
            ))

    # ---------------- coverage availability (the honesty column) -----------
    avail_rows = []
    for name in names:
        for grp in GROUPS:
            sub = [r for r in var_rows if r["sample"] == name and r["group"] == grp]
            n_obs = sum(r["in_pileup_pool"] for r in sub)
            dps = [r["DP"] for r in sub if r["in_pileup_pool"]]
            avail_rows.append(dict(
                sample=name, group=grp, n_total=len(sub), n_dp_observed=n_obs,
                pct_dp_observed=round(100 * n_obs / len(sub), 3) if sub else "",
                median_DP=float(np.median(dps)) if dps else "",
                mean_DP=round(float(np.mean(dps)), 3) if dps else "",
            ))

    # ---------------- raw (unadjusted) rates -------------------------------
    raw_rows = []
    for name in names:
        for subset, keep in (("all", lambda r: True), ("MHC_excl", lambda r: not r["mhc"])):
            for grp in GROUPS:
                sub = [r for r in var_rows
                       if r["sample"] == name and r["group"] == grp and keep(r)]
                obs = [r for r in sub if r["in_pileup_pool"]]
                h = sum(r["cosmic_hit"] for r in sub)
                h_obs = sum(r["cosmic_hit"] for r in obs)
                raw_rows.append(dict(
                    sample=name, subset=subset, group=grp,
                    n=len(sub), n_cosmic=h,
                    rate_pct=round(100 * h / len(sub), 4) if sub else "",
                    n_dp_observed=len(obs), n_cosmic_dp_observed=h_obs,
                    rate_pct_dp_observed=round(100 * h_obs / len(obs), 4) if obs else "",
                ))

    # ---------------- depth-stratified rates -------------------------------
    strat_rows = []
    for name in names:
        for grp in GROUPS:
            sub = [r for r in var_rows if r["sample"] == name and r["group"] == grp
                   and not r["mhc"] and r["in_pileup_pool"]]
            for lab in DP_LABELS:
                g = [r for r in sub if dp_bin(r["DP"]) == lab]
                if not g:
                    continue
                h = sum(r["cosmic_hit"] for r in g)
                strat_rows.append(dict(
                    sample=name, group=grp, depth_bin=lab, n=len(g), n_cosmic=h,
                    rate_pct=round(100 * h / len(g), 4)))

    # ---------------- depth-controlled logistic ----------------------------
    # group is dummy-coded with sparcal_private as reference, so the two
    # coefficients answer the two questions we actually have:
    #   concordant vs SPARCAL-private  -> does agreement add over SPARCAL alone?
    #   SpatialSNV-private vs SPARCAL-private -> baseline difference between tools.
    # A second fit uses spatialsnv_private as reference for the mirror contrast.
    logit_rows = []
    for name in names:
        for subset, keep in (("all", lambda r: True), ("MHC_excl", lambda r: not r["mhc"])):
            sub = [r for r in var_rows
                   if r["sample"] == name and keep(r) and r["in_pileup_pool"]]
            if len(sub) < 100:
                continue
            y = np.array([r["cosmic_hit"] for r in sub], float)
            logdp = np.log10(np.clip(np.array([r["DP"] for r in sub], float), 1, None))
            grp = np.array([r["group"] for r in sub])
            for ref_grp in ("sparcal_private", "spatialsnv_private"):
                others = [g for g in GROUPS if g != ref_grp]
                cols = [np.ones(len(sub))] + [(grp == g).astype(float) for g in others] + [logdp]
                X = np.column_stack(cols)
                if y.sum() < 10 or np.linalg.matrix_rank(X) < X.shape[1]:
                    continue
                beta, se, pv = logit_irls(X, y)
                for i, g in enumerate(others, start=1):
                    # Fisher on the unadjusted 2x2, for the side-by-side comparison
                    a = [r for r in sub if r["group"] == g]
                    b = [r for r in sub if r["group"] == ref_grp]
                    ah, bh = sum(r["cosmic_hit"] for r in a), sum(r["cosmic_hit"] for r in b)
                    _, p_raw = fisher_exact([[ah, len(a) - ah], [bh, len(b) - bh]])
                    logit_rows.append(dict(
                        sample=name, subset=subset, reference=ref_grp, group=g, n=len(sub),
                        n_group=len(a), n_reference=len(b),
                        raw_rate_group_pct=round(100 * ah / len(a), 4) if a else "",
                        raw_rate_reference_pct=round(100 * bh / len(b), 4) if b else "",
                        raw_ratio=round((ah / len(a)) / (bh / len(b)), 4)
                        if a and b and bh else "",
                        fisher_p_unadjusted=f"{p_raw:.3e}",
                        adj_OR=_expc(beta[i]),
                        adj_OR_lo=_expc(beta[i] - 1.96 * se[i]),
                        adj_OR_hi=_expc(beta[i] + 1.96 * se[i]),
                        adj_p=f"{pv[i]:.3e}",
                        logDP_OR=_expc(beta[-1]),
                        logDP_p=f"{pv[-1]:.3e}",
                    ))

    _write(os.path.join(outdir, "concordance_group_sizes.csv"), size_rows)
    _write(os.path.join(outdir, "concordance_coverage_availability.csv"), avail_rows)
    _write(os.path.join(outdir, "concordance_cosmic_raw.csv"), raw_rows)
    _write(os.path.join(outdir, "concordance_depth_stratified.csv"), strat_rows)
    _write(os.path.join(outdir, "concordance_depth_logistic.csv"), logit_rows)
    _write_gz(os.path.join(outdir, "concordance_variant_level.csv.gz"), var_rows)
    sys.stderr.write(f"\nWrote 6 files to {outdir}\n")


def _expc(b: float):
    """exp() that reports separation instead of overflowing. Quasi-complete
    separation is real here (a group with 0 COSMIC hits), and an 'inf' in the CSV
    is the honest answer -- silently clipping it would look like a finite effect."""
    if not np.isfinite(b) or abs(b) > 700:
        return "inf" if b > 0 else "0"
    return round(float(np.exp(b)), 4)


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _write_gz(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
