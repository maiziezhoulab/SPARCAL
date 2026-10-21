#!/usr/bin/env python
"""COSMIC hit rates partitioned INTO vs OUT OF the extended MHC (chr6:28-34 Mb).

Fig. 5b currently reports the somatic-vs-unresolved ratio raw and with xMHC
REMOVED. That answers "does the contrast survive without the MHC?" but never
shows what is happening inside the region it removes. This script reports the
xMHC interval as its own stratum so the exclusion can be justified from the
numbers rather than asserted.

For every sample x class it computes, separately inside and outside xMHC:
  - set size, COSMIC hits, hit rate
and then the somatic-vs-unresolved contrast (ratio, 95% Katz CI, one-sided
Fisher) within each stratum, plus the within-class MHC-vs-nonMHC enrichment.

Coordinates: 28-34 Mb is applied on both builds (GRCh37 for P4/P6, GRCh38 for
DCIS1/2), matching fig5_cosmic_cascade.py. The xMHC shifts by ~1 Mb between
builds, so this is an approximation held constant across samples on purpose.

INPUTS (same as fig5_cosmic_cascade.py):
  cosmic_amb/{sample}_{class}_nochr.vcf.gz        -- full per-class call sets
  cosmic_amb/isec_{sample}_{class}/0003.vcf       -- their COSMIC-hit subset

OUTPUT:
  data/paper_figs_2026-07-29/cosmic_mhc_partition_by_class.csv
  data/paper_figs_2026-07-29/cosmic_mhc_partition_contrast.csv

Run (env snv_caller, CPU, ~1 min): python scripts/postanalyze/cosmic_mhc_partition.py
"""
import os
import subprocess

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
AMB_DIR = os.path.join(PROJECT, "cosmic_amb")
BCFTOOLS = os.path.join(PROJECT, "apps", "bcftools")
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-07-29")
os.makedirs(DERIVED_DIR, exist_ok=True)

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
SAMPLE_TOKEN = {"P4": "p4_tumor", "P6": "p6_tumor", "DCIS1": "dcis1", "DCIS2": "dcis2"}
CLASS_ORDER = ["defined", "upv", "somatic", "ambiguous"]
CLASS_LABEL = {"defined": "germline (1kG)", "upv": "UPV", "somatic": "somatic",
               "ambiguous": "unresolved"}

MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000


def _in_mhc(chrom, pos):
    return chrom == MHC_CHR and MHC_LO <= pos <= MHC_HI


def _count_vcf(path):
    """Return (total, in_mhc) record counts for a bgzipped VCF."""
    p = subprocess.run([BCFTOOLS, "view", "-H", path],
                       capture_output=True, text=True, check=True)
    tot = mhc = 0
    for line in p.stdout.splitlines():
        if not line:
            continue
        c = line.split("\t", 2)
        tot += 1
        if _in_mhc(c[0], int(c[1])):
            mhc += 1
    return tot, mhc


def _count_isec(path):
    """Return (total, in_mhc) record counts for a plain isec output VCF."""
    if not os.path.exists(path):
        return 0, 0
    tot = mhc = 0
    with open(path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t", 2)
            tot += 1
            if _in_mhc(c[0], int(c[1])):
                mhc += 1
    return tot, mhc


def _ratio_ci(h1, n1, h2, n2):
    """95% CI on rate1/rate2 via the log-ratio (Katz) normal approximation."""
    if min(h1, h2) == 0 or min(n1, n2) == 0:
        return np.nan, np.nan
    p1, p2 = h1 / n1, h2 / n2
    logr = np.log(p1 / p2)
    se = np.sqrt((1 - p1) / (p1 * n1) + (1 - p2) / (p2 * n2))
    return np.exp(logr - 1.96 * se), np.exp(logr + 1.96 * se)


def _contrast(sh, st, ah, at, alternative="greater"):
    """Somatic-vs-unresolved rate ratio with CI and Fisher p."""
    if st == 0 or at == 0 or ah == 0:
        return dict(ratio=np.nan, ci_lo=np.nan, ci_hi=np.nan, p_value=np.nan)
    ratio = (sh / st) / (ah / at)
    _, p = fisher_exact([[sh, st - sh], [ah, at - ah]], alternative=alternative)
    lo, hi = _ratio_ci(sh, st, ah, at)
    return dict(ratio=ratio, ci_lo=lo, ci_hi=hi, p_value=p)


def main():
    per_class, contrasts = [], []

    for sk in SAMPLES:
        tok = SAMPLE_TOKEN[sk]
        counts = {}
        for cls in CLASS_ORDER:
            tot, tot_mhc = _count_vcf(os.path.join(AMB_DIR, f"{tok}_{cls}_nochr.vcf.gz"))
            hit, hit_mhc = _count_isec(os.path.join(AMB_DIR, f"isec_{tok}_{cls}", "0003.vcf"))
            counts[cls] = dict(tot=tot, tot_mhc=tot_mhc, hit=hit, hit_mhc=hit_mhc)

            in_n, in_h = tot_mhc, hit_mhc
            out_n, out_h = tot - tot_mhc, hit - hit_mhc
            enr = _contrast(in_h, in_n, out_h, out_n, alternative="two-sided")
            per_class.append({
                "sample": sk, "class_code": cls, "class_label": CLASS_LABEL[cls],
                "n_total": tot, "n_hits": hit,
                "rate_all_pct": 100 * hit / tot if tot else np.nan,
                "n_total_mhc": in_n, "n_hits_mhc": in_h,
                "rate_mhc_pct": 100 * in_h / in_n if in_n else np.nan,
                "n_total_nonmhc": out_n, "n_hits_nonmhc": out_h,
                "rate_nonmhc_pct": 100 * out_h / out_n if out_n else np.nan,
                "pct_of_set_in_mhc": 100 * in_n / tot if tot else np.nan,
                "pct_of_hits_in_mhc": 100 * in_h / hit if hit else np.nan,
                "mhc_vs_nonmhc_ratio": enr["ratio"],
                "mhc_vs_nonmhc_ci_lo": enr["ci_lo"],
                "mhc_vs_nonmhc_ci_hi": enr["ci_hi"],
                "mhc_vs_nonmhc_p": enr["p_value"],
            })

        s, a = counts["somatic"], counts["ambiguous"]
        strata = {
            "all": (s["hit"], s["tot"], a["hit"], a["tot"]),
            "mhc_only": (s["hit_mhc"], s["tot_mhc"], a["hit_mhc"], a["tot_mhc"]),
            "nonmhc": (s["hit"] - s["hit_mhc"], s["tot"] - s["tot_mhc"],
                       a["hit"] - a["hit_mhc"], a["tot"] - a["tot_mhc"]),
        }
        for stratum, (sh, st, ah, at) in strata.items():
            row = dict(sample=sk, stratum=stratum, somatic_hit=sh, somatic_tot=st,
                       unresolved_hit=ah, unresolved_tot=at,
                       somatic_rate_pct=100 * sh / st if st else np.nan,
                       unresolved_rate_pct=100 * ah / at if at else np.nan)
            row.update(_contrast(sh, st, ah, at))
            contrasts.append(row)

    df_class = pd.DataFrame(per_class)
    df_contrast = pd.DataFrame(contrasts)

    p1 = os.path.join(DERIVED_DIR, "cosmic_mhc_partition_by_class.csv")
    p2 = os.path.join(DERIVED_DIR, "cosmic_mhc_partition_contrast.csv")
    df_class.to_csv(p1, index=False)
    df_contrast.to_csv(p2, index=False)

    pd.set_option("display.width", 200, "display.max_columns", 50)

    print("\n=== Per class: COSMIC hit rate inside vs outside xMHC (chr6:28-34 Mb) ===")
    print(df_class[["sample", "class_label", "n_total", "pct_of_set_in_mhc",
                    "pct_of_hits_in_mhc", "rate_mhc_pct", "rate_nonmhc_pct",
                    "mhc_vs_nonmhc_ratio", "mhc_vs_nonmhc_p"]].round(3).to_string(index=False))

    print("\n=== Somatic vs unresolved, by stratum ===")
    print(df_contrast[["sample", "stratum", "somatic_hit", "somatic_tot",
                       "somatic_rate_pct", "unresolved_hit", "unresolved_tot",
                       "unresolved_rate_pct", "ratio", "ci_lo", "ci_hi",
                       "p_value"]].round(4).to_string(index=False))

    print(f"\nwrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
