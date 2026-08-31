#!/usr/bin/env python3
"""P1-1 three-way germline concordance: SPARCAL vs GATK4 vs Strelka2 vs matched-normal WES.

Uses the truth table already built by p1_1_germline_concordance.py
(data/germline_concordance_2026-08-23/truth_site_details.csv.gz -- WES truth
GTs, RNA depth, SPARCAL detection/GT) and adds two independent RNA germline
callers run on the IDENTICAL site set
(scripts/postanalyze/run_gatk_strelka2_targeted.sh -- GATK4 HaplotypeCaller
and Strelka2 germline workflow, both restricted via -L/--callRegions to the
exact truth-union-SPARCAL target positions, both on the same whole-section
RNA BAM), so the P1-1 comparison is three-way rather than self-referential.

Non-destructive: reads existing outputs read-only, writes only under the P1
task output directory.
"""
from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
TRUTH_DETAILS = PROJECT / "data/germline_concordance_2026-08-23/truth_site_details.csv.gz"
THREE_WAY_ROOT = PROJECT / "data/germline_and_contrasts_2026-08-27/three_way_calls"
DEPTH_ORDER = ["0", "1-3", "4-9", "10-29", "30+"]


def norm_chrom(c: str) -> str:
    c = str(c)
    return "chr" + c[3:] if c.startswith("chr") else "chr" + c


def load_caller_gt(path: Path) -> dict[tuple, str]:
    """Parse a targeted GATK/Strelka2 VCF into {(chrom,pos,ref,alt): gt_0/1_1/1_etc}.

    Handles multiallelic records: the GT index of the matching ALT determines
    whether that specific allele is present (het if GT has exactly one copy of
    that allele's index, hom if two).
    """
    out = {}
    if not path.exists():
        return out
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 10:
                continue
            chrom, pos, ref, alt_field = f[0], int(f[1]), f[3], f[4]
            if alt_field in (".", ""):
                continue
            alts = alt_field.split(",")
            fmt_keys = f[8].split(":")
            sample = f[9].split(":")
            fmt = dict(zip(fmt_keys, sample))
            gt_raw = fmt.get("GT", ".")
            gt_raw = gt_raw.replace("|", "/")
            if gt_raw in (".", "./.", ".|."):
                continue
            try:
                alleles = [int(x) for x in gt_raw.split("/") if x != "."]
            except ValueError:
                continue
            if not alleles:
                continue
            for i, alt in enumerate(alts, start=1):
                if len(ref) != 1 or len(alt) != 1:
                    continue
                count = alleles.count(i)
                if count == 0:
                    gt = "0/0"
                elif count == len(alleles):
                    gt = "1/1"
                else:
                    gt = "0/1"
                key = (norm_chrom(chrom), pos, ref.upper(), alt.upper())
                out[key] = gt
    return out


def wilson(k, n):
    if not n:
        return math.nan, math.nan
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - half, center + half


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    truth = pd.read_csv(TRUTH_DETAILS)
    truth["key"] = list(zip(truth.chrom.map(norm_chrom), truth.pos, truth.ref.str.upper(), truth.alt.str.upper()))

    caller_gts = {}
    for sample in ["P4", "P6"]:
        caller_gts[(sample, "GATK")] = load_caller_gt(THREE_WAY_ROOT / sample / "gatk" / f"{sample}_gatk_targeted.vcf.gz")
        caller_gts[(sample, "Strelka2")] = load_caller_gt(THREE_WAY_ROOT / sample / "strelka2" / f"{sample}_strelka2_targeted.vcf.gz")
        print(f"[{sample}] GATK targeted calls parsed: {len(caller_gts[(sample,'GATK')])}")
        print(f"[{sample}] Strelka2 targeted calls parsed: {len(caller_gts[(sample,'Strelka2')])}")

    rows = []
    for _, r in truth.iterrows():
        sample = r["sample"]
        if sample not in ("P4", "P6"):
            continue
        base = dict(sample=sample, chrom=r.chrom, pos=r.pos, ref=r.ref, alt=r.alt,
                    panel=r.panel, wes_gt=r.wes_gt, rna_dp=r.rna_dp, rna_depth_bin=r.rna_depth_bin)
        # SPARCAL
        rows.append({**base, "caller": "SPARCAL",
                    "detected": int(r.detected), "call_gt": r.raw_gt if r.detected else None})
        # GATK / Strelka2
        for caller in ["GATK", "Strelka2"]:
            gt = caller_gts.get((sample, caller), {}).get(r["key"])
            detected = int(gt is not None and gt != "0/0")
            rows.append({**base, "caller": caller, "detected": detected, "call_gt": gt})

    long_df = pd.DataFrame(rows)
    long_df["rna_depth_bin"] = pd.Categorical(long_df["rna_depth_bin"], DEPTH_ORDER, ordered=True)

    # ---- three-way summary by depth bin x panel x caller ----
    summary_rows = []
    expanded = pd.concat([long_df, long_df.assign(panel="all")], ignore_index=True)
    for (sample, caller, panel, dbin), g in expanded.groupby(
            ["sample", "caller", "panel", "rna_depth_bin"], observed=True):
        n = len(g)
        detected = int(g.detected.sum())
        lo, hi = wilson(detected, n)
        dg = g[g.detected == 1]
        gt_ok = dg[dg.call_gt.isin(["0/1", "1/1"]) & (dg.call_gt == dg.wes_gt)]
        n_gt_eval = int(dg.call_gt.isin(["0/1", "1/1"]).sum())
        summary_rows.append({
            "sample": sample, "caller": caller, "panel": panel, "rna_depth_bin": dbin,
            "n_wes_truth": n, "n_detected": detected,
            "sensitivity": detected / n if n else math.nan,
            "sensitivity_ci_low": lo, "sensitivity_ci_high": hi,
            "n_gt_evaluable": n_gt_eval,
            "gt_accuracy": len(gt_ok) / n_gt_eval if n_gt_eval else math.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary["rna_depth_bin"] = pd.Categorical(summary["rna_depth_bin"], DEPTH_ORDER, ordered=True)
    summary = summary.sort_values(["sample", "panel", "rna_depth_bin", "caller"])
    summary.to_csv(args.outdir / "concordance_three_way.csv", index=False)

    # ---- confusion matrices: WES truth GT (0/1,1/1) vs each caller's call GT (incl not-detected/0-0) ----
    # Reported at two depth strata: all callable sites (depth=0 dominates, since no
    # method can call an uncovered site), and RNA depth>=10 (the regime the headline
    # sensitivity/accuracy numbers describe).
    conf_rows = []
    for depth_stratum, base in [("all_depths", long_df), ("rna_dp_ge10", long_df[long_df.rna_dp >= 10])]:
        for (sample, caller), g in base.groupby(["sample", "caller"], observed=True):
            g = g.copy()
            g["call_bucket"] = g["call_gt"].where(g["call_gt"].isin(["0/1", "1/1"]), other=None)
            g.loc[g.detected == 0, "call_bucket"] = "not_detected"
            g["call_bucket"] = g["call_bucket"].fillna("called_but_ambiguous")
            for wes_gt, gg in g.groupby("wes_gt"):
                counts = gg["call_bucket"].value_counts()
                for bucket, n in counts.items():
                    conf_rows.append({"sample": sample, "caller": caller, "depth_stratum": depth_stratum,
                                      "wes_truth_gt": wes_gt, "call_bucket": bucket, "n": int(n)})
    confusion = pd.DataFrame(conf_rows)
    confusion.to_csv(args.outdir / "concordance_confusion.csv", index=False)

    # also refresh concordance_by_depth.csv (SPARCAL-only view, copied through for the
    # requested filename, identical content to germline_concordance_2026-08-23's file
    # but regenerated from this run's merged long table for provenance).
    sparcal_only = summary[summary.caller == "SPARCAL"]
    sparcal_only.to_csv(args.outdir / "concordance_by_depth.csv", index=False)

    print(summary.to_string(index=False))
    print(f"\nWrote three-way P1-1 package to {args.outdir}")


if __name__ == "__main__":
    main()
