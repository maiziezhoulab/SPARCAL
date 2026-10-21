#!/usr/bin/env python
"""Does HLA divergence from GRCh38 explain where SPARCAL somatic-class calls land?

Reference-divergence predicts a dose-response: the HLA genes whose typed alleles
differ MOST from the GRCh38 reference allele should accumulate the MOST
reference-relative "somatic" calls.  This joins the arcasHLA genotypes
(data/hla_typing_2026-09-07/) to the per-locus call counts
(data/hla_immune_escape_2026-08-22/hla_locus_summary.csv) and checks.

Divergence is scored at the ALLELE GROUP (first field) level -- the field that
low-depth RNA typing actually resolves reliably.  0 = both alleles in the
reference's group, 1 = one differs, 2 = both differ.

CAVEATS
  * arcasHLA was run on Visium RNA at modest depth.  First-field calls are
    trustworthy; 3rd/4th-field designations are not.  Several DCIS2 calls
    (A*01:436Q, B*07:02:113, C*07:744, DPB1*1686:01N, DPB1*1954:01Q,
    DQB1*02:251N) carry implausible subtype numbers and N/Q suffixes and are
    low-confidence beyond the first field.
  * The typing fastq excluded unmapped reads and alt/decoy contigs, which biases
    toward the reference allele -- divergence here is a LOWER bound.
  * GRCh38 primary assembly carries the PGF haplotype; those reference alleles
    are hard-coded below.

Read-only; writes only under --out-dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
GENO = f"{REPO}/data/hla_typing_2026-09-07"
LOCUS = f"{REPO}/data/hla_immune_escape_2026-08-22/hla_locus_summary.csv"

# GRCh38 primary assembly = PGF haplotype
REF = {"A": "03", "B": "07", "C": "07", "DRA": "01", "DRB1": "15", "DRB5": "01",
       "DQA1": "01", "DQB1": "06", "DPA1": "01", "DPB1": "04", "E": "01",
       "F": "01", "G": "01", "DMA": "01", "DMB": "01", "DOA": "01", "DOB": "01"}
SAMPLES = ["DCIS1", "DCIS2"]


def first_field(allele):
    # "B*57:03:01" -> "57"
    return allele.split("*", 1)[1].split(":")[0] if "*" in allele else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    calls = {}
    if os.path.exists(LOCUS):
        with open(LOCUS) as fh:
            for r in csv.DictReader(fh):
                g = r["gene"].replace("HLA-", "")
                calls[(r["sample"], g)] = int(r["n_calls"])

    genos = {}
    for s in SAMPLES:
        p = os.path.join(GENO, s, "hla.genotype.json")
        if os.path.exists(p):
            genos[s] = json.load(open(p))

    # Are the two sections the same individual?
    if len(genos) == 2:
        print("=== Are DCIS1 and DCIS2 the same individual? ===")
        print(f"  {'gene':6}{'DCIS1':34}{'DCIS2':34}match")
        same = diff = 0
        for g in ("A", "B", "C", "DRB1", "DQA1", "DQB1", "DPA1", "DPB1"):
            a1 = genos["DCIS1"].get(g, []); a2 = genos["DCIS2"].get(g, [])
            if not a1 or not a2:
                continue
            f1 = sorted(first_field(x) for x in a1)
            f2 = sorted(first_field(x) for x in a2)
            ok = f1 == f2
            same += ok; diff += not ok
            print(f"  {g:6}{','.join(a1):34}{','.join(a2):34}{'YES' if ok else 'NO'}")
        print(f"  -> {same} loci match, {diff} differ at the allele-group level.")
        print("  HLA type is fixed per individual, so group-level mismatch at the "
              "classical loci\n     means DIFFERENT INDIVIDUALS "
              "(answers referee item M10).\n")

    rows = []
    print("=== Divergence from GRCh38 vs number of somatic-class calls ===")
    print(f"  {'sample':7}{'gene':6}{'typed alleles':38}{'ref':5}{'divergence':>11}{'n_calls':>9}")
    for s in SAMPLES:
        if s not in genos:
            continue
        for g, alleles in sorted(genos[s].items()):
            if g not in REF:
                continue
            d = sum(1 for x in alleles if first_field(x) != REF[g])
            n = calls.get((s, g), 0)
            print(f"  {s:7}{g:6}{','.join(alleles):38}{REF[g]:5}{d:>11}{n:>9}")
            rows.append(dict(sample=s, gene=g, alleles=";".join(alleles),
                             ref_group=REF[g], divergence_0to2=d, n_calls=n))

    print("\n=== The dose-response test ===")
    for g in ("A", "B", "C", "DRB1", "DQA1", "DPB1"):
        r = [x for x in rows if x["gene"] == g]
        if len(r) != 2:
            continue
        r.sort(key=lambda x: x["sample"])
        print(f"  HLA-{g:5} DCIS1 div={r[0]['divergence_0to2']} calls={r[0]['n_calls']:<4}"
              f"  DCIS2 div={r[1]['divergence_0to2']} calls={r[1]['n_calls']:<4}", end="")
        if r[0]["divergence_0to2"] == r[1]["divergence_0to2"]:
            print("   (equal divergence)")
        else:
            more_div = r[0] if r[0]["divergence_0to2"] > r[1]["divergence_0to2"] else r[1]
            less_div = r[1] if more_div is r[0] else r[0]
            verdict = ("CONSISTENT with divergence"
                       if more_div["n_calls"] > less_div["n_calls"]
                       else "AGAINST divergence" if more_div["n_calls"] < less_div["n_calls"]
                       else "tie")
            print(f"   -> {verdict}")

    p = os.path.join(a.out_dir, "12_hla_divergence_vs_calls.csv")
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
