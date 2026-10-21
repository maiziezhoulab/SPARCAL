#!/usr/bin/env python3
"""
cosmic_somatic_gene_annotation.py  —  2026-07-28

Gene- and variant-level annotation of the COSMIC hits in the CURRENT SPARCAL
variant sets (PAPER_PLAN.md Story C5, item 4.6a/4.6c).

WHY THIS EXISTS
---------------
The manuscript's "reproducible keratinocyte/skin-barrier signature (KRT10, KRT14,
KRT6B, SPINK5, ITGB4) ... KRT6B p.N227S and SPINK5 p.G519= shared by P4 and P6"
comes from `COSMIC/P4_sec1_COSMIC_analysis.md` (2026-05-08), which was computed on
a P4 somatic callset of **8 variants**. The current P4 somatic set has 19,523 calls.
That claim must be recomputed on the current sets or dropped.

INPUT (already computed, no bcftools needed)
--------------------------------------------
`cosmic_amb/isec_{sample}_{cls}/0003.vcf` — the COSMIC records shared with each
SPARCAL class VCF, carrying COSMIC's own INFO annotation (GENE, AA, CDS,
LEGACY_ID, GENOME_SCREEN_SAMPLE_COUNT, IS_CANONICAL, SO_TERM). Builds are
correct per sample: P4/P6 = GRCh37, DCIS1/2 = GRCh38.
`cosmic_amb/cgc_genes.txt` — COSMIC Cancer Gene Census gene symbols.

OUTPUT -> data/somatic_hits_2026-07-28/
---------------------------------------
  cosmic_hits_annotated.csv  every hit x sample x class, with consequence class
  cosmic_gene_counts.csv     per sample x class x gene hit counts
  cosmic_cgc_hits.csv        hits falling in Cancer Gene Census genes
  cscc_shared_p4_p6.csv      genes shared by the two cSCC somatic sets
  dcis_shared_1_2.csv        genes shared by the two DCIS somatic sets
  SUMMARY.md                 the numbers to quote in the paper

FRAMING GUARDRAIL (PAPER_PLAN Decision D2)
------------------------------------------
COSMIC membership is evaluated allele-exactly on the same basis across classes.
Report these as *which genes the retained calls fall in*, NOT as evidence that
the calls are cancer-driving.

Usage (env `snv_caller`, CPU, seconds):
    python scripts/postanalyze/cosmic_somatic_gene_annotation.py
"""
from __future__ import annotations

import os
import re
from collections import Counter, defaultdict

import pandas as pd

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
ISEC_ROOT = os.path.join(PROJECT, "cosmic_amb")
OUT_DIR = os.path.join(PROJECT, "data", "somatic_hits_2026-07-28")
CGC_TSV = "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_CancerGeneCensus_v103_GRCh37.tsv"

SAMPLES = {
    "P4":    ("p4_tumor", "GRCh37", "cSCC"),
    "P6":    ("p6_tumor", "GRCh37", "cSCC"),
    "DCIS1": ("dcis1",    "GRCh38", "DCIS"),
    "DCIS2": ("dcis2",    "GRCh38", "DCIS"),
}
# Paper labels for the code tokens (CLAUDE.md "Variant Categories & Naming").
CLASSES = {"defined": "germline", "upv": "UPV", "somatic": "somatic",
           "ambiguous": "unresolved"}

# Set sizes on the CURRENT sets, from COSMIC/somatic_vs_ambiguous_rates_2026-07-13.csv
# (used only to re-derive hit rates in the summary; the hit counts are recomputed here).
SET_TOTALS = {
    ("P4", "defined"): 42375, ("P4", "upv"): 9491,
    ("P4", "somatic"): 19523, ("P4", "ambiguous"): 175711,
    ("P6", "defined"): 125269, ("P6", "upv"): 1744,
    ("P6", "somatic"): 65655, ("P6", "ambiguous"): 590897,
    ("DCIS1", "defined"): 140773, ("DCIS1", "upv"): 53120,
    ("DCIS1", "somatic"): 18536, ("DCIS1", "ambiguous"): 166825,
    ("DCIS2", "defined"): 129764, ("DCIS2", "upv"): 36323,
    ("DCIS2", "somatic"): 25154, ("DCIS2", "ambiguous"): 226386,
}

INFO_RE = re.compile(r"([^=;]+)=([^;]*)")


def parse_info(info: str) -> dict:
    return dict(INFO_RE.findall(info))


def consequence(aa: str, so_term: str) -> str:
    """Coarse consequence class from COSMIC's AA field.

    COSMIC writes 'p.?' for variants with no protein consequence annotated on the
    chosen transcript (intronic/UTR/intergenic), 'p.X123=' for synonymous, and
    'p.X123Y' for an amino-acid substitution ('*' or 'Ter' = nonsense).
    """
    if not aa or aa in ("p.?", "."):
        return "non_coding_or_unannotated"
    if aa.endswith("="):
        return "synonymous"
    if aa.endswith("*") or aa.endswith("Ter"):
        return "nonsense"
    if so_term and so_term != "SNV":
        return so_term
    return "missense"


def read_isec(sample_key: str, cls: str) -> pd.DataFrame:
    token = SAMPLES[sample_key][0]
    path = os.path.join(ISEC_ROOT, f"isec_{token}_{cls}", "0003.vcf")
    if not os.path.exists(path):
        print(f"  [warn] missing {path}")
        return pd.DataFrame()
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 8:
                continue
            info = parse_info(f[7])
            aa = info.get("AA", "")
            rows.append({
                "sample": sample_key,
                "tumor_type": SAMPLES[sample_key][2],
                "build": SAMPLES[sample_key][1],
                "class_code": cls,
                "class": CLASSES[cls],
                "chrom": f[0], "pos": int(f[1]),
                "ref": f[3], "alt": f[4],
                "cosmic_id": f[2],
                "gene": info.get("GENE", ""),
                "transcript": info.get("TRANSCRIPT", ""),
                "cds": info.get("CDS", ""),
                "aa": aa,
                "so_term": info.get("SO_TERM", ""),
                "is_canonical": info.get("IS_CANONICAL", ""),
                "screen_sample_count": int(info.get("GENOME_SCREEN_SAMPLE_COUNT", 0) or 0),
                "legacy_id": info.get("LEGACY_ID", ""),
                "consequence": consequence(aa, info.get("SO_TERM", "")),
            })
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading COSMIC intersections (current sets) ...")
    frames = []
    for sk in SAMPLES:
        for cls in CLASSES:
            df = read_isec(sk, cls)
            if len(df):
                print(f"  {sk:6s} {CLASSES[cls]:11s} {len(df):>6,} hits")
                frames.append(df)
    hits = pd.concat(frames, ignore_index=True)
    hits.to_csv(os.path.join(OUT_DIR, "cosmic_hits_annotated.csv"), index=False)

    # ── per-gene counts ──────────────────────────────────────────────────────
    gene_counts = (hits[hits.gene != ""]
                   .groupby(["sample", "class", "gene"])
                   .agg(n_hits=("cosmic_id", "size"),
                        n_coding=("consequence",
                                  lambda s: int((~s.isin(
                                      ["non_coding_or_unannotated", "synonymous"])).sum())))
                   .reset_index()
                   .sort_values(["sample", "class", "n_hits"], ascending=[True, True, False]))
    gene_counts.to_csv(os.path.join(OUT_DIR, "cosmic_gene_counts.csv"), index=False)

    # ── Cancer Gene Census ───────────────────────────────────────────────────
    cgc = pd.read_csv(CGC_TSV, sep="\t")
    cgc_genes = set(cgc["GENE_SYMBOL"].dropna())
    tumour_types = dict(zip(cgc["GENE_SYMBOL"], cgc.get("TUMOUR_TYPES_SOMATIC", "")))
    cgc_hits = hits[hits.gene.isin(cgc_genes)].copy()
    cgc_hits["cgc_tumour_types_somatic"] = cgc_hits.gene.map(tumour_types)
    cgc_hits.to_csv(os.path.join(OUT_DIR, "cosmic_cgc_hits.csv"), index=False)

    # ── shared signatures within a tumor type (somatic class only) ───────────
    som = hits[hits.class_code == "somatic"]

    def shared(a: str, b: str, out_name: str) -> pd.DataFrame:
        ga = set(som[(som["sample"] == a) & (som.gene != "")].gene)
        gb = set(som[(som["sample"] == b) & (som.gene != "")].gene)
        both = sorted(ga & gb)
        rec = []
        for g in both:
            sub = som[(som.gene == g) & (som["sample"].isin([a, b]))]
            coding = sub[~sub.consequence.isin(["non_coding_or_unannotated", "synonymous"])]
            rec.append({
                "gene": g,
                "in_cgc": g in cgc_genes,
                f"n_hits_{a}": int((sub["sample"] == a).sum()),
                f"n_hits_{b}": int((sub["sample"] == b).sum()),
                "n_protein_altering": len(coding),
                "shared_positions": sum(
                    1 for _, grp in sub.groupby(["chrom", "pos", "ref", "alt"])
                    if grp["sample"].nunique() == 2),
                "example_aa": "; ".join(sorted(set(coding.aa))[:3]),
            })
        out = pd.DataFrame(rec).sort_values(
            ["in_cgc", "n_protein_altering", "shared_positions"], ascending=False)
        out.to_csv(os.path.join(OUT_DIR, out_name), index=False)
        return out

    cscc = shared("P4", "P6", "cscc_shared_p4_p6.csv")
    dcis = shared("DCIS1", "DCIS2", "dcis_shared_1_2.csv")

    # positions (not just genes) called somatic in BOTH cSCC patients
    pos_key = ["chrom", "pos", "ref", "alt"]
    cs = som[som["sample"].isin(["P4", "P6"])]
    shared_pos = (cs.groupby(pos_key)
                    .filter(lambda g: g["sample"].nunique() == 2)
                    .drop_duplicates(pos_key + ["gene", "aa"]))
    shared_pos.to_csv(os.path.join(OUT_DIR, "cscc_shared_positions_p4_p6.csv"), index=False)

    # ── summary ──────────────────────────────────────────────────────────────
    lines = [
        "# COSMIC gene-level annotation of the CURRENT SPARCAL sets — 2026-07-28",
        "",
        "Recompute of the stale `COSMIC/P4_sec1_COSMIC_analysis.md` (2026-05-08), which was",
        "built on an 8-variant P4 somatic callset. All numbers below are the **current**",
        "(post-dedup) sets — the version locked for the paper (PAPER_PLAN Decision D1).",
        "",
        "⚠️ **Framing (Decision D2):** COSMIC membership is evaluated allele-exactly on",
        "the same basis across classes. Report these as *where the retained calls fall*,",
        "never as evidence that the calls are cancer-driving.",
        "",
        "## Hits and unique genes per class",
        "",
        "| sample | class | set size | COSMIC hits | rate | unique genes | protein-altering |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for sk in SAMPLES:
        for cls, label in CLASSES.items():
            sub = hits[(hits["sample"] == sk) & (hits.class_code == cls)]
            if not len(sub):
                continue
            tot = SET_TOTALS.get((sk, cls))
            rate = f"{100 * len(sub) / tot:.3f}%" if tot else "n/a"
            coding = sub[~sub.consequence.isin(
                ["non_coding_or_unannotated", "synonymous"])]
            lines.append(
                f"| {sk} | {label} | {tot:,} | {len(sub):,} | {rate} | "
                f"{sub.gene.nunique():,} | {len(coding):,} |")

    lines += ["", "## Consequence breakdown of the somatic-class hits", "",
              "| sample | " + " | ".join(
                  sorted(som.consequence.unique())) + " |",
              "|---" * (1 + som.consequence.nunique()) + "|"]
    for sk in SAMPLES:
        sub = som[som["sample"] == sk]
        c = Counter(sub.consequence)
        lines.append(f"| {sk} | " + " | ".join(
            str(c.get(k, 0)) for k in sorted(som.consequence.unique())) + " |")

    lines += ["", "## Cancer Gene Census genes hit by the somatic class", ""]
    for sk in SAMPLES:
        sub = som[(som["sample"] == sk) & (som.gene.isin(cgc_genes))]
        coding = sub[~sub.consequence.isin(
            ["non_coding_or_unannotated", "synonymous"])]
        top = ", ".join(f"{g} ({n})" for g, n in
                        Counter(sub.gene).most_common(12)) or "none"
        lines.append(f"- **{sk}**: {sub.gene.nunique()} CGC genes, {len(sub)} hits "
                     f"({len(coding)} protein-altering). Top: {top}")

    lines += ["", "## Shared between the two cSCC patients (P4 ∩ P6, somatic class)", "",
              f"- **{len(cscc)} genes** shared; "
              f"**{int(cscc.in_cgc.sum())}** are Cancer Gene Census genes.",
              f"- **{len(shared_pos.drop_duplicates(pos_key))} exact positions** "
              f"(chrom/pos/ref/alt) called somatic in both patients.", ""]
    if len(cscc):
        lines += ["| gene | CGC | hits P4 | hits P6 | protein-altering | shared positions |",
                  "|---|---|---:|---:|---:|---:|"]
        for _, r in cscc.head(25).iterrows():
            lines.append(f"| {r.gene} | {'yes' if r.in_cgc else ''} | {r.n_hits_P4} | "
                         f"{r.n_hits_P6} | {r.n_protein_altering} | {r.shared_positions} |")

    lines += ["", "## Shared between the two DCIS sections (somatic class)", "",
              f"- **{len(dcis)} genes** shared; "
              f"**{int(dcis.in_cgc.sum())}** in the Cancer Gene Census.", ""]

    lines += ["", "## Status of the manuscript's old claim", ""]
    for g in ["KRT10", "KRT14", "KRT6B", "SPINK5", "ITGB4", "NOTCH1", "TP53", "CDKN2A"]:
        where = []
        for sk in SAMPLES:
            sub = som[(som["sample"] == sk) & (som.gene == g)]
            if len(sub):
                aas = sorted(set(a for a in sub.aa if a and a != "p.?"))
                where.append(f"{sk} n={len(sub)}" + (f" [{', '.join(aas[:3])}]" if aas else ""))
        lines.append(f"- **{g}**: " + ("; ".join(where) if where
                                       else "**absent from every current somatic set**"))
    lines += ["", "Outputs: `cosmic_hits_annotated.csv`, `cosmic_gene_counts.csv`, "
                  "`cosmic_cgc_hits.csv`, `cscc_shared_p4_p6.csv`, "
                  "`cscc_shared_positions_p4_p6.csv`, `dcis_shared_1_2.csv`.", ""]

    with open(os.path.join(OUT_DIR, "SUMMARY.md"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"\nWrote {OUT_DIR}/SUMMARY.md and 6 CSVs")


if __name__ == "__main__":
    main()
