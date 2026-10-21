#!/usr/bin/env python3
"""Mutational spectrum of every SPARCAL output class (referee item M3).

Per section and class (1000G germline / UPV / somatic / ambiguous):
  * 6-class SBS spectrum, pyrimidine-collapsed (C>A C>G C>T T>A T>C T>G) and the 96-context spectrum
  * UV hallmark: C>T at dipyrimidines (5' base C or T on the pyrimidine strand; SBS7-like)
  * RNA-editing signal: A>G on the transcript strand (gene strand from the GTF), the strand-agnostic
    T>C fraction, and exact overlap with the SComatic editing catalogue
  * cosine similarity of the 96-spectrum to the same section's 1000G class (empirical germline)
Wilson 95% CIs on fractions. P4/P6 are cutaneous SCC (UV-driven); DCIS is breast; the UV hallmark
should therefore separate the cSCC somatic class from both germline and DCIS if the calls are real.

    python scripts/postanalyze/mutational_spectrum_v2.py --out data/mutational_spectrum_2026-10-20
"""
from __future__ import annotations

import argparse
import gzip
import sys
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
sys.path.insert(0, str(PROJECT / "scripts/6_spatial_filter"))
from upv_v1_annotate import load_bed, in_bed  # noqa: E402

SW = Path("/data/maiziezhou_lab/Softwares")
SECTIONS = {
    "P4_rep1": dict(root=PROJECT / "data/P4_tumor/1", build="hg19", tissue="cSCC",
                    fasta=SW / "refdata-hg19-2.1.0/fasta/genome.fa",
                    gtf=SW / "refdata-hg19-2.1.0/genes/gene_annotations.gtf"),
    "P6_rep1": dict(root=PROJECT / "data/P6_tumor/1", build="hg19", tissue="cSCC",
                    fasta=SW / "refdata-hg19-2.1.0/fasta/genome.fa",
                    gtf=SW / "refdata-hg19-2.1.0/genes/gene_annotations.gtf"),
    "DCIS1": dict(root=PROJECT / "data/dcis1", build="hg38", tissue="breast DCIS",
                  fasta=SW / "GRCh38-3.0.0/fasta/genome.fa",
                  gtf=SW / "refdata-GRCh38-2.1.0/genes/gene_annotations.gtf.gz"),
    "DCIS2": dict(root=PROJECT / "data/dcis2", build="hg38", tissue="breast DCIS",
                  fasta=SW / "GRCh38-3.0.0/fasta/genome.fa",
                  gtf=SW / "refdata-GRCh38-2.1.0/genes/gene_annotations.gtf.gz"),
}
EDIT_BED = {"hg19": PROJECT / "resources/artifact_masks/SComatic/AllEditingSites.hg19.bed.gz",
            "hg38": PROJECT / "resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz"}
COMP = str.maketrans("ACGT", "TGCA")
SUBS6 = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
CONTEXT96 = [f"{a}[{s}]{b}" for s in SUBS6 for a in "ACGT" for b in "ACGT"]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def load_classes(root, tag):
    sf = root / "spatial_filter_purity" / tag
    g = pd.read_csv(sf / "germline/germline_variants.txt", sep="\t", dtype={"chrom": str})
    return {"1000G": g[g.race == "defined"], "UPV": g[g.race == "denovo"],
            "somatic": pd.read_csv(sf / "somatic/somatic_variants.txt", sep="\t", dtype={"chrom": str}),
            "ambiguous": pd.read_csv(sf / "ambiguous/ambiguous_variants.txt", sep="\t", dtype={"chrom": str})}


def gene_strands(gtf):
    """chrom(no chr) -> (sorted starts, ends, strands) of gene records."""
    op = gzip.open if str(gtf).endswith(".gz") else open
    rows = defaultdict(list)
    with op(gtf, "rt") as h:
        for line in h:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if len(f) < 8 or f[2] != "gene":
                continue
            rows[f[0].removeprefix("chr")].append((int(f[3]), int(f[4]), f[6]))
    out = {}
    for c, r in rows.items():
        r.sort(); out[c] = ([x[0] for x in r], [x[1] for x in r], [x[2] for x in r])
    return out


def strand_at(index, chrom, pos):
    """'+', '-', or '' (intergenic or genes on both strands)."""
    d = index.get(chrom)
    if not d:
        return ""
    st, en, sd = d
    i = bisect_right(st, pos)
    found = {sd[j] for j in range(max(0, i - 200), i) if en[j] >= pos}
    return found.pop() if len(found) == 1 else ""


def annotate(df, fasta, contig, strands, edit_bed):
    recs = []
    for chrom, pos, ref, alt in df[["chrom", "pos", "ref", "alt"]].itertuples(index=False):
        ref, alt = ref.upper(), alt.upper(); pos = int(pos); c = str(chrom).removeprefix("chr")
        if len(ref) != 1 or len(alt) != 1 or ref not in "ACGT" or alt not in "ACGT":
            continue
        tri = fasta.fetch(contig(c), pos - 2, pos + 1).upper()
        if len(tri) != 3 or tri[1] != ref or "N" in tri:
            recs.append(dict(ok=False)); continue
        if ref in "GA":                                   # move to the pyrimidine strand
            tri = tri.translate(COMP)[::-1]; r, a = ref.translate(COMP), alt.translate(COMP)
        else:
            r, a = ref, alt
        sub = f"{r}>{a}"
        g = strand_at(strands, c, pos)
        tx_ref, tx_alt = (ref, alt) if g == "+" else (ref.translate(COMP), alt.translate(COMP)) if g == "-" else ("", "")
        recs.append(dict(ok=True, sub=sub, ctx=f"{tri[0]}[{sub}]{tri[2]}",
                         uv=(sub == "C>T") and tri[0] in "CT",
                         a2g_tx=(tx_ref == "A" and tx_alt == "G"), stranded=bool(g),
                         edit_cat=in_bed(edit_bed, "chr" + c, pos - 1)))
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS), choices=list(SECTIONS))
    ap.add_argument("--tag", default="baseQ0mapQ0")
    ap.add_argument("--section-root-base", default=None,
                    help="e.g. data/freeze_v1 to read <base>/<P4_tumor/1|P6_tumor/1|dcis1|dcis2> instead")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)

    s6, s96, diag, checks = [], [], [], []
    strand_cache, bed_cache = {}, {}
    for name in a.sections:
        cfg = SECTIONS[name]
        root = cfg["root"] if a.section_root_base is None else \
            PROJECT / a.section_root_base / cfg["root"].relative_to(PROJECT / "data")
        fasta = pysam.FastaFile(str(cfg["fasta"]))
        refs = set(fasta.references)
        contig = (lambda c: c if c in refs else "chr" + c) if "1" in refs else (lambda c: "chr" + c)
        if cfg["gtf"] not in strand_cache:
            strand_cache[cfg["gtf"]] = gene_strands(cfg["gtf"])
        if cfg["build"] not in bed_cache:
            bed_cache[cfg["build"]] = load_bed(str(EDIT_BED[cfg["build"]]))
        classes = load_classes(root, a.tag)
        germ96 = None
        for cls in ["1000G", "UPV", "somatic", "ambiguous"]:
            df = classes[cls]
            ann = annotate(df, fasta, contig, strand_cache[cfg["gtf"]], bed_cache[cfg["build"]])
            bad = int((~ann.ok).sum()) if len(ann) else 0
            ann = ann[ann.ok] if len(ann) else ann
            n = len(ann)
            c6 = ann["sub"].value_counts().reindex(SUBS6, fill_value=0) if n else pd.Series(0, index=SUBS6)
            c96 = ann["ctx"].value_counts().reindex(CONTEXT96, fill_value=0) if n else pd.Series(0, index=CONTEXT96)
            if cls == "1000G":
                germ96 = c96.to_numpy(float)
            v = c96.to_numpy(float)
            cos = float(v @ germ96 / (np.linalg.norm(v) * np.linalg.norm(germ96))) if n and germ96 is not None else np.nan
            uv = wilson(int(ann.uv.sum()) if n else 0, n)
            ns = int(ann.stranded.sum()) if n else 0
            a2g = wilson(int(ann.a2g_tx.sum()) if n else 0, ns)
            tc = wilson(int(c6["T>C"]), n)
            ed = wilson(int(ann.edit_cat.sum()) if n else 0, n)
            for s, k in c6.items():
                s6.append(dict(section=name, tissue=cfg["tissue"], cls=cls, sub=s, count=int(k), frac=k / n if n else np.nan))
            for s, k in c96.items():
                s96.append(dict(section=name, cls=cls, context=s, count=int(k)))
            diag.append(dict(section=name, tissue=cfg["tissue"], cls=cls, class_size=len(df), n_snv=n,
                             ref_mismatch_or_N=bad, uv_frac=uv[0], uv_lo=uv[1], uv_hi=uv[2],
                             a2g_tx_frac=a2g[0], a2g_tx_lo=a2g[1], a2g_tx_hi=a2g[2], n_stranded=ns,
                             t2c_frac=tc[0], t2c_lo=tc[1], t2c_hi=tc[2],
                             editing_catalogue_frac=ed[0], editing_lo=ed[1], editing_hi=ed[2],
                             cosine_to_1000G=cos))
        # hand-check five random somatic contexts against the FASTA
        som = classes["somatic"].sample(min(5, len(classes["somatic"])), random_state=1)
        for chrom, pos, ref, alt in som[["chrom", "pos", "ref", "alt"]].itertuples(index=False):
            c = str(chrom).removeprefix("chr"); pos = int(pos)
            checks.append(dict(section=name, variant=f"{c}:{pos}{ref}>{alt}",
                               fasta_tri=fasta.fetch(contig(c), pos - 2, pos + 1).upper(), ref=ref))
        print(f"[{name}] done", file=sys.stderr)

    pd.DataFrame(s6).to_csv(out / "spectrum_6.csv", index=False)
    pd.DataFrame(s96).to_csv(out / "spectrum_96.csv", index=False)
    d = pd.DataFrame(diag); d.to_csv(out / "diagnostics.csv", index=False)
    pd.DataFrame(checks).to_csv(out / "fasta_spot_checks.csv", index=False)
    (out / "run_meta.txt").write_text(f"tag={a.tag}\nsection_root_base={a.section_root_base}\n")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(d[["section", "cls", "class_size", "n_snv", "ref_mismatch_or_N", "uv_frac", "uv_lo", "uv_hi",
                 "a2g_tx_frac", "t2c_frac", "editing_catalogue_frac", "cosine_to_1000G"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
