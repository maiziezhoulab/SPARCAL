#!/usr/bin/env python3
"""Same spectrum diagnostics (mutational_spectrum_v2.py) for the other callers' SOMATIC callsets.

Question: is the low UV / high editing signal specific to SPARCAL's somatic class, or shared by
every RNA-based somatic caller on the same BAMs? Callsets: Monopogen somatic (putativeSNVs),
SpatialSNV (FilterMutectCalls PASS biallelic SNVs), SpaceTracer final calls (DCIS only).
WES-confirmed P4/P6 mutations are the DNA positive control for the UV hallmark.

    python scripts/postanalyze/spectrum_other_callers.py --out data/mutational_spectrum_2026-10-20_callers
"""
from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import pandas as pd
import pysam

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
sys.path.insert(0, str(PROJECT / "scripts/postanalyze"))
sys.path.insert(0, str(PROJECT / "scripts/6_spatial_filter"))
import mutational_spectrum_v2 as ms  # noqa: E402
from upv_v1_annotate import load_bed  # noqa: E402

VIEWER = Path("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer")
SAMPLES = {  # sample -> (spectrum section config key, Monopogen tree, SpatialSNV id, SpaceTracer vcf)
    "P4_rep1": ("P4_rep1", "Monopogen_P4_rep1/out/classified/P4_rep1.somatic.csv", "p4", None),
    "P6_rep1": ("P6_rep1", "Monopogen_P6_rep1/out/classified/P6_rep1.somatic.csv", "p6", None),
    "DCIS1": ("DCIS1", "Monopogen_DCIS1/out/classified/DCIS1.somatic.csv", "dcis1",
              VIEWER / "DCIS_1_SpaceTracer/spacetracer_calls.vcf.gz"),
    "DCIS2": ("DCIS2", "Monopogen_DCIS2/out/classified/DCIS2.somatic.csv", "dcis2",
              VIEWER / "DCIS_2_SpaceTracer/spacetracer_calls.vcf.gz"),
}


def vcf_snvs(path, pass_only=True):
    rows = []
    with gzip.open(path, "rt") as h:
        for line in h:
            if line.startswith("#"):
                continue
            f = line.split("\t", 8)
            if pass_only and f[6] != "PASS":
                continue
            for alt in f[4].split(","):
                if len(f[3]) == 1 and len(alt) == 1:
                    rows.append((f[0], int(f[1]), f[3], alt))
    return pd.DataFrame(rows, columns=["chrom", "pos", "ref", "alt"]).drop_duplicates()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sparcal-tag", default="baseQ0mapQ0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True)
    strands, beds, rows = {}, {}, []
    wes = pd.read_csv(PROJECT / "references/STmut_Table_S1_hg19_autosomal_snvs.tsv", sep="\t")
    for sample, (sec, mono, ssnv, st) in SAMPLES.items():
        cfg = ms.SECTIONS[sec]
        fasta = pysam.FastaFile(str(cfg["fasta"])); refs = set(fasta.references)
        contig = (lambda c, refs=refs: c if c in refs else "chr" + c) if "1" in refs else (lambda c: "chr" + c)
        strands.setdefault(cfg["gtf"], ms.gene_strands(cfg["gtf"]))
        beds.setdefault(cfg["build"], load_bed(str(ms.EDIT_BED[cfg["build"]])))
        sp = ms.load_classes(cfg["root"], a.sparcal_tag)
        sets = {"SPARCAL somatic": sp["somatic"], "SPARCAL 1000G germline": sp["1000G"],
                "Monopogen somatic": pd.read_csv(PROJECT / mono, dtype={"chr": str}).rename(columns={"chr": "chrom"}),
                "SpatialSNV PASS": vcf_snvs(PROJECT / f"SpatialSNV/results/{ssnv}/{ssnv}.vcf.gz")}
        if st is not None and st.exists():
            sets["SpaceTracer final"] = vcf_snvs(st)
        if sample in ("P4_rep1", "P6_rep1"):
            w = wes[wes["sample"] == sample[:2]].rename(columns={"chrom_hg19": "chrom", "pos_hg19": "pos"})
            sets["WES-confirmed (DNA truth)"] = w[["chrom", "pos", "ref", "alt"]]
        autosomes = {str(i) for i in range(1, 23)}
        for name, df in sets.items():
            df = df[df["chrom"].astype(str).str.removeprefix("chr").isin(autosomes)]   # like-for-like with SPARCAL/WES
            ann = ms.annotate(df[["chrom", "pos", "ref", "alt"]], fasta, contig, strands[cfg["gtf"]], beds[cfg["build"]])
            ann = ann[ann.ok] if len(ann) else ann
            n = len(ann); ns = int(ann.stranded.sum()) if n else 0
            uv = ms.wilson(int(ann.uv.sum()) if n else 0, n)
            tc = ms.wilson(int((ann["sub"] == "T>C").sum()) if n else 0, n)
            ed = ms.wilson(int(ann.edit_cat.sum()) if n else 0, n)
            a2g = ms.wilson(int(ann.a2g_tx.sum()) if n else 0, ns)
            rows.append(dict(sample=sample, tissue=cfg["tissue"], callset=name, n_snv=n,
                             uv_frac=uv[0], uv_lo=uv[1], uv_hi=uv[2], t2c_frac=tc[0],
                             a2g_tx_frac=a2g[0], editing_catalogue_frac=ed[0], editing_lo=ed[1], editing_hi=ed[2]))
        print(f"[{sample}] done", file=sys.stderr)
    d = pd.DataFrame(rows); d.to_csv(out / "caller_spectrum_diagnostics.csv", index=False)
    print(d[["sample", "callset", "n_snv", "uv_frac", "t2c_frac", "a2g_tx_frac", "editing_catalogue_frac"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
