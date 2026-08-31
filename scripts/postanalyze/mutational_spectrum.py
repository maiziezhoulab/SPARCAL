#!/usr/bin/env python3
"""Mutational spectrum of every SPARCAL output class (PAPER_WORK P1-2 / X-3).

For each of the four tumour sections (P4, P6, DCIS1, DCIS2) and each of the four
mutually exclusive output classes (germline, UPV, retained somatic, unresolved),
computes:

  * 6-channel strand-collapsed substitution spectrum (counts + fractions)
  * 96-channel trinucleotide (SigProfiler-style, pyrimidine-strand-collapsed)
    spectrum, using the reference FASTA appropriate to that sample's genome build
  * the A>G + T>C fraction (the ADAR-editing diagnostic)
  * overlap with the SComatic AllEditingSites catalogue (RADAR + DARNED + REDIportal
    per SComatic's own documentation; resources/artifact_masks/SComatic/), the only
    RNA-editing catalogue found on disk

Class VCFs and sample roots match the canonical mapping used throughout this repo's
postanalyze scripts (see leaked_allele_confusion.py, somatic_evidence_package.py):

    germline (paper "germline") = germline/defined/germline_defined.vcf.gz
    UPV      (paper "UPV")      = germline/denovo/germline_denovo.vcf.gz
    somatic  (paper "somatic")  = somatic/denovo/somatic_denovo.vcf.gz
    unresolved (paper "unresolved") = ambiguous/denovo/ambiguous_denovo.vcf.gz

READ-ONLY: reads only the shipped class-stratified VCFs, the two reference FASTAs
already on disk, and the two SComatic editing-site BED files already on disk.
Writes only into --out-dir (default: data/mutational_spectrum_2026-08-DD/).

Run (env snv_caller): python scripts/postanalyze/mutational_spectrum.py
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import subprocess
from pathlib import Path

import pandas as pd
import pysam

REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")
BCFTOOLS = REPO / "apps/bcftools"

SAMPLE_ROOTS = {
    "P4": REPO / "data/P4_tumor/1",
    "P6": REPO / "data/P6_tumor/1",
    "DCIS1": REPO / "data/dcis1",
    "DCIS2": REPO / "data/dcis2",
}
CLASS_PATHS = {
    "germline": "spatial_filter_purity/baseQ0mapQ0/germline/defined/germline_defined.vcf.gz",
    "UPV": "spatial_filter_purity/baseQ0mapQ0/germline/denovo/germline_denovo.vcf.gz",
    "somatic": "spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
    "unresolved": "spatial_filter_purity/baseQ0mapQ0/ambiguous/denovo/ambiguous_denovo.vcf.gz",
}
CLASS_ORDER = ["germline", "UPV", "somatic", "unresolved"]
SAMPLE_ORDER = ["P4", "P6", "DCIS1", "DCIS2"]

BUILD = {
    "P4": "hg19", "P6": "hg19", "DCIS1": "hg38", "DCIS2": "hg38",
}
FASTA = {
    "hg19": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
    "hg38": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
}
# FASTA contig naming differs by build: hg19 fasta is chr-prefixed (matches our
# VCF records directly); GRCh38-3.0.0 fasta is bare ("1", not "chr1"), while our
# DCIS1/DCIS2 VCF records are still "chr1" (see GENOME BUILDS table) -> strip
# "chr" before querying the GRCh38 fasta.
FASTA_STRIP_CHR = {"hg19": False, "hg38": True}
EDITING_BED = {
    "hg19": REPO / "resources/artifact_masks/SComatic/AllEditingSites.hg19.bed.gz",
    "hg38": REPO / "resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz",
}
# The editing BED files are themselves chr-prefixed regardless of build (verified
# by inspection), matching our VCF records directly -> no stripping needed here.

COMP = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}
PYRIMIDINE_REF = {"C", "T"}

SIX_CHANNELS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def rc(seq: str) -> str:
    return "".join(COMP.get(b, "N") for b in reversed(seq))


def read_snps(path: Path) -> list[tuple[str, int, str, str]]:
    """Biallelic single-base substitutions only (indels/MNPs excluded)."""
    out = []
    proc = subprocess.run(
        [str(BCFTOOLS), "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", str(path)],
        capture_output=True, text=True, check=True,
    )
    for line in proc.stdout.splitlines():
        chrom, pos, ref, alts = line.split("\t")
        if len(ref) != 1 or ref not in "ACGT":
            continue
        for alt in alts.split(","):
            if len(alt) == 1 and alt in "ACGT" and alt != ref:
                out.append((chrom, int(pos), ref, alt))
    return out


class EditingMask:
    """Sorted-position lookup for a tabix bed.gz, one chromosome cached at a time."""

    def __init__(self, path: Path):
        self.tabix = pysam.TabixFile(str(path))
        self.chrom = None
        self.positions: list[int] = []

    def contains(self, chrom: str, pos: int) -> bool:
        if chrom != self.chrom:
            self.chrom = chrom
            try:
                self.positions = sorted(int(row.split("\t")[2]) for row in self.tabix.fetch(chrom))
            except ValueError:
                self.positions = []
        i = bisect.bisect_left(self.positions, pos)
        return i < len(self.positions) and self.positions[i] == pos


def trinucleotide_context(fasta: pysam.FastaFile, strip_chr: bool, chrom: str, pos: int) -> str | None:
    query_chrom = chrom[3:] if strip_chr and chrom.startswith("chr") else chrom
    if query_chrom not in fasta.references:
        return None
    try:
        seq = fasta.fetch(query_chrom, pos - 2, pos + 1).upper()
    except (ValueError, IndexError):
        return None
    if len(seq) != 3:
        return None
    return seq


def pyrimidine_collapse(prev_b: str, ref: str, next_b: str, alt: str) -> tuple[str, str, str, str]:
    if ref in PYRIMIDINE_REF:
        return prev_b, ref, next_b, alt
    return COMP[next_b], COMP[ref], COMP[prev_b], COMP[alt]


def build_all_96_contexts() -> list[str]:
    bases = "ACGT"
    out = []
    for sub in SIX_CHANNELS:
        ref, alt = sub.split(">")
        for p5 in bases:
            for p3 in bases:
                out.append(f"{p5}[{sub}]{p3}")
    return out


def process_sample_class(sample: str, cls: str, vcf_path: Path,
                          fasta: pysam.FastaFile, strip_chr: bool,
                          editing_mask: EditingMask) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    variants = read_snps(vcf_path)
    n_total = len(variants)

    six_counts = {c: 0 for c in SIX_CHANNELS}
    ninety_six_counts: dict[str, int] = {}
    n_ag_tc = 0
    n_context_ok = 0
    n_editing_overlap = 0
    n_editing_compatible_sub = 0

    for chrom, pos, ref, alt in variants:
        if (ref, alt) in {("A", "G"), ("T", "C")}:
            n_ag_tc += 1
            n_editing_compatible_sub += 1
        ctx = trinucleotide_context(fasta, strip_chr, chrom, pos)
        if ctx is None:
            continue
        prev_b, mid_b, next_b = ctx[0], ctx[1], ctx[2]
        if mid_b != ref:
            # Reference mismatch (rare; build/liftover edge case) -> skip from
            # context-dependent tallies but keep in the raw variant list.
            continue
        n_context_ok += 1
        p5, r, p3, a = pyrimidine_collapse(prev_b, ref, next_b, alt)
        six_counts[f"{r}>{a}"] += 1
        key = f"{p5}[{r}>{a}]{p3}"
        ninety_six_counts[key] = ninety_six_counts.get(key, 0) + 1
        if editing_mask.contains(chrom, pos):
            n_editing_overlap += 1

    six_rows = [
        {"sample": sample, "class": cls, "channel": c, "count": six_counts[c],
         "fraction": (six_counts[c] / n_context_ok) if n_context_ok else float("nan")}
        for c in SIX_CHANNELS
    ]
    six_rows.append({
        "sample": sample, "class": cls, "channel": "n_total_snvs", "count": n_total,
        "fraction": float("nan"),
    })
    six_rows.append({
        "sample": sample, "class": cls, "channel": "n_context_resolved", "count": n_context_ok,
        "fraction": (n_context_ok / n_total) if n_total else float("nan"),
    })
    six_rows.append({
        "sample": sample, "class": cls, "channel": "A>G_plus_T>C_fraction_of_total",
        "count": n_ag_tc, "fraction": (n_ag_tc / n_total) if n_total else float("nan"),
    })
    six_df = pd.DataFrame(six_rows)

    ninety_six_rows = []
    for key in build_all_96_contexts():
        cnt = ninety_six_counts.get(key, 0)
        ninety_six_rows.append({
            "sample": sample, "class": cls, "context": key, "count": cnt,
            "fraction": (cnt / n_context_ok) if n_context_ok else float("nan"),
        })
    ninety_six_df = pd.DataFrame(ninety_six_rows)

    editing_row = pd.DataFrame([{
        "sample": sample, "class": cls,
        "n_total_snvs": n_total,
        "n_editing_compatible_substitution_AG_TC": n_editing_compatible_sub,
        "pct_editing_compatible_substitution": 100 * n_editing_compatible_sub / n_total if n_total else float("nan"),
        "n_overlap_SComatic_AllEditingSites": n_editing_overlap,
        "pct_overlap_SComatic_AllEditingSites": 100 * n_editing_overlap / n_total if n_total else float("nan"),
    }])

    return six_df, ninety_six_df, editing_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fasta_handles = {b: pysam.FastaFile(FASTA[b]) for b in set(BUILD.values())}
    editing_masks = {b: EditingMask(EDITING_BED[b]) for b in set(BUILD.values())}

    all_six, all_96, all_edit = [], [], []
    for sample in SAMPLE_ORDER:
        build = BUILD[sample]
        fasta = fasta_handles[build]
        strip_chr = FASTA_STRIP_CHR[build]
        emask = editing_masks[build]
        root = SAMPLE_ROOTS[sample]
        for cls in CLASS_ORDER:
            vcf_path = root / CLASS_PATHS[cls]
            if not vcf_path.exists():
                raise FileNotFoundError(vcf_path)
            print(f"[{sample}/{cls}] {vcf_path}", flush=True)
            six_df, n96_df, edit_row = process_sample_class(sample, cls, vcf_path, fasta, strip_chr, emask)
            all_six.append(six_df)
            all_96.append(n96_df)
            all_edit.append(edit_row)
            ag_tc = six_df[six_df.channel == "A>G_plus_T>C_fraction_of_total"].iloc[0]
            print(f"    n={ag_tc['count']} SNVs is A>G/T>C-compatible substitution; "
                  f"total class n={six_df[six_df.channel=='n_total_snvs'].iloc[0]['count']}, "
                  f"fraction={ag_tc['fraction']:.4f}" if ag_tc["fraction"] == ag_tc["fraction"] else "    n/a")

    six_out = pd.concat(all_six, ignore_index=True)
    n96_out = pd.concat(all_96, ignore_index=True)
    edit_out = pd.concat(all_edit, ignore_index=True)

    six_out.to_csv(out_dir / "spectrum_6channel.csv", index=False)
    n96_out.to_csv(out_dir / "spectrum_96channel.csv", index=False)
    edit_out.to_csv(out_dir / "editing_overlap.csv", index=False)
    print(f"WROTE {out_dir}/spectrum_6channel.csv")
    print(f"WROTE {out_dir}/spectrum_96channel.csv")
    print(f"WROTE {out_dir}/editing_overlap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
