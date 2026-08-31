#!/usr/bin/env python3
"""P1-5: Monopogen depth-floor matched ablation on SPARCAL's somatic candidates.

The manuscript asserts that Monopogen's better support statistics are "a
threshold difference rather than better discrimination" -- Monopogen's somatic
module only scores loci with >=4 high-quality REF and >=4 high-quality ALT
bases (verified: Monopogen_DCIS2/Monopogen/src/somatic.py line 156,
`ref_depth>=4 and alt_depth>=4`, computed from bcftools mpileup `-q 20 -Q 20`
I16 counts -- Monopogen_DCIS2/Monopogen/src/bamProcess.py line 148). That is an
explanation, not a result. This script applies the IDENTICAL floor (mpileup
-q 20 -Q 20, ref>=4 AND alt>=4) to SPARCAL's final somatic-class candidates on
the whole-section BAM, then recomputes the same three statistics
monopogen_callset_quality.py already reports for Monopogen, so the two numbers
are produced by the same measurement pipeline and are directly comparable:
  - % >=2 high-quality ALT reads (same q20/Q20 depth used for the floor itself)
  - % >=2 spots showing the ALT allele (from each side's own presence matrix)
  - COSMIC v103 allele-exact hit rate (build-matched panel)
  - callset size, before vs after the floor

Non-destructive: reads existing SPARCAL/Monopogen assets read-only; writes only
under the P1 task output directory.
"""
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
SAMTOOLS = PROJECT / "apps/samtools"

COSMIC = {
    "hg19": Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz"),
    "GRCh38": Path("/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz"),
}

SAMPLES = {
    "P4": dict(
        build="hg19",
        somatic_vcf=PROJECT / "data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
        bam=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam"),
        reference=Path("/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa"),
        chr_prefix=True,
        matrix=PROJECT / "data/P4_tumor/1/matrix/P4_TUMOR_1_SPARCAL_somatic_matrix.pkl",
        mono_ge2_alt=89.7, mono_ge2_spots=None, mono_cosmic=4.567, mono_n=4905,
    ),
    "P6": dict(
        build="hg19",
        somatic_vcf=PROJECT / "data/P6_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
        bam=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam"),
        reference=Path("/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa"),
        chr_prefix=True,
        matrix=PROJECT / "data/P6_tumor/1/matrix/P6_TUMOR_1_SPARCAL_somatic_matrix.pkl",
        mono_ge2_alt=91.6, mono_ge2_spots=44.8, mono_cosmic=3.201, mono_n=7373,
    ),
    "DCIS1": dict(
        build="GRCh38",
        somatic_vcf=PROJECT / "data/dcis1/spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
        bam=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.bam"),
        reference=Path("/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"),
        chr_prefix=False,
        matrix=PROJECT / "data/dcis1/matrix/DCIS_dcis1_SPARCAL_somatic_matrix.pkl",
        mono_ge2_alt=89.1, mono_ge2_spots=51.6, mono_cosmic=2.555, mono_n=44102,
    ),
    "DCIS2": dict(
        build="GRCh38",
        somatic_vcf=PROJECT / "data/dcis2/spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz",
        bam=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.bam"),
        reference=Path("/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"),
        chr_prefix=False,
        matrix=PROJECT / "data/dcis2/matrix/DCIS_dcis2_SPARCAL_somatic_matrix.pkl",
        mono_ge2_alt=83.7, mono_ge2_spots=45.8, mono_cosmic=2.628, mono_n=42657,
    ),
}


def strip_chr(c: str) -> str:
    c = str(c)
    return c[3:] if c.startswith("chr") else c


def load_somatic_vcf(path: Path, want_chr_prefix: bool):
    """Return list of (chrom_as_in_bam, pos, ref, alt) and set of chrom_pos (no chr) keys."""
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            chrom, pos, ref, alt = f[0], int(f[1]), f[3], f[4]
            if len(ref) != 1 or len(alt) != 1 or "," in alt:
                continue
            bare = strip_chr(chrom)
            bam_chrom = ("chr" + bare) if want_chr_prefix else bare
            rows.append((bam_chrom, pos, ref.upper(), alt.upper()))
    return rows


def write_positions(rows, path: Path):
    with open(path, "w") as out:
        for chrom, pos, *_ in rows:
            out.write(f"{chrom}\t{pos}\n")


def run_mpileup(bam: Path, reference: Path, positions: Path, out: Path, force: bool):
    if out.exists() and out.stat().st_size > 0 and not force:
        return
    cmd = [str(SAMTOOLS), "mpileup", "-f", str(reference), "-l", str(positions),
           "-q", "20", "-Q", "20", "-d", "1000000", "-o", str(out), str(bam)]
    subprocess.run(cmd, check=True)


def parse_bases(bases: str, ref: str) -> Counter:
    counts: Counter = Counter()
    i = 0
    while i < len(bases):
        c = bases[i]
        if c == "^":
            i += 2
            continue
        if c == "$":
            i += 1
            continue
        if c in "+-":
            m = re.match(r"(\d+)", bases[i + 1:])
            if not m:
                i += 1
                continue
            i += 1 + len(m.group(1)) + int(m.group(1))
            continue
        if c in ".,":
            counts[ref.upper()] += 1
        elif c.upper() in {"A", "C", "G", "T"}:
            counts[c.upper()] += 1
        i += 1
    return counts


def load_pileup(path: Path):
    result = {}
    with open(path) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 5:
                continue
            chrom, pos, ref = f[0], int(f[1]), f[2].upper()
            counts = parse_bases(f[4], ref)
            result[(chrom, pos)] = counts
    return result


def cosmic_hits(build: str, keys: set[str]) -> set[str]:
    """keys are 'bare_pos_ref_alt' (no chr). Returns the subset that hit COSMIC."""
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


def spot_ge2_fraction(matrix_path: Path, keys_no_chr: set[str], want_chr_prefix: bool):
    if not matrix_path.exists():
        return None, None
    mat = pd.read_pickle(matrix_path)
    ns = (mat.values > 0).sum(axis=0)
    cols = list(mat.columns)

    def col_key(col):
        chrom, pos = col.split("_")[0], col.split("_")[1]
        return f"{strip_chr(chrom)}_{pos}"

    keyed = {col_key(c): n for c, n in zip(cols, ns)}
    sub = [keyed[k] for k in keys_no_chr if k in keyed]
    if not sub:
        return None, None
    ge2 = sum(1 for v in sub if v >= 2)
    return len(sub), ge2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES))
    ap.add_argument("--force-pileup", action="store_true")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    workdir = args.outdir / "work"
    workdir.mkdir(exist_ok=True)

    rows_out = []
    detail_frames = []
    for sample in args.samples:
        cfg = SAMPLES[sample]
        print(f"[{sample}] loading somatic candidates: {cfg['somatic_vcf']}", flush=True)
        rows = load_somatic_vcf(cfg["somatic_vcf"], cfg["chr_prefix"])
        n_total = len(rows)
        print(f"[{sample}] n_total_somatic_candidates={n_total}", flush=True)

        pos_path = workdir / f"{sample}.positions.tsv"
        pileup_path = workdir / f"{sample}.q20Q20.mpileup.tsv"
        write_positions(rows, pos_path)
        run_mpileup(cfg["bam"], cfg["reference"], pos_path, pileup_path, args.force_pileup)
        pile = load_pileup(pileup_path)

        detail = []
        for chrom, pos, ref, alt in rows:
            counts = pile.get((chrom, pos), Counter())
            ref_hq = int(counts.get(ref, 0))
            alt_hq = int(counts.get(alt, 0))
            detail.append({
                "sample": sample, "chrom": chrom, "pos": pos, "ref": ref, "alt": alt,
                "ref_hq_q20Q20": ref_hq, "alt_hq_q20Q20": alt_hq,
                "passes_monopogen_floor": int(ref_hq >= 4 and alt_hq >= 4),
                "key_no_chr": f"{strip_chr(chrom)}_{pos}_{ref}_{alt}",
                "key_pos_no_chr": f"{strip_chr(chrom)}_{pos}",
            })
        ddf = pd.DataFrame(detail)
        detail_frames.append(ddf)

        for stage, sub in [("pre_floor", ddf), ("post_floor", ddf[ddf.passes_monopogen_floor == 1])]:
            n = len(sub)
            n_ge2_alt = int((sub.alt_hq_q20Q20 >= 2).sum())
            keys = set(sub.key_no_chr)
            keys_pos = set(sub.key_pos_no_chr)
            c_hits = cosmic_hits(cfg["build"], keys) if n else set()
            n_matrix, n_ge2_spots = spot_ge2_fraction(cfg["matrix"], keys_pos, cfg["chr_prefix"]) if n else (None, None)
            rows_out.append({
                "sample": sample, "build": cfg["build"], "stage": stage,
                "n_calls": n,
                "pct_of_pre_floor": round(100.0 * n / n_total, 2) if n_total else None,
                "n_ge2_alt_reads_q20Q20": n_ge2_alt,
                "pct_ge2_alt_reads_q20Q20": round(100.0 * n_ge2_alt / n, 1) if n else None,
                "n_cosmic_hits": len(c_hits),
                "pct_cosmic": round(100.0 * len(c_hits) / n, 3) if n else None,
                "n_calls_with_matrix": n_matrix,
                "n_ge2_spots": n_ge2_spots,
                "pct_ge2_spots": round(100.0 * n_ge2_spots / n_matrix, 1) if n_matrix else None,
            })
        print(f"[{sample}] done.", flush=True)

    result = pd.DataFrame(rows_out)
    # attach Monopogen's own already-published numbers (from
    # data/monopogen_callset_quality_2026-08-20{,_dcis1}/summary.csv) for direct
    # side-by-side comparison, read-only quoting, not re-derived here.
    mono_rows = []
    for sample, cfg in SAMPLES.items():
        mono_rows.append({
            "sample": sample, "stage": "monopogen_own_callset",
            "n_calls": cfg["mono_n"],
            "pct_ge2_alt_reads_q20Q20": cfg["mono_ge2_alt"],
            "pct_ge2_spots": cfg["mono_ge2_spots"],
            "pct_cosmic": cfg["mono_cosmic"],
        })
    mono_df = pd.DataFrame(mono_rows)
    result = pd.concat([result, mono_df], ignore_index=True)
    result.to_csv(args.outdir / "monopogen_matched_floor.csv", index=False)

    detail_all = pd.concat(detail_frames, ignore_index=True)
    detail_all.to_csv(args.outdir / "monopogen_matched_floor_detail.csv.gz", index=False, compression="gzip")

    print(result.to_string(index=False))
    print(f"\nWrote P1-5 package to {args.outdir}")


if __name__ == "__main__":
    main()
