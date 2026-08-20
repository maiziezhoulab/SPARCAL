#!/usr/bin/env python
"""
Callset-quality measurements for Monopogen's SOMATIC module, on the same footing
as `spatialsnv_callset_quality.py` runs them for the published SpatialSNV calls.

Adds Monopogen to three comparisons the manuscript currently makes without it:

  (1) COMMON-GERMLINE CONTAMINATION.  Fraction of Monopogen's putative somatic
      callset that is an allele-exact match to a biallelic SNV in the 1000 Genomes
      panel.  Directly comparable to the SpatialSNV rates in the "population-variant
      leakage" subsection.

      *** ALLELE-EXACT IS NOT OPTIONAL. *** A position-only / region query silently
      counts panel indel+SV records whose SPAN covers the query point; that artifact
      inflated an earlier WES figure ~3x.  Same rule as the SpatialSNV script.

      Monopogen is interesting here precisely because it is population-aware: it
      uses the same 1KGP panel for Beagle phasing and LD refinement.  Whatever rate
      comes out is a property of its somatic module, not an accident of resources.

  (2) COSMIC MEMBERSHIP.  Allele-exact hit rate against COSMIC v103 Genome Screens
      Mutant, matching the catalog and matching rule used for the SPARCAL
      somatic/unresolved contrast.  Report as *where the calls fall*, never as
      evidence that they are cancer-driving -- catalog membership is
      depth-associated for every class alike (PAPER_PLAN Decision D2).

  (3) PER-CALL SUPPORT.  Fraction of calls with >= 2 ALT reads (from Monopogen's
      own Depth_alt) and, where a presence matrix has been built, the fraction seen
      in >= 2 spots.  Comparable to the "evidence available to spatial-RNA somatic
      callers" subsection.

Builds: P4/P6 are hg19 (COSMIC GRCh37); DCIS1/DCIS2 are GRCh38.

Outputs (under data/monopogen_callset_quality_<date>/):
  summary.csv          per sample: set size, 1KGP hits+rate, COSMIC hits+rate, support
  leaked_sites.csv     per 1KGP-leaked call, with panel AF
  cosmic_hits.csv      per COSMIC-matching call, with gene

Run under env `snv_caller`.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import date

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"

PANELS = {
    "hg19": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19",
        "hg19_chr{chrom}.vcf.gz",
    ),
    "GRCh38": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}"
        ".filtered.shapeit2-duohmm-phased.vcf.gz",
    ),
}

COSMIC = {
    "hg19": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz",
    "GRCh38": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz",
}

MONO_LFS = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/"
            "CanLuo/ST_SNV/Monopogen")

# sample -> (build, monopogen run dir, presence-matrix pkl or None)
BENCH = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/monopogen"
SAMPLES = {
    "P4": ("hg19", f"{MONO_LFS}/P4_rep1", None),
    "P6": ("hg19", f"{MONO_LFS}/P6_rep1",
           f"{BENCH}/P6_rep1_som/P6_rep1_som_monopogen_somatic_presence_matrix.pkl"),
    "DCIS1": ("GRCh38", "/data/maiziezhou_lab/leiy4/snv_calling/Monopogen_DCIS1",
              f"{BENCH}/dcis1/dcis1_monopogen_somatic_presence_matrix.pkl"),
    "DCIS2": ("GRCh38", "/data/maiziezhou_lab/leiy4/snv_calling/Monopogen_DCIS2",
              f"{BENCH}/dcis2/dcis2_monopogen_somatic_presence_matrix.pkl"),
}

AUTOSOMES = [str(c) for c in range(1, 23)]
VALID_CHROMS = set(AUTOSOMES)


def norm_chrom(c: str) -> str:
    c = c.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c


def key(chrom: str, pos, ref: str, alt: str) -> str:
    return f"{norm_chrom(chrom)}_{pos}_{ref.upper()}_{alt.upper()}"


# ------------------------------------------------------------------ callset
def read_monopogen_somatic(run_dir: str) -> dict[str, dict]:
    """putativeSNVs.csv for chr1..22 -> {key: {...call fields...}}."""
    som = os.path.join(run_dir, "out", "somatic")
    calls: dict[str, dict] = {}
    for i in range(1, 23):
        path = os.path.join(som, f"chr{i}.putativeSNVs.csv")
        if not os.path.exists(path):
            sys.stderr.write(f"[warn] missing {path}\n")
            continue
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if norm_chrom(row["chr"]) not in VALID_CHROMS:
                    continue
                k = key(row["chr"], row["pos"], row["Ref_allele"], row["Alt_allele"])
                calls[k] = {
                    "chrom": norm_chrom(row["chr"]),
                    "pos": row["pos"],
                    "ref": row["Ref_allele"],
                    "alt": row["Alt_allele"],
                    "depth_total": row["Depth_total"],
                    "depth_alt": row["Depth_alt"],
                    "svm": row["SVM_pos_score"],
                    "ld_merged": row["LDrefine_merged_score"],
                    "baf": row["BAF_alt"],
                }
    return calls


# ------------------------------------------------------------------- panel
def scan_panel_chrom(args) -> tuple[str, dict[str, str]]:
    """One streaming pass over a panel chromosome; allele-exact hits only."""
    build, chrom, query_keys = args
    directory, template = PANELS[build]
    path = os.path.join(directory, template.format(chrom=chrom))
    if not os.path.exists(path):
        sys.stderr.write(f"[warn] missing panel file: {path}\n")
        return chrom, {}

    hits: dict[str, str] = {}
    proc = subprocess.Popen(["zcat", path], stdout=subprocess.PIPE, text=True,
                            stderr=subprocess.DEVNULL, bufsize=1 << 20)
    assert proc.stdout is not None
    for line in proc.stdout:
        if line[0] == "#":
            continue
        f = line.split("\t", 8)
        if len(f) < 8:
            continue
        ref, alt = f[3], f[4]
        if len(ref) != 1 or len(alt) != 1:      # biallelic SNV records only
            continue
        k = f"{norm_chrom(f[0])}_{f[1]}_{ref.upper()}_{alt.upper()}"
        if k in query_keys:
            af = ""
            for field in f[7].split(";"):
                if field.startswith("AF="):
                    af = field[3:].split(",")[0]
                    break
            hits[k] = af
    proc.wait()
    return chrom, hits


def panel_hits(build: str, keys: set[str], workers: int) -> dict[str, str]:
    by_chrom: dict[str, set[str]] = {c: set() for c in AUTOSOMES}
    for k in keys:
        c = k.split("_", 1)[0]
        if c in by_chrom:
            by_chrom[c].add(k)
    jobs = [(build, c, by_chrom[c]) for c in AUTOSOMES if by_chrom[c]]
    out: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for chrom, hits in ex.map(scan_panel_chrom, jobs):
            out.update(hits)
    return out


# ------------------------------------------------------------------ cosmic
def cosmic_hits(build: str, keys: set[str]) -> dict[str, str]:
    """Single streaming pass over the COSMIC VCF; allele-exact SNV matches -> gene."""
    path = COSMIC[build]
    if not os.path.exists(path):
        sys.stderr.write(f"[warn] missing COSMIC file: {path}\n")
        return {}
    hits: dict[str, str] = {}
    proc = subprocess.Popen(["zcat", path], stdout=subprocess.PIPE, text=True,
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
        k = f"{norm_chrom(f[0])}_{f[1]}_{ref.upper()}_{alt.upper()}"
        if k in keys and k not in hits:
            gene = ""
            for field in f[7].split(";"):
                if field.startswith("GENE="):
                    gene = field[5:]
                    break
            hits[k] = gene
    proc.wait()
    return hits


# ----------------------------------------------------------------- support
def spot_counts(pkl: str | None, keys: set[str]) -> dict[str, int]:
    """SNV -> number of spots carrying ALT, from the benchmark presence matrix."""
    if not pkl or not os.path.exists(pkl):
        return {}
    import pandas as pd
    mat = pd.read_pickle(pkl)
    sums = mat.sum(axis=0)
    return {k: int(v) for k, v in sums.items() if k in keys}


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", default=list(SAMPLES),
                    choices=list(SAMPLES))
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    outdir = args.outdir or os.path.join(
        REPO, "data", f"monopogen_callset_quality_{date.today()}")
    os.makedirs(outdir, exist_ok=True)

    summary, leaked_rows, cosmic_rows = [], [], []

    for sample in args.samples:
        build, run_dir, pkl = SAMPLES[sample]
        calls = read_monopogen_somatic(run_dir)
        if not calls:
            sys.stderr.write(f"[skip] {sample}: no putative somatic calls found\n")
            continue
        keys = set(calls)
        print(f"[{sample}] {len(keys)} putative somatic calls ({build})", flush=True)

        p_hits = panel_hits(build, keys, args.workers)
        print(f"[{sample}] 1KGP allele-exact hits: {len(p_hits)}", flush=True)

        c_hits = cosmic_hits(build, keys)
        print(f"[{sample}] COSMIC allele-exact hits: {len(c_hits)}", flush=True)

        spots = spot_counts(pkl, keys)

        n = len(keys)
        ge2_reads = sum(1 for c in calls.values()
                        if c["depth_alt"] not in ("", "NA") and float(c["depth_alt"]) >= 2)
        ge2_spots = sum(1 for v in spots.values() if v >= 2) if spots else ""
        n_spot_known = len(spots) if spots else ""

        summary.append({
            "sample": sample,
            "build": build,
            "n_somatic_calls": n,
            "n_1kgp_hits": len(p_hits),
            "pct_1kgp": round(100.0 * len(p_hits) / n, 3),
            "n_cosmic_hits": len(c_hits),
            "pct_cosmic": round(100.0 * len(c_hits) / n, 3),
            "n_ge2_alt_reads": ge2_reads,
            "pct_ge2_alt_reads": round(100.0 * ge2_reads / n, 1),
            "n_calls_with_matrix": n_spot_known,
            "n_ge2_spots": ge2_spots,
            "pct_ge2_spots": (round(100.0 * ge2_spots / n_spot_known, 1)
                              if n_spot_known else ""),
        })

        for k, af in sorted(p_hits.items()):
            c = calls[k]
            leaked_rows.append({"sample": sample, **c, "panel_af": af,
                                "n_spots": spots.get(k, "")})
        for k, gene in sorted(c_hits.items()):
            c = calls[k]
            cosmic_rows.append({"sample": sample, **c, "cosmic_gene": gene,
                                "n_spots": spots.get(k, "")})

    _write(os.path.join(outdir, "summary.csv"), summary)
    _write(os.path.join(outdir, "leaked_sites.csv"), leaked_rows)
    _write(os.path.join(outdir, "cosmic_hits.csv"), cosmic_rows)
    print(f"\nWrote {outdir}")
    for r in summary:
        print(f"  {r['sample']:6s} n={r['n_somatic_calls']:>7,}  "
              f"1KGP {r['pct_1kgp']:>6.3f}%  COSMIC {r['pct_cosmic']:>6.3f}%  "
              f">=2 alt reads {r['pct_ge2_alt_reads']:>5.1f}%  "
              f">=2 spots {r['pct_ge2_spots']}")


def _write(path: str, rows: list[dict]) -> None:
    if not rows:
        open(path, "w").close()
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
