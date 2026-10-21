#!/usr/bin/env python
"""
Callset-quality comparison: SPARCAL somatic vs the published SpatialSNV callsets.

Two measurements, both allele-exact, both independent of read depth / coverage
(which is the confound that sank the region-detection comparison, see
PAPER_PLAN_DEPRECATED.md sections 1 and 4):

  (1) RETAINED MUTECT2 ARTIFACT FLAGS.  SpatialSNV calls with Mutect2 in tumor-only
      mode.  For every SNV in their FINAL callset (the post-CallBack feature list
      that their matrices are keyed on), we look up the FILTER field Mutect2 wrote
      for that site and report what fraction of the retained callset never cleared
      the caller's own filters, broken down by flag.
      NOTE (fairness): they DO drop Mutect2's `germline`-flagged records -- verified
      on P6 chr1, 0 of 12,973 retained calls carried it.  Do not claim otherwise.

  (2) COMMON-GERMLINE CONTAMINATION.  Fraction of each final callset that is an
      allele-exact match to a biallelic SNV in the 1000 Genomes panel.

      *** ALLELE-EXACT IS NOT OPTIONAL. ***  A position-only (or bcftools -R region)
      match against the panel silently counts panel indel/SV records whose SPAN
      merely covers the query point.  That artifact inflated an earlier WES-leakage
      figure by ~3x (28% -> the correct 8.3%); see PAPER_PLAN.md section 3 Story C2.
      For the same reason we do NOT use data/1kG_positions/GRCh38/*_snps_with_af.txt,
      which is position-only.

      SPARCAL's somatic class is 0% here BY CONSTRUCTION (1KGP-panel variants are
      routed to the germline class before the spatial cascade runs).  That is the
      point of the comparison, not a hidden win: report SpatialSNV's rate as a
      property of their callset and SPARCAL's as a design consequence.  The
      informative extra column is `n_in_sparcal_germline` -- of the panel variants
      SpatialSNV retained as somatic, how many did SPARCAL actually see in the same
      tissue and correctly route to germline.

Outputs (under data/spatialsnv_callset_quality_<date>/):
  filter_summary.csv        per sample: retained-callset FILTER breakdown
  filter_by_flag.csv        per sample x flag counts
  germline_summary.csv      per sample x method: 1KGP allele-exact contamination
  germline_leaked_sites.csv per leaked site, with panel AF and SPARCAL's class

Run under env `snv_caller` (needs pysam only for nothing -- stdlib + bcftools/zcat).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
BCFTOOLS = os.path.join(REPO, "apps", "bcftools")

PANELS = {
    # build -> (directory, filename template with {chrom} as a bare number)
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

# SPARCAL class VCFs are all under <root>/spatial_filter_purity/baseQ0mapQ0/
SPARCAL_SOMATIC = "spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz"
SPARCAL_GERMLINE_DEF = (
    "spatial_filter_purity/baseQ0mapQ0/germline/defined/germline_defined.vcf.gz"
)

SAMPLES = {
    "P4": {
        "build": "hg19",
        "ssnv_dir": "SpatialSNV/results/p4",
        "ssnv_tag": "p4",
        "sparcal_root": "data/P4_tumor/1",
    },
    "P6": {
        "build": "hg19",
        "ssnv_dir": "SpatialSNV/results/p6",
        "ssnv_tag": "p6",
        "sparcal_root": "data/P6_tumor/1",
    },
    "DCIS1": {
        "build": "GRCh38",
        "ssnv_dir": "SpatialSNV/results/dcis1",
        "ssnv_tag": "dcis1",
        "sparcal_root": "data/dcis1",
    },
    "DCIS2": {
        "build": "GRCh38",
        "ssnv_dir": "SpatialSNV/results/dcis2",
        "ssnv_tag": "dcis2",
        "sparcal_root": "data/dcis2",
    },
}

AUTOSOMES = [str(c) for c in range(1, 23)]


# ---------------------------------------------------------------- normalisation
def norm_chrom(c: str) -> str:
    """Bare autosome number. Inputs disagree: DCIS SpatialSNV features use `12`,
    P4/P6 use `chr11`, both panels and all SPARCAL VCFs use `chr…`."""
    c = c.strip()
    if c.lower().startswith("chr"):
        c = c[3:]
    return c


VALID_CHROMS = set(str(c) for c in range(1, 23))


def is_autosome(c: str) -> bool:
    """Reject scaffolds on the WHOLE normalised name, not its first token.
    `chr7_gl000195_random` -> `7_gl000195_random`, whose leading token is `7`;
    a `split("_", 1)[0] in chroms` test would admit it and inflate the callset
    denominators (P4 +153, P6 +232 records; hg19 features only, DCIS has none)."""
    return norm_chrom(c) in VALID_CHROMS


def key(chrom: str, pos: str | int, ref: str, alt: str) -> str:
    return f"{norm_chrom(chrom)}_{pos}_{ref.upper()}_{alt.upper()}"


# ---------------------------------------------------------------- readers
def read_spatialsnv_features(path: str) -> set[str]:
    """Their final callset. Feature ids look like `chr11_74456788:G>A` (P4/P6) or
    `12_54858388:T>A` (DCIS). SNVs only (single-base ref and alt)."""
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


def read_vcf_keys(path: str, with_filter: bool = False):
    """Stream a VCF via bcftools. Returns a set of keys, or {key: FILTER}."""
    fmt = "%CHROM\t%POS\t%REF\t%ALT\t%FILTER\n" if with_filter else "%CHROM\t%POS\t%REF\t%ALT\n"
    proc = subprocess.Popen(
        [BCFTOOLS, "query", "-f", fmt, path],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    result_set: set[str] = set()
    result_map: dict[str, str] = {}
    assert proc.stdout is not None
    for line in proc.stdout:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            continue
        chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
        if len(ref) != 1 or len(alt) != 1:
            continue
        k = key(chrom, pos, ref, alt)
        if with_filter:
            result_map[k] = parts[4] if len(parts) > 4 else "."
        else:
            result_set.add(k)
    proc.wait()
    return result_map if with_filter else result_set


# ---------------------------------------------------------------- panel scan
def scan_panel_chrom(args) -> tuple[str, dict[str, str]]:
    """Single pass over one panel chromosome, keeping only allele-exact hits to the
    supplied query keys. Returns {key: panel_AF or ''}.

    Streaming + hashing rather than indexed lookup: we need every query key tested,
    and one sequential read beats tens of thousands of random seeks.
    """
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
        # biallelic SNV records only -- this is what makes the match allele-exact
        # and is precisely what a span/region query fails to enforce.
        if len(ref) != 1 or len(alt) != 1:
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


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--chroms", default=",".join(AUTOSOMES),
                    help="comma-separated autosomes (default 1-22)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--samples", default=",".join(SAMPLES),
                    help="comma-separated subset of " + ",".join(SAMPLES))
    args = ap.parse_args()

    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    sample_names = [s.strip() for s in args.samples.split(",") if s.strip()]
    outdir = args.outdir or os.path.join(
        REPO, f"data/spatialsnv_callset_quality_{date.today().isoformat()}")
    os.makedirs(outdir, exist_ok=True)

    chrom_set = set(chroms)

    # ---------------- load every callset -------------------------------------
    loaded: dict[str, dict] = {}
    for name in sample_names:
        cfg = SAMPLES[name]
        ssnv_feat = os.path.join(REPO, cfg["ssnv_dir"], "matrix",
                                 f"{cfg['ssnv_tag']}_alt", "features.tsv.gz")
        ssnv_vcf = os.path.join(REPO, cfg["ssnv_dir"], f"{cfg['ssnv_tag']}.vcf.gz")
        sparcal_som = os.path.join(REPO, cfg["sparcal_root"], SPARCAL_SOMATIC)
        sparcal_germ = os.path.join(REPO, cfg["sparcal_root"], SPARCAL_GERMLINE_DEF)

        sys.stderr.write(f"[{name}] loading callsets\n")
        final = {k for k in read_spatialsnv_features(ssnv_feat)
                 if k.split("_", 1)[0] in chrom_set}
        filters = read_vcf_keys(ssnv_vcf, with_filter=True)
        somatic = {k for k in read_vcf_keys(sparcal_som)
                   if k.split("_", 1)[0] in chrom_set}
        germline = {k for k in read_vcf_keys(sparcal_germ)
                    if k.split("_", 1)[0] in chrom_set} if os.path.exists(sparcal_germ) else set()

        loaded[name] = dict(build=cfg["build"], final=final, filters=filters,
                            somatic=somatic, germline=germline)
        sys.stderr.write(
            f"[{name}] SpatialSNV final={len(final):,}  "
            f"SPARCAL somatic={len(somatic):,}  germline(1kG)={len(germline):,}\n")

    # ---------------- (1) Mutect2 FILTER retention ---------------------------
    filt_rows, flag_rows = [], []
    for name in sample_names:
        d = loaded[name]
        counts: Counter = Counter()
        flags: Counter = Counter()
        n_unmatched = 0
        for k in d["final"]:
            f = d["filters"].get(k)
            if f is None:
                n_unmatched += 1
                continue
            counts[f] += 1
            if f != "PASS":
                for flag in f.split(";"):
                    flags[flag] += 1
        n_matched = sum(counts.values())
        n_pass = counts.get("PASS", 0)
        n_nonpass = n_matched - n_pass
        filt_rows.append(dict(
            sample=name, n_final_callset=len(d["final"]),
            n_matched_to_vcf=n_matched, n_unmatched_to_vcf=n_unmatched,
            n_PASS=n_pass, n_non_PASS=n_nonpass,
            pct_non_PASS=round(100.0 * n_nonpass / n_matched, 3) if n_matched else "",
            n_germline_flagged=flags.get("germline", 0),
        ))
        for flag, n in flags.most_common():
            flag_rows.append(dict(sample=name, flag=flag, n=n,
                                  pct_of_matched=round(100.0 * n / n_matched, 3)
                                  if n_matched else ""))

    # ---------------- (2) 1KGP allele-exact contamination --------------------
    # Pool queries by build so each panel chromosome is read exactly once.
    per_build_queries: dict[str, set[str]] = defaultdict(set)
    for name in sample_names:
        d = loaded[name]
        per_build_queries[d["build"]] |= d["final"] | d["somatic"]

    panel_hits: dict[str, dict[str, str]] = {}
    for build, qkeys in per_build_queries.items():
        by_chrom: dict[str, set[str]] = defaultdict(set)
        for k in qkeys:
            by_chrom[k.split("_", 1)[0]].add(k)
        jobs = [(build, c, by_chrom[c]) for c in chroms if by_chrom.get(c)]
        sys.stderr.write(f"[panel:{build}] scanning {len(jobs)} chromosomes "
                         f"for {len(qkeys):,} query alleles\n")
        merged: dict[str, str] = {}
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for chrom, hits in ex.map(scan_panel_chrom, jobs):
                sys.stderr.write(f"[panel:{build}] chr{chrom}: {len(hits):,} hits\n")
                merged.update(hits)
        panel_hits[build] = merged

    germ_rows, leaked_rows = [], []
    for name in sample_names:
        d = loaded[name]
        hits = panel_hits[d["build"]]
        for method, calls in (("SpatialSNV", d["final"]), ("SPARCAL_somatic", d["somatic"])):
            leaked = calls & hits.keys()
            common = sum(1 for k in leaked
                         if hits[k] not in ("", None) and _f(hits[k]) >= 0.01)
            germ_rows.append(dict(
                sample=name, method=method, build=d["build"], n_calls=len(calls),
                n_in_1KGP=len(leaked),
                pct_in_1KGP=round(100.0 * len(leaked) / len(calls), 3) if calls else "",
                n_in_1KGP_AF_ge_1pct=common,
                n_also_in_sparcal_germline=len(leaked & d["germline"]),
                note=("0 by construction: 1KGP variants bypass the cascade as germline"
                      if method == "SPARCAL_somatic" else ""),
            ))
            if method == "SpatialSNV":
                for k in sorted(leaked):
                    chrom, pos, ref, alt = k.split("_")
                    leaked_rows.append(dict(
                        sample=name, chrom=chrom, pos=pos, ref=ref, alt=alt,
                        panel_AF=hits[k],
                        mutect2_FILTER=d["filters"].get(k, "NA"),
                        in_sparcal_germline_1kG=int(k in d["germline"]),
                        in_sparcal_somatic=int(k in d["somatic"]),
                    ))

    _write(os.path.join(outdir, "filter_summary.csv"), filt_rows)
    _write(os.path.join(outdir, "filter_by_flag.csv"), flag_rows)
    _write(os.path.join(outdir, "germline_summary.csv"), germ_rows)
    _write(os.path.join(outdir, "germline_leaked_sites.csv"), leaked_rows)
    sys.stderr.write(f"\nWrote 4 CSVs to {outdir}\n")


def _f(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return -1.0


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
