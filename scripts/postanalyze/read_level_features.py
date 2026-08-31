#!/usr/bin/env python3
"""Strand bias and position-in-read for ALT-supporting reads, all four output
classes, all four sections (PAPER_WORK P1-2 / X-3, remaining bullets).

This is a thin, non-destructive wrapper around the already-existing, unmodified
scripts/postanalyze/collect_spatial_artifact_features.py collector (used
elsewhere in this repo for the same BAMs/masks; see On_going.md 2026-07-15/16).
It is invoked as a subprocess -- this script does not edit it -- once per
section, with all four output classes (germline/UPV/somatic/unresolved) as
labelled candidate sources, writing into a NEW directory under --out-dir (never
into the existing data/<sample>/artifact_evidence/ trees). It then summarizes
the resulting site_features.tsv.gz into strand_bias.csv and
position_in_read.csv.

Candidate sites are a deterministic stable-hash subsample (see collector
--max-sites-per-source) because the unresolved class alone reaches 590,897
sites for P6; the collector pileups every read at every site, so BAM-scale
random access is the cost driver, not VCF size. The subsample is per class,
seeded, and reproducible.

READ-ONLY except for --out-dir. Run (env snv_caller):
    python scripts/postanalyze/read_level_features.py --out-dir data/mutational_spectrum_2026-08-DD
"""
from __future__ import annotations

import argparse
import gzip
import subprocess
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency

REPO = Path("/data/maiziezhou_lab/leiy4/snv_calling")
COLLECTOR = REPO / "scripts/postanalyze/collect_spatial_artifact_features.py"
PYTHON = "/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python"

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
BUILD = {"P4": "hg19", "P6": "hg19", "DCIS1": "hg38", "DCIS2": "hg38"}

BAM_PATH = {
    "P4": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/"
          "spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.dedup.bam",
    "P6": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/"
          "spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.dedup.bam",
    "DCIS1": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/"
             "DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.dedup.bam",
    "DCIS2": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/"
             "DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.dedup.bam",
}
# NOTE: --editing-bed/--pon-bed are deliberately NOT passed to the collector here.
# collect_spatial_artifact_features.py's TabixCoordinateMask re-fetches and rebuilds
# the ENTIRE chromosome's position array from tabix every time the candidate's
# chromosome differs from the previous candidate's (no re-sorting by chromosome
# happens upstream), which thrashes catastrophically once candidates are drawn from
# multiple classes (chromosome order is effectively random). Confirmed by profiling:
# 40 sites reached 47M generator calls / 40s in TabixCoordinateMask.contains() alone.
# Editing-site overlap is already computed correctly and efficiently (chromosome-
# sorted, cached once per chromosome) in mutational_spectrum.py's own EditingMask ->
# editing_overlap.csv, so this collector run does not need those masks at all.
MAX_SITES_PER_CLASS = 2000  # matches the precedent pilot run (On_going.md 2026-07-15)
MAX_PILEUP_DEPTH = 5000  # bound tail risk from very high-coverage pooled hotspot positions


def run_collector(sample: str, out_dir: Path, overwrite: bool) -> Path:
    feat_dir = out_dir / "read_features" / sample
    site_path = feat_dir / "site_features.tsv.gz"
    if site_path.exists() and not overwrite:
        print(f"[{sample}] {site_path} already exists, skipping collector (use --overwrite to redo)")
        return site_path
    build = BUILD[sample]
    root = SAMPLE_ROOTS[sample]
    cmd = [
        PYTHON, str(COLLECTOR),
        "--bam", BAM_PATH[sample],
        "--out-dir", str(feat_dir),
        "--max-sites-per-source", str(MAX_SITES_PER_CLASS),
        "--seed", "mutspec-2026-08-27",
        "--max-depth", str(MAX_PILEUP_DEPTH),
    ]
    for cls in CLASS_ORDER:
        cmd += ["--candidates", f"{cls}={root / CLASS_PATHS[cls]}"]
    if overwrite:
        cmd.append("--overwrite")
    print(f"[{sample}] running collector: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    return site_path


def load_site_features(sample: str, site_path: Path) -> pd.DataFrame:
    df = pd.read_csv(site_path, sep="\t")
    df["sample"] = sample
    # candidate_sources can carry multiple labels if the same site was sampled under
    # more than one class (rare at this subsample size, but handle it): explode.
    df["class_label"] = df["candidate_sources"].str.split(",")
    df = df.explode("class_label")
    return df


def summarize_strand_bias(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, cls), g in df.groupby(["sample", "class_label"]):
        alt_fwd = g["alt_forward_frac"].apply(pd.to_numeric, errors="coerce")
        ref_fwd = g["ref_forward_frac"].apply(pd.to_numeric, errors="coerce")
        alt_n = g["alt_n"].apply(pd.to_numeric, errors="coerce")
        ref_n = g["ref_n"].apply(pd.to_numeric, errors="coerce")
        alt_fwd_reads = (alt_fwd * alt_n).sum()
        alt_rev_reads = alt_n.sum() - alt_fwd_reads
        ref_fwd_reads = (ref_fwd * ref_n).sum()
        ref_rev_reads = ref_n.sum() - ref_fwd_reads
        row = {
            "sample": sample, "class": cls, "n_sites": len(g),
            "n_sites_with_alt_read": int((alt_n > 0).sum()),
            "mean_alt_forward_frac": alt_fwd.mean(),
            "mean_ref_forward_frac": ref_fwd.mean(),
            "pooled_alt_forward_reads": alt_fwd_reads, "pooled_alt_reverse_reads": alt_rev_reads,
            "pooled_ref_forward_reads": ref_fwd_reads, "pooled_ref_reverse_reads": ref_rev_reads,
        }
        table = [[alt_fwd_reads, alt_rev_reads], [ref_fwd_reads, ref_rev_reads]]
        if min(alt_fwd_reads + alt_rev_reads, ref_fwd_reads + ref_rev_reads) > 0:
            try:
                chi2, p, _, _ = chi2_contingency(table)
                row["strand_chi2_alt_vs_ref"] = chi2
                row["strand_chi2_p"] = p
            except ValueError:
                row["strand_chi2_alt_vs_ref"] = float("nan")
                row["strand_chi2_p"] = float("nan")
        else:
            row["strand_chi2_alt_vs_ref"] = float("nan")
            row["strand_chi2_p"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_position_in_read(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, cls), g in df.groupby(["sample", "class_label"]):
        alt_pos = g["alt_mean_read_pos_frac"].apply(pd.to_numeric, errors="coerce")
        ref_pos = g["ref_mean_read_pos_frac"].apply(pd.to_numeric, errors="coerce")
        alt_term = g["alt_terminal_frac"].apply(pd.to_numeric, errors="coerce")
        ref_term = g["ref_terminal_frac"].apply(pd.to_numeric, errors="coerce")
        rows.append({
            "sample": sample, "class": cls, "n_sites": len(g),
            "mean_alt_read_pos_frac": alt_pos.mean(), "median_alt_read_pos_frac": alt_pos.median(),
            "mean_ref_read_pos_frac": ref_pos.mean(), "median_ref_read_pos_frac": ref_pos.median(),
            "mean_alt_terminal_frac": alt_term.mean(), "mean_ref_terminal_frac": ref_term.mean(),
            "note": "read_pos_frac = normalized distance to the NEAREST read end (0=at a read end, "
                    "~0.5=read center; collector default read_end_bases=5); terminal_frac = fraction "
                    "of ALT/REF-supporting reads within 5bp of either read end",
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_site = []
    for sample in SAMPLE_ORDER:
        site_path = run_collector(sample, out_dir, args.overwrite)
        df = load_site_features(sample, site_path)
        all_site.append(df)
    site_df = pd.concat(all_site, ignore_index=True)

    strand_df = summarize_strand_bias(site_df)
    pos_df = summarize_position_in_read(site_df)
    strand_df.to_csv(out_dir / "strand_bias.csv", index=False)
    pos_df.to_csv(out_dir / "position_in_read.csv", index=False)
    print(f"WROTE {out_dir}/strand_bias.csv")
    print(strand_df.to_string())
    print(f"WROTE {out_dir}/position_in_read.csv")
    print(pos_df.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
