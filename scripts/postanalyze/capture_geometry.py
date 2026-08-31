#!/usr/bin/env python3
"""Capture-geometry waterfall (PAPER_WORK P1-3 / M4) and per-variant 3'-distance
feature emission (PAPER_WORK X-4).

TASK 2 (waterfall, P4 & P6 only, which have matched-WES somatic callsets):
decomposes "RNA covers ~1% of WES-somatic positions" into four staged
mechanisms: gene expression -> 3' capture-window membership -> any pooled-BAM
read depth -> ALT-allele presence. Builds on, and cross-checks against, the
already-parked 3'-shift measurement in
data/spatialsnv_reanalysis_2026-07-17/{p6_3prime_bias_summary,p6_gene_expression_bias}.csv
and the already-parked callability numbers (P4 40/3451, P6 27/2604; re-derived
allele-exact in scripts/postanalyze/somatic_evidence_package.py as P4 34/3450,
P6 22/2585) -- this script's final stage must reproduce those.

TASK 3 (X-4): for every SPARCAL candidate in all four output classes, all four
sections, emits the distance from the assigned transcript's 3' end (bp and
relative position), so it can serve as a per-variant classifier feature.

Gene assignment is gene-body-level (smallest-span gene wins on overlap), same
sweep-line method as scripts/postanalyze/ssnv_3prime_bias_p6.py. Genome-build
handling matches mutational_spectrum.py in this same directory:
  - P4/P6 (hg19): GTF and BAM contigs are chr-prefixed, matching VCF records.
  - DCIS1/DCIS2 (GRCh38): GTF/FASTA contigs are bare ("1"), VCF records are
    "chr1" -> strip "chr" before querying GTF/FASTA; the pooled BAMs for DCIS
    are also chr-prefixed (Cell Ranger convention), so BAM queries use the
    original "chr1" form.

READ-ONLY: reads shipped class VCFs, WES matched-somatic VCFs, GTFs, pooled
dedup BAMs, and SpaceRanger filtered_feature_bc_matrix directories already on
disk. Writes only into --out-dir.

Run (env snv_caller): python scripts/postanalyze/capture_geometry.py --out-dir data/mutational_spectrum_2026-08-DD
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import scipy.io
import scipy.sparse as sp
from scipy.stats import ks_2samp, mannwhitneyu

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
BUILD = {"P4": "hg19", "P6": "hg19", "DCIS1": "hg38", "DCIS2": "hg38"}
STRIP_CHR_FOR_ANNOTATION = {"hg19": False, "hg38": True}  # strip when querying GTF (hg38 GTF is bare)

GTF_PATH = {
    "hg19": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/cellranger_hg19/cellranger_hg19/genes/genes.gtf",
    "hg38": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/genes/genes.gtf.gz",
}

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
MATRIX_DIR = {
    "P4": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/"
          "spaceranger_align_rep1_hg19/P4_Tumor_output/outs/filtered_feature_bc_matrix",
    "P6": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/"
          "spaceranger_align_rep1_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix",
}
# Matched-WES somatic exome-restricted SNP callsets (hg19, chr-prefixed; confirmed against
# scripts/postanalyze/somatic_evidence_package.py -- 3,451/2,604 raw records, 3,450/2,585
# true single-base-substitution alleles after resolving co-located multiallelic indel records).
WES_SOMATIC_VCF = {
    "P4": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
          "P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz",
    "P6": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
          "P6_Somatic_Mutect2/P6_somatic_exome_snps_chr1_22.vcf.gz",
}
WATERFALL_SAMPLES = ["P4", "P6"]
WINDOW_BP_OPTIONS = [300, 500, 1000]  # primary = 500bp; 300/1000 as sensitivity bounds


def canon_chrom(c: str) -> str:
    return c if c.startswith("chr") else f"chr{c}"


def load_genes(build: str) -> dict[str, list[tuple[int, int, str, str]]]:
    """Return {canonical chr-prefixed chrom: [(start,end,strand,gene_name), ...] sorted by start}."""
    path = GTF_PATH[build]
    opener = gzip.open if path.endswith(".gz") else open
    by_chrom: dict[str, list] = {}
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom = canon_chrom(f[0])
            start, end, strand = int(f[3]), int(f[4]), f[6]
            attrs = f[8]
            gi = attrs.find('gene_name "')
            gname = attrs[gi + 11: attrs.find('"', gi + 11)] if gi >= 0 else "NA"
            by_chrom.setdefault(chrom, []).append((start, end, strand, gname))
    for c in by_chrom:
        by_chrom[c].sort(key=lambda t: t[0])
    return by_chrom


def assign_genes(genes_by_chrom, positions_by_chrom) -> dict:
    """Sweep-line smallest-span-wins gene assignment. positions_by_chrom keys are
    canonical chr-prefixed chrom strings, matching genes_by_chrom keys."""
    result = {}
    for chrom, qpos in positions_by_chrom.items():
        genes = genes_by_chrom.get(chrom, [])
        if not genes:
            for p in qpos:
                result[(chrom, p)] = None
            continue
        qpos_sorted = sorted(set(qpos))
        gi = 0
        active: list = []
        for p in qpos_sorted:
            while gi < len(genes) and genes[gi][0] <= p:
                s, e, strand, name = genes[gi]
                heapq.heappush(active, (e, s, strand, name))
                gi += 1
            while active and active[0][0] < p:
                heapq.heappop(active)
            cands = [g for g in active if g[1] <= p <= g[0]]
            if not cands:
                result[(chrom, p)] = None
            else:
                best = min(cands, key=lambda g: g[0] - g[1])
                e, s, strand, name = best
                result[(chrom, p)] = (s, e, strand, name)
    return result


def dist_and_relpos(pos: int, start: int, end: int, strand: str) -> tuple[int, float]:
    length = max(end - start, 1)
    if strand == "+":
        dist3p = end - pos
        relpos = (pos - start) / length
    else:
        dist3p = pos - start
        relpos = (end - pos) / length
    return dist3p, relpos


def read_snps(path) -> list[tuple[str, int, str, str]]:
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


# ---------------------------------------------------------------------------
# TASK 3: X-4 per-variant 3' distance feature
# ---------------------------------------------------------------------------

def build_three_prime_distance(out_dir: Path) -> pd.DataFrame:
    rows = []
    genes_cache: dict[str, dict] = {}
    for sample in SAMPLE_ORDER:
        build = BUILD[sample]
        if build not in genes_cache:
            print(f"Loading {build} gene models...", flush=True)
            genes_cache[build] = load_genes(build)
        genes_by_chrom = genes_cache[build]
        root = SAMPLE_ROOTS[sample]
        for cls in CLASS_ORDER:
            vcf_path = root / CLASS_PATHS[cls]
            variants = read_snps(vcf_path)
            by_chrom: dict[str, list[int]] = {}
            for chrom, pos, ref, alt in variants:
                by_chrom.setdefault(chrom, []).append(pos)
            assigned = assign_genes(genes_by_chrom, by_chrom)
            n_assigned = 0
            for chrom, pos, ref, alt in variants:
                g = assigned.get((chrom, pos))
                if g is None:
                    continue
                s, e, strand, gname = g
                dist3p, relpos = dist_and_relpos(pos, s, e, strand)
                rows.append({
                    "sample": sample, "class": cls, "chrom": chrom, "pos": pos,
                    "ref": ref, "alt": alt, "gene": gname, "strand": strand,
                    "gene_len": e - s, "dist_from_3prime_bp": dist3p, "relpos_0tss_1_3prime": round(relpos, 6),
                })
                n_assigned += 1
            print(f"[{sample}/{cls}] {len(variants):,} SNVs -> {n_assigned:,} gene-assigned "
                  f"({100*n_assigned/len(variants):.1f}%)" if variants else f"[{sample}/{cls}] 0 SNVs", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "three_prime_distance.csv", index=False)
    print(f"WROTE {out_dir}/three_prime_distance.csv ({len(df):,} rows)")
    return df


# Two distinct, sometimes-disagreeing metrics -- both reported, neither privileged
# a priori. "higher_is_more_3prime" controls how ordering is read off the medians.
DISTANCE_METRICS = {
    "dist_from_3prime_bp": {
        "label": "absolute distance from 3' end (bp)",
        "higher_is_more_3prime": False,  # smaller bp distance = closer to the 3' end
    },
    "relpos_0tss_1_3prime": {
        "label": "gene-length-normalized relative position (0=TSS, 1=3' end)",
        "higher_is_more_3prime": True,
    },
}


def three_prime_distance_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Retained-somatic vs unresolved (and vs germline) comparison per sample, for
    BOTH the absolute (bp) and gene-length-normalized (relative) 3'-distance metrics.

    IMPORTANT CAVEATS (do not drop when reporting these numbers):
    1. Absolute and relative position can rank classes differently when the classes'
       underlying gene-length distributions differ -- check both before claiming an
       ordering; either metric alone is not sufficient.
    2. KS statistics here are small (typically 0.05-0.22) against very large n
       (tens of thousands vs hundreds of thousands): the resulting p-values are
       driven by sample size, not effect size. Report the KS/effect-size magnitude
       alongside p, never p alone.
    3. This comparison is confounded by capture depth: positions closer to the 3'
       end are, by the platform-limit mechanism this same script's waterfall
       quantifies, more likely to be covered at all and therefore more likely to be
       *retained* by the calling cascade in ANY class. A 3'-distance difference
       between classes is therefore not evidence of an independent (depth-free)
       positional effect until depth is matched between the compared groups -- that
       matching has NOT been done here and is not claimed.
    """
    rows = []
    for sample in SAMPLE_ORDER:
        for metric, meta in DISTANCE_METRICS.items():
            som = df[(df["sample"] == sample) & (df["class"] == "somatic")][metric].dropna()
            unr = df[(df["sample"] == sample) & (df["class"] == "unresolved")][metric].dropna()
            germ = df[(df["sample"] == sample) & (df["class"] == "germline")][metric].dropna()
            if len(som) > 1 and len(unr) > 1:
                ks = ks_2samp(som, unr)
                mw = mannwhitneyu(som, unr, alternative="two-sided")
                rows.append({
                    "sample": sample, "metric": metric, "metric_label": meta["label"],
                    "comparison": "somatic_vs_unresolved",
                    "n_somatic": len(som), "n_other": len(unr),
                    "median_somatic": som.median(), "median_other": unr.median(),
                    "mean_somatic": som.mean(), "mean_other": unr.mean(),
                    "more_3prime_class_by_median": (
                        "somatic" if (som.median() > unr.median()) == meta["higher_is_more_3prime"]
                        else "unresolved"
                    ),
                    "ks_stat": ks.statistic, "ks_p": ks.pvalue,
                    "mannwhitney_p": mw.pvalue,
                    "caveat": "KS p reflects n, not effect size (see docstring); comparison is "
                              "depth-confounded and NOT depth-matched -- not evidence of an "
                              "independent positional effect on its own.",
                })
            if len(som) > 1 and len(germ) > 1:
                ks = ks_2samp(som, germ)
                rows.append({
                    "sample": sample, "metric": metric, "metric_label": meta["label"],
                    "comparison": "somatic_vs_germline",
                    "n_somatic": len(som), "n_other": len(germ),
                    "median_somatic": som.median(), "median_other": germ.median(),
                    "mean_somatic": som.mean(), "mean_other": germ.mean(),
                    "more_3prime_class_by_median": (
                        "somatic" if (som.median() > germ.median()) == meta["higher_is_more_3prime"]
                        else "germline"
                    ),
                    "ks_stat": ks.statistic, "ks_p": ks.pvalue, "mannwhitney_p": np.nan,
                    "caveat": "KS p reflects n, not effect size (see docstring); comparison is "
                              "depth-confounded and NOT depth-matched -- not evidence of an "
                              "independent positional effect on its own.",
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# TASK 2: P1-3 capture-geometry waterfall
# ---------------------------------------------------------------------------

def gene_expression(matrix_dir: str) -> pd.Series:
    with gzip.open(f"{matrix_dir}/features.tsv.gz", "rt") as fh:
        feats = [l.rstrip("\n").split("\t") for l in fh]
    names = [f[1] for f in feats]
    with gzip.open(f"{matrix_dir}/matrix.mtx.gz", "rb") as fh:
        M = sp.csr_matrix(scipy.io.mmread(fh))
    total = np.asarray(M.sum(axis=1)).ravel()
    return pd.Series(total, index=names).groupby(level=0).sum()


def pileup_depth_and_alt(bam: pysam.AlignmentFile, chrom: str, pos: int, alt: str) -> tuple[int, int]:
    """Return (total read depth, ALT-supporting read count) at a single 1-based position."""
    depth = 0
    alt_n = 0
    for col in bam.pileup(chrom, pos - 1, pos, truncate=True, stepper="all",
                           max_depth=100_000, min_base_quality=0, min_mapping_quality=0):
        if col.reference_pos != pos - 1:
            continue
        for pr in col.pileups:
            if pr.is_del or pr.is_refskip or pr.query_position is None:
                continue
            depth += 1
            base = pr.alignment.query_sequence[pr.query_position].upper()
            if base == alt:
                alt_n += 1
    return depth, alt_n


def build_waterfall(sample: str, genes_by_chrom: dict, out_dir: Path) -> pd.DataFrame:
    build = BUILD[sample]
    wes_variants = read_snps(WES_SOMATIC_VCF[sample])
    n_stage1 = len(wes_variants)
    print(f"[{sample}] matched-WES somatic exome SNP alleles: {n_stage1:,}", flush=True)

    by_chrom: dict[str, list[int]] = {}
    for chrom, pos, ref, alt in wes_variants:
        by_chrom.setdefault(chrom, []).append(pos)
    assigned = assign_genes(genes_by_chrom, by_chrom)

    expr = gene_expression(MATRIX_DIR[sample])

    bam = pysam.AlignmentFile(BAM_PATH[sample], "rb")

    per_variant = []
    for chrom, pos, ref, alt in wes_variants:
        g = assigned.get((chrom, pos))
        row = {"chrom": chrom, "pos": pos, "ref": ref, "alt": alt}
        if g is None:
            row.update({"gene": None, "strand": None, "gene_len": None,
                        "dist_from_3prime_bp": None, "expressed": False})
        else:
            s, e, strand, gname = g
            dist3p, relpos = dist_and_relpos(pos, s, e, strand)
            gene_umi = expr.get(gname, 0.0)
            row.update({"gene": gname, "strand": strand, "gene_len": e - s,
                        "dist_from_3prime_bp": dist3p, "relpos": relpos,
                        "expressed": bool(gene_umi > 0), "gene_umi": gene_umi})
        per_variant.append(row)
    per_df = pd.DataFrame(per_variant)

    # Stage-4/5: pileup only at sites that reach stage 3 for each window (cheapest -> most
    # informative order), but for transparency we pileup ALL stage-2 (expressed) sites once
    # and apply window membership as a boolean column afterward.
    expressed_mask = per_df["expressed"] == True  # noqa: E712
    depths, alts = [], []
    for _, r in per_df.iterrows():
        if not r["expressed"]:
            depths.append(None); alts.append(None)
            continue
        d, a = pileup_depth_and_alt(bam, r["chrom"], int(r["pos"]), r["alt"])
        depths.append(d); alts.append(a)
    per_df["pooled_bam_depth"] = depths
    per_df["pooled_bam_alt_reads"] = alts
    bam.close()

    per_df.to_csv(out_dir / f"waterfall_{sample.lower()}_pervariant.csv", index=False)

    stage_rows = []
    for window_bp in WINDOW_BP_OPTIONS:
        s1 = n_stage1
        gene_assigned = per_df["gene"].notna().sum()
        s2 = int((per_df["expressed"] == True).sum())  # noqa: E712
        in_window = per_df["expressed"] & (per_df["dist_from_3prime_bp"].notna()) & \
                    (per_df["dist_from_3prime_bp"] <= window_bp)
        s3 = int(in_window.sum())
        covered = in_window & (per_df["pooled_bam_depth"].fillna(0) >= 1)
        s4 = int(covered.sum())
        alt_present = covered & (per_df["pooled_bam_alt_reads"].fillna(0) >= 1)
        s5 = int(alt_present.sum())

        def pct(n, d):
            return 100 * n / d if d else float("nan")

        stage_rows.extend([
            {"sample": sample, "window_bp": window_bp, "stage": "1_wes_somatic_total",
             "n": s1, "pct_of_total": 100.0, "pct_of_prev_stage": float("nan")},
            {"sample": sample, "window_bp": window_bp, "stage": "1b_gene_assigned_any",
             "n": int(gene_assigned), "pct_of_total": pct(gene_assigned, s1), "pct_of_prev_stage": pct(gene_assigned, s1)},
            {"sample": sample, "window_bp": window_bp, "stage": "2_gene_expressed_in_section",
             "n": s2, "pct_of_total": pct(s2, s1), "pct_of_prev_stage": pct(s2, gene_assigned)},
            {"sample": sample, "window_bp": window_bp, "stage": "3_within_3prime_capture_window",
             "n": s3, "pct_of_total": pct(s3, s1), "pct_of_prev_stage": pct(s3, s2)},
            {"sample": sample, "window_bp": window_bp, "stage": "4_ge1_read_pooled_bam",
             "n": s4, "pct_of_total": pct(s4, s1), "pct_of_prev_stage": pct(s4, s3)},
            {"sample": sample, "window_bp": window_bp, "stage": "5_alt_allele_present",
             "n": s5, "pct_of_total": pct(s5, s1), "pct_of_prev_stage": pct(s5, s4)},
        ])
    return pd.DataFrame(stage_rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-waterfall", action="store_true", help="Debug: skip Task 2 (BAM pileups)")
    ap.add_argument("--skip-3prime", action="store_true", help="Debug: skip Task 3 (3' distance feature)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_3prime:
        df3p = build_three_prime_distance(out_dir)
        stats3p = three_prime_distance_stats(df3p)
        stats3p.to_csv(out_dir / "three_prime_distance_stats.csv", index=False)
        print(f"WROTE {out_dir}/three_prime_distance_stats.csv")
        print(stats3p.to_string())

    if not args.skip_waterfall:
        genes_hg19 = load_genes("hg19")
        for sample in WATERFALL_SAMPLES:
            wf = build_waterfall(sample, genes_hg19, out_dir)
            wf.to_csv(out_dir / f"waterfall_{sample.lower()}.csv", index=False)
            print(f"WROTE {out_dir}/waterfall_{sample.lower()}.csv")
            print(wf.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
