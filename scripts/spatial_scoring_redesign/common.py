#!/usr/bin/env python
"""
common.py
=========
Shared, READ-ONLY configuration and utilities for the Stage-2 spatial-scoring
redesign side-car (see data/spatial_scoring_redesign_2026-08-31/RESULTS.md).

This module never writes into any shipped pipeline directory
(data/<sample>/output_VCFs/, data/dlpfc/) and never runs / imports the shipped
scripts/1_calling .. scripts/6_spatial_filter code. It re-derives what it needs
(spot positions, the hexagonal neighbour graph, per-spot candidate presence)
straight from the same input files those scripts read, using the SAME
algorithm (ported by hand from scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py,
read there only).

Builds: P4/P6 = GRCh37, BAM contigs `chr`-prefixed. DCIS1/DCIS2 = GRCh38, BAM
contigs bare (no `chr`); the pipeline's own per-section text tables
(all_variant_scores.txt, vcf_by_spot/*.vcf.gz) use bare chrom for ALL FOUR
sections (verified directly, see RESULTS.md) -- only BAM contig lookups need
the chr_prefix_in_bam switch below.
"""
import glob
import gzip
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors

REPO = "/data/maiziezhou_lab/leiy4/snv_calling"
OUT_ROOT = f"{REPO}/data/spatial_scoring_redesign_2026-08-31"
CACHE_ROOT = f"{OUT_ROOT}/_cache"
CODE_ROOT = f"{REPO}/scripts/spatial_scoring_redesign"

QUALITY_FILTER = "baseQ0mapQ0"
NEIGHBOR_DISTANCE = 2.0   # same default used by every production run (see run_slurm/*/7_*.sh)
KNN_K = 13                # same as build_spatial_graph() in the shipped filter

# xMHC window used identically for GRCh37 and GRCh38 sections, matching the
# convention already established in data/xmhc_artifact_mechanism_2026-08-31/
XMHC_CHROM = "6"
XMHC_START = 28_000_000
XMHC_END = 34_000_000
XMHC_FRACTION_OF_GENOME = 0.00194   # 0.194%, as given in the task spec

# ---------------------------------------------------------------------------
# Section configuration. BAM paths + chr_prefix_in_bam copied verbatim from
# scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py
# (read-only source, not imported -- that script has its own CLI/output
# contract and this project must not couple to it).
# ---------------------------------------------------------------------------
SECTIONS = {
    "P4": dict(
        build="GRCh37",
        all_variant_scores=f"{REPO}/data/P4_tumor/1/spatial_filter_purity/{QUALITY_FILTER}/all_variant_scores.txt",
        vcf_by_spot_dir=f"{REPO}/data/P4_tumor/1/output_VCFs/spotprofiles/{QUALITY_FILTER}/vcf_by_spot",
        positions_file="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
                        "P4_Visium/spaceranger_align_rep1_hg19/Meta_Data/GSM4565823_P4_rep1_tissue_positions_list.csv",
        positions_has_header=False,
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        chr_prefix_in_bam=True,
    ),
    "P6": dict(
        build="GRCh37",
        all_variant_scores=f"{REPO}/data/P6_tumor/1/spatial_filter_purity/{QUALITY_FILTER}/all_variant_scores.txt",
        vcf_by_spot_dir=f"{REPO}/data/P6_tumor/1/output_VCFs/spotprofiles/{QUALITY_FILTER}/vcf_by_spot",
        positions_file="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
                        "P6_Visium/spaceranger_align_rep1_hg19/Meta_Data/GSM4565825_P6_rep1_tissue_positions_list.csv",
        positions_has_header=False,
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/"
            "P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        chr_prefix_in_bam=True,
    ),
    "DCIS1": dict(
        build="GRCh38",
        all_variant_scores=f"{REPO}/data/dcis1/spatial_filter_purity/{QUALITY_FILTER}/all_variant_scores.txt",
        vcf_by_spot_dir=f"{REPO}/data/dcis1/output_VCFs/spotprofiles/{QUALITY_FILTER}/vcf_by_spot",
        positions_file="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
                        "10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/spatial/tissue_positions.csv",
        positions_has_header=True,
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.dedup.bam",
        chr_prefix_in_bam=False,
    ),
    "DCIS2": dict(
        build="GRCh38",
        all_variant_scores=f"{REPO}/data/dcis2/spatial_filter_purity/{QUALITY_FILTER}/all_variant_scores.txt",
        vcf_by_spot_dir=f"{REPO}/data/dcis2/output_VCFs/spotprofiles/{QUALITY_FILTER}/vcf_by_spot",
        positions_file="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
                        "10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/spatial/tissue_positions.csv",
        positions_has_header=True,
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/"
            "10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.dedup.bam",
        chr_prefix_in_bam=False,
    ),
}

BARCODE_RE = re.compile(r'^[ACGTacgt]+-\d+$')


def cache_dir(section):
    d = os.path.join(CACHE_ROOT, section)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Spot positions -- ported from EnhancedSpatialSNVFilter._read_positions_csv /
# load_spot_positions (read-only; scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py)
# ---------------------------------------------------------------------------
def load_spot_positions(section):
    cfg = SECTIONS[section]
    path = cfg["positions_file"]
    if cfg["positions_has_header"]:
        with open(path) as fh:
            first_line = fh.readline().strip()
        first_col = first_line.split(',')[0]
        has_header = not bool(BARCODE_RE.match(first_col))
        df = pd.read_csv(path, header=0 if has_header else None)
        if has_header:
            df.columns = range(len(df.columns))
    else:
        df = pd.read_csv(path, header=None)

    positions = {}
    for _, row in df.iterrows():
        bc = str(row[0])
        if int(row[1]) != 1:
            continue
        pxl_row = float(row[4])
        pxl_col = float(row[5])
        positions[bc] = (pxl_col, pxl_row)
    return positions


# ---------------------------------------------------------------------------
# Hexagonal neighbour graph -- ported from
# EnhancedSpatialSNVFilter.build_spatial_graph (read-only).
# Symmetrized (union, i.e. W = max(W, W^T)) because join-count / Moran's I
# statistics require an undirected weight matrix; the shipped kNN+threshold
# construction is not guaranteed symmetric. Diagonal forced to zero.
# ---------------------------------------------------------------------------
def build_neighbor_graph(barcodes, positions_dict, neighbor_distance=NEIGHBOR_DISTANCE, k=KNN_K):
    positions = np.array([positions_dict[b] for b in barcodes])
    n = len(barcodes)

    tree = KDTree(positions)
    nn_dists, _ = tree.query(positions, k=2)
    median_pitch = float(np.median(nn_dists[:, 1]))
    neighbor_threshold = neighbor_distance * median_pitch

    kk = min(k, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(positions)
    dists, idxs = nn.kneighbors(positions)

    rows, cols = [], []
    for i in range(n):
        for j, d in zip(idxs[i][1:], dists[i][1:]):
            if d <= neighbor_threshold:
                rows.append(i)
                cols.append(j)
    data = np.ones(len(rows), dtype=np.float64)
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    W = W.maximum(W.T)
    W.setdiag(0)
    W.eliminate_zeros()
    W.data[:] = 1.0   # binary undirected adjacency
    return W, median_pitch, neighbor_threshold


# ---------------------------------------------------------------------------
# Candidate pool: the Stage-2 input population, read straight from the
# ALREADY-COMPUTED all_variant_scores.txt (race == denovo AND shipped
# classification in {somatic, ambiguous}, i.e. NOT excluded by Stage-1
# germline exclusion). This is an exact, read-only re-derivation -- verified
# arithmetically against the shipped pipeline's own log line
# ("Stage 1 -- ... N proceed to somatic voting") for DCIS2 in RESULTS.md.
# ---------------------------------------------------------------------------
def load_candidate_pool(section):
    cfg = SECTIONS[section]
    df = pd.read_csv(cfg["all_variant_scores"], sep="\t")
    denovo = df[df["race"] == "denovo"].copy()
    pool = denovo[denovo["classification"].isin(["somatic", "ambiguous"])].copy()
    pool = pool.reset_index(drop=True)
    chrom = pool["variant"].str.split("_", n=1).str[0]
    pos = pool["variant"].str.split("_", n=1).str[1].astype(int)
    pool["chrom"] = chrom
    pool["pos"] = pos
    pool = pool.rename(columns={"classification": "shipped_classification",
                                 "somatic_score": "shipped_vote_fraction"})
    return pool[["variant", "chrom", "pos", "shipped_classification", "shipped_vote_fraction"]]


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR (no statsmodels in snv_caller env -- implemented
# directly; standard step-up procedure).
# ---------------------------------------------------------------------------
def bh_qvalues(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    ranks = np.arange(1, n + 1)
    q = ranked * n / ranks
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def variant_to_chrom_pos(variant):
    chrom, pos = variant.split("_", 1)
    return chrom, int(pos)
