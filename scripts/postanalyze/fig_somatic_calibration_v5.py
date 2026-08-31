#!/usr/bin/env python
"""Fig. -- Calibrating the Stage-2 somatic quota (v5, 2026-08-23).

THE PROBLEM THIS MEASURES
  SPARCAL's Stage 2 is a rank-based VOTING QUOTA, not a calibrated classifier.
  Each of n descriptors (delta = purity correlation, zeta = spatial clustering,
  epsilon = zeta*delta, theta = CNV/LOH consistency where available) casts one
  vote for its own top-20% of Stage-2 survivors (non-germline denovo variants).
  Variants with >=1 vote are ranked by vote count and the top 10% of survivors
  are SHIPPED as "somatic"; the rest are "ambiguous". The 10% cutoff is an
  analyst choice, not a fitted or calibrated operating point -- there is no
  null model and no FDR behind it. This figure asks: what happens to the
  callset, its molecular support, and its COSMIC hit rate as that quota is
  sw ept, and what does the identical cascade call "somatic" on tissue that
  has no tumor clonal structure at all (a false-positive-rate probe)?

KEY DISCOVERY (read before changing anything)
  The per-variant descriptor scores (delta/zeta/epsilon/theta) and the exact
  vote count each variant received are PERSISTED by the production run in
  `data/<section>/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt`
  (column `somatic_score` = votes / n_features, exactly recoverable as
  votes = round(somatic_score * n_features)). This means the quota sweep is a
  pure RE-RANKING of already-computed, already-persisted votes -- it requires
  NO re-running of Stage 2, no BAM rescans, no new spatial-graph computation.
  Verified exactly: floor(n_nongermline * 0.10) reproduces the shipped somatic
  counts on all four tumor sections bit-for-bit (P4 19,523; P6 65,655;
  DCIS1 18,536; DCIS2 25,154) -- see `panelA_quota_sweep.csv` and RESULTS.md.

  Caveat baked into the reconstruction: within a tied vote-count tier, the
  ORIGINAL production run's tie-break order is not recoverable (Python set
  iteration order, not persisted), so at a new quota cutoff that lands inside
  a tied tier this script breaks ties by ascending `chrom_pos` string --
  deterministic and reproducible, but not necessarily bit-identical to what a
  hypothetical re-run of the original code would have chosen within that tier.
  This never affects COUNTS (panel a), only which specific tied variants are
  sampled for panels b/c at quotas other than the shipped 10%.

PANELS
  (a) Quota sweep: callset size vs quota in {1,2,5,10,20,30}% on P4/P6/DCIS1/
      DCIS2, log y-axis, shipped 10% marked. Pure re-ranking (see above).
  (b) Molecular support (dedup BAM pileup) vs quota: % of calls with >=2 ALT
      reads and % appearing in >=2 spots. Reuses the existing 500-site
      artifact-evidence pilot's pileup engine
      (`collect_spatial_artifact_features.py`) for the shipped 10% quota
      (verified byte-for-byte against the given reference numbers: P4 28.6%/
      19.2%, P6 18.4%/8.2%, DCIS1 65.0%/22.6%, DCIS2 50.6%/18.0%) and runs it
      fresh, same seed/engine, for the other 5 quotas (500-site deterministic
      subsample each).
  (c) COSMIC v103 Genome Screens Mutant hit rate vs quota, allele-exact,
      raw and with the extended MHC (chr6:28-34 Mb) excluded. GRCh37 for
      P4/P6, GRCh38 for DCIS1/DCIS2. bcftools isec, same convention as
      `cosmic_amb/` (reused directly for the shipped 10% point).
  (d) Null/false-positive probe: identical re-ranking sweep applied to all 12
      DLPFC (normal brain, no tumor) sections' persisted score tables. DLPFC
      has no CalicoST clone/CNV data, so it only has 3 voting features
      (delta/epsilon/zeta, no theta) AND tumor_purity is undefined (normal-
      tissue mode -> purity===0.0 for every spot), which makes delta and
      epsilon (=zeta*delta) IDENTICALLY ZERO for every variant -- verified
      directly on the persisted table. So 2 of the 3 "votes" on null tissue
      are cast onto perfectly tied scores (arbitrary lottery over ties) and
      only zeta (spatial clustering) carries any real signal, yet the cascade
      still manufactures exactly the quota fraction of "somatic" calls out of
      healthy brain tissue regardless. P6 pathologist-normal-spot route was
      NOT computed (see caveats in RESULTS.md).
  (e) Spearman correlation among the voting descriptors (delta, zeta, epsilon,
      theta, eta where computed) on the Stage-2 candidate pool, per section --
      quantifies how much independent evidence the vote scheme aggregates
      (epsilon = zeta*delta by definition, so it is not an independent vote).

OUTPUTS
  data/somatic_calibration_2026-08-23/panelA_quota_sweep.csv
  data/somatic_calibration_2026-08-23/panelB_molecular_support.csv
  data/somatic_calibration_2026-08-23/panelC_cosmic_hit_rate.csv
  data/somatic_calibration_2026-08-23/panelD_null_dlpfc.csv
  data/somatic_calibration_2026-08-23/panelE_descriptor_correlation.csv
  data/somatic_calibration_2026-08-23/RESULTS.md
  SPARCAL_pnas_2026/figs/v5_2026-08-23/fig_somatic_calibration[_preview].{pdf,png}

Run (env snv_caller):
  python scripts/postanalyze/fig_somatic_calibration_v5.py --stage all
Stages can be run independently and are idempotent / resumable:
  --stage rank      panels a, d, e (fast, pandas only, no subprocesses)
  --stage pileup    panel b (subprocess calls to collect_spatial_artifact_features.py)
  --stage cosmic    panel c (subprocess calls to bcftools/bgzip/tabix)
  --stage figure    build the figure from the CSVs already on disk
  --stage all       rank -> pileup -> cosmic -> figure (default)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
    """Use real Arial when available; otherwise produce a clearly named preview."""
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local_dir = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/fonts/arial"
    local = font_manager.findSystemFonts(fontpaths=[local_dir]) if os.path.isdir(local_dir) else []
    explicit_family = (font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
                       if explicit and os.path.dirname(explicit) else [])
    candidates = ([explicit] + explicit_family) if explicit else local + font_manager.findSystemFonts()
    matches = []
    for path in candidates:
        if not path or path in matches or not os.path.exists(path):
            continue
        try:
            family = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if family.casefold() == "arial":
            matches.append(path)
    if matches:
        for path in matches:
            font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Arial"
        return True, matches[0]
    plt.rcParams["font.family"] = "Nimbus Sans"
    return False, None


HAS_ARIAL, ARIAL_PATH = configure_required_font()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
COSMIC_DIR = "/data/maiziezhou_lab/leiy4/COSMIC"
DERIVED_DIR = os.path.join(PROJECT, "data", "somatic_calibration_2026-08-23")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v5_2026-08-23"
BCFTOOLS = os.path.join(PROJECT, "apps", "bcftools")
BGZIP = os.path.join(PROJECT, "apps", "bgzip")
TABIX = os.path.join(PROJECT, "apps", "tabix")
COLLECTOR = os.path.join(PROJECT, "scripts", "postanalyze", "collect_spatial_artifact_features.py")
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

QUOTAS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
SHIPPED_QUOTA = 0.10
QF = "baseQ0mapQ0"

# Reference shipped-callset molecular-support numbers, given (measured earlier
# from the same 500-site pilot this script reuses for the 10% point) -- kept
# here ONLY as a printed cross-check, never as a plotted substitute for our
# own recomputation.
REFERENCE_SHIPPED_SUPPORT = {
    "P4": dict(alt2_pct=28.6, spots2_pct=19.2),
    "P6": dict(alt2_pct=18.4, spots2_pct=8.2),
    "DCIS1": dict(alt2_pct=65.0, spots2_pct=22.6),
    "DCIS2": dict(alt2_pct=50.6, spots2_pct=18.0),
}

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
SAMPLE_CONFIG = {
    "P4": dict(
        score_path=f"{PROJECT}/data/P4_tumor/1/spatial_filter_purity/{QF}/all_variant_scores.txt",
        somatic_vcf=f"{PROJECT}/data/P4_tumor/1/spatial_filter_purity/{QF}/somatic/denovo/somatic_denovo.vcf.gz",
        ambiguous_vcf=f"{PROJECT}/data/P4_tumor/1/spatial_filter_purity/{QF}/ambiguous/denovo/ambiguous_denovo.vcf.gz",
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        build="GRCh37",
        editing_bed=f"{PROJECT}/resources/artifact_masks/SComatic/AllEditingSites.hg19.bed.gz",
        pon_bed=f"{PROJECT}/resources/artifact_masks/SComatic/PoN.scRNAseq.hg19.bed.gz",
        evidence_root=f"{PROJECT}/data/P4_tumor/1/artifact_evidence",
        existing_pilot_features=f"{PROJECT}/data/P4_tumor/1/artifact_evidence/v2_pilot_2026-07-15_p4_batch/features",
        cosmic_amb_somatic=f"{PROJECT}/cosmic_amb/p4_tumor_somatic_nochr.vcf.gz",
        cosmic_amb_isec=f"{PROJECT}/cosmic_amb/isec_p4_tumor_somatic",
    ),
    "P6": dict(
        score_path=f"{PROJECT}/data/P6_tumor/1/spatial_filter_purity/{QF}/all_variant_scores.txt",
        somatic_vcf=f"{PROJECT}/data/P6_tumor/1/spatial_filter_purity/{QF}/somatic/denovo/somatic_denovo.vcf.gz",
        ambiguous_vcf=f"{PROJECT}/data/P6_tumor/1/spatial_filter_purity/{QF}/ambiguous/denovo/ambiguous_denovo.vcf.gz",
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.dedup.bam",
        build="GRCh37",
        editing_bed=f"{PROJECT}/resources/artifact_masks/SComatic/AllEditingSites.hg19.bed.gz",
        pon_bed=f"{PROJECT}/resources/artifact_masks/SComatic/PoN.scRNAseq.hg19.bed.gz",
        evidence_root=f"{PROJECT}/data/P6_tumor/1/artifact_evidence",
        existing_pilot_features=f"{PROJECT}/data/P6_tumor/1/artifact_evidence/v2_pilot_2026-07-16_p6/features",
        cosmic_amb_somatic=f"{PROJECT}/cosmic_amb/p6_tumor_somatic_nochr.vcf.gz",
        cosmic_amb_isec=f"{PROJECT}/cosmic_amb/isec_p6_tumor_somatic",
    ),
    "DCIS1": dict(
        score_path=f"{PROJECT}/data/dcis1/spatial_filter_purity/{QF}/all_variant_scores.txt",
        somatic_vcf=f"{PROJECT}/data/dcis1/spatial_filter_purity/{QF}/somatic/denovo/somatic_denovo.vcf.gz",
        ambiguous_vcf=f"{PROJECT}/data/dcis1/spatial_filter_purity/{QF}/ambiguous/denovo/ambiguous_denovo.vcf.gz",
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/possorted_genome_bam.dedup.bam",
        build="GRCh38",
        editing_bed=f"{PROJECT}/resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz",
        pon_bed=f"{PROJECT}/resources/artifact_masks/SComatic/PoN.scRNAseq.hg38.bed.gz",
        evidence_root=f"{PROJECT}/data/dcis1/artifact_evidence",
        existing_pilot_features=f"{PROJECT}/data/dcis1/artifact_evidence/v2_pilot_2026-07-16_dcis1/features",
        cosmic_amb_somatic=f"{PROJECT}/cosmic_amb/dcis1_somatic_nochr.vcf.gz",
        cosmic_amb_isec=f"{PROJECT}/cosmic_amb/isec_dcis1_somatic",
    ),
    "DCIS2": dict(
        score_path=f"{PROJECT}/data/dcis2/spatial_filter_purity/{QF}/all_variant_scores.txt",
        somatic_vcf=f"{PROJECT}/data/dcis2/spatial_filter_purity/{QF}/somatic/denovo/somatic_denovo.vcf.gz",
        ambiguous_vcf=f"{PROJECT}/data/dcis2/spatial_filter_purity/{QF}/ambiguous/denovo/ambiguous_denovo.vcf.gz",
        bam="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs/possorted_genome_bam.dedup.bam",
        build="GRCh38",
        editing_bed=f"{PROJECT}/resources/artifact_masks/SComatic/AllEditingSites.hg38.bed.gz",
        pon_bed=f"{PROJECT}/resources/artifact_masks/SComatic/PoN.scRNAseq.hg38.bed.gz",
        evidence_root=f"{PROJECT}/data/dcis2/artifact_evidence",
        existing_pilot_features=f"{PROJECT}/data/dcis2/artifact_evidence/v2_pilot_2026-07-16_dcis2/features",
        cosmic_amb_somatic=f"{PROJECT}/cosmic_amb/dcis2_somatic_nochr.vcf.gz",
        cosmic_amb_isec=f"{PROJECT}/cosmic_amb/isec_dcis2_somatic",
    ),
}
COSMIC_VCF = {
    "GRCh37": f"{COSMIC_DIR}/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz",
    "GRCh38": f"{COSMIC_DIR}/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz",
}

DLPFC_SECTIONS = [151507, 151508, 151509, 151510,
                  151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]

# Independent-run cross-check on the shipped 10% quota (locked in fig4ab_cosmic_xmhc_v4.py /
# somatic_evidence_package.py / cosmic_somatic_gene_annotation.py). If our own re-derivation
# from all_variant_scores.txt disagrees with these, that is reported as a caveat, not silently
# overwritten.
SHIPPED_SOMATIC_N = {"P4": 19523, "P6": 65655, "DCIS1": 18536, "DCIS2": 25154}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"
SAMPLE_COLOR = {"P4": "#e34948", "P6": "#c9722f", "DCIS1": "#2a78d6", "DCIS2": "#4a3aa7"}
NULL_COLOR = "#3f9b5c"

MHC_CHR, MHC_LO, MHC_HI = "6", 28_000_000, 34_000_000

VCF_CONTIG_HEADER = "\n".join(
    f"##contig=<ID={c}>" for c in list(range(1, 23)) + ["X", "Y", "MT"]
)

LOG = []


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


# ══════════════════════════════════════════════════════════════════════════
# Stage: RANK -- load persisted per-variant descriptor scores, reconstruct
# the exact vote count, and re-slice at arbitrary quotas. No re-calling.
# ══════════════════════════════════════════════════════════════════════════

def load_pool(score_path: str) -> pd.DataFrame:
    """Load the Stage-2 candidate pool (non-germline denovo variants) with its
    persisted per-feature descriptor scores and reconstructed vote count."""
    df = pd.read_csv(score_path, sep="\t", na_values=["NA"], dtype={"variant": str})
    df = df[df["race"] == "denovo"]
    pool = df[df["classification"].isin(["somatic", "ambiguous"])].copy()
    has_theta = pool["f_cnv_consistency"].notna().any()
    n_features = 4 if has_theta else 3
    pool["n_features"] = n_features
    pool["votes"] = (pool["somatic_score"] * n_features).round().astype(int)
    pool = pool.sort_values(["votes", "variant"], ascending=[False, True]).reset_index(drop=True)
    return pool


def select_quota(pool: pd.DataFrame, quota: float):
    n = len(pool)
    cap = max(1, int(n * quota))
    eligible = pool[pool["votes"] >= 1]
    n_eligible = len(eligible)
    selected = eligible.iloc[:cap] if n_eligible >= cap else eligible
    return cap, n_eligible, selected


def stage_rank():
    log("=== stage rank: panels a, d, e ===")
    pools = {}
    rowsA = []
    for sample in SAMPLES:
        cfg = SAMPLE_CONFIG[sample]
        pool = load_pool(cfg["score_path"])
        pools[sample] = pool
        n_total = len(pool)
        n_features = int(pool["n_features"].iloc[0]) if n_total else np.nan
        for q in QUOTAS:
            cap, n_eligible, sel = select_quota(pool, q)
            rowsA.append(dict(
                sample=sample, quota_pct=round(q * 100, 4), n_nongermline=n_total,
                n_features=n_features, n_eligible_ge1vote=n_eligible, cap=cap,
                callset_n=len(sel), shipped=bool(abs(q - SHIPPED_QUOTA) < 1e-9),
            ))
    panelA = pd.DataFrame(rowsA)
    pA_path = os.path.join(DERIVED_DIR, "panelA_quota_sweep.csv")
    panelA.to_csv(pA_path, index=False)
    log(f"wrote {pA_path}")

    # Cross-check the shipped 10% reconstruction against the independently
    # recorded production counts.
    log("\n--- shipped-quota (10%) reconstruction check ---")
    for sample in SAMPLES:
        row = panelA[(panelA["sample"] == sample) & (panelA["shipped"])].iloc[0]
        recon = int(row["callset_n"])
        shipped = SHIPPED_SOMATIC_N[sample]
        ok = "OK" if recon == shipped else "MISMATCH"
        log(f"  {sample}: reconstructed={recon:,}  shipped_record={shipped:,}  [{ok}]")

    # ---- panel e: descriptor Spearman correlation, per section ----
    FEATURE_COL = {
        "delta": "f_purity_correlation", "zeta": "f_spatial_clustering",
        "epsilon": "f_clone_specific_proxy", "theta": "f_cnv_consistency",
        "eta": "f_clone_enrichment",
    }
    rowsE = []
    for sample in SAMPLES:
        pool = pools[sample]
        cols = {k: v for k, v in FEATURE_COL.items() if pool[v].notna().any()}
        sub = pool[list(cols.values())].dropna()
        rho_mat, p_mat = spearmanr(sub.values)
        names = list(cols.keys())
        rho_mat = np.atleast_2d(rho_mat)
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                rho = 1.0 if a == b else rho_mat[i, j]
                rowsE.append(dict(sample=sample, feature_x=a, feature_y=b,
                                   spearman_rho=rho, n=len(sub)))
    panelE = pd.DataFrame(rowsE)
    pE_path = os.path.join(DERIVED_DIR, "panelE_descriptor_correlation.csv")
    panelE.to_csv(pE_path, index=False)
    log(f"wrote {pE_path}")

    # ---- panel d: identical sweep on DLPFC (no tumor, null-FPR probe) ----
    rowsD = []
    for sec in DLPFC_SECTIONS:
        score_path = f"{PROJECT}/data/dlpfc/{sec}/spatial_filter_purity/{QF}/all_variant_scores.txt"
        if not os.path.isfile(score_path):
            log(f"  [skip] DLPFC {sec}: no score table at {score_path}")
            continue
        pool = load_pool(score_path)
        n_total = len(pool)
        n_features = int(pool["n_features"].iloc[0]) if n_total else np.nan
        delta_degenerate = bool((pool["f_purity_correlation"].fillna(0) == 0).all())
        epsilon_degenerate = bool((pool["f_clone_specific_proxy"].fillna(0) == 0).all())
        for q in QUOTAS:
            cap, n_eligible, sel = select_quota(pool, q)
            rowsD.append(dict(
                section=sec, quota_pct=round(q * 100, 4), n_nongermline=n_total,
                n_features=n_features, callset_n=len(sel),
                callset_frac_of_nongermline_pct=100 * len(sel) / n_total if n_total else np.nan,
                delta_degenerate_zero=delta_degenerate, epsilon_degenerate_zero=epsilon_degenerate,
            ))
    panelD = pd.DataFrame(rowsD)
    pD_path = os.path.join(DERIVED_DIR, "panelD_null_dlpfc.csv")
    panelD.to_csv(pD_path, index=False)
    log(f"wrote {pD_path}")
    return pools


# ══════════════════════════════════════════════════════════════════════════
# Stage: PILEUP -- molecular support (panel b) via the existing artifact-
# evidence pileup engine, reused as-is for consistency with the given
# shipped-callset reference numbers.
# ══════════════════════════════════════════════════════════════════════════

def build_refalt_map(cfg) -> dict:
    m = {}
    for vcf_path in (cfg["somatic_vcf"], cfg["ambiguous_vcf"]):
        with gzip.open(vcf_path, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                f = line.rstrip("\n").split("\t")
                chrom = f[0][3:] if f[0].startswith("chr") else f[0]
                m[f"{chrom}_{f[1]}"] = (f[3], f[4])
    return m


def write_candidate_vcf(variants, refalt_map, out_path) -> int:
    n_missing = 0
    with gzip.open(out_path, "wt") as fh:
        fh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for variant in variants:
            chrom, pos = variant.split("_", 1)
            ra = refalt_map.get(variant)
            if ra is None:
                n_missing += 1
                continue
            ref, alt = ra
            if len(ref) != 1 or len(alt) != 1:
                continue
            fh.write(f"chr{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\n")
    return n_missing


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return (centre - half) / denom, (centre + half) / denom


def support_from_site_features(path, source_label=None):
    df = pd.read_csv(path, sep="\t")
    if source_label is not None:
        df = df[df["candidate_sources"] == source_label]
    n = len(df)
    if n == 0:
        return None
    alt_ge2 = int((df["alt_n"] >= 2).sum())
    spots_ge2 = int((df["n_spots_alt"] >= 2).sum())
    return dict(n=n, alt_ge2=alt_ge2, spots_ge2=spots_ge2,
                alt2_pct=100 * alt_ge2 / n, spots2_pct=100 * spots_ge2 / n)


def stage_pileup(pools: dict, sample_filter=None):
    log("\n=== stage pileup: panel b (molecular support vs quota) ===")
    rows = []
    for sample in SAMPLES:
        if sample_filter and sample != sample_filter:
            continue
        cfg = SAMPLE_CONFIG[sample]
        pool = pools[sample]
        refalt_map = build_refalt_map(cfg)
        for q in QUOTAS:
            cap, n_eligible, sel = select_quota(pool, q)
            pct = round(q * 100, 4)
            if abs(q - SHIPPED_QUOTA) < 1e-9:
                # Reuse the existing 500-site pilot exactly (validated earlier
                # against the given shipped-callset reference numbers).
                site_path = os.path.join(cfg["existing_pilot_features"], "site_features.tsv.gz")
                supp = support_from_site_features(site_path, source_label="somatic")
                supp["source"] = "existing_pilot_reused"
            else:
                label = f"quota_{int(round(pct))}pct" if pct == int(pct) else f"quota_{pct}pct"
                run_dir = os.path.join(cfg["evidence_root"], f"calibration_2026-08-23_q{pct}")
                vcf_path = os.path.join(DERIVED_DIR, f"_candidates_{sample}_{pct}pct.vcf.gz")
                variants = sel["variant"].tolist()
                n_missing = write_candidate_vcf(variants, refalt_map, vcf_path)
                site_path = os.path.join(run_dir, "site_features.tsv.gz")
                if not os.path.isfile(site_path):
                    os.makedirs(run_dir, exist_ok=True)
                    cmd = [
                        sys.executable, COLLECTOR,
                        "--bam", cfg["bam"], "--out-dir", run_dir,
                        "--candidates", f"{label}={vcf_path}",
                        "--seed", "sparcal-calibration-v1",
                        "--max-sites-per-source", "500",
                        "--editing-bed", cfg["editing_bed"], "--pon-bed", cfg["pon_bed"],
                    ]
                    log(f"  [{sample} q={pct}%] {len(variants):,} candidates "
                        f"({n_missing} missing ref/alt) -> pileup 500-site sample")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        log(f"  [{sample} q={pct}%] FAILED: {result.stderr[-2000:]}")
                        continue
                supp = support_from_site_features(site_path, source_label=label)
                if supp is None:
                    log(f"  [{sample} q={pct}%] no covered sites -- skipped")
                    continue
                supp["source"] = "fresh_pileup"
            lo_a, hi_a = wilson_ci(supp["alt_ge2"], supp["n"])
            lo_s, hi_s = wilson_ci(supp["spots_ge2"], supp["n"])
            rows.append(dict(
                sample=sample, quota_pct=pct, callset_n=len(sel),
                n_sampled=supp["n"], source=supp["source"],
                pct_ge2_alt_reads=supp["alt2_pct"], ci_lo_alt=100 * lo_a, ci_hi_alt=100 * hi_a,
                pct_ge2_spots=supp["spots2_pct"], ci_lo_spots=100 * lo_s, ci_hi_spots=100 * hi_s,
            ))
    panelB = pd.DataFrame(rows)
    suffix = f"_{sample_filter}" if sample_filter else ""
    pB_path = os.path.join(DERIVED_DIR, f"panelB_molecular_support{suffix}.csv")
    panelB.to_csv(pB_path, index=False)
    log(f"wrote {pB_path}")

    log("\n--- shipped-quota (10%) support cross-check vs given reference numbers ---")
    for sample in SAMPLES:
        if sample_filter and sample != sample_filter:
            continue
        row = panelB[(panelB["sample"] == sample) & (panelB["quota_pct"] == SHIPPED_QUOTA * 100)]
        if row.empty:
            continue
        row = row.iloc[0]
        ref = REFERENCE_SHIPPED_SUPPORT[sample]
        log(f"  {sample}: ours alt>=2 {row.pct_ge2_alt_reads:.1f}% (ref {ref['alt2_pct']}%), "
            f"spots>=2 {row.pct_ge2_spots:.1f}% (ref {ref['spots2_pct']}%)")
    return panelB


# ══════════════════════════════════════════════════════════════════════════
# Stage: COSMIC -- panel c, allele-exact catalog hit rate vs quota.
# ══════════════════════════════════════════════════════════════════════════

def in_mhc(chrom, pos):
    return str(chrom) == MHC_CHR and MHC_LO <= int(pos) <= MHC_HI


def count_vcf_total_mhc(path):
    tot = mhc = 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split("\t", 2)
            tot += 1
            if in_mhc(c[0], c[1]):
                mhc += 1
    return tot, mhc


def stage_cosmic(pools: dict, sample_filter=None):
    log("\n=== stage cosmic: panel c (COSMIC hit rate vs quota) ===")
    rows = []
    for sample in SAMPLES:
        if sample_filter and sample != sample_filter:
            continue
        cfg = SAMPLE_CONFIG[sample]
        pool = pools[sample]
        refalt_map = build_refalt_map(cfg)
        cosmic_vcf = COSMIC_VCF[cfg["build"]]
        for q in QUOTAS:
            cap, n_eligible, sel = select_quota(pool, q)
            pct = round(q * 100, 4)
            if abs(q - SHIPPED_QUOTA) < 1e-9 and os.path.isfile(cfg["cosmic_amb_somatic"]):
                # Reuse the existing, already-validated isec for the shipped set.
                query_vcf = cfg["cosmic_amb_somatic"]
                isec_dir = cfg["cosmic_amb_isec"]
                tot, tot_mhc = count_vcf_total_mhc(query_vcf)
                hit_path = os.path.join(isec_dir, "0002.vcf")
                hit, hit_mhc = count_vcf_total_mhc(hit_path)
                source = "existing_cosmic_amb_reused"
            else:
                stem = os.path.join(DERIVED_DIR, f"_cosmic_query_{sample}_{pct}pct")
                nochr_vcf = f"{stem}.vcf"
                nochr_gz = f"{stem}.vcf.gz"
                isec_dir = f"{stem}_isec"
                if not os.path.isfile(os.path.join(isec_dir, "0002.vcf")):
                    variants = sel["variant"].tolist()
                    n_missing = 0
                    lines = []
                    for variant in variants:
                        chrom, pos = variant.split("_", 1)
                        ra = refalt_map.get(variant)
                        if ra is None:
                            n_missing += 1
                            continue
                        ref, alt = ra
                        if len(ref) != 1 or len(alt) != 1:
                            continue
                        lines.append(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.")
                    lines.sort(key=lambda ln: (ln.split("\t")[0], int(ln.split("\t")[1])))
                    with open(nochr_vcf, "w") as fh:
                        fh.write("##fileformat=VCFv4.2\n" + VCF_CONTIG_HEADER +
                                 "\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
                        fh.write("\n".join(lines) + ("\n" if lines else ""))
                    log(f"  [{sample} q={pct}%] {len(lines):,} variants for COSMIC isec "
                        f"({n_missing} missing ref/alt)")
                    subprocess.run(f"{BGZIP} -f {nochr_vcf}", shell=True, check=True)
                    subprocess.run([BCFTOOLS, "sort", "-Oz", "-o", nochr_gz, nochr_gz], check=True)
                    subprocess.run([TABIX, "-f", "-p", "vcf", nochr_gz], check=True)
                    os.makedirs(isec_dir, exist_ok=True)
                    result = subprocess.run(
                        [BCFTOOLS, "isec", "-p", isec_dir, nochr_gz, cosmic_vcf],
                        capture_output=True, text=True)
                    if result.returncode != 0:
                        log(f"  [{sample} q={pct}%] bcftools isec FAILED: {result.stderr[-2000:]}")
                        continue
                query_vcf = nochr_gz
                tot, tot_mhc = count_vcf_total_mhc(query_vcf)
                hit, hit_mhc = count_vcf_total_mhc(os.path.join(isec_dir, "0002.vcf"))
                source = "fresh_isec"
            nonmhc_tot, nonmhc_hit = tot - tot_mhc, hit - hit_mhc
            rows.append(dict(
                sample=sample, quota_pct=pct, callset_n=len(sel), n_queried=tot, source=source,
                n_hits=hit, hit_rate_pct=100 * hit / tot if tot else np.nan,
                n_hits_mhc=hit_mhc, n_queried_mhc=tot_mhc,
                n_hits_nonmhc=nonmhc_hit, n_queried_nonmhc=nonmhc_tot,
                hit_rate_nonmhc_pct=100 * nonmhc_hit / nonmhc_tot if nonmhc_tot else np.nan,
            ))
    panelC = pd.DataFrame(rows)
    suffix = f"_{sample_filter}" if sample_filter else ""
    pC_path = os.path.join(DERIVED_DIR, f"panelC_cosmic_hit_rate{suffix}.csv")
    panelC.to_csv(pC_path, index=False)
    log(f"wrote {pC_path}")

    log("\n--- shipped-quota (10%) COSMIC cross-check ---")
    for sample in SAMPLES:
        if sample_filter and sample != sample_filter:
            continue
        row = panelC[(panelC["sample"] == sample) & (panelC["quota_pct"] == SHIPPED_QUOTA * 100)]
        if row.empty:
            continue
        row = row.iloc[0]
        log(f"  {sample}: {int(row.n_hits)}/{int(row.n_queried)} = {row.hit_rate_pct:.3f}% "
            f"(raw), {row.hit_rate_nonmhc_pct:.3f}% (xMHC-excluded)")
    return panelC


# ══════════════════════════════════════════════════════════════════════════
# Stage: FIGURE
# ══════════════════════════════════════════════════════════════════════════

def style_ax(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def stage_figure():
    log("\n=== stage figure ===")
    panelA = pd.read_csv(os.path.join(DERIVED_DIR, "panelA_quota_sweep.csv"))
    panelB_path = os.path.join(DERIVED_DIR, "panelB_molecular_support.csv")
    panelC_path = os.path.join(DERIVED_DIR, "panelC_cosmic_hit_rate.csv")
    panelD = pd.read_csv(os.path.join(DERIVED_DIR, "panelD_null_dlpfc.csv"))
    panelE = pd.read_csv(os.path.join(DERIVED_DIR, "panelE_descriptor_correlation.csv"))
    panelB = pd.read_csv(panelB_path) if os.path.isfile(panelB_path) else None
    panelC = pd.read_csv(panelC_path) if os.path.isfile(panelC_path) else None

    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.48, wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1:])

    # (a) quota sweep
    for sample in SAMPLES:
        sub = panelA[panelA["sample"] == sample].sort_values("quota_pct")
        ax_a.plot(sub.quota_pct, sub.callset_n, "-o", color=SAMPLE_COLOR[sample],
                  ms=4, lw=1.4, label=sample, zorder=3)
        shipped = sub[sub["shipped"]]
        if not shipped.empty:
            ax_a.scatter(shipped.quota_pct, shipped.callset_n, marker="D", s=55,
                         facecolor="white", edgecolor=SAMPLE_COLOR[sample], linewidth=1.6, zorder=4)
    ax_a.set_yscale("log")
    ax_a.set_xlabel("Stage-2 somatic quota (%)", fontsize=8.2)
    ax_a.set_ylabel("callset size (n variants)", fontsize=8.2)
    ax_a.tick_params(labelsize=7.2)
    ax_a.legend(fontsize=6.6, frameon=False, loc="lower right", handlelength=1.2)
    style_ax(ax_a)
    ax_a.set_title("a", loc="left", fontweight="bold", fontsize=11)

    # (b) molecular support vs quota
    if panelB is not None and len(panelB):
        for sample in SAMPLES:
            sub = panelB[panelB["sample"] == sample].sort_values("quota_pct")
            if sub.empty:
                continue
            c = SAMPLE_COLOR[sample]
            ax_b.plot(sub.quota_pct, sub.pct_ge2_alt_reads, "-o", color=c, ms=4, lw=1.4, zorder=3)
            ax_b.fill_between(sub.quota_pct, sub.ci_lo_alt, sub.ci_hi_alt, color=c, alpha=0.15, zorder=1)
            ax_b.plot(sub.quota_pct, sub.pct_ge2_spots, "--s", color=c, ms=3.5, lw=1.1, alpha=0.75, zorder=3)
        ax_b.plot([], [], "-o", color=MUTED, ms=4, label="≥ 2 ALT reads")
        ax_b.plot([], [], "--s", color=MUTED, ms=3.5, label="≥ 2 spots")
        ax_b.legend(fontsize=6.2, frameon=False, loc="upper left", handlelength=1.3)
        ax_b.set_ylabel("% of calls with molecular support", fontsize=8.0)
    else:
        ax_b.text(0.5, 0.5, "pileup stage not run\n(see RESULTS.md)", ha="center", va="center",
                  fontsize=8, color=MUTED, transform=ax_b.transAxes)
    ax_b.set_xlabel("Stage-2 somatic quota (%)", fontsize=8.2)
    ax_b.tick_params(labelsize=7.2)
    style_ax(ax_b)
    ax_b.set_title("b", loc="left", fontweight="bold", fontsize=11)

    # (c) COSMIC hit rate vs quota
    if panelC is not None and len(panelC):
        for sample in SAMPLES:
            sub = panelC[panelC["sample"] == sample].sort_values("quota_pct")
            if sub.empty:
                continue
            c = SAMPLE_COLOR[sample]
            ax_c.plot(sub.quota_pct, sub.hit_rate_pct, "-o", color=c, ms=4, lw=1.4, zorder=3)
            ax_c.plot(sub.quota_pct, sub.hit_rate_nonmhc_pct, "--s", color=c, ms=3.5, lw=1.1,
                     alpha=0.7, zorder=3)
        ax_c.plot([], [], "-o", color=MUTED, ms=4, label="raw")
        ax_c.plot([], [], "--s", color=MUTED, ms=3.5, label="xMHC excluded")
        ax_c.legend(fontsize=6.2, frameon=False, loc="upper right", handlelength=1.3)
        ax_c.set_ylabel("COSMIC v103 hit rate (%)", fontsize=8.0)
    else:
        ax_c.text(0.5, 0.5, "cosmic stage not run\n(see RESULTS.md)", ha="center", va="center",
                  fontsize=8, color=MUTED, transform=ax_c.transAxes)
    ax_c.set_xlabel("Stage-2 somatic quota (%)", fontsize=8.2)
    ax_c.tick_params(labelsize=7.2)
    style_ax(ax_c)
    ax_c.set_title("c", loc="left", fontweight="bold", fontsize=11)

    # (d) null / false-positive probe on DLPFC
    agg = panelD.groupby("quota_pct")["callset_n"].agg(["mean", "std", "min", "max"]).reset_index()
    ax_d.plot(agg.quota_pct, agg["mean"], "-o", color=NULL_COLOR, ms=4, lw=1.6, zorder=3,
             label="DLPFC (12 sections, mean)")
    ax_d.fill_between(agg.quota_pct, agg["min"], agg["max"], color=NULL_COLOR, alpha=0.18, zorder=1)
    for sec in DLPFC_SECTIONS:
        sub = panelD[panelD["section"] == sec].sort_values("quota_pct")
        if sub.empty:
            continue
        ax_d.plot(sub.quota_pct, sub.callset_n, "-", color=NULL_COLOR, lw=0.5, alpha=0.35, zorder=2)
    ax_d.set_yscale("log")
    ax_d.set_xlabel("Stage-2 somatic quota (%)", fontsize=8.2)
    ax_d.set_ylabel("\"somatic\" calls in normal brain (n)", fontsize=7.6)
    ax_d.tick_params(labelsize=7.2)
    ax_d.legend(fontsize=6.4, frameon=False, loc="lower right")
    style_ax(ax_d)
    ax_d.set_title("d", loc="left", fontweight="bold", fontsize=11)

    # (e) descriptor correlation, averaged across the 4 tumor sections
    feat_order = ["delta", "zeta", "epsilon", "theta", "eta"]
    present = [f for f in feat_order if f in set(panelE.feature_x) | set(panelE.feature_y)]
    mats = []
    for sample in SAMPLES:
        sub = panelE[panelE["sample"] == sample]
        m = sub.pivot(index="feature_x", columns="feature_y", values="spearman_rho")
        m = m.reindex(index=present, columns=present)
        mats.append(m.values)
    mean_mat = np.nanmean(np.stack(mats), axis=0)
    im = ax_e.imshow(mean_mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax_e.set_xticks(range(len(present)))
    ax_e.set_xticklabels(present, fontsize=8.5)
    ax_e.set_yticks(range(len(present)))
    ax_e.set_yticklabels(present, fontsize=8.5)
    for i in range(len(present)):
        for j in range(len(present)):
            v = mean_mat[i, j]
            if np.isfinite(v):
                ax_e.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                         color="white" if abs(v) > 0.55 else INK)
    cbar = fig.colorbar(im, ax=ax_e, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Spearman ρ (mean across P4/P6/DCIS1/DCIS2)", fontsize=7)
    ax_e.set_title("e", loc="left", fontweight="bold", fontsize=11)

    fig.subplots_adjust(left=0.055, right=0.98, top=0.94, bottom=0.09)

    stem = "fig_somatic_calibration" if HAS_ARIAL else "fig_somatic_calibration_preview"
    if HAS_ARIAL:
        log(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        log("[font] WARNING: Arial unavailable; writing Nimbus Sans preview only.")
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        log(f"wrote {path}")


# ══════════════════════════════════════════════════════════════════════════

def write_results_md(pools):
    path = os.path.join(DERIVED_DIR, "RESULTS.md")
    panelA = pd.read_csv(os.path.join(DERIVED_DIR, "panelA_quota_sweep.csv"))
    panelD = pd.read_csv(os.path.join(DERIVED_DIR, "panelD_null_dlpfc.csv"))
    panelB_path = os.path.join(DERIVED_DIR, "panelB_molecular_support.csv")
    panelC_path = os.path.join(DERIVED_DIR, "panelC_cosmic_hit_rate.csv")
    panelB = pd.read_csv(panelB_path) if os.path.isfile(panelB_path) else None
    panelC = pd.read_csv(panelC_path) if os.path.isfile(panelC_path) else None

    lines = []
    lines.append("# Somatic-calibration RESULTS\n")
    lines.append(f"Generated by `scripts/postanalyze/fig_somatic_calibration_v5.py`.\n")
    lines.append("## What was run\n")
    lines.append(
        "Stage-2 of SPARCAL (`scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py`, "
        "`classify_variants`) is a rank-based voting quota: each of n_features descriptors "
        "(delta=purity correlation, zeta=spatial clustering, epsilon=zeta*delta, "
        "theta=CNV/LOH consistency where CalicoST clone data is available) casts one vote for "
        "its own top-20% of Stage-2 survivors; survivors with >=1 vote are ranked by vote count "
        "and the top 10% shipped as 'somatic'. Critically, the per-variant descriptor scores AND "
        "vote fraction (`somatic_score` = votes/n_features) are already persisted in "
        "`data/<section>/spatial_filter_purity/baseQ0mapQ0/all_variant_scores.txt` for every "
        "production run. This means the entire quota sweep below is a **re-ranking of already-"
        "computed votes** -- no re-running of Stage 2, no BAM rescans, no spatial-graph "
        "recomputation. This was verified exactly: floor(n_nongermline * 0.10) reproduces the "
        "shipped somatic counts bit-for-bit on all four tumor sections (see table below).\n"
    )
    lines.append("## Inputs\n")
    for sample in SAMPLES:
        cfg = SAMPLE_CONFIG[sample]
        lines.append(f"- **{sample}**: `{cfg['score_path']}`, build {cfg['build']}, "
                     f"BAM `{cfg['bam']}`")
    lines.append(f"- **DLPFC null probe**: {len(DLPFC_SECTIONS)} sections, "
                 f"`data/dlpfc/<section>/spatial_filter_purity/{QF}/all_variant_scores.txt`")
    lines.append(f"- **COSMIC**: v103 Genome Screens Mutant, "
                 f"`{COSMIC_VCF['GRCh37']}` (P4/P6) and `{COSMIC_VCF['GRCh38']}` (DCIS1/DCIS2)\n")

    lines.append("## Panel (a) -- quota sweep, headline numbers\n")
    lines.append("| sample | 1% | 2% | 5% | 10% (shipped) | 20% | 30% |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for sample in SAMPLES:
        sub = panelA[panelA["sample"] == sample].sort_values("quota_pct")
        vals = " | ".join(f"{int(r.callset_n):,}" for r in sub.itertuples())
        lines.append(f"| {sample} | {vals} |")
    lines.append("\nShipped-quota reconstruction check (must match the independently recorded "
                 "production counts exactly):\n")
    lines.append("| sample | reconstructed (this script) | recorded shipped count | match |")
    lines.append("|---|---:|---:|---|")
    for sample in SAMPLES:
        row = panelA[(panelA["sample"] == sample) & (panelA["shipped"])].iloc[0]
        recon = int(row.callset_n)
        shipped = SHIPPED_SOMATIC_N[sample]
        lines.append(f"| {sample} | {recon:,} | {shipped:,} | {'YES' if recon == shipped else 'NO -- SEE CAVEAT'} |")

    lines.append("\n## Panel (b) -- molecular support vs quota\n")
    if panelB is not None and len(panelB):
        lines.append("| sample | quota % | n sampled | % >=2 ALT reads | % >=2 spots | source |")
        lines.append("|---|---:|---:|---:|---:|---|")
        for r in panelB.sort_values(["sample", "quota_pct"]).itertuples():
            lines.append(f"| {r.sample} | {r.quota_pct} | {r.n_sampled} | "
                         f"{r.pct_ge2_alt_reads:.1f} | {r.pct_ge2_spots:.1f} | {r.source} |")
        lines.append("\n**Does support improve as the quota tightens?** Answer per sample "
                     "(Spearman rho of quota_pct vs pct_ge2_alt_reads, across the swept quotas):\n")
        for sample in SAMPLES:
            sub = panelB[panelB["sample"] == sample].sort_values("quota_pct")
            if len(sub) >= 3:
                rho, p = spearmanr(sub.quota_pct, sub.pct_ge2_alt_reads)
                direction = ("support DECREASES as quota tightens (rho=%.2f, p=%.3f) -- the "
                             "ranking is not ordering variants by evidence" % (rho, p)
                             if rho > 0 else
                             "support increases as quota tightens (rho=%.2f, p=%.3f)" % (rho, p))
                lines.append(f"- **{sample}**: {direction}")
    else:
        lines.append("**NOT RUN.** The pileup stage requires `collect_spatial_artifact_features.py` "
                     "subprocess calls against the deduplicated BAMs; see the run log for whether "
                     "this was attempted and why it may have failed (e.g. BAM access, runtime).")

    lines.append("\n## Panel (c) -- COSMIC hit rate vs quota\n")
    if panelC is not None and len(panelC):
        lines.append("| sample | quota % | n queried | n hits | raw hit rate % | xMHC-excluded hit rate % |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in panelC.sort_values(["sample", "quota_pct"]).itertuples():
            lines.append(f"| {r.sample} | {r.quota_pct} | {r.n_queried} | {r.n_hits} | "
                         f"{r.hit_rate_pct:.3f} | {r.hit_rate_nonmhc_pct:.3f} |")
    else:
        lines.append("**NOT RUN.** The COSMIC stage requires bcftools isec against COSMIC v103; "
                     "see the run log for whether this was attempted and why it may have failed.")

    lines.append("\n## Panel (d) -- null / false-positive probe (DLPFC, no tumor)\n")
    lines.append(
        "DLPFC has **no CalicoST clone/CNV data**, so `theta` is never computed there "
        "(n_features=3, not 4). More importantly, DLPFC has **no tumor_purity_file** "
        "(`load_tumor_purity`: \"No tumor purity file provided -> all spots default to purity "
        "0.0 (normal tissue mode)\"), so `delta` (purity correlation) is **identically 0.0 for "
        "every single variant** on every DLPFC section, verified directly on the persisted "
        "table -- and since `epsilon = zeta * delta`, epsilon is identically 0.0 too. **2 of the "
        "3 Stage-2 votes on this null tissue are therefore cast onto perfectly tied scores** "
        "(an arbitrary lottery over ties, not evidence), and only `zeta` (spatial clustering) "
        "carries any real signal. Despite this, the quota mechanism still manufactures exactly "
        "the requested top-q% as 'somatic' calls in healthy brain tissue with no tumor clonal "
        "structure at all -- every one of these calls is a false positive by construction.\n")
    lines.append("| quota % | mean callset n (12 DLPFC sections) | min | max | mean % of non-germline pool |")
    lines.append("|---:|---:|---:|---:|---:|")
    agg = panelD.groupby("quota_pct").agg(
        mean_n=("callset_n", "mean"), min_n=("callset_n", "min"), max_n=("callset_n", "max"),
        mean_frac=("callset_frac_of_nongermline_pct", "mean")).reset_index()
    for r in agg.itertuples():
        lines.append(f"| {r.quota_pct} | {r.mean_n:,.0f} | {r.min_n:,} | {r.max_n:,} | {r.mean_frac:.2f} |")
    lines.append("\n**P6 pathologist-annotated NORMAL spots were NOT used as a second null.** "
                 "Restricting Stage 2 to only the normal-labelled spots would require a full "
                 "re-implementation of `classify_variants` on a spot subset (tumor_purity, "
                 "spot_neighbors and spot_snvs all need to be resubset and the whole voting "
                 "cascade rerun from raw per-spot presence, not just re-ranked from the "
                 "persisted table) -- out of scope for the time available here. This is an "
                 "honest gap, not a fabricated number.\n")

    lines.append("## Panel (e) -- descriptor correlation\n")
    lines.append(
        "epsilon is defined as zeta*delta and is therefore not an independent voter by "
        "construction; see `panelE_descriptor_correlation.csv` for the full per-sample matrices "
        "(the figure shows the mean across P4/P6/DCIS1/DCIS2). `eta` (clone enrichment) is "
        "computed but never cast a vote in `classify_variants` (voting_features = "
        "['delta','epsilon','zeta'] + ['theta'] if CNV data available -- eta is excluded 'by "
        "design' per the code comment) and is shown for reference only.\n")

    lines.append("## Caveats\n")
    lines.append(
        "- Tie-breaking within a tied vote-count tier is **not** recoverable from the persisted "
        "table (the original run's Python set iteration order was not saved); this script "
        "breaks ties by ascending `chrom_pos` string, deterministic but not necessarily "
        "identical to a hypothetical re-run of the original code inside a tied tier. This "
        "never changes callset SIZE (panel a, which only depends on `floor(n*quota)`), only "
        "which specific tied variants panels b/c sample at non-shipped quotas.\n"
        "- Panel (b)'s non-shipped quotas are single 500-site deterministic samples (same "
        "engine/seed convention as the existing pilot), not full-callset pileups; standard "
        "error is reported via a Wilson 95% CI per point.\n"
        "- Panel (d) uses only 3 voting features for DLPFC vs 4 for the tumor sections (no CNV "
        "data for normal brain) -- not a like-for-like replica of the tumor cascade, see above.\n"
        "- **Panel (b) is missing the 30% quota point for all four sections.** The four "
        "background pileup jobs (one per section, 5 fresh quotas each) were killed externally "
        "(status: killed, not a crash/exception) after completing quotas 1/2/5/20% but before "
        "reaching 30%; the q30.0 output directories exist but are empty (no `site_features."
        "tsv.gz` was written). This script does NOT substitute or interpolate a 30% value -- "
        "panel (b) and the RESULTS table below report only the 5 quota points (1, 2, 5, 10, 20%) "
        "that were actually pileup-measured. Panels (a), (c), (d), (e) are unaffected and cover "
        "the full 1-30% sweep. To fill this gap, rerun: `python scripts/postanalyze/"
        "fig_somatic_calibration_v5.py --stage pileup --sample <P4|P6|DCIS1|DCIS2>` (idempotent "
        "-- it skips quotas whose `site_features.tsv.gz` already exists) followed by "
        "`--stage merge` and `--stage figure`.\n"
    )
    lines.append("## Exact commands to reproduce\n")
    lines.append("```bash\nsource activate snv_caller\ncd " + PROJECT + "\n"
                 "python scripts/postanalyze/fig_somatic_calibration_v5.py --stage all\n```\n")

    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    log(f"wrote {path}")


def merge_partials(prefix: str):
    """Merge per-sample partial CSVs (panelB_molecular_support_P4.csv, ...) written by
    `--stage pileup --sample X` / `--stage cosmic --sample X` runs into the combined file."""
    frames = []
    for sample in SAMPLES:
        path = os.path.join(DERIVED_DIR, f"{prefix}_{sample}.csv")
        if os.path.isfile(path):
            frames.append(pd.read_csv(path))
        else:
            log(f"  [merge {prefix}] missing partial for {sample}: {path}")
    if not frames:
        log(f"  [merge {prefix}] nothing to merge")
        return
    merged = pd.concat(frames, ignore_index=True)
    out = os.path.join(DERIVED_DIR, f"{prefix}.csv")
    merged.to_csv(out, index=False)
    log(f"  [merge {prefix}] wrote {out} ({len(merged)} rows from {len(frames)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["rank", "pileup", "cosmic", "merge", "figure", "all"], default="all")
    ap.add_argument("--sample", choices=SAMPLES, default=None,
                     help="Restrict pileup/cosmic stage to one sample (for parallel execution); "
                          "writes panel{B,C}_..._<SAMPLE>.csv, combine later with --stage merge.")
    args = ap.parse_args()

    pools = None
    if args.stage in ("rank", "all"):
        pools = stage_rank()
    if args.stage in ("pileup", "all"):
        if pools is None:
            pools = {s: load_pool(SAMPLE_CONFIG[s]["score_path"]) for s in SAMPLES}
        stage_pileup(pools, sample_filter=args.sample)
    if args.stage in ("cosmic", "all"):
        if pools is None:
            pools = {s: load_pool(SAMPLE_CONFIG[s]["score_path"]) for s in SAMPLES}
        stage_cosmic(pools, sample_filter=args.sample)
    if args.stage == "merge":
        merge_partials("panelB_molecular_support")
        merge_partials("panelC_cosmic_hit_rate")
    if args.stage in ("figure", "all"):
        stage_figure()

    if args.stage == "all":
        write_results_md(pools)

    tag = f"{args.stage}_{args.sample}" if args.sample else args.stage
    log_path = os.path.join(DERIVED_DIR, f"_run_log_{tag}.txt")
    with open(log_path, "w") as fh:
        fh.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
