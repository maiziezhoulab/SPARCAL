#!/usr/bin/env python
"""
compute_endpoints.py
=====================
The four non-circular evaluation endpoints (task spec), computed per section
per method_variant (as listed in selected_sets/manifest.csv), and written to
data/spatial_scoring_redesign_2026-08-31/endpoint_comparison.csv.

1. Molecular support -- fraction of a selected set with >=2 DISTINCT ALT-UMI
   families (molecular_support_scan.py; reused logic from
   scripts/sidecar_phasing_feasibility/measure_readbacked_feasibility.py).
2. Depth-dependence -- Spearman rho between selection indicator (over the
   FULL Stage-2 candidate pool, not just the selected set) and per-variant
   read depth (candidates.tsv 'info_dp', the position-level total depth from
   the pooled multi-BAM mpileup -- already computed by the shipped pipeline's
   own step 1, re-used here read-only, no BAM rescan). Also reports the
   selected vs. unselected depth distributions.
3. xMHC enrichment -- fold-enrichment of chrom==6, 28-34Mb among the selected
   set vs. the 0.194% genome-length expectation (task spec). LOWER is better
   (xMHC is a known artifact sink, see data/xmhc_artifact_mechanism_2026-08-31/RESULTS.md).
4. Cross-section reproducibility -- DCIS1 vs DCIS2, P4 vs P6: gene/position
   overlap of selected sets vs. a size-matched random-selection baseline.
   Only computed once >=2 sections of a pair are available; the single-section
   smoke test reports this endpoint as not-yet-computable, honestly.
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import gene_annotation

PAIR_GROUPS = [("DCIS1", "DCIS2"), ("P4", "P6")]
BUILD_OF_SECTION = {s: common.SECTIONS[s]["build"] for s in common.SECTIONS}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def endpoint_molecular_support(section, selected_variants):
    outdir = common.cache_dir(section)
    ms_path = os.path.join(outdir, "molecular_support.tsv")
    if not os.path.exists(ms_path):
        return dict(molecular_support_frac_ge2_families=np.nan, molecular_support_n_with_data=0,
                    molecular_support_n_total=len(selected_variants))
    ms = pd.read_csv(ms_path, sep="\t").set_index("variant")
    sub = ms.reindex(selected_variants)
    n_with_data = sub["n_distinct_alt_umi_families"].notna().sum()
    frac = (sub["n_distinct_alt_umi_families"] >= 2).sum() / len(selected_variants) if len(selected_variants) else np.nan
    return dict(molecular_support_frac_ge2_families=frac, molecular_support_n_with_data=int(n_with_data),
                molecular_support_n_total=len(selected_variants))


def endpoint_depth(candidates_df, selected_variants):
    sel_mask = candidates_df["variant"].isin(selected_variants)
    depth = candidates_df["info_dp"].values
    valid = ~np.isnan(depth)
    sel_indicator = sel_mask.values.astype(int)
    if valid.sum() < 3 or sel_indicator[valid].sum() == 0 or sel_indicator[valid].sum() == valid.sum():
        rho, p = np.nan, np.nan
    else:
        rho, p = stats.spearmanr(sel_indicator[valid], depth[valid])
    dep_sel = depth[sel_mask.values & valid]
    dep_unsel = depth[(~sel_mask.values) & valid]
    return dict(
        depth_spearman_rho=rho, depth_spearman_p=p,
        depth_median_selected=float(np.median(dep_sel)) if len(dep_sel) else np.nan,
        depth_median_unselected=float(np.median(dep_unsel)) if len(dep_unsel) else np.nan,
        depth_iqr_selected=float(np.subtract(*np.percentile(dep_sel, [75, 25]))) if len(dep_sel) else np.nan,
        depth_iqr_unselected=float(np.subtract(*np.percentile(dep_unsel, [75, 25]))) if len(dep_unsel) else np.nan,
    )


def endpoint_xmhc(sel_df):
    n_sel = len(sel_df)
    if n_sel == 0:
        return dict(xmhc_n_in_xmhc=0, xmhc_frac=np.nan, xmhc_fold_enrichment=np.nan)
    in_xmhc = (sel_df["chrom"].astype(str) == common.XMHC_CHROM) & \
              (sel_df["pos"] >= common.XMHC_START) & (sel_df["pos"] <= common.XMHC_END)
    n_in = int(in_xmhc.sum())
    frac = n_in / n_sel
    fold = frac / common.XMHC_FRACTION_OF_GENOME if frac is not None else np.nan
    return dict(xmhc_n_in_xmhc=n_in, xmhc_frac=frac, xmhc_fold_enrichment=fold)


def run_endpoints_for_section(section, manifest):
    outdir = common.cache_dir(section)
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t")
    rows = []
    sec_manifest = manifest[manifest["section"] == section]
    for _, mrow in sec_manifest.iterrows():
        method_variant = mrow["method_variant"]
        sel_df = pd.read_csv(mrow["path"], sep="\t")
        sel_variants = sel_df["variant"].tolist()
        row = dict(section=section, method_variant=method_variant, n_selected=len(sel_variants))
        row.update(endpoint_molecular_support(section, sel_variants))
        row.update(endpoint_depth(candidates, sel_variants))
        row.update(endpoint_xmhc(sel_df))
        rows.append(row)
        log(f"{section}/{method_variant}: n={len(sel_variants)} "
            f"molsupport={row.get('molecular_support_frac_ge2_families')} "
            f"depth_rho={row.get('depth_spearman_rho')} "
            f"xmhc_fold={row.get('xmhc_fold_enrichment')}")
    return pd.DataFrame(rows)


def gene_annotate_section(section):
    """Cache (chrom,pos)->frozenset(gene names) for the FULL candidate pool of one
    section, computed once from the pre-existing GENCODE GTF for that section's
    build (read-only, see gene_annotation.py). Reused by both the observed
    gene-level overlap and its random baseline (avoids re-annotating per-rep)."""
    outdir = common.cache_dir(section)
    cache_path = os.path.join(outdir, "candidates_genes.tsv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, sep="\t", dtype={"chrom": str})
        return {(r.chrom, r.pos): (set(r.genes.split("|")) if isinstance(r.genes, str) and r.genes else set())
                for r in df.itertuples()}
    candidates = pd.read_csv(os.path.join(outdir, "candidates.tsv"), sep="\t", dtype={"chrom": str})
    build = BUILD_OF_SECTION[section]
    log(f"{section}: annotating {len(candidates)} candidate positions against the {build} GENCODE GTF "
        f"(gene_annotation.py, cached to {cache_path}) ...")
    pairs = list(zip(candidates["chrom"].astype(str), candidates["pos"].astype(int)))
    ann = gene_annotation.annotate_positions(build, pairs)
    rows = [dict(chrom=c, pos=p, genes="|".join(sorted(ann.get((c, p), frozenset()))))
            for c, p in pairs]
    out = pd.DataFrame(rows)
    out.to_csv(cache_path, sep="\t", index=False)
    log(f"{section}: gene annotation cached ({(out['genes'] != '').sum()}/{len(out)} candidates hit a gene body)")
    return {(r.chrom, r.pos): (set(r.genes.split("|")) if isinstance(r.genes, str) and r.genes else set())
            for r in out.itertuples()}


def genes_for(sel_df, gene_map):
    genes = set()
    for c, p in zip(sel_df["chrom"].astype(str), sel_df["pos"].astype(int)):
        genes |= gene_map.get((c, p), set())
    return genes


def random_baseline_gene_overlap(cand_a, gene_map_a, cand_b, gene_map_b, size_a, size_b, n_reps=200, seed=0):
    rng = np.random.default_rng(seed)
    idx_a = cand_a.index.values
    idx_b = cand_b.index.values
    overlaps = []
    for _ in range(n_reps):
        draw_a = cand_a.loc[rng.choice(idx_a, size=min(size_a, len(idx_a)), replace=False)]
        draw_b = cand_b.loc[rng.choice(idx_b, size=min(size_b, len(idx_b)), replace=False)]
        ga = genes_for(draw_a, gene_map_a)
        gb = genes_for(draw_b, gene_map_b)
        overlaps.append(len(ga & gb))
    return float(np.mean(overlaps)), float(np.std(overlaps))


def random_baseline_overlap(pool_a, pool_b, size_a, size_b, n_reps=200, seed=0):
    rng = np.random.default_rng(seed)
    pool_a = list(pool_a)
    pool_b = list(pool_b)
    overlaps = []
    for _ in range(n_reps):
        sa = set(rng.choice(pool_a, size=min(size_a, len(pool_a)), replace=False))
        sb = set(rng.choice(pool_b, size=min(size_b, len(pool_b)), replace=False))
        overlaps.append(len(sa & sb))
    return float(np.mean(overlaps)), float(np.std(overlaps))


def endpoint_cross_section(manifest):
    rows = []
    for sec_a, sec_b in PAIR_GROUPS:
        have_a = (manifest["section"] == sec_a).any()
        have_b = (manifest["section"] == sec_b).any()
        if not (have_a and have_b):
            rows.append(dict(pair=f"{sec_a}_vs_{sec_b}", method_variant="ALL", status="NOT_COMPUTABLE",
                              reason=f"section(s) missing from this run: "
                                     f"{sec_a if not have_a else ''} {sec_b if not have_b else ''}".strip()))
            continue
        common_methods = set(manifest[manifest["section"] == sec_a]["method_variant"]) & \
                          set(manifest[manifest["section"] == sec_b]["method_variant"])
        cand_a = pd.read_csv(os.path.join(common.cache_dir(sec_a), "candidates.tsv"), sep="\t", dtype={"chrom": str})
        cand_b = pd.read_csv(os.path.join(common.cache_dir(sec_b), "candidates.tsv"), sep="\t", dtype={"chrom": str})
        gene_map_a = gene_annotate_section(sec_a)
        gene_map_b = gene_annotate_section(sec_b)
        for mv in sorted(common_methods):
            path_a = manifest[(manifest["section"] == sec_a) & (manifest["method_variant"] == mv)]["path"].iloc[0]
            path_b = manifest[(manifest["section"] == sec_b) & (manifest["method_variant"] == mv)]["path"].iloc[0]
            sel_a = pd.read_csv(path_a, sep="\t", dtype={"chrom": str})
            sel_b = pd.read_csv(path_b, sep="\t", dtype={"chrom": str})
            pos_a = set(zip(sel_a["chrom"].astype(str), sel_a["pos"].astype(int)))
            pos_b = set(zip(sel_b["chrom"].astype(str), sel_b["pos"].astype(int)))
            pos_overlap = len(pos_a & pos_b)
            mean_rand, std_rand = random_baseline_overlap(
                cand_a["chrom"].astype(str) + "_" + cand_a["pos"].astype(str),
                cand_b["chrom"].astype(str) + "_" + cand_b["pos"].astype(str),
                len(sel_a), len(sel_b))

            genes_a = genes_for(sel_a, gene_map_a)
            genes_b = genes_for(sel_b, gene_map_b)
            gene_overlap = len(genes_a & genes_b)
            gene_mean_rand, gene_std_rand = random_baseline_gene_overlap(
                cand_a, gene_map_a, cand_b, gene_map_b, len(sel_a), len(sel_b))

            rows.append(dict(pair=f"{sec_a}_vs_{sec_b}", method_variant=mv, status="OK",
                              n_selected_a=len(sel_a), n_selected_b=len(sel_b),
                              position_overlap=pos_overlap,
                              random_baseline_overlap_mean=mean_rand,
                              random_baseline_overlap_std=std_rand,
                              n_genes_a=len(genes_a), n_genes_b=len(genes_b),
                              gene_overlap=gene_overlap,
                              random_baseline_gene_overlap_mean=gene_mean_rand,
                              random_baseline_gene_overlap_std=gene_std_rand))
            log(f"{sec_a}_vs_{sec_b}/{mv}: pos_overlap={pos_overlap} (rand {mean_rand:.1f}±{std_rand:.1f}) "
                f"gene_overlap={gene_overlap} (rand {gene_mean_rand:.1f}±{gene_std_rand:.1f})")
    return pd.DataFrame(rows)


def qvalue_distribution(section):
    """Full q-value distribution for Method A (coordinator request 2026-09-03,
    point 2): never present p<1 as if it were a significance threshold -- show
    the whole picture, including how many pass at the FDR thresholds the task
    actually specifies (expected: zero)."""
    outdir = common.cache_dir(section)
    a = pd.read_csv(os.path.join(outdir, f"methodA_{section}_scores.tsv"), sep="\t")
    q = a["q_joincount"].values
    p = a["p_joincount"].values
    row = dict(section=section, n_candidates=len(a), n_testable=int((a["n_pos_spots"] >= 2).sum()))
    for thresh in (0.05, 0.10, 0.20, 0.50, 0.80):
        row[f"n_q_lt_{thresh}"] = int((q < thresh).sum())
    row["n_q_eq_1"] = int((q >= 1.0).sum())
    row["n_p_lt_1"] = int((p < 1.0).sum())
    row["n_p_eq_1"] = int((p >= 1.0).sum())
    for pct in (50, 90, 99, 99.9):
        row[f"p_percentile_{pct}"] = float(np.percentile(p, pct))
    return row


def main(sections):
    sel_dir = os.path.join(common.OUT_ROOT, "selected_sets")
    manifest = pd.read_csv(os.path.join(sel_dir, "manifest.csv"))

    all_rows = []
    for section in sections:
        df = run_endpoints_for_section(section, manifest)
        all_rows.append(df)
    endpoint_df = pd.concat(all_rows, ignore_index=True)

    out_path = os.path.join(common.OUT_ROOT, "endpoint_comparison.csv")
    endpoint_df.to_csv(out_path, index=False)
    log(f"wrote {out_path} ({len(endpoint_df)} rows)")

    qdist_rows = [qvalue_distribution(s) for s in sections]
    qdist_df = pd.DataFrame(qdist_rows)
    qdist_path = os.path.join(common.OUT_ROOT, "qvalue_distribution.csv")
    qdist_df.to_csv(qdist_path, index=False)
    log(f"wrote {qdist_path} ({len(qdist_df)} rows)")

    cross_df = endpoint_cross_section(manifest)
    cross_path = os.path.join(common.OUT_ROOT, "endpoint4_cross_section.csv")
    cross_df.to_csv(cross_path, index=False)
    log(f"wrote {cross_path} ({len(cross_df)} rows)")
    return endpoint_df, cross_df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", nargs="+", required=True, choices=list(common.SECTIONS.keys()))
    args = ap.parse_args()
    main(args.sections)
