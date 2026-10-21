#!/usr/bin/env python
"""3'-bias and expressed-gene characterization of P6 spatial-SNV calls (SPARCAL somatic,
SpatialSNV). Deepens the platform-limit story: 10x Visium 3' chemistry should concentrate
calls near the 3' end of the gene body and in highly expressed genes -- a second,
independent signature of the same coverage confound (not just per-spot burden).

Method (gene-body-level, not exon-spliced -- documented caveat): each SNV is assigned to
the overlapping Ensembl gene span (GTF used by the actual P6 SpaceRanger alignment,
refdata-hg19-2.1.0/cellranger_hg19/cellranger_hg19/genes/genes.gtf) via a sweep-line
interval join (smallest-span gene wins on overlap). Relative position in [0,1] is defined
as distance from the gene's TSS, oriented by strand so 1.0 = the transcript's 3' end. A
gene-length-matched null (uniform random position within the SAME assigned gene, one per
call) is generated for comparison -- this controls for the gene-length distribution so
"all near 1.0" cannot be an artifact of long-gene bias alone.

Expression-bias part: per-gene total UMI (P6 SpaceRanger filtered matrix, summed across
all spots) vs per-gene SNV-call count -> Spearman correlation, both call sets.

READ-ONLY. Run (env snv_caller, NOT sbatch): python scripts/postanalyze/ssnv_3prime_bias_p6.py
Outputs: p6_3prime_bias_summary.csv, p6_3prime_bias_relpos.csv (per-call relpos + null)
"""
import gzip
import heapq
import random
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import spearmanr, ks_2samp

GTF = "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/cellranger_hg19/cellranger_hg19/genes/genes.gtf"
GT_BUNDLE = ("/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/"
             "p6_gt/extracted/P6_rep1_SPARCAL_somatic/snv_matrix.pkl")
SSNV_FEAT = "/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/results/p6/matrix/p6_alt/features.tsv.gz"
SRB = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/"
       "spaceranger_align_rep1_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix")
OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"
random.seed(0)


def strip_chr(c):
    return c[3:] if c.startswith("chr") else c


def load_genes():
    """feature=='gene' lines -> {chrom: [(start,end,strand,gene_name), ...] sorted by start}"""
    by_chrom = {}
    with open(GTF) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom = strip_chr(f[0])
            start, end, strand = int(f[3]), int(f[4]), f[6]
            attrs = f[8]
            gi = attrs.find('gene_name "')
            gname = attrs[gi + 11:attrs.find('"', gi + 11)] if gi >= 0 else "NA"
            by_chrom.setdefault(chrom, []).append((start, end, strand, gname))
    for c in by_chrom:
        by_chrom[c].sort(key=lambda t: t[0])
    return by_chrom


def assign_genes(genes_by_chrom, positions_by_chrom):
    """Sweep-line: for each chrom, walk sorted query positions, maintain a min-heap of
    active genes (by end) whose start<=pos; among active genes with end>=pos pick the
    smallest span. Returns {(chrom,pos): (start,end,strand,gene_name) or None}."""
    result = {}
    for chrom, qpos in positions_by_chrom.items():
        genes = genes_by_chrom.get(chrom, [])
        if not genes:
            for p in qpos:
                result[(chrom, p)] = None
            continue
        qpos_sorted = sorted(set(qpos))
        gi = 0
        active = []  # heap of (end, start, strand, name)
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
                best = min(cands, key=lambda g: g[0] - g[1])  # smallest span
                e, s, strand, name = best
                result[(chrom, p)] = (s, e, strand, name)
    return result


def relpos(pos, start, end, strand):
    length = max(end - start, 1)
    return (pos - start) / length if strand == "+" else (end - pos) / length


def load_sparcal_positions():
    M = pd.read_pickle(GT_BUNDLE)
    out = []
    for col in M.columns:
        chrom, pos = col.split("_", 1)
        out.append((strip_chr(chrom), int(pos)))
    return out


def load_spatialsnv_positions():
    out = []
    with gzip.open(SSNV_FEAT, "rt") as fh:
        for line in fh:
            key = line.split("\t", 1)[0]
            chrom_pos, _allele = key.split(":", 1)
            chrom, pos = chrom_pos.rsplit("_", 1)
            out.append((strip_chr(chrom), int(pos)))
    return out


def positions_to_relpos(name, positions, genes_by_chrom):
    by_chrom = {}
    for c, p in positions:
        by_chrom.setdefault(c, []).append(p)
    assigned = assign_genes(genes_by_chrom, by_chrom)
    rows = []
    n_unassigned = 0
    for c, p in positions:
        g = assigned.get((c, p))
        if g is None:
            n_unassigned += 1
            continue
        s, e, strand, gname = g
        rp = relpos(p, s, e, strand)
        null_p = random.uniform(s, e)
        null_rp = relpos(null_p, s, e, strand)
        rows.append({"chrom": c, "pos": p, "gene": gname, "strand": strand,
                      "gene_len": e - s, "relpos": rp, "null_relpos": null_rp})
    df = pd.DataFrame(rows)
    print(f"[{name}] {len(positions):,} positions -> {len(df):,} genic ({100*len(df)/len(positions):.1f}%), "
          f"{n_unassigned:,} intergenic/unassigned")
    return df


def gene_expression():
    with gzip.open(f"{SRB}/features.tsv.gz", "rt") as fh:
        feats = [l.rstrip("\n").split("\t") for l in fh]
    names = [f[1] for f in feats]
    with gzip.open(f"{SRB}/matrix.mtx.gz", "rb") as fh:
        G = sp.csr_matrix(scipy.io.mmread(fh))  # genes x spots
    total = np.asarray(G.sum(axis=1)).ravel()
    return pd.Series(total, index=names).groupby(level=0).sum()  # dedupe repeated symbols


if __name__ == "__main__":
    print("Loading gene models...")
    genes_by_chrom = load_genes()
    n_genes = sum(len(v) for v in genes_by_chrom.values())
    print(f"{n_genes:,} genes across {len(genes_by_chrom)} chromosomes\n")

    sparcal_pos = load_sparcal_positions()
    ssnv_pos = load_spatialsnv_positions()

    df_sparcal = positions_to_relpos("SPARCAL_somatic", sparcal_pos, genes_by_chrom)
    df_ssnv = positions_to_relpos("SpatialSNV", ssnv_pos, genes_by_chrom)

    summary_rows = []
    relpos_frames = []
    for name, df in [("SPARCAL_somatic", df_sparcal), ("SpatialSNV", df_ssnv)]:
        ks = ks_2samp(df.relpos, df.null_relpos)
        row = {"set": name, "n_genic": len(df),
               "mean_relpos": round(df.relpos.mean(), 4),
               "median_relpos": round(df.relpos.median(), 4),
               "mean_null_relpos": round(df.null_relpos.mean(), 4),
               "pct_in_top_decile_3prime": round(100 * (df.relpos >= 0.9).mean(), 2),
               "pct_null_in_top_decile_3prime": round(100 * (df.null_relpos >= 0.9).mean(), 2),
               "ks_stat": round(ks.statistic, 4), "ks_p": ks.pvalue}
        summary_rows.append(row)
        print(f"[{name}] mean relpos (1.0=3' end) = {row['mean_relpos']:.3f} "
              f"vs gene-length-matched null {row['mean_null_relpos']:.3f}  "
              f"(KS stat={row['ks_stat']:.3f}, p={ks.pvalue:.2e})")
        print(f"         top-decile-near-3'-end: calls {row['pct_in_top_decile_3prime']:.1f}% "
              f"vs null {row['pct_null_in_top_decile_3prime']:.1f}%")
        df2 = df.copy(); df2["set"] = name
        relpos_frames.append(df2)

    print("\nExpression-bias:")
    expr = gene_expression()
    for name, df in [("SPARCAL_somatic", df_sparcal), ("SpatialSNV", df_ssnv)]:
        gcount = df.groupby("gene").size()
        joined = pd.DataFrame({"n_calls": gcount}).join(expr.rename("total_umi"), how="left").fillna(0)
        rho, p = spearmanr(joined.n_calls, joined.total_umi)
        all_expr_median = expr.median()
        called_gene_expr_median = expr.reindex(gcount.index).dropna().median()
        row = summary_rows[0 if name == "SPARCAL_somatic" else 1]
        row["n_genes_hit"] = len(gcount)
        row["spearman_rho_calls_vs_expr"] = round(rho, 4)
        row["spearman_p"] = p
        row["median_expr_all_genes"] = round(all_expr_median, 1)
        row["median_expr_called_genes"] = round(called_gene_expr_median, 1)
        print(f"[{name}] {len(gcount):,} genes hit; Spearman(#calls/gene, total UMI/gene) = "
              f"{rho:.3f} (p={p:.2e}); median expr all genes={all_expr_median:.0f} vs "
              f"called genes={called_gene_expr_median:.0f}")

    pd.DataFrame(summary_rows).to_csv(f"{OUT}/p6_3prime_bias_summary.csv", index=False)
    pd.concat(relpos_frames, ignore_index=True).to_csv(f"{OUT}/p6_3prime_bias_relpos.csv", index=False)
    print(f"\nWROTE: {OUT}/p6_3prime_bias_summary.csv")
    print(f"WROTE: {OUT}/p6_3prime_bias_relpos.csv")
