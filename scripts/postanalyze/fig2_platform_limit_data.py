#!/usr/bin/env python
"""Fig 2 data prep: per-gene expression table for P6 (needed for panel d, gene-level
expression bias). Reuses the same gene-expression load as ssnv_3prime_bias_p6.py but
saves a standalone per-gene CSV (total_umi, called-by-SPARCAL, called-by-SpatialSNV) so
the figure script does not need to recompute the interval-sweep gene assignment.

READ-ONLY. Run (env snv_caller, NOT sbatch): python scripts/postanalyze/fig2_platform_limit_data.py
Output: data/spatialsnv_reanalysis_2026-07-17/p6_gene_expression_bias.csv
"""
import gzip
import pandas as pd
import scipy.io
import scipy.sparse as sp

# reuse the exact gene-assignment logic from the 3'-bias script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "ssnv_3prime_bias_p6",
    "/data/maiziezhou_lab/leiy4/snv_calling/scripts/postanalyze/ssnv_3prime_bias_p6.py")
m3p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m3p)

SRB = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/"
       "spaceranger_align_rep1_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix")
OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"

if __name__ == "__main__":
    expr = m3p.gene_expression()  # per-gene total UMI, symbol-deduped
    genes_by_chrom = m3p.load_genes()

    sparcal_pos = m3p.load_sparcal_positions()
    ssnv_pos = m3p.load_spatialsnv_positions()
    df_sparcal = m3p.positions_to_relpos("SPARCAL_somatic", sparcal_pos, genes_by_chrom)
    df_ssnv = m3p.positions_to_relpos("SpatialSNV", ssnv_pos, genes_by_chrom)

    sparcal_genes = set(df_sparcal.gene)
    ssnv_genes = set(df_ssnv.gene)

    out = pd.DataFrame({"gene": expr.index, "total_umi": expr.values})
    out["hit_by_SPARCAL"] = out.gene.isin(sparcal_genes)
    out["hit_by_SpatialSNV"] = out.gene.isin(ssnv_genes)
    out.to_csv(f"{OUT}/p6_gene_expression_bias.csv", index=False)
    print(f"n genes total: {len(out)}")
    print(f"n genes hit by SPARCAL: {out.hit_by_SPARCAL.sum()}  by SpatialSNV: {out.hit_by_SpatialSNV.sum()}")
    print(f"WROTE: {OUT}/p6_gene_expression_bias.csv")
