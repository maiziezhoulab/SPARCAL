#!/usr/bin/env python
"""Fig 3 data prep: re-fit the coverage-conditional logistic for the Fig-1-selected
predictors on P6 and DCIS2, this time SAVING the standard error (not just the
rounded p-value) so a forest plot can draw a real coef +/- 1.96*SE interval. The
earlier per-sample scripts (ssnv_axis2_p6.py, ssnv_axis2_monopogen_p6.py,
ssnv_dcis2_poscontrol.py) only persisted coef+p, and several p-values underflow to
exactly 0.0 at this N x collinearity (float64 can't represent p that small) -- SE
does not have that problem, so we recompute here instead of trying to invert p.

STMut_burden (P6, WES-gold sites) is the one exception: its SE would require
re-scanning the BAM (the slow part of stmut_p6_extension.py) and is not worth
re-running for a supplementary CI. Its point estimate (coef=-0.757) is reused with
a conservative capped SE (z-floor=20, i.e. treating it as "only" p~2.8e-89 rather
than the true underflowed value) -- this makes its displayed CI WIDER than the
true one, which is the safe direction (not overclaiming precision). Flagged in the
output table via `se_is_capped=True`.

READ-ONLY. Run (env snv_caller, NOT sbatch): python scripts/postanalyze/fig3_forest_data.py
Output: data/spatialsnv_reanalysis_2026-07-17/fig3_forest_data.csv
"""
import gzip
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import norm

OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"


def logit_irls(X, y):
    X = np.asarray(X, float); y = np.asarray(y, float); b = np.zeros(X.shape[1])
    for _ in range(200):
        mu = 1 / (1 + np.exp(-(X @ b))); W = np.clip(mu * (1 - mu), 1e-9, None)
        z = X @ b + (y - mu) / W; XtW = X.T * W
        try:
            bn = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(bn - b)) < 1e-8:
            b = bn; break
        b = bn
    mu = 1 / (1 + np.exp(-(X @ b))); W = np.clip(mu * (1 - mu), 1e-9, None)
    se = np.sqrt(np.diag(np.linalg.inv((X.T * W) @ X)))
    return b, se


def fit_one(df, k):
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
    one = np.ones(len(df))
    X = np.column_stack([one, z(np.log1p(df[k])).values, z(df.log_umi).values])
    b, se = logit_irls(X, df.tumor.values)
    return b[1], se[1]


rows = []

# ---------------- P6 ----------------
GT = ("/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/"
      "p6_gt/extracted/P6_rep1_SPARCAL_somatic")
SSNV_P6 = "/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/results/p6/matrix/p6_alt"
SRB_P6 = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/"
          "spaceranger_align_rep1_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix")
MONO_P6 = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/monopogen/P6_rep1/P6_rep1_monopogen_presence_matrix.pkl"
TSK_MARKERS = ["MMP10", "MMP1", "KRT6A", "KRT16", "INHBA", "PTHLH"]

tg = pd.read_csv(f"{GT}/tumor_groups.csv"); gt = set(tg[tg.profile == "GroundTruth"].barcode.astype(str))
pos = pd.read_csv(f"{GT}/tissue_positions.csv"); intissue = [b for b in pos[pos.in_tissue == 1].barcode.astype(str)]
cov = pd.read_csv(f"{GT}/spot_coverage.csv").set_index("barcode")["total_umi"]

burdens = {}
M = pd.read_pickle(f"{GT}/snv_matrix.pkl")
burdens["SPARCAL_somatic"] = (M > 0).sum(axis=1)
with gzip.open(f"{SSNV_P6}/matrix.mtx.gz", "rb") as fh:
    A = sp.csr_matrix(scipy.io.mmread(fh))
with gzip.open(f"{SSNV_P6}/barcodes.tsv.gz", "rt") as fh:
    sbc = np.array([x.strip() for x in fh])
spp = np.asarray((A >= 1).sum(axis=1)).ravel()
burdens["SpatialSNV_effective"] = pd.Series(np.asarray((A[spp >= 2, :] >= 1).sum(axis=0)).ravel(), index=sbc)
Mono = pd.read_pickle(MONO_P6)
burdens["Monopogen_burden"] = (Mono > 0).sum(axis=1)

with gzip.open(f"{SRB_P6}/features.tsv.gz", "rt") as fh:
    feats = [l.rstrip("\n").split("\t") for l in fh]
name2row = {f[1]: i for i, f in enumerate(feats) if len(f) > 1}
with gzip.open(f"{SRB_P6}/barcodes.tsv.gz", "rt") as fh:
    gbc = np.array([x.strip() for x in fh])
with gzip.open(f"{SRB_P6}/matrix.mtx.gz", "rb") as fh:
    G = sp.csr_matrix(scipy.io.mmread(fh))
gene_series = lambda names: pd.Series(np.asarray(G[[name2row[n] for n in names if n in name2row], :].sum(axis=0)).ravel(), index=gbc)
burdens["TSK_sig_expr"] = gene_series(TSK_MARKERS)

bc = [b for b in intissue if b in cov.index]
df6 = pd.DataFrame(index=bc)
df6["tumor"] = [1 if b in gt else 0 for b in bc]
df6["umi"] = cov.reindex(bc).values
for k, v in burdens.items():
    df6[k] = v.reindex(bc).fillna(0).values
df6 = df6.dropna(subset=["umi"]); df6 = df6[df6.umi > 0]
df6["log_umi"] = np.log1p(df6.umi)
print(f"P6: {len(df6)} spots")

for k, kind in [("SPARCAL_somatic", "SNV caller"), ("SpatialSNV_effective", "SNV caller"),
                ("Monopogen_burden", "SNV caller"), ("TSK_sig_expr", "expression (positive control)")]:
    coef, se = fit_one(df6, k)
    rows.append({"sample": "P6", "predictor": k, "kind": kind, "coef": coef, "se": se, "se_is_capped": False})

# STMut: reuse reported coef, cap SE (see module docstring)
stmut_coef = -0.757
z_floor = 20.0
rows.append({"sample": "P6", "predictor": "STMut_burden", "kind": "SNV caller (gold WES sites)",
             "coef": stmut_coef, "se": abs(stmut_coef) / z_floor, "se_is_capped": True})

# ---------------- DCIS2 ----------------
V = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL"
MAT = "/data/maiziezhou_lab/leiy4/snv_calling/data/dcis2_pre_umidedup_2026-06-25/final_matrices/baseQ0mapQ0"
SSNV_D2 = "/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/results/dcis2/matrix/dcis2_alt"
SRB_D2 = ("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS2/"
          "spaceranger_align_DCIS2_hg38/DCIS2_output/outs/filtered_feature_bc_matrix")
EPI = ["EPCAM", "KRT8", "KRT18", "KRT19", "KRT7"]

pos2 = pd.read_csv(f"{V}/tissue_positions.csv"); pos2 = pos2[pos2.in_tissue == 1].copy()
pos2["x"] = pos2.pxl_col_in_fullres; pos2["y"] = pos2.pxl_row_in_fullres
x0, x1 = pos2.x.min(), pos2.x.max(); y0, y1 = pos2.y.min(), pos2.y.max()
clip = pos2[(pos2.x >= x0 + 0.4 * (x1 - x0)) & (pos2.y >= y0 + 0.4 * (y1 - y0))].set_index("barcode")
tg2 = pd.read_csv(f"{V}/tumor_groups.csv"); gt2 = set(tg2[tg2.profile.isin(["Ground Truth", "GroundTruth"])].barcode)
cov2 = pd.read_csv(f"{V}/spot_coverage.csv").set_index("barcode")["total_umi"]

pred2 = {}
pred2["SPARCAL_somatic"] = (pd.read_pickle(f"{MAT}/somatic_denovo_spot_snv_matrix.pkl") > 0).sum(axis=1)
with gzip.open(f"{SSNV_D2}/matrix.mtx.gz", "rb") as fh:
    A2 = sp.csr_matrix(scipy.io.mmread(fh))
with gzip.open(f"{SSNV_D2}/barcodes.tsv.gz", "rt") as fh:
    sbc2 = np.array([x.strip() for x in fh])
spp2 = np.asarray((A2 >= 1).sum(axis=1)).ravel()
pred2["SpatialSNV_effective"] = pd.Series(np.asarray((A2[spp2 >= 2, :] >= 1).sum(axis=0)).ravel(), index=sbc2)

with gzip.open(f"{SRB_D2}/features.tsv.gz", "rt") as fh:
    feats2 = [l.rstrip().split("\t") for l in fh]
n2r2 = {f[1]: i for i, f in enumerate(feats2) if len(f) > 1}
with gzip.open(f"{SRB_D2}/barcodes.tsv.gz", "rt") as fh:
    gbc2 = np.array([x.strip() for x in fh])
with gzip.open(f"{SRB_D2}/matrix.mtx.gz", "rb") as fh:
    G2 = sp.csr_matrix(scipy.io.mmread(fh))
gs2 = lambda names: pd.Series(np.asarray(G2[[n2r2[n] for n in names if n in n2r2], :].sum(axis=0)).ravel(), index=gbc2)
pred2["epithelial_sig_expr"] = gs2(EPI)

bc2 = [b for b in clip.index if b in cov2.index]
df2 = pd.DataFrame(index=bc2)
df2["tumor"] = [1 if b in gt2 else 0 for b in bc2]
df2["umi"] = cov2.reindex(bc2).values
for k, v in pred2.items():
    df2[k] = v.reindex(bc2).fillna(0).values
df2 = df2.dropna(subset=["umi"]); df2 = df2[df2.umi > 0]
df2["log_umi"] = np.log1p(df2.umi)
print(f"DCIS2: {len(df2)} spots")

for k, kind in [("SPARCAL_somatic", "SNV caller"), ("SpatialSNV_effective", "SNV caller"),
                ("epithelial_sig_expr", "expression (positive control)")]:
    coef, se = fit_one(df2, k)
    rows.append({"sample": "DCIS2", "predictor": k, "kind": kind, "coef": coef, "se": se, "se_is_capped": False})

out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/fig3_forest_data.csv", index=False)
print(out.to_string(index=False))
print(f"\nWROTE: {OUT}/fig3_forest_data.csv")
