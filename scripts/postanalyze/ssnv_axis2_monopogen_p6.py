#!/usr/bin/env python
"""Add Monopogen P6 burden as a third caller in the P6 pathology coverage-conditional
ladder (extends ssnv_axis2_p6.py / stmut_p6_extension.py to a germline-side caller).
Same conditional logistic test vs the P6 pathology GT (Ji-2020-derived leading-edge mask).
READ-ONLY; merges one new row into the existing p6_axis2_pathology_GT.csv (keeps all
previously computed predictors intact).

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/ssnv_axis2_monopogen_p6.py
"""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

GT  = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/p6_gt/extracted/P6_rep1_SPARCAL_somatic"
MONO = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/monopogen/P6_rep1/P6_rep1_monopogen_presence_matrix.pkl"
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
    cov = np.linalg.inv((X.T * W) @ X); se = np.sqrt(np.diag(cov)); p = 2 * (1 - norm.cdf(np.abs(b / se)))
    return b, se, p


tg = pd.read_csv(f"{GT}/tumor_groups.csv"); gt = set(tg[tg.profile == "GroundTruth"].barcode.astype(str))
pos = pd.read_csv(f"{GT}/tissue_positions.csv"); intissue = [b for b in pos[pos.in_tissue == 1].barcode.astype(str)]
cov = pd.read_csv(f"{GT}/spot_coverage.csv").set_index("barcode")["total_umi"]

M = pd.read_pickle(MONO)
burden = (M > 0).sum(axis=1)
print(f"Monopogen P6 matrix: {M.shape[0]} spots x {M.shape[1]} SNVs")

bc = [b for b in intissue if b in cov.index]
df = pd.DataFrame(index=bc)
df["tumor"] = [1 if b in gt else 0 for b in bc]
df["umi"] = cov.reindex(bc).values
df["Monopogen_burden"] = burden.reindex(bc).fillna(0).values
df = df.dropna(subset=["umi"]); df = df[df.umi > 0]
df["log_umi"] = np.log1p(df.umi)
print(f"P6 analysis spots: {len(df)}  tumor={int(df.tumor.sum())}  normal={len(df) - int(df.tumor.sum())}")

auc_cov = roc_auc_score(df.tumor, df.umi)
z = lambda s: (s - s.mean()) / (s.std() + 1e-9); one = np.ones(len(df))
k = "Monopogen_burden"
r = np.corrcoef(df.log_umi, np.log1p(df[k]))[0, 1]
a = roc_auc_score(df.tumor, df[k]) if df[k].nunique() > 1 else np.nan
X = np.column_stack([one, z(np.log1p(df[k])).values, z(df.log_umi).values])
b, se, p = logit_irls(X, df.tumor.values); j = roc_auc_score(df.tumor, X @ b)
print(f"coverage-only AUC: {auc_cov:.3f}")
print(f"{k:22s} corr(UMI)={r:.3f} alone_auc={a:.3f} joint_auc={j:.3f} coef={b[1]:+.3f} p={p[1]:.2e}")

new_row = {"predictor": k, "corr_UMI": round(r, 3), "alone_auc": round(a, 3),
           "joint_auc": round(j, 3), "coef": round(b[1], 3), "p_given_UMI": p[1]}

csv_path = f"{OUT}/p6_axis2_pathology_GT.csv"
existing = pd.read_csv(csv_path)
existing = existing[existing.predictor != k]
merged = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
merged.to_csv(csv_path, index=False)
print(f"\nWROTE (merged): {csv_path}")
print(merged.to_string(index=False))
