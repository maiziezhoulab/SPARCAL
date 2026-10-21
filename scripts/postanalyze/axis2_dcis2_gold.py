#!/usr/bin/env python
"""Axis 2 — DCIS2 GOLD coverage-controlled test.
tumor_label (pathology, clip window) ~ set_burden + total_UMI.
Does somatic/UPV burden predict tumor BEYOND coverage?"""
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

V   = "/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL"
MAT = "/data/maiziezhou_lab/leiy4/snv_calling/data/dcis2_pre_umidedup_2026-06-25/final_matrices/baseQ0mapQ0"

def logit_irls(X, y):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(100):
        mu=1/(1+np.exp(-(X@b))); W=np.clip(mu*(1-mu),1e-9,None)
        z=X@b+(y-mu)/W; XtW=X.T*W
        bn=np.linalg.solve(XtW@X, XtW@z)
        if np.max(np.abs(bn-b))<1e-8: b=bn; break
        b=bn
    mu=1/(1+np.exp(-(X@b))); W=np.clip(mu*(1-mu),1e-9,None)
    cov=np.linalg.inv((X.T*W)@X); se=np.sqrt(np.diag(cov))
    p=2*(1-norm.cdf(np.abs(b/se)))
    return b,se,p

# ---- positions + clip window (bottom-right 0.6x0.6) ----
pos=pd.read_csv(f"{V}/tissue_positions.csv")
pos=pos[pos.in_tissue==1].copy()
pos["x"]=pos.pxl_col_in_fullres; pos["y"]=pos.pxl_row_in_fullres
x0,x1=pos.x.min(),pos.x.max(); y0,y1=pos.y.min(),pos.y.max()
xm=x0+0.4*(x1-x0); ym=y0+0.4*(y1-y0)
clip=pos[(pos.x>=xm)&(pos.y>=ym)].set_index("barcode")
# ---- GT ----
tg=pd.read_csv(f"{V}/tumor_groups.csv"); gt=set(tg[tg.profile=="Ground Truth"].barcode)
print(f"clip window spots: {len(clip)} | GT tumor total: {len(gt)} | GT in clip: {len(gt & set(clip.index))}")
# ---- coverage ----
cov=pd.read_csv(f"{V}/spot_coverage.csv").set_index("barcode")["total_umi"]
# ---- burdens ----
def burden(f):
    m=pd.read_pickle(f"{MAT}/{f}")
    return (m>0).sum(axis=1)   # per-spot count of distinct SNVs
bs={"somatic":burden("somatic_denovo_spot_snv_matrix.pkl"),
    "upv":     burden("germline_denovo_spot_snv_matrix.pkl"),
    "defined": burden("germline_defined_spot_snv_matrix.pkl")}

# ---- assemble table over clip-window spots ----
bc=[b for b in clip.index if b in cov.index]
df=pd.DataFrame(index=bc)
df["tumor"]=[1 if b in gt else 0 for b in bc]
df["umi"]=cov.reindex(bc).values
for k,v in bs.items(): df[k]=v.reindex(bc).fillna(0).values
df=df.dropna(subset=["umi"])
df["log_umi"]=np.log1p(df.umi)
print(f"analysis spots: {len(df)}  tumor={df.tumor.sum()}  normal={len(df)-df.tumor.sum()}")
print(f"corr(log_umi, log_somatic_burden) = {np.corrcoef(df.log_umi, np.log1p(df.somatic))[0,1]:.3f}\n")

z=lambda s:(s-s.mean())/s.std()
one=np.ones(len(df))
# baseline: coverage only
auc_cov=roc_auc_score(df.tumor, df.umi)
print(f"{'model':34s} {'AUC':>6s}   burden_coef  p(burden|umi)")
print(f"{'coverage_only (UMI)':34s} {auc_cov:6.3f}      --          --")
print("-"*72)
for k in ["somatic","upv","defined"]:
    lb=z(np.log1p(df[k])); lu=z(df.log_umi)
    # burden only
    auc_b=roc_auc_score(df.tumor, df[k])
    # burden + umi
    X=np.column_stack([one, lb.values, lu.values])
    b,se,p=logit_irls(X, df.tumor.values)
    # AUC of the 2-var model
    eta=X@b; auc_full=roc_auc_score(df.tumor, eta)
    print(f"{'tumor~'+k+'_burden (alone)':34s} {auc_b:6.3f}")
    print(f"{'tumor~'+k+'_burden + UMI':34s} {auc_full:6.3f}   {b[1]:+.3f}      {p[1]:.2e}")
    print("-"*72)
