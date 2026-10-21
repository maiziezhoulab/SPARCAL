#!/usr/bin/env python
"""Axis 2 — coverage-controlled spatial concordance on P4/P6/DCIS1 (+DCIS2 silver).

The decisive test (On_going.md 2026-07-13): does SPARCAL's *somatic* (or UPV) per-spot
burden predict the CNV-defined tumor region BEYOND total UMI (coverage)?  DCIS2 with
pathology GOLD already said NO (somatic p=0.77, only germline/1kG survives). This extends
the SAME coverage-controlled logistic to the high-TMB cSCC cases (P4/P6) and DCIS1, using a
CNV-tool "silver" tumor label.

Silver GT = copyKAT (aneuploid=tumor / diploid=normal).  Calibrated against DCIS2 pathology
(230/249 recall = 0.924; precision 0.627; F1 0.747, clip window).  NOTE (verified 2026-07-13):
inferCNV's HMM cell_groupings and siCNV's annotations are **100% identical to copyKAT** on all
4 samples (Weiman's pipeline seeds inferCNV/siCNV with copyKAT's aneuploid/diploid calls), and
numbat/clonalscope have no parseable per-spot tumor label -> the "5-tool consensus" reduces to
the copyKAT anchor.  So the frozen consensus rule IS copyKAT.

Burden source = step-7 per-spot lists (spatial_filter_purity/baseQ0mapQ0/{germline,somatic}/
{barcode}.txt; germline race col = defined[1kG]/denovo[UPV]).  Coverage = total UMI per spot
from the SpaceRanger filtered_feature_bc_matrix.  Identical model to axis2_dcis2_gold.py:
tumor ~ z(log burden) + z(log UMI).
"""
import os, glob, gzip, re
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.io import mmread
from sklearn.metrics import roc_auc_score

PROJ = "/data/maiziezhou_lab/leiy4/snv_calling"
CK   = "/data/maiziezhou_lab/Weiman/ST_CNV/copyKAT"
IC   = "/data/maiziezhou_lab/Weiman/ST_CNV/inferCNV"
SI   = "/data/maiziezhou_lab/Weiman/ST_CNV/siCNV"
QF   = "baseQ0mapQ0"
OUTDIR = f"{PROJ}/data/somatic_validation_2026-07-13/data"

# sample -> (copyKAT dir, {burden_version: burden dir}, spaceranger outs dir)
# Two burden conventions:
#   pre_dedup = the May-18 run used by the established DCIS2-gold anchor (the whole tumor
#               benchmark is "pre-UMI-dedup" per On_going.md); PRIMARY for consistency.
#   current   = the Jun/Jul post-dedup "1" dirs (larger sets: dedup + code-drift inflation).
SP = "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets"
SAMPLES = {
    "P4":   ("P4_rep1_hg38",
             {"pre_dedup": f"{PROJ}/data/P4_tumor/1_pre_umidedup_2026-06-25",
              "current":   f"{PROJ}/data/P4_tumor/1"},
             f"{SP}/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs"),
    "P6":   ("P6_rep1_hg38",
             {"pre_dedup": f"{PROJ}/data/P6_tumor/1_pre_umidedup_2026-06-24",
              "current":   f"{PROJ}/data/P6_tumor/1"},
             f"{SP}/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs"),
    "DCIS1":("DCIS_1",
             {"pre_dedup": f"{PROJ}/data/dcis1_pre_umidedup_2026-06-25",
              "current":   f"{PROJ}/data/dcis1"},
             f"{SP}/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs"),
    "DCIS2":("DCIS_2",
             {"pre_dedup": f"{PROJ}/data/dcis2_pre_umidedup_2026-06-25",
              "current":   f"{PROJ}/data/dcis2"},
             f"{SP}/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs"),
}
BARCODE_RE = re.compile(r"[ACGT]+-\d+$")   # excludes stray aggregate files (somatic_variants.txt, ...)

# ---------------------------------------------------------------- loaders
def load_copykat(s):
    d={}
    with open(f"{CK}/{s}/prediction_aligned.txt") as f:
        next(f)
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] in ("aneuploid","diploid"):
                d[p[0]] = 1 if p[1]=="aneuploid" else 0
    return d

def load_infercnv(s):
    d={}
    with open(f"{IC}/{s}/infercnv_annotations.txt") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)>=2 and p[1] in ("tumor","normal"):
                d[p[0]] = 1 if p[1]=="tumor" else 0
    return d

def load_sicnv(s):
    fn=[x for x in os.listdir(f"{SI}/{s}") if x.endswith("_copykat_annotations.tsv")]
    d={}
    if not fn: return d
    with open(f"{SI}/{s}/{fn[0]}") as f:
        for ln in f:
            p=ln.rstrip("\n").split("\t")
            if len(p)<2: continue
            m=re.search(r"([ACGT]{16})\.1$", p[0])
            if m and p[1] in ("aneuploid","diploid"):
                d[m.group(1)+"-1"] = 1 if p[1]=="aneuploid" else 0
    return d

def spot_burden(burden_dir):
    """Per-spot distinct-SNV counts: somatic, upv(=germline denovo), defined(=germline 1kG)."""
    gdir=f"{burden_dir}/spatial_filter_purity/{QF}/germline"
    sdir=f"{burden_dir}/spatial_filter_purity/{QF}/somatic"
    bcs=set()
    for d in (gdir,sdir):
        if os.path.isdir(d):
            bcs |= {os.path.basename(x)[:-4] for x in glob.glob(f"{d}/*.txt")
                    if BARCODE_RE.fullmatch(os.path.basename(x)[:-4])}
    rows={}
    for bc in bcs:
        defi=upv=som=0
        gf=f"{gdir}/{bc}.txt"
        if os.path.isfile(gf):
            with open(gf) as f:
                next(f,None)
                for ln in f:
                    r=ln.rstrip("\n").rsplit("\t",1)
                    if len(r)==2:
                        if r[1]=="defined": defi+=1
                        elif r[1]=="denovo": upv+=1
        sf=f"{sdir}/{bc}.txt"
        if os.path.isfile(sf):
            with open(sf) as f:
                next(f,None)
                som=sum(1 for _ in f)
        rows[bc]=(defi,upv,som)
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["defined","upv","somatic"])

def total_umi(outs):
    mdir=f"{outs}/filtered_feature_bc_matrix"
    with gzip.open(f"{mdir}/barcodes.tsv.gz","rt") as f:
        bcs=[ln.strip() for ln in f if ln.strip()]
    m=mmread(f"{mdir}/matrix.mtx.gz").tocsc()      # genes x spots
    umi=np.asarray(m.sum(axis=0)).ravel()
    return pd.Series(umi, index=bcs, dtype=np.int64)

# ---------------------------------------------------------------- stats
def logit_irls(X, y):
    X=np.asarray(X,float); y=np.asarray(y,float); b=np.zeros(X.shape[1])
    for _ in range(200):
        mu=1/(1+np.exp(-(X@b))); W=np.clip(mu*(1-mu),1e-9,None)
        z=X@b+(y-mu)/W; XtW=X.T*W
        bn=np.linalg.solve(XtW@X, XtW@z)
        if np.max(np.abs(bn-b))<1e-9: b=bn; break
        b=bn
    mu=1/(1+np.exp(-(X@b))); W=np.clip(mu*(1-mu),1e-9,None)
    cov=np.linalg.inv((X.T*W)@X); se=np.sqrt(np.diag(cov))
    p=2*(1-norm.cdf(np.abs(b/se)))
    return b,se,p

z=lambda s:(s-s.mean())/s.std()

# ---------------------------------------------------------------- calibration (DCIS2 pathology)
def calibrate_dcis2():
    V="/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL"
    pos=pd.read_csv(f"{V}/tissue_positions.csv"); pos=pos[pos.in_tissue==1].copy()
    pos["x"]=pos.pxl_col_in_fullres; pos["y"]=pos.pxl_row_in_fullres
    x0,x1=pos.x.min(),pos.x.max(); y0,y1=pos.y.min(),pos.y.max()
    xm=x0+0.4*(x1-x0); ym=y0+0.4*(y1-y0)
    clip=set(pos[(pos.x>=xm)&(pos.y>=ym)].barcode)
    tg=pd.read_csv(f"{V}/tumor_groups.csv"); gt=set(tg[tg.profile=="Ground Truth"].barcode)
    rows=[]
    for tool,loader in [("copyKAT",load_copykat),("inferCNV",load_infercnv),("siCNV",load_sicnv)]:
        lab=loader("DCIS_2")
        spots=[b for b in clip if b in lab]
        tp=sum(1 for b in spots if lab[b]==1 and b in gt)
        fp=sum(1 for b in spots if lab[b]==1 and b not in gt)
        fn=sum(1 for b in spots if lab[b]==0 and b in gt)
        tn=sum(1 for b in spots if lab[b]==0 and b not in gt)
        prec=tp/(tp+fp) if tp+fp else 0; rec=tp/(tp+fn) if tp+fn else 0
        f1=2*prec*rec/(prec+rec) if prec+rec else 0
        rows.append(dict(tool=tool,n_spots=len(spots),TP=tp,FP=fp,FN=fn,TN=tn,
                         precision=round(prec,3),recall=round(rec,3),F1=round(f1,3),
                         accuracy=round((tp+tn)/len(spots),3)))
    # independence check
    ck=load_copykat("DCIS_2"); ic=load_infercnv("DCIS_2"); si=load_sicnv("DCIS_2")
    for name,other in [("inferCNV",ic),("siCNV",si)]:
        common=set(ck)&set(other); agr=sum(1 for b in common if ck[b]==other[b])
        rows.append(dict(tool=f"{name}_vs_copyKAT_agreement",n_spots=len(common),
                         TP="",FP="",FN="",TN="",precision="",recall="",
                         F1=round(agr/len(common),3) if common else "",
                         accuracy="100% identical" if agr==len(common) else f"{agr}/{len(common)}"))
    return pd.DataFrame(rows)

# ---------------------------------------------------------------- per-sample regression
def _regress(df, name, gt_kind, bver):
    """df has cols: tumor, umi, defined/upv/somatic. Returns rows w/ coef, se, 95% CI."""
    df=df[df.umi>0].copy()
    df["log_umi"]=np.log1p(df.umi)
    n=len(df); ntum=int(df.tumor.sum()); nnorm=n-ntum
    corr=np.corrcoef(df.log_umi, np.log1p(df.somatic))[0,1]
    auc_cov=roc_auc_score(df.tumor, df.umi)
    one=np.ones(n)
    base=dict(sample=name,gt=gt_kind,burden_version=bver)
    out=[dict(**base,model="coverage_only_UMI",n=n,n_tumor=ntum,n_normal=nnorm,
              auc_alone="",auc_with_umi=round(auc_cov,3),burden_coef="",burden_se="",
              ci_lo="",ci_hi="",p_burden_given_umi="",corr_umi_som=round(corr,3))]
    for k in ["somatic","upv","defined"]:
        auc_b=roc_auc_score(df.tumor, df[k])
        X=np.column_stack([one, z(np.log1p(df[k])).values, z(df.log_umi).values])
        b,se,p=logit_irls(X, df.tumor.values)
        auc_full=roc_auc_score(df.tumor, X@b)
        out.append(dict(**base,model=f"{k}_burden",n=n,n_tumor=ntum,n_normal=nnorm,
                        auc_alone=round(auc_b,3),auc_with_umi=round(auc_full,3),
                        burden_coef=round(b[1],3),burden_se=round(se[1],3),
                        ci_lo=round(b[1]-1.96*se[1],3),ci_hi=round(b[1]+1.96*se[1],3),
                        p_burden_given_umi=f"{p[1]:.2e}",corr_umi_som=round(corr,3)))
    return pd.DataFrame(out)

def run_sample(name, bver):
    ckdir,bdirs,outs=SAMPLES[name]
    lab=load_copykat(ckdir)                       # silver GT
    bur=spot_burden(bdirs[bver])
    umi=total_umi(outs)
    bcs=[b for b in bur.index if b in lab and b in umi.index]
    df=pd.DataFrame(index=bcs)
    df["tumor"]=[lab[b] for b in bcs]
    df["umi"]=umi.reindex(bcs).values
    df=df.join(bur.reindex(bcs)).dropna(subset=["umi"])
    return _regress(df, name, "silver_copyKAT", bver)

def run_dcis2_gold(bver):
    """Pathology GOLD reproduction (clip window). bver='pre_dedup' reproduces the
    established anchor (somatic p=0.77); 'current' uses the larger post-dedup set."""
    V="/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/DCIS_2_SPARCAL"
    pos=pd.read_csv(f"{V}/tissue_positions.csv"); pos=pos[pos.in_tissue==1].copy()
    pos["x"]=pos.pxl_col_in_fullres; pos["y"]=pos.pxl_row_in_fullres
    x0,x1=pos.x.min(),pos.x.max(); y0,y1=pos.y.min(),pos.y.max()
    xm=x0+0.4*(x1-x0); ym=y0+0.4*(y1-y0)
    clip=set(pos[(pos.x>=xm)&(pos.y>=ym)].barcode)
    tg=pd.read_csv(f"{V}/tumor_groups.csv"); gt=set(tg[tg.profile=="Ground Truth"].barcode)
    cov=pd.read_csv(f"{V}/spot_coverage.csv").set_index("barcode")["total_umi"]
    bur=spot_burden(SAMPLES["DCIS2"][1][bver])
    bcs=[b for b in clip if b in cov.index and b in bur.index]
    df=pd.DataFrame(index=bcs)
    df["tumor"]=[1 if b in gt else 0 for b in bcs]
    df["umi"]=cov.reindex(bcs).values
    df=df.join(bur.reindex(bcs)).dropna(subset=["umi"])
    return _regress(df, "DCIS2", "gold_pathology", bver)

# ---------------------------------------------------------------- main
if __name__=="__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    print("="*84)
    print("CALIBRATION — CNV tools vs DCIS2 pathology GOLD (clip window)")
    print("="*84)
    cal=calibrate_dcis2()
    print(cal.to_string(index=False))
    cal.to_csv(f"{OUTDIR}/axis2_cnv_calibration_dcis2.csv", index=False)

    print("\n"+"="*84)
    print("COVERAGE-CONTROLLED REGRESSION  (tumor ~ z(log burden)+z(log UMI))")
    print("silver GT = copyKAT; DCIS2 also has gold pathology GT")
    print("="*84)
    allres=[]
    for bver in ["pre_dedup","current"]:
        allres.append(run_dcis2_gold(bver))
        for name in ["P4","P6","DCIS1","DCIS2"]:
            allres.append(run_sample(name, bver))
    for r in allres:
        hd=r.iloc[0]
        print(f"\n--- {hd['sample']} [{hd['gt']}] ({hd['burden_version']}): {hd.n} spots  "
              f"{hd.n_tumor}T/{hd.n_normal}N  corr(logUMI,log_som)={hd.corr_umi_som}  "
              f"coverage_only AUC={hd.auc_with_umi}")
        print(f"    {'model':20s} {'AUC_alone':>9s} {'AUC+UMI':>8s} {'coef':>8s} {'95% CI':>16s} {'p(burden|UMI)':>14s}")
        for _,row in r.iloc[1:].iterrows():
            sig="  <-- SIG" if row.p_burden_given_umi!="" and float(row.p_burden_given_umi)<0.05 else ""
            ci=f"[{row.ci_lo:+.2f},{row.ci_hi:+.2f}]"
            print(f"    {row.model:20s} {str(row.auc_alone):>9s} {str(row.auc_with_umi):>8s} "
                  f"{str(row.burden_coef):>8s} {ci:>16s} {row.p_burden_given_umi:>14s}{sig}")
    res=pd.concat(allres, ignore_index=True)
    res.to_csv(f"{OUTDIR}/axis2_cnv_consensus_regression.csv", index=False)
    print(f"\nWrote {OUTDIR}/axis2_cnv_consensus_regression.csv")
    print(f"Wrote {OUTDIR}/axis2_cnv_calibration_dcis2.csv")
