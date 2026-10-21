#!/usr/bin/env python
"""STMut extension (P6): per-spot ALT-UMI burden at matched-WES somatic sites, then the
same coverage-conditional region test vs the P6 pathology GT. The fairest somatic test:
TRUE DNA somatic positions, not RNA-called sites. READ-ONLY; parallel pileup.

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/stmut_p6_extension.py
Output: data/spatialsnv_reanalysis_2026-07-17/stmut_p6_extension.csv
"""
import numpy as np, pandas as pd, pysam
from multiprocessing import Pool
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

pysam.set_verbosity(0)   # silence the "index older than data" warning (harmless)

VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Somatic_Mutect2/P6_somatic_snp_chr1_22.vcf.gz"
BAM="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam"
GT ="/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/p6_gt/extracted/P6_rep1_SPARCAL_somatic"
OUT="/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"
NWORK=12

def load_sites():
    s=[]
    with pysam.VariantFile(VCF) as vf:
        for r in vf:
            if r.alts and len(r.ref)==1 and len(r.alts[0])==1:
                s.append((r.chrom,r.pos,r.alts[0].upper()))
    return s

def scan(args):
    lo,hi,sites=args
    bam=pysam.AlignmentFile(BAM,"rb"); out={}
    for i in range(lo,hi):
        c,p,alt=sites[i]
        try: cols=bam.pileup(c,p-1,p,truncate=True,min_mapping_quality=1,stepper="samtools")
        except ValueError: continue
        for col in cols:
            for pr in col.pileups:
                a=pr.alignment
                if pr.is_del or pr.is_refskip or a.is_duplicate or a.is_qcfail: continue
                if not (a.has_tag("CB") and a.has_tag("UB")): continue
                if a.query_sequence[pr.query_position]!=alt: continue
                out.setdefault(a.get_tag("CB"),set()).add((i,a.get_tag("UB")))
    bam.close(); return out

def logit(X,y):
    X=np.asarray(X,float);y=np.asarray(y,float);b=np.zeros(X.shape[1])
    for _ in range(200):
        mu=1/(1+np.exp(-(X@b)));W=np.clip(mu*(1-mu),1e-9,None);z=X@b+(y-mu)/W;XtW=X.T*W
        try:bn=np.linalg.solve(XtW@X,XtW@z)
        except np.linalg.LinAlgError:break
        if np.max(np.abs(bn-b))<1e-8:b=bn;break
        b=bn
    mu=1/(1+np.exp(-(X@b)));W=np.clip(mu*(1-mu),1e-9,None)
    se=np.sqrt(np.diag(np.linalg.inv((X.T*W)@X)));p=2*(1-norm.cdf(np.abs(b/se)));return b,se,p

if __name__=="__main__":
    sites=load_sites(); n=len(sites); print(f"somatic SNV sites: {n}",flush=True)
    step=(n+NWORK-1)//NWORK; chunks=[(i,min(i+step,n),sites) for i in range(0,n,step)]
    merged={}
    with Pool(NWORK) as pool:
        for part in pool.imap_unordered(scan,chunks):
            for bc,s in part.items(): merged.setdefault(bc,set()).update(s)
    burden=pd.Series({bc:len({s for s,_ in v}) for bc,v in merged.items()})   # distinct sites/spot
    altumi=pd.Series({bc:len(v) for bc,v in merged.items()})                   # distinct ALT UMIs
    print(f"spots with signal:{len(burden)} median sites/spot:{burden.median() if len(burden) else 0} max:{burden.max() if len(burden) else 0}",flush=True)

    tg=pd.read_csv(f"{GT}/tumor_groups.csv");gt=set(tg[tg.profile=="GroundTruth"].barcode.astype(str))
    pos=pd.read_csv(f"{GT}/tissue_positions.csv");intissue=[b for b in pos[pos.in_tissue==1].barcode.astype(str)]
    cov=pd.read_csv(f"{GT}/spot_coverage.csv").set_index("barcode")["total_umi"]
    bc=[b for b in intissue if b in cov.index]
    df=pd.DataFrame(index=bc);df["tumor"]=[1 if b in gt else 0 for b in bc];df["umi"]=cov.reindex(bc).values
    df["STMut_burden"]=burden.reindex(bc).fillna(0).values;df["STMut_altumi"]=altumi.reindex(bc).fillna(0).values
    df=df.dropna(subset=["umi"]);df=df[df.umi>0];df["log_umi"]=np.log1p(df.umi)
    nz=int((df.STMut_burden>0).sum())
    print(f"P6 spots:{len(df)} tumor={int(df.tumor.sum())} | spots with STMut signal:{nz} ({100*nz/len(df):.1f}%)")
    print(f"tumor STMut mean:{df[df.tumor==1].STMut_burden.mean():.3f} normal:{df[df.tumor==0].STMut_burden.mean():.3f}")
    ac=roc_auc_score(df.tumor,df.umi);z=lambda s:(s-s.mean())/(s.std()+1e-9);one=np.ones(len(df))
    print(f"coverage-only AUC:{ac:.3f}")
    rows=[{"predictor":"coverage_only","corrUMI":np.nan,"alone_auc":np.nan,"joint_auc":round(ac,3),"coef":np.nan,"p":np.nan,"pct_signal":round(100*nz/len(df),1)}]
    for k in ["STMut_burden","STMut_altumi"]:
        r=np.corrcoef(df.log_umi,np.log1p(df[k]))[0,1];a=roc_auc_score(df.tumor,df[k]) if df[k].nunique()>1 else np.nan
        X=np.column_stack([one,z(np.log1p(df[k])).values,z(df.log_umi).values]);b,se,p=logit(X,df.tumor.values)
        j=roc_auc_score(df.tumor,X@b)
        print(f"{k:16s} corr(UMI)={r:.3f} alone_auc={a:.3f} joint_auc={j:.3f} coef={b[1]:+.3f} p={p[1]:.2e}")
        rows.append({"predictor":k,"corrUMI":round(r,3),"alone_auc":round(a,3),"joint_auc":round(j,3),"coef":round(b[1],3),"p":p[1],"pct_signal":round(100*nz/len(df),1)})
    pd.DataFrame(rows).to_csv(f"{OUT}/stmut_p6_extension.csv",index=False);print(f"WROTE {OUT}/stmut_p6_extension.csv",flush=True)
