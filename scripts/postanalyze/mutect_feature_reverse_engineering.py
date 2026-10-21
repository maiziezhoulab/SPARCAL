#!/usr/bin/env python3
"""Reverse-engineer transferable read-evidence features from matched WES Mutect2.

Published filtered STmut Table S1 calls are the positive stratum; the remainder
of the local unfiltered Mutect2 VCF is the comparison stratum. This is a
filter-faithfulness audit, not independent biological validation.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,pysam
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import fisher_exact,mannwhitneyu
from sklearn.metrics import roc_auc_score

ROOT=Path("/data/maiziezhou_lab/leiy4/snv_calling")
STMUT=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data")
TRUTH=ROOT/"references/STmut_Table_S1_hg19_autosomal_snvs.tsv"

def chrom(x):return str(x).removeprefix("chr")
def key(c,p,r,a):return f"{chrom(c)}_{int(p)}_{str(r).upper()}_{str(a).upper()}"
def scalar(x,i=0):
 if x is None:return np.nan
 if isinstance(x,(tuple,list)):
  return x[min(i,len(x)-1)] if len(x) else np.nan
 return x
def bal(a,b):
 return min(a,b)/(a+b) if np.isfinite(a) and np.isfinite(b) and a+b>0 else np.nan
def allele(sample,field,i):
 x=sample.get(field)
 try:return float(x[i+1])
 except (IndexError,TypeError,ValueError):return np.nan

def parse(patient,truth):
 path=STMUT/f"{patient}_Somatic_Mutect2"/f"{patient}_somatic_exome_snps_chr1_22.vcf.gz";rows=[]
 with pysam.VariantFile(str(path)) as v:
  names=list(v.header.samples);nn,tn=names[0],names[-1]
  for rec in v:
   ref=rec.ref.upper();c=chrom(rec.chrom)
   if c not in {str(i) for i in range(1,23)} or len(ref)!=1:continue
   ns,ts=rec.samples[nn],rec.samples[tn]
   for ai,alt in enumerate(rec.alts or ()):
    alt=alt.upper()
    if len(alt)!=1 or ref not in "ACGT" or alt not in "ACGT":continue
    k=key(c,rec.pos,ref,alt)
    nr,na,tr,ta=allele(ns,"AD",-1),allele(ns,"AD",ai),allele(ts,"AD",-1),allele(ts,"AD",ai)
    nr=nr if np.isfinite(nr) else 0;na=na if np.isfinite(na) else 0;tr=tr if np.isfinite(tr) else 0;ta=ta if np.isfinite(ta) else 0
    ndp,tdp=nr+na,tr+ta
    sb=ts.get("SB");af,ar=(float(sb[2]),float(sb[3])) if sb is not None and len(sb)>=4 else (np.nan,np.nan)
    f1,f2=allele(ts,"F1R2",ai),allele(ts,"F2R1",ai);info=rec.info
    def num(name,i=0):
     try:return float(scalar(info.get(name),i))
     except (TypeError,ValueError):return np.nan
    rows.append(dict(patient=patient,variant=k,published_strict=k in truth,ref=ref,alt=alt,
      edit_class=(ref,alt) in {("A","G"),("T","C")},tumor_dp=tdp,tumor_alt=ta,
      tumor_vaf=ta/tdp if tdp else np.nan,normal_dp=ndp,normal_alt=na,
      normal_vaf=na/ndp if ndp else np.nan,
      delta_vaf=(ta/tdp if tdp else np.nan)-(na/ndp if ndp else np.nan),
      tumor_alt_strand_balance=bal(af,ar),tumor_alt_orientation_balance=bal(f1,f2),
      tumor_alt_fad=allele(ts,"FAD",ai),tlod=num("TLOD",ai),nlod=num("NLOD",ai),
      nalod=num("NALOD",ai),popaf=num("POPAF",ai),mbq_alt=num("MBQ",ai+1),
      mmq_alt=num("MMQ",ai+1),mpos=num("MPOS",ai),mfrl_alt=num("MFRL",ai+1),
      ecnt=num("ECNT"),contq=num("CONTQ"),germq=num("GERMQ"),roq=num("ROQ"),
      seqq=num("SEQQ"),strandq=num("STRANDQ"),unique_alt_reads=num("AS_UNIQ_ALT_READ_COUNT",ai),
      pon=bool(info.get("PON",False))))
 return pd.DataFrame(rows)

def bh(s):
 p=pd.to_numeric(s,errors="coerce").to_numpy(float);q=np.full(len(p),np.nan);ix=np.where(np.isfinite(p))[0]
 if len(ix):
  o=ix[np.argsort(p[ix])];x=p[o]*len(o)/np.arange(1,len(o)+1);x=np.minimum.accumulate(x[::-1])[::-1];q[o]=np.minimum(x,1)
 return q

def audit(d):
 skip={"patient","variant","published_strict","ref","alt","edit_class","pon"};rows=[]
 for f in [c for c in d if c not in skip]:
  z=d[["published_strict",f]].dropna();a=z.loc[z.published_strict,f].astype(float);b=z.loc[~z.published_strict,f].astype(float)
  if len(a)<10 or len(b)<10 or z[f].nunique()<2:continue
  auc=roc_auc_score(z.published_strict.astype(int),z[f]);rows.append(dict(feature=f,n_strict=len(a),n_unfiltered_only=len(b),strict_median=float(a.median()),unfiltered_only_median=float(b.median()),direction="higher_is_strict" if auc>=.5 else "lower_is_strict",oriented_auc=max(auc,1-auc),mannwhitney_p=mannwhitneyu(a,b).pvalue))
 out=pd.DataFrame(rows).sort_values("oriented_auc",ascending=False);out["mannwhitney_fdr"]=bh(out.mannwhitney_p);return out

def support_match(d):
 """Match 1:1 on patient, tumor ALT count, tumor depth, and normal depth."""
 frames=[];diagnostics=[]
 for patient,g in d.groupby("patient"):
  pos=g[g.published_strict].copy();neg=g[(~g.published_strict)&(g.tumor_alt>=3)].copy();fields=["tumor_alt","tumor_dp","normal_dp"]
  both=pd.concat([pos,neg]);x=np.log1p(both[fields].to_numpy(float));mu=x.mean(0);sd=x.std(0);sd[sd==0]=1
  xp=(np.log1p(pos[fields].to_numpy(float))-mu)/sd;xn=(np.log1p(neg[fields].to_numpy(float))-mu)/sd
  ri,ci=linear_sum_assignment(cdist(xp,xn,metric="sqeuclidean"))
  matched=pd.concat([pos.iloc[ri],neg.iloc[ci]],ignore_index=True);frames.append(matched)
  for field in fields:
   a=pos.iloc[ri][field].astype(float);b=neg.iloc[ci][field].astype(float);pool=np.sqrt((a.var(ddof=1)+b.var(ddof=1))/2)
   diagnostics.append(dict(patient=patient,field=field,strict_mean=a.mean(),unfiltered_only_mean=b.mean(),standardized_mean_difference=(a.mean()-b.mean())/pool if pool else 0))
 return pd.concat(frames,ignore_index=True),pd.DataFrame(diagnostics)

def gates(d):
 ori=(d.tumor_alt<4)|(d.tumor_alt_orientation_balance.fillna(0)>0);strand=(d.tumor_alt<4)|(d.tumor_alt_strand_balance.fillna(0)>0)
 g0=d.tumor_alt>=3
 g1=g0&(d.normal_alt<=1)&(d.normal_vaf.fillna(0)<=.01)
 g2=g1&ori&strand
 g3=g2&(d.mbq_alt>=20)&(d.mmq_alt>=40)&(d.mpos>=5)
 gs=[("G0_alt_support",g0),("G1_plus_clean_normal",g1),("G2_plus_bidirectional",g2),("G3_plus_read_quality",g3),("G4_plus_population_context",g3&(~d.pon)&(d.popaf>=5)),("G5_plus_non_editing",g3&(~d.pon)&(d.popaf>=5)&(~d.edit_class))]
 rows=[];P=int(d.published_strict.sum());N=int((~d.published_strict).sum())
 for name,x in gs:
  a=int((x&d.published_strict).sum());b=P-a;c=int((x&~d.published_strict).sum());e=N-c;od,p=fisher_exact([[a,b],[c,e]])
  rows.append(dict(gate=name,strict_pass=a,strict_total=P,strict_recall=a/P,unfiltered_only_pass=c,unfiltered_only_total=N,unfiltered_only_retention=c/N,enrichment_odds_ratio=od,fisher_p=p))
 return pd.DataFrame(rows)

def md(d):
 def f(x):
  if pd.isna(x):return "NA"
  if isinstance(x,(float,np.floating)):return f"{x:.4g}"
  return str(x).replace("|","\\|")
 cols=list(d);return "\n".join(["| "+" | ".join(cols)+" |","| "+" | ".join("---" for x in cols)+" |"]+["| "+" | ".join(f(x) for x in row)+" |" for row in d.itertuples(index=False,name=None)])

def main():
 ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=ROOT/"data/mutect_feature_reverse_engineering_2026-09-10");a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
 t=pd.read_csv(TRUTH,sep="\t");truth={p:{key(*x) for x in t.loc[t["sample"]==p,["chrom_hg19","pos_hg19","ref","alt"]].itertuples(index=False,name=None)} for p in ["P4","P6"]}
 d=pd.concat([parse(p,truth[p]) for p in ["P4","P6"]],ignore_index=True);fa=audit(d);matched,match_diag=support_match(d);matched_fa=audit(matched)
 ga=pd.concat([gates(d).assign(patient="pooled")]+[gates(d[d.patient==p]).assign(patient=p) for p in ["P4","P6"]],ignore_index=True)
 d.to_csv(a.output_dir/"mutect_variant_features.tsv.gz",sep="\t",index=False,compression="gzip");fa.to_csv(a.output_dir/"univariate_feature_audit.tsv",sep="\t",index=False);matched.to_csv(a.output_dir/"support_depth_matched_records.tsv.gz",sep="\t",index=False,compression="gzip");matched_fa.to_csv(a.output_dir/"support_depth_matched_feature_audit.tsv",sep="\t",index=False);match_diag.to_csv(a.output_dir/"support_depth_match_diagnostics.tsv",sep="\t",index=False);ga.to_csv(a.output_dir/"mechanistic_gate_combinations.tsv",sep="\t",index=False)
 mapping=pd.DataFrame([
 ("tumor ALT support/TLOD","ALT UMI count plus beta-binomial variant-vs-error likelihood"),
 ("normal ALT/VAF, NLOD/NALOD","low-purity or normal-clone ALT evidence; matched normal when available"),
 ("SB and F1R2/F2R1 balance","strand and read-orientation balance from deduplicated spatial BAM"),
 ("MBQ/MMQ/MPOS/MFRL","ALT base quality, mapping quality, read-end position, fragment context"),
 ("unique ALT reads","unique CB-UB molecules and unique alignment starts"),
 ("POPAF/PON","population panel, cohort recurrence, matched normals"),
 ("ECNT/local events","nearby multi-event and homopolymer/repeat context"),
 ("tumor-normal delta VAF","high-purity minus low-purity/normal-clone VAF with coverage conditioning")],columns=["Mutect2 evidence","SPARCAL analogue"])
 mapping.to_csv(a.output_dir/"transferable_feature_mapping.tsv",sep="\t",index=False)
 top=fa.head(15).sort_values("oriented_auc");fig,ax=plt.subplots(1,2,figsize=(13,6));ax[0].barh(top.feature,top.oriented_auc);ax[0].axvline(.5,color="k",ls="--");ax[0].set_xlim(.5,1);ax[0].set_title("Strict published vs unfiltered-only");ax[0].set_xlabel("Oriented AUC")
 z=ga[ga.patient=="pooled"];x=np.arange(len(z));ax[1].plot(x,z.strict_recall,marker="o",label="strict recall");ax[1].plot(x,z.unfiltered_only_retention,marker="o",label="unfiltered-only retention");ax[1].set_xticks(x,[q.replace("_plus_","\n+") for q in z.gate],rotation=25,ha="right");ax[1].set_ylim(0,1);ax[1].set_title("Non-trained mechanistic gates");ax[1].legend();fig.tight_layout();fig.savefig(a.output_dir/"mutect_feature_audit.png",dpi=200);fig.savefig(a.output_dir/"mutect_feature_audit.pdf");plt.close(fig)
 report=["# Mutect2 feature reverse engineering","",f"Pooled records: {len(d):,}; published-strict overlap: {int(d.published_strict.sum()):,}; unfiltered-only: {int((~d.published_strict).sum()):,}.","",
 "Positive stratum is the published filtered STmut Table S1 set; comparison is the rest of the local unfiltered Mutect2 VCF. This measures filter-faithfulness, not independent truth.","","## Strongest individual fields","",md(fa.head(20)),"","## After 1:1 matching on tumor ALT count, tumor depth, and normal depth","",md(matched_fa.head(20)),"","Matching diagnostics (absolute standardized mean differences near zero are desired):","",md(match_diag),"","## Non-trained gate combinations","",md(ga),"","## Transfer to spatial RNA","",md(mapping),"","Directional conclusion only: normal contamination, ALT support, and read-level artifact balance are useful evidence families. Exact WES cutoffs must not be copied to sparse spatial RNA; use likelihood terms with explicit missingness and coverage conditioning.",""]
 (a.output_dir/"REPORT.md").write_text("\n".join(report));(a.output_dir/"manifest.json").write_text(json.dumps(dict(records=len(d),strict=int(d.published_strict.sum()),outputs=sorted(p.name for p in a.output_dir.iterdir())),indent=2)+"\n")
 print(fa.head(20).to_string(index=False));print(ga.to_string(index=False));print(a.output_dir)
if __name__=="__main__":main()
