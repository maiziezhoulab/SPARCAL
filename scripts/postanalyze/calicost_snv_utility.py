#!/usr/bin/env python3
"""Test whether CalicoST CNV and purity help P4/P6 spatial SNV identification."""

from __future__ import annotations
import argparse, bisect, gzip, hashlib, json
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
from scipy.stats import binomtest, fisher_exact, spearmanr

ROOT=Path("/data/maiziezhou_lab/leiy4/snv_calling")
CAL=Path("/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906")
STMUT=Path("/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data")
TRUTH=ROOT/"references/STmut_Table_S1_hg19_autosomal_snvs.tsv"
CYTO=ROOT/"references/hg19_cytoBand.txt.gz"
BED=Path("/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed")
FULL="baseQ0mapQ0_full_calicost"; ABL="baseQ0mapQ0_ablation_no_calicost"
SECTIONS=[
 ("P4","1","clone2_rectangle0_w1.0","cnv_seglevel.tsv"),
 ("P4","2","clone2_rectangle0_w1.0","cnv_seglevel.tsv"),
 ("P6","1","clone3_rectangle0_w1.0","cnv_seglevel.tsv"),
 ("P6","2","clone3_rectangle0_w1.0","cnv_diploid_seglevel.tsv")]
GAINS={"1p","3q","8q","9q","11q","14q","17q","20p","20q"}
LOSSES={"3p","4q","5q","10p","10q","13p","13q","21q"}

def chrom(x): return str(x).removeprefix("chr")
def vkey(c,p,r,a): return f"{chrom(c)}_{int(p)}_{str(r).upper()}_{str(a).upper()}"
def unkey(k):
    c,p,r,a=k.split("_"); return c,int(p),r,a

def paths(p,r,clone,cnv):
    sec=ROOT/"data"/f"{p}_tumor"/r
    co=CAL/f"{p}_sec{r}"
    out=STMUT/f"{p}_Visium"/f"spaceranger_align_rep{r}_hg19"/f"{p}_Tumor_output"/"outs"
    return dict(
      full=sec/"spatial_filter_purity"/FULL/"somatic/somatic_variants.txt",
      ablation=sec/"spatial_filter_purity"/ABL/"somatic/somatic_variants.txt",
      cnv=co/"calicost"/clone/cnv,
      purity=co/"estimate_tumor_prop/loh_estimator_tumor_prop.tsv",
      bam=out/"possorted_genome_bam.dedup.bam")

def calls(path):
    d=pd.read_csv(path,sep="\t",dtype={"chrom":str})
    return {vkey(*x) for x in d[["chrom","pos","ref","alt"]].itertuples(index=False,name=None)
            if len(str(x[2]))==len(str(x[3]))==1}

def vcf(path):
    out=set()
    with gzip.open(path,"rt") as h:
      for line in h:
        if line.startswith("#"): continue
        f=line.rstrip().split("\t"); c,p,r=chrom(f[0]),int(f[1]),f[3].upper()
        for a in f[4].upper().split(","):
          if c in {str(i) for i in range(1,23)} and r in "ACGT" and a in "ACGT" and len(r)==len(a)==1:
            out.add(vkey(c,p,r,a))
    return out

class Intervals:
  def __init__(self,path):
    x=defaultdict(list)
    with open(path) as h:
      for line in h:
        if not line.strip() or line.startswith(("#","track","browser")): continue
        f=line.split("\t"); x[chrom(f[0])].append((int(f[1]),int(f[2])))
    self.x={}
    for c,z in x.items():
      z.sort(); m=[]
      for a,b in z:
        if m and a<=m[-1][1]: m[-1]=(m[-1][0],max(b,m[-1][1]))
        else: m.append((a,b))
      self.x[c]=([a for a,b in m],m)
  def has(self,c,p):
    if c not in self.x:return False
    starts,z=self.x[c]; p-=1; i=bisect.bisect_right(starts,p)-1
    return i>=0 and z[i][0]<=p<z[i][1]

class CNV:
  def __init__(self,path):
    d=pd.read_csv(path,sep="\t"); d.columns=[str(c).replace(" ","_") for c in d]
    cc=[c for c in d if c.startswith("clone") and c.endswith(("_A","_B")) and not c.startswith("clone0_")]
    if not cc: cc=[c for c in d if c.startswith("clone") and c.endswith(("_A","_B"))]
    self.clones=sorted({c.rsplit("_",1)[0] for c in cc}); self.x={}
    for c,g in d.groupby(d["CHR"].map(chrom)):
      z=[]
      for _,r in g.sort_values("START").iterrows():
        states=[(float(r[f"{q}_A"]),float(r[f"{q}_B"])) for q in self.clones]
        z.append((int(r.START),int(r.END),states))
      self.x[c]=([a for a,b,s in z],z)
  def get(self,c,p):
    empty=dict(covered=False,any_alteration=False,total_cna=False,imbalance=False,loh=False,total_cn=np.nan,direction="uncovered")
    if c not in self.x:return empty
    starts,z=self.x[c]; i=bisect.bisect_right(starts,p)-1
    if i<0 or not z[i][0]<=p<=z[i][1]:return empty
    states=z[i][2]; totals=[a+b for a,b in states]; total=float(np.mean(totals))
    return dict(covered=True,
      any_alteration=any(a!=1 or b!=1 for a,b in states),
      total_cna=any(not np.isclose(a+b,2) for a,b in states),
      imbalance=any(not np.isclose(a,b) for a,b in states),
      loh=any((np.isclose(a,0) or np.isclose(b,0)) and a+b>0 for a,b in states),
      total_cn=total,direction="gain" if total>2 else ("loss" if total<2 else "neutral"))

def purity(path):
    d=pd.read_csv(path,sep="\t"); d["barcode"]=d.BARCODES.astype(str).str.split("_").str[0]
    d["purity"]=pd.to_numeric(d.Tumor,errors="coerce")
    return d[["barcode","purity"]].drop_duplicates("barcode")

def spatial_pileup(path,variants,p,r):
    rows=[]
    with pysam.AlignmentFile(str(path),"rb") as bam:
      refs=set(bam.references)
      for n,k in enumerate(sorted(variants),1):
        c,pos,ref,alt=unkey(k); ct=c if c in refs else f"chr{c}"; mol=defaultdict(dict)
        if ct in refs:
          for col in bam.pileup(ct,pos-1,pos,truncate=True,stepper="all",min_base_quality=0,max_depth=1000000):
            if col.reference_pos!=pos-1:continue
            for pr in col.pileups:
              rd=pr.alignment
              if pr.is_del or pr.is_refskip or rd.is_unmapped or rd.is_secondary or rd.is_supplementary or rd.is_qcfail or rd.is_duplicate or not rd.has_tag("CB"):continue
              qp=pr.query_position
              if qp is None:continue
              base=rd.query_sequence[qp].upper()
              if base not in "ACGT":continue
              bq=int(rd.query_qualities[qp]) if rd.query_qualities is not None else 0
              bc=rd.get_tag("CB"); umi=rd.get_tag("UB") if rd.has_tag("UB") else rd.query_name
              old=mol[(bc,umi)].get(base)
              if old is None or (bq,rd.mapping_quality)>old:mol[(bc,umi)][base]=(bq,int(rd.mapping_quality))
        by=defaultdict(lambda:dict(dp0=0,ref0=0,alt0=0,dpq=0,refq=0,altq=0))
        for (bc,u),base_map in mol.items():
          base,(bq,mq)=max(base_map.items(),key=lambda x:x[1])
          by[bc]["dp0"]+=1; by[bc]["ref0"]+=base==ref; by[bc]["alt0"]+=base==alt
          if bq>=13 and mq>=20:
            by[bc]["dpq"]+=1; by[bc]["refq"]+=base==ref; by[bc]["altq"]+=base==alt
        rows.extend(dict(patient=p,rep=r,sample=f"{p}_rep{r}",variant=k,barcode=bc,**z) for bc,z in by.items())
        if n%200==0:print(f"  {p}_rep{r}: {n}/{len(variants)} WES loci",flush=True)
    return pd.DataFrame(rows)

def permute_purity(spots,pur,label,rng,nperm):
    d=spots.merge(pur,on="barcode",how="left"); finite=pur.purity.dropna()
    informative=len(finite)>=20 and finite.nunique()>1; threshold=float(finite.median()) if len(finite) else np.nan
    out=[]
    for q,dp,ac in [("q13_mq20","dpq","altq"),("q0_mq0","dp0","alt0")]:
      for policy in ["complete_case","pipeline_missing_as_zero"]:
        u=d[d[dp]>0].copy()
        if policy=="complete_case":u=u[u.purity.notna()]
        else:u["purity"]=u.purity.fillna(0)
        base=dict(sample=label,quality=q,missing_policy=policy,informative_purity=informative,
          finite_purity_spots=int(pur.purity.notna().sum()),total_purity_rows=len(pur),purity_threshold=threshold,
          covered_spot_loci=len(u),alt_positive_spot_loci=int((u[ac]>0).sum()))
        if not informative or u.empty:
          out.append({**base,"loci_with_alt":0,"observed_high_purity_alt":np.nan,"null_mean_high_purity_alt":np.nan,
                      "enrichment_ratio":np.nan,"permutation_pvalue":np.nan,"spearman_rho":np.nan});continue
        u["high"]=u.purity>threshold; observed=0; draws=np.zeros(nperm,dtype=np.int32); loci=0
        for k,g in u.groupby("variant"):
          N=len(g); A=int((g[ac]>0).sum()); H=int(g.high.sum())
          if A==0:continue
          loci+=1; observed+=int(((g[ac]>0)&g.high).sum()); draws+=rng.hypergeometric(H,N-H,A,size=nperm)
        mean=float(draws.mean()) if loci else np.nan
        rho=spearmanr(u.purity,u[ac]/u[dp]).statistic if u.purity.nunique()>1 else np.nan
        out.append({**base,"loci_with_alt":loci,"observed_high_purity_alt":observed,"null_mean_high_purity_alt":mean,
          "enrichment_ratio":observed/mean if mean>0 else np.nan,
          "permutation_pvalue":float((1+(draws>=observed).sum())/(nperm+1)) if loci else np.nan,"spearman_rho":rho})
    return out,d

def wes_pileup(path,variants,prefix):
    rows=[]
    with pysam.AlignmentFile(str(path),"rb") as bam:
      refs=set(bam.references)
      for k in variants:
        c,pos,ref,alt=unkey(k); ct=c if c in refs else f"chr{c}"; z=defaultdict(int)
        if ct in refs:
          for col in bam.pileup(ct,pos-1,pos,truncate=True,stepper="samtools",min_base_quality=13,min_mapping_quality=20,max_depth=1000000):
            if col.reference_pos!=pos-1:continue
            for pr in col.pileups:
              rd=pr.alignment
              if pr.is_del or pr.is_refskip or rd.is_unmapped or rd.is_secondary or rd.is_supplementary or rd.is_qcfail or rd.is_duplicate:continue
              qp=pr.query_position
              if qp is not None and rd.query_sequence[qp].upper() in "ACGT":z[rd.query_sequence[qp].upper()]+=1
        dp=sum(z.values()); rows.append(dict(variant=k,**{f"{prefix}_dp":dp,f"{prefix}_ref":z[ref],f"{prefix}_alt":z[alt],f"{prefix}_vaf":z[alt]/dp if dp else np.nan}))
    return pd.DataFrame(rows)

def bh(s):
    p=pd.to_numeric(s,errors="coerce").to_numpy(float); q=np.full(len(p),np.nan); ok=np.isfinite(p)
    ix=np.where(ok)[0]
    if len(ix):
      order=ix[np.argsort(p[ix])]; x=p[order]*len(order)/np.arange(1,len(order)+1); x=np.minimum.accumulate(x[::-1])[::-1];q[order]=np.minimum(x,1)
    return q

def candidate_screen(p,shared,truth,raw,bed,spatial):
    variants=[]
    for k in sorted(shared-truth):
      c,pos,ref,alt=unkey(k)
      if bed.has(c,pos) and not(c=="6" and 28510120<=pos<=33480577) and (ref,alt) not in {("A","G"),("T","C")}:variants.append(k)
    t=wes_pileup(STMUT/f"{p}_cSCC_WES"/f"{p}_cSCC_WES.bam",variants,"tumor")
    n=wes_pileup(STMUT/f"{p}_Normal_WES"/f"{p}_Normal_WES.bam",variants,"normal")
    d=t.merge(n,on="variant");d.insert(0,"patient",p);d["in_local_unfiltered_mutect2"]=d.variant.isin(raw)
    d["fisher_p"]=[fisher_exact([[x.tumor_alt,x.tumor_dp-x.tumor_alt],[x.normal_alt,x.normal_dp-x.normal_alt]],alternative="greater").pvalue for x in d.itertuples()]
    d["fisher_fdr"]=bh(d.fisher_p)
    spatial=[]
    for rr in ["1","2"]:
      cfg=next(x for x in SECTIONS if x[0]==p and x[1]==rr)
      print(f"  scanning {p}_rep{rr} candidate RNA evidence",flush=True)
      spatial.append(spatial_pileup(paths(*cfg)["bam"],set(variants),p,rr))
    for frame in spatial:
      r=str(frame.rep.iloc[0]); s=frame.groupby("variant").agg(**{f"rep{r}_alt_umis":("altq","sum"),f"rep{r}_alt_spots":("altq",lambda x:int((x>0).sum()))}).reset_index()
      d=d.merge(s,on="variant",how="left")
    for r in ["1","2"]:
      for x in ["alt_umis","alt_spots"]:d[f"rep{r}_{x}"]=d[f"rep{r}_{x}"].fillna(0).astype(int)
    d["provisional_silver_candidate"]=d.in_local_unfiltered_mutect2&(d.tumor_dp>=20)&(d.normal_dp>=20)&(d.tumor_alt>=3)&(d.tumor_vaf>=.02)&(d.normal_alt<=1)&(d.normal_vaf.fillna(0)<=.01)&(d.fisher_fdr<=.1)&(d.rep1_alt_umis>=1)&(d.rep2_alt_umis>=1)
    return d.sort_values(["provisional_silver_candidate","fisher_fdr","tumor_alt"],ascending=[False,True,False])

def arms():
    x=defaultdict(list)
    with gzip.open(CYTO,"rt") as h:
      for line in h:
        c,a,b,band,stain=line.rstrip().split("\t");c=chrom(c)
        if c in {str(i) for i in range(1,23)} and band[0] in "pq":x[(c,band[0])]+=[int(a)+1,int(b)]
    return {f"{c}{a}":(c,min(z),max(z)) for (c,a),z in x.items()}

def arm_check(idx,p,r):
    rows=[]
    for name,(c,a,b) in arms().items():
      w=defaultdict(int); imbalance_bp=0; loh_bp=0
      if c in idx.x:
        for s,e,states in idx.x[c][1]:
          left,right=max(a,s),min(b,e)
          if left<=right:
            bp=right-left+1; total=np.mean([x+y for x,y in states])
            pred="gain" if total>2 else ("loss" if total<2 else "neutral");w[pred]+=bp
            imbalance_bp+=bp*any(not np.isclose(x,y) for x,y in states)
            loh_bp+=bp*any((np.isclose(x,0) or np.isclose(y,0)) and x+y>0 for x,y in states)
      cov=sum(w.values());pred=max(w,key=w.get) if cov else "uncovered"
      exp="neutral" if p=="P4" else ("gain" if name in GAINS else ("loss" if name in LOSSES else "neutral"))
      rows.append(dict(patient=p,rep=r,sample=f"{p}_rep{r}",arm=name,expected_bulk_wes=exp,predicted_calicost=pred,
        direction_correct=pred==exp if cov else np.nan,covered_bp=cov,
        altered_bp_fraction=(w["gain"]+w["loss"])/cov if cov else np.nan,
        imbalance_bp=imbalance_bp,imbalance_bp_fraction=imbalance_bp/cov if cov else np.nan,
        loh_bp=loh_bp,loh_bp_fraction=loh_bp/cov if cov else np.nan))
    return rows

def table(d):
    def f(x):
      if pd.isna(x):return "NA"
      if isinstance(x,(float,np.floating)):return f"{x:.4g}"
      return str(x).replace("|","\\|")
    cols=[str(c) for c in d];return "\n".join(["| "+" | ".join(cols)+" |","| "+" | ".join("---" for c in cols)+" |"]+["| "+" | ".join(f(x) for x in row)+" |" for row in d.itertuples(index=False,name=None)])

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output-dir",type=Path,default=ROOT/"data/calicost_snv_utility_2026-09-08");ap.add_argument("--n-permutations",type=int,default=50000);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    td=pd.read_csv(TRUTH,sep="\t");td["variant"]=[vkey(*x) for x in td[["chrom_hg19","pos_hg19","ref","alt"]].itertuples(index=False,name=None)]
    truth={p:set(td.loc[td["sample"]==p,"variant"]) for p in ["P4","P6"]};bed=Intervals(BED)
    raw={p:vcf(STMUT/f"{p}_Somatic_Mutect2"/f"{p}_somatic_exome_snps_chr1_22.vcf.gz") for p in ["P4","P6"]}
    rng=np.random.default_rng(20260908);callsets={};cnvann=[];rec=[];rv=[];ar=[];pres=[];spots=[];spatial=defaultdict(list)
    for p,r,cl,cn in SECTIONS:
      label=f"{p}_rep{r}";q=paths(p,r,cl,cn)
      missing=[str(x) for x in q.values() if not x.exists()]
      if missing:raise FileNotFoundError("\n".join(missing))
      full,abl=calls(q["full"]),calls(q["ablation"]);callsets[(p,r)]=dict(full=full,ablation=abl);idx=CNV(q["cnv"])
      fh,ah=full&truth[p],abl&truth[p];b=len(fh-ah);c=len(ah-fh)
      rec.append(dict(patient=p,rep=r,sample=label,truth_n=len(truth[p]),full_calls=len(full),ablation_calls=len(abl),full_truth_hits=len(fh),ablation_truth_hits=len(ah),full_hit_rate=len(fh)/len(truth[p]),ablation_hit_rate=len(ah)/len(truth[p]),full_only_truth_hits=b,ablation_only_truth_hits=c,mcnemar_exact_p=float(binomtest(min(b,c),b+c,.5).pvalue) if b+c else 1))
      rv.extend(dict(patient=p,rep=r,sample=label,variant=k,full_called=k in full,ablation_called=k in abl) for k in truth[p])
      for source,vs in [("published_Table_S1_strict",truth[p]),("local_Mutect2_only_unfiltered",raw[p]-truth[p])]:
        for k in vs:
          cc,pos,ref,alt=unkey(k);cnvann.append(dict(sample=label,source=source,variant=k,**idx.get(cc,pos)))
      ar.extend(arm_check(idx,p,r));print(f"Scanning UMI-deduplicated spatial BAM: {label}",flush=True)
      sp=spatial_pileup(q["bam"],truth[p],p,r);spatial[p].append(sp);pr=purity(q["purity"]);z,m=permute_purity(sp,pr,label,rng,a.n_permutations);pres.extend(z);spots.append(m)
    rec=pd.DataFrame(rec);rv=pd.DataFrame(rv);cnvann=pd.DataFrame(cnvann);ar=pd.DataFrame(ar);pres=pd.DataFrame(pres);spots=pd.concat(spots,ignore_index=True)
    pres["permutation_fdr"]=np.nan
    primary_mask=(pres.quality=="q13_mq20")&(pres.missing_policy=="complete_case")&pres.informative_purity
    pres.loc[primary_mask,"permutation_fdr"]=bh(pres.loc[primary_mask,"permutation_pvalue"])
    rv["rna_alt_observed_q13"]=False
    rv["rna_alt_observed_q0"]=False
    for i,row in rec.iterrows():
      label=row["sample"]; p=row["patient"]; r=str(row["rep"])
      q13=set(spots[(spots["sample"]==label)&(spots.altq>0)].variant)
      q0=set(spots[(spots["sample"]==label)&(spots.alt0>0)].variant)
      rv.loc[rv["sample"].eq(label),"rna_alt_observed_q13"]=rv.loc[rv["sample"].eq(label),"variant"].isin(q13)
      rv.loc[rv["sample"].eq(label),"rna_alt_observed_q0"]=rv.loc[rv["sample"].eq(label),"variant"].isin(q0)
      rec.loc[i,"rna_observed_truth_n"]=len(q13)
      rec.loc[i,"full_hits_among_rna_observed"]=len(q13&callsets[(p,r)]["full"])
      rec.loc[i,"ablation_hits_among_rna_observed"]=len(q13&callsets[(p,r)]["ablation"])
      rec.loc[i,"full_conditional_recall"]=len(q13&callsets[(p,r)]["full"])/len(q13) if q13 else np.nan
      rec.loc[i,"ablation_conditional_recall"]=len(q13&callsets[(p,r)]["ablation"])/len(q13) if q13 else np.nan
    pubcomp=[]
    for p in ["P4","P6"]:
      for r in ["1","2"]:
        col=f"st_rep{r}_alt"; paper=set(td.loc[(td["sample"]==p)&(pd.to_numeric(td[col],errors="coerce").fillna(0)>0),"variant"])
        current=set(spots[(spots["sample"]==f"{p}_rep{r}")&(spots.alt0>0)].variant)
        pubcomp.append(dict(sample=f"{p}_rep{r}",paper_alt_observed=len(paper),current_dedup_q0_alt_observed=len(current),
          shared=len(paper&current),paper_only=len(paper-current),current_only=len(current-paper)))
    pubcomp=pd.DataFrame(pubcomp)
    rep=[]
    for p in ["P4","P6"]:
      for mode in ["full","ablation"]:
        x,y=callsets[(p,"1")][mode],callsets[(p,"2")][mode];rep.append(dict(patient=p,mode=mode,rep1_calls=len(x),rep2_calls=len(y),intersection=len(x&y),union=len(x|y),jaccard=len(x&y)/len(x|y),replicate_shared_truth_hits=len(x&y&truth[p]),either_rep_truth_hits=len((x|y)&truth[p])))
    rep=pd.DataFrame(rep)
    er=[]
    for (sample,outcome),g in pd.DataFrame([dict(sample=s,outcome=o,source=src,n=len(h),positive=int(h[o].sum()),fraction=float(h[o].mean())) for (s,src),h in cnvann[cnvann.covered].groupby(["sample","source"]) for o in ["any_alteration","total_cna","imbalance","loh"]]).groupby(["sample","outcome"]):
      z=g.set_index("source")
      if len(z)==2:
        x=z.loc["published_Table_S1_strict"];y=z.loc["local_Mutect2_only_unfiltered"];tab=[[int(x.positive),int(x.n-x.positive)],[int(y.positive),int(y.n-y.positive)]];od,pv=fisher_exact(tab)
        er.append(dict(sample=sample,outcome=outcome,strict_positive=tab[0][0],strict_negative=tab[0][1],unfiltered_only_positive=tab[1][0],unfiltered_only_negative=tab[1][1],odds_ratio_strict_vs_unfiltered_only=od,fisher_p=pv))
    er=pd.DataFrame(er);er["fisher_fdr"]=bh(er.fisher_p)
    detect=[]
    for p,r,cl,cn in SECTIONS:
      label=f"{p}_rep{r}";g=rv[(rv.patient==p)&(rv.rep==r)];ca=cnvann[(cnvann["sample"]==label)&(cnvann.source=="published_Table_S1_strict")];d=g.merge(ca,on=["sample","variant"]);d=d[d.covered].copy();obs=set(spots[(spots["sample"]==label)&(spots.altq>0)].variant);d["observed"]=d.variant.isin(obs)
      for o in ["any_alteration","total_cna","imbalance","loh"]:
        tab=pd.crosstab(d[o],d.observed).reindex(index=[True,False],columns=[True,False],fill_value=0);od,pv=fisher_exact(tab.to_numpy());detect.append(dict(sample=label,outcome=o,truth_in_cnv=int(d[o].sum()),truth_outside_cnv=int((~d[o]).sum()),rna_observed_in_cnv=int((d[o]&d.observed).sum()),rna_observed_outside_cnv=int(((~d[o])&d.observed).sum()),odds_ratio_rna_observation=od,fisher_p=pv))
    detect=pd.DataFrame(detect);detect["fisher_fdr"]=bh(detect.fisher_p)
    asum=[]
    for s,g in ar.groupby("sample"):
      alt=g[g.expected_bulk_wes!="neutral"];neu=g[g.expected_bulk_wes=="neutral"];asum.append(dict(sample=s,expected_altered_arms=len(alt),altered_arm_direction_accuracy=float(alt.direction_correct.mean()) if len(alt) else np.nan,neutral_arm_accuracy=float(neu.direction_correct.mean()),all_arm_accuracy=float(g.direction_correct.mean()),neutral_bp_total_cna_fraction=float((neu.altered_bp_fraction*neu.covered_bp).sum()/neu.covered_bp.sum()),neutral_bp_imbalance_fraction=float(neu.imbalance_bp.sum()/neu.covered_bp.sum()),neutral_bp_loh_fraction=float(neu.loh_bp.sum()/neu.covered_bp.sum())))
    asum=pd.DataFrame(asum);silver=[]
    for p in ["P4","P6"]:
      print(f"Screening {p} replicate-shared candidates in matched WES",flush=True);silver.append(candidate_screen(p,callsets[(p,"1")]["full"]&callsets[(p,"2")]["full"],truth[p],raw[p],bed,spatial[p]))
    silver=pd.concat(silver,ignore_index=True);provisional=silver[silver.provisional_silver_candidate]
    funnel=pd.DataFrame([dict(patient=p,total_screened=len(g),in_local_unfiltered_mutect2=int(g.in_local_unfiltered_mutect2.sum()),
      tumor_alt_ge3=int((g.tumor_alt>=3).sum()),tumor_specific=int(((g.tumor_alt>=3)&(g.normal_alt<=1)).sum()),
      spatial_alt_both_reps=int(((g.rep1_alt_umis>0)&(g.rep2_alt_umis>0)).sum()),
      provisional_silver_candidates=int(g.provisional_silver_candidate.sum())) for p,g in silver.groupby("patient")])
    outputs={"wes_recovery":rec,"published_vs_current_spatial_observation":pubcomp,"candidate_screen_funnel":funnel,"replicate_concordance":rep,"wes_variant_cnv_annotation":cnvann,"wes_cnv_enrichment_fisher":er,"wes_cnv_rna_observation":detect,"bulk_wes_arm_cnv_concordance":ar,"bulk_wes_arm_cnv_summary":asum,"purity_coverage_conditioned_permutation":pres,"provisional_silver_candidates":silver}
    for name,d in outputs.items():d.to_csv(a.output_dir/f"{name}.tsv",sep="\t",index=False)
    rv.to_csv(a.output_dir/"wes_recovery_variant_level.tsv",sep="\t",index=False);spots.to_csv(a.output_dir/"wes_spot_locus_umi_evidence.tsv.gz",sep="\t",index=False,compression="gzip")
    fig,ax=plt.subplots(2,2,figsize=(12,9));x=np.arange(4);ax[0,0].bar(x-.18,rec.full_truth_hits,.36,label="full");ax[0,0].bar(x+.18,rec.ablation_truth_hits,.36,label="ablation");ax[0,0].set_xticks(x,rec["sample"],rotation=25);ax[0,0].set_title("Allele-exact published WES recovery");ax[0,0].legend()
    for mode,off in [("full",-.18),("ablation",.18)]:
      q=rep[rep["mode"]==mode];ax[0,1].bar(np.arange(2)+off,q.jaccard,.36,label=mode)
    ax[0,1].set_xticks(np.arange(2),["P4","P6"]);ax[0,1].set_title("Cross-replicate exact-call Jaccard");ax[0,1].legend()
    primary=pres[(pres.quality=="q13_mq20")&(pres.missing_policy=="complete_case")];ax[1,0].bar(primary["sample"],primary.enrichment_ratio);ax[1,0].axhline(1,color="k",ls="--");ax[1,0].tick_params(axis="x",rotation=25);ax[1,0].set_title("High-purity ALT spot-locus enrichment")
    ax[1,1].bar(asum["sample"],asum.all_arm_accuracy);ax[1,1].set_ylim(0,1);ax[1,1].tick_params(axis="x",rotation=25);ax[1,1].set_title("CalicoST vs bulk-WES CNA arms");fig.tight_layout();fig.savefig(a.output_dir/"calicost_snv_utility_summary.png",dpi=200);fig.savefig(a.output_dir/"calicost_snv_utility_summary.pdf");plt.close(fig)
    pc=["sample","finite_purity_spots","total_purity_rows","covered_spot_loci","alt_positive_spot_loci","enrichment_ratio","permutation_pvalue","permutation_fdr","spearman_rho"]
    report=["# CalicoST CNV/purity utility for spatial SNV identification","",
      "Independent endpoint: filtered somatic SNVs in published STmut Table S1, lifted from hg38 to hg19 and matched allele-exactly. The local Mutect2 VCF has no FILTER labels and is exploratory only.","",
      "## Strict WES recovery: full versus no-CalicoST ablation","",table(rec),"","## Published versus current UMI-deduplicated spatial observation","",table(pubcomp),"","## Cross-replicate reproducibility","",table(rep),"",
      "## CNV enrichment: strict WES versus local unfiltered-only candidates","",table(er),"",
      "## Does CNV predict spatial-RNA observation of strict WES SNVs?","",table(detect),"",
      "## CalicoST versus published bulk-WES CNA arms","",table(asum),"",
      "P4 is the negative control because the paper reported no discernible CNA or allelic imbalance. P6 expected arms are taken from the paper.","",
      "## Purity and ALT-positive spot-loci","",
      "Primary test: UMI-deduplicated BAM, baseQ >= 13, mapQ >= 20, finite-purity complete cases, and within-locus hypergeometric permutations preserving coverage and ALT-positive counts.","",table(primary[pc]),"",
      "P4 rep2 is non-informative: 29 finite purity values, all zero. P6 rep1 has substantial missingness; missing-as-zero is reported only as sensitivity analysis.","",
      "## Provisional new silver candidates","",table(funnel),"",f"Passing candidates: {len(provisional)}. These are hypotheses, not truth; IGV/manual and orthogonal validation are required.","",table(provisional.head(30)) if len(provisional) else "No candidate passed all criteria.","",
      "## Guardrails","","- CalicoST and SPARCAL share spatial-RNA evidence, so association is not independent validation.","- Bulk WES misses spatial subclones; non-overlap is not proof of a false call.","- RNA VAF is affected by allele-specific expression and 3-prime capture; purity/CNV-adjusted VAF should be a feature, not a hard filter.","- Copy-neutral LOH is separate from total CNA.","- The P6 tumor-WES BAM index is readable and passes pysam check_index, but its timestamp predates the BAM; candidate pileups should be reconfirmed after reindexing if any site is promoted.",""]
    (a.output_dir/"REPORT.md").write_text("\n".join(report))
    manifest=dict(created="2026-09-08",truth=str(TRUTH),truth_sha256=hashlib.sha256(TRUTH.read_bytes()).hexdigest(),truth_counts={p:len(truth[p]) for p in truth},truth_present_local_unfiltered={p:len(truth[p]&raw[p]) for p in truth},permutations=a.n_permutations,outputs=sorted(x.name for x in a.output_dir.iterdir()))
    (a.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(rec.to_string(index=False));print(primary[pc].to_string(index=False));print(f"Provisional silver candidates: {len(provisional)}");print(a.output_dir)
if __name__=="__main__":main()
