#!/usr/bin/env python
"""sparcal_set_benchmark.py — 1-click tumor set-validation benchmark for SPARCAL.

Scores SPARCAL's per-spot variant sets — germline/1kG (`defined`), UPV (`denovo`
germline), `somatic`, and the DISCARDED `ambiguous` set — against cancer-catalogue
and tumor-region ground truth. **Re-run after retraining the spatial filter** to
instantly see whether the somatic / UPV / ambiguous separation improved.

The question every module answers is the ⭐ one: does SPARCAL's *model output*
(somatic / UPV) carry tumor signal that 1kG germline (Beagle) and a coverage
baseline cannot? As of 2026-07-13 the answer is "not for region detection"; this
tool freezes the comparison so a better filter can be measured against it.

Modules (each degrades gracefully if its inputs are missing):
  set_sizes : distinct-SNV counts per set per sample                    (drift check)
  cosmic    : COSMIC hit-rate gradient + somatic-vs-ambiguous Fisher    (variant-level;
              the clean contrast is somatic vs AMBIGUOUS, both non-1kG) (+ HLA-excluded)
  axis2     : coverage-controlled logistic  tumor ~ z(log burden)+z(log UMI)
              (does burden beat coverage for the copyKAT/pathology tumor region?)
  wes       : (P4/P6 only) WES-somatic confirmation overlap of the somatic set

Outputs (timestamped dir): scorecard.csv (one row/sample), SUMMARY.md, per-module CSVs.

Usage:
  python sparcal_set_benchmark.py                                   # all samples, all modules, pre_dedup
  python sparcal_set_benchmark.py --burden-version current          # score the post-dedup / retrained sets
  python sparcal_set_benchmark.py --samples P4 P6 --modules axis2,cosmic
  python sparcal_set_benchmark.py --burden-version current --samples DCIS2
  python sparcal_set_benchmark.py --burden-version current \\
      --burden-root P4=/path/to/postfilter/P4 --burden-root P6=/path/to/postfilter/P6
"""
import os, sys, glob, re, argparse, subprocess, gzip, tempfile, datetime, shutil
import numpy as np, pandas as pd
from scipy.stats import fisher_exact

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# reuse the validated Axis-2 machinery (import is side-effect-free; __main__-guarded)
import axis2_cnv_consensus as ax2

PROJ    = "/data/maiziezhou_lab/leiy4/snv_calling"
BCFTOOLS= f"{PROJ}/apps/bcftools"
BGZIP   = f"{PROJ}/apps/bgzip"
QF      = "baseQ0mapQ0"
COSMIC  = {"GRCh37": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh37.vcf.gz",
           "GRCh38": "/data/maiziezhou_lab/leiy4/COSMIC/Cosmic_GenomeScreensMutant_v103_GRCh38.vcf.gz"}
MHC = ("6", 28_000_000, 34_000_000)   # broad xMHC, chr-stripped contig name

# set token -> VCF path under {burden_dir}/spatial_filter_purity/{QF}/
SET_VCF = {"defined":  "germline/defined/germline_defined.vcf.gz",
           "upv":      "germline/denovo/germline_denovo.vcf.gz",
           "somatic":  "somatic/denovo/somatic_denovo.vcf.gz",
           "ambiguous":"ambiguous/denovo/ambiguous_denovo.vcf.gz"}
SET_ORDER = ["defined","upv","somatic","ambiguous"]

# per-sample metadata beyond what ax2.SAMPLES holds
META = {
    "P4":   dict(build="GRCh37", wes="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz"),
    "P6":   dict(build="GRCh37", wes="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Somatic_Mutect2/P6_somatic_exome_snps_chr1_22.vcf.gz"),
    "DCIS1":dict(build="GRCh38", wes=None),
    "DCIS2":dict(build="GRCh38", wes=None),
}

def burden_dir(sample, version):
    return ax2.SAMPLES[sample][1][version]

# ------------------------------------------------------------------ helpers
def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)

def _vcf_keys(path, strip_chr=True, work=None):
    """Return set of (chrom,pos,ref,alt) from a (bgzipped) VCF via bcftools query."""
    p = _run([BCFTOOLS, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\n", path])
    keys=set()
    for ln in p.stdout.splitlines():
        c=ln.split("\t")
        if len(c)<4: continue
        chrom=c[0][3:] if (strip_chr and c[0].startswith("chr")) else c[0]
        keys.add((chrom, c[1], c[2], c[3]))
    return keys

def _prep_nochr(src, dst):
    """Strip 'chr', bgzip, index (cached)."""
    if os.path.exists(dst+".tbi"):
        return dst
    view=_run([BCFTOOLS, "view", src])
    stripped="\n".join(
        (ln if ln.startswith("#") else re.sub(r"^chr","",ln)) for ln in view.stdout.splitlines()
    )+"\n"
    with open(dst[:-3], "w") as f: f.write(stripped)
    _run([BGZIP, "-f", dst[:-3]])
    _run([BCFTOOLS, "index", "-f", "-t", dst])
    return dst

# ------------------------------------------------------------------ modules
def module_set_sizes(samples, version):
    rows=[]
    for s in samples:
        b=f"{burden_dir(s,version)}/spatial_filter_purity/{QF}"
        row=dict(sample=s)
        for tok in SET_ORDER:
            f=f"{b}/{SET_VCF[tok]}"
            n=""
            if os.path.exists(f):
                q=_run([BCFTOOLS,"view","-H",f]); n=len(q.stdout.splitlines())
            row[f"n_{tok}"]=n
        rows.append(row)
    return pd.DataFrame(rows)

def module_cosmic(samples, version, work):
    rows=[]
    for s in samples:
        build=META[s]["build"]; cat=COSMIC[build]
        b=f"{burden_dir(s,version)}/spatial_filter_purity/{QF}"
        counts={}   # tok -> (hits, total, hits_nomhc, total_nomhc)
        for tok in SET_ORDER:
            src=f"{b}/{SET_VCF[tok]}"
            if not os.path.exists(src):
                counts[tok]=None; continue
            nochr=f"{work}/{s}_{tok}_nochr.vcf.gz"
            _prep_nochr(src, nochr)
            setk=_vcf_keys(nochr, strip_chr=False)
            # catalogue records only at the set's positions (fast, index-targeted)
            reg=f"{work}/{s}_{tok}_pos.txt"
            with open(reg,"w") as fh:
                for (c,p,_,_) in setk: fh.write(f"{c}\t{p}\n")
            cq=_run([BCFTOOLS,"view","-H","-R",reg,cat])
            catk=set()
            for ln in cq.stdout.splitlines():
                cc=ln.split("\t")
                if len(cc)>=5: catk.add((cc[0],cc[1],cc[3],cc[4]))
            hit=setk & catk
            def nomhc(ks): return {k for k in ks if not (k[0]==MHC[0] and MHC[1]<=int(k[1])<=MHC[2])}
            counts[tok]=(len(hit), len(setk), len(nomhc(hit)), len(nomhc(setk)))
        row=dict(sample=s, build=build)
        for tok in SET_ORDER:
            c=counts[tok]
            row[f"{tok}_pct"]= round(100*c[0]/c[1],3) if c and c[1] else ""
        # somatic-vs-ambiguous Fisher (full + HLA-excluded)
        cs, ca = counts.get("somatic"), counts.get("ambiguous")
        for suf,idx in [("", (0,1)), ("_hlaexcl",(2,3))]:
            if cs and ca and cs[idx[1]] and ca[idx[1]]:
                sh,st=cs[idx[0]],cs[idx[1]]; ah,at=ca[idx[0]],ca[idx[1]]
                ratio=(sh/st)/(ah/at) if ah else float("nan")
                _,pp=fisher_exact([[sh,st-sh],[ah,at-ah]], alternative="greater")
                row[f"som_amb_ratio{suf}"]=round(ratio,3); row[f"fisher_p{suf}"]=f"{pp:.2e}"
            else:
                row[f"som_amb_ratio{suf}"]=""; row[f"fisher_p{suf}"]=""
        rows.append(row)
    return pd.DataFrame(rows)

def module_axis2(samples, version):
    """Coverage-controlled regression via the validated ax2 machinery."""
    frames=[]
    for s in samples:
        try:
            frames.append(ax2.run_sample(s, version))
            if s=="DCIS2":
                frames.append(ax2.run_dcis2_gold(version))
        except Exception as e:
            print(f"  [axis2] {s}: FAILED ({e})")
    if not frames: return pd.DataFrame(), pd.DataFrame()
    full=pd.concat(frames, ignore_index=True)
    # compact per-sample somatic headline
    sm=full[full.model=="somatic_burden"][["sample","gt","auc_alone","auc_with_umi",
                                            "burden_coef","p_burden_given_umi","corr_umi_som"]]
    cov=full[full.model=="coverage_only_UMI"][["sample","gt","auc_with_umi"]].rename(
        columns={"auc_with_umi":"coverage_auc"})
    head=sm.merge(cov,on=["sample","gt"]).rename(columns={
        "burden_coef":"somatic_coef","p_burden_given_umi":"somatic_p","auc_with_umi":"somatic+umi_auc"})
    return full, head

def module_wes(samples, version, work):
    rows=[]
    for s in samples:
        wes=META[s].get("wes")
        if not wes or not os.path.exists(wes): continue
        b=f"{burden_dir(s,version)}/spatial_filter_purity/{QF}"
        wesk=_vcf_keys(wes, strip_chr=True)      # WES is chr-prefixed hg19
        row=dict(sample=s, n_wes_somatic=len(wesk))
        for tok in ["somatic","ambiguous","defined","upv"]:
            src=f"{b}/{SET_VCF[tok]}"
            if not os.path.exists(src): row[f"{tok}_wes_overlap"]=""; continue
            nochr=f"{work}/{s}_{tok}_nochr.vcf.gz"; _prep_nochr(src,nochr)
            setk=_vcf_keys(nochr, strip_chr=False)
            row[f"{tok}_wes_overlap"]=len(setk & wesk)
        rows.append(row)
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ driver
def main():
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", default=["P4","P6","DCIS1","DCIS2"])
    ap.add_argument("--burden-version", choices=["pre_dedup","current"], default="pre_dedup",
                    help="which spatial-filter output to score (pre_dedup=established anchor; "
                         "current=post-dedup/retrained live '1' dirs)")
    ap.add_argument("--burden-root", action="append", default=[], metavar="SAMPLE=DIR",
                    help="replace one sample's burden directory for this run; repeat as needed. "
                         "Useful for non-destructive post-filter experiments.")
    ap.add_argument("--modules", default="set_sizes,cosmic,axis2,wes",
                    help="comma list of {set_sizes,cosmic,axis2,wes}")
    ap.add_argument("--outdir", default=None)
    args=ap.parse_args()
    # Keep experimental outputs outside the canonical data trees.  axis2 imports
    # the same dictionary, so changing this in-process mapping keeps all modules
    # (set sizes, COSMIC, WES, and burden summaries) coherent.
    for item in args.burden_root:
        if "=" not in item:
            ap.error(f"--burden-root must be SAMPLE=DIR, got: {item}")
        sample, path = item.split("=", 1)
        if sample not in META:
            ap.error(f"unknown sample in --burden-root: {sample}")
        if not os.path.isdir(path):
            ap.error(f"--burden-root directory does not exist: {path}")
        ax2.SAMPLES[sample][1][args.burden_version] = os.path.abspath(path)
    mods=set(args.modules.split(","))
    stamp=datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    out=args.outdir or f"{PROJ}/data/set_benchmark/{stamp}_{args.burden_version}"
    os.makedirs(out, exist_ok=True)
    work=tempfile.mkdtemp(prefix="setbench_")
    print(f"# SPARCAL set benchmark | samples={args.samples} | version={args.burden_version}")
    print(f"# out={out}\n# work={work}\n")

    tables={}
    if "set_sizes" in mods:
        print("== set_sizes =="); tables["set_sizes"]=module_set_sizes(args.samples,args.burden_version)
        print(tables["set_sizes"].to_string(index=False)); print()
    if "cosmic" in mods:
        print("== cosmic (somatic vs ambiguous is the clean contrast) ==")
        tables["cosmic"]=module_cosmic(args.samples,args.burden_version,work)
        print(tables["cosmic"].to_string(index=False)); print()
    axis2_full=axis2_head=None
    if "axis2" in mods:
        print("== axis2 (coverage-controlled; somatic_coef>0 w/ p<0.05 = beats coverage) ==")
        axis2_full,axis2_head=module_axis2(args.samples,args.burden_version)
        tables["axis2_full"]=axis2_full; tables["axis2"]=axis2_head
        if not axis2_head.empty: print(axis2_head.to_string(index=False)); print()
    if "wes" in mods:
        wes=module_wes(args.samples,args.burden_version,work)
        if not wes.empty:
            print("== wes (P4/P6; expression-restricted, diagnostic only) ==")
            tables["wes"]=wes; print(wes.to_string(index=False)); print()

    # write per-module CSVs
    for k,v in tables.items():
        if v is not None and not v.empty: v.to_csv(f"{out}/{k}.csv", index=False)

    # ---- scorecard (one row per sample) ----
    sc=pd.DataFrame({"sample":args.samples})
    def _mrg(df, cols):
        nonlocal sc
        if df is not None and not df.empty:
            sc=sc.merge(df[["sample"]+[c for c in cols if c in df.columns]], on="sample", how="left")
    _mrg(tables.get("set_sizes"), [f"n_{t}" for t in SET_ORDER])
    _mrg(tables.get("cosmic"), ["som_amb_ratio","fisher_p","som_amb_ratio_hlaexcl","fisher_p_hlaexcl"])
    if axis2_head is not None and not axis2_head.empty:
        sil=axis2_head[axis2_head["gt"]=="silver_copyKAT"][["sample","somatic_coef","somatic_p","coverage_auc"]]
        sc=sc.merge(sil, on="sample", how="left")
    _mrg(tables.get("wes"), ["somatic_wes_overlap","n_wes_somatic"])
    sc.to_csv(f"{out}/scorecard.csv", index=False)
    print("== SCORECARD =="); print(sc.to_string(index=False)); print()

    write_summary(out, args, sc, tables)
    print(f"Wrote {out}/scorecard.csv + SUMMARY.md + per-module CSVs")
    shutil.rmtree(work, ignore_errors=True)

def _md_table(df):
    cols=list(df.columns)
    head="| "+" | ".join(cols)+" |"
    sep="|"+"|".join(["---"]*len(cols))+"|"
    rows=["| "+" | ".join("" if pd.isna(v) else str(v) for v in r)+" |"
          for r in df.itertuples(index=False)]
    return "\n".join([head,sep]+rows)

def write_summary(out, args, sc, tables):
    L=[]
    L.append(f"# SPARCAL set-validation benchmark — {datetime.datetime.now():%Y-%m-%d %H:%M}")
    L.append("")
    L.append(f"- **samples:** {', '.join(args.samples)}  |  **burden version:** `{args.burden_version}`")
    L.append(f"- **modules:** {args.modules}")
    L.append("")
    L.append("## How to read this")
    L.append("- **cosmic `som_amb_ratio`** — COSMIC hit-rate of the somatic set ÷ ambiguous (discarded) set. "
             ">1 with small `fisher_p` = the KEPT somatic calls are more cancer-catalogued than what the "
             "filter DISCARDS (a clean, coverage-free contrast; the HLA-excluded columns are the robust ones).")
    L.append("- **axis2 `somatic_coef` / `somatic_p`** — standardized somatic-burden coefficient after "
             "controlling for total UMI (copyKAT silver GT). **coef>0 with p<0.05 = somatic burden predicts "
             "the tumor region BEYOND coverage.** As of 2026-07-13 this is never robustly true.")
    L.append("- **coverage_auc** — AUC of total-UMI alone for the tumor region (context: if ~1.0 the silver "
             "GT is coverage-dominated and the somatic test there is near-degenerate).")
    L.append("- **wes overlap** — |somatic ∩ WES-somatic| (P4/P6). Expression-restricted (~1% of WES-somatic "
             "positions are RNA-covered) so this is diagnostic, not a precision estimate.")
    L.append("")
    L.append("## Scorecard")
    L.append("")
    L.append(_md_table(sc))
    L.append("")
    L.append("## Verdict heuristic (for a RETRAINED filter)")
    L.append("A better spatial filter should push, vs this baseline: (1) cosmic `som_amb_ratio_hlaexcl` "
             "UP (esp. cSCC P4/P6) and its `fisher_p` DOWN; (2) axis2 `somatic_coef` positive and "
             "`somatic_p`<0.05 in ≥1 coverage-independent case (DCIS2 gold / P6 where coverage_auc<0.5); "
             "(3) somatic set size not exploding (drift). None of these hold at baseline → somatic region "
             "signal is currently absent; the open avenue is Axis 3 (sub-clonal structure).")
    L.append("")
    L.append("## Re-run after retraining")
    L.append("```")
    L.append("python scripts/postanalyze/sparcal_set_benchmark.py --burden-version current")
    L.append("```")
    L.append("(the retrained filter writes to the live `data/{sample}/…` dirs = `current`; "
             "compare its scorecard against this one.)")
    with open(f"{out}/SUMMARY.md","w") as f: f.write("\n".join(L))

if __name__=="__main__":
    main()
