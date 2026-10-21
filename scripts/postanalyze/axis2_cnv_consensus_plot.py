#!/usr/bin/env python
"""Forest + AUC figure for Axis-2 coverage-controlled concordance.
Reads axis2_cnv_consensus_regression.csv. Headline: SPARCAL somatic burden adds no
robust tumor-region signal beyond coverage in ANY sample (high- or low-TMB)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42,"font.size":10})

OUT="/data/maiziezhou_lab/leiy4/snv_calling/data/somatic_validation_2026-07-13"
df=pd.read_csv(f"{OUT}/data/axis2_cnv_consensus_regression.csv")

# paper colours: germline=blue, UPV=purple, somatic=red
COL={"defined":"#2166AC","upv":"#9970AB","somatic":"#B2182B"}
LAB={"defined":"germline / 1kG","upv":"UPV","somatic":"somatic"}
SETS=["defined","upv","somatic"]

def cond_label(r):
    tag="pathology GOLD" if r["gt"]=="gold_pathology" else "copyKAT silver"
    return f"{r['sample']}  ({tag})"

# conditions in display order
order=[("DCIS2","gold_pathology"),("P4","silver_copyKAT"),("P6","silver_copyKAT"),
       ("DCIS1","silver_copyKAT"),("DCIS2","silver_copyKAT")]

fig,(axA,axB)=plt.subplots(1,2,figsize=(13.5,5.6),gridspec_kw={"width_ratios":[1.35,1]})

# ============================================================ Panel A: forest (pre_dedup, all 3 sets)
prim=df[df.burden_version=="pre_dedup"]
yticks=[]; ylabs=[]
y=0; gap=1.1
for (samp,gt) in order[::-1]:
    sub=prim[(prim['sample']==samp)&(prim["gt"]==gt)]
    base_y=y
    for j,s in enumerate(SETS):
        row=sub[sub.model==f"{s}_burden"]
        if row.empty: continue
        row=row.iloc[0]
        yy=y+ (j-1)*0.24
        coef=float(row.burden_coef); lo=float(row.ci_lo); hi=float(row.ci_hi)
        p=float(row.p_burden_given_umi)
        sig = p<0.05
        axA.plot([lo,hi],[yy,yy],color=COL[s],lw=2,solid_capstyle="round",zorder=2)
        axA.plot([coef],[yy],"o",color=COL[s],ms=8 if sig else 7,
                 mec="white",mew=1.2,zorder=3)
        if sig:
            axA.annotate("*",(coef,yy),textcoords="offset points",xytext=(0,3),
                         ha="center",va="bottom",color=COL[s],fontsize=13,fontweight="bold")
    yticks.append(base_y); ylabs.append(cond_label(sub.iloc[0]))
    y+=gap
axA.axvline(0,color="#444",ls="--",lw=1,zorder=1)
axA.set_yticks(yticks); axA.set_yticklabels(ylabs)
axA.set_xlim(-6.6,4.2)
axA.set_xlabel("standardized burden coefficient  (controlling for total UMI)")
axA.set_title("A   Does per-spot SNV burden predict tumor beyond coverage?\n"
              "     (coef>0 = adds signal; * = p<0.05)",loc="left",fontsize=11)
# annotation for somatic null
axA.text(0.015,0.34,"somatic: no significant positive effect in any sample.\n"
         "large negative germline/UPV coefs = collinearity\n"
         "artifact (burden~UMI r≈0.8–0.94), not biology.",
         transform=axA.transAxes,fontsize=7.6,color="#555",va="top",
         bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#ccc",alpha=0.9))
leg=[Line2D([0],[0],color=COL[s],lw=2,marker="o",mec="white",label=LAB[s]) for s in SETS]
axA.legend(handles=leg,loc="lower left",frameon=False,fontsize=9,title="SNV set")
for sp in ("top","right"): axA.spines[sp].set_visible(False)
axA.grid(axis="x",color="#eee",zorder=0)

# ============================================================ Panel B: SOMATIC coef robustness (both burden versions)
# The decisive quantity: is the somatic coefficient (controlling UMI) ever robustly >0?
VER={"pre_dedup":("#7A0018","o","pre-dedup (primary)"),
     "current":  ("#E08A8A","s","current (post-dedup)")}
ycond=[]; ylab2=[]
y=0
for (samp,gt) in order[::-1]:
    for bv,(c,mk,_) in VER.items():
        row=df[(df['sample']==samp)&(df["gt"]==gt)&(df.burden_version==bv)&(df.model=="somatic_burden")]
        if row.empty: continue
        row=row.iloc[0]
        yy=y+(0.16 if bv=="current" else -0.16)
        coef=float(row.burden_coef); lo=float(row.ci_lo); hi=float(row.ci_hi)
        sig=float(row.p_burden_given_umi)<0.05
        axB.plot([lo,hi],[yy,yy],color=c,lw=2,solid_capstyle="round",zorder=2)
        axB.plot([coef],[yy],mk,color=c,ms=7,mec="white",mew=1.1,zorder=3)
        if sig:
            axB.annotate("*",(coef,yy),textcoords="offset points",xytext=(0,2),
                         ha="center",va="bottom",color=c,fontsize=12,fontweight="bold")
    ycond.append(y); ylab2.append(cond_label(df[(df['sample']==samp)&(df["gt"]==gt)].iloc[0]))
    y+=1
axB.axvline(0,color="#444",ls="--",lw=1,zorder=1)
axB.set_yticks(ycond); axB.set_yticklabels(ylab2,fontsize=8.5)
axB.set_xlabel("somatic burden coefficient  (controlling for total UMI)")
axB.set_title("B   Somatic effect is not robust to burden version\n"
              "     (only 1 weak positive: DCIS2-gold current, p=0.029; * = p<0.05)",
              loc="left",fontsize=11)
legB=[Line2D([0],[0],color=c,lw=2,marker=mk,mec="white",label=lb) for c,mk,lb in VER.values()]
axB.legend(handles=legB,loc="lower left",frameon=False,fontsize=8.5,title="burden set version")
for sp in ("top","right"): axB.spines[sp].set_visible(False)
axB.grid(axis="x",color="#eee",zorder=0)
axB.set_xlim(-3.0,2.0)

fig.suptitle("Axis 2 — coverage-controlled spatial concordance (pre-dedup burden; CNV silver GT + DCIS2 pathology gold)",
             fontsize=12,y=1.02,x=0.02,ha="left")
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figures/axis2_cnv_consensus_forest_2026-07-13.{ext}",
                dpi=200,bbox_inches="tight")
print(f"Wrote {OUT}/figures/axis2_cnv_consensus_forest_2026-07-13.png / .pdf")
