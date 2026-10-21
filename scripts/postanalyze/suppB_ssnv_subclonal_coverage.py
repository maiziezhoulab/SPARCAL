#!/usr/bin/env python
"""Supp B -- SpatialSNV's published "subclonal" structure is coverage-dominated
(DCIS1/DCIS2). SNV-window PC1 tracks total UMI almost perfectly; the KMeans-3
"subclone" partition is mostly explained by coverage strata (eta^2); and SNV PC1 does
not out-predict UMI at copyKAT malignancy.

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/suppB_ssnv_subclonal_coverage.py
Output: data/spatialsnv_reanalysis_2026-07-17/figures/suppB_ssnv_subclonal_coverage.{png,pdf}
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

BASE = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"
OUT = f"{BASE}/figures"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
DCIS1_C = "#eb6834"
DCIS2_C = "#4a3aa7"

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(f"{BASE}/ssnv_step3_subclonal_coverage.csv").set_index("sample")

    metrics = [
        ("corr_PC1_logumi", "corr(SNV-window\nPC1, log UMI)"),
        ("kmeans3_eta2_UMI", "KMeans-3 \"subclone\"\nη²(UMI)"),
        ("copykat_AUC_from_UMI", "copyKAT malignancy\nAUC from UMI alone"),
        ("copykat_AUC_from_SNV_PC1", "copyKAT malignancy\nAUC from SNV PC1 alone"),
    ]
    samples = ["dcis1", "dcis2"]
    colors = {"dcis1": DCIS1_C, "dcis2": DCIS2_C}

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    x = np.arange(len(metrics))
    width = 0.32
    for i, s in enumerate(samples):
        vals = [df.loc[s, m] for m, _ in metrics]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, color=colors[s], zorder=3,
                      label=s.upper())
        for xi, v in zip(x + offset, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.3, color=INK)

    ax.set_xticks(x); ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Value", fontsize=10, color=INK)
    ax.set_title("SpatialSNV's \"subclonal\" trajectory is coverage-dominated (DCIS1/DCIS2)\n"
                 "SNV-PC1 tracks UMI almost 1:1, and does not out-predict UMI at malignancy",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=9.5)
    ax.axhline(1.0, color=GRID, lw=1, ls=":", zorder=1)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

    fig.text(0.01, -0.02,
              "SNV_PC1 coef given UMI: DCIS1 −2.01 (p=0.0023, negative = collinearity-suppression "
              "artifact), DCIS2 +0.55 (p=0.080, ns)", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(f"{OUT}/suppB_ssnv_subclonal_coverage.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/suppB_ssnv_subclonal_coverage.pdf", bbox_inches="tight")
    print(f"WROTE: {OUT}/suppB_ssnv_subclonal_coverage.{{png,pdf}}")
