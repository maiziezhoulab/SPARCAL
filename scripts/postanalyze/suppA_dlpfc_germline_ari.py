#!/usr/bin/env python
"""Supp A -- germline positive control (what spatial-RNA SNVs CAN do). DLPFC 12-section
ARI: SPARCAL's 1kG-defined, 250kb-binned germline representation ties GATK and beats
Strelka2, approaching the gene-expression upper reference. Data already computed
(clustering_benchmark project); this script only re-renders it to match this paper's
figure style/palette (source: data/dlpfc/clustering_benchmark/ari_boxplot.png already
exists but predates this palette).

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/suppA_dlpfc_germline_ari.py
Output: data/spatialsnv_reanalysis_2026-07-17/figures/suppA_dlpfc_germline_ari.{png,pdf}
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

SRC = "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/clustering_benchmark/ari_matrix_mean.csv"
OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/figures"

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
COLS = ["gene_expr", "defined_bin250kb", "gatk_bin250kb", "strelka2_bin250kb"]
LABELS = ["Gene expression\n(upper reference)", "SPARCAL germline\n(1kG-defined, 250kb bin)",
          "GATK\n(250kb bin)", "Strelka2\n(250kb bin)"]
COLORS = ["#898781", "#2a78d6", "#eda100", "#1baf7a"]

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(SRC)[COLS].dropna()
    print(f"n sections: {len(df)}")
    print(df.mean())

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    x = np.arange(len(COLS))
    bp = ax.boxplot([df[c].values for c in COLS], positions=x, widths=0.42,
                     patch_artist=True, showfliers=False, zorder=3,
                     medianprops=dict(color="white", lw=2),
                     boxprops=dict(lw=0), whiskerprops=dict(color=MUTED, lw=1.3),
                     capprops=dict(color=MUTED, lw=1.3))
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color); patch.set_alpha(0.85)

    rng = np.random.default_rng(0)
    for xi, c, color in zip(x, COLS, COLORS):
        jitter = rng.uniform(-0.12, 0.12, size=len(df))
        ax.scatter(xi + jitter, df[c].values, s=22, color=color, alpha=0.55,
                   edgecolor="white", linewidth=0.4, zorder=4)

    means = [df[c].mean() for c in COLS]
    for xi, m in zip(x, means):
        ax.text(xi, 0.06, f"mean\n{m:.3f}", ha="center", va="bottom", fontsize=8.3, color=INK,
                fontweight="medium")

    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
    ax.set_ylim(0, 0.65)
    ax.set_ylabel("ARI vs DLPFC cortical layers (12 sections)", fontsize=10, color=INK)
    ax.set_title("SPARCAL's germline representation ties GATK and beats Strelka2\n"
                 "— this is the positive control, not the somatic/UPV model output",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=12)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(f"{OUT}/suppA_dlpfc_germline_ari.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/suppA_dlpfc_germline_ari.pdf", bbox_inches="tight")
    print(f"WROTE: {OUT}/suppA_dlpfc_germline_ari.{{png,pdf}}")
