#!/usr/bin/env python
"""Supp C -- 3' bias and expressed-gene bias of P6 spatial-SNV calls (Step 9).
(a) relative position within the gene body (1.0 = 3' end) vs a gene-length-matched
null, SPARCAL and SpatialSNV. (b) per-gene call count vs per-gene total UMI --
the gene-level restatement of the coverage confound (bigger effect than 3a).

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/suppC_3prime_expression_bias.py
Output: data/spatialsnv_reanalysis_2026-07-17/figures/suppC_3prime_expression_bias.{png,pdf}
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
SPARCAL_C = "#2a78d6"
SSNV_C = "#eb6834"

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    relpos = pd.read_csv(f"{BASE}/p6_3prime_bias_relpos.csv")
    expr = pd.read_csv(f"{BASE}/p6_gene_expression_bias.csv").set_index("gene")["total_umi"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # panel a: relpos density, SPARCAL and SpatialSNV vs their null
    ax = axes[0]
    bins = np.linspace(0, 1, 41)
    for name, color in [("SPARCAL_somatic", SPARCAL_C), ("SpatialSNV", SSNV_C)]:
        d = relpos[relpos.set == name]
        ax.hist(d.relpos, bins=bins, density=True, histtype="step", color=color, lw=2.2,
                 label=f"{name.replace('_somatic','')} calls (mean={d.relpos.mean():.3f})", zorder=3)
        ax.hist(d.null_relpos, bins=bins, density=True, histtype="step", color=color, lw=1.3,
                 ls=":", label=f"{name.replace('_somatic','')} null (mean={d.null_relpos.mean():.3f})",
                 zorder=2)
    ax.set_xlabel("Relative position in gene body (0 = 5' / TSS, 1 = 3' end)", fontsize=9.5, color=INK)
    ax.set_ylabel("Density", fontsize=9.5, color=INK)
    ax.set_title("a. Sharp pileup at the exact 3′ end vs a\ngene-length-matched null (otherwise ~uniform)",
                  fontsize=10.8, color=INK, fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.6)

    # panel b: per-gene call count vs per-gene expression
    ax2 = axes[1]
    for name, color in [("SPARCAL_somatic", SPARCAL_C), ("SpatialSNV", SSNV_C)]:
        gcount = relpos[relpos.set == name].groupby("gene").size()
        joined = pd.DataFrame({"n_calls": gcount}).join(expr, how="left").dropna()
        ax2.scatter(np.log1p(joined.total_umi), np.log1p(joined.n_calls), s=10, alpha=0.35,
                    color=color, edgecolor="none", zorder=3,
                    label=name.replace("_somatic", ""))
    ax2.set_xlabel("log(1 + total UMI per gene)", fontsize=9.5, color=INK)
    ax2.set_ylabel("log(1 + #calls per gene)", fontsize=9.5, color=INK)
    ax2.set_title("b. Calls per gene scale with\ngene expression (bigger effect)",
                   fontsize=10.8, color=INK, fontweight="bold", loc="left")
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, markerscale=2)
    ax2.text(0.98, 0.97, "Spearman:\nSPARCAL 0.22\nSpatialSNV 0.26\n(both p≈0)",
             transform=ax2.transAxes, ha="right", va="top", fontsize=8, color=MUTED,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2))

    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

    fig.suptitle("3′-capture and expression bias in P6 spatial-SNV calls (Step 9)",
                 fontsize=12.5, color=INK, fontweight="bold", x=0.02, ha="left", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/suppC_3prime_expression_bias.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/suppC_3prime_expression_bias.pdf", bbox_inches="tight")
    print(f"WROTE: {OUT}/suppC_3prime_expression_bias.{{png,pdf}}")
