#!/usr/bin/env python
"""Fig 3 -- Conditional-test forest plot (even-handed: SPARCAL included alongside
every other method). coef +/- 95% CI of z-scored burden|UMI, per method x sample.

IMPORTANT interpretive note carried into the figure: joint coefficients are unstable
under burden~UMI collinearity (r=0.80-0.98 for several predictors here) -- a large
coefficient is NOT the same as a large, trustworthy effect. Standalone AUC (shown as
a text annotation per row) is the robust companion metric and is what the text
should be quoted from; this plot exists to show that most SNV-caller coefficients
that look "significant" are collinearity-inflated, not to rank effect sizes at face
value.

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/fig3_forest_plot.py
Output: data/spatialsnv_reanalysis_2026-07-17/figures/fig3_forest_plot.{png,pdf}
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
COLOR = {
    "SPARCAL_somatic": "#2a78d6",
    "SpatialSNV_effective": "#eb6834",
    "Monopogen_burden": "#4a3aa7",
    "STMut_burden": "#e34948",
    "TSK_sig_expr": "#008300",
    "epithelial_sig_expr": "#008300",
}
LABEL = {
    "SPARCAL_somatic": "SPARCAL (somatic)",
    "SpatialSNV_effective": "SpatialSNV (effective)",
    "Monopogen_burden": "Monopogen (germline)",
    "STMut_burden": "STMut (WES-gold sites)",
    "TSK_sig_expr": "Expression (TSK sig)",
    "epithelial_sig_expr": "Expression (epithelial sig)",
}
ALONE_AUC = {
    ("P6", "SPARCAL_somatic"): 0.631, ("P6", "SpatialSNV_effective"): 0.582,
    ("P6", "Monopogen_burden"): 0.570, ("P6", "TSK_sig_expr"): 0.807,
    ("P6", "STMut_burden"): 0.432,
    ("DCIS2", "SPARCAL_somatic"): 0.858, ("DCIS2", "SpatialSNV_effective"): 0.907,
    ("DCIS2", "epithelial_sig_expr"): 0.946,
}
ROW_ORDER = {
    "P6": ["TSK_sig_expr", "SPARCAL_somatic", "SpatialSNV_effective", "Monopogen_burden", "STMut_burden"],
    "DCIS2": ["epithelial_sig_expr", "SPARCAL_somatic", "SpatialSNV_effective"],
}

if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(f"{BASE}/fig3_forest_data.csv")

    rows = []
    for sample in ["DCIS2", "P6"]:
        for pred in ROW_ORDER[sample]:
            r = df[(df["sample"] == sample) & (df.predictor == pred)].iloc[0]
            rows.append(r)
    order = list(reversed(rows))  # top-to-bottom plotting order

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    y = np.arange(len(order))
    ax.axvline(0, color=GRID, lw=1.4, ls="-", zorder=1)

    ci_low_all, ci_high_all = [], []
    for yi, r in zip(y, order):
        color = COLOR[r.predictor]
        lo, hi = r.coef - 1.96 * r.se, r.coef + 1.96 * r.se
        ci_low_all.append(lo); ci_high_all.append(hi)
        ax.plot([lo, hi], [yi, yi], color=color, lw=2.2, zorder=3,
                solid_capstyle="round")
        ax.scatter([r.coef], [yi], s=70, color=color, zorder=4, edgecolor="white", linewidth=1.0)
        auc = ALONE_AUC[(r["sample"], r.predictor)]
        cap = " (SE capped, conservative)" if r.se_is_capped else ""
        ax.text(hi + 0.18, yi, f"alone AUC={auc:.3f}{cap}", va="center", fontsize=8.3, color=MUTED)

    labels = [f"{LABEL[r.predictor]}" for r in order]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9.5, color=INK)

    # sample group brackets on the left
    p6_idx = [i for i, r in enumerate(order) if r["sample"] == "P6"]
    d2_idx = [i for i, r in enumerate(order) if r["sample"] == "DCIS2"]
    xmin = min(ci_low_all) - 1.3
    for idx, name in [(p6_idx, "P6"), (d2_idx, "DCIS2")]:
        ax.text(xmin, np.mean(idx), name, rotation=90, va="center", ha="center",
                fontsize=10.5, color=INK, fontweight="bold")

    ax.axhline(min(p6_idx) - 0.5 if p6_idx and d2_idx else -1, color=GRID, lw=1, zorder=1)

    ax.set_xlim(xmin - 0.4, max(ci_high_all) + 2.6)
    ax.set_xlabel("Coefficient (z-scored burden | log UMI), 95% CI", fontsize=10, color=INK)
    ax.set_title("Even-handed conditional test: coefficient magnitude is unreliable under\n"
                 "burden~UMI collinearity — standalone AUC (right) is the metric to trust",
                 fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=14)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_forest_plot.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig3_forest_plot.pdf", bbox_inches="tight")
    print(f"WROTE: {OUT}/fig3_forest_plot.{{png,pdf}}")
