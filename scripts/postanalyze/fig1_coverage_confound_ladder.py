#!/usr/bin/env python
"""Fig 1 -- The coverage confound across methods (MAIN figure).

Two panels, one per pathology-ground-truth sample (DCIS2 gold, P6 leading-edge mask):
grouped bars of STANDALONE AUC at predicting the tumor region, per predictor. A dashed
line marks the coverage-only (total UMI) AUC and a dotted line marks chance (0.5).
Standalone AUC is used (not the joint coefficient) because joint coefficients are
unstable under burden~UMI collinearity (0.8-0.98) -- see PAPER_PLAN.md guardrail.

Message: an expression positive control clears the coverage line on both samples;
no SNV-burden method (SPARCAL, SpatialSNV, Monopogen, STMut) does.

Color mapping is FIXED per entity across both panels (identity, not rank):
  SPARCAL = blue, SpatialSNV = orange, Expression = green,
  Monopogen = violet, STMut = red  (Monopogen/STMut: P6 only, no DCIS2 matrix exists)

Run (env snv_caller, NOT sbatch): python scripts/postanalyze/fig1_coverage_confound_ladder.py
Output: data/spatialsnv_reanalysis_2026-07-17/figures/fig1_coverage_confound_ladder.{png,pdf}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

OUT = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17/figures"

COLOR = {
    "SPARCAL": "#2a78d6",
    "SpatialSNV": "#eb6834",
    "Expression": "#008300",
    "Monopogen": "#4a3aa7",
    "STMut": "#e34948",
}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
COVERAGE_LINE = "#52514e"

# (label, entity_key, alone_auc)
dcis2 = [
    ("Expression\n(epithelial sig)", "Expression", 0.946),
    ("SpatialSNV\n(effective)", "SpatialSNV", 0.907),
    ("SPARCAL\n(somatic)", "SPARCAL", 0.858),
]
dcis2_coverage = 0.913
dcis2_n = "789 spots (249 tumor / 540 normal)"

p6 = [
    ("Expression\n(TSK sig)", "Expression", 0.807),
    ("SPARCAL\n(somatic)", "SPARCAL", 0.631),
    ("SpatialSNV\n(effective)", "SpatialSNV", 0.582),
    ("Monopogen\n(germline)", "Monopogen", 0.570),
    ("STMut\n(WES sites)", "STMut", 0.432),
]
p6_coverage = 0.664
p6_n = "3,719 spots (1,504 tumor / 2,215 normal)"


def draw_panel(ax, data, coverage_auc, title, subtitle):
    data = sorted(data, key=lambda r: -r[2])
    labels = [d[0] for d in data]
    colors = [COLOR[d[1]] for d in data]
    values = [d[2] for d in data]
    x = range(len(data))

    right_edge = len(data) - 1 + 0.55
    ax.set_xlim(-0.7, len(data) - 1 + 0.7)

    ax.axhline(0.5, color=GRID, lw=1.2, ls=":", zorder=1)
    ax.text(-0.6, 0.5, "chance", color=MUTED, fontsize=8, va="bottom", ha="left", zorder=6)

    ax.axhline(coverage_auc, color=COVERAGE_LINE, lw=1.6, ls="--", zorder=2)
    ax.text(right_edge, coverage_auc + 0.012, f"coverage (UMI) = {coverage_auc:.3f}",
            color=COVERAGE_LINE, fontsize=8.5, va="bottom", ha="right", fontweight="medium", zorder=6)

    bars = ax.bar(x, values, width=0.6, color=colors, zorder=3,
                   edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, values):
        # inside the bar, near the top -- never collides with the coverage/chance lines
        ax.text(b.get_x() + b.get_width() / 2, v - 0.022, f"{v:.3f}",
                ha="center", va="top", fontsize=9.5, color="white", fontweight="bold",
                zorder=5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8, color=INK)
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Standalone AUC (predicting tumor region)", fontsize=9.5, color=INK)
    ax.set_title(title, fontsize=11.5, color=INK, fontweight="bold", loc="left", pad=36)
    ax.text(0, 1.045, subtitle, transform=ax.transAxes, fontsize=8.5, color=MUTED, va="bottom")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    draw_panel(axes[0], dcis2, dcis2_coverage,
               "DCIS2 — pathology gold standard", dcis2_n)
    draw_panel(axes[1], p6, p6_coverage,
               "P6 — pathology leading-edge mask", p6_n)

    fig.suptitle("Expression beats coverage; no spatial-RNA somatic/germline burden method does",
                 fontsize=12.5, color=INK, y=0.99, x=0.02, ha="left", fontweight="bold")
    fig.subplots_adjust(top=0.78, bottom=0.12, wspace=0.28)
    fig.savefig(f"{OUT}/fig1_coverage_confound_ladder.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/fig1_coverage_confound_ladder.pdf", bbox_inches="tight")
    print(f"WROTE: {OUT}/fig1_coverage_confound_ladder.{{png,pdf}}")
