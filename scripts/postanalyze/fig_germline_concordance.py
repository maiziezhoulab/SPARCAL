#!/usr/bin/env python
"""fig_germline_concordance (2026-08-27) -- how well SPARCAL's germline pathway
(a 1KGP lookup-and-refine, NOT a general variant caller) recovers WES-truth
germline SNPs from RNA, how accurate its genotypes are, and whether somatic
calls are contaminated by leaked common alleles.

Panels
  a. three-way sensitivity vs RNA depth bin (SPARCAL / GATK / Strelka2), split
     into 1KGP-panel-defined loci (left) and de novo / non-1KGP loci (right).
     SPARCAL is competitive with GATK/Strelka2 on panel loci at every depth and
     collapses to near zero on de novo loci at every depth -- because its
     germline pathway only refines loci it already expects from the 1KGP panel;
     it does not discover new germline sites.
  b. genotype accuracy vs RNA depth bin (SPARCAL only), split the same way
     (panel-defined vs de novo), per sample.
  c. leaked common-allele outcomes as stacked bars per sample: of common
     (1KGP-panel) WES alleles that also physically overlap the somatic
     candidate space ("leaked" positions), how many are (i) not detected at
     all, (ii) correctly routed to germline, or misassigned to UPV/somatic/
     unresolved. Headline: 0 of 29,882 leaked alleles were misassigned somatic.
CLAIM BOUNDARY THAT MUST TRAVEL WITH THIS FIGURE: panels a/b describe SPARCAL's
1KGP lookup-and-refine germline pathway, not a general de novo germline caller.
Do not present the panel-allele sensitivity/accuracy numbers as general germline
calling performance -- the near-zero de novo sensitivity is the other half of
the same result, not a separate failure.

Run:
  python scripts/postanalyze/fig_germline_concordance.py

Outputs
  data/paper_figs_2026-08-27/fig_germline_concordance_panel_{a,b,c}_source.csv
  SPARCAL_pnas_2026/figs/v7_2026-08-27/fig_germline_concordance[_preview].{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local_dir = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/fonts/arial"
    local = font_manager.findSystemFonts(fontpaths=[local_dir]) if os.path.isdir(local_dir) else []
    explicit_family = (font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
                       if explicit and os.path.dirname(explicit) else [])
    candidates = ([explicit] + explicit_family) if explicit else local + font_manager.findSystemFonts()
    matches = []
    for path in candidates:
        if not path or path in matches or not os.path.exists(path):
            continue
        try:
            family = font_manager.FontProperties(fname=path).get_name()
        except Exception:
            continue
        if family.casefold() == "arial":
            matches.append(path)
    if matches:
        for path in matches:
            font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Arial"
        return True, matches[0]
    plt.rcParams["font.family"] = "Nimbus Sans"
    return False, None


HAS_ARIAL, ARIAL_PATH = configure_required_font()

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
SRC_DIR = f"{PROJECT}/data/germline_and_contrasts_2026-08-27"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-08-27"
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v7_2026-08-27"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"
GATK_C = "#45a66a"   # established project convention (make_selected_modality_performance.py)

CALLER_COLOR = {"SPARCAL": SPARCAL_C, "GATK": GATK_C, "Strelka2": SSNV_C}
DEPTH_BINS = ["0", "1-3", "4-9", "10-29", "30+"]
SAMPLES = ["P4", "P6"]
SAMPLE_LS = {"P4": "-", "P6": "--"}
SAMPLE_MARKER = {"P4": "o", "P6": "s"}

# Project-standard variant-class colors (shared with fig_mutational_spectrum.py).
CLASS_COLOR = {
    "not detected": "#b6b4ae",
    "germline": SSNV_C,
    "UPV": MONO_C,
    "somatic": SPARCAL_C,
    "unresolved": "#3f9b5c",
}
CLASS_ORDER = ["not detected", "germline", "UPV", "somatic", "unresolved"]
ALL_SAMPLES4 = ["P4", "P6", "DCIS1", "DCIS2"]


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.0)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def load():
    three_way = pd.read_csv(f"{SRC_DIR}/concordance_three_way.csv", dtype={"rna_depth_bin": str})
    by_depth = pd.read_csv(f"{SRC_DIR}/concordance_by_depth.csv", dtype={"rna_depth_bin": str})
    leaked = pd.read_csv(f"{SRC_DIR}/leaked_allele_confusion.csv")
    return three_way, by_depth, leaked


# ---------------------------------------------------------------------------
# Panel a -- three-way sensitivity, panel-defined vs de novo
# ---------------------------------------------------------------------------
def panel_a(axes, three_way):
    x = np.arange(len(DEPTH_BINS))
    panels = [("defined_1kgp", "1KGP-panel-defined loci"), ("non_1kgp", "de novo (non-1KGP) loci")]
    rows_out = []
    for ax, (panel_key, panel_title) in zip(axes, panels):
        for caller in ["GATK", "Strelka2", "SPARCAL"]:
            for sample in SAMPLES:
                sub = three_way[three_way.caller.eq(caller) & three_way["sample"].eq(sample)
                                & three_way.panel.eq(panel_key)].set_index("rna_depth_bin").reindex(DEPTH_BINS)
                ax.plot(x, sub.sensitivity.values, marker=SAMPLE_MARKER[sample],
                        linestyle=SAMPLE_LS[sample], color=CALLER_COLOR[caller], markersize=5,
                        markeredgecolor=INK, markeredgewidth=0.4, linewidth=1.6, zorder=3, alpha=0.95)
                for db, v in zip(DEPTH_BINS, sub.sensitivity.values):
                    rows_out.append(dict(panel=panel_key, caller=caller, sample=sample,
                                         rna_depth_bin=db, sensitivity=v))
        ax.set_xticks(x)
        ax.set_xticklabels(DEPTH_BINS, fontsize=8.0)
        ax.set_xlabel("RNA depth bin", fontsize=8.6, color=INK)
        ax.set_title(panel_title, fontsize=9.0, color=MUTED, loc="center", pad=4)
        style_axes(ax)
    axes[0].set_ylabel("sensitivity vs WES truth", fontsize=9.0, color=INK)
    axes[0].set_ylim(0, 1.0)
    axes[1].set_ylim(0, 0.40)
    caller_handles = [Line2D([0], [0], color=CALLER_COLOR[c], lw=2.0, label=c)
                      for c in ["SPARCAL", "GATK", "Strelka2"]]
    sample_handles = [Line2D([0], [0], color=INK, marker=SAMPLE_MARKER[s], linestyle=SAMPLE_LS[s],
                             markersize=5, label=s) for s in SAMPLES]
    axes[0].legend(handles=caller_handles + sample_handles, loc="upper left", frameon=False,
                  fontsize=7.0, handlelength=1.6, ncol=1, borderaxespad=0.2, labelspacing=0.3)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Panel b -- genotype accuracy by depth (SPARCAL only)
# ---------------------------------------------------------------------------
def panel_b(axes, by_depth):
    x = np.arange(len(DEPTH_BINS))
    panel_style = {"defined_1kgp": (SPARCAL_C, "panel-defined"), "non_1kgp": (MUTED, "de novo")}
    rows_out = []
    for ax, sample in zip(axes, SAMPLES):
        for panel_key, (color, label) in panel_style.items():
            sub = by_depth[by_depth["sample"].eq(sample) & by_depth.panel.eq(panel_key)
                          ].set_index("rna_depth_bin").reindex(DEPTH_BINS)
            ax.plot(x, sub.gt_accuracy.values, marker="o", color=color, markersize=5,
                    markeredgecolor=INK, markeredgewidth=0.4, linewidth=1.8, zorder=3, label=label)
            for db, v, n in zip(DEPTH_BINS, sub.gt_accuracy.values, sub.n_gt_evaluable.values):
                rows_out.append(dict(sample=sample, panel=panel_key, rna_depth_bin=db,
                                     gt_accuracy=v, n_gt_evaluable=n))
        ax.set_xticks(x)
        ax.set_xticklabels(DEPTH_BINS, fontsize=8.0)
        ax.set_xlabel("RNA depth bin", fontsize=8.6, color=INK)
        ax.set_ylim(0, 1.05)
        ax.set_title(sample, fontsize=9.0, color=MUTED, loc="center", pad=4)
        style_axes(ax)
    axes[0].set_ylabel("SPARCAL genotype accuracy", fontsize=8.8, color=INK)
    axes[0].legend(loc="lower right", frameon=False, fontsize=7.4, handlelength=1.4, borderaxespad=0.2)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Panel c -- leaked-allele outcomes, stacked bars
# ---------------------------------------------------------------------------
def panel_c(ax, leaked):
    x = np.arange(len(ALL_SAMPLES4))
    bottoms = np.zeros(len(ALL_SAMPLES4))
    for cls in CLASS_ORDER:
        vals = []
        for s in ALL_SAMPLES4:
            row = leaked[(leaked["sample"].eq(s)) & (leaked.outcome.eq(cls))]
            vals.append(row["n"].values[0] if len(row) else 0)
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, bottom=bottoms, width=0.62, color=CLASS_COLOR[cls], edgecolor=INK,
              linewidth=0.5, zorder=3, label=cls)
        bottoms += vals
    totals = leaked.groupby("sample")["n_total_leaked"].first().reindex(ALL_SAMPLES4)
    for xi, tot in zip(x, totals.values):
        ax.text(xi, tot * 1.02, f"n = {int(tot):,}", ha="center", va="bottom", fontsize=7.4, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(ALL_SAMPLES4, fontsize=8.6)
    ax.set_ylabel("leaked common (1KGP) alleles\noverlapping somatic candidate space", fontsize=8.4,
                 color=INK, linespacing=1.25)
    style_axes(ax)
    n_somatic_misassigned = int(leaked.loc[leaked.outcome.eq("somatic"), "n"].sum())
    n_total = int(totals.sum())
    ax.set_ylim(0, bottoms.max() * 1.16)
    ax.text(0.02, 0.985, f"{n_somatic_misassigned} of {n_total:,} leaked alleles misassigned somatic",
            transform=ax.transAxes, fontsize=7.6, color=SPARCAL_C, fontweight="bold", va="top", ha="left")
    handles = [Patch(facecolor=CLASS_COLOR[c], edgecolor=INK, label=c) for c in CLASS_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
             fontsize=7.2, handlelength=1.1, borderaxespad=0.2, labelspacing=0.4)


def main():
    three_way, by_depth, leaked = load()

    fig = plt.figure(figsize=(14.5, 9.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.55, wspace=0.42,
                          left=0.055, right=0.97, top=0.90, bottom=0.08)
    ax_a = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax_b = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 3])]
    ax_c = fig.add_subplot(gs[1, 0:4])

    a_src = panel_a(ax_a, three_way)
    b_src = panel_b(ax_b, by_depth)
    panel_c(ax_c, leaked)

    a_src.to_csv(f"{DERIVED_DIR}/fig_germline_concordance_panel_a_source.csv", index=False)
    b_src.to_csv(f"{DERIVED_DIR}/fig_germline_concordance_panel_b_source.csv", index=False)
    leaked.to_csv(f"{DERIVED_DIR}/fig_germline_concordance_panel_c_source.csv", index=False)

    ax_a[0].text(-0.22, 1.18, "a", transform=ax_a[0].transAxes, fontsize=13, fontweight="bold",
                color=INK, va="top", ha="left")
    ax_b[0].text(-0.20, 1.18, "b", transform=ax_b[0].transAxes, fontsize=13, fontweight="bold",
                color=INK, va="top", ha="left")
    ax_c.text(-0.09, 1.12, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold",
             color=INK, va="top", ha="left")

    fig.text(0.055, 0.975, "SPARCAL's germline pathway is a 1KGP lookup-and-refine, not a "
                          "general de novo germline caller (panels a-b).", fontsize=7.4, color=MUTED,
             ha="left", va="top")

    stem = "fig_germline_concordance" if HAS_ARIAL else "fig_germline_concordance_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans preview only.")
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{stem}.{ext}")
        fig.savefig(path, dpi=300)
        print(f"[fig] wrote {path}")


if __name__ == "__main__":
    main()
