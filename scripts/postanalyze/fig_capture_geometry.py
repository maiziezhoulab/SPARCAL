#!/usr/bin/env python
"""fig_capture_geometry (2026-08-27) -- where WES-confirmed somatic calls are lost
on the way to being observable in 3'-captured spatial RNA, for P4 and P6.

The pipeline stage order (matches the `stage` column in waterfall_{p4,p6}.csv):
  1_wes_somatic_total          -- WES Mutect2 somatic calls (100% denominator)
  1b_gene_assigned_any         -- call falls inside an annotated gene
  2_gene_expressed_in_section  -- that gene is expressed (>=1 count) in the section
  3_within_3prime_capture_window -- call falls within N bp of the annotated 3' end
                                    (N = 300 / 500 / 1000, the window swept here)
  4_ge1_read_pooled_bam        -- >=1 read observed at the site in the pooled BAM
  5_alt_allele_present         -- the ALT allele is observed at all

Story this figure must make visually obvious: **the collapse is at stage 3 (the
3' capture window), not at stage 2 (expression) or stage 4/5 (depth/allele
sampling).** Gene assignment and expression only lose ~1/3 of calls; the window
step alone loses 95-99% of what remains. Doubling the window from 300bp to
500bp (and again to 1000bp) roughly doubles the pass rate at stage 3 -- the
signature of a geometric distance cutoff, not a biological expression effect.

CAVEAT THAT MUST TRAVEL WITH THIS FIGURE: the "3' capture window" is defined
by this analysis's own distance-from-annotated-3'-end rule; it is an assumption
about where a poly-A-primed capture platform's read density concentrates, not a
manufacturer-published capture-efficiency curve. The qualitative point (window
tightening is a geometric, not biological, bottleneck) generalizes to any
3'-capture platform; the exact percentages are specific to this window
definition and to P4/P6.

Run:
  python scripts/postanalyze/fig_capture_geometry.py

Outputs
  data/paper_figs_2026-08-27/fig_capture_geometry_source.csv
  SPARCAL_pnas_2026/figs/v7_2026-08-27/fig_capture_geometry[_preview].{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D

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
SRC_DIR = f"{PROJECT}/data/mutational_spectrum_2026-08-27"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-08-27"
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v7_2026-08-27"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"

STAGE_ORDER = ["1_wes_somatic_total", "1b_gene_assigned_any", "2_gene_expressed_in_section",
              "3_within_3prime_capture_window", "4_ge1_read_pooled_bam", "5_alt_allele_present"]
STAGE_LABEL = {
    "1_wes_somatic_total": "WES\nsomatic\ntotal",
    "1b_gene_assigned_any": "gene\nassigned",
    "2_gene_expressed_in_section": "gene\nexpressed",
    "3_within_3prime_capture_window": "within 3'\ncapture\nwindow",
    "4_ge1_read_pooled_bam": "≥1 read\n(pooled BAM)",
    "5_alt_allele_present": "ALT\npresent",
}
WINDOWS = [300, 500, 1000]
# Sequential shades of SSNV blue: darkest = tightest (300bp), lightest = widest (1000bp).
WIN_COLOR = {}
base = np.array(to_rgb(SSNV_C))
white = np.array([1, 1, 1])
for w, frac in zip(WINDOWS, [0.0, 0.32, 0.62]):
    WIN_COLOR[w] = tuple(base * (1 - frac) + white * frac)
SAMPLE_MARKER = {"P4": "o", "P6": "s"}


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.2)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0, which="both")
    ax.set_axisbelow(True)


def load():
    p4 = pd.read_csv(f"{SRC_DIR}/waterfall_p4.csv")
    p6 = pd.read_csv(f"{SRC_DIR}/waterfall_p6.csv")
    df = pd.concat([p4, p6], ignore_index=True)
    return df


def panel(ax, df, sample, show_ylabel):
    sub = df[df["sample"].eq(sample)]
    x = np.arange(len(STAGE_ORDER))
    for w in WINDOWS:
        wsub = sub[sub.window_bp.eq(w)].set_index("stage").reindex(STAGE_ORDER)
        color = WIN_COLOR[w]
        ax.plot(x, wsub.pct_of_total.values, marker="o", markersize=5.5, color=color,
                markeredgecolor=INK, markeredgewidth=0.5, linewidth=2.0, zorder=4,
                label=f"{w} bp window")
    ax.set_yscale("log")
    ax.set_ylim(0.15, 140)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGE_ORDER], fontsize=7.6, linespacing=1.15)
    ax.set_xlim(-0.4, len(STAGE_ORDER) - 0.6)
    if show_ylabel:
        ax.set_ylabel("% of WES somatic total\n(log scale)", fontsize=9.0, color=INK, linespacing=1.3)
    else:
        ax.tick_params(labelleft=False)
    style_axes(ax)
    ax.set_title(sample, fontsize=10.5, color=MUTED, loc="center", pad=6)  # data-subset label

    # Shade the collapse step (stage index 2->3) to make the location of the drop
    # visually unmissable.
    ax.axvspan(1.5, 2.5, color=SPARCAL_L, alpha=0.28, zorder=0)
    ax.annotate("the collapse is here", xy=(2, ax.get_ylim()[1] * 0.55), xytext=(3.35, ax.get_ylim()[1] * 0.55),
               fontsize=8.0, color=SPARCAL_C, fontweight="bold", ha="left", va="center",
               arrowprops=dict(arrowstyle="->", color=SPARCAL_C, lw=1.3))

    # Annotate the stage-3 values for each window (the key spec numbers).
    for w in WINDOWS:
        row = sub[(sub.window_bp.eq(w)) & (sub.stage.eq("3_within_3prime_capture_window"))]
        if not row.empty:
            v = row.pct_of_total.values[0]
            ax.text(2, v * 1.35, f"{v:.2f}%", ha="center", va="bottom", fontsize=7.0,
                    color=WIN_COLOR[w], fontweight="bold")
    return sub


def main():
    df = load()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.1), gridspec_kw=dict(
        left=0.10, right=0.985, top=0.86, bottom=0.20, wspace=0.10))

    panel(axes[0], df, "P4", True)
    panel(axes[1], df, "P6", False)

    handles = [Line2D([0], [0], marker="o", color=WIN_COLOR[w], markeredgecolor=INK,
                      markersize=6, linewidth=2.0, label=f"{w} bp window") for w in WINDOWS]
    axes[0].legend(handles=handles, loc="upper right", frameon=False, fontsize=7.6,
                  handlelength=1.6, borderaxespad=0.2)

    fig.text(0.10, 0.965, "3' capture window is an analysis-defined distance-from-3'-end rule, "
                          "not a manufacturer capture-efficiency curve (see dossier).",
             fontsize=7.0, color=MUTED, ha="left", va="top")

    df.to_csv(f"{DERIVED_DIR}/fig_capture_geometry_source.csv", index=False)

    axes[0].text(-0.16, 1.14, "a", transform=axes[0].transAxes, fontsize=13, fontweight="bold",
                color=INK, va="top", ha="left")
    axes[1].text(-0.10, 1.14, "b", transform=axes[1].transAxes, fontsize=13, fontweight="bold",
                color=INK, va="top", ha="left")

    stem = "fig_capture_geometry" if HAS_ARIAL else "fig_capture_geometry_preview"
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
