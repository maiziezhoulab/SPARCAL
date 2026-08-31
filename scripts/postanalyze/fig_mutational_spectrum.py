#!/usr/bin/env python
"""fig_mutational_spectrum (2026-08-27) -- SPARCAL's four spatial-RNA variant
classes (germline / UPV / somatic / unresolved) have different substitution
signatures, and the two failure modes for somatic calls in P4/P6 versus DCIS1/2
look nothing alike.

Panels
  a. 6-channel substitution spectrum, grouped bars, all 4 samples x 4 classes.
     Germline is structured in every sample (C>T and T>C dominate, as expected
     for a real germline/deamination-shaped spectrum). P4/P6 somatic is FLAT
     (close to uniform across the 6 channels -- consistent with residual
     sequencing/mapping noise, not a mutational process). DCIS somatic is
     T>C-dominated (~52%), the classic signature of RNA A-to-I editing being
     read out as a DNA substitution.
  b. A>G+T>C fraction per class per sample (a single scalar summary of the
     same table), with the germline baseline drawn as a reference band so the
     UPV/somatic/unresolved excess above germline is visible directly.
  c. RNA-editing catalogue overlap (% of a class's SNVs that coincide with a
     literature editing-site catalogue): germline ~3.1-3.5% in every sample
     (background rate), P4/P6 somatic ~8.6% (elevated but modest), DCIS1/2
     somatic 26-31% (dominated by editing sites).
  d. DCIS1 somatic 96-channel profile (SigProfiler-style), to show directly
     that the top contexts are essentially all *[T>C]* -- i.e. the elevated
     T>C/editing-overlap numbers in a/b/c are not an artifact of 6-channel
     collapsing.

DO NOT MERGE THE P4/P6 AND DCIS SOMATIC STORIES. They fail in different ways:
P4/P6 somatic looks like unstructured residual noise (flat spectrum, modest
editing overlap); DCIS somatic looks like RNA-editing readthrough (T>C peak,
high editing overlap). See the dossier for the "must not claim" list.

Run:
  python scripts/postanalyze/fig_mutational_spectrum.py

Outputs
  data/paper_figs_2026-08-27/fig_mutational_spectrum_panel_source.csv (a+b+c, tidy)
  data/paper_figs_2026-08-27/fig_mutational_spectrum_panel_d_dcis1_somatic_96ch.csv
  SPARCAL_pnas_2026/figs/v7_2026-08-27/fig_mutational_spectrum[_preview].{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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

# Project-standard variant-class colors (matches plot_priority_figures_v5.py and
# snv_calling/CLAUDE.md "Variant Categories & Naming": germline=blue, UPV=purple,
# somatic=red). "unresolved" reuses the established green from fig4ab_cosmic_xmhc_v4.py.
CLASS_ORDER = ["germline", "UPV", "somatic", "unresolved"]
CLASS_COLOR = {
    "germline": (SSNV_C, SSNV_L),
    "UPV": (MONO_C, MONO_L),
    "somatic": (SPARCAL_C, SPARCAL_L),
    "unresolved": ("#3f9b5c", "#c3e6d0"),
}
SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
CHANNELS = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.0)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def load():
    six = pd.read_csv(f"{SRC_DIR}/spectrum_6channel.csv")
    ninety_six = pd.read_csv(f"{SRC_DIR}/spectrum_96channel.csv")
    edit = pd.read_csv(f"{SRC_DIR}/editing_overlap.csv")
    return six, ninety_six, edit


# ---------------------------------------------------------------------------
# Panel a -- 6-channel spectrum, one subplot per sample
# ---------------------------------------------------------------------------
def panel_a(axes, six):
    ch_rows = six[six.channel.isin(CHANNELS)]
    width = 0.19
    x = np.arange(len(CHANNELS))
    for ax, sample in zip(axes, SAMPLES):
        sub = ch_rows[ch_rows["sample"].eq(sample)]
        for i, cls in enumerate(CLASS_ORDER):
            css = sub[sub["class"].eq(cls)].set_index("channel").reindex(CHANNELS)
            dark, _ = CLASS_COLOR[cls]
            xp = x + (i - 1.5) * width
            ax.bar(xp, css.fraction.values, width=width * 0.92, color=dark, edgecolor=INK,
                  linewidth=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(CHANNELS, fontsize=7.4, rotation=0)
        ax.set_ylim(0, 0.58)
        ax.set_title(sample, fontsize=9.0, color=MUTED, loc="center", pad=3)  # data-subset label
        style_axes(ax)
    axes[0].set_ylabel("fraction of class SNVs", fontsize=8.6, color=INK)
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    handles = [Patch(facecolor=CLASS_COLOR[c][0], edgecolor=INK, label=c) for c in CLASS_ORDER]
    axes[0].legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.30), ncol=4,
                  frameon=False, fontsize=7.6, handlelength=1.1, columnspacing=1.1,
                  borderaxespad=0.0)
    return ch_rows


# ---------------------------------------------------------------------------
# Panel b -- A>G + T>C fraction per class per sample
# ---------------------------------------------------------------------------
def panel_b(ax, six):
    ag_tc = six[six.channel.eq("A>G_plus_T>C_fraction_of_total")].copy()
    ag_tc = ag_tc.rename(columns={"fraction": "value"})
    germ_mean = ag_tc.loc[ag_tc["class"].eq("germline"), "value"].mean()
    x = np.arange(len(SAMPLES))
    width = 0.20
    for i, cls in enumerate(CLASS_ORDER):
        sub = ag_tc[ag_tc["class"].eq(cls)].set_index("sample").reindex(SAMPLES)
        dark, _ = CLASS_COLOR[cls]
        xp = x + (i - 1.5) * width
        ax.bar(xp, sub.value.values, width=width * 0.92, color=dark, edgecolor=INK, linewidth=0.5,
              zorder=3)
    ax.axhline(germ_mean, color=CLASS_COLOR["germline"][0], lw=1.3, linestyle=(0, (5, 2)), zorder=2)
    ax.text(len(SAMPLES) - 0.55, germ_mean + 0.012, f"germline baseline ({germ_mean:.2f})",
            fontsize=6.8, color=CLASS_COLOR["germline"][0], ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8.4)
    ax.set_ylabel("A>G + T>C fraction of class SNVs", fontsize=8.6, color=INK)
    ax.set_ylim(0, 0.62)
    style_axes(ax)
    return ag_tc[["sample", "class", "value"]]


# ---------------------------------------------------------------------------
# Panel c -- RNA-editing catalogue overlap
# ---------------------------------------------------------------------------
def panel_c(ax, edit):
    x = np.arange(len(SAMPLES))
    width = 0.20
    for i, cls in enumerate(CLASS_ORDER):
        sub = edit[edit["class"].eq(cls)].set_index("sample").reindex(SAMPLES)
        dark, _ = CLASS_COLOR[cls]
        xp = x + (i - 1.5) * width
        ax.bar(xp, sub.pct_overlap_SComatic_AllEditingSites.values, width=width * 0.92, color=dark,
              edgecolor=INK, linewidth=0.5, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8.4)
    ax.set_ylabel("% overlap with RNA-editing catalogue\n(SComatic AllEditingSites)", fontsize=8.4,
                 color=INK, linespacing=1.25)
    style_axes(ax)


# ---------------------------------------------------------------------------
# Panel d -- DCIS1 somatic 96-channel profile
# ---------------------------------------------------------------------------
CH96_COLORS = {"C>A": "#5ec2e8", "C>G": "#0b0b0b", "C>T": "#e34948",
               "T>A": "#b6b4ae", "T>C": "#3f9b5c", "T>G": "#f0b8c4"}


def parse_context(ctx):
    left, rest = ctx.split("[")
    channel, right = rest.split("]")
    return channel, f"{left}[{channel[0]}]{right}"


def panel_d(ax, ninety_six):
    sub = ninety_six[ninety_six["sample"].eq("DCIS1") & ninety_six["class"].eq("somatic")].copy()
    sub["channel"] = sub["context"].str.extract(r"\[(.\>.)\]")
    order = []
    for ch in CHANNELS:
        ctxs = sorted(sub.loc[sub.channel.eq(ch), "context"])
        order.extend(ctxs)
    sub = sub.set_index("context").loc[order].reset_index()
    colors = [CH96_COLORS[c] for c in sub.channel]
    x = np.arange(len(sub))
    ax.bar(x, sub.fraction.values, width=0.82, color=colors, edgecolor="none", zorder=3)
    # block labels
    pos = 0
    for ch in CHANNELS:
        n = (sub.channel == ch).sum()
        ax.text(pos + n / 2 - 0.5, sub.fraction.max() * 1.10, ch, ha="center", va="bottom",
                fontsize=7.6, color=CH96_COLORS[ch], fontweight="bold")
        ax.axvspan(pos - 0.5, pos + n - 0.5, color=CH96_COLORS[ch], alpha=0.05, zorder=0)
        pos += n
    top3 = sub.nlargest(3, "fraction")
    for _, row in top3.iterrows():
        xi = list(sub.context).index(row.context)
        ax.annotate(row.context, xy=(xi, row.fraction), xytext=(xi, row.fraction + sub.fraction.max() * 0.22),
                    fontsize=6.2, color=INK, ha="center", va="bottom", rotation=90,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xlim(-1, len(sub))
    ax.set_ylim(0, sub.fraction.max() * 1.75)
    ax.set_xticks([])
    ax.set_ylabel("fraction of DCIS1 somatic SNVs", fontsize=8.4, color=INK)
    style_axes(ax)
    n_tc = int((sub.channel == "T>C").sum())
    tc_share = sub.loc[sub.channel.eq("T>C"), "fraction"].sum()
    ax.text(0.02, 0.985, f"T>C contexts: {tc_share*100:.0f}% of somatic SNVs ({n_tc}/96 contexts)",
            transform=ax.transAxes, fontsize=7.0, color=CH96_COLORS["T>C"], fontweight="bold",
            va="top", ha="left")
    return sub[["context", "channel", "count", "fraction"]]


def main():
    six, ninety_six, edit = load()

    fig = plt.figure(figsize=(14.5, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.05], hspace=0.55, wspace=0.28,
                          left=0.055, right=0.985, top=0.90, bottom=0.07)
    axes_a = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_b = fig.add_subplot(gs[1, 0:2])
    ax_c = fig.add_subplot(gs[1, 2])
    ax_d = fig.add_subplot(gs[1, 3])

    panel_a(axes_a, six)
    b_src = panel_b(ax_b, six)
    panel_c(ax_c, edit)
    d_src = panel_d(ax_d, ninety_six)

    tidy = six[six.channel.isin(CHANNELS)][["sample", "class", "channel", "fraction"]].copy()
    tidy["panel"] = "a"
    b_src2 = b_src.rename(columns={"value": "fraction"}).copy()
    b_src2["channel"] = "A>G_plus_T>C_fraction_of_total"
    b_src2["panel"] = "b"
    edit_src = edit[["sample", "class", "pct_overlap_SComatic_AllEditingSites"]].copy()
    edit_src["panel"] = "c"
    combined = pd.concat([tidy, b_src2, edit_src], ignore_index=True, sort=False)
    combined.to_csv(f"{DERIVED_DIR}/fig_mutational_spectrum_panel_source.csv", index=False)
    d_src.to_csv(f"{DERIVED_DIR}/fig_mutational_spectrum_panel_d_dcis1_somatic_96ch.csv", index=False)

    axes_a[0].text(-0.28, 1.42, "a", transform=axes_a[0].transAxes, fontsize=13, fontweight="bold",
                  color=INK, va="top", ha="left")
    ax_b.text(-0.11, 1.10, "b", transform=ax_b.transAxes, fontsize=13, fontweight="bold",
             color=INK, va="top", ha="left")
    ax_c.text(-0.22, 1.10, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold",
             color=INK, va="top", ha="left")
    ax_d.text(-0.14, 1.10, "d", transform=ax_d.transAxes, fontsize=13, fontweight="bold",
             color=INK, va="top", ha="left")

    stem = "fig_mutational_spectrum" if HAS_ARIAL else "fig_mutational_spectrum_preview"
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
