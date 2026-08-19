#!/usr/bin/env python
"""Figure 4 -- spatial-RNA evidence limits and a direct SPARCAL/SpatialSNV comparison.

The four panels distinguish shared platform limits from a method-specific result:
  (a) most SNV calls in SpatialSNV's results occur in a single spot (4 samples)
  (b) RNA covers ~1% of matched-WES Mutect2 alleles, both in and out of the
      Beagle/1KGP panel (P4/P6)
  (c) a three-set Venn shows the position-level overlap among SPARCAL,
      SpatialSNV, and matched-WES Mutect2 calls (P6)
  (d) a direct comparison shows common-1KGP contamination in each reported
      somatic callset and how SPARCAL routes SpatialSNV's leaked sites

Panel b is a platform/input mismatch, not a Mutect2 accuracy test. Panel d is
the method-specific classification result.

Run (env snv_caller, NOT sbatch):
  python scripts/postanalyze/fig2_platform_limit.py

To create the final Arial export:
  SPARCAL_ARIAL_FONT=/path/to/Arial.ttf python scripts/postanalyze/fig2_platform_limit.py

Output: SPARCAL_pnas_2026/figs/v2_2026-07-29/fig4_platform_limit[_preview].{png,pdf}
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Patch

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def configure_required_font():
    """Use real Arial when available; otherwise produce a named preview."""
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

BASE = "/data/maiziezhou_lab/leiy4/snv_calling/data/spatialsnv_reanalysis_2026-07-17"
OUT = f"{BASE}/figures"
PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
EVID_DIR = f"{PROJECT}/data/somatic_evidence_2026-07-28"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-07-29"
PAPER_FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v2_2026-07-29"

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SPARCAL_C = "#e34948"
SSNV_C = "#2a78d6"
SSNV_LIGHT = "#9ec4ec"
GRAY_LIGHT = "#e1e0d9"
ALLGENE_C = "#898781"
WES_C = "#7651a6"


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def panel_a(ax):
    df = pd.read_csv(f"{BASE}/ssnv_support_summary.csv").set_index("sample")
    samples = ["p4", "p6", "dcis1", "dcis2"]
    labels = ["P4", "P6", "DCIS1", "DCIS2"]
    single = df.loc[samples, "snv_in_1_spot_pct"].values
    multi = 100 - single
    x = np.arange(len(samples))
    ax.bar(x, single, width=0.6, color=SSNV_C, label="in 1 spot", edgecolor=INK,
           linewidth=0.7, zorder=3)
    ax.bar(x, multi, width=0.6, bottom=single, color=SSNV_LIGHT, label="in ≥2 spots",
           edgecolor=INK, linewidth=0.7, zorder=3)
    for i, v in enumerate(single):
        ax.text(i, v / 2, f"{v:.0f}%", ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of SpatialSNV calls", fontsize=9.5, color=INK)
    ax.set_title("a. Most calls in their results are single-spot\n(SpatialSNV callset)",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False, fontsize=8.5,
              handlelength=1.4, columnspacing=1.2)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def beagle_rna_coverage():
    """Return allele-exact WES/RNA coverage stratified by 1KGP membership."""
    accounting = pd.read_csv(f"{EVID_DIR}/wes_confirmed_full_accounting.csv")
    leakage = pd.read_csv(f"{EVID_DIR}/wes_leakage_af_stratified.csv")
    keys = ["sample", "chrom", "pos", "ref", "alt"]
    merged = leakage.merge(
        accounting[keys + ["rna_covered"]], on=keys, how="left", validate="one_to_one")
    if merged["rna_covered"].isna().any():
        raise ValueError("WES leakage rows did not all match the RNA-coverage accounting table")
    merged["beagle_group"] = np.where(
        merged["category"].eq("leaked_exact"), "In Beagle/1KGP", "Out of Beagle/1KGP")
    out = (merged.groupby(["sample", "beagle_group"], as_index=False)
           .agg(n_wes=("pos", "size"), n_rna_covered=("rna_covered", "sum")))
    out["pct_rna_covered"] = 100 * out["n_rna_covered"] / out["n_wes"]
    return out


def panel_b(ax, coverage):
    samples = ["P4", "P6"]
    groups = ["In Beagle/1KGP", "Out of Beagle/1KGP"]
    colors = {"In Beagle/1KGP": "#2a78d6", "Out of Beagle/1KGP": "#eb6834"}
    pale = {"In Beagle/1KGP": "#c9dcf4", "Out of Beagle/1KGP": "#f8d6c5"}
    x = np.arange(len(samples))
    width = 0.34
    for gi, group in enumerate(groups):
        sub = coverage[coverage.beagle_group.eq(group)].set_index("sample").loc[samples]
        xp = x + (gi - 0.5) * width
        ax.bar(xp, [100] * len(samples), width=width * 0.86, color=pale[group],
               edgecolor=INK, linewidth=0.8, zorder=2)
        ax.bar(xp, sub.pct_rna_covered, width=width * 0.86, color=colors[group],
               edgecolor=INK, linewidth=0.8, zorder=3)
        for xi, row in zip(xp, sub.itertuples()):
            ax.text(xi, 5.2,
                    f"{int(row.n_rna_covered)}/{int(row.n_wes):,}\n{row.pct_rna_covered:.2f}%",
                    ha="center", va="bottom", fontsize=7.5, color=INK, zorder=4)
    ax.set_xlim(-0.55, 1.55)
    ax.set_xticks(x); ax.set_xticklabels(samples, fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of exome-filtered WES Mutect2 SNP alleles", fontsize=9.5, color=INK)
    ax.set_title("b. We observed limited spatial-RNA coverage\nin and out of the Beagle/1KGP panel",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")
    ax.legend(handles=[Patch(facecolor=pale[g], edgecolor=INK, label=g) for g in groups],
              loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=2,
              frameon=False, fontsize=7.7, handlelength=1.2, columnspacing=0.9)
    ax.text(0.99, 0.94, "saturated sub-bar = RNA-covered portion",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=MUTED)
    style_axes(ax)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0); ax.set_axisbelow(True)


def panel_c(ax):
    df = pd.read_csv(f"{BASE}/p6_crossmethod_jaccard.csv")
    pair = {(r.set_A, r.set_B): int(r.intersection) for r in df.itertuples()}
    n_a = int(df.loc[df.set_A.eq("SPARCAL"), "n_A"].iloc[0])
    n_b = int(df.loc[df.set_B.eq("SpatialSNV"), "n_B"].iloc[0])
    n_c = int(df.loc[df.set_B.eq("WES_somatic"), "n_B"].iloc[0])
    n_ab = pair[("SPARCAL", "SpatialSNV")]
    n_ac = pair[("SPARCAL", "WES_somatic")]
    n_bc = pair[("SpatialSNV", "WES_somatic")]
    n_abc = pair[("SPARCAL", "SpatialSNV∩WES_somatic (3-way)")]
    compartments = {
        "a": n_a - n_ab - n_ac + n_abc,
        "b": n_b - n_ab - n_bc + n_abc,
        "c": n_c - n_ac - n_bc + n_abc,
        "ab": n_ab - n_abc,
        "ac": n_ac - n_abc,
        "bc": n_bc - n_abc,
        "abc": n_abc,
    }

    circles = [((0.39, 0.58), 0.27, SPARCAL_C),
               ((0.61, 0.58), 0.27, SSNV_C),
               ((0.50, 0.39), 0.27, WES_C)]
    for center, radius, color in circles:
        ax.add_patch(Circle(center, radius, facecolor=color, edgecolor=INK,
                            linewidth=1.0, alpha=0.28))
    positions = {
        "a": (0.25, 0.65), "b": (0.75, 0.65), "c": (0.50, 0.22),
        "ab": (0.50, 0.69), "ac": (0.38, 0.42), "bc": (0.62, 0.42),
        "abc": (0.50, 0.50),
    }
    for key, (xp, yp) in positions.items():
        ax.text(xp, yp, f"{compartments[key]:,}", ha="center", va="center",
                fontsize=8.2 if key != "abc" else 9.0,
                fontweight="bold" if key == "abc" else "normal", color=INK)
    ax.text(0.08, 0.92, f"SPARCAL somatic\nn = {n_a:,}", ha="left", va="top",
            fontsize=8.2, color=SPARCAL_C, fontweight="bold")
    ax.text(0.92, 0.92, f"SpatialSNV\nn = {n_b:,}", ha="right", va="top",
            fontsize=8.2, color=SSNV_C, fontweight="bold")
    ax.text(0.50, 0.02, f"Broad matched-WES Mutect2 positions\nn = {n_c:,}",
            ha="center", va="bottom", fontsize=8.2, color=WES_C, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("c. We compared the position-level overlap\namong three P6 callsets",
                  fontsize=10.5, color=INK, fontweight="bold", loc="left")


def panel_d(ax):
    quality_dirs = sorted(
        d for d in glob.glob(f"{PROJECT}/data/spatialsnv_callset_quality_*")
        if os.path.exists(os.path.join(d, "germline_summary.csv")))
    if not quality_dirs:
        raise FileNotFoundError("No completed SpatialSNV callset-quality analysis")
    germ = pd.read_csv(os.path.join(quality_dirs[-1], "germline_summary.csv"))
    samples = ["P4", "P6", "DCIS1", "DCIS2"]
    x = np.arange(len(samples))
    ax.axis("off")
    ax.set_title("d. We compared 1KGP routing in SPARCAL and SpatialSNV",
                 fontsize=10.5, color=INK, fontweight="bold", loc="left")
    left = ax.inset_axes([0.00, 0.04, 0.52, 0.82])
    right = ax.inset_axes([0.61, 0.04, 0.39, 0.82])

    width = 0.34
    for i, (method, color, label) in enumerate([
            ("SpatialSNV", SSNV_C, "SpatialSNV"),
            ("SPARCAL_somatic", SPARCAL_C, "SPARCAL")]):
        sub = germ[germ.method.eq(method)].set_index("sample").loc[samples]
        xp = x + (i - 0.5) * width
        left.bar(xp, sub.pct_in_1KGP, width=width * 0.88, color=color,
                 edgecolor=INK, linewidth=0.7, label=label, zorder=3)
        for xi, pct, n in zip(xp, sub.pct_in_1KGP, sub.n_in_1KGP):
            label_y = pct + 0.10 if pct >= 0.05 else 0.10
            txt = f"{pct:.2f}" if pct >= 0.05 else f"n={int(n)}"
            left.text(xi, label_y, txt, ha="center", va="bottom", fontsize=5.8, color=INK)
    left.set_xticks(x); left.set_xticklabels(samples, rotation=35, ha="right", fontsize=6.7)
    left.set_ylim(0, 5.5)
    left.set_ylabel("somatic callset matching 1KGP (%)", fontsize=7.0)
    left.set_title("Reported somatic calls", fontsize=8.0, color=INK)
    left.legend(frameon=False, fontsize=6.3, loc="upper left", ncol=1,
                handlelength=1.0, borderaxespad=0.2)

    ssnv = germ[germ.method.eq("SpatialSNV")].set_index("sample").loc[samples]
    routed = 100 * ssnv.n_also_in_sparcal_germline / ssnv.n_in_1KGP
    right.bar(x, routed, width=0.58, color=SPARCAL_C, edgecolor=INK,
              linewidth=0.7, zorder=3)
    for xi, pct in zip(x, routed):
        right.text(xi, pct + 2.0, f"{pct:.0f}%", ha="center", va="bottom",
                   fontsize=6.4, fontweight="bold", color=INK)
    right.set_xticks(x); right.set_xticklabels(samples, rotation=35, ha="right", fontsize=6.7)
    right.set_ylim(0, 100)
    right.set_ylabel("leaked sites routed to germline (%)", fontsize=7.0)
    right.set_title("SPARCAL routing of SpatialSNV matches", fontsize=8.0, color=INK)

    for inner in (left, right):
        style_axes(inner)
        inner.tick_params(axis="y", labelsize=6.5)
        inner.grid(axis="y", color=GRID, lw=0.7, zorder=0)
        inner.set_axisbelow(True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    os.makedirs(DERIVED_DIR, exist_ok=True)
    coverage = beagle_rna_coverage()
    coverage.to_csv(f"{DERIVED_DIR}/fig4b_beagle_rna_coverage.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 9.4))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1], coverage)
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    fig.suptitle("We compared spatial-RNA evidence limits and population-panel routing",
                 fontsize=13, color=INK, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.subplots_adjust(hspace=0.55, wspace=0.32)
    stem = "fig4_platform_limit" if HAS_ARIAL else "fig4_platform_limit_preview"
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial is unavailable; writing Nimbus Sans preview only. "
              "The manuscript asset is not being overwritten.")
    png = os.path.join(PAPER_FIG_DIR, f"{stem}.png")
    pdf = os.path.join(PAPER_FIG_DIR, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"WROTE: {png}")
    print(f"WROTE: {pdf}")
