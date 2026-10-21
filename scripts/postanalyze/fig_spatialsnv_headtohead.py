#!/usr/bin/env python
"""Main-text head-to-head figure: SPARCAL vs the published SpatialSNV callsets.

WHY THIS FIGURE HAS TO EXIST.  Before it, the manuscript contained ZERO results
where SPARCAL and SpatialSNV are compared directly and SPARCAL comes out ahead:
SpatialSNV appeared in the platform-limit figure (illustrating a limit the two
methods SHARE) and in the region-detection figure (where they beat us, 0.671 vs
0.612).  Once region detection moved to SI, their only remaining main-text
appearance was the shared-limit panel.  The introduction promises a design
advantage, so some Results section has to cash it.  This is that section.

WHAT IT SHOWS, and what it deliberately does not.

  (a) COMMON-GERMLINE CONTAMINATION.  Fraction of each delivered somatic callset
      that is an allele-exact match to a biallelic 1000 Genomes SNV.  SPARCAL is
      ~0 BY CONSTRUCTION -- panel variants are routed to the germline class before
      the spatial cascade runs -- so the panel is captioned as a design
      consequence, not a contest won.  DCIS2's single leak in 25,154 is drawn and
      labelled rather than rounded to zero: "0 by construction" is false as an
      absolute and must not be shown as one.

  (b) THE CONSTRUCTIVE HALF, which is the actual argument.  Of the panel variants
      SpatialSNV retained as somatic, what fraction did SPARCAL see in the SAME
      TISSUE and route to germline?  64-82%.  This is what turns a definitional
      zero into a measured behavioural difference: both methods had access to
      these positions, and they disposed of them differently.

  (c) CALLSET OVERLAP, as context and as an honest limit.  The two methods agree
      on 2.7-5.0% of their union (Jaccard).  Drawn on a log axis because the
      partition sizes span two orders of magnitude and a linear axis would render
      the concordant slice invisible.

NOT IN THIS FIGURE, on purpose:
  * The Mutect2 retained-FILTER breakdown (48.8-58.0% non-PASS).  Both
    `clustered_events` and `haplotype` are DNA-calibrated and fire readily at
    high-expression RNA loci where clustered variation is legitimate, so
    retaining them may be a defensible choice on their part.  It is callset
    composition, not an error rate, and putting it beside a contamination
    panel would read as a second accuracy claim.  It stays in the CSV.
  * COSMIC catalogue membership. It is reported for all callers on the same
    basis elsewhere as descriptive corroboration, not variant-level validation.
    Panel (c) reports only the set arithmetic, which is model-free.

Colour: SPARCAL red / SpatialSNV blue, both already locked project series
colours (fig5_cosmic_cascade.py).  Red-blue is the CVD-safest available pair,
and every bar is additionally identified by its x-axis label, so identity is
never carried by hue alone.

INPUTS (read, never hardcoded):
  data/spatialsnv_callset_quality_<date>/germline_summary.csv
  data/spatialsnv_concordance_2026-08-06/concordance_group_sizes.csv

OUTPUT:
  SPARCAL_pnas_2026/figs/v2_2026-07-29/fig_spatialsnv_headtohead.{pdf,png}
  data/paper_figs_2026-07-29/fig_spatialsnv_headtohead_source.csv

Run (env snv_caller, seconds):
  python scripts/postanalyze/fig_spatialsnv_headtohead.py
"""
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
CONC_DIR = os.path.join(PROJECT, "data", "spatialsnv_concordance_2026-08-06")
DERIVED = os.path.join(PROJECT, "data", "paper_figs_2026-07-29")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v2_2026-07-29"

SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]      # locked sample order

SPARCAL_C = "#e34948"      # locked somatic red
SSNV_C = "#2a78d6"         # locked blue
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def latest_quality_dir() -> str:
    """Newest run that actually has results.

    spatialsnv_callset_quality.py creates its dated output directory up front and
    fills it at the end, so a run in flight leaves an empty directory that sorts
    newest. Requiring the CSV avoids silently picking it (or, worse, half of it)."""
    dirs = sorted(d for d in glob.glob(
        os.path.join(PROJECT, "data", "spatialsnv_callset_quality_*"))
        if os.path.exists(os.path.join(d, "germline_summary.csv")))
    if not dirs:
        raise SystemExit("no completed spatialsnv_callset_quality_* run found")
    return dirs[-1]


def main():
    qdir = latest_quality_dir()
    print(f"[source] {qdir}")
    germ = pd.read_csv(os.path.join(qdir, "germline_summary.csv"))
    sizes = pd.read_csv(os.path.join(CONC_DIR, "concordance_group_sizes.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    x = np.arange(len(SAMPLES))

    # ---------------- (a) contamination rate ------------------------------
    ax = axes[0]
    width = 0.36
    for i, (method, colour, label) in enumerate(
            [("SpatialSNV", SSNV_C, "SpatialSNV"),
             ("SPARCAL_somatic", SPARCAL_C, "SPARCAL")]):
        sub = germ[germ.method == method].set_index("sample").loc[SAMPLES]
        xp = x + (i - 0.5) * width
        ax.bar(xp, sub.pct_in_1KGP, width=width * 0.9, color=colour, label=label,
               edgecolor="white", linewidth=0.5, zorder=3)
        for xi, v, n in zip(xp, sub.pct_in_1KGP, sub.n_in_1KGP):
            if v >= 0.05:
                ax.text(xi, v + 0.09, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.0, color=INK)
            else:
                # SPARCAL's bars are invisible at this scale, which is the point --
                # but an unlabelled gap reads as "no data", so the count is written
                # in. DCIS2's single leak is drawn as n=1, never rounded to zero.
                ax.text(xi, 0.09, f"n={int(n)}", ha="center", va="bottom",
                        fontsize=5.6, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8)
    ax.set_ylabel("callset that is a common\n1000G variant (%)", fontsize=7.8)
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=6.4, frameon=False, loc="upper left", handlelength=1.1,
              handletextpad=0.5, borderaxespad=0.2)
    ax.set_title("a", fontsize=10, fontweight="bold", loc="left", x=-0.20, y=1.02)

    # ---------------- (b) SPARCAL's disposal of the same positions ---------
    ax = axes[1]
    sub = germ[germ.method == "SpatialSNV"].set_index("sample").loc[SAMPLES]
    frac = 100 * sub.n_also_in_sparcal_germline / sub.n_in_1KGP
    ax.bar(x, frac, width=0.52, color=SPARCAL_C, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, v, n, d in zip(x, frac, sub.n_also_in_sparcal_germline, sub.n_in_1KGP):
        ax.text(xi, v + 1.4, f"{v:.0f}%", ha="center", va="bottom", fontsize=6.6,
                color=INK, fontweight="bold")
        ax.text(xi, 3, f"{int(n):,}/{int(d):,}", ha="center", va="bottom",
                fontsize=5.4, color="white", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8)
    ax.set_ylabel("of SpatialSNV's leaked sites,\nSPARCAL routes to germline (%)", fontsize=7.8)
    ax.set_ylim(0, 100)
    ax.set_title("b", fontsize=10, fontweight="bold", loc="left", x=-0.20, y=1.02)

    # ---------------- (c) callset overlap ---------------------------------
    ax = axes[2]
    s = sizes.set_index("sample").loc[SAMPLES]
    width = 0.26
    parts = [("n_sparcal_private", SPARCAL_C, "SPARCAL only"),
             ("n_concordant", "#6f4fa8", "both"),
             ("n_spatialsnv_private", SSNV_C, "SpatialSNV only")]
    for i, (col, colour, label) in enumerate(parts):
        ax.bar(x + (i - 1) * width, s[col], width=width * 0.9, color=colour,
               label=label, edgecolor="white", linewidth=0.5, zorder=3)
    for xi, j in zip(x, s.jaccard):
        ax.text(xi, 4.2e5, f"J={j:.3f}", ha="center", va="bottom", fontsize=6.0, color=INK)
    ax.set_yscale("log")
    # Headroom above the tallest bar (~2.6e5) so the legend has somewhere to sit
    # that is not on top of the DCIS bars.
    ax.set_ylim(1e3, 4e7)
    ax.set_xticks(x)
    ax.set_xticklabels(SAMPLES, fontsize=8)
    ax.set_ylabel("variants (log scale)", fontsize=7.8)
    ax.legend(fontsize=6.2, frameon=False, loc="upper center", ncol=3,
              handlelength=1.0, handletextpad=0.4, columnspacing=1.0,
              borderaxespad=0.15)
    ax.set_title("c", fontsize=10, fontweight="bold", loc="left", x=-0.20, y=1.02)

    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(MUTED)
        ax.spines["bottom"].set_color(MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.42)

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(DERIVED, exist_ok=True)
    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"fig_spatialsnv_headtohead.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"wrote {out}")

    src = germ.merge(sizes[["sample", "n_concordant", "n_sparcal_private",
                            "n_spatialsnv_private", "jaccard"]], on="sample", how="left")
    out_csv = os.path.join(DERIVED, "fig_spatialsnv_headtohead_source.csv")
    src.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
