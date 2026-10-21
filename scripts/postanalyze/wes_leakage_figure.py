#!/usr/bin/env python3
"""
wes_leakage_figure.py  —  2026-08-07

Figure for PAPER_PLAN.md §3 Story C2 / §4.4 — Mutect2 germline leakage in the WES
"truth" set, and the allele-frequency evidence that the leak is common-SNP
carryover rather than a tumor-specific process.

REPLACES the retracted `COSMIC/wes_somatic_vs_sparcal_2026-07-13.png` panel A,
which plots the 976/811 (28%/31%) numbers diagnosed as an artifact on 2026-07-29
(naive single-point region query against a 1000G panel containing structural
variants returns span-overlapping records, not position matches). Panel B of that
old figure (where RNA-recovered WES-somatic land in SPARCAL's classes) is
unaffected by the correction and is NOT reproduced here.

INPUT (both written by scripts/postanalyze/somatic_evidence_package.py, 2026-07-29)
    data/somatic_evidence_2026-07-28/wes_leakage_af_stratified.csv
        one row per WES(Mutect2)-exome somatic SNP allele x {P4,P6}, with its
        1000G-panel category and panel AF. Categories:
          leaked_exact             CHROM/POS/REF/ALT all match a panel record
          near_miss_position_only  a panel record sits at that exact POS, but no
                                   allele match  -> the contrast group
          absent_from_panel        no panel record at that POS at all
    data/somatic_evidence_2026-07-28/wes_leakage_af_summary.csv
        (read only to cross-check the plotted %>=AF values; not plotted directly)

OUTPUT -> data/somatic_evidence_2026-07-28/figures/
    wes_germline_leakage_2026-08-07.png / .pdf
    wes_germline_leakage_2026-08-07_plotted_values.csv   table-view twin

Panels
    A   leakage composition per sample (stacked counts, leaked % direct-labeled)
    B   P4 - complementary ECDF of panel AF, leaked vs near-miss
    C   P6 - same

B/C are complementary ECDFs (y = % of sites with panel AF >= x) so the AF>=1%
figures quoted in the text are read directly off the curves at the x=0.01 rule.

Palette: slots 1-2 of the validated categorical default (blue #2a78d6 /
orange #eb6834); validated all-pairs, light mode, all six checks PASS.

Usage (env `snv_caller`, seconds, NOT sbatch):
    python scripts/postanalyze/wes_leakage_figure.py
Idempotent: outputs are fully overwritten on each run.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
SRC_DIR = os.path.join(PROJECT, "data", "somatic_evidence_2026-07-28")
OUT_DIR = os.path.join(SRC_DIR, "figures")
STEM = "wes_germline_leakage_2026-08-07"

SAMPLES = ["P4", "P6"]
AF_RULE = 0.01  # the "common variant" line quoted in the text

# ── palette (validated categorical slots 1-2 + chrome) ────────────────────────
LEAKED, NEARMISS = "#2a78d6", "#eb6834"
ABSENT = "#d9d8d1"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

CAT_LABEL = {
    "leaked_exact": "leaked — allele-exact match to 1000G",
    "near_miss_position_only": "near miss — same position, other allele",
    "absent_from_panel": "absent from 1000G panel",
}
CAT_SHORT = {"leaked_exact": "leaked", "near_miss_position_only": "near miss"}
X_LEFT = 1.5e-4  # left edge of the AF axis in B/C
CAT_COLOR = {"leaked_exact": LEAKED, "near_miss_position_only": NEARMISS,
             "absent_from_panel": ABSENT}


def ccdf(vals: np.ndarray, x_left: float):
    """Complementary ECDF for `plt.step(..., where='pre')`.

    Returns (x, y) with y = % of vals >= x. Under where='pre' the interval
    (x[i-1], x[i]] carries y[i], which is exactly the survival semantics: for
    x just above v_(i-1) the count is n-i, not n-i+1. (where='post' would hold
    each level one step too far right and overstate every point by one tie
    group — the AF>=1% readout would then not match the CSV.)
    """
    v = np.sort(np.asarray(vals, dtype=float))
    n = len(v)
    x = np.concatenate([[x_left], v])
    y = np.concatenate([[100.0], 100.0 * (n - np.arange(n)) / n])
    return x, y


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    leak = pd.read_csv(os.path.join(SRC_DIR, "wes_leakage_af_stratified.csv"))

    counts = (leak.groupby(["sample", "category"]).size()
              .unstack(fill_value=0).reindex(SAMPLES))
    totals = counts.sum(axis=1)

    plt.rcParams.update({
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(13.2, 4.3), gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})

    # ── Panel A — leakage composition ────────────────────────────────────────
    order = ["absent_from_panel", "near_miss_position_only", "leaked_exact"]
    x = np.arange(len(SAMPLES))
    bottom = np.zeros(len(SAMPLES))
    for cat in order:
        vals = counts[cat].to_numpy(dtype=float)
        axA.bar(x, vals, 0.5, bottom=bottom, color=CAT_COLOR[cat],
                label=CAT_LABEL[cat], zorder=3,
                linewidth=1.6, edgecolor=SURFACE)  # 2px surface gap between segments
        bottom += vals

    # labels sit ABOVE each bar — the leaked/near-miss segments are far too thin
    # to hold text, and side callouts collided with the neighbouring panel
    for i, s in enumerate(SAMPLES):
        n_leak, n_tot = int(counts.loc[s, "leaked_exact"]), int(totals[s])
        n_near = int(counts.loc[s, "near_miss_position_only"])
        axA.text(i, n_tot + 0.055 * totals.max(), f"{n_leak} leaked · {100*n_leak/n_tot:.1f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=LEAKED)
        axA.text(i, n_tot + 0.012 * totals.max(), f"+{n_near} near miss",
                 ha="center", va="bottom", fontsize=8.5, color=NEARMISS)

    axA.set_xticks(x)
    axA.set_xticklabels([f"{s}\nn = {int(totals[s]):,}" for s in SAMPLES],
                        fontsize=10, color=INK)
    axA.set_xlim(-0.6, len(SAMPLES) - 0.4)
    axA.set_ylim(0, totals.max() * 1.20)
    axA.set_ylabel("Mutect2 tumor/normal 'somatic' SNVs")
    axA.set_title("A · Leakage in the WES 'somatic' truth",
                  fontsize=10.5, fontweight="bold", loc="left", color=INK, pad=10)
    axA.legend(frameon=False, fontsize=8.0, loc="upper left",
               bbox_to_anchor=(-0.02, -0.22), labelcolor=INK2,
               handlelength=1.1, handleheight=1.0, borderpad=0, labelspacing=0.55)
    axA.spines[["top", "right"]].set_visible(False)
    axA.set_axisbelow(True)
    axA.yaxis.grid(True, color=GRID, lw=0.8, linestyle="-")

    # ── Panels B/C — AF distribution, leaked vs near miss ────────────────────
    plotted = []
    for ax, s, tag in [(axB, "P4", "B"), (axC, "P6", "C")]:
        for cat, col in [("leaked_exact", "panel_af_exact"),
                         ("near_miss_position_only", "panel_af_position_any_allele")]:
            sub = leak[(leak["sample"] == s) & (leak["category"] == cat)]
            vals = sub[col].dropna().astype(float).to_numpy()
            if not len(vals):
                continue
            xs, ys = ccdf(vals, X_LEFT)
            ax.step(xs, ys, where="pre", color=CAT_COLOR[cat], lw=2.0,
                    zorder=4, solid_capstyle="round")
            pct = 100 * (vals >= AF_RULE).mean()
            plotted.append({"sample": s, "category": cat, "n": len(sub),
                            "n_with_af": len(vals), "pct_af_ge_0.01": round(pct, 1),
                            "af_median": round(float(np.median(vals)), 5)})
            # Direct label at the AF>=1% rule, parked in axes-fraction space:
            # the leaked label above both curves (upper right), the near-miss
            # label below both (lower left), each with a thin leader to its
            # marker. Point-offset placement kept landing the label on top of a
            # curve or the x-axis ticks, because the two %s differ per sample.
            ax.plot([AF_RULE], [pct], "o", ms=8, color=CAT_COLOR[cat],
                    mec=SURFACE, mew=2, zorder=6)
            anchor = (0.50, 0.84) if cat == "leaked_exact" else (0.03, 0.11)
            ax.annotate(f"{pct:.0f}%  {CAT_SHORT[cat]}  (n={len(sub)})",
                        xy=(AF_RULE, pct), xytext=anchor,
                        textcoords="axes fraction",
                        fontsize=8.8, fontweight="bold", color=CAT_COLOR[cat],
                        va="center", ha="left", zorder=7,
                        arrowprops=dict(arrowstyle="-", color=CAT_COLOR[cat],
                                        lw=0.9, shrinkA=3, shrinkB=4))

        ax.axvline(AF_RULE, color=MUTED, lw=0.9, zorder=2)
        ax.annotate("AF = 1%", xy=(AF_RULE, 103), xytext=(-5, 0),
                    textcoords="offset points",
                    fontsize=8, color=MUTED, ha="right", va="top")
        ax.set_xscale("log")
        ax.set_xlim(X_LEFT, 1.0)
        ax.set_ylim(0, 108)
        ax.set_xlabel("1000 Genomes panel allele frequency")
        if tag == "B":
            ax.set_ylabel("% of sites with panel AF ≥ x")
        ax.set_title(f"{tag} · {s} — leaked sites are commoner alleles",
                     fontsize=10.5, fontweight="bold", loc="left", color=INK, pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, color=GRID, lw=0.8, linestyle="-")

    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.text(0.008, 0.055,
             "Allele-exact matching against the 1000 Genomes panel (hg19), autosomes, Mutect2 matched tumor/normal exome SNVs. 'Near miss' = a panel record at the same position but a\n"
             "different allele — the coordinate-only match, kept as the contrast group so the AF skew cannot be an artifact of position lookup. B/C plot only the sites carrying a panel AF\n"
             "(the leaked and near-miss groups); sites absent from the panel have no AF to plot.",
             fontsize=7.4, color=MUTED, ha="left", va="top", style="italic",
             linespacing=1.5)
    for ext in ("png", "pdf"):
        p = os.path.join(OUT_DIR, f"{STEM}.{ext}")
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor=SURFACE)
        print(f"WROTE {p}")

    # ── table-view twin ──────────────────────────────────────────────────────
    comp = counts.reset_index().melt(id_vars="sample", var_name="category",
                                     value_name="n")
    comp["pct_of_sample_total"] = [
        round(100 * r.n / totals[r.sample], 2) for r in comp.itertuples()]
    tbl = comp.merge(pd.DataFrame(plotted)[
        ["sample", "category", "pct_af_ge_0.01", "af_median"]],
        on=["sample", "category"], how="left")
    tbl = tbl.sort_values(["sample", "category"]).reset_index(drop=True)
    p = os.path.join(OUT_DIR, f"{STEM}_plotted_values.csv")
    tbl.to_csv(p, index=False)
    print(f"WROTE {p}")
    print(tbl.to_string(index=False))


if __name__ == "__main__":
    main()
