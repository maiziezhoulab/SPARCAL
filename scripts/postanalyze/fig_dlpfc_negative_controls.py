#!/usr/bin/env python
"""fig_dlpfc_negative_controls (2026-08-27, updated 2026-08-28) -- the P0-1 DLPFC
negative-control battery, now with a density-matched fifth control closing the
density confound in the original detection_only comparison.

Reads the aggregated negative-control clustering results (already computed by
`aggregate_dlpfc_negative_controls.py` for the original four controls and
`aggregate_density_matched_control.py` for the new fifth) and draws the figure
that decides whether "panel-locus matrix clustering decodes cortical layers"
survives five label-free controls that share the matrix's shape (and, for the
newest one, also its density) but destroy the allele/locus signal in different
ways:

  detection_only         -- binary locus-detected/not, no allele information.
                             CONFOUNDED WITH DENSITY: 3.5-4.3x denser (nonzero
                             fraction) and 5.4-7.9x more signal per spot (mean
                             row sum) than the real matrix, across all 12
                             sections (see fig-dlpfc-negative-controls.md's
                             "density confound" section and panel d below) --
                             its high ARI conflates "gene detection instead of
                             variant detection" with "much more data."
  detection_downsampled   -- (2026-08-28) detection_only subsampled per spot so
                             its nonzero-bin count AND total count exactly
                             match the real matrix's own per-spot values (see
                             build_detection_downsampled.py for the exact
                             method). Isolates the representation question
                             (gene- vs variant-detection) from density.
  allele_permuted         -- real detected loci, REF/ALT identity permuted
                             within spot
  smoothed_random         -- random detection pattern with the section's own
                             spatial autocorrelation structure imposed
  coverage_only           -- total UMI/read depth only (no locus identity at
                             all); degenerate in every run (effective rank 2 <
                             7 clusters)

Panels
  a. per-section mean ARI for each control, with the 10-run spread (individual
     points), grouped by the three donors (four serial sections each).
     coverage_only is annotated as DEGENERATE, never plotted as ARI=0.
  b. the comparison that matters: each control's cohort distribution against
     three CROSS-RUN reference points from the paper's own published
     representations (spatially-augmented 250 kb, 1KGP-only 250 kb, gene
     expression) -- explicitly labelled as not measured in this batch.
  c. donor-level means (n=3) beside section-level means (n=12) for every
     control, to make the pseudoreplication gap visible (referee finding C7).
  d. (2026-08-28) THE DENSITY CONFOUND, made visible directly: nonzero fraction
     and mean row sum (log scale, 12 sections/modality) for the real matrix and
     every control, including detection_downsampled -- shows detection_only's
     density inflation and detection_downsampled's exact match to the real
     matrix, by construction.

SOURCE OF THE THREE CROSS-RUN REFERENCE NUMBERS (panel b dashed lines) --
these are NOT recomputed here, they are the existing 12-section column means
already on disk from the main-text clustering benchmark:
  data/dlpfc/clustering_benchmark/ari_matrix_mean.csv
    column somtop25_bin250kb  -> mean 0.3500  (spatially-augmented 250 kb)
    column defined_bin250kb   -> mean 0.3626  (1KGP-only 250 kb)
    column gene_expr          -> mean 0.4116  (gene expression)
  (verified by direct column-mean computation from that file, 2026-08-27)

Run (env: base python3 with matplotlib/pandas/numpy; no GPU needed):
  python scripts/postanalyze/fig_dlpfc_negative_controls.py

Outputs
  data/paper_figs_2026-08-27/fig_dlpfc_negative_controls_panel_{a,c,d}_source.csv
  SPARCAL_pnas_2026/figs/v7_2026-08-27/fig_dlpfc_negative_controls[_preview].{png,pdf}
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
SRC_DIR = f"{PROJECT}/data/dlpfc_negative_controls_2026-08-27"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-08-27"
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v7_2026-08-27"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# The 12-section column means already published for the main-text clustering
# benchmark -- read once here, at import time, so the numbers are traceable to a
# concrete file rather than typed in as literals.
REF_TABLE = f"{PROJECT}/data/dlpfc/clustering_benchmark/ari_matrix_mean.csv"
_ref = pd.read_csv(REF_TABLE)
REAL_REFS = {
    "SPARCAL spat. aug. 250kb": float(_ref["somtop25_bin250kb"].mean()),
    "SPARCAL 1KGP-only 250kb": float(_ref["defined_bin250kb"].mean()),
    "Gene expression": float(_ref["gene_expr"].mean()),
}

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C, MONO_L = "#4a3aa7", "#c5bfe6"
SSNV_C, SSNV_L = "#2a78d6", "#bcd6f2"
WES_C, WES_L = "#7651a6", "#d5c8e6"

# Control-modality palette. These are label-free negative controls, not methods,
# so they intentionally avoid the reserved SPARCAL/Monopogen/SpatialSNV/WES colors.
# detection_downsampled (added 2026-08-28) is detection_only's density-matched sibling
# -- see fig-dlpfc-negative-controls.md's density-confound section -- given a distinct
# teal so it is never visually confused with detection_only's near-black.
CTRL_ORDER = ["detection_only", "detection_downsampled", "smoothed_random", "allele_permuted",
              "coverage_only"]
CTRL_LABEL = {
    "detection_only": "detection-\nonly",
    "detection_downsampled": "detection-\ndownsampled",
    "smoothed_random": "smoothed\nrandom",
    "allele_permuted": "allele-\npermuted",
    "coverage_only": "coverage-\nonly",
}
CTRL_SHORT = {
    "detection_only": "detection-only",
    "detection_downsampled": "detection-downsampled\n(density-matched)",
    "smoothed_random": "smoothed random",
    "allele_permuted": "allele-permuted",
    "coverage_only": "coverage-only",
}
CTRL_COLOR = {
    "detection_only": INK,
    "detection_downsampled": "#2e8b6f",  # teal -- density-matched sibling of detection_only
    "smoothed_random": SSNV_C,
    "allele_permuted": MUTED,
    "coverage_only": "#c94f2f",  # degenerate -- warm/alert tone, never implies a fitted ARI
}

DONOR = {}
for s in ("151507", "151508", "151509", "151510"):
    DONOR[s] = "Br5292"
for s in ("151669", "151670", "151671", "151672"):
    DONOR[s] = "Br5595"
for s in ("151673", "151674", "151675", "151676"):
    DONOR[s] = "Br8100"
SECTIONS = list(DONOR.keys())


def style_axes(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


DENSITY_MATCHED_DIR = f"{SRC_DIR}/density_matched"


def load():
    # Original four controls -- frozen CSVs, read-only, never rewritten by this script.
    long = pd.read_csv(f"{SRC_DIR}/control_ari_long.csv", dtype={"section": str})
    summary = pd.read_csv(f"{SRC_DIR}/control_summary.csv")
    donor = pd.read_csv(f"{SRC_DIR}/donor_level_tests.csv")
    # detection_downsampled (2026-08-28) -- separate density_matched/ subdirectory,
    # aggregated by aggregate_density_matched_control.py; merged in-memory here only,
    # never written back into the original three CSVs above.
    dm_long = pd.read_csv(f"{DENSITY_MATCHED_DIR}/control_ari_long.csv", dtype={"section": str})
    dm_summary = pd.read_csv(f"{DENSITY_MATCHED_DIR}/control_summary.csv")
    dm_donor = pd.read_csv(f"{DENSITY_MATCHED_DIR}/donor_level_tests.csv")
    long = pd.concat([long, dm_long], ignore_index=True)
    summary = pd.concat([summary, dm_summary], ignore_index=True)
    donor = pd.concat([donor, dm_donor], ignore_index=True)
    return long, summary, donor


def load_density():
    return pd.read_csv(f"{DENSITY_MATCHED_DIR}/matrix_density_comparison.csv", dtype={"section": str})


# ---------------------------------------------------------------------------
# Panel a -- per-section spread
# ---------------------------------------------------------------------------
def panel_a(ax, long):
    valid_mods = [m for m in CTRL_ORDER if m != "coverage_only"]
    n_sec = len(SECTIONS)
    section_x = {s: i for i, s in enumerate(SECTIONS)}
    width_per_mod = 0.62 / len(valid_mods)

    rows_out = []
    for mi, mod in enumerate(valid_mods):
        color = CTRL_COLOR[mod]
        sub = long[long.modality.eq(mod)]
        for s in SECTIONS:
            ss = sub[sub.section.eq(s)]
            if ss.empty:
                continue
            x0 = section_x[s] + (mi - (len(valid_mods) - 1) / 2) * width_per_mod
            jitter = (np.random.RandomState(abs(hash((s, mod))) % (2**32)).uniform(-0.16, 0.16, len(ss))
                      * width_per_mod)
            ax.scatter(x0 + jitter, ss.ari, s=7, color=color, alpha=0.45, linewidths=0, zorder=3)
            mean_ari = ss.ari.mean()
            ax.scatter([x0], [mean_ari], s=32, color=color, edgecolor=INK, linewidth=0.7, zorder=4)
            rows_out.append(dict(section=s, donor=DONOR[s], modality=mod,
                                  section_mean_ari=mean_ari, n_runs=len(ss)))

    # coverage_only: degenerate in all 120/120 runs -- mark, never plot a fake ARI.
    deg_color = CTRL_COLOR["coverage_only"]
    y_deg = -0.045
    for s in SECTIONS:
        ax.scatter([section_x[s]], [y_deg], marker="x", s=26, color=deg_color, linewidths=1.4, zorder=5)
        rows_out.append(dict(section=s, donor=DONOR[s], modality="coverage_only",
                              section_mean_ari=np.nan, n_runs=10, note="degenerate: rank 2 < G=7, all 10 runs"))

    # donor boundaries
    for b in (3.5, 7.5):
        ax.axvline(b, color=GRID, lw=1.0, linestyle="--", zorder=1)
    for donor_name, cx in [("Br5292", 1.5), ("Br5595", 5.5), ("Br8100", 9.5)]:
        ax.text(cx, 1.045, donor_name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.0, color=MUTED, fontweight="bold")

    ax.set_xticks(range(n_sec))
    ax.set_xticklabels(SECTIONS, rotation=45, ha="right", fontsize=7.6)
    ax.set_xlim(-0.7, n_sec - 0.3)
    ax.set_ylim(-0.09, 0.78)
    ax.set_ylabel("ARI vs cortical layer\n(10 optimizer runs per section)", fontsize=9.2, color=INK,
                  linespacing=1.3)
    style_axes(ax)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6,
                      markerfacecolor=CTRL_COLOR[m], markeredgecolor=INK,
                      label=CTRL_SHORT[m]) for m in valid_mods]
    handles.append(Line2D([0], [0], marker="x", linestyle="none", markersize=6, color=deg_color,
                          markeredgewidth=1.4, label="coverage-only (degenerate, all runs)"))
    ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=7.0, handlelength=1.1,
             borderaxespad=0.2, labelspacing=0.35)
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Panel b -- the comparison that matters
# ---------------------------------------------------------------------------
def panel_b(ax, long, summary):
    valid_mods = [m for m in CTRL_ORDER if m != "coverage_only"]
    x = np.arange(len(valid_mods))
    for xi, mod in zip(x, valid_mods):
        sub = long[long.modality.eq(mod)]
        color = CTRL_COLOR[mod]
        jitter = np.random.RandomState(abs(hash(mod)) % (2**32)).uniform(-0.12, 0.12, len(sub))
        ax.scatter(xi + jitter, sub.ari, s=8, color=color, alpha=0.35, linewidths=0, zorder=2)
        row = summary[summary.modality.eq(mod)].iloc[0]
        mean_v, sd_v = row.mean_ari_over_runs, row.sd_ari_over_runs
        ax.errorbar([xi], [mean_v], yerr=[[mean_v - max(mean_v - sd_v, 0)], [sd_v]], fmt="D",
                    color=color, markeredgecolor=INK, markersize=8, elinewidth=1.4, capsize=3.5, zorder=4)
        ax.text(xi, mean_v + sd_v + 0.02, f"{mean_v:.3f}", ha="center", va="bottom", fontsize=8.0,
                color=color, fontweight="bold")

    # coverage_only degenerate marker at the same x-spacing (for completeness / scale)
    x_cov = len(valid_mods)
    ax.scatter([x_cov], [-0.03], marker="x", s=40, color=CTRL_COLOR["coverage_only"], linewidths=1.6)
    ax.text(x_cov, -0.06, "degenerate\n(0/120 fit)", ha="center", va="top", fontsize=7.0,
            color=CTRL_COLOR["coverage_only"], linespacing=1.15)

    ax.set_xticks(list(x) + [x_cov])
    ax.set_xticklabels([CTRL_LABEL[m] for m in valid_mods] + [CTRL_LABEL["coverage_only"]],
                       fontsize=8.0, linespacing=1.15)
    ax.set_xlim(-0.6, x_cov + 0.6)
    ax.set_ylim(-0.09, 0.95)
    ax.set_ylabel("ARI vs cortical layer", fontsize=9.2, color=INK)
    style_axes(ax)

    # Cross-run reference lines -- explicitly NOT measured in this batch. Stack the
    # text labels above their lines (offset alternately) so they never collide.
    ref_items = sorted(REAL_REFS.items(), key=lambda kv: kv[1])
    ref_colors_by_val = {ref_items[0][0]: MUTED, ref_items[1][0]: SPARCAL_L, ref_items[2][0]: SPARCAL_C}
    for label, val in ref_items:
        rc = ref_colors_by_val[label]
        ax.axhline(val, color=rc, lw=1.6, linestyle=(0, (5, 2)), zorder=1)
    # Place labels in a stacked block on the right margin, ordered by value, using
    # annotate + arrow so text never overlaps the lines or each other.
    label_ys = np.linspace(0.55, 0.76, len(ref_items))
    for (label, val), ly in zip(reversed(ref_items), sorted(label_ys, reverse=True)):
        rc = ref_colors_by_val[label]
        ax.annotate(f"{label}\n= {val:.3f}", xy=(x_cov + 0.05, val),
                    xytext=(x_cov + 0.55, ly), fontsize=6.6, color=rc, va="center", ha="left",
                    linespacing=1.1, annotation_clip=False, clip_on=False,
                    arrowprops=dict(arrowstyle="-", color=rc, lw=0.9, shrinkA=0, shrinkB=0))
    ax.text(-0.55, 0.93, "dashed lines = cross-run reference (paper's own runs, not re-measured here)",
            fontsize=6.8, color=MUTED, va="top", ha="left")


# ---------------------------------------------------------------------------
# Panel c -- donor-level vs section-level (pseudoreplication)
# ---------------------------------------------------------------------------
def panel_c(ax, long, donor_df):
    valid_mods = [m for m in CTRL_ORDER if m != "coverage_only"]
    x = np.arange(len(valid_mods))
    rows_out = []
    for xi, mod in zip(x, valid_mods):
        color = CTRL_COLOR[mod]
        # section-level means (n=12), small dots with jitter
        sec_means = (long[long.modality.eq(mod)].groupby("section").ari.mean())
        jitter = np.random.RandomState(abs(hash(("c", mod))) % (2**32)).uniform(-0.13, 0.13, len(sec_means))
        ax.scatter(xi + jitter, sec_means.values, s=16, color=color, alpha=0.55, linewidths=0.4,
                   edgecolor=INK, zorder=3, label="_nolegend_")
        # donor-level means (n=3), diamonds
        drow = donor_df[donor_df.modality.eq(mod)].iloc[0]
        donor_vals = [drow.donor_Br5292_mean_ari, drow.donor_Br5595_mean_ari, drow.donor_Br8100_mean_ari]
        djit = np.array([-0.05, 0.0, 0.05])
        ax.scatter(xi + djit, donor_vals, marker="D", s=52, color=color, edgecolor=INK, linewidth=0.9,
                   zorder=5)
        for s_name, s_x, val in zip(["Br5292", "Br5595", "Br8100"], xi + djit, donor_vals):
            rows_out.append(dict(modality=mod, donor=s_name, donor_mean_ari=val))
        for s, v in sec_means.items():
            rows_out.append(dict(modality=mod, section=s, donor=DONOR[s], section_mean_ari=v))

    ax.set_xticks(x)
    ax.set_xticklabels([CTRL_SHORT[m].split("\n")[0] for m in valid_mods], fontsize=7.4,
                       rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.5, len(valid_mods) - 0.5)
    ax.set_ylim(-0.03, 0.72)
    ax.set_ylabel("ARI vs cortical layer", fontsize=9.2, color=INK)
    style_axes(ax)
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6, markerfacecolor=MUTED,
                      markeredgecolor=INK, label="section mean (n = 12)"),
              Line2D([0], [0], marker="D", linestyle="none", markersize=7, markerfacecolor=MUTED,
                      markeredgecolor=INK, label="donor mean (n = 3)")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.2, handlelength=1.0,
             borderaxespad=0.2)
    ax.text(0.02, -0.34, "coverage-only omitted: degenerate in 120/120 runs (no ARI to average)",
            transform=ax.transAxes, fontsize=6.6, color=CTRL_COLOR["coverage_only"], ha="left", va="top")
    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Panel d -- density confound: nonzero fraction & mean row sum, every matrix
# ---------------------------------------------------------------------------
DENSITY_MOD_ORDER = ["real_somtop25_bin250kb", "detection_only", "detection_downsampled",
                     "allele_permuted", "smoothed_random", "coverage_only"]
DENSITY_MOD_LABEL = {
    "real_somtop25_bin250kb": "real (somtop25)",
    "detection_only": "detection-only",
    "detection_downsampled": "detection-downsampled",
    "allele_permuted": "allele-permuted",
    "smoothed_random": "smoothed-random",
    "coverage_only": "coverage-only",
}
DENSITY_MOD_COLOR = {
    "real_somtop25_bin250kb": SPARCAL_C,
    "detection_only": CTRL_COLOR["detection_only"],
    "detection_downsampled": CTRL_COLOR["detection_downsampled"],
    "allele_permuted": CTRL_COLOR["allele_permuted"],
    "smoothed_random": CTRL_COLOR["smoothed_random"],
    "coverage_only": CTRL_COLOR["coverage_only"],
}


def panel_d(fig, gs_cell, density_df):
    sub = gs_cell.subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.75)
    ax_nz = fig.add_subplot(sub[0, 0])
    ax_sum = fig.add_subplot(sub[0, 1])

    x = np.arange(len(DENSITY_MOD_ORDER))
    for xi, mod in zip(x, DENSITY_MOD_ORDER):
        color = DENSITY_MOD_COLOR[mod]
        sub_df = density_df[density_df.modality.eq(mod)]
        jitter_nz = np.random.RandomState(abs(hash(("d_nz", mod))) % (2**32)).uniform(-0.14, 0.14, len(sub_df))
        jitter_sum = np.random.RandomState(abs(hash(("d_sum", mod))) % (2**32)).uniform(-0.14, 0.14, len(sub_df))
        ax_nz.scatter(xi + jitter_nz, sub_df.nonzero_frac, s=14, color=color, alpha=0.55, linewidths=0,
                     zorder=3)
        ax_nz.scatter([xi], [sub_df.nonzero_frac.mean()], s=42, color=color, edgecolor=INK,
                     linewidth=0.8, marker="D", zorder=4)
        ax_sum.scatter(xi + jitter_sum, sub_df.mean_row_sum, s=14, color=color, alpha=0.55, linewidths=0,
                      zorder=3)
        ax_sum.scatter([xi], [sub_df.mean_row_sum.mean()], s=42, color=color, edgecolor=INK,
                      linewidth=0.8, marker="D", zorder=4)

    for ax, col, ylab in [(ax_nz, "nonzero_frac", "nonzero fraction\n(log scale)"),
                          (ax_sum, "mean_row_sum", "mean row sum\n(log scale)")]:
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([DENSITY_MOD_LABEL[m] for m in DENSITY_MOD_ORDER], fontsize=6.8,
                           rotation=42, ha="right", rotation_mode="anchor")
        ax.set_ylabel(ylab, fontsize=8.2, color=INK, linespacing=1.2)
        style_axes(ax)
    ax_nz.text(0.0, 1.16, "12 sections/modality; diamond = section mean", transform=ax_nz.transAxes,
              fontsize=6.4, color=MUTED, ha="left", va="bottom")
    return ax_nz, ax_sum


def main():
    np.random.seed(0)
    long, summary, donor_df = load()
    density_df = load_density()

    fig = plt.figure(figsize=(19.6, 4.9))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.5, 1.25, 0.95, 1.15], wspace=0.55,
                          left=0.04, right=0.865, top=0.84, bottom=0.26)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    panel_a_src = panel_a(ax_a, long)
    panel_b(ax_b, long, summary)
    panel_c_src = panel_c(ax_c, long, donor_df)
    panel_d(fig, gs[0, 3], density_df)

    panel_a_src.to_csv(f"{DERIVED_DIR}/fig_dlpfc_negative_controls_panel_a_source.csv", index=False)
    panel_c_src.to_csv(f"{DERIVED_DIR}/fig_dlpfc_negative_controls_panel_c_source.csv", index=False)
    density_df.to_csv(f"{DERIVED_DIR}/fig_dlpfc_negative_controls_panel_d_source.csv", index=False)

    ax_a.text(-0.16, 1.14, "a", transform=ax_a.transAxes, fontsize=13, fontweight="bold",
              color=INK, va="top", ha="left")
    ax_b.text(-0.19, 1.14, "b", transform=ax_b.transAxes, fontsize=13, fontweight="bold",
              color=INK, va="top", ha="left")
    ax_c.text(-0.19, 1.14, "c", transform=ax_c.transAxes, fontsize=13, fontweight="bold",
              color=INK, va="top", ha="left")
    fig.text(0.712, 0.955, "d", fontsize=13, fontweight="bold", color=INK, va="top", ha="left")

    stem = "fig_dlpfc_negative_controls" if HAS_ARIAL else "fig_dlpfc_negative_controls_preview"
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
