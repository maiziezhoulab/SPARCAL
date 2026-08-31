#!/usr/bin/env python
"""fig_phasing_chain (2026-08-28) -- the read-backed/absolute-phasing evidence chain.

No figure existed for the phasing work spread across four independent analyses
(2026-08-23/24). This pulls the already-computed numbers from all four into one
figure that tells the chain of the story end to end:

  (a) Beagle input-mode probe -- gl=/gtgl= genotype-likelihood input NEVER phases
      (0.0% of het sites) on this data; only feeding genotypes directly via gt=
      phases (100.0%). Beagle 4.1 additionally fails outright on 29/88 genome-wide
      chromosome runs (all in DCIS, an output-time crash under the dense GRCh38
      panel); Beagle 5.4 succeeds on all 88/88.
  (b) Read-backed gate funnel (per section): somatic candidates -> >=2 ALT reads
      -> a heterozygous germline anchor within 1kb -> actually co-observed on a
      read or UMI family ("testable"), with the testable count broken into
      read-level vs UMI-level-only support.
  (c) Anchor scarcity -- the binding constraint on the whole chain: the fraction
      of UNTESTABLE candidates that have no heterozygous anchor within 1kb at all
      (90.5-94.9%), and the gulf between testable and untestable candidates'
      median distance to their nearest heterozygous site (tens of bp vs tens of
      kb).
  (d) Anchored (non-read-backed) phase-prediction model: 5-fold CV AUROC for
      gradient boosting vs logistic regression against the 0.5 chance line --
      real signal, far from a deployable per-candidate tool.
  (e) Beagle 4.1 vs 5.4 engine concordance on population-reference phasing --
      barely above the 50% chance line in every sample, the direct evidence that
      ABSOLUTE haplotype orientation (which allele is on which physical
      chromosome copy) is not resolved by either engine on this data, only
      RELATIVE (same/opposite) phase from real molecular read/UMI evidence is
      trustworthy.

Sources (all read-only; see each analysis's own RESULTS.md for full methodology):
  data/sidecar_phasing_probe_2026-08-23/probe_comparison.csv
  data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/_manifest.json (4.1)
  data/confident_set_phasing_2026-08-24/genomewide_beagle_gt/_manifest_beagle54.json (5.4)
  data/confident_set_phasing_2026-08-24/engine_concordance.json
  data/readbacked_feasibility_2026-08-23/summary.json (gate funnel, per section)
  data/anchored_phase_model_2026-08-24/cv_performance.csv (feature_variant=
    clean_excl_label_spots, label_subset=confident -- the leakage-audited,
    non-read-backed, primary answer)
  data/anchored_phase_model_2026-08-24/covariate_shift_summary.csv (anchor scarcity)

Run:
  python scripts/postanalyze/fig_phasing_chain.py

Outputs
  data/paper_figs_2026-08-27/fig_phasing_chain_panel_{a,b,c,d,e}_source.csv
  SPARCAL_pnas_2026/figs/v7_2026-08-27/fig_phasing_chain[_preview].{png,pdf}
"""
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
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
PROBE_DIR = f"{PROJECT}/data/sidecar_phasing_probe_2026-08-23"
CONFIDENT_DIR = f"{PROJECT}/data/confident_set_phasing_2026-08-24"
FEAS_DIR = f"{PROJECT}/data/readbacked_feasibility_2026-08-23"
MODEL_DIR = f"{PROJECT}/data/anchored_phase_model_2026-08-24"
DERIVED_DIR = f"{PROJECT}/data/paper_figs_2026-08-27"
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v7_2026-08-27"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C, SPARCAL_L = "#e34948", "#f6c3c2"
MONO_C = "#4a3aa7"
SSNV_C = "#2a78d6"
WES_C = "#7651a6"

# Per-sample palette -- distinct from the reserved method colors above.
SAMPLE_ORDER = ["P4", "P6", "DCIS1", "DCIS2"]
SAMPLE_COLOR = {
    "P4": "#c98a2b",     # amber
    "P6": "#4f9d69",     # green
    "DCIS1": "#b0495a",  # brick rose
    "DCIS2": "#5b7aa6",  # slate blue
}


def style_axes(ax, grid_axis="y"):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=7.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.7, zorder=0, which="major")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel a -- Beagle input-mode probe + 4.1-vs-5.4 run success
# ---------------------------------------------------------------------------
def load_panel_a():
    probe = pd.read_csv(os.path.join(PROBE_DIR, "probe_comparison.csv"))
    tags = {"gl=": "shipped_niter0_imputeF_gl", "gtgl=": "niter5_imputeF_gtgl", "gt=": "niter5_imputeF_gt"}
    rows = []
    for label, tag in tags.items():
        r = probe[probe.run_tag.eq(tag)].iloc[0]
        rows.append(dict(input_field=label, pct_het_phased=float(r.pct_het_phased),
                          n_het=int(r.n_het), run_tag=tag))
    df = pd.DataFrame(rows)

    m41 = json.load(open(os.path.join(CONFIDENT_DIR, "genomewide_beagle_gt", "_manifest.json")))
    m54 = json.load(open(os.path.join(CONFIDENT_DIR, "genomewide_beagle_gt", "_manifest_beagle54.json")))
    ok41 = sum(1 for r in m41 if r.get("returncode") == 0 and r.get("stats"))
    ok54 = sum(1 for r in m54 if r.get("returncode") == 0 and r.get("stats"))
    fail41_samples = sorted({r["label"] for r in m41 if not (r.get("returncode") == 0 and r.get("stats"))})
    success = dict(engine=["Beagle 4.1", "Beagle 5.4"], n_ok=[ok41, ok54], n_total=[len(m41), len(m54)],
                    fail_samples=["+".join(fail41_samples), ""])
    return df, pd.DataFrame(success)


def panel_a(ax, df, success):
    x = np.arange(len(df))
    colors = [MUTED, MUTED, SPARCAL_C]
    bars = ax.bar(x, df.pct_het_phased, width=0.55, color=colors, edgecolor=INK, linewidth=0.6, zorder=3)
    for xi, v in zip(x, df.pct_het_phased):
        ax.text(xi, v + 2.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=8.2, color=INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df.input_field, fontsize=9.0)
    ax.set_ylim(0, 118)
    ax.set_ylabel("% het sites phased\n(P4 chr10 probe)", fontsize=8.8, color=INK, linespacing=1.25)
    style_axes(ax)

    r41, r54 = success.iloc[0], success.iloc[1]
    txt = (f"genome-wide\n(4 samples x 22 chrom):\n"
           f"Beagle 4.1: {int(r41.n_ok)}/{int(r41.n_total)} OK\n"
           f"(29 fail = all DCIS)\n"
           f"Beagle 5.4: {int(r54.n_ok)}/{int(r54.n_total)} OK")
    ax.text(0.05, 0.62, txt, transform=ax.transAxes, fontsize=6.5, color=MUTED, ha="left", va="top",
            linespacing=1.35, clip_on=False,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))


# ---------------------------------------------------------------------------
# Panel b -- read-backed gate funnel per section (log-y) + read/UMI split
# ---------------------------------------------------------------------------
def load_panel_b():
    summ = json.load(open(os.path.join(FEAS_DIR, "summary.json")))
    rows = []
    for s in SAMPLE_ORDER:
        v = summ[f"{s}_somatic"]
        rows.append(dict(
            section=s,
            candidates=v["gate_n_candidates"], ge2_alt=v["gate_n_ge2_alt"],
            het_in_range=v["gate_n_het_in_range"], testable=v["gate_n_testable"],
            testable_read_level=v["gate_n_testable_via_read_level_only"],
            testable_umi_only=v["gate_n_testable_via_umi_level_only"],
        ))
    return pd.DataFrame(rows)


def panel_b(fig, gs_cell, df):
    sub = gs_cell.subgridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.55)
    ax_funnel = fig.add_subplot(sub[0, 0])
    ax_split = fig.add_subplot(sub[0, 1])

    stages = ["candidates", "ge2_alt", "het_in_range", "testable"]
    stage_labels = ["candidates", ">=2 ALT\nreads", "het\n<=1kb", "testable"]
    xs = np.arange(len(stages))
    for _, r in df.iterrows():
        color = SAMPLE_COLOR[r.section]
        ys = [r[s] for s in stages]
        ax_funnel.plot(xs, ys, marker="o", markersize=5, color=color, linewidth=1.8,
                       markeredgecolor=INK, markeredgewidth=0.5, zorder=4, label=r.section)
    ax_funnel.set_yscale("log")
    ax_funnel.set_xticks(xs)
    ax_funnel.set_xticklabels(stage_labels, fontsize=7.6)
    ax_funnel.set_ylabel("n candidates\n(log scale)", fontsize=8.6, color=INK, linespacing=1.25, labelpad=2)
    style_axes(ax_funnel, grid_axis="y")
    ax_funnel.legend(loc="lower left", frameon=False, fontsize=6.6, handlelength=1.1, borderaxespad=0.2,
                     labelspacing=0.3)

    xb = np.arange(len(df))
    bottoms = df.testable_read_level.values
    tops = df.testable_umi_only.values
    for i, (s, bo, to) in enumerate(zip(df.section, bottoms, tops)):
        color = SAMPLE_COLOR[s]
        ax_split.bar(i, bo, width=0.6, color=color, edgecolor=INK, linewidth=0.5, zorder=3, alpha=0.95)
        ax_split.bar(i, to, bottom=bo, width=0.6, color=color, edgecolor=INK, linewidth=0.5, zorder=3,
                    alpha=0.45, hatch="///")
        ax_split.text(i, bo + to + max(tops + bottoms) * 0.02, f"{bo + to:,}", ha="center", va="bottom",
                     fontsize=6.6, color=INK)
    ax_split.set_xticks(xb)
    ax_split.set_xticklabels(df.section, rotation=35, ha="right", fontsize=7.2)
    ax_split.set_ylabel("testable candidates", fontsize=8.2, color=INK)
    style_axes(ax_split, grid_axis="y")
    handles = [Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=MUTED,
                      markeredgecolor=INK, alpha=0.95, label="read-level"),
              Line2D([0], [0], marker="s", linestyle="none", markersize=8, markerfacecolor=MUTED,
                      markeredgecolor=INK, alpha=0.45, label="UMI-only", markeredgewidth=0.5)]
    ax_split.legend(handles=handles, loc="upper right", frameon=False, fontsize=6.2, handlelength=1.0,
                    borderaxespad=0.15)
    return ax_funnel, ax_split


# ---------------------------------------------------------------------------
# Panel c -- anchor scarcity: frac untestable w/ no het <=1kb, + median distance gap
# ---------------------------------------------------------------------------
def load_panel_c():
    df = pd.read_csv(os.path.join(MODEL_DIR, "covariate_shift_summary.csv"))
    df = df.set_index("section").loc[SAMPLE_ORDER].reset_index()
    return df


def panel_c(fig, gs_cell, df):
    sub = gs_cell.subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.55)
    ax_frac = fig.add_subplot(sub[0, 0])
    ax_dist = fig.add_subplot(sub[0, 1])

    x = np.arange(len(df))
    colors = [SAMPLE_COLOR[s] for s in df.section]
    ax_frac.bar(x, df.untestable_frac_no_het_within_1000bp * 100, width=0.6, color=colors,
               edgecolor=INK, linewidth=0.6, zorder=3)
    for xi, v in zip(x, df.untestable_frac_no_het_within_1000bp * 100):
        ax_frac.text(xi, v + 1.2, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.2, color=INK,
                    fontweight="bold")
    ax_frac.set_xticks(x)
    ax_frac.set_xticklabels(df.section, rotation=35, ha="right", fontsize=7.2)
    ax_frac.set_ylim(0, 108)
    ax_frac.set_ylabel("% untestable w/\nno het <=1kb", fontsize=8.0, color=INK,
                      linespacing=1.25, labelpad=2)
    style_axes(ax_frac, grid_axis="y")

    for i, (_, r) in enumerate(df.iterrows()):
        color = SAMPLE_COLOR[r.section]
        ax_dist.plot([i, i], [r.testable_median_nearest_het_dist_bp, r.untestable_median_nearest_het_dist_bp],
                    color=color, lw=1.6, zorder=2)
        ax_dist.scatter([i], [r.testable_median_nearest_het_dist_bp], color=color, edgecolor=INK,
                       linewidth=0.6, s=42, zorder=4, marker="o")
        ax_dist.scatter([i], [r.untestable_median_nearest_het_dist_bp], color=color, edgecolor=INK,
                       linewidth=0.6, s=42, zorder=4, marker="^")
    ax_dist.set_yscale("log")
    ax_dist.set_xticks(np.arange(len(df)))
    ax_dist.set_xticklabels(df.section, rotation=35, ha="right", fontsize=7.2)
    ax_dist.set_ylabel("median dist. to nearest het\n(bp, log scale)", fontsize=8.0, color=INK, linespacing=1.25)
    style_axes(ax_dist, grid_axis="y")
    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=6, markerfacecolor=MUTED,
                      markeredgecolor=INK, label="testable"),
              Line2D([0], [0], marker="^", linestyle="none", markersize=6, markerfacecolor=MUTED,
                      markeredgecolor=INK, label="untestable")]
    ax_dist.legend(handles=handles, loc="upper left", frameon=False, fontsize=6.4, handlelength=1.0,
                   borderaxespad=0.15)
    return ax_frac, ax_dist


# ---------------------------------------------------------------------------
# Panel d -- anchored (non-read-backed) phase model CV AUROC
# ---------------------------------------------------------------------------
def load_panel_d():
    cv = pd.read_csv(os.path.join(MODEL_DIR, "cv_performance.csv"))
    cv = cv[(cv.feature_variant.eq("clean_excl_label_spots")) & (cv.label_subset.eq("confident"))]
    return cv


def panel_d(ax, cv):
    models = [("logistic_regression", "LR"), ("gradient_boosting", "GBM")]
    x = np.arange(len(models))
    for xi, (mkey, mlabel) in zip(x, models):
        sub = cv[cv.model.eq(mkey)]
        color = MUTED if mkey == "logistic_regression" else SPARCAL_C
        jitter = np.random.RandomState(abs(hash(mkey)) % (2**32)).uniform(-0.10, 0.10, len(sub))
        ax.scatter(xi + jitter, sub.auroc, s=22, color=color, alpha=0.55, linewidths=0, zorder=3)
        mean_v, sd_v = sub.auroc.mean(), sub.auroc.std()
        ax.errorbar([xi], [mean_v], yerr=[[sd_v], [sd_v]], fmt="D", color=color, markeredgecolor=INK,
                   markersize=8, elinewidth=1.5, capsize=4, zorder=5)
        ax.text(xi, mean_v + sd_v + 0.025, f"{mean_v:.3f}\n±{sd_v:.3f}", ha="center", va="bottom",
               fontsize=7.4, color=color, fontweight="bold", linespacing=1.15)
    ax.axhline(0.5, color=MUTED, lw=1.4, linestyle=(0, (5, 2)), zorder=1)
    ax.text(1.55, 0.503, "chance", fontsize=6.8, color=MUTED, va="bottom", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in models], fontsize=9.0)
    ax.set_xlim(-0.5, 1.9)
    ax.set_ylim(0.35, 0.85)
    ax.set_ylabel("5-fold CV AUROC\n(confident set, n=897)", fontsize=8.6, color=INK, linespacing=1.25)
    style_axes(ax)


# ---------------------------------------------------------------------------
# Panel e -- Beagle 4.1 vs 5.4 engine concordance
# ---------------------------------------------------------------------------
def load_panel_e():
    d = json.load(open(os.path.join(CONFIDENT_DIR, "engine_concordance.json")))
    rows = []
    for s in SAMPLE_ORDER:
        v = d["per_sample"][s]
        rows.append(dict(sample=s, agreement_rate=v["overall_agreement_rate"] * 100,
                          n_shared_het_sites=v["n_shared_het_sites_total"]))
    rows.append(dict(sample="overall", agreement_rate=d["overall"]["agreement_rate"] * 100,
                      n_shared_het_sites=d["overall"]["n_shared_het_sites_total"]))
    return pd.DataFrame(rows)


def panel_e(ax, df):
    x = np.arange(len(df))
    colors = [SAMPLE_COLOR.get(s, INK) for s in df["sample"]]
    ax.bar(x, df.agreement_rate, width=0.6, color=colors, edgecolor=INK, linewidth=0.6, zorder=3)
    for xi, v in zip(x, df.agreement_rate):
        ax.text(xi, v + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=7.6, color=INK, fontweight="bold")
    ax.axhline(50, color=MUTED, lw=1.4, linestyle=(0, (5, 2)), zorder=1)
    ax.text(len(df) - 0.4, 50.3, "50% chance", fontsize=6.8, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample"], rotation=35, ha="right", fontsize=7.6)
    ax.set_ylim(45, 60)
    ax.set_ylabel("4.1 vs 5.4 phase\nagreement rate", fontsize=8.6, color=INK, linespacing=1.25)
    style_axes(ax)


def main():
    np.random.seed(0)
    a_df, a_success = load_panel_a()
    b_df = load_panel_b()
    c_df = load_panel_c()
    d_df = load_panel_d()
    e_df = load_panel_e()

    a_df.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_a_source.csv", index=False)
    a_success.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_a_engine_success.csv", index=False)
    b_df.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_b_source.csv", index=False)
    c_df.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_c_source.csv", index=False)
    d_df.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_d_source.csv", index=False)
    e_df.to_csv(f"{DERIVED_DIR}/fig_phasing_chain_panel_e_source.csv", index=False)

    fig = plt.figure(figsize=(20.5, 4.9))
    gs = fig.add_gridspec(1, 5, width_ratios=[0.85, 1.6, 1.6, 0.85, 0.95], wspace=0.32,
                          left=0.032, right=0.99, top=0.86, bottom=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    panel_a(ax_a, a_df, a_success)

    panel_b(fig, gs[0, 1], b_df)
    panel_c(fig, gs[0, 2], c_df)

    ax_d = fig.add_subplot(gs[0, 3])
    panel_d(ax_d, d_df)

    ax_e = fig.add_subplot(gs[0, 4])
    panel_e(ax_e, e_df)

    for letter, xpos in zip("abcde", [0.008, 0.185, 0.435, 0.685, 0.855]):
        fig.text(xpos, 0.955, letter, fontsize=13, fontweight="bold", color=INK, va="top", ha="left")

    stem = "fig_phasing_chain" if HAS_ARIAL else "fig_phasing_chain_preview"
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
