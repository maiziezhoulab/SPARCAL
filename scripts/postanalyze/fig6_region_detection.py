#!/usr/bin/env python
"""Fig 6 -- cross-tool tumor-region-detection ARI (PAPER_PLAN.md Story C4, Fig 6).

Two panels, DCIS2 and P6, sharing ONE y-axis scale (0-0.75) on purpose: the
absolute ARI levels differ enormously between samples (DCIS2 ~0.6-0.67, P6
~0.09-0.16) and that gap IS the finding -- a per-panel autoscaled axis would
visually flatten it away. `coverage_only` is drawn as a labelled horizontal
dashed reference line (not a competing bar) per PAPER_PLAN Decision L3/D4.
A companion F1 figure is emitted the same way, at the SAME operating point
(method/norm/params) that produced each bar's best raw ARI -- not an
independently-best-F1 point -- so the two panels describe one run each.

CRITICAL FRAMING (PAPER_PLAN.md Sec 3 Story C4, Sec 6 guardrail 7):
  - This is NOT a caller ranking. The ordering flips between samples (SPARCAL
    somatic is best on P6, 3rd on DCIS2) and nothing separates meaningfully
    from the coverage_only baseline on either sample (DCIS2: -0.045; P6: SPARCAL
    +0.005 -- noise either way).
  - DCIS2 is drawn from SpatialSNV's own published dataset, so the one
    comparison SpatialSNV wins there is not an independent test -- flagged with
    a dagger on the bar and stated in the caption.
  - P6 absolute ARIs (0.09-0.16, every method) are near-useless in absolute
    terms versus DCIS2 (0.60-0.67) -- annotation geometry (compact foci vs
    diffuse leading edge), not caller identity, drives this gap.

INPUT (committed CSV, no re-derivation needed):
  SPARCAL_Benchmarking/analysis/region_method_benchmark/current_2026-07-28/
    benchmark_best_ari_all.csv
  -- "best" here means: restricted to norm=='raw' rows (the raw-burden,
     non-coverage-normalized detector output -- coverage-normalization is the
     deferred paper #2's axis, PAPER_PLAN_DEPRECATED.md Sec 1), max over
     method x params within that subset. VERIFIED: taking the true global max
     (raw+norm together) reproduces every number in this figure's brief
     EXCEPT p6/SpatialSNV/all, where the coverage-normalized value (0.299) is
     higher than any raw value and would silently swap in a different burden
     convention for one bar only -- see fig6_region_detection_VERIFY_NOTE
     printed at runtime.

OUTPUT:
  data/paper_figs_2026-07-29/fig6_best_ari_raw.csv
  data/paper_figs_2026-07-29/fig6_best_f1_at_ari_op_point.csv
  SPARCAL_pnas_2026/figs/v2_2026-07-29/fig6_region_detection_ari.{png,pdf}
  SPARCAL_pnas_2026/figs/v2_2026-07-29/fig6_region_detection_f1.{png,pdf}

Run (env snv_caller, CPU, seconds): python scripts/postanalyze/fig6_region_detection.py
"""
import os

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PROJECT = "/data/maiziezhou_lab/leiy4/snv_calling"
PAPER = os.path.join(os.path.dirname(PROJECT), "SPARCAL_pnas_2026")
FONT_DIR = os.path.join(PAPER, "fonts", "arial")
REGION_CSV = ("/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/analysis/"
              "region_method_benchmark/current_2026-07-28/benchmark_best_ari_all.csv")
DERIVED_DIR = os.path.join(PROJECT, "data", "paper_figs_2026-07-29")
FIG_DIR = "/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026/figs/v2_2026-07-29"
os.makedirs(DERIVED_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def configure_required_font():
    """Require genuine Arial for a final-basename manuscript export."""
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local = font_manager.findSystemFonts(fontpaths=[FONT_DIR]) if os.path.isdir(FONT_DIR) else []
    explicit_family = (font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
                       if explicit and os.path.dirname(explicit) else [])
    candidates = ([explicit] + explicit_family) if explicit else local + font_manager.findSystemFonts()
    matches = []
    for path in dict.fromkeys(path for path in candidates if path):
        if not os.path.exists(path):
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

INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"

SAMPLE_ORDER = ["dcis2", "p6"]
SAMPLE_TITLE = {"dcis2": "DCIS2", "p6": "P6"}

# Bar order is fixed across both panels for direct comparability; a tool/set
# simply absent from a sample's CSV rows is omitted, never drawn as a zero bar.
BAR_ORDER = [
    ("SPARCAL", "somatic"), ("SPARCAL", "upv_somatic"), ("SPARCAL", "merged"),
    ("SpatialSNV", "all"), ("Monopogen", "all"),
]
BAR_LABEL = {
    ("SPARCAL", "somatic"): "somatic",
    ("SPARCAL", "upv_somatic"): "UPV+som",
    ("SPARCAL", "merged"): "merged",
    ("SpatialSNV", "all"): "SpatialSNV",
    ("Monopogen", "all"): "Monopogen",
}
BAR_COLOR = {
    ("SPARCAL", "somatic"): "#e34948",       # somatic -- locked red
    ("SPARCAL", "upv_somatic"): "#4a3aa7",   # UPV+somatic -- locked purple
    ("SPARCAL", "merged"): "#898781",        # merged (all classes) -- neutral gray
    ("SpatialSNV", "all"): "#1baf7a",        # comparator tool -- aqua
    ("Monopogen", "all"): "#eda100",         # comparator tool -- yellow
}
DAGGER = {("dcis2", "SpatialSNV", "all")}    # same dataset as SpatialSNV's own publication


def build_tables():
    bench = pd.read_csv(REGION_CSV)

    # -- verification note: raw-only vs global-max disagree for one cell --
    global_best = bench.groupby(["sample", "tool", "snv_set"])["ari"].idxmax()
    raw_best = bench[bench.norm == "raw"].groupby(["sample", "tool", "snv_set"])["ari"].idxmax()
    mismatch = set(global_best) - set(raw_best.values) | set(raw_best.values) - set(global_best)
    for idx in sorted(mismatch):
        r = bench.loc[idx]
        print(f"[fig6 VERIFY NOTE] {r['sample']}/{r.tool}/{r.snv_set}: "
              f"global-max row uses norm={r.norm} ari={r.ari:.3f} -- excluded because "
              f"this figure fixes norm=='raw' throughout (coverage-normalization is "
              f"paper #2's axis, not this one).")

    raw = bench[bench.norm == "raw"].copy()
    idx = raw.groupby(["sample", "tool", "snv_set"])["ari"].idxmax()
    best = raw.loc[idx, ["sample", "tool", "snv_set", "ari", "f1", "precision", "recall",
                          "method", "norm", "params"]].reset_index(drop=True)
    best_ari_csv = os.path.join(DERIVED_DIR, "fig6_best_ari_raw.csv")
    best.to_csv(best_ari_csv, index=False)
    print(f"[fig6] wrote {best_ari_csv}")
    return best


def coverage_value(best, sample):
    row = best[(best["sample"] == sample) & (best.tool == "coverage")]
    return float(row.ari.iloc[0]) if len(row) else None, \
           float(row.f1.iloc[0]) if len(row) else None


def plot_panel(ax, best, sample, metric, ymax, ylabel):
    sub = best[best["sample"] == sample]
    xt, xl, vals, colors = [], [], [], []
    for i, (tool, sset) in enumerate(BAR_ORDER):
        row = sub[(sub.tool == tool) & (sub.snv_set == sset)]
        if len(row) == 0:
            continue
        xt.append(len(xt))
        xl.append(BAR_LABEL[(tool, sset)])
        vals.append(float(row[metric].iloc[0]))
        colors.append(BAR_COLOR[(tool, sset)])

    bars = ax.bar(xt, vals, color=colors, width=0.62, edgecolor="white",
                   linewidth=0.5, zorder=3)
    # which (tool,sset) pairs were actually plotted, in BAR_ORDER order
    plotted = [(tool, sset) for (tool, sset) in BAR_ORDER
               if len(sub[(sub.tool == tool) & (sub.snv_set == sset)])]
    for b, v, key in zip(bars, vals, plotted):
        label = f"{v:.3f}"
        if (sample, key[0], key[1]) in DAGGER:
            label += "†"
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012, label,
                 ha="center", va="bottom", fontsize=6.3, color=INK,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.4))

    # bracket over the SPARCAL bars (always the leading block, when present)
    sparcal_idx = [i for i, (tool, _) in enumerate(plotted) if tool == "SPARCAL"]
    if sparcal_idx:
        x0, x1 = sparcal_idx[0] - 0.35, sparcal_idx[-1] + 0.35
        yb = ymax * 0.955
        ax.plot([x0, x1], [yb, yb], color=MUTED, lw=0.8, clip_on=False)
        ax.plot([x0, x0], [yb, yb - ymax * 0.018], color=MUTED, lw=0.8, clip_on=False)
        ax.plot([x1, x1], [yb, yb - ymax * 0.018], color=MUTED, lw=0.8, clip_on=False)
        ax.text((x0 + x1) / 2, yb + ymax * 0.012, "SPARCAL", fontsize=6.2,
                 color=MUTED, ha="center", va="bottom")

    ax.set_xlim(-0.65, len(xt) - 1 + 1.55)  # right margin so the coverage label clears the bars

    cov_val, cov_f1 = coverage_value(sub, sample)
    cov = cov_val if metric == "ari" else cov_f1
    if cov is not None:
        ax.axhline(cov, color=MUTED, linestyle="--", linewidth=1.1, zorder=2)
        ax.text(len(xt) - 1 + 0.55, cov, f"coverage_only\n{cov:.3f}",
                 fontsize=5.8, color=MUTED, ha="left", va="center",
                 bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.6))

    ax.set_xticks(xt)
    ax.set_xticklabels(xl, fontsize=7, rotation=25, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MUTED)
    ax.spines["bottom"].set_color(MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


def make_figure(best, metric, ylabel, out_stub, ymax):
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.1), sharey=True)
    for ax, sample in zip(axes, SAMPLE_ORDER):
        plot_panel(ax, best, sample, metric, ymax, ylabel if sample == "dcis2" else "")
    axes[0].set_title("a  DCIS2", fontsize=9, fontweight="bold", loc="left")
    axes[1].set_title("b  P6", fontsize=9, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.12)
    suffix = "" if HAS_ARIAL else "_preview"
    png = os.path.join(FIG_DIR, f"{out_stub}{suffix}.png")
    pdf = os.path.join(FIG_DIR, f"{out_stub}{suffix}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[fig6] wrote {png}")
    print(f"[fig6] wrote {pdf}")


def main():
    if HAS_ARIAL:
        print(f"[font] Arial loaded from {ARIAL_PATH}")
    else:
        print("[font] WARNING: Arial unavailable; writing Nimbus Sans previews only.")
    best = build_tables()

    f1_csv = os.path.join(DERIVED_DIR, "fig6_best_f1_at_ari_op_point.csv")
    best.to_csv(f1_csv, index=False)  # same table carries f1 at the ARI-best op point
    print(f"[fig6] wrote {f1_csv}")

    ymax_ari = 0.75  # shared across DCIS2 and P6 on purpose -- see module docstring
    make_figure(best, "ari", "Best raw ARI (tumor-region detection)",
                "fig6_region_detection_ari", ymax_ari)

    ymax_f1 = 0.90
    make_figure(best, "f1", "F1 at the best-ARI operating point",
                "fig6_region_detection_f1", ymax_f1)


if __name__ == "__main__":
    main()
