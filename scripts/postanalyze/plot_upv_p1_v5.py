#!/usr/bin/env python3
"""Generate the documented v5 figures for P1-1 and the deep UPV audit."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plot_priority_figures_v5 import (
    ARIAL_PATH, DERIVED_DIR, HAS_ARIAL, INK, MUTED, PROJECT,
    save_figure, style_axis,
)


P1 = PROJECT / "data/germline_concordance_2026-08-23"
UPV = PROJECT / "data/upv_baf_assessment_2026-08-23"
DELTA = PROJECT / "data/upv_delta_baf_2026-08-23"
SAMPLES = ["P4", "P6", "DCIS1", "DCIS2"]
SAMPLE_COLORS = {"P4": "#2a78d6", "P6": "#4a3aa7", "DCIS1": "#e34948", "DCIS2": "#45a66a"}
DEFINED, NONPANEL = "#2a78d6", "#e07a3f"


def letter(axis, value):
    axis.set_title(value, loc="left", fontweight="bold", fontsize=11)


def p1_figure():
    depth = pd.read_csv(P1 / "concordance_by_depth.csv")
    summary = pd.read_csv(P1 / "concordance_summary.csv")
    depth_source = depth[
        depth.panel.eq("all") & depth.wes_truth_gt.eq("all")
    ].copy()
    threshold_source = summary[
        summary.threshold.isin(["RNA_DP>=10", "normal_WES_pileup_DP>=10"])
    ].copy()
    depth_source.to_csv(DERIVED_DIR / "fig_v5_p1_depth_source.csv", index=False)
    threshold_source.to_csv(DERIVED_DIR / "fig_v5_p1_threshold_source.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4))
    order = ["0", "1-3", "4-9", "10-29", "30+"]
    axis = axes[0, 0]
    style_axis(axis); letter(axis, "a")
    for sample in ["P4", "P6"]:
        g = depth_source[depth_source["sample"].eq(sample)].copy()
        g["x"] = g.rna_depth_bin.astype(str).map({v: i for i, v in enumerate(order)})
        g = g.sort_values("x")
        axis.errorbar(
            g.x, g.sensitivity,
            yerr=[g.sensitivity - g.sensitivity_ci_low, g.sensitivity_ci_high - g.sensitivity],
            marker="o", markersize=4, linewidth=1.4, capsize=2,
            color=SAMPLE_COLORS[sample], label=sample,
        )
    axis.set_xticks(range(len(order)), order)
    axis.set_xlabel("Visium RNA depth at WES truth allele")
    axis.set_ylabel("Exact-allele sensitivity")
    axis.set_ylim(0, 1.03)
    axis.legend(frameon=False, fontsize=7.5)

    axis = axes[0, 1]
    style_axis(axis); letter(axis, "b")
    g = threshold_source[
        threshold_source.analysis.eq("truth_sensitivity") &
        threshold_source.panel.isin(["defined_1kgp", "non_1kgp"])
    ].copy()
    slots = [(s, p) for s in ["P4", "P6"] for p in ["defined_1kgp", "non_1kgp"]]
    for i, (sample, panel) in enumerate(slots):
        row = g[g["sample"].eq(sample) & g.panel.eq(panel)].iloc[0]
        color = DEFINED if panel == "defined_1kgp" else NONPANEL
        axis.bar(i, row.estimate, color=color, edgecolor=INK, linewidth=0.5, width=0.72, zorder=2)
        axis.errorbar(i, row.estimate, yerr=[[row.estimate-row.ci_low], [row.ci_high-row.estimate]],
                      color=INK, capsize=2, linewidth=0.8, zorder=3)
        axis.text(i, min(1.01, row.estimate + 0.08), f"n={int(row.n):,}", ha="center", fontsize=6.5)
    axis.set_xticks(range(4), ["P4\n1KGP", "P4\nnon-1KGP", "P6\n1KGP", "P6\nnon-1KGP"])
    axis.set_ylabel("Sensitivity at RNA depth >=10")
    axis.set_ylim(0, 1.08)

    axis = axes[1, 0]
    style_axis(axis); letter(axis, "c")
    g = threshold_source[
        threshold_source.analysis.isin(["raw_gt_accuracy", "model_gt_accuracy"]) &
        threshold_source.panel.eq("all")
    ].copy()
    x = np.arange(2); width = 0.32
    for j, (analysis, label, color) in enumerate([
        ("raw_gt_accuracy", "Raw GT", MUTED),
        ("model_gt_accuracy", "SparcalNet class", "#4a3aa7"),
    ]):
        values, labels = [], []
        for sample in ["P4", "P6"]:
            row = g[g["sample"].eq(sample) & g.analysis.eq(analysis)].iloc[0]
            values.append(row.estimate); labels.append(int(row.n))
        axis.bar(x + (j - 0.5) * width, values, width=width, color=color,
                 edgecolor=INK, linewidth=0.5, label=label, zorder=2)
        for xpos, value, n in zip(x + (j - 0.5) * width, values, labels):
            axis.text(xpos, min(1.01, value + 0.055), f"n={n:,}", ha="center", fontsize=6.3, rotation=90)
    axis.set_xticks(x, ["P4", "P6"])
    axis.set_ylabel("Agreement with normal-DNA GT\nat RNA depth >=10")
    axis.set_ylim(0, 1.08)
    axis.legend(frameon=False, fontsize=7.2, loc="lower right")

    axis = axes[1, 1]
    style_axis(axis); letter(axis, "d")
    g = threshold_source[
        threshold_source.analysis.isin(["wes_supported_fraction", "wes_alt_evidence_fraction"]) &
        threshold_source.panel.isin(["defined_1kgp", "upv"])
    ].copy()
    slots = [(s, p) for s in ["P4", "P6"] for p in ["defined_1kgp", "upv"]]
    x = np.arange(len(slots)); width = 0.32
    for j, (analysis, label, color) in enumerate([
        ("wes_supported_fraction", "High-confidence VCF", DEFINED),
        ("wes_alt_evidence_fraction", "VCF or ALT-read evidence", NONPANEL),
    ]):
        values = []
        for sample, panel in slots:
            row = g[g["sample"].eq(sample) & g.panel.eq(panel) & g.analysis.eq(analysis)].iloc[0]
            values.append(row.estimate)
        axis.bar(x + (j - 0.5) * width, values, width=width, color=color,
                 edgecolor=INK, linewidth=0.5, label=label, zorder=2)
    axis.set_xticks(x, ["P4\n1KGP", "P4\nUPV", "P6\n1KGP", "P6\nUPV"])
    axis.set_ylabel("Normal-WES support among calls\nwith pileup depth >=10")
    axis.set_ylim(0, 1.03)
    axis.legend(frameon=False, fontsize=6.8, loc="upper right")
    fig.subplots_adjust(hspace=0.43, wspace=0.33)
    save_figure(fig, "fig_p1_germline_validation")


def upv_figure():
    stage1 = pd.read_csv(UPV / "upv_stage1_threshold_sensitivity.csv")
    gmm = pd.read_csv(UPV / "upv_gmm_sensitivity.csv")
    subgroup = pd.read_csv(UPV / "upv_subgroup_comparison.csv")
    delta = pd.read_csv(DELTA / "upv_delta_baf_summary.csv")
    stage1.to_csv(DERIVED_DIR / "fig_v5_upv_stage1_source.csv", index=False)
    gmm.to_csv(DERIVED_DIR / "fig_v5_upv_gmm_source.csv", index=False)
    subgroup.to_csv(DERIVED_DIR / "fig_v5_upv_evidence_source.csv", index=False)
    delta.to_csv(DERIVED_DIR / "fig_v5_upv_delta_source.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.35))
    axis = axes[0, 0]
    style_axis(axis); letter(axis, "a")
    for i, sample in enumerate(SAMPLES):
        g = stage1[stage1["sample"].eq(sample)]
        production = g[g.is_production_setting].iloc[0].n_upv
        axis.vlines(i, g.n_upv.min(), g.n_upv.max(), color=MUTED, linewidth=3, zorder=2)
        axis.scatter(i, production, marker="D", s=45, color=SAMPLE_COLORS[sample],
                     edgecolor=INK, linewidth=0.5, zorder=3)
    axis.set_yscale("log")
    axis.set_xticks(range(4), SAMPLES)
    axis.set_ylabel("UPVs across Stage-1 threshold grid")

    axis = axes[0, 1]
    style_axis(axis); letter(axis, "b")
    production_n = subgroup[subgroup.subgroup.eq("original_gmm_candidate")].set_index("sample").n
    for i, sample in enumerate(SAMPLES):
        g = gmm[gmm["sample"].eq(sample)]
        axis.vlines(i, g.n_called.min(), g.n_called.max(), color=MUTED, linewidth=3, zorder=2)
        axis.scatter(i, production_n[sample], marker="D", s=45, color=SAMPLE_COLORS[sample],
                     edgecolor=INK, linewidth=0.5, zorder=3)
    axis.set_xticks(range(4), SAMPLES)
    axis.set_ylabel("GMM somatic-candidate UPVs")

    axis = axes[1, 0]
    style_axis(axis); letter(axis, "c")
    g = delta[delta.current_class.eq("all")].set_index("sample").loc[SAMPLES]
    bottoms = np.zeros(4)
    categories = [
        ("n_negative_delta_lt_minus_005", "Negative", "#2a78d6"),
        ("n_near_zero_minus_005_to_005", "Near zero", "#b6b4ae"),
        ("n_positive_delta_gt_005", "Positive", "#e34948"),
    ]
    for column, label, color in categories:
        values = g[column].to_numpy() / g.n_evaluable.to_numpy()
        axis.bar(range(4), values, bottom=bottoms, width=0.68, color=color,
                 edgecolor=INK, linewidth=0.45, label=label, zorder=2)
        bottoms += values
    axis.set_xticks(range(4), [f"{s}\n(n={int(g.loc[s, 'n_evaluable']):,})" for s in SAMPLES])
    axis.set_ylim(0, 1)
    axis.set_ylabel("Evaluable low-BAF UPVs")
    axis.legend(frameon=False, fontsize=7, ncol=3, loc="lower center")

    axis = axes[1, 1]
    style_axis(axis); letter(axis, "d")
    g = subgroup[subgroup.subgroup.eq("all")].set_index("sample").loc[SAMPLES]
    x = np.arange(4); width = 0.32
    axis.bar(x - width/2, g.artifact_mask_fraction, width=width, color="#e34948",
             edgecolor=INK, linewidth=0.5, label="Editing/PON mask", zorder=2)
    axis.bar(x + width/2, g.editing_compatible_fraction, width=width, color="#4a3aa7",
             edgecolor=INK, linewidth=0.5, label="A>G or T>C", zorder=2)
    axis.set_xticks(x, SAMPLES)
    axis.set_ylim(0, 0.62)
    axis.set_ylabel("Fraction of all UPVs")
    axis.legend(frameon=False, fontsize=7.1)
    fig.subplots_adjust(hspace=0.42, wspace=0.33)
    save_figure(fig, "fig_upv_assessment")


def main():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    p1_figure()
    upv_figure()
    print(f"Arial: {ARIAL_PATH}" if HAS_ARIAL else "Arial unavailable; wrote preview assets")


if __name__ == "__main__":
    main()
