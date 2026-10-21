#!/usr/bin/env python3
"""Generate v5 figures for donor-aware DLPFC inference and leakage outcomes."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

PROJECT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
MANUSCRIPT = Path("/data/maiziezhou_lab/leiy4/SPARCAL_pnas_2026")
FIG_DIR = MANUSCRIPT / "figs/v5"
DERIVED_DIR = PROJECT / "data/paper_figs_2026-08-23"
DONOR_DIR = PROJECT / "data/dlpfc_donor_stats_2026-08-23"
LEAK_DIR = PROJECT / "data/leaked_allele_confusion_2026-08-23"

INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
SPARCAL_C = "#e34948"
DONOR_COLORS = {
    "Donor 1": "#2a78d6",
    "Donor 2": "#4a3aa7",
    "Donor 3": "#45a66a",
}
OUTCOME_COLORS = {
    "not detected": "#b6b4ae",
    "germline": "#2a78d6",
    "UPV": "#4a3aa7",
    "somatic": "#e34948",
    "unresolved": "#3f9b5c",
}


def configure_required_font() -> tuple[bool, str | None]:
    explicit = os.environ.get("SPARCAL_ARIAL_FONT")
    local_dir = MANUSCRIPT / "fonts/arial"
    local = (
        font_manager.findSystemFonts(fontpaths=[str(local_dir)])
        if local_dir.is_dir() else []
    )
    explicit_family = (
        font_manager.findSystemFonts(fontpaths=[os.path.dirname(explicit)])
        if explicit and os.path.dirname(explicit) else []
    )
    candidates = ([explicit] + explicit_family) if explicit else (
        local + font_manager.findSystemFonts()
    )
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


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.spines["bottom"].set_color(MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.7, zorder=0)
    axis.tick_params(colors=INK, labelsize=8)


def save_figure(fig, stem: str) -> None:
    suffix = "" if HAS_ARIAL else "_preview"
    for extension in ("pdf", "png"):
        fig.savefig(
            FIG_DIR / f"{stem}{suffix}.{extension}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def donor_effect_figure() -> None:
    detail = pd.read_csv(
        DONOR_DIR / "comparison_values.csv", dtype={"section": str}
    )
    tests = pd.read_csv(DONOR_DIR / "donor_level_tests.csv")

    panels = [
        [
            ("knn_spatial_aug_vs_gene", "Gene\nexpression"),
            ("knn_spatial_aug_vs_gatk250", "GATK\n250 kb"),
            ("knn_spatial_aug_vs_strelka250", "Strelka2\n250 kb"),
        ],
        [
            ("ari_spatial_aug_vs_gene", "Gene\nexpression"),
            ("ari_spatial_aug_vs_gatk250", "GATK\n250 kb"),
            ("ari_spatial_aug_vs_defined250", "1KGP-only\n250 kb"),
            ("ari_defined250_vs_defined_unbinned", "Binning gain:\n250 kb − raw"),
        ],
    ]
    selected_ids = [item[0] for panel in panels for item in panel]
    source = detail[detail["comparison"].isin(selected_ids)].copy()
    source.to_csv(
        DERIVED_DIR / "fig_v5_dlpfc_donor_effects_source.csv", index=False
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 3.15), gridspec_kw={"width_ratios": [0.88, 1.12]}
    )
    rng = np.random.default_rng(230823)
    for panel_index, (axis, comparisons) in enumerate(zip(axes, panels)):
        style_axis(axis)
        axis.axhline(0, color=INK, linewidth=0.9, zorder=1)
        for x_pos, (comparison, _) in enumerate(comparisons):
            frame = source[source["comparison"] == comparison]
            donor_means = frame.groupby("donor", sort=True)["difference"].mean()
            for donor, donor_frame in frame.groupby("donor", sort=True):
                jitter = rng.uniform(-0.115, 0.115, len(donor_frame))
                axis.scatter(
                    x_pos + jitter,
                    donor_frame["difference"],
                    s=18,
                    color=DONOR_COLORS[donor],
                    alpha=0.55,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
                axis.scatter(
                    x_pos,
                    donor_means[donor],
                    s=52,
                    marker="D",
                    color=DONOR_COLORS[donor],
                    edgecolor=INK,
                    linewidth=0.45,
                    zorder=4,
                )
            row = tests.loc[tests["comparison"] == comparison].iloc[0]
            annotation_y = 0.118 if panel_index == 0 else 0.205
            axis.text(
                x_pos,
                annotation_y,
                f"donor P={row.donor_wilcoxon_p:.2g}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color=MUTED,
            )
        axis.set_xticks(range(len(comparisons)))
        axis.set_xticklabels([label for _, label in comparisons], fontsize=7.4)
        axis.set_ylabel(
            "10-NN purity difference" if panel_index == 0 else "ARI difference",
            fontsize=8.5,
            color=INK,
        )
        axis.set_title("a" if panel_index == 0 else "b", loc="left",
                       fontweight="bold", fontsize=11)
    axes[0].set_ylim(-0.055, 0.137)
    axes[1].set_ylim(-0.19, 0.225)

    legend_handles = [
        Line2D(
            [0], [0], marker="D", linestyle="", markersize=6,
            markerfacecolor=color, markeredgecolor=INK, markeredgewidth=0.4,
            label=donor,
        )
        for donor, color in DONOR_COLORS.items()
    ]
    legend_handles.append(
        Line2D(
            [0], [0], marker="o", linestyle="", markersize=5,
            markerfacecolor=MUTED, markeredgecolor="white", alpha=0.55,
            label="section (color = donor)",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=7.5,
    )
    fig.subplots_adjust(wspace=0.32, bottom=0.26)
    save_figure(fig, "fig_dlpfc_donor_reanalysis")


def leakage_figure() -> None:
    summary = pd.read_csv(LEAK_DIR / "confusion_by_section.csv")
    summary.to_csv(
        DERIVED_DIR / "fig_v5_leaked_allele_confusion_source.csv", index=False
    )
    samples = ["P4", "P6", "DCIS1", "DCIS2"]
    outcomes = ["not detected", "germline", "UPV", "somatic", "unresolved"]

    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 3.15), gridspec_kw={"width_ratios": [1.42, 0.58]}
    )
    axis = axes[0]
    style_axis(axis)
    bottoms = np.zeros(len(samples))
    for outcome in outcomes:
        frame = (
            summary[summary["outcome"] == outcome]
            .set_index("sample")
            .loc[samples]
        )
        values = frame["pct_of_leaked"].to_numpy()
        axis.bar(
            np.arange(len(samples)),
            values,
            bottom=bottoms,
            width=0.67,
            color=OUTCOME_COLORS[outcome],
            edgecolor=INK,
            linewidth=0.55,
            label=outcome,
            zorder=2,
        )
        if outcome in {"not detected", "germline"}:
            for x_pos, value, bottom, count in zip(
                range(len(samples)), values, bottoms, frame["n"]
            ):
                if value > 7:
                    text_color = "white" if outcome == "germline" else INK
                    axis.text(
                        x_pos,
                        bottom + value / 2,
                        f"{value:.1f}%\n(n={int(count):,})",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=text_color,
                    )
        bottoms += values
    axis.set_xticks(range(len(samples)), samples)
    axis.set_ylim(0, 100)
    axis.set_ylabel("SpatialSNV 1KGP-matched alleles (%)", fontsize=8.5)
    axis.set_title("a", loc="left", fontweight="bold", fontsize=11)

    axis = axes[1]
    style_axis(axis)
    non_germline = ["UPV", "somatic", "unresolved"]
    x = np.arange(len(samples))
    offsets = np.linspace(-0.22, 0.22, len(non_germline))
    for offset, outcome in zip(offsets, non_germline):
        frame = (
            summary[summary["outcome"] == outcome]
            .set_index("sample")
            .loc[samples]
        )
        axis.bar(
            x + offset,
            frame["n"],
            width=0.2,
            color=OUTCOME_COLORS[outcome],
            edgecolor=INK,
            linewidth=0.5,
            label=outcome,
            zorder=2,
        )
        for xpos, value in zip(x + offset, frame["n"]):
            if value:
                axis.text(
                    xpos, value + 0.04, str(int(value)),
                    ha="center", va="bottom", fontsize=7, color=INK,
                )
    axis.set_xticks(x, samples)
    axis.set_ylim(0, 1.25)
    axis.set_yticks([0, 1])
    axis.set_ylabel("Non-germline assignments (n)", fontsize=8.5)
    axis.set_title("b", loc="left", fontweight="bold", fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.06),
        fontsize=7.3,
    )
    fig.subplots_adjust(wspace=0.34, bottom=0.25)
    save_figure(fig, "fig_leaked_allele_confusion")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    donor_effect_figure()
    leakage_figure()
    font_note = f"Arial: {ARIAL_PATH}" if HAS_ARIAL else "Arial unavailable; wrote preview assets"
    print(font_note)
    print(f"Wrote v5 figures to {FIG_DIR}")
    print(f"Wrote plotted source tables to {DERIVED_DIR}")


if __name__ == "__main__":
    main()
