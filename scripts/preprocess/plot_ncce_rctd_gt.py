#!/usr/bin/env python3
"""Plot NCCE working ground truth as RCTD cell types overlaid with copyKAT tumor.

The working convention was requested by the project owner:

* RCTD labels in ``JapanNCCE_finalCT`` are the cell-type ground truth.
* A copyKAT ``aneuploid`` call is shown as ``tumor`` and overrides the RCTD
  cell-type colour, matching the tumour-plus-cell-type convention of the
  supplied colleague figure.

Source annotations and SpaceRanger outputs are read only.  All figures,
per-section counts, and diagnostics are written beneath
``data/ncce/rctd_gt_comparison``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/data/maiziezhou_lab/leiy4/snv_calling")
SOURCE_ROOT = Path("/data/maiziezhou_lab/Weiman/ST_CNV")
RCTD_ROOT = SOURCE_ROOT / "data" / "JapanNCCE_finalCT"
SPACERANGER_ROOT = SOURCE_ROOT / "data" / "spaceranger_outs"
COPYKAT_ROOT = SOURCE_ROOT / "copyKAT"
SAMPLES = ["6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C", "11A", "11B", "11C", "14A", "14B", "14C"]
COLLEAGUE_PANEL = ["8A", "8B", "8C", "11A", "11B", "11C", "14A", "14B", "14C"]

# These labels and colours reproduce the legend vocabulary of the supplied
# colleague figure.  The panel sections contain no NK/Treg/mast labels.
PLOT_ORDER = [
    "B", "CD8", "DC", "endothelial", "epithelial", "fibroblasts",
    "macrophage", "pericytes", "plasma", "tumor", "Unknown",
]
PALETTE = {
    "B": "#2d6fa3",
    "CD8": "#9eb9d6",
    "DC": "#ed7d21",
    "endothelial": "#f4b36c",
    "epithelial": "#f08a8d",
    "fibroblasts": "#8fd17e",
    "macrophage": "#229943",
    "pericytes": "#ab9abd",
    "plasma": "#78509b",
    "tumor": "#d62d27",
    "Unknown": "#bdbdbd",
}


def load_section(section: str) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Load in-tissue coordinates, RCTD labels, and copyKAT calls for one section."""
    positions_path = SPACERANGER_ROOT / f"NCCE_{section}" / "outs" / "spatial" / "tissue_positions.csv"
    rctd_path = RCTD_ROOT / f"{section}.csv"
    copykat_path = COPYKAT_ROOT / f"NCCE_{section}" / "prediction_aligned.txt"

    positions = pd.read_csv(positions_path, dtype={"barcode": str})
    positions = positions[positions["in_tissue"].astype(int) == 1].copy()
    if positions["barcode"].duplicated().any():
        positions = positions.drop_duplicates("barcode", keep="first")

    rctd = pd.read_csv(rctd_path, index_col=0, dtype={"pixel": str})
    rctd = rctd.rename(columns={"pixel": "barcode"})[["barcode", "cell_type"]]
    rctd["barcode"] = rctd["barcode"].astype(str)
    rctd = rctd.drop_duplicates("barcode", keep="first")

    copykat = pd.read_csv(copykat_path, sep="\t", dtype=str)
    if not {"cell.names", "copykat.pred"}.issubset(copykat.columns):
        raise ValueError(f"Unexpected copyKAT columns in {copykat_path}: {copykat.columns.tolist()}")
    copykat = copykat.rename(columns={"cell.names": "barcode", "copykat.pred": "copykat_prediction"})
    copykat = copykat.drop_duplicates("barcode", keep="first")
    copykat["is_tumor"] = copykat["copykat_prediction"].str.contains("aneuploid", case=False, na=False)

    frame = positions.merge(rctd, on="barcode", how="left").merge(
        copykat[["barcode", "copykat_prediction", "is_tumor"]], on="barcode", how="left"
    )
    frame["rctd_cell_type"] = frame["cell_type"].fillna("Unknown").astype(str).str.strip()
    frame.loc[frame["rctd_cell_type"].isin(["", "NA", "nan"]), "rctd_cell_type"] = "Unknown"
    frame["display_label"] = frame["rctd_cell_type"]
    frame.loc[frame["is_tumor"].fillna(False), "display_label"] = "tumor"
    frame.loc[~frame["display_label"].isin(PLOT_ORDER), "display_label"] = "Unknown"

    diagnostics = {
        "in_tissue_spots": len(frame),
        "rctd_labelled_spots": int((frame["rctd_cell_type"] != "Unknown").sum()),
        "copykat_labelled_spots": int(frame["copykat_prediction"].notna().sum()),
        "copykat_aneuploid_spots": int(frame["is_tumor"].fillna(False).sum()),
    }
    return frame, diagnostics


def draw_section(ax: plt.Axes, frame: pd.DataFrame, section: str, dot_size: float = 36) -> None:
    """Draw a Visium grid with the same orientation and label ordering as the slide."""
    ax.set_facecolor("#f5f5f5")
    # Draw unknown first so labelled spots stay visible at edges.
    ordered = list(reversed(PLOT_ORDER))
    for label in ordered:
        subset = frame[frame["display_label"] == label]
        if subset.empty:
            continue
        ax.scatter(
            subset["array_col"], subset["array_row"],
            s=dot_size, c=PALETTE[label], edgecolors="#8f8f8f", linewidths=0.25,
            alpha=0.97, rasterized=True,
        )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"GC Slice {section}", fontsize=11, weight="bold", pad=4)


def counts_table(frames: Dict[str, pd.DataFrame], diagnostics: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    rows = []
    for section in SAMPLES:
        labels = frames[section]["display_label"].value_counts()
        row = {"section": section, **diagnostics[section]}
        row.update({label: int(labels.get(label, 0)) for label in PLOT_ORDER})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_atlas(frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    patients = ["6", "8", "10", "11", "14"]
    response = {"6": "non-responder", "8": "non-responder", "10": "non-responder", "11": "non-responder", "14": "responder"}
    fig, axes = plt.subplots(len(patients), 3, figsize=(13, 20), constrained_layout=True)
    fig.suptitle("NCCE working ground truth: RCTD cell type + copyKAT aneuploid = tumor", fontsize=17, weight="bold")
    for row, patient in enumerate(patients):
        for col, phase in enumerate("ABC"):
            section = f"{patient}{phase}"
            draw_section(axes[row, col], frames[section], section)
            if col == 0:
                axes[row, col].set_ylabel(f"Patient {patient}\n{response[patient]}", fontsize=11, weight="bold")
    legend = [Patch(facecolor=PALETTE[label], edgecolor="none", label=label) for label in PLOT_ORDER]
    fig.legend(handles=legend, loc="lower center", ncol=6, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_colleague_panel(frames: Dict[str, pd.DataFrame], output_path: Path) -> None:
    """Build a 8/11/14 panel whose content/layout mirrors the supplied slide."""
    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, width_ratios=[1.05, 1.35, 1.6, 1.35])
    fig.suptitle("Cell type composition — RCTD working GT with copyKAT tumor overlay", fontsize=19, weight="bold")

    count_ax = fig.add_subplot(grid[0, 0])
    table = pd.DataFrame({section: frames[section]["display_label"].value_counts() for section in COLLEAGUE_PANEL}).T
    table = table.reindex(columns=PLOT_ORDER, fill_value=0).fillna(0)
    bottom = np.zeros(len(table))
    for label in PLOT_ORDER:
        values = table[label].to_numpy()
        count_ax.bar(table.index, values, bottom=bottom, color=PALETTE[label], width=0.78, label=label)
        bottom += values
    count_ax.set_title("GC", fontsize=12, weight="bold")
    count_ax.set_ylabel("Number of spots")
    count_ax.set_xlabel("Slice")
    count_ax.tick_params(axis="x", rotation=45, labelsize=8)
    count_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, frameon=True)

    slots = {
        "8A": grid[0, 1:3], "8B": grid[0, 3],
        "8C": grid[1, 0], "11A": grid[1, 1:3], "11B": grid[1, 3],
        "11C": grid[2, 0], "14A": grid[2, 1], "14B": grid[2, 2], "14C": grid[2, 3],
    }
    for section, slot in slots.items():
        axis = fig.add_subplot(slot)
        draw_section(axis, frames[section], section, dot_size=30)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "data" / "ncce" / "rctd_gt_comparison",
        help="Project-local output directory",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[str, pd.DataFrame] = {}
    diagnostics: Dict[str, Dict[str, int]] = {}
    for section in SAMPLES:
        frame, section_diagnostics = load_section(section)
        frames[section] = frame
        diagnostics[section] = section_diagnostics

    summary = counts_table(frames, diagnostics)
    summary.to_csv(args.output_dir / "NCCE_RCTD_copyKAT_working_GT_counts.tsv", sep="\t", index=False)
    plot_atlas(frames, args.output_dir / "NCCE_RCTD_copyKAT_working_GT_atlas.png")
    plot_colleague_panel(frames, args.output_dir / "NCCE_RCTD_copyKAT_colleague_panel.png")
    print(f"Wrote figures and counts to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
