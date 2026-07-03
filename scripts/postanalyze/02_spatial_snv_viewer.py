#!/usr/bin/env python3
"""
02_spatial_snv_viewer.py  —  Interactive spatial SNV viewer (matplotlib)

Run locally:
    python 02_spatial_snv_viewer.py \\
        --matrix_dir  /path/to/final_matrices/baseQ0mapQ0 \\
        --positions   /path/to/tissue_positions.csv \\
        [--flip_x]

Controls
--------
  Click a spot          toggle select / deselect
  Drag (no key)         lasso — adds enclosed spots to selection
  Radio (bottom-left)   switch matrix type
  [Clear]               deselect all
  [Export]              save selected barcodes + SNV profile CSVs
"""

import argparse
import os
import pickle
import re
import sys

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.patches import Patch
from matplotlib.path import Path as MplPath
from matplotlib.widgets import Button, LassoSelector, RadioButtons

# ── Palette ──────────────────────────────────────────────────────────────────
CMAP_HEAT   = "YlOrRd"
C_SELECTED  = "#4fc3f7"   # light-blue ring for selected spots
C_UNIQUE    = "#66bb6a"   # green
C_UBIQ      = "#ef5350"   # red
C_SHARED    = "#ffa726"   # orange
C_OTHER     = "#90a4ae"   # grey-blue

BARCODE_RE = re.compile(r"^[ACGTacgt]+-\d+$")
MATRIX_LABELS = ["germline_denovo", "germline_defined", "somatic_denovo"]

# Hardcoded ACCRE paths (used as defaults; override with --positions)
DATASET_POSITIONS = {
    "dcis1": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets"
        "/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38"
        "/DCIS1_output/outs/spatial/tissue_positions.csv"
    ),
    "dcis2": (
        "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets"
        "/spatialSNV/10x-Visium/DCIS2/spaceranger_align_DCIS2_hg38"
        "/DCIS2_output/outs/spatial/tissue_positions.csv"
    ),
}

# ── I/O helpers ───────────────────────────────────────────────────────────────

def read_positions(filepath: str, flip_x: bool = False) -> dict:
    first_col = open(filepath).readline().strip().split(",")[0]
    has_header = not bool(BARCODE_RE.match(first_col))
    df = pd.read_csv(filepath, header=(0 if has_header else None))
    df.columns = range(len(df.columns))
    pos = {}
    for _, row in df.iterrows():
        bc = str(row[0])
        if int(row[1]) != 1:
            continue
        pos[bc] = (float(row[5]), float(row[4]))   # (pxl_col, pxl_row)
    if flip_x and pos:
        max_x = max(v[0] for v in pos.values())
        pos = {bc: (max_x - x, y) for bc, (x, y) in pos.items()}
    return pos


def load_pkl(path: str) -> pd.DataFrame:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, np.ndarray):
        return pd.DataFrame(obj)
    if isinstance(obj, dict):
        all_snvs = sorted({s for snvs in obj.values() for s in snvs})
        barcodes = sorted(obj.keys())
        mat = pd.DataFrame(0, index=barcodes, columns=all_snvs, dtype=np.int8)
        for bc, snvs in obj.items():
            for s in snvs:
                mat.at[bc, s] = 1
        return mat
    raise TypeError(f"Unsupported pkl type: {type(obj)}")


def discover_matrices(matrix_dir: str) -> dict:
    found = {}
    for label in MATRIX_LABELS:
        p = os.path.join(matrix_dir, f"{label}_spot_snv_matrix.pkl")
        if os.path.exists(p):
            found[label] = p
    return found

# ── Analysis ──────────────────────────────────────────────────────────────────

def compute_profile(df: pd.DataFrame, selected: set,
                    ubiq_threshold: float = 0.5, top_n: int = 25) -> dict:
    barcodes  = list(df.index)
    sel_mask  = np.array([b in selected for b in barcodes])
    n_sel, n_total = int(sel_mask.sum()), len(df)

    mat        = df.values.astype(np.int8)
    sel_counts = mat[sel_mask].sum(axis=0)
    non_counts = mat[~sel_mask].sum(axis=0)
    all_counts = mat.sum(axis=0)

    in_sel    = sel_counts > 0
    ubiq_mask = all_counts > (ubiq_threshold * n_total)
    shr_mask  = sel_counts > (0.5 * n_sel) if n_sel > 0 else np.zeros(df.shape[1], bool)
    uniq_mask = in_sel & (non_counts == 0)

    freq = sel_counts / max(n_sel, 1)
    present_idx = np.where(in_sel)[0]
    top_idx = present_idx[np.argsort(-freq[present_idx])[:top_n]]

    rows = []
    for i in top_idx:
        col = df.columns[i]
        cat = ("unique"     if uniq_mask[i] else
               "ubiquitous" if ubiq_mask[i] else
               "shared"     if shr_mask[i]  else "other")
        rows.append({"snv": str(col), "freq": float(freq[i]), "category": cat})

    return {
        "n_sel": n_sel, "n_total": n_total,
        "n_union":  int(in_sel.sum()),
        "n_uniq":   int(uniq_mask.sum()),
        "n_ubiq":   int(ubiq_mask.sum()),
        "n_shr":    int((shr_mask & in_sel).sum()),
        "n_all_snvs": df.shape[1],
        "ubiq_threshold": ubiq_threshold,
        "top_df": pd.DataFrame(rows),
    }

# ── Viewer class ──────────────────────────────────────────────────────────────

class Viewer:

    def __init__(self, matrix_dir: str, positions: dict,
                 ubiq_threshold: float = 0.5, top_n: int = 25):

        self.positions       = positions
        self.ubiq_threshold  = ubiq_threshold
        self.top_n           = top_n
        self.matrix_dir      = matrix_dir
        self.selected        = set()
        self._lasso          = None

        self.matrix_paths = discover_matrices(matrix_dir)
        if not self.matrix_paths:
            sys.exit(f"No *_spot_snv_matrix.pkl found in {matrix_dir}")
        print(f"Matrices found: {list(self.matrix_paths.keys())}")

        self.current_label = list(self.matrix_paths.keys())[0]
        self._load_matrix(self.current_label)
        self._build_figure()
        self._draw_heatmap()
        self._draw_profile()

    # ── data ─────────────────────────────────────────────────────────────────

    def _load_matrix(self, label: str):
        print(f"Loading {label} …")
        df_raw = load_pkl(self.matrix_paths[label])
        common = sorted(set(df_raw.index) & set(self.positions.keys()))
        self.df = df_raw.loc[common]
        self.current_label  = label
        self.barcodes       = np.array(list(self.df.index))
        self.xs             = np.array([self.positions[b][0] for b in self.barcodes])
        self.ys             = np.array([self.positions[b][1] for b in self.barcodes])
        self.snv_counts     = self.df.values.sum(axis=1).astype(int)
        print(f"  {len(self.df):,} spots × {self.df.shape[1]:,} SNVs")

    # ── figure layout ─────────────────────────────────────────────────────────

    def _build_figure(self):
        self.fig = plt.figure(figsize=(17, 9), facecolor="#1a1a2e")

        # Main split: left=heatmap, right=profile (70/30 height for controls)
        outer = gridspec.GridSpec(
            2, 2,
            figure=self.fig,
            width_ratios=[1.15, 1],
            height_ratios=[11, 1],
            hspace=0.06, wspace=0.18,
            left=0.04, right=0.97, top=0.95, bottom=0.04,
        )
        self.ax_map     = self.fig.add_subplot(outer[0, 0])
        self.ax_profile = self.fig.add_subplot(outer[0, 1])
        self.ax_ctrl    = self.fig.add_subplot(outer[1, :])

        for ax in (self.ax_map, self.ax_profile, self.ax_ctrl):
            ax.set_facecolor("#0f0f23")
            for sp in ax.spines.values():
                sp.set_edgecolor("#44446a")

        self.ax_ctrl.set_xticks([])
        self.ax_ctrl.set_yticks([])

        self._build_controls()

    def _build_controls(self):
        n = len(self.matrix_paths)
        radio_w = 0.10 + 0.06 * n

        # RadioButtons for matrix type
        ax_radio = self.fig.add_axes(
            [0.04, 0.005, radio_w, 0.075], facecolor="#1a1a2e"
        )
        active_idx = list(self.matrix_paths.keys()).index(self.current_label)
        self.radio = RadioButtons(
            ax_radio, list(self.matrix_paths.keys()), active=active_idx
        )
        for lbl in self.radio.labels:
            lbl.set_color("white"); lbl.set_fontsize(8.5)
        for c in self.radio.circles:
            c.set_radius(0.06)

        # Clear button
        ax_clr = self.fig.add_axes(
            [0.04 + radio_w + 0.01, 0.01, 0.07, 0.055]
        )
        self.btn_clear = Button(ax_clr, "Clear", color="#2a2a4a", hovercolor="#3a3a6a")
        self.btn_clear.label.set_color("white"); self.btn_clear.label.set_fontsize(9)

        # Export button
        ax_exp = self.fig.add_axes(
            [0.04 + radio_w + 0.10, 0.01, 0.09, 0.055]
        )
        self.btn_export = Button(ax_exp, "Export CSV", color="#2a2a4a", hovercolor="#3a3a6a")
        self.btn_export.label.set_color("white"); self.btn_export.label.set_fontsize(9)

        # Status text
        self.status = self.ax_ctrl.text(
            0.55, 0.5,
            "Click to select a spot. Drag to lasso-select multiple spots.",
            transform=self.ax_ctrl.transAxes,
            ha="center", va="center", fontsize=9, color="#9999cc",
        )

        # Wire callbacks
        self.radio.on_clicked(self._cb_radio)
        self.btn_clear.on_clicked(self._cb_clear)
        self.btn_export.on_clicked(self._cb_export)
        self.fig.canvas.mpl_connect("pick_event", self._cb_pick)

        self._attach_lasso()

    def _attach_lasso(self):
        if self._lasso is not None:
            self._lasso.disconnect_events()
        self._lasso = LassoSelector(
            self.ax_map, self._cb_lasso,
            props={"color": "#4fc3f7", "linewidth": 1.2},
            button=1,
            useblit=True,
        )

    # ── heatmap ───────────────────────────────────────────────────────────────

    def _draw_heatmap(self):
        self.ax_map.cla()
        self.ax_map.set_facecolor("#0f0f23")
        self.ax_map.set_title(
            f"Spatial SNV Map  ·  {self.current_label}",
            color="white", fontsize=11, pad=7,
        )

        sel_mask  = np.array([b in self.selected for b in self.barcodes])
        norm      = plt.Normalize(self.snv_counts.min(), self.snv_counts.max())
        cmap      = plt.get_cmap(CMAP_HEAT)
        facecolors = cmap(norm(self.snv_counts))

        sizes      = np.where(sel_mask, 45, 16)
        linewidths = np.where(sel_mask, 1.5, 0.0)
        edge_cols  = [C_SELECTED if s else "none" for s in sel_mask]

        self.scatter = self.ax_map.scatter(
            self.xs, self.ys,
            c=facecolors,
            s=sizes,
            linewidths=linewidths,
            edgecolors=edge_cols,
            picker=6,           # pick radius in points
            zorder=2,
        )
        self.ax_map.invert_yaxis()

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = self.fig.colorbar(sm, ax=self.ax_map, fraction=0.028, pad=0.02)
        cb.set_label("SNVs / spot", color="white", fontsize=8)
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

        self.ax_map.tick_params(colors="#8888aa", labelsize=7)
        self.ax_map.set_xlabel("pixel col", color="#8888aa", fontsize=8)
        self.ax_map.set_ylabel("pixel row", color="#8888aa", fontsize=8)
        for sp in self.ax_map.spines.values():
            sp.set_edgecolor("#44446a")

        self._attach_lasso()
        self.fig.canvas.draw_idle()

    # ── profile panel ─────────────────────────────────────────────────────────

    def _draw_profile(self):
        self.ax_profile.cla()
        self.ax_profile.set_facecolor("#0f0f23")
        self.ax_profile.tick_params(colors="#8888aa", labelsize=7)
        for sp in self.ax_profile.spines.values():
            sp.set_edgecolor("#44446a")

        if not self.selected:
            self.ax_profile.set_title(
                "SNV Profile", color="white", fontsize=11, pad=7
            )
            self.ax_profile.text(
                0.5, 0.5,
                "No spots selected.\n\nClick a spot or drag to lasso.",
                ha="center", va="center", fontsize=11,
                color="#6666aa", transform=self.ax_profile.transAxes,
            )
            self.fig.canvas.draw_idle()
            return

        prof = compute_profile(
            self.df, self.selected, self.ubiq_threshold, self.top_n
        )
        n_s, n_t = prof["n_sel"], prof["n_total"]
        n_u   = prof["n_union"]
        n_uniq = prof["n_uniq"]
        n_ubiq = prof["n_ubiq"]
        n_shr  = prof["n_shr"]
        n_all  = prof["n_all_snvs"]

        self.ax_profile.set_title(
            f"SNV Profile  ·  {n_s} spot{'s' if n_s>1 else ''} selected",
            color="white", fontsize=11, pad=7,
        )

        # ── summary text box (top-left of axes) ──────────────────────────
        summary = (
            f"Spots:  {n_s} / {n_t}  ({n_s/n_t:.1%} of tissue)\n"
            f"SNVs in selection (union):  {n_u:,}\n"
            f"  Unique to selection:       {n_uniq:,}  ({n_uniq/max(n_u,1):.1%})\n"
            f"  Shared in sel  (>50 %):    {n_shr:,}  ({n_shr/max(n_u,1):.1%})\n"
            f"  Ubiquitous (>{self.ubiq_threshold:.0%} all):  "
            f"{n_ubiq:,}  ({n_ubiq/max(n_all,1):.1%} of all SNVs)"
        )
        self.ax_profile.text(
            0.02, 0.99, summary,
            transform=self.ax_profile.transAxes,
            va="top", ha="left", fontsize=8.5,
            color="#ddddff", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#1e1e40",
                      alpha=0.95, edgecolor="#44446a"),
            zorder=5,
        )

        # ── horizontal bar chart ──────────────────────────────────────────
        top_df = prof["top_df"]
        if top_df.empty:
            self.fig.canvas.draw_idle()
            return

        cat_col = {
            "unique": C_UNIQUE, "ubiquitous": C_UBIQ,
            "shared": C_SHARED, "other": C_OTHER,
        }
        colors = [cat_col.get(c, C_OTHER) for c in top_df["category"]]
        ys     = np.arange(len(top_df))

        self.ax_profile.barh(ys, top_df["freq"], color=colors,
                             height=0.7, alpha=0.88, linewidth=0)

        for i, row in top_df.reset_index(drop=True).iterrows():
            self.ax_profile.text(
                row["freq"] + 0.01, i, row["snv"],
                va="center", ha="left", fontsize=6.5, color="#ccccee",
            )

        self.ax_profile.set_xlim(0, 1.05)
        self.ax_profile.set_ylim(-0.5, len(top_df) - 0.5)
        self.ax_profile.invert_yaxis()
        self.ax_profile.set_xlabel(
            "Frequency within selection", color="#8888aa", fontsize=8
        )
        self.ax_profile.set_yticks([])

        # Legend
        legend_els = [
            Patch(facecolor=C_UNIQUE,  label=f"Unique to selection ({n_uniq:,})"),
            Patch(facecolor=C_UBIQ,    label=f"Ubiquitous >{self.ubiq_threshold:.0%} ({n_ubiq:,})"),
            Patch(facecolor=C_SHARED,  label=f"Shared in sel >50 % ({n_shr:,})"),
            Patch(facecolor=C_OTHER,   label="Other"),
        ]
        self.ax_profile.legend(
            handles=legend_els, loc="lower right",
            fontsize=7.5, framealpha=0.8,
            facecolor="#1e1e40", edgecolor="#44446a",
            labelcolor="white",
        )
        self.fig.canvas.draw_idle()

    # ── callbacks ─────────────────────────────────────────────────────────────

    def _cb_pick(self, event):
        if event.mouseevent.inaxes != self.ax_map:
            return
        if not isinstance(event.artist, PathCollection):
            return
        indices = list(event.ind)
        if not indices:
            return
        # If multiple hits, take the nearest
        if len(indices) > 1:
            mx, my = event.mouseevent.xdata, event.mouseevent.ydata
            d = (self.xs[indices] - mx) ** 2 + (self.ys[indices] - my) ** 2
            indices = [indices[int(np.argmin(d))]]

        bc = self.barcodes[indices[0]]
        if bc in self.selected:
            self.selected.discard(bc)
        else:
            self.selected.add(bc)
        self._refresh()

    def _cb_lasso(self, verts):
        path = MplPath(verts)
        pts  = np.column_stack([self.xs, self.ys])
        hit  = path.contains_points(pts)
        if hit.any():
            self.selected |= set(self.barcodes[hit])
            self._refresh()

    def _cb_radio(self, label):
        if label == self.current_label:
            return
        self._load_matrix(label)
        self.selected = set()
        self._draw_heatmap()
        self._draw_profile()
        self._update_status()

    def _cb_clear(self, _event):
        self.selected = set()
        self._refresh()

    def _cb_export(self, _event):
        if not self.selected:
            self._update_status("Nothing to export — select some spots first.")
            return

        sel_bcs   = sorted(self.selected)
        idx_map   = {b: i for i, b in enumerate(self.barcodes)}
        counts    = [self.snv_counts[idx_map[b]] for b in sel_bcs if b in idx_map]
        bc_df     = pd.DataFrame({"barcode": sel_bcs, "snv_count": counts})

        prof      = compute_profile(self.df, self.selected, self.ubiq_threshold,
                                    top_n=self.df.shape[1])
        bc_path   = os.path.join(
            self.matrix_dir, f"selected_{self.current_label}_barcodes.csv"
        )
        prof_path = os.path.join(
            self.matrix_dir, f"selected_{self.current_label}_snv_profile.csv"
        )
        bc_df.to_csv(bc_path, index=False)
        prof["top_df"].to_csv(prof_path, index=False)

        msg = f"Exported {len(sel_bcs)} barcodes → {os.path.basename(bc_path)}"
        print(msg)
        self._update_status(msg)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _refresh(self):
        self._draw_heatmap()
        self._draw_profile()
        self._update_status()

    def _update_status(self, msg: str = None):
        if msg is None:
            n = len(self.selected)
            if n == 0:
                msg = "Click to select a spot. Drag to lasso-select multiple spots."
            else:
                sample = ", ".join(sorted(self.selected)[:4])
                suffix = f"  … +{n-4} more" if n > 4 else ""
                msg = f"{n} spot{'s' if n>1 else ''} selected: {sample}{suffix}"
        self.status.set_text(msg)
        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix_dir", required=True,
                    help="Folder containing *_spot_snv_matrix.pkl files")
    ap.add_argument("--positions",  required=True,
                    help="Path to tissue_positions.csv")
    ap.add_argument("--flip_x", action="store_true",
                    help="Mirror x-axis (needed for DCIS)")
    ap.add_argument("--ubiq_threshold", type=float, default=0.5)
    ap.add_argument("--top_n",          type=int,   default=25)
    ap.add_argument("--backend", default=None,
                    help="Matplotlib backend, e.g. TkAgg, Qt5Agg")
    args = ap.parse_args()

    if args.backend:
        matplotlib.use(args.backend)

    if not os.path.exists(args.positions):
        sys.exit(f"Positions file not found: {args.positions}")

    print(f"Loading positions from {args.positions} …")
    positions = read_positions(args.positions, flip_x=args.flip_x)
    print(f"  {len(positions):,} in-tissue spots")

    viewer = Viewer(
        matrix_dir     = args.matrix_dir,
        positions      = positions,
        ubiq_threshold = args.ubiq_threshold,
        top_n          = args.top_n,
    )
    viewer.show()


if __name__ == "__main__":
    main()
