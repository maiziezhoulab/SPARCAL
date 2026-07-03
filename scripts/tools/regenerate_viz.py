#!/usr/bin/env python3
"""
Standalone visualization regenerator.
Reads existing per-barcode txt files + all_variant_scores.txt
and reproduces all 5 plots without rerunning the pipeline.

Usage:
  python regenerate_viz.py --dataset p6_tumor --section_id 1
  python regenerate_viz.py --dataset p4_tumor --section_id 2
  python regenerate_viz.py --dataset dcis --section_id dcis1
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from collections import defaultdict
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

DPI = 300
FIGURE_SIZE = (12, 10)
COLORMAP_GERMLINE = ['white', 'lightblue', 'blue', 'darkblue']
COLORMAP_SOMATIC  = ['white', 'lightyellow', 'orange', 'red']
COLORMAP_COMBINED = ['white', 'lightgray', 'gray', 'darkgray']

# ── Path configs (mirrors enhanced filter) ────────────────────────────────────
CONFIGS = {
    "P4_TUMOR": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_positions_list.csv",
            "2": "GSM4565824_P4_rep2_tissue_positions_list.csv",
        },
        "scale_file_patterns": {
            "1": "GSM4565823_P4_rep1_scalefactors_json.json",
            "2": "GSM4565824_P4_rep2_scalefactors_json.json",
        },
        "image_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_hires_image.png",
            "2": "GSM4565824_P4_rep2_tissue_hires_image.png",
        },
        "missing_barcodes_template": "spaceranger_align_rep{section_id}_hg19/Meta_Data/missing_barcodes.txt",
        "coord_transform": "yx",   # x_plot = col5*scale, y_plot = col4*scale
    },
    "P6_TUMOR": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "spaceranger_dir_template": "spaceranger_align_rep{section_id}_hg19",
        "spatial_subdir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_positions_list.csv",
            "2": "GSM4565826_P6_rep2_tissue_positions_list.csv",
        },
        "scale_file_patterns": {
            "1": "GSM4565825_P6_rep1_scalefactors_json.json",
            "2": "GSM4565826_P6_rep2_scalefactors_json.json",
        },
        "image_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_hires_image.png",
            "2": "GSM4565826_P6_rep2_tissue_hires_image.png",
        },
        "missing_barcodes_template": "spaceranger_align_rep{section_id}_hg19/Meta_Data/missing_barcodes.txt",
        "coord_transform": "yx",
    },
    "DLPFC": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc",
        "base_path":   "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12",
        "coord_transform": "yx",
    },
    "DCIS": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data",
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "positions_dir_patterns": {"dcis1": "DCIS1", "dcis2": "DCIS2"},
        "spaceranger_dir_patterns": {
            "dcis1": "DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs",
            "dcis2": "DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs",
        },
        "position_file": "tissue_positions_list.csv",
        "scale_file":    "scalefactors_json.json",
        "image_file":    "tissue_hires_image.png",
        "spatial_subdir": "spatial",
        "coord_transform": "flip_x",  # x-flip then xy
    },
}


def resolve_paths(dataset, section_id, quality_filter):
    cfg = CONFIGS[dataset]
    out_base = cfg["output_base"]

    if dataset == "DCIS":
        output_dir   = os.path.join(out_base, section_id,
                                    f"spatial_filter_purity/{quality_filter}")
        pos_dir      = cfg["positions_dir_patterns"][section_id]
        positions_file = os.path.join(cfg["base_path"], pos_dir, cfg["position_file"])
        outs_dir     = cfg["spaceranger_dir_patterns"][section_id]
        spatial_base = os.path.join(cfg["base_path"], outs_dir, cfg["spatial_subdir"])
        scale_file   = os.path.join(spatial_base, cfg["scale_file"])
        image_file   = os.path.join(spatial_base, cfg["image_file"])
        missing_file = None
    elif dataset == "DLPFC":
        output_dir   = os.path.join(out_base, section_id,
                                    f"spatial_filter_purity/{quality_filter}")
        spatial_base = os.path.join(cfg["base_path"], section_id, "spatial")
        positions_file = os.path.join(spatial_base, "tissue_positions_list.csv")
        scale_file   = os.path.join(spatial_base, "scalefactors_json.json")
        image_file   = os.path.join(spatial_base, "tissue_hires_image.png")
        missing_file = None
    else:  # P4/P6
        output_dir   = os.path.join(out_base, section_id,
                                    f"spatial_filter_purity/{quality_filter}")
        spaceranger_dir = cfg["spaceranger_dir_template"].format(section_id=section_id)
        spatial_base = os.path.join(cfg["base_path"], spaceranger_dir, cfg["spatial_subdir"])
        positions_file = os.path.join(spatial_base, cfg["position_file_patterns"][section_id])
        scale_file   = os.path.join(spatial_base, cfg["scale_file_patterns"][section_id])
        image_file   = os.path.join(spatial_base, cfg["image_file_patterns"][section_id])
        missing_file = os.path.join(cfg["base_path"],
                                    cfg["missing_barcodes_template"].format(section_id=section_id))

    return output_dir, positions_file, scale_file, image_file, missing_file


def load_spot_positions(dataset, positions_file, missing_file):
    """Load spot positions; return dict barcode -> (x_fullres, y_fullres)."""
    out_tissue = set()
    if missing_file and os.path.exists(missing_file):
        with open(missing_file) as f:
            for line in f:
                bc = line.strip().split()[0]
                if bc:
                    out_tissue.add(bc)
        print(f"  Loaded {len(out_tissue)} out-tissue barcodes")

    df = pd.read_csv(positions_file, header=None)
    positions = {}

    if dataset == "DLPFC":
        for _, row in df.iterrows():
            bc = str(row[0])
            if int(row[1]) != 1:
                continue
            if bc in out_tissue:
                continue
            positions[bc] = (float(row[4]), float(row[5]))
    elif dataset == "DCIS":
        xs, rows_to_add = [], []
        for _, row in df.iterrows():
            bc = str(row[0])
            if int(row[1]) != 1:
                continue
            x, y = float(row[4]), float(row[5])
            xs.append(x)
            rows_to_add.append((bc, x, y))
        if xs:
            x_max, x_min = max(xs), min(xs)
            for bc, x, y in rows_to_add:
                positions[bc] = (x_max + x_min - x, y)
    else:  # P4/P6 — no in_tissue column, filter by out_tissue list
        for _, row in df.iterrows():
            bc = str(row[0])
            if bc in out_tissue:
                continue
            positions[bc] = (float(row[4]), float(row[5]))

    print(f"  Loaded {len(positions)} spot positions")
    return positions


def count_variants_from_txt(dir_path, spot_positions):
    """Count variants per barcode from per-barcode txt files in dir_path."""
    counts = {bc: 0 for bc in spot_positions}
    txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
    for fpath in txt_files:
        bc = os.path.basename(fpath).replace(".txt", "")
        if bc not in spot_positions:
            continue
        with open(fpath) as f:
            # header line + data lines
            lines = [l for l in f if not l.startswith("chrom")]
        counts[bc] = len(lines)
    return counts


def plot_variant_map(count_map, spot_positions, scale, img, 
                     variant_type, dataset, section_id, output_file):
    """Render one spatial SNV count map."""
    cmap_colors = {
        "germline":  COLORMAP_GERMLINE,
        "somatic":   COLORMAP_SOMATIC,
        "ambiguous": COLORMAP_COMBINED,
        "combined":  COLORMAP_COMBINED,
    }
    cmap = LinearSegmentedColormap.from_list('', cmap_colors.get(variant_type, COLORMAP_COMBINED))
    max_count = max(count_map.values()) if count_map else 1
    norm = Normalize(vmin=0, vmax=max_count)

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.imshow(img, origin='upper')

    x_coords, y_coords, colors, sizes = [], [], [], []
    for bc, count in count_map.items():
        if bc not in spot_positions:
            continue
        px, py = spot_positions[bc]
        # P4/P6/DLPFC: swap (row,col) → (col*scale, row*scale)
        if dataset in ("P4_TUMOR", "P6_TUMOR", "DLPFC"):
            x_plot = py * scale
            y_plot = px * scale
        else:  # DCIS: already flipped in load; direct scale
            x_plot = px * scale
            y_plot = py * scale
        x_coords.append(x_plot)
        y_coords.append(y_plot)
        colors.append(count)
        sizes.append(30 + count * 20 / max(1, max_count))

    if x_coords:
        sc = ax.scatter(x_coords, y_coords, c=colors, cmap=cmap,
                        norm=norm, s=sizes, alpha=0.7)
        cb = plt.colorbar(sc, ax=ax, shrink=0.6)
        cb.set_label(f'{variant_type.capitalize()} SNV Count')

    non_zero = [c for c in count_map.values() if c > 0]
    stats_text = (
        f"Total Spots: {len(count_map)}\n"
        f"Spots with SNVs: {len(non_zero)}\n"
        f"Total SNVs: {sum(count_map.values())}\n"
        f"Max SNVs/spot: {max_count}\n"
        f"Mean SNVs/spot: {np.mean(list(count_map.values())):.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_title(f"{variant_type.capitalize()} Variants — {dataset} Section {section_id}",
                 fontsize=14)
    ax.set_xticks([]); ax.set_yticks([])
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def plot_violin(scores_file, output_file):
    """Regenerate tricolor violin from all_variant_scores.txt."""
    if not os.path.exists(scores_file):
        print(f"  Skipping violin — {scores_file} not found")
        return

    df = pd.read_csv(scores_file, sep='\t')
    # Only denovo variants get real feature scores
    df = df[df['race'] == 'denovo'].copy()
    if df.empty:
        print("  No denovo variants — skipping violin")
        return

    feature_info = [
        ('f_spatial_uniformity',  'Spatial Uniformity\n(germline)'),
        ('f_global_prevalence',   'Global Prevalence\n(germline)'),
        ('f_purity_correlation',  'Purity Correlation\n(somatic)'),
        ('f_clone_specific_proxy','Clone-Specific Proxy\n(somatic)'),
        ('f_spatial_clustering',  'Spatial Clustering\n(somatic)'),
    ]
    if 'f_cnv_consistency' in df.columns and df['f_cnv_consistency'].notna().any():
        feature_info.append(('f_cnv_consistency', 'CNV Consistency\n(somatic)'))

    germline_df  = df[df['classification'] == 'germline']
    somatic_df   = df[df['classification'] == 'somatic']
    ambiguous_df = df[df['classification'] == 'ambiguous']

    import random
    random.seed(42)
    n_other = max(1, int(len(ambiguous_df) * 0.10))
    other_idx = random.sample(list(ambiguous_df.index), min(n_other, len(ambiguous_df)))
    other_df = ambiguous_df.loc[other_idx]

    groups = [
        (germline_df,  '#4878CF', 'Germline'),
        (somatic_df,   '#D65F5F', 'Somatic'),
        (other_df,     '#888888', 'Other (10%)'),
    ]

    n_features = len(feature_info)
    ncols = min(4, n_features)
    nrows = int(np.ceil(n_features / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), squeeze=False)
    rng = np.random.default_rng(42)

    for feat_idx, (feat_col, feat_label) in enumerate(feature_info):
        if feat_col not in df.columns:
            continue
        row, col = feat_idx // ncols, feat_idx % ncols
        ax = axes[row][col]

        plot_data, plot_labels, plot_colors = [], [], []
        for grp_df, grp_color, grp_label in groups:
            vals = grp_df[feat_col].dropna().tolist()
            plot_data.append(vals if vals else [float('nan')])
            plot_labels.append(f"{grp_label}\nn={len(vals)}")
            plot_colors.append(grp_color)

        valid_data = [[x for x in d if not np.isnan(x)] for d in plot_data]
        positions_to_draw = [i for i, d in enumerate(valid_data) if len(d) >= 2]

        if positions_to_draw:
            parts = ax.violinplot(
                [valid_data[i] for i in positions_to_draw],
                positions=positions_to_draw,
                showmedians=True, showextrema=False,
            )
            for body, pos in zip(parts['bodies'], positions_to_draw):
                body.set_facecolor(plot_colors[pos])
                body.set_alpha(0.6)
            parts['cmedians'].set_color('black')
            parts['cmedians'].set_linewidth(1.5)

        for i, (vals, color) in enumerate(zip(valid_data, plot_colors)):
            if vals:
                jitter = rng.uniform(-0.08, 0.08, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals,
                           color=color, alpha=0.3, s=6, zorder=3)

        g_vals = germline_df[feat_col].dropna().tolist()
        s_vals = somatic_df[feat_col].dropna().tolist()
        if len(g_vals) >= 3 and len(s_vals) >= 3:
            _, pval = scipy.stats.mannwhitneyu(g_vals, s_vals, alternative='two-sided')
            ax.text(0.98, 0.98, f"p={pval:.2e}", transform=ax.transAxes,
                    ha='right', va='top', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', ec='black', lw=0.8))

        ax.set_title(feat_label, fontsize=10, fontweight='bold')
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=8)
        ax.set_ylabel("Score", fontsize=9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    for feat_idx in range(n_features, nrows * ncols):
        axes[feat_idx // ncols][feat_idx % ncols].set_visible(False)

    fig.suptitle(
        f"Feature Score Distributions\n"
        f"Germline n={len(germline_df)} | Somatic n={len(somatic_df)} | Other (10%) n={len(other_df)}",
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout()
    plt.savefig(output_file, dpi=DPI, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['dlpfc', 'p4_tumor', 'p6_tumor', 'dcis'])
    parser.add_argument('--section_id', required=True)
    parser.add_argument('--quality_filter', default='baseQ0mapQ0')
    args = parser.parse_args()

    dataset    = args.dataset.upper().replace("_TUMOR", "_TUMOR")
    # normalize dataset name
    ds_map = {'DLPFC': 'DLPFC', 'P4_TUMOR': 'P4_TUMOR',
              'P6_TUMOR': 'P6_TUMOR', 'DCIS': 'DCIS'}
    dataset = ds_map.get(args.dataset.upper(), args.dataset.upper())

    section_id = args.section_id
    qf         = args.quality_filter

    print(f"\n=== Regenerating visualizations: {dataset} sec {section_id} ===")

    output_dir, positions_file, scale_file, image_file, missing_file = \
        resolve_paths(dataset, section_id, qf)

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)

    print(f"Output dir : {output_dir}")
    print(f"Viz dir    : {viz_dir}")

    # ── Load spatial data ──────────────────────────────────────────────────
    print("\nLoading spot positions...")
    spot_positions = load_spot_positions(dataset, positions_file, missing_file)

    print("Loading scale factors...")
    with open(scale_file) as f:
        sf = json.load(f)
    scale = sf.get('tissue_hires_scalef', 1.0)
    print(f"  hires_scalef = {scale}")

    print("Loading tissue image...")
    img = plt.imread(image_file)

    # ── Count variants per spot ────────────────────────────────────────────
    print("\nCounting variants per spot...")
    germline_dir  = os.path.join(output_dir, "germline")
    somatic_dir   = os.path.join(output_dir, "somatic")
    ambiguous_dir = os.path.join(output_dir, "ambiguous")

    germline_counts  = count_variants_from_txt(germline_dir,  spot_positions)
    somatic_counts   = count_variants_from_txt(somatic_dir,   spot_positions)
    ambiguous_counts = count_variants_from_txt(ambiguous_dir, spot_positions)
    combined_counts  = {bc: germline_counts.get(bc, 0) +
                             somatic_counts.get(bc, 0) +
                             ambiguous_counts.get(bc, 0)
                        for bc in spot_positions}

    total_g = sum(germline_counts.values())
    total_s = sum(somatic_counts.values())
    total_a = sum(ambiguous_counts.values())
    print(f"  Germline  SNV instances: {total_g}")
    print(f"  Somatic   SNV instances: {total_s}")
    print(f"  Ambiguous SNV instances: {total_a}")

    # ── Generate spatial plots ─────────────────────────────────────────────
    print("\nGenerating spatial plots...")
    plot_variant_map(germline_counts,  spot_positions, scale, img,
                     "germline",  dataset, section_id,
                     os.path.join(viz_dir, "germline_variants.png"))
    plot_variant_map(somatic_counts,   spot_positions, scale, img,
                     "somatic",   dataset, section_id,
                     os.path.join(viz_dir, "somatic_variants.png"))
    plot_variant_map(ambiguous_counts, spot_positions, scale, img,
                     "ambiguous", dataset, section_id,
                     os.path.join(viz_dir, "ambiguous_variants.png"))
    plot_variant_map(combined_counts,  spot_positions, scale, img,
                     "combined",  dataset, section_id,
                     os.path.join(viz_dir, "all_variants.png"))

    # ── Generate violin plot ───────────────────────────────────────────────
    print("\nGenerating violin plot...")
    scores_file = os.path.join(output_dir, "all_variant_scores.txt")
    plot_violin(scores_file, os.path.join(viz_dir, "feature_violins_tricolor.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()