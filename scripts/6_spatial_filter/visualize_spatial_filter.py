#!/usr/bin/env python3
"""
Standalone spatial SNV filter visualization.

Reads existing outputs from run_spatial_snv_filter_enhanced.py and overlays
per-spot SNV counts on the tissue hires image.  Run this after the filter
completes, or re-run it alone to regenerate plots without repeating filtering.

Usage:
    # Re-visualize after filter has run (viz-only)
    python visualize_spatial_filter.py \
        --dataset dcis --section_id dcis1 --quality_filter baseQ0mapQ0

    # Override output directory
    python visualize_spatial_filter.py \
        --dataset p4_tumor --section_id 1 --quality_filter baseQ0mapQ0 \
        --output_dir /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0
"""

import os
import sys
import csv
import glob
import gzip
import json
import logging
import argparse
from typing import Dict, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("spatial_viz")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FIGURE_SIZE = (12, 10)
DPI = 300
COLORMAP_GERMLINE = ["white", "lightblue", "blue", "darkblue"]
COLORMAP_SOMATIC  = ["white", "lightyellow", "orange", "red"]
COLORMAP_COMBINED = ["white", "lightgray", "gray", "darkgray"]

# ---------------------------------------------------------------------------
# Dataset configuration  (mirrors run_spatial_snv_filter_enhanced.py)
# ---------------------------------------------------------------------------
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path":   "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC12",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc",
        "spatial_subdir": "spatial",
        "position_file": "tissue_positions_list.csv",
        "scale_file":    "scalefactors_json.json",
        "image_file":    "tissue_hires_image.png",
        "has_header": False,
        "x_flip":     False,
    },
    "P4_TUMOR": {
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
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
        "has_header": False,
        "x_flip":     False,
    },
    "P6_TUMOR": {
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
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
        "has_header": False,
        "x_flip":     False,
    },
    "DCIS": {
        "base_path":   "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data",
        "spaceranger_dir_patterns": {
            "dcis1": "DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs",
            "dcis2": "DCIS2/spaceranger_align_DCIS2_hg38/DCIS2_output/outs",
        },
        "spatial_subdir": "spatial",
        "position_file": "tissue_positions.csv",
        "scale_file":    "scalefactors_json.json",
        "image_file":    "tissue_hires_image.png",
        "has_header": True,
        "x_flip":     True,
    },
}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def resolve_spatial_files(dataset_upper: str, section_id: str) -> Tuple[str, str, str, bool, bool]:
    """
    Return (positions_file, scale_file, image_file, has_header, x_flip).
    """
    cfg = DATASET_CONFIGS[dataset_upper]

    if dataset_upper == "DLPFC":
        spatial_base = os.path.join(cfg["base_path"], section_id, cfg["spatial_subdir"])
        pos   = os.path.join(spatial_base, cfg["position_file"])
        scale = os.path.join(spatial_base, cfg["scale_file"])
        img   = os.path.join(spatial_base, cfg["image_file"])

    elif dataset_upper == "DCIS":
        outs_dir     = cfg["spaceranger_dir_patterns"][section_id]
        spatial_base = os.path.join(cfg["base_path"], outs_dir, cfg["spatial_subdir"])
        pos   = os.path.join(spatial_base, cfg["position_file"])
        scale = os.path.join(spatial_base, cfg["scale_file"])
        img   = os.path.join(spatial_base, cfg["image_file"])

    else:  # P4_TUMOR / P6_TUMOR
        spaceranger_dir = cfg["spaceranger_dir_template"].format(section_id=section_id)
        spatial_base = os.path.join(cfg["base_path"], spaceranger_dir, cfg["spatial_subdir"])
        pos   = os.path.join(spatial_base, cfg["position_file_patterns"][section_id])
        scale = os.path.join(spatial_base, cfg["scale_file_patterns"][section_id])
        img   = os.path.join(spatial_base, cfg["image_file_patterns"][section_id])

    return pos, scale, img, cfg["has_header"], cfg["x_flip"]


def resolve_output_dir(dataset_upper: str, section_id: str, quality_filter: str,
                       override: str = None) -> str:
    if override:
        return override
    cfg = DATASET_CONFIGS[dataset_upper]
    return os.path.join(cfg["output_base"], section_id,
                        f"spatial_filter_purity/{quality_filter}")


def resolve_snv_pos_dir(dataset_upper: str, section_id: str, quality_filter: str) -> str:
    cfg = DATASET_CONFIGS[dataset_upper]
    return os.path.join(cfg["output_base"], section_id,
                        f"output_VCFs/spotprofiles/{quality_filter}/vcf_by_spot")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spot_positions(positions_file: str, has_header: bool) -> Dict[str, Tuple[float, float]]:
    """
    Load in-tissue spot positions.
    Returns {barcode: (pxl_col, pxl_row)} where pxl_col is horizontal and
    pxl_row is vertical in fullres pixel coordinates.

    CSV layout (header-less):  barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
    CSV layout (with header):  barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
    """
    positions: Dict[str, Tuple[float, float]] = {}

    with open(positions_file, newline="") as fh:
        if has_header:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    if int(float(row["in_tissue"])) != 1:
                        continue
                    bc      = row["barcode"]
                    pxl_row = float(row["pxl_row_in_fullres"])
                    pxl_col = float(row["pxl_col_in_fullres"])
                    positions[bc] = (pxl_col, pxl_row)
                except (KeyError, ValueError):
                    continue
        else:
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    if int(row[1]) != 1:
                        continue
                    bc      = row[0]
                    pxl_row = float(row[4])
                    pxl_col = float(row[5])
                    positions[bc] = (pxl_col, pxl_row)
                except (ValueError, IndexError):
                    continue

    logger.info(f"Loaded {len(positions)} in-tissue spot positions")
    return positions


def load_scale_factor(scale_file: str) -> float:
    with open(scale_file) as fh:
        sf = json.load(fh)
    return float(sf.get("tissue_hires_scalef", 1.0))


def load_classifications(scores_file: str) -> Dict[str, str]:
    """
    Read all_variant_scores.txt produced by the filter.
    Returns {snv_key: 'germline'|'somatic'|'ambiguous'}.
    """
    classifications: Dict[str, str] = {}
    if not os.path.exists(scores_file):
        logger.warning(f"Scores file not found: {scores_file}")
        return classifications

    with open(scores_file) as fh:
        for i, line in enumerate(fh):
            if i == 0:
                continue  # skip header
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            snv_key = parts[0]
            cls     = parts[3]  # 'germline', 'somatic', or 'ambiguous'
            if cls in ("germline", "somatic", "ambiguous"):
                classifications[snv_key] = cls

    logger.info(f"Loaded {len(classifications)} classified variants from {scores_file}")
    return classifications


def load_spot_snvs(snv_pos_dir: str,
                   positions: Dict[str, Tuple[float, float]]) -> Dict[str, Set[str]]:
    """
    Read per-barcode vcf.gz files from the filter input SNV directory.
    Only processes barcodes that are in `positions` (in-tissue spots).
    Returns {barcode: set(snv_key)} where snv_key = "{chrom}_{pos}" (no chr prefix).
    """
    vcf_files = glob.glob(os.path.join(snv_pos_dir, "*.vcf.gz"))
    logger.info(f"Found {len(vcf_files)} SNV vcf.gz files in {snv_pos_dir}")

    spot_snvs: Dict[str, Set[str]] = {}
    for vcf in vcf_files:
        bc = os.path.basename(vcf).replace(".vcf.gz", "")
        if bc not in positions:
            continue
        snvs: Set[str] = set()
        try:
            with gzip.open(vcf, "rt") as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 5:
                        continue
                    chrom = parts[0].replace("chr", "")
                    pos   = parts[1]
                    alt   = parts[4]
                    if "," in alt:
                        continue  # skip multi-allelic
                    snvs.add(f"{chrom}_{pos}")
        except Exception as exc:
            logger.warning(f"Error reading {vcf}: {exc}")
        spot_snvs[bc] = snvs

    logger.info(f"Loaded SNV sets for {len(spot_snvs)} barcodes")
    return spot_snvs


def build_count_maps(
    spot_snvs: Dict[str, Set[str]],
    classifications: Dict[str, str],
    positions: Dict[str, Tuple[float, float]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Return (germline_map, somatic_map, ambiguous_map), each {barcode: count}.
    All in-tissue barcodes are present in the maps (0 if no SNVs).
    """
    germline_map  = {bc: 0 for bc in positions}
    somatic_map   = {bc: 0 for bc in positions}
    ambiguous_map = {bc: 0 for bc in positions}

    for bc, snvs in spot_snvs.items():
        if bc not in positions:
            continue
        for snv_key in snvs:
            cls = classifications.get(snv_key)
            if cls == "germline":
                germline_map[bc] += 1
            elif cls == "somatic":
                somatic_map[bc] += 1
            elif cls == "ambiguous":
                ambiguous_map[bc] += 1

    return germline_map, somatic_map, ambiguous_map


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_spot_counts(
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    scale: float,
    image_path: str,
    x_flip: bool,
    title: str,
    out_path: str,
    colormap: list,
) -> None:
    """
    Overlay per-spot SNV counts on the tissue hires image.

    Coordinate convention:
        positions stores (pxl_col, pxl_row) in fullres pixels.
        pxl_col (horizontal) → scatter x-axis  (multiply by scale)
        pxl_row (vertical)   → scatter y-axis  (multiply by scale)
        x_flip=True          → ax.invert_xaxis() for correct tissue orientation
    No image rotation is applied.
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image not found: {image_path} — skipping")
        return

    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.imshow(img, origin="upper")

    max_count = max(count_map.values()) if count_map else 0
    if max_count == 0:
        max_count = 1

    xs, ys, colors, sizes = [], [], [], []
    for bc, (pxl_col, pxl_row) in positions.items():
        count = count_map.get(bc, 0)
        xs.append(pxl_col * scale)
        ys.append(pxl_row * scale)
        colors.append(count)
        sizes.append(30 + count * 20.0 / max_count)

    cmap = LinearSegmentedColormap.from_list("snv_cmap", colormap)
    norm = Normalize(vmin=0, vmax=max(1, max(colors) if colors else 1))

    sc = ax.scatter(xs, ys, c=colors, cmap=cmap, norm=norm, s=sizes, alpha=0.75)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("SNV count per spot")

    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    if x_flip:
        ax.invert_xaxis()

    counts_arr = np.array([count_map.get(bc, 0) for bc in positions])
    nz = counts_arr[counts_arr > 0]
    stats = (
        f"In-tissue spots: {len(positions)}\n"
        f"Spots with SNVs: {len(nz)}\n"
        f"Total SNVs: {int(counts_arr.sum())}\n"
        f"Max/spot: {int(counts_arr.max())}\n"
        f"Mean/spot: {counts_arr.mean():.1f}"
    )
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Standalone spatial SNV filter visualization.\n"
            "Reads existing filter outputs and overlays spot SNV counts on tissue image.\n"
            "Run after run_spatial_snv_filter_enhanced.py, or standalone to regenerate plots."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        choices=["dlpfc", "p4_tumor", "p6_tumor", "dcis"],
                        help="Dataset name (case-insensitive)")
    parser.add_argument("--section_id", required=True,
                        help="Section ID (e.g., 1 for P4/P6, 151507 for DLPFC, dcis1 for DCIS)")
    parser.add_argument("--quality_filter", required=True,
                        help="Quality filter string (e.g., baseQ0mapQ0)")
    parser.add_argument("--output_dir", default=None,
                        help="Filter output directory (default: auto-resolved from dataset/section_id)")
    args = parser.parse_args()

    dataset_upper = args.dataset.upper()
    if dataset_upper not in DATASET_CONFIGS:
        logger.error(f"Unknown dataset: {args.dataset}")
        sys.exit(1)

    output_dir  = resolve_output_dir(dataset_upper, args.section_id, args.quality_filter, args.output_dir)
    snv_pos_dir = resolve_snv_pos_dir(dataset_upper, args.section_id, args.quality_filter)
    pos_file, scale_file, img_file, has_header, x_flip = resolve_spatial_files(dataset_upper, args.section_id)
    viz_dir     = os.path.join(output_dir, "visualizations")
    scores_file = os.path.join(output_dir, "all_variant_scores.txt")

    logger.info("=" * 60)
    logger.info(f"Dataset:        {dataset_upper}")
    logger.info(f"Section:        {args.section_id}")
    logger.info(f"Quality filter: {args.quality_filter}")
    logger.info(f"Output dir:     {output_dir}")
    logger.info(f"SNV pos dir:    {snv_pos_dir}")
    logger.info(f"Positions file: {pos_file}")
    logger.info(f"Image file:     {img_file}")
    logger.info(f"x_flip:         {x_flip}")
    logger.info("=" * 60)

    if not os.path.exists(scores_file):
        logger.error(
            f"Filter output not found: {scores_file}\n"
            f"Run run_spatial_snv_filter_enhanced.py first."
        )
        sys.exit(1)

    positions = load_spot_positions(pos_file, has_header)
    if not positions:
        logger.error("No in-tissue positions loaded.")
        sys.exit(1)

    try:
        scale = load_scale_factor(scale_file)
    except Exception as exc:
        logger.warning(f"Could not load scale factor ({exc}); using 1.0")
        scale = 1.0

    classifications = load_classifications(scores_file)
    spot_snvs       = load_spot_snvs(snv_pos_dir, positions)
    germline_map, somatic_map, ambiguous_map = build_count_maps(spot_snvs, classifications, positions)
    combined_map = {
        bc: germline_map.get(bc, 0) + somatic_map.get(bc, 0) + ambiguous_map.get(bc, 0)
        for bc in positions
    }

    os.makedirs(viz_dir, exist_ok=True)

    label = f"{dataset_upper} {args.section_id} ({args.quality_filter})"
    plots = [
        (germline_map,  "germline_variants.png",  f"Germline Variants — {label}",  COLORMAP_GERMLINE),
        (somatic_map,   "somatic_variants.png",   f"Somatic Variants — {label}",   COLORMAP_SOMATIC),
        (ambiguous_map, "ambiguous_variants.png", f"Ambiguous Variants — {label}", COLORMAP_COMBINED),
        (combined_map,  "all_variants.png",       f"All Variants — {label}",       COLORMAP_COMBINED),
    ]
    for count_map, fname, title, cmap in plots:
        out_path = os.path.join(viz_dir, fname)
        visualize_spot_counts(
            count_map=count_map,
            positions=positions,
            scale=scale,
            image_path=img_file,
            x_flip=x_flip,
            title=title,
            out_path=out_path,
            colormap=cmap,
        )

    logger.info(f"Done. Visualizations in: {viz_dir}")


if __name__ == "__main__":
    main()
