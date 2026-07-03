#!/usr/bin/env python3
"""
VCF Visualizer (per-barcode presented): spatial SNV distribution from pre-computed per-barcode TXT files.

Each barcode TXT file has tab-separated columns: chrom  pos  ref  alt  race
Counts are derived directly from these files — no BAM scan or VCF parsing needed.

Usage:
    python vcf_visualizer_per_barcode_presented.py \
        --barcode-dir .../spatial_filter_purity/baseQ0mapQ0/germline \
        --race denovo --dataset DCIS --section-id 1

    python vcf_visualizer_per_barcode_presented.py \
        --barcode-dir .../spatial_filter_purity/baseQ0mapQ0/somatic \
        --race denovo --dataset DCIS --section-id 1 --no-bed
"""

import os
import csv
import glob
import json
import bisect
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
}

DATASET_CONFIGS = {
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "section_ids": ["1", "2"],
        "spatial_dir": "spaceranger_align_rep{section_id}_hg19/Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_positions_list.csv",
            "2": "GSM4565824_P4_rep2_tissue_positions_list.csv",
        },
        "scale_factor_file_patterns": {
            "1": "GSM4565823_P4_rep1_scalefactors_json.json",
            "2": "GSM4565824_P4_rep2_scalefactors_json.json",
        },
        "image_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_hires_image.png",
            "2": "GSM4565824_P4_rep2_tissue_hires_image.png",
        },
        "position_file_has_header": False,
        "x_flip": False,
        "bed_file": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed",
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "section_ids": ["1", "2"],
        "spatial_dir": "spaceranger_align_rep{section_id}_hg19/Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_positions_list.csv",
            "2": "GSM4565826_P6_rep2_tissue_positions_list.csv",
        },
        "scale_factor_file_patterns": {
            "1": "GSM4565825_P6_rep1_scalefactors_json.json",
            "2": "GSM4565826_P6_rep2_scalefactors_json.json",
        },
        "image_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_hires_image.png",
            "2": "GSM4565826_P6_rep2_tissue_hires_image.png",
        },
        "position_file_has_header": False,
        "x_flip": False,
        "bed_file": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed",
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "section_ids": ["1", "2"],
        "spatial_dir": "DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/spatial",
        "position_file": "tissue_positions.csv",
        "scale_factor_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "position_file_has_header": True,
        "x_flip": True,
        "bed_file": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed",
    },
}

COLORMAP    = ["#e6e6e6", "#0000ff"]
FIGURE_SIZE = (16, 16)
DPI         = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vcf_vis_pb")


# ============================================================================
# Spatial utilities
# ============================================================================

def resolve_spatial_paths(dataset: str, section_id: str) -> Dict:
    cfg = DATASET_CONFIGS[dataset]
    spatial_dir = os.path.join(
        cfg["base_path"], cfg["spatial_dir"].format(section_id=section_id)
    )
    if "position_file_patterns" in cfg:
        position_file     = os.path.join(spatial_dir, cfg["position_file_patterns"][section_id])
        scale_factor_file = os.path.join(spatial_dir, cfg["scale_factor_file_patterns"][section_id])
        image_file        = os.path.join(spatial_dir, cfg["image_file_patterns"][section_id])
    else:
        position_file     = os.path.join(spatial_dir, cfg["position_file"])
        scale_factor_file = os.path.join(spatial_dir, cfg["scale_factor_file"])
        image_file        = os.path.join(spatial_dir, cfg["image_file"])
    return {
        "position_file":     position_file,
        "scale_factor_file": scale_factor_file,
        "image_file":        image_file,
        "has_header":        cfg["position_file_has_header"],
        "x_flip":            cfg["x_flip"],
    }


def load_spot_positions(positions_file: str, has_header: bool) -> Dict[str, Tuple[float, float]]:
    positions: Dict[str, Tuple[float, float]] = {}
    with open(positions_file, "r", newline="") as fh:
        if has_header:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    if int(float(row["in_tissue"])) != 1:
                        continue
                    positions[row["barcode"]] = (float(row["pxl_col_in_fullres"]), float(row["pxl_row_in_fullres"]))
                except (KeyError, ValueError):
                    continue
        else:
            for row in csv.reader(fh):
                if len(row) < 6:
                    continue
                try:
                    if int(row[1]) != 1:
                        continue
                    positions[row[0]] = (float(row[5]), float(row[4]))
                except ValueError:
                    continue
    logger.info(f"Loaded {len(positions)} in-tissue spot positions")
    return positions


def load_scale_factor(scale_factor_file: str) -> float:
    with open(scale_factor_file) as fh:
        sf = json.load(fh)
    return float(sf.get("tissue_hires_scalef", 1.0))


# ============================================================================
# Exome BED filtering
# ============================================================================

# BedIndex: chrom -> (sorted starts list, intervals list)
BedIndex = Dict[str, Tuple[List[int], List[Tuple[int, int]]]]


def load_bed_index(bed_path: str) -> BedIndex:
    """Load BED file into a fast-lookup index: chrom -> (starts, intervals)."""
    raw: Dict[str, List[Tuple[int, int]]] = {}
    with open(bed_path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                raw.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    index: BedIndex = {}
    for chrom, ivs in raw.items():
        ivs.sort()
        index[chrom] = ([s for s, _ in ivs], ivs)
    total = sum(len(v[1]) for v in index.values())
    logger.info(f"Loaded BED: {total:,} intervals across {len(index)} chroms from {bed_path}")
    return index


def _in_bed(chrom: str, pos: int, bed_index: BedIndex) -> bool:
    """Return True if the 1-based VCF position falls within any BED interval."""
    entry = bed_index.get(chrom) or bed_index.get(
        chrom[3:] if chrom.startswith("chr") else "chr" + chrom
    )
    if entry is None:
        return False
    starts, ivs = entry
    pos0 = pos - 1  # convert to 0-based
    idx = bisect.bisect_right(starts, pos0) - 1
    return idx >= 0 and ivs[idx][0] <= pos0 < ivs[idx][1]


# ============================================================================
# Per-barcode counting
# ============================================================================

def _process_one_barcode(
    txt_file: str,
    race_filter: Optional[str],
    bed_index: Optional[BedIndex],
) -> Tuple[str, int, Set[Tuple[str, int]]]:
    """Read one barcode TXT and return (barcode, count, unique_positions)."""
    barcode = os.path.splitext(os.path.basename(txt_file))[0]
    count = 0
    positions: Set[Tuple[str, int]] = set()
    with open(txt_file) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if race_filter and row.get("race") != race_filter:
                continue
            try:
                chrom = row["chrom"]
                pos   = int(row["pos"])
            except (KeyError, ValueError):
                continue
            if bed_index is not None and not _in_bed(chrom, pos, bed_index):
                continue
            count += 1
            positions.add((chrom, pos))
    return barcode, count, positions


def count_snvs_per_barcode(
    barcode_dir: str,
    race_filter: Optional[str],
    bed_index: Optional[BedIndex],
    max_workers: int,
) -> Tuple[Dict[str, int], int]:
    """
    Read all *.txt barcode files in barcode_dir and return:
      (count_map, unique_snv_count)
    where count_map is barcode -> SNV count and unique_snv_count is the number
    of distinct (chrom, pos) positions across all barcodes.
    """
    txt_files = sorted(glob.glob(os.path.join(barcode_dir, "*.txt")))
    if not txt_files:
        raise RuntimeError(f"No .txt files found in {barcode_dir}")
    logger.info(f"Processing {len(txt_files)} barcode files from {barcode_dir}")

    count_map: Dict[str, int] = {}
    all_positions: Set[Tuple[str, int]] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_process_one_barcode, f, race_filter, bed_index): f
            for f in txt_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Reading barcode files"):
            try:
                barcode, cnt, positions = fut.result()
                count_map[barcode] = cnt
                all_positions.update(positions)
            except Exception as exc:
                logger.warning(f"{futures[fut]}: {exc}")

    logger.info(f"Done: {len(count_map)} barcodes, {len(all_positions):,} unique positions")
    return count_map, len(all_positions)


# ============================================================================
# Visualization & output
# ============================================================================

def visualize_spot_counts(
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    scale: float,
    x_flip: bool,
    title: str,
    unique_snvs: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    xs, ys, colors = [], [], []
    counts_for_stats = []
    max_count = max(count_map.values()) if count_map else 1

    for barcode, (px_col, px_row) in positions.items():
        c = count_map.get(barcode, 0)
        xs.append(px_col * scale)
        ys.append(px_row * scale)
        colors.append(c)
        counts_for_stats.append(c)

    cmap = LinearSegmentedColormap.from_list("snv_cmap", COLORMAP)
    norm = Normalize(vmin=0, vmax=max(1, max(colors) if colors else 1))
    sizes = [30 + (c * 20.0 / max_count) for c in colors]

    sc = ax.scatter(xs, ys, c=colors, cmap=cmap, norm=norm, s=sizes, alpha=0.75)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label("SNV count per spot")

    ax.set_title(title, fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])

    if x_flip:
        ax.invert_xaxis()

    counts_arr = np.array(counts_for_stats) if counts_for_stats else np.array([0])
    nz = counts_arr[counts_arr > 0]
    stats = (
        f"Unique SNVs: {unique_snvs:,}\n"
        f"In-tissue spots: {len(positions)}\n"
        f"Spots with SNVs: {len(nz)}\n"
        f"Total SNVs (sum): {int(counts_arr.sum())}\n"
        f"Max SNVs / spot: {int(counts_arr.max())}\n"
        f"Mean SNVs / spot: {counts_arr.mean():.2f}\n"
        f"Median SNVs / spot: {np.median(counts_arr):.0f}"
    )
    ax.text(
        0.02, 0.98, stats, transform=ax.transAxes, fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {out_path}")


def write_counts_csv(count_map: Dict[str, int], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["barcode", "snv_count"])
        for barcode, count in sorted(count_map.items()):
            writer.writerow([barcode, count])
    logger.info(f"Wrote counts CSV: {out_path}")


def write_summary(
    out_path: str,
    barcode_dir: str,
    race_filter: Optional[str],
    dataset: str,
    section_id: str,
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    unique_snvs: int,
    bed_used: bool,
) -> None:
    counts_arr = np.array(list(count_map.values())) if count_map else np.array([0])
    nz         = counts_arr[counts_arr > 0]
    matched    = set(positions.keys()) & set(count_map.keys())
    with open(out_path, "w") as f:
        f.write("VCF Visualizer (per-barcode) — summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Barcode dir:       {barcode_dir}\n")
        f.write(f"Race filter:       {race_filter or 'all'}\n")
        f.write(f"Exome BED filter:  {'yes' if bed_used else 'no'}\n")
        f.write(f"Dataset:           {dataset}\n")
        f.write(f"Section:           {section_id}\n\n")
        f.write(f"Unique SNVs:                  {unique_snvs:,}\n")
        f.write(f"Barcodes in count map:        {len(count_map)}\n")
        f.write(f"In-tissue spots:              {len(positions)}\n")
        f.write(f"In-tissue spots with profile: {len(matched)}\n")
        f.write(f"Spots with >=1 SNV:           {len(nz)}\n\n")
        f.write("Per-spot SNV count statistics:\n")
        f.write(f"  total (sum):  {int(counts_arr.sum())}\n")
        f.write(f"  max:          {int(counts_arr.max())}\n")
        f.write(f"  mean:         {counts_arr.mean():.2f}\n")
        f.write(f"  median:       {np.median(counts_arr):.0f}\n")
    logger.info(f"Wrote summary: {out_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Spatial SNV visualizer from per-barcode TXT files.")
    parser.add_argument("--barcode-dir", required=True,
                        help="Directory containing per-barcode *.txt files (chrom/pos/ref/alt/race columns).")
    parser.add_argument("--race",        default="denovo", choices=["denovo", "defined", "all"],
                        help="Row filter on the 'race' column (default: denovo).")
    parser.add_argument("--dataset",     required=True, choices=list(DATASET_CONFIGS.keys()),
                        help=f"Dataset: {list(DATASET_CONFIGS.keys())}")
    parser.add_argument("--section-id",  required=True)
    parser.add_argument("--output-dir",  default=None,
                        help="Output directory (default: <barcode_dir>/per_barcode_visualizer/)")
    parser.add_argument("--title",       default=None,
                        help="Plot title (auto-generated if omitted).")
    parser.add_argument("--max-workers", type=int, default=22)
    parser.add_argument("--no-bed",      action="store_true",
                        help="Disable exome BED filtering (default: on-target positions only).")
    args = parser.parse_args()

    if args.section_id not in DATASET_CONFIGS[args.dataset]["section_ids"]:
        parser.error(f"--section-id {args.section_id} not valid for {args.dataset}. "
                     f"Valid: {DATASET_CONFIGS[args.dataset]['section_ids']}")

    barcode_dir = os.path.abspath(args.barcode_dir)
    if not os.path.isdir(barcode_dir):
        raise NotADirectoryError(f"barcode-dir not found: {barcode_dir}")

    category = os.path.basename(barcode_dir)           # e.g. "germline", "somatic"
    race_filter = None if args.race == "all" else args.race
    stem        = f"{category}_{args.race}"            # e.g. "germline_denovo"

    out_dir = args.output_dir or os.path.join(barcode_dir, "per_barcode_visualizer")
    os.makedirs(out_dir, exist_ok=True)

    bed_suffix = "" if args.no_bed else " (exome)"
    title = args.title or f"{args.dataset} sec{args.section_id} — {category} {args.race}{bed_suffix}"

    spatial = resolve_spatial_paths(args.dataset, args.section_id)
    for key in ("position_file", "scale_factor_file"):
        if not os.path.exists(spatial[key]):
            raise FileNotFoundError(f"Missing spatial file ({key}): {spatial[key]}")

    logger.info("=" * 60)
    logger.info(f"Barcode dir: {barcode_dir}")
    logger.info(f"Race filter: {race_filter or 'all'}")
    logger.info(f"Dataset:     {args.dataset}  Section: {args.section_id}")
    logger.info(f"Output dir:  {out_dir}")
    logger.info("=" * 60)

    # BED index
    bed_index: Optional[BedIndex] = None
    if not args.no_bed:
        bed_path = DATASET_CONFIGS[args.dataset].get("bed_file")
        if bed_path and os.path.exists(bed_path):
            bed_index = load_bed_index(bed_path)
        elif bed_path:
            logger.warning(f"BED file not found, skipping exome filter: {bed_path}")

    # Count SNVs per barcode from pre-computed TXT files
    count_map, unique_snvs = count_snvs_per_barcode(
        barcode_dir, race_filter, bed_index, max_workers=args.max_workers
    )
    write_counts_csv(count_map, os.path.join(out_dir, f"{stem}_counts.csv"))

    # Visualization
    positions = load_spot_positions(spatial["position_file"], spatial["has_header"])
    scale     = load_scale_factor(spatial["scale_factor_file"])

    visualize_spot_counts(
        count_map   = count_map,
        positions   = positions,
        scale       = scale,
        x_flip      = spatial["x_flip"],
        title       = title,
        unique_snvs = unique_snvs,
        out_path    = os.path.join(out_dir, f"{stem}_spot_snv_counts.png"),
    )

    write_summary(
        out_path    = os.path.join(out_dir, f"{stem}_summary.txt"),
        barcode_dir = barcode_dir,
        race_filter = race_filter,
        dataset     = args.dataset,
        section_id  = args.section_id,
        count_map   = count_map,
        positions   = positions,
        unique_snvs = unique_snvs,
        bed_used    = bed_index is not None,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
