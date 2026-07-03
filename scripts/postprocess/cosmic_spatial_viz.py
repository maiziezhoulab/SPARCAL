#!/usr/bin/env python3
"""
cosmic_spatial_viz.py
Visualizes COSMIC-hit somatic/germline variants on tissue image.
Spots are colored by the number of COSMIC-hit variants they carry:
  - 0 hits  : light gray (background spots)
  - 1+ hits : blue, darker with more hits

Usage:
    python cosmic_spatial_viz.py \
        --dataset P4_tumor \
        --section 1 \
        --cosmic_hits_vcf /data/.../isec/0002.vcf \
        --variant_type somatic \
        --outdir /data/maiziezhou_lab/leiy4/COSMIC/visualization
"""

import argparse
import os
import gzip
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ── Dataset configs (mirrors enhanced spatial filter) ─────────────────────────
DATASET_CONFIGS = {
    "P4_TUMOR": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor",
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
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
    },
    "P6_TUMOR": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor",
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
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
    },
    "DCIS": {
        "output_base": "/data/maiziezhou_lab/leiy4/snv_calling/data/DCIS",
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/DCIS",
        "spaceranger_dir_template": "{section_id}",
        "spatial_subdir": "spatial",
        "position_file_patterns": {
            "dcis1": "tissue_positions_list.csv",
            "dcis2": "tissue_positions_list.csv",
        },
        "scale_file_patterns": {
            "dcis1": "scalefactors_json.json",
            "dcis2": "scalefactors_json.json",
        },
        "image_file_patterns": {
            "dcis1": "tissue_hires_image.png",
            "dcis2": "tissue_hires_image.png",
        },
    },
}

QUALITY_FILTER = "baseQ0mapQ0"


# ── Path resolution ───────────────────────────────────────────────────────────
def resolve_paths(dataset_key, section_id):
    cfg = DATASET_CONFIGS[dataset_key]
    spaceranger_dir = cfg["spaceranger_dir_template"].format(section_id=section_id)
    spatial_base    = os.path.join(cfg["base_path"], spaceranger_dir, cfg["spatial_subdir"])

    positions_file = os.path.join(spatial_base, cfg["position_file_patterns"][section_id])
    scale_file     = os.path.join(spatial_base, cfg["scale_file_patterns"][section_id])
    image_file     = os.path.join(spatial_base, cfg["image_file_patterns"][section_id])
    vcf_by_spot_dir = os.path.join(
        cfg["output_base"], section_id,
        f"output_VCFs/spotprofiles/{QUALITY_FILTER}/vcf_by_spot"
    )
    return positions_file, scale_file, image_file, vcf_by_spot_dir


# ── Load COSMIC hits VCF → set of variant keys ────────────────────────────────
def load_cosmic_hits(vcf_path):
    """
    Load CHROM/POS/REF/ALT from the isec 0002.vcf.
    Returns a set of (chrom_no_chr, pos, ref, alt) tuples.
    Also returns a dict: variant_key -> {GENE, TIER, SAMPLE_CNT}
    """
    hits     = set()
    hit_info = {}

    opener = gzip.open if vcf_path.endswith(".gz") else open
    mode   = "rt"
    with opener(vcf_path, mode) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            chrom = parts[0].replace("chr", "")
            pos, ref, alt = parts[1], parts[3], parts[4]
            key = (chrom, pos, ref, alt)
            hits.add(key)

            # Parse INFO for annotation
            info = {}
            if len(parts) > 7:
                for kv in parts[7].split(";"):
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        info[k] = v
            hit_info[key] = {
                "GENE":       info.get("GENE", "."),
                "TIER":       info.get("TIER", "."),
                "SAMPLE_CNT": info.get("GENOME_SCREEN_SAMPLE_COUNT", "."),
            }

    print(f"[INFO] Loaded {len(hits)} COSMIC-hit variants")
    return hits, hit_info


# ── Load spot positions ───────────────────────────────────────────────────────
def load_positions(positions_file):
    """
    Returns dict: barcode -> (pxl_row_fullres, pxl_col_fullres)
    Col 4 = pxl_row_in_fullres, Col 5 = pxl_col_in_fullres
    """
    df = pd.read_csv(positions_file, header=None)
    positions = {}
    for _, row in df.iterrows():
        bc = str(row[0])
        # For datasets with in_tissue column (col 1), skip out-of-tissue
        # P4/P6 have no in_tissue filter column so include all
        pxl_row = float(row[4])
        pxl_col = float(row[5])
        positions[bc] = (pxl_row, pxl_col)
    print(f"[INFO] Loaded {len(positions)} spot positions")
    return positions


# ── Load scale factors ────────────────────────────────────────────────────────
def load_scale_factors(scale_file):
    with open(scale_file) as f:
        sf = json.load(f)
    return sf.get("tissue_hires_scalef", 1.0)


# ── Count COSMIC hits per spot from per-barcode VCFs ─────────────────────────
def count_cosmic_hits_per_spot(vcf_by_spot_dir, cosmic_hits):
    """
    Iterates through per-barcode VCFs. For each barcode, counts how many
    variants in that VCF are in cosmic_hits.
    Returns dict: barcode -> int (hit count)
    """
    hit_counts = defaultdict(int)

    if not os.path.isdir(vcf_by_spot_dir):
        print(f"[WARN] vcf_by_spot_dir not found: {vcf_by_spot_dir}")
        return hit_counts

    vcf_files = [f for f in os.listdir(vcf_by_spot_dir)
                 if f.endswith(".vcf.gz") or f.endswith(".vcf")]
    print(f"[INFO] Scanning {len(vcf_files)} per-barcode VCFs...")

    for fname in vcf_files:
        barcode = fname.replace(".vcf.gz", "").replace(".vcf", "")
        fpath   = os.path.join(vcf_by_spot_dir, fname)

        try:
            opener = gzip.open if fpath.endswith(".gz") else open
            with opener(fpath, "rt") as f:
                for line in f:
                    if line.startswith("#"):
                        continue
                    parts = line.strip().split("\t")
                    if len(parts) < 5:
                        continue
                    chrom = parts[0].replace("chr", "")
                    pos, ref, alt = parts[1], parts[3], parts[4]
                    if (chrom, pos, ref, alt) in cosmic_hits:
                        hit_counts[barcode] += 1
        except Exception as e:
            print(f"[WARN] Could not read {fpath}: {e}")
            continue

    n_spots_with_hits = sum(1 for v in hit_counts.values() if v > 0)
    print(f"[INFO] {n_spots_with_hits} spots carry at least one COSMIC-hit variant")
    return hit_counts


# ── Visualization ─────────────────────────────────────────────────────────────
def plot_cosmic_hits(positions, hit_counts, image_file, scale_factor,
                     dataset, section_id, variant_type, outdir):
    """
    Plot spots on tissue image colored by COSMIC hit count.
    0 hits = light gray; 1+ hits = blue (darker = more hits).
    """
    img = plt.imread(image_file)

    # Convert full-res pixel coords to hires image coords
    # pxl_row = y on image, pxl_col = x on image
    barcodes = list(positions.keys())
    xs = np.array([positions[bc][1] * scale_factor for bc in barcodes])  # pxl_col -> img x
    ys = np.array([positions[bc][0] * scale_factor for bc in barcodes])  # pxl_row -> img y
    counts = np.array([hit_counts.get(bc, 0) for bc in barcodes])

    max_count = max(counts.max(), 1)

    # Custom colormap: white/lightgray for 0, blue gradient for 1+
    blue_cmap = LinearSegmentedColormap.from_list(
        "cosmic_blue", ["#d0d0d0", "#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    )

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(img, aspect="equal")

    # Plot non-hit spots first (background)
    mask_zero = counts == 0
    ax.scatter(xs[mask_zero], ys[mask_zero],
               c="#d0d0d0", s=8, alpha=0.3, linewidths=0, zorder=2,
               label="No COSMIC hit")

    # Plot hit spots on top, colored by count
    mask_hit = counts > 0
    if mask_hit.sum() > 0:
        norm   = mcolors.Normalize(vmin=1, vmax=max_count)
        colors = blue_cmap(norm(counts[mask_hit]))
        sc = ax.scatter(xs[mask_hit], ys[mask_hit],
                        c=counts[mask_hit], cmap=blue_cmap,
                        norm=norm, s=20, alpha=0.9, linewidths=0.3,
                        edgecolors="white", zorder=3,
                        label="COSMIC hit")
        cb = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("# COSMIC-hit variants per spot", fontsize=11)
        if max_count <= 5:
            cb.set_ticks(range(1, max_count + 1))

    n_hits     = mask_hit.sum()
    n_total    = len(barcodes)
    title = (f"COSMIC Variant Hits — {dataset} Section {section_id} "
             f"({variant_type})\n"
             f"{n_hits} / {n_total} spots carry ≥1 COSMIC-recorded variant")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.8)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(
        outdir,
        f"cosmic_hits_{dataset}_sec{section_id}_{variant_type}.png"
    )
    plt.tight_layout()
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved: {outpath}")
    return outpath


# ── Summary TSV of hit spots ──────────────────────────────────────────────────
def write_spot_summary(positions, hit_counts, outdir, dataset, section_id, variant_type):
    rows = []
    for bc, (pxl_row, pxl_col) in positions.items():
        cnt = hit_counts.get(bc, 0)
        if cnt > 0:
            rows.append({"barcode": bc, "pxl_row": pxl_row, "pxl_col": pxl_col,
                         "cosmic_hit_count": cnt})
    df = pd.DataFrame(rows).sort_values("cosmic_hit_count", ascending=False)
    outpath = os.path.join(
        outdir,
        f"cosmic_hit_spots_{dataset}_sec{section_id}_{variant_type}.tsv"
    )
    df.to_csv(outpath, sep="\t", index=False)
    print(f"[INFO] Spot summary saved: {outpath}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Visualize COSMIC hits on spatial tissue")
    parser.add_argument("--dataset",         required=True,
                        help="Dataset key: P4_tumor, P6_tumor, DCIS")
    parser.add_argument("--section",         required=True,
                        help="Section ID: 1, 2, dcis1, dcis2")
    parser.add_argument("--cosmic_hits_vcf", required=True,
                        help="Path to bcftools isec 0002.vcf (COSMIC hits)")
    parser.add_argument("--variant_type",    required=True,
                        choices=["somatic", "germline"],
                        help="somatic or germline (for labeling only)")
    parser.add_argument("--outdir",          required=True,
                        help="Output directory for plots and TSV")
    args = parser.parse_args()

    dataset_key = args.dataset.upper().replace("-", "_")
    if dataset_key not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}. "
                         f"Choices: {list(DATASET_CONFIGS.keys())}")

    positions_file, scale_file, image_file, vcf_by_spot_dir = \
        resolve_paths(dataset_key, args.section)

    # 1. Load COSMIC hits
    cosmic_hits, hit_info = load_cosmic_hits(args.cosmic_hits_vcf)

    # 2. Load spatial data
    positions    = load_positions(positions_file)
    scale_factor = load_scale_factors(scale_file)

    # 3. Count COSMIC hits per spot
    hit_counts = count_cosmic_hits_per_spot(vcf_by_spot_dir, cosmic_hits)

    # 4. Visualize
    plot_cosmic_hits(
        positions, hit_counts, image_file, scale_factor,
        args.dataset, args.section, args.variant_type, args.outdir
    )

    # 5. Write spot-level summary TSV
    df = write_spot_summary(
        positions, hit_counts, args.outdir,
        args.dataset, args.section, args.variant_type
    )

    # 6. Print top hits
    print("\n[INFO] Top 10 spots by COSMIC hit count:")
    print(df.head(10).to_string(index=False))

    total_hits_variants = len(cosmic_hits)
    total_spots_hit = sum(1 for v in hit_counts.values() if v > 0)
    print(f"\n[SUMMARY] {total_hits_variants} unique COSMIC-matched variants")
    print(f"[SUMMARY] {total_spots_hit} spots carry ≥1 COSMIC-hit variant")


if __name__ == "__main__":
    main()