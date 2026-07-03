#!/usr/bin/env python3
"""
VCF Visualizer: spatial distribution of SNVs from any VCF onto Visium tissue coordinates.

Routing:
  - Sites-only or single-sample VCF (0-1 sample columns): BAM scan via pysam CB tags.
  - Multi-sample VCF (>=2 sample columns, one column per barcode): streaming VCF split
    + bcftools filter of ref/missing genotypes, then count remaining records per barcode.

"Unique SNVs" = distinct (CHROM, POS) positions in the input VCF.

Usage:
    python vcf_visualizer.py --vcf path/to/file.vcf.gz --dataset DCIS --section-id 1
    python vcf_visualizer.py --vcf path/to/file.vcf.gz --dataset DCIS --section-id 1 \
        --output-dir path/to/output --title "My Plot" --max-workers 22
"""

import os
import csv
import gzip
import glob
import json
import logging
import argparse
import resource
import shutil
import bisect
import subprocess
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from tqdm import tqdm

try:
    import pysam
    HAS_PYSAM = True
except ImportError:
    HAS_PYSAM = False


# ============================================================================
# Configuration
# ============================================================================

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR":    "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "BCFTOOLS":    "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP":       "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX":       "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix",
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
        "bam_file_pattern": "{base_path}/spaceranger_align_rep{section_id}_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam",
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
        "bam_file_pattern": "{base_path}/spaceranger_align_rep{section_id}_hg19/P6_Tumor_output/outs/possorted_genome_bam.bam",
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
        "bam_file_pattern": "{base_path}/DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/possorted_genome_bam.bam",
        "bed_file": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/regions/Twist_Exome_Core_Covered_Targets_hg38.bed",
    },
}

COLORMAP    = ["#e6e6e6", "#4578b4"]
FIGURE_SIZE = (16, 16)
DPI         = 200

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("vcf_visualizer")


# ============================================================================
# Utilities
# ============================================================================

def setup_environment() -> None:
    apps = PATH_CONFIG["APPS_DIR"]
    os.environ["PATH"] = f"{apps}:{os.environ.get('PATH', '')}"
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{apps}:{cur}" if cur else apps


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
    bam_file = cfg["bam_file_pattern"].format(
        base_path=cfg["base_path"], section_id=section_id
    )
    return {
        "position_file":     position_file,
        "scale_factor_file": scale_factor_file,
        "image_file":        image_file,
        "has_header":        cfg["position_file_has_header"],
        "x_flip":            cfg["x_flip"],
        "bam_file":          bam_file,
    }


def count_vcf_samples(vcf_path: str) -> int:
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                return max(0, len(line.rstrip("\n").split("\t")) - 9)
    return 0


def count_unique_snvs(vcf_path: str) -> int:
    """Count distinct (CHROM, POS) positions in the VCF (SNVs only, single-base REF and ALT)."""
    seen = set()
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 5)
            if len(parts) < 5:
                continue
            chrom, pos, _, ref, alt_field = parts[0], parts[1], parts[2], parts[3], parts[4]
            if len(ref) != 1:
                continue
            alts = [a for a in alt_field.split(",") if len(a) == 1 and a not in (".", "*", "N")]
            if alts:
                seen.add((chrom, pos))
    return len(seen)


# ============================================================================
# Exome BED filtering
# ============================================================================

def load_bed_intervals(bed_path: str) -> Dict[str, List[Tuple[int, int]]]:
    """Load BED file into dict of chrom -> sorted list of (start, end) 0-based half-open intervals."""
    intervals: Dict[str, List[Tuple[int, int]]] = {}
    with open(bed_path) as fh:
        for line in fh:
            if line.startswith(("#", "track", "browser")):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                intervals.setdefault(parts[0], []).append((int(parts[1]), int(parts[2])))
            except ValueError:
                continue
    for chrom in intervals:
        intervals[chrom].sort()
    total = sum(len(v) for v in intervals.values())
    logger.info(f"Loaded BED: {total:,} intervals across {len(intervals)} chroms from {bed_path}")
    return intervals


def filter_positions_by_bed(
    positions: Dict[str, Dict[int, FrozenSet[str]]],
    bed_intervals: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, Dict[int, FrozenSet[str]]]:
    """Keep only SNV positions overlapping a BED interval; handles chr-prefix mismatch."""
    filtered: Dict[str, Dict[int, FrozenSet[str]]] = {}
    n_kept = n_drop = 0
    for chrom, pos_map in positions.items():
        ivs = bed_intervals.get(chrom) or bed_intervals.get(
            chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        )
        if ivs is None:
            n_drop += len(pos_map)
            continue
        starts = [s for s, _ in ivs]
        kept: Dict[int, FrozenSet[str]] = {}
        for pos, alts in pos_map.items():
            pos0 = pos - 1  # VCF is 1-based; BED is 0-based half-open
            idx = bisect.bisect_right(starts, pos0) - 1
            if idx >= 0 and ivs[idx][0] <= pos0 < ivs[idx][1]:
                kept[pos] = alts
                n_kept += 1
            else:
                n_drop += 1
        if kept:
            filtered[chrom] = kept
    logger.info(f"BED filter: {n_kept:,} positions kept, {n_drop:,} dropped")
    return filtered


# ============================================================================
# Path A — BAM scan (sites-only or single-sample VCF)
# ============================================================================

def load_variant_positions(vcf_path: str) -> Dict[str, Dict[int, FrozenSet[str]]]:
    positions: Dict[str, Dict[int, FrozenSet[str]]] = {}
    n_snvs = n_skip = 0
    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 5)
            if len(parts) < 5:
                continue
            chrom, pos_str, _, ref, alt_field = parts[0], parts[1], parts[2], parts[3], parts[4]
            if len(ref) != 1:
                n_skip += 1
                continue
            alts = [a for a in alt_field.split(",") if len(a) == 1 and a not in (".", "*", "N")]
            if not alts:
                n_skip += 1
                continue
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            positions.setdefault(chrom, {})[pos] = frozenset(alts)
            n_snvs += 1
    logger.info(f"Loaded {n_snvs:,} SNV positions across {len(positions)} chromosomes ({n_skip:,} skipped)")
    return positions


def _build_chrom_map(bam_path: str, vcf_chroms: List[str]) -> Dict[str, str]:
    bam = pysam.AlignmentFile(bam_path, "rb")
    bam_refs = set(bam.references)
    bam.close()
    mapping: Dict[str, str] = {}
    for c in vcf_chroms:
        if c in bam_refs:
            mapping[c] = c
        elif c.startswith("chr") and c[3:] in bam_refs:
            mapping[c] = c[3:]
        elif not c.startswith("chr") and ("chr" + c) in bam_refs:
            mapping[c] = "chr" + c
    n_skip = len(vcf_chroms) - len(mapping)
    logger.info(f"Chrom map: {len(mapping)} matched, {n_skip} unmatched (BAM has {len(bam_refs)} contigs)")
    if n_skip:
        missing = [c for c in vcf_chroms if c not in mapping]
        logger.warning(f"VCF chroms with no BAM contig: {missing}")
    return mapping


def _scan_one_chrom(bam_path: str, chrom: str, variant_pos: Dict[int, FrozenSet[str]]) -> Dict[str, int]:
    if not variant_pos:
        return {}
    pos_sorted  = sorted(variant_pos.keys())
    start_0     = pos_sorted[0] - 1
    stop_0      = pos_sorted[-1]
    count_map: Dict[str, int] = {}
    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
    except Exception as exc:
        logger.warning(f"Could not open BAM for {chrom}: {exc}")
        return {}
    try:
        for pcol in bam.pileup(
            contig=chrom, start=start_0, stop=stop_0,
            max_depth=2_000_000, min_base_quality=0,
            stepper="all", truncate=True, ignore_overlaps=False,
        ):
            pos1 = pcol.reference_pos + 1
            if pos1 not in variant_pos:
                continue
            alts = variant_pos[pos1]
            credited_at_pos: set = set()
            for pread in pcol.pileups:
                if pread.is_del or pread.is_refskip or pread.query_position is None:
                    continue
                base = pread.alignment.query_sequence[pread.query_position]
                if base not in alts:
                    continue
                try:
                    cb = pread.alignment.get_tag("CB")
                except KeyError:
                    continue
                if cb not in credited_at_pos:
                    credited_at_pos.add(cb)
                    count_map[cb] = count_map.get(cb, 0) + 1
    except Exception as exc:
        logger.warning(f"Pileup error on {chrom}: {exc}")
    finally:
        bam.close()
    return count_map


def scan_bam_for_spot_counts(
    bam_path: str,
    variant_positions: Dict[str, Dict[int, FrozenSet[str]]],
    max_workers: int = 22,
) -> Dict[str, int]:
    if not HAS_PYSAM:
        raise RuntimeError("pysam not installed: conda install -c bioconda pysam")
    vcf_chroms  = sorted(variant_positions.keys())
    chrom_map   = _build_chrom_map(bam_path, vcf_chroms)
    to_scan     = [c for c in vcf_chroms if c in chrom_map]
    n_variants  = sum(len(variant_positions[c]) for c in to_scan)
    logger.info(f"BAM scan: {len(to_scan)} chroms, {n_variants:,} positions, {max_workers} threads")
    total: Dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_scan_one_chrom, bam_path, chrom_map[c], variant_positions[c]): c
            for c in to_scan
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scanning chromosomes"):
            chrom = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                logger.warning(f"scan_chrom({chrom}) raised: {exc}")
                continue
            for bc, cnt in result.items():
                total[bc] = total.get(bc, 0) + cnt
    logger.info(f"BAM scan complete: {len(total)} barcodes, {sum(total.values()):,} total hits")
    return total


# ============================================================================
# Path B — VCF split (multi-sample, one column per barcode)
# ============================================================================

def split_merged_vcf(merged_vcf: str, raw_split_dir: str) -> List[str]:
    os.makedirs(raw_split_dir, exist_ok=True)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 65536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except Exception as exc:
        logger.warning(f"Could not raise RLIMIT_NOFILE: {exc}")

    header_lines: List[str] = []
    with gzip.open(merged_vcf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                header_lines.append(line)
            else:
                break
    if not header_lines:
        raise RuntimeError(f"No header in {merged_vcf}")
    chrom_fields = header_lines[-1].rstrip("\n").split("\t")
    if len(chrom_fields) < 10:
        raise RuntimeError(f"#CHROM line has only {len(chrom_fields)} columns; expected >=10")
    sample_names = chrom_fields[9:]
    header_prefix = "".join(header_lines[:-1])
    fixed_cols    = chrom_fields[:9]
    logger.info(f"Streaming split: {len(sample_names)} samples -> {raw_split_dir}")

    n_variants = 0
    handles: Dict[str, gzip.GzipFile] = {}
    try:
        for s in sample_names:
            h = gzip.open(os.path.join(raw_split_dir, f"{s}.vcf.gz"), "wt")
            h.write(header_prefix)
            h.write("\t".join(fixed_cols + [s]) + "\n")
            handles[s] = h
        with gzip.open(merged_vcf, "rt") as fh:
            for line in fh:
                if line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                fixed = parts[:9]
                for i, s in enumerate(sample_names):
                    gt = parts[9 + i] if (9 + i) < len(parts) else "."
                    handles[s].write("\t".join(fixed + [gt]) + "\n")
                n_variants += 1
                if n_variants % 100_000 == 0:
                    logger.info(f"  …{n_variants:,} variant lines processed")
    finally:
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    files = sorted(glob.glob(os.path.join(raw_split_dir, "*.vcf.gz")))
    logger.info(f"Split produced {len(files)} per-sample VCFs ({n_variants:,} lines)")
    if not files:
        raise RuntimeError(f"No per-sample VCFs in {raw_split_dir}")
    return files


def _filter_one_vcf(input_vcf: str, output_vcf: str) -> Tuple[bool, int, str]:
    r1 = subprocess.run(
        f"{PATH_CONFIG['BCFTOOLS']} view -e 'GT=\"ref\" || GT=\"mis\"' -Oz -o {output_vcf} {input_vcf}",
        shell=True, capture_output=True, text=True,
    )
    if r1.returncode != 0:
        return False, 0, (r1.stderr or "view failed").strip()
    r2 = subprocess.run(
        f"{PATH_CONFIG['BCFTOOLS']} view -H {output_vcf} | wc -l",
        shell=True, capture_output=True, text=True,
    )
    count = int(r2.stdout.strip()) if r2.stdout.strip().isdigit() else 0
    r3 = subprocess.run(
        f"{PATH_CONFIG['BCFTOOLS']} index -t -f {output_vcf}",
        shell=True, capture_output=True, text=True,
    )
    if r3.returncode != 0:
        return True, count, f"index_warning: {r3.stderr.strip()}"
    return True, count, ""


def filter_all_parallel(raw_files: List[str], output_dir: str, max_workers: int = 30) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    count_map: Dict[str, int] = {}
    failures: List[Tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for raw in raw_files:
            barcode = os.path.basename(raw).replace(".vcf.gz", "")
            futures[ex.submit(_filter_one_vcf, raw, os.path.join(output_dir, f"{barcode}.vcf.gz"))] = barcode
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Filter 0/0 + index"):
            barcode = futures[fut]
            try:
                ok, count, err = fut.result()
            except Exception as exc:
                failures.append((barcode, str(exc)))
                continue
            if ok:
                count_map[barcode] = count
                if err.startswith("index_warning"):
                    logger.warning(f"[{barcode}] {err}")
            else:
                failures.append((barcode, err))
    logger.info(f"Filter: {len(count_map)} OK, {len(failures)} failed")
    if failures:
        for b, e in failures[:5]:
            logger.warning(f"  {b}: {e}")
    return count_map


# ============================================================================
# Spatial / visualization
# ============================================================================

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


def visualize_spot_counts(
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    scale: float,
    image_path: str,
    x_flip: bool,
    title: str,
    unique_snvs: int,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.imshow(img, origin="upper")

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

    # Lock axes to image dimensions so scatter auto-scaling can't push the image out of view
    img_h, img_w = img.shape[:2]
    ax.set_xlim(0, img_w)
    ax.set_ylim(img_h, 0)
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
    out_path: str, vcf_path: str, dataset: str, section_id: str,
    count_map: Dict[str, int], positions: Dict[str, Tuple[float, float]],
    unique_snvs: int, source_desc: str,
) -> None:
    counts_arr = np.array(list(count_map.values())) if count_map else np.array([0])
    nz         = counts_arr[counts_arr > 0]
    matched    = set(positions.keys()) & set(count_map.keys())
    with open(out_path, "w") as f:
        f.write("VCF Visualizer — summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"VCF:               {vcf_path}\n")
        f.write(f"Dataset:           {dataset}\n")
        f.write(f"Section:           {section_id}\n")
        f.write(f"Source:            {source_desc}\n\n")
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
    parser = argparse.ArgumentParser(description="Spatial VCF visualizer for Visium data.")
    parser.add_argument("--vcf",        required=True,  help="Input VCF (sites-only, single-sample, or multi-sample .vcf.gz)")
    parser.add_argument("--dataset",    required=True,  choices=list(DATASET_CONFIGS.keys()),
                        help=f"Dataset: {list(DATASET_CONFIGS.keys())}")
    parser.add_argument("--section-id", required=True)
    parser.add_argument("--output-dir", default=None,   help="Output directory (default: <vcf_dir>/vcf_visualizer/)")
    parser.add_argument("--title",      default=None,   help="Plot title (auto-generated if omitted)")
    parser.add_argument("--max-workers",type=int, default=22)
    parser.add_argument("--keep-tmp",   action="store_true", help="Keep intermediate split VCF directory.")
    parser.add_argument("--no-bed",     action="store_true", help="Disable exome BED filtering (default: on-target positions only).")
    args = parser.parse_args()

    if args.section_id not in DATASET_CONFIGS[args.dataset]["section_ids"]:
        parser.error(f"--section-id {args.section_id} not valid for {args.dataset}. "
                     f"Valid: {DATASET_CONFIGS[args.dataset]['section_ids']}")

    setup_environment()

    vcf_path = os.path.abspath(args.vcf)
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"VCF not found: {vcf_path}")

    # Derive a clean stem from the VCF filename for use in output file names
    vcf_basename = os.path.basename(vcf_path)
    if vcf_basename.endswith(".vcf.gz"):
        vcf_stem = vcf_basename[:-7]
    elif vcf_basename.endswith(".vcf"):
        vcf_stem = vcf_basename[:-4]
    else:
        vcf_stem = vcf_basename

    out_dir = args.output_dir or os.path.join(os.path.dirname(vcf_path), "vcf_visualizer")
    os.makedirs(out_dir, exist_ok=True)

    title = args.title or (
        f"{args.dataset} sec{args.section_id} — {vcf_basename}"
        + ("" if args.no_bed else " (exome)")
    )

    spatial = resolve_spatial_paths(args.dataset, args.section_id)

    for key in ("position_file", "scale_factor_file", "image_file"):
        if not os.path.exists(spatial[key]):
            raise FileNotFoundError(f"Missing spatial file ({key}): {spatial[key]}")

    logger.info("=" * 60)
    logger.info(f"VCF:         {vcf_path}")
    logger.info(f"Dataset:     {args.dataset}  Section: {args.section_id}")
    logger.info(f"Output dir:  {out_dir}")
    logger.info(f"BAM:         {spatial['bam_file']}")
    logger.info("=" * 60)

    # Count unique SNVs (positions) in the VCF
    logger.info("Counting unique SNV positions...")
    unique_snvs = count_unique_snvs(vcf_path)
    logger.info(f"Unique SNVs: {unique_snvs:,}")

    # Route: BAM scan for sites-only / single-sample; VCF split for multi-sample
    n_samples = count_vcf_samples(vcf_path)
    logger.info(f"VCF sample columns: {n_samples}")

    if n_samples <= 1:
        # BAM scan path
        bam = spatial["bam_file"]
        if not os.path.exists(bam):
            raise FileNotFoundError(f"BAM not found: {bam}")
        bai = bam + ".bai"
        if not os.path.exists(bai):
            raise FileNotFoundError(f"BAM index not found: {bai}")
        logger.info(f"Routing: BAM scan (pysam CB tags) from {bam}")
        variant_positions = load_variant_positions(vcf_path)
        if not args.no_bed:
            bed_path = DATASET_CONFIGS[args.dataset].get("bed_file")
            if bed_path and os.path.exists(bed_path):
                bed_ivs = load_bed_intervals(bed_path)
                variant_positions = filter_positions_by_bed(variant_positions, bed_ivs)
                unique_snvs = sum(len(v) for v in variant_positions.values())
                logger.info(f"Unique SNVs after exome filter: {unique_snvs:,}")
            elif bed_path:
                logger.warning(f"BED file not found, skipping exome filter: {bed_path}")
        count_map = scan_bam_for_spot_counts(bam, variant_positions, max_workers=min(args.max_workers, 22))
        write_counts_csv(count_map, os.path.join(out_dir, f"{vcf_stem}_counts.csv"))
        source_desc = f"BAM pysam scan of {bam}"
    else:
        # VCF split path
        if not args.no_bed:
            logger.warning("BED exome filter is not applied in multi-sample VCF split mode.")
        logger.info(f"Routing: streaming VCF split ({n_samples} samples)")
        raw_split_dir = os.path.join(out_dir, "_raw_split_tmp")
        if os.path.exists(raw_split_dir):
            shutil.rmtree(raw_split_dir)
        try:
            raw_files = split_merged_vcf(vcf_path, raw_split_dir)
            split_out = os.path.join(out_dir, "_filtered_split")
            count_map = filter_all_parallel(raw_files, split_out, max_workers=args.max_workers)
        finally:
            if not args.keep_tmp and os.path.exists(raw_split_dir):
                shutil.rmtree(raw_split_dir, ignore_errors=True)
        write_counts_csv(count_map, os.path.join(out_dir, f"{vcf_stem}_counts.csv"))
        source_desc = f"VCF split of {vcf_path}"

    # Visualization
    positions = load_spot_positions(spatial["position_file"], spatial["has_header"])
    scale     = load_scale_factor(spatial["scale_factor_file"])

    visualize_spot_counts(
        count_map    = count_map,
        positions    = positions,
        scale        = scale,
        image_path   = spatial["image_file"],
        x_flip       = spatial["x_flip"],
        title        = title,
        unique_snvs  = unique_snvs,
        out_path     = os.path.join(out_dir, f"{vcf_stem}_spot_snv_counts.png"),
    )

    write_summary(
        out_path    = os.path.join(out_dir, f"{vcf_stem}_summary.txt"),
        vcf_path    = vcf_path,
        dataset     = args.dataset,
        section_id  = args.section_id,
        count_map   = count_map,
        positions   = positions,
        unique_snvs = unique_snvs,
        source_desc = source_desc,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
