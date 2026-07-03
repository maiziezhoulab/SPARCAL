#!/usr/bin/env python3
"""
Generate per-spot SNP profile from raw mpileup output.

For multi-sample merged VCFs (DCIS / DLPFC): splits the VCF by sample column,
filters GT=0/0 / missing, and counts per-barcode records.

For single-sample merged VCFs (P4_TUMOR / P6_TUMOR, where mpileup ran on the
merged SpaceRanger BAM): scans the BAM directly with pysam, tracks the CB
(cell barcode) tag at every variant position, and counts how many variant
positions each barcode has an ALT-supporting read for.

Usage:
    python generate_original_snp_profile.py --dataset P4_TUMOR --section-id 1
    python generate_original_snp_profile.py --dataset DCIS --section-id 1 --max-workers 30
"""

import os
import sys
import csv
import json
import glob
import gzip
import shutil
import logging
import argparse
import resource
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
    "SAMTOOLS":    "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools",
    "BCFTOOLS":    "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP":       "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX":       "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix",
}

QUALITY_FILTER = "baseQ0mapQ0"

REFERENCE_CONFIGS = {
    "DLPFC":       {"chr_prefix": ""},
    "FFPE_VISIUM": {"chr_prefix": "chr"},
    "TUMOR":       {"chr_prefix": "chr"},
}

DATASET_CONFIGS = {
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
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
        # Mpileup ran on the merged SpaceRanger BAM (single sample per VCF).
        # Per-spot counts are obtained by scanning the BAM with pysam CB tags.
        "bam_file_pattern": "{base_path}/spaceranger_align_rep{section_id}_hg19/P4_Tumor_output/outs/possorted_genome_bam.bam",
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
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
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "section_ids": ["1", "2"],
        "reference": "FFPE_VISIUM",
        "spatial_dir": "DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/spatial",
        "position_file": "tissue_positions.csv",
        "scale_factor_file": "scalefactors_json.json",
        "image_file": "tissue_hires_image.png",
        "position_file_has_header": True,
        "x_flip": True,
        # Mpileup also ran on the merged SpaceRanger BAM for DCIS.
        "bam_file_pattern": "{base_path}/DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/possorted_genome_bam.bam",
    },
}

COLORMAP = ["#4578b4", "#e6e6e6", "red"]
FIGURE_SIZE = (16, 16)
DPI = 200


# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("original_snp_profile")


# ============================================================================
# Utilities
# ============================================================================

def setup_environment() -> None:
    apps = PATH_CONFIG["APPS_DIR"]
    os.environ["PATH"] = f"{apps}:{os.environ.get('PATH', '')}"
    cur_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = f"{apps}:{cur_ld}" if cur_ld else apps


def resolve_paths(dataset: str, section_id: str) -> Dict[str, str]:
    cfg = DATASET_CONFIGS[dataset]
    out_dir_rel = cfg["output_dir"].format(section_id=section_id)
    section_root = os.path.join(PATH_CONFIG["PROJECT_DIR"], out_dir_rel)

    mpileup_dir = os.path.join(
        section_root, "output_VCFs", "mpileup_multi_bam", QUALITY_FILTER
    )
    merged_main    = os.path.join(mpileup_dir, "merged_sorted_gt.vcf.gz")
    merged_exclude = os.path.join(mpileup_dir, "merged_sorted_gt_exclude00.vcf.gz")
    merged_vcf = merged_exclude if os.path.exists(merged_exclude) else merged_main

    profile_dir = os.path.join(section_root, "original_snp_profile")

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

    bam_file: Optional[str] = None
    if "bam_file_pattern" in cfg:
        bam_file = cfg["bam_file_pattern"].format(
            base_path=cfg["base_path"],
            section_id=section_id,
        )

    return {
        "section_root":      section_root,
        "merged_vcf":        merged_vcf,
        "profile_dir":       profile_dir,
        "position_file":     position_file,
        "scale_factor_file": scale_factor_file,
        "image_file":        image_file,
        "has_header":        cfg["position_file_has_header"],
        "x_flip":            cfg["x_flip"],
        "bam_file":          bam_file,
    }


def check_inputs(paths: Dict, require_bam: bool = False) -> None:
    required = ["merged_vcf", "position_file", "scale_factor_file", "image_file"]
    for k in required:
        p = paths[k]
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input ({k}): {p}")
    if require_bam:
        bam = paths.get("bam_file")
        if not bam:
            raise RuntimeError("BAM scan requested but no bam_file configured for this dataset.")
        if not os.path.exists(bam):
            raise FileNotFoundError(f"Missing BAM file: {bam}")
        bai = bam + ".bai"
        if not os.path.exists(bai):
            raise FileNotFoundError(f"Missing BAM index (.bai): {bai}")
    logger.info("All required inputs present.")


def count_vcf_samples(vcf_path: str) -> int:
    """Return the number of sample columns in a VCF (reads only the #CHROM line)."""
    with gzip.open(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                fields = line.rstrip("\n").split("\t")
                return max(0, len(fields) - 9)
    return 0


# ============================================================================
# Path A — BAM scan with pysam (single-sample VCF datasets: P4/P6 TUMOR)
# ============================================================================

def load_variant_positions(
    vcf_path: str,
) -> Dict[str, Dict[int, FrozenSet[str]]]:
    """
    Read all SNV positions from a VCF.
    Returns {chrom: {pos_1based: frozenset(alts)}}.

    Only single-base ALTs are included; indels and symbolic alleles are skipped.
    Uses split("\t", 5) to avoid parsing sample columns in large multi-sample VCFs.
    """
    positions: Dict[str, Dict[int, FrozenSet[str]]] = {}
    n_snvs = 0
    n_skipped = 0

    opener = gzip.open if vcf_path.endswith(".gz") else open
    with opener(vcf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 5)
            if len(parts) < 5:
                continue
            chrom, pos_str, _, ref, alt_field = (
                parts[0], parts[1], parts[2], parts[3], parts[4]
            )
            # SNVs only: single-base REF
            if len(ref) != 1:
                n_skipped += 1
                continue
            alts = [
                a for a in alt_field.split(",")
                if len(a) == 1 and a not in (".", "*", "N")
            ]
            if not alts:
                n_skipped += 1
                continue
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            if chrom not in positions:
                positions[chrom] = {}
            positions[chrom][pos] = frozenset(alts)
            n_snvs += 1

    logger.info(
        f"Loaded {n_snvs:,} SNV positions across {len(positions)} chromosomes "
        f"({n_skipped:,} indels/multi-base skipped)"
    )
    return positions


def _build_chrom_map(bam_path: str, vcf_chroms: List[str]) -> Dict[str, str]:
    """
    Map VCF chromosome names to BAM contig names, handling chr-prefix mismatches.
    Returns {vcf_chrom: bam_chrom} for every VCF chrom that has a matching BAM contig.
    Unmatched chroms are silently omitted (caller logs a warning).
    """
    bam = pysam.AlignmentFile(bam_path, "rb")
    bam_refs = set(bam.references)
    bam.close()

    mapping: Dict[str, str] = {}
    for vcf_chrom in vcf_chroms:
        if vcf_chrom in bam_refs:
            mapping[vcf_chrom] = vcf_chrom
        elif vcf_chrom.startswith("chr") and vcf_chrom[3:] in bam_refs:
            # VCF has chr prefix, BAM does not (e.g. chr1 → 1)
            mapping[vcf_chrom] = vcf_chrom[3:]
        elif not vcf_chrom.startswith("chr") and ("chr" + vcf_chrom) in bam_refs:
            # BAM has chr prefix, VCF does not (e.g. 1 → chr1)
            mapping[vcf_chrom] = "chr" + vcf_chrom
        # else: no match; will be logged as skipped

    n_mapped  = len(mapping)
    n_skipped = len(vcf_chroms) - n_mapped
    logger.info(
        f"Chromosome name mapping: {n_mapped} matched, {n_skipped} unmatched "
        f"(BAM has {len(bam_refs)} contigs)"
    )
    if n_skipped:
        missing = [c for c in vcf_chroms if c not in mapping]
        logger.warning(f"VCF chromosomes with no BAM contig: {missing}")
    return mapping


def _scan_one_chrom(
    bam_path: str,
    chrom: str,
    variant_pos: Dict[int, FrozenSet[str]],
) -> Dict[str, int]:
    """
    Scan one chromosome of the BAM via pysam pileup.
    At each variant position, reads that carry the ALT base and have a CB tag
    are counted.  Returns {barcode: count_of_positions_with_ALT_read}.

    Each thread opens its own AlignmentFile handle (pysam handles are not
    thread-safe).
    """
    if not variant_pos:
        return {}

    pos_sorted = sorted(variant_pos.keys())
    start_0based = pos_sorted[0] - 1   # pysam uses 0-based half-open
    stop_0based  = pos_sorted[-1]       # exclusive → covers the last 1-based pos

    count_map: Dict[str, int] = {}

    try:
        bam = pysam.AlignmentFile(bam_path, "rb")
    except Exception as exc:
        logger.warning(f"Could not open BAM for chromosome {chrom}: {exc}")
        return {}

    try:
        for pcol in bam.pileup(
            contig=chrom,
            start=start_0based,
            stop=stop_0based,
            max_depth=2_000_000,
            min_base_quality=0,
            stepper="all",
            truncate=True,
            ignore_overlaps=False,
        ):
            pos1 = pcol.reference_pos + 1
            if pos1 not in variant_pos:
                continue
            alts = variant_pos[pos1]

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
    """
    Scan the BAM in parallel across chromosomes using ThreadPoolExecutor.
    pysam releases the GIL during C-level I/O, so threads run concurrently.
    Returns merged {barcode: total_snv_count}.
    """
    if not HAS_PYSAM:
        raise RuntimeError(
            "pysam is not installed. Install it with: conda install -c bioconda pysam"
        )

    vcf_chroms = sorted(variant_positions.keys())
    chrom_map  = _build_chrom_map(bam_path, vcf_chroms)

    # Only scan chroms that have a matching BAM contig
    chroms_to_scan = [c for c in vcf_chroms if c in chrom_map]
    n_variants_total = sum(len(variant_positions[c]) for c in chroms_to_scan)
    logger.info(
        f"BAM scan: {len(chroms_to_scan)} chromosomes, {n_variants_total:,} SNV positions, "
        f"up to {max_workers} parallel threads"
    )

    total_map: Dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(
                _scan_one_chrom,
                bam_path,
                chrom_map[vcf_chrom],       # use the BAM's contig name
                variant_positions[vcf_chrom],
            ): vcf_chrom
            for vcf_chrom in chroms_to_scan
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Scanning chromosomes"):
            chrom = futures[fut]
            try:
                chrom_map = fut.result()
            except Exception as exc:
                logger.warning(f"scan_chrom({chrom}) raised: {exc}")
                continue
            for bc, cnt in chrom_map.items():
                total_map[bc] = total_map.get(bc, 0) + cnt

    n_bc = len(total_map)
    total_hits = sum(total_map.values())
    logger.info(
        f"BAM scan complete: {n_bc} barcodes with ≥1 ALT read, "
        f"{total_hits:,} total (barcode × variant position) counts"
    )
    return total_map


def write_counts_csv(count_map: Dict[str, int], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["barcode", "snv_count"])
        for barcode, count in sorted(count_map.items()):
            writer.writerow([barcode, count])
    logger.info(f"Wrote per-barcode counts CSV: {out_path}")


# ============================================================================
# Path B — Python streaming VCF split + per-spot filter (multi-sample VCFs)
# ============================================================================

def split_merged_vcf(merged_vcf: str, raw_split_dir: str) -> List[str]:
    """
    Split a multi-sample VCF into one per-sample VCF.gz using a single Python
    streaming pass.  Replaces `bcftools +split`, which requires compiled plugin
    support not available with the bundled apps/bcftools binary.
    """
    os.makedirs(raw_split_dir, exist_ok=True)

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = min(hard, 65536)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
            logger.info(f"Raised RLIMIT_NOFILE: {soft} -> {target}")
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
        raise RuntimeError(f"No header lines found in {merged_vcf}")
    chrom_fields = header_lines[-1].rstrip("\n").split("\t")
    if len(chrom_fields) < 10:
        raise RuntimeError(
            f"#CHROM line has only {len(chrom_fields)} columns; expected ≥10"
        )
    fixed_cols   = chrom_fields[:9]
    sample_names = chrom_fields[9:]
    header_prefix = "".join(header_lines[:-1])

    logger.info(
        f"Streaming split of {len(sample_names)} samples "
        f"from {merged_vcf} -> {raw_split_dir}"
    )

    n_variants = 0
    handles: Dict[str, gzip.GzipFile] = {}
    try:
        for s in sample_names:
            p = os.path.join(raw_split_dir, f"{s}.vcf.gz")
            h = gzip.open(p, "wt")
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
                    logger.info(f"  …processed {n_variants:,} variant lines")
    finally:
        for h in handles.values():
            try:
                h.close()
            except Exception:
                pass

    files = sorted(glob.glob(os.path.join(raw_split_dir, "*.vcf.gz")))
    logger.info(
        f"Python streaming split produced {len(files)} per-sample VCFs "
        f"({n_variants:,} variant lines processed)"
    )
    if not files:
        raise RuntimeError(f"No per-sample VCFs produced in {raw_split_dir}.")
    return files


def filter_one_vcf(input_vcf: str, output_vcf: str) -> Tuple[bool, int, str]:
    """Filter GT=ref/missing, compress, tabix-index. Returns (ok, count, err_msg)."""
    cmd_view = (
        f"{PATH_CONFIG['BCFTOOLS']} view "
        f"-e 'GT=\"ref\" || GT=\"mis\"' "
        f"-Oz -o {output_vcf} {input_vcf}"
    )
    r1 = subprocess.run(cmd_view, shell=True, capture_output=True, text=True)
    if r1.returncode != 0:
        return False, 0, (r1.stderr or "view failed").strip()

    r2 = subprocess.run(
        f"{PATH_CONFIG['BCFTOOLS']} view -H {output_vcf} | wc -l",
        shell=True, capture_output=True, text=True,
    )
    try:
        count = int(r2.stdout.strip())
    except (ValueError, AttributeError):
        count = 0

    r3 = subprocess.run(
        f"{PATH_CONFIG['BCFTOOLS']} index -t -f {output_vcf}",
        shell=True, capture_output=True, text=True,
    )
    if r3.returncode != 0:
        return True, count, f"index_warning: {r3.stderr.strip()}"

    return True, count, ""


def filter_all_parallel(raw_files: List[str], output_dir: str,
                        max_workers: int = 30) -> Dict[str, int]:
    os.makedirs(output_dir, exist_ok=True)
    count_map: Dict[str, int] = {}
    failures: List[Tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        for raw in raw_files:
            barcode = os.path.basename(raw).replace(".vcf.gz", "")
            out_vcf = os.path.join(output_dir, f"{barcode}.vcf.gz")
            futures[ex.submit(filter_one_vcf, raw, out_vcf)] = barcode

        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Filter 0/0 + index"):
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

    logger.info(f"Filter complete: {len(count_map)} OK, {len(failures)} failed")
    if failures:
        logger.warning("First few failures:")
        for b, e in failures[:5]:
            logger.warning(f"  {b}: {e}")
    return count_map


# ============================================================================
# Spot positions / visualization / summary
# ============================================================================

def load_spot_positions(positions_file: str, has_header: bool) -> Dict[str, Tuple[float, float]]:
    """
    Load in-tissue spot positions from a Visium tissue_positions* CSV.
    Returns {barcode -> (pxl_col, pxl_row)}.
    """
    positions: Dict[str, Tuple[float, float]] = {}

    with open(positions_file, "r", newline="") as fh:
        if has_header:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    if int(float(row["in_tissue"])) != 1:
                        continue
                    barcode  = row["barcode"]
                    pxl_row  = float(row["pxl_row_in_fullres"])
                    pxl_col  = float(row["pxl_col_in_fullres"])
                    positions[barcode] = (pxl_col, pxl_row)
                except (KeyError, ValueError):
                    continue
        else:
            # Legacy header-less: barcode, in_tissue, array_row, array_col, pxl_row, pxl_col
            reader = csv.reader(fh)
            for row in reader:
                if len(row) < 6:
                    continue
                try:
                    if int(row[1]) != 1:
                        continue
                    barcode  = row[0]
                    pxl_row  = float(row[4])
                    pxl_col  = float(row[5])
                    positions[barcode] = (pxl_col, pxl_row)
                except ValueError:
                    continue

    logger.info(f"Loaded {len(positions)} in-tissue spot positions from {positions_file}")
    return positions


def load_scale_factor(scale_factor_file: str) -> float:
    with open(scale_factor_file, "r") as fh:
        sf = json.load(fh)
    return float(sf.get("tissue_hires_scalef", 1.0))


def visualize_spot_counts(
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    scale: float,
    image_path: str,
    x_flip: bool,
    title: str,
    out_path: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.imshow(img, origin="upper")

    xs, ys, colors, sizes = [], [], [], []
    counts_for_stats = []
    max_count = max(count_map.values()) if count_map else 0
    if max_count == 0:
        max_count = 1

    for barcode, (px_col, px_row) in positions.items():
        c = count_map.get(barcode, 0)
        xs.append(px_col * scale)
        ys.append(px_row * scale)
        colors.append(c)
        sizes.append(30 + (c * 20.0 / max_count))
        counts_for_stats.append(c)

    cmap = LinearSegmentedColormap.from_list("snv_count_cmap", COLORMAP)
    norm = Normalize(vmin=0, vmax=max(1, max(colors) if colors else 1))

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
        f"In-tissue spots: {len(positions)}\n"
        f"Spots with SNVs: {len(nz)}\n"
        f"Total SNVs (sum): {int(counts_arr.sum())}\n"
        f"Max SNVs / spot: {int(counts_arr.max())}\n"
        f"Mean SNVs / spot: {counts_arr.mean():.2f}\n"
        f"Median SNVs / spot: {np.median(counts_arr):.0f}"
    )
    ax.text(
        0.02, 0.98, stats,
        transform=ax.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved visualization: {out_path}")


def write_summary(
    profile_dir: str,
    dataset: str,
    section_id: str,
    count_map: Dict[str, int],
    positions: Dict[str, Tuple[float, float]],
    source_desc: str,
) -> None:
    summary_path = os.path.join(profile_dir, "summary.txt")
    counts_arr = np.array(list(count_map.values())) if count_map else np.array([0])
    nz = counts_arr[counts_arr > 0]

    in_tissue_set  = set(positions.keys())
    profile_set    = set(count_map.keys())
    matched        = in_tissue_set & profile_set

    with open(summary_path, "w") as f:
        f.write("Original SNP profile (raw mpileup) - summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Dataset:           {dataset}\n")
        f.write(f"Section:           {section_id}\n")
        f.write(f"Quality filter:    {QUALITY_FILTER}\n")
        f.write(f"Source:            {source_desc}\n")
        f.write("\n")
        f.write(f"Barcodes in count map:        {len(profile_set)}\n")
        f.write(f"In-tissue spots:              {len(in_tissue_set)}\n")
        f.write(f"In-tissue spots with profile: {len(matched)}\n")
        f.write(f"Spots with >=1 SNV:           {int(len(nz))}\n")
        f.write("\n")
        f.write("Per-spot SNV count statistics:\n")
        f.write(f"  total SNVs (sum):  {int(counts_arr.sum())}\n")
        f.write(f"  max:               {int(counts_arr.max())}\n")
        f.write(f"  mean:              {counts_arr.mean():.2f}\n")
        f.write(f"  median:            {np.median(counts_arr):.0f}\n")

    logger.info(f"Wrote summary: {summary_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-spot SNP profile from raw mpileup output.",
    )
    parser.add_argument(
        "--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--section-id", required=True)
    parser.add_argument(
        "--max-workers", type=int, default=22,
        help="Parallel workers: chromosomes for BAM scan, or bcftools threads for VCF split (default 22)",
    )
    parser.add_argument(
        "--vcf-override", default=None,
        help="Use this VCF instead of the default merged VCF resolved for this dataset/section.",
    )
    parser.add_argument(
        "--profile-subdir", default="original_snp_profile",
        help="Output subdirectory name under the section root (default: original_snp_profile).",
    )
    parser.add_argument(
        "--skip-vcf", action="store_true",
        help="Skip VCF/BAM processing; only re-render the visualization from an existing counts.csv",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep intermediate split VCF directory for debugging.",
    )
    args = parser.parse_args()

    valid_sections = DATASET_CONFIGS[args.dataset]["section_ids"]
    if args.section_id not in valid_sections:
        parser.error(
            f"--section-id {args.section_id} not valid for {args.dataset}. "
            f"Valid: {valid_sections}"
        )

    setup_environment()

    paths = resolve_paths(args.dataset, args.section_id)

    if args.vcf_override:
        paths["merged_vcf"] = args.vcf_override
    if args.profile_subdir != "original_snp_profile":
        paths["profile_dir"] = os.path.join(paths["section_root"], args.profile_subdir)

    logger.info("=" * 60)
    logger.info(f"Dataset:            {args.dataset}")
    logger.info(f"Section:            {args.section_id}")
    logger.info(f"Quality filter:     {QUALITY_FILTER}")
    logger.info(f"Merged VCF:         {paths['merged_vcf']}")
    logger.info(f"Profile output dir: {paths['profile_dir']}")
    logger.info(f"Position file:      {paths['position_file']}")
    logger.info(f"Image file:         {paths['image_file']}")
    if paths.get("bam_file"):
        logger.info(f"BAM file:           {paths['bam_file']}")
    logger.info("=" * 60)

    # Decide routing: BAM scan vs VCF split
    n_samples = count_vcf_samples(paths["merged_vcf"])
    use_bam_scan = n_samples <= 1 and paths.get("bam_file") is not None

    check_inputs(paths, require_bam=use_bam_scan)

    profile_dir = paths["profile_dir"]
    os.makedirs(profile_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Generate per-barcode count_map
    # ------------------------------------------------------------------
    count_map: Dict[str, int] = {}

    if not args.skip_vcf:
        if use_bam_scan:
            # Single-sample VCF → scan BAM for CB-tagged reads at variant positions
            logger.info(
                f"Merged VCF has {n_samples} sample column(s): "
                f"using pysam BAM scan (CB tag) from {paths['bam_file']}"
            )
            variant_positions = load_variant_positions(paths["merged_vcf"])
            count_map = scan_bam_for_spot_counts(
                paths["bam_file"],
                variant_positions,
                max_workers=min(args.max_workers, 22),
            )
            counts_csv = os.path.join(profile_dir, "counts.csv")
            write_counts_csv(count_map, counts_csv)
            source_desc = f"BAM pysam scan of {paths['bam_file']}"

        else:
            # Multi-sample VCF → Python streaming split + bcftools filter
            logger.info(
                f"Merged VCF has {n_samples} sample columns: "
                f"using streaming VCF split + per-spot filter"
            )
            for old in (
                glob.glob(os.path.join(profile_dir, "*.vcf.gz")) +
                glob.glob(os.path.join(profile_dir, "*.vcf.gz.tbi"))
            ):
                try:
                    os.remove(old)
                except OSError:
                    pass

            raw_split_dir = os.path.join(profile_dir, "_raw_split_tmp")
            if os.path.exists(raw_split_dir):
                shutil.rmtree(raw_split_dir)

            try:
                raw_files = split_merged_vcf(paths["merged_vcf"], raw_split_dir)
                count_map = filter_all_parallel(
                    raw_files, profile_dir, max_workers=args.max_workers,
                )
            finally:
                if not args.keep_tmp and os.path.exists(raw_split_dir):
                    shutil.rmtree(raw_split_dir, ignore_errors=True)

            source_desc = f"VCF split of {paths['merged_vcf']}"

    else:
        # Reload counts from existing outputs
        logger.info("--skip-vcf set: loading counts from existing outputs")
        counts_csv = os.path.join(profile_dir, "counts.csv")
        if os.path.exists(counts_csv):
            with open(counts_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        count_map[row["barcode"]] = int(row["snv_count"])
                    except (KeyError, ValueError):
                        pass
            logger.info(f"Loaded {len(count_map)} barcodes from {counts_csv}")
        else:
            for vcf in tqdm(sorted(glob.glob(os.path.join(profile_dir, "*.vcf.gz"))),
                            desc="Counting VCF records"):
                barcode = os.path.basename(vcf).replace(".vcf.gz", "")
                r = subprocess.run(
                    f"{PATH_CONFIG['BCFTOOLS']} view -H {vcf} | wc -l",
                    shell=True, capture_output=True, text=True,
                )
                try:
                    count_map[barcode] = int(r.stdout.strip())
                except ValueError:
                    count_map[barcode] = 0
        source_desc = "reloaded from existing outputs"

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------
    try:
        positions = load_spot_positions(paths["position_file"], paths["has_header"])
    except Exception as exc:
        logger.error(f"Failed to load spot positions: {exc}")
        positions = {}

    if positions:
        try:
            scale = load_scale_factor(paths["scale_factor_file"])
        except Exception as exc:
            logger.warning(f"Could not load scale factor ({exc}); using 1.0")
            scale = 1.0

        plot_path = os.path.join(profile_dir, "plots", "spot_snv_counts.png")
        title = (
            f"Original mpileup SNP profile — "
            f"{args.dataset} sec{args.section_id} ({QUALITY_FILTER})"
        )
        try:
            visualize_spot_counts(
                count_map=count_map,
                positions=positions,
                scale=scale,
                image_path=paths["image_file"],
                x_flip=paths["x_flip"],
                title=title,
                out_path=plot_path,
            )
        except Exception as exc:
            logger.error(f"Visualization failed: {exc}")
    else:
        logger.warning("Skipping visualization: no positions loaded.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    try:
        write_summary(
            profile_dir=profile_dir,
            dataset=args.dataset,
            section_id=args.section_id,
            count_map=count_map,
            positions=positions,
            source_desc=source_desc,
        )
    except Exception as exc:
        logger.warning(f"Failed to write summary: {exc}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
