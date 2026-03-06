"""
SPARCAL Step 1: Mpileup Variant Calling Pipeline
=================================================
Calls variants from split BAM files using samtools mpileup + bcftools.
Supports both multi-BAM (merge all spots per region) and single-BAM
(per-spot) calling modes.

Usage:
    python mpileup_pipeline.py --dataset DLPFC --section_id 151507
    python mpileup_pipeline.py --dataset P4_TUMOR --section_id 1 --filter_out_tissue
    python mpileup_pipeline.py --dataset DCIS --section_id 1 --call_mode single
"""

import os
import sys
import time
import glob
import gzip
import shutil
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config loader — replaces all hardcoded DATASET_CONFIGS / PATH_CONFIG / etc.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig


# ---------------------------------------------------------------------------
# Default mpileup-specific parameters (non-path, non-dataset)
# ---------------------------------------------------------------------------
MPILEUP_DEFAULTS = {
    "MAX_DEPTH": 10000000,
    "THREADS": 30,
    "MAX_FILES": None,
    "MIN_DEPTH": 0,
    "MIN_GQ": 8,
    "MIN_QUAL": 0,
}


# ========================== Environment Setup ===============================

def setup_environment(cfg: PipelineConfig) -> Dict[str, str]:
    """Prepend apps_dir to PATH and LD_LIBRARY_PATH."""
    os.environ['PATH'] = f"{cfg.apps_dir}:{os.environ.get('PATH', '')}"
    current_ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{cfg.apps_dir}:{current_ld}" if current_ld else cfg.apps_dir
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH'],
    }


# ========================== Output Directories ==============================

def setup_output_dirs(output_base: str, call_mode: str) -> Dict[str, str]:
    """Create and return output directory structure."""
    base_dir = "mpileup_multi_bam" if call_mode == "multi" else "mpileup_single_bam"

    output_structure = {
        "vcf_dir":     os.path.join(output_base, "output_VCFs", base_dir),
        "log_dir":     os.path.join(output_base, "logs", base_dir),
        "metrics_dir": os.path.join(output_base, "metrics", base_dir),
    }
    for d in output_structure.values():
        os.makedirs(d, exist_ok=True)
    return output_structure


# ========================== Shell Helpers ===================================

def run_command(cmd, env=None, **kwargs) -> subprocess.CompletedProcess:
    """Run a shell command with optional environment."""
    if env is None:
        env = os.environ.copy()
    else:
        env = {**os.environ.copy(), **env}
    return subprocess.run(cmd, env=env, **kwargs)


# ========================== BAM / Barcode Helpers ===========================

def read_barcode_list(barcode_file: str) -> List[str]:
    """Read barcodes from a TSV file (compressed or uncompressed)."""
    opener = gzip.open if barcode_file.endswith('.gz') else open
    mode = 'rt' if barcode_file.endswith('.gz') else 'r'
    with opener(barcode_file, mode) as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]


def load_in_tissue_barcodes(cfg: PipelineConfig) -> Set[str]:
    """
    Load barcodes that are within tissue boundaries.

    Supports three strategies:
      1. Position file with in_tissue column (DLPFC, DCIS).
      2. Missing-barcodes blacklist + position file (P4/P6 TUMOR).
      3. Barcode whitelist file (fallback for DCIS).

    Returns an empty set when all BAMs should be used.
    """
    in_tissue = set()

    # ---- Strategy A: position file with in_tissue column ----
    position_path = _resolve_position_path(cfg)
    if position_path and os.path.exists(position_path):
        try:
            header = 0 if cfg.has_header else None
            df = pd.read_csv(position_path, header=header)
            col = cfg.in_tissue_column
            for _, row in df.iterrows():
                if int(row.iloc[col]) == 1:
                    in_tissue.add(str(row.iloc[0]))
            print(f"Loaded {len(in_tissue)} in-tissue barcodes from {position_path}")
            return in_tissue
        except Exception as e:
            print(f"Warning: could not parse position file {position_path}: {e}")

    # ---- Strategy B: missing_tissue_file (blacklist) ----
    missing_file_raw = cfg.raw.get("spatial", {}).get("missing_tissue_file", "")
    if missing_file_raw and cfg.section_id:
        missing_file = missing_file_raw.replace("{section_id}", str(cfg.section_id))
        if not os.path.isabs(missing_file):
            missing_file = os.path.join(cfg.bam_base_path, missing_file)

        if os.path.exists(missing_file):
            out_tissue = set()
            with open(missing_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        out_tissue.add(parts[0])
            print(f"Loaded {len(out_tissue)} out-of-tissue barcodes from {missing_file}")

            # Now read ALL barcodes from position file and subtract
            if position_path and os.path.exists(position_path):
                header = 0 if cfg.has_header else None
                df = pd.read_csv(position_path, header=header)
                for _, row in df.iterrows():
                    bc = str(row.iloc[0])
                    if bc not in out_tissue:
                        in_tissue.add(bc)
                print(f"After blacklist: {len(in_tissue)} in-tissue barcodes")
                return in_tissue

    # ---- Fallback: use all BAMs ----
    print("No tissue filtering info available — using all BAMs.")
    return set()


def _resolve_position_path(cfg: PipelineConfig) -> Optional[str]:
    """Build the full path to the tissue positions file."""
    if not cfg.position_file:
        return None
    spatial_dir = cfg.spatial_dir
    if spatial_dir:
        return os.path.join(spatial_dir, cfg.position_file)
    return None


def get_bam_list_for_tumor(cfg: PipelineConfig) -> List[str]:
    """Generate list of BAM files from barcode reference file."""
    barcode_pattern = cfg.raw.get("input", {}).get("barcode_file_pattern", "")
    if not barcode_pattern:
        return []

    barcode_file = barcode_pattern.replace("{section_id}", str(cfg.section_id))
    if not os.path.isabs(barcode_file):
        barcode_file = os.path.join(cfg.bam_base_path, barcode_file)

    barcodes = read_barcode_list(barcode_file)
    split_bam_dir = os.path.dirname(
        os.path.join(cfg.bam_base_path, cfg.bam_pattern)
    )

    bam_files = []
    for bc in barcodes:
        bam = os.path.join(split_bam_dir, f"{bc}.bam")
        if os.path.exists(bam):
            bam_files.append(bam)
    return bam_files


# ========================== Genotype Inference ==============================

def infer_gt_from_pl(input_vcf: str, output_vcf: str,
                     min_depth: int = 5, min_gq: float = 7, min_qual: int = 0):
    """
    Infer genotypes from PL fields and calculate BAF using I16 values.
    Filters hom-ref calls and applies quality thresholds.
    """

    def calculate_gq(pl_values):
        sorted_pls = sorted(pl_values)
        return sorted_pls[1] - sorted_pls[0]

    def calculate_baf_from_i16(i16_values):
        try:
            ref_depth = i16_values[0] + i16_values[1]
            alt_depth = i16_values[2] + i16_values[3]
            total = ref_depth + alt_depth
            return alt_depth / total if total > 0 else 0.0
        except (IndexError, ZeroDivisionError):
            return 0.0

    def parse_i16(i16_str):
        try:
            return [int(x) for x in i16_str.split(',')]
        except (ValueError, AttributeError):
            return [0] * 16

    infile = gzip.open(input_vcf, 'rt') if input_vcf.endswith('.gz') else open(input_vcf, 'r')

    with infile, open(output_vcf, 'w') as f_out:
        for line in infile:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    f_out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                    f_out.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
                    f_out.write('##FORMAT=<ID=BAF,Number=1,Type=Float,Description="B-Allele Frequency from I16">\n')
                    f_out.write('##FILTER=<ID=LowQual,Description="Low quality variant call">\n')
                    f_out.write('##FILTER=<ID=LowDP,Description="Low read depth">\n')
                    f_out.write('##FILTER=<ID=LowGQ,Description="Low genotype quality">\n')
                    f_out.write('##FILTER=<ID=DiscordantBAF,Description="BAF inconsistent with genotype">\n')
                f_out.write(line)
                continue

            fields = line.strip().split('\t')
            info_dict = dict(item.split('=') for item in fields[7].split(';') if '=' in item)
            depth = int(info_dict.get('DP', 0))
            i16_values = parse_i16(info_dict.get('I16', ''))

            format_field = fields[8]
            sample_field = fields[9]
            format_dict = dict(zip(format_field.split(':'), sample_field.split(':')))

            try:
                pl_values = [int(x) for x in format_dict.get('PL', '').split(',')]
                if len(pl_values) != 3:
                    continue
                qual = float(fields[5]) if fields[5] != '.' else 0
                baf = calculate_baf_from_i16(i16_values)
            except (ValueError, KeyError, IndexError):
                continue

            min_pl_idx = pl_values.index(min(pl_values))
            inferred_gt = {0: '0/0', 1: '0/1', 2: '1/1'}[min_pl_idx]
            if inferred_gt == '0/0':
                continue

            gq = calculate_gq(pl_values)

            # BAF concordance check
            baf_valid = True
            if inferred_gt == '0/1' and (baf < 0.35 or baf > 0.65):
                baf_valid = False
            elif inferred_gt == '1/1' and baf < 0.85:
                baf_valid = False

            filters = []
            if depth < min_depth:
                filters.append('LowDP')
            if gq < min_gq:
                filters.append('LowGQ')
            if qual < min_qual:
                filters.append('LowQual')
            if not baf_valid:
                filters.append('DiscordantBAF')

            fields[6] = ';'.join(filters) if filters else 'PASS'
            fields[8] = f"GT:GQ:BAF:{format_field}"
            fields[9] = f"{inferred_gt}:{gq}:{baf:.3f}:{sample_field}"
            f_out.write('\t'.join(fields) + '\n')


# ========================== VCF Merging =====================================

def merge_vcfs(vcf_files: List[str], output_dir: str,
               params: Dict, cfg: PipelineConfig) -> str:
    """Merge per-region VCFs, infer genotypes, sort, compress, index."""
    os.makedirs(output_dir, exist_ok=True)

    temp_merged     = os.path.join(output_dir, "temp_merged.vcf")
    temp_gt         = os.path.join(output_dir, "temp_gt_inferred.vcf")
    final_vcf       = os.path.join(output_dir, "merged_sorted_gt.vcf")
    final_vcf_gz    = f"{final_vcf}.gz"

    # Concatenate VCFs (header from first, variants from all)
    with open(temp_merged, 'w') as out:
        with gzip.open(vcf_files[0], 'rt') as first:
            for line in first:
                if line.startswith('#'):
                    out.write(line)
                else:
                    break
        for vcf in vcf_files:
            with gzip.open(vcf, 'rt') as fh:
                for line in fh:
                    if not line.startswith('#'):
                        out.write(line)

    infer_gt_from_pl(temp_merged, temp_gt,
                     params.get('MIN_DEPTH', 0),
                     params.get('MIN_GQ', 8),
                     params.get('MIN_QUAL', 0))

    run_command(f"{cfg.tool('bcftools')} sort {temp_gt} -o {final_vcf}", shell=True)
    run_command([cfg.tool('bgzip'), '-f', final_vcf])
    run_command([cfg.tool('tabix'), '-p', 'vcf', final_vcf_gz])

    os.remove(temp_merged)
    if os.path.exists(temp_gt):
        os.remove(temp_gt)

    return final_vcf_gz


# ========================== Chr-prefix Handling =============================

def add_chr_prefix_to_vcf(input_vcf: str, output_vcf: str, cfg: PipelineConfig):
    """Add 'chr' prefix to contig names in a VCF."""
    temp_vcf = output_vcf.replace('.gz', '')

    opener = gzip.open if input_vcf.endswith('.gz') else open
    mode = 'rt' if input_vcf.endswith('.gz') else 'r'

    with opener(input_vcf, mode) as f_in, open(temp_vcf, 'w') as f_out:
        for line in f_in:
            if line.startswith('#'):
                f_out.write(line)
                continue
            fields = line.strip().split('\t')
            if fields[0].isdigit() or fields[0] in ('X', 'Y'):
                fields[0] = 'chr' + fields[0]
            f_out.write('\t'.join(fields) + '\n')

    run_command(f"{cfg.tool('bgzip')} -f {temp_vcf}", shell=True)
    run_command([cfg.tool('tabix'), '-p', 'vcf', output_vcf])
    if os.path.exists(temp_vcf):
        os.remove(temp_vcf)


def process_vcfs_for_chr_prefix(output_dirs: Dict[str, str], call_mode: str,
                                cfg: PipelineConfig):
    """Add chr prefix to all output VCFs if the reference has no chr prefix."""
    if cfg.chr_prefix != '':
        return  # already has chr prefix

    vcf_dir = output_dirs['vcf_dir']
    if call_mode == 'multi':
        merged = os.path.join(vcf_dir, "merged_sorted_gt.vcf.gz")
        if os.path.exists(merged):
            tmp = os.path.join(vcf_dir, "merged_sorted_gt_chr.vcf.gz")
            add_chr_prefix_to_vcf(merged, tmp, cfg)
            os.rename(tmp, merged)
            if os.path.exists(tmp + '.tbi'):
                os.rename(tmp + '.tbi', merged + '.tbi')
    else:
        for vcf in glob.glob(os.path.join(vcf_dir, "*.vcf.gz")):
            if vcf.endswith('.tbi'):
                continue
            tmp = vcf.replace('.vcf.gz', '_chr.vcf.gz')
            add_chr_prefix_to_vcf(vcf, tmp, cfg)
            os.rename(tmp, vcf)
            if os.path.exists(tmp + '.tbi'):
                os.rename(tmp + '.tbi', vcf + '.tbi')


# ========================== Per-BAM Processing ==============================

def process_single_bam(bam_file: str, output_dirs: Dict[str, str],
                       params: Dict, cfg: PipelineConfig) -> Dict:
    """Process a single BAM file to generate one VCF."""
    bam_name = os.path.basename(bam_file).replace('.bam', '')
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"{bam_name}.vcf.gz")
    temp_vcf   = os.path.join(output_dirs["vcf_dir"], f"{bam_name}_temp.vcf")
    log_file   = os.path.join(output_dirs["log_dir"], f"{bam_name}.log")

    start_time = time.time()
    env = setup_environment(cfg)

    try:
        mpileup_cmd = (
            f"{cfg.tool('samtools')} mpileup "
            f"-f {cfg.reference_fasta} "
            f"-q {cfg.mapping_quality} "
            f"-Q {cfg.base_quality} "
            f"-d {params['MAX_DEPTH']} -v "
            f"{bam_file} | "
            f"{cfg.tool('bcftools')} view | "
            f"{cfg.tool('bcftools')} filter -e 'REF !~ \"^[ATGC]$\"' | "
            f"{cfg.tool('bcftools')} norm -m-both -f {cfg.reference_fasta} | "
            f"grep -v '<X>\\|INDEL' > {temp_vcf}"
        )

        with open(log_file, 'w') as log:
            log.write(f"Command: {mpileup_cmd}\n")
            proc = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
            if proc.returncode != 0:
                raise Exception(f"mpileup failed (rc={proc.returncode})")

        snp_count = sum(1 for l in open(temp_vcf) if not l.startswith('#'))

        run_command(f"{cfg.tool('bgzip')} -c {temp_vcf} > {output_vcf}", shell=True)
        run_command([cfg.tool('tabix'), '-p', 'vcf', output_vcf])
        os.remove(temp_vcf)

        return {"bam": bam_file, "duration": time.time() - start_time,
                "snp_count": snp_count, "output_vcf": output_vcf, "status": "completed"}
    except Exception as e:
        print(f"Error processing {bam_file}: {e}")
        return {"bam": bam_file, "duration": time.time() - start_time,
                "snp_count": 0, "status": "failed", "error": str(e)}


def process_single_bam_by_region(bam_file: str, output_dirs: Dict[str, str],
                                 params: Dict, cfg: PipelineConfig) -> Dict:
    """Process a single BAM file region-by-region, then merge."""
    bam_name = os.path.basename(bam_file).replace('.bam', '')
    final_vcf = os.path.join(output_dirs["vcf_dir"], f"{bam_name}.vcf.gz")
    temp_dir  = os.path.join(output_dirs["vcf_dir"], f"temp_{bam_name}")
    os.makedirs(temp_dir, exist_ok=True)

    start_time = time.time()
    region_results = []
    region_vcfs = []

    for region in cfg.regions:
        region_vcf = os.path.join(temp_dir, f"{region}.vcf.gz")
        log_file   = os.path.join(output_dirs["log_dir"], f"{bam_name}_{region}.log")

        try:
            mpileup_cmd = (
                f"{cfg.tool('samtools')} mpileup "
                f"-f {cfg.reference_fasta} -r {region} "
                f"-q {cfg.mapping_quality} -Q {cfg.base_quality} "
                f"-d {params['MAX_DEPTH']} -v "
                f"{bam_file} | "
                f"{cfg.tool('bcftools')} view | "
                f"{cfg.tool('bcftools')} filter -e 'REF !~ \"^[ATGC]$\"' | "
                f"{cfg.tool('bcftools')} norm -m-both -f {cfg.reference_fasta} | "
                f"grep -v '<X>\\|INDEL' | "
                f"{cfg.tool('bgzip')} -c > {region_vcf}"
            )
            with open(log_file, 'w') as log:
                env = setup_environment(cfg)
                proc = run_command(mpileup_cmd, env=env, shell=True, stderr=log)

            if proc.returncode == 0 and os.path.exists(region_vcf) and os.path.getsize(region_vcf) > 0:
                run_command([cfg.tool('tabix'), '-p', 'vcf', region_vcf], check=True)
                snp_count = sum(1 for l in gzip.open(region_vcf, 'rt') if not l.startswith('#'))
                region_results.append({"region": region, "snp_count": snp_count, "status": "completed"})
                region_vcfs.append(region_vcf)
            else:
                with gzip.open(region_vcf, 'wt') as f:
                    f.write("##fileformat=VCFv4.2\n")
                    f.write(f"##reference={cfg.reference_fasta}\n")
                    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                run_command([cfg.tool('tabix'), '-p', 'vcf', region_vcf], check=True)
                region_vcfs.append(region_vcf)
                region_results.append({"region": region, "snp_count": 0, "status": "failed"})

        except Exception as e:
            print(f"Error region {region} for {bam_name}: {e}")
            region_results.append({"region": region, "snp_count": 0, "status": "failed", "error": str(e)})

    if region_vcfs:
        try:
            merged = merge_vcfs(region_vcfs, temp_dir, params, cfg)
            shutil.move(merged, final_vcf)
            if os.path.exists(merged + '.tbi'):
                shutil.move(merged + '.tbi', final_vcf + '.tbi')
            shutil.rmtree(temp_dir)
            total_snps = sum(r["snp_count"] for r in region_results if r["status"] == "completed")
            return {"bam": bam_file, "duration": time.time() - start_time,
                    "snp_count": total_snps, "output_vcf": final_vcf,
                    "status": "completed", "region_results": region_results}
        except Exception as e:
            return {"bam": bam_file, "duration": time.time() - start_time,
                    "snp_count": 0, "status": "failed",
                    "error": f"Merge error: {e}", "region_results": region_results}

    return {"bam": bam_file, "duration": time.time() - start_time,
            "snp_count": 0, "status": "failed",
            "error": "No regions processed", "region_results": region_results}


# ========================== Per-Region Processing (multi-BAM) ===============

def process_region(region: str, bam_input: str, output_dirs: Dict[str, str],
                   params: Dict, cfg: PipelineConfig,
                   is_bam_list: bool = True) -> Optional[Dict]:
    """Process a specific genomic region across all BAMs."""
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"region_{region}.vcf.gz")
    log_file   = os.path.join(output_dirs["log_dir"], f"region_{region}.log")
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)

    start_time = time.time()
    env = setup_environment(cfg)

    bam_input_cmd = f"-b {bam_input}" if is_bam_list else bam_input

    try:
        mpileup_cmd = (
            f"{cfg.tool('samtools')} mpileup "
            f"-f {cfg.reference_fasta} -r {region} "
            f"-q {cfg.mapping_quality} -Q {cfg.base_quality} "
            f"-d {params['MAX_DEPTH']} -v "
            f"{bam_input_cmd} | "
            f"{cfg.tool('bcftools')} view | "
            f"{cfg.tool('bcftools')} filter -e 'REF !~ \"^[ATGC]$\"' | "
            f"{cfg.tool('bcftools')} norm -m-both -f {cfg.reference_fasta} | "
            f"grep -v '<X>\\|INDEL' | "
            f"{cfg.tool('bgzip')} -c > {output_vcf}"
        )

        with open(log_file, 'w') as log:
            proc = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
            if proc.returncode != 0:
                raise Exception(f"mpileup failed (rc={proc.returncode})")

        snp_count = sum(1 for l in gzip.open(output_vcf, 'rt') if not l.startswith('#'))
        return {"region": region, "duration": time.time() - start_time,
                "snp_count": snp_count, "output_vcf": output_vcf}
    except Exception as e:
        print(f"Error processing region {region}: {e}")
        return None


# ========================== Main Pipeline ===================================

def run_pipeline(cfg: PipelineConfig, call_mode: str = "multi",
                 filter_out_tissue: bool = False,
                 custom_params: Optional[Dict] = None):
    """Run the mpileup variant calling pipeline for a dataset section."""
    params = MPILEUP_DEFAULTS.copy()
    if custom_params:
        params.update(custom_params)

    # Setup
    output_dirs = setup_output_dirs(cfg.output_dir, call_mode)

    # Discover BAM files
    bam_pattern = os.path.join(cfg.bam_base_path, cfg.bam_pattern)
    bam_files = sorted(glob.glob(bam_pattern))
    print(f"Found {len(bam_files)} BAM files matching {bam_pattern}")
    if bam_files:
        print(f"  Example: {bam_files[0]}")

    if params["MAX_FILES"]:
        bam_files = bam_files[:params["MAX_FILES"]]

    if not bam_files:
        raise ValueError(f"No BAM files found at: {bam_pattern}")

    # Tissue filtering
    if filter_out_tissue:
        print("Filtering out-of-tissue spots...")
        in_tissue_barcodes = load_in_tissue_barcodes(cfg)
        if in_tissue_barcodes:
            before = len(bam_files)
            bam_files = [b for b in bam_files
                         if os.path.basename(b).replace('.bam', '') in in_tissue_barcodes]
            print(f"Filtered {before - len(bam_files)} out-of-tissue BAMs → {len(bam_files)} remain")

            # Save barcode list for reference
            bc_file = os.path.join(output_dirs["log_dir"], "in_tissue_barcodes.txt")
            with open(bc_file, 'w') as f:
                for bc in sorted(in_tissue_barcodes):
                    f.write(f"{bc}\n")
    else:
        print(f"Processing {len(bam_files)} BAM files (no tissue filtering)")

    # ---- Run ----
    results = []

    if call_mode == "multi":
        # Write BAM list file
        bam_list_file = os.path.join(output_dirs["log_dir"], "bam_list.txt")
        with open(bam_list_file, 'w') as f:
            for bam in bam_files:
                f.write(f"{bam}\n")

        # Parallel region processing
        with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
            futures = {
                executor.submit(process_region, region, bam_list_file,
                                output_dirs, params, cfg, True): region
                for region in cfg.regions
            }
            with tqdm(total=len(cfg.regions), desc="Processing regions") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)

        # Merge per-region VCFs
        region_vcfs = [r["output_vcf"] for r in results]
        merge_vcfs(region_vcfs, output_dirs["vcf_dir"], params, cfg)

    else:
        # Single-BAM mode: parallel per BAM
        with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
            futures = {
                executor.submit(process_single_bam, bam, output_dirs, params, cfg): bam
                for bam in bam_files
            }
            with tqdm(total=len(bam_files), desc="Processing BAMs") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)

    # Add chr prefix if needed
    process_vcfs_for_chr_prefix(output_dirs, call_mode, cfg)

    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_file = os.path.join(output_dirs["metrics_dir"], "processing_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    return metrics_df


# ========================== CLI =============================================

def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 1: Mpileup Variant Calling Pipeline")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match a YAML config file)")
    parser.add_argument("--section_id",
                        help="Section ID (required for multi-section datasets)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max BAM files to process (for debugging)")
    parser.add_argument("--threads", type=int, default=30,
                        help="Number of parallel threads")
    parser.add_argument("--call_mode", choices=["single", "multi"], default="multi",
                        help="single = per-BAM VCFs; multi = merge all BAMs per region")
    parser.add_argument("--filter_out_tissue", action="store_true",
                        help="Exclude spots outside tissue boundaries")
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.dataset, args.section_id)
    print(f"Dataset:   {cfg.dataset_name}")
    print(f"Section:   {cfg.section_id}")
    print(f"Tissue:    {cfg.tissue_type}")
    print(f"Reference: {cfg.reference_fasta}")
    print(f"Regions:   {cfg.regions[0]}..{cfg.regions[-1]}")

    env = setup_environment(cfg)
    print(f"Apps PATH: {cfg.apps_dir}")

    custom_params = {
        "MAX_FILES": args.max_files,
        "THREADS": args.threads,
    }

    metrics_df = run_pipeline(cfg, args.call_mode, args.filter_out_tissue, custom_params)

    # Summary
    print("\n--- Pipeline Summary ---")
    if args.call_mode == "multi":
        print(f"Regions processed: {len(metrics_df)}")
        print(f"Avg time/region:   {metrics_df['duration'].mean():.2f}s")
        print(f"Total SNPs:        {metrics_df['snp_count'].sum()}")
    else:
        ok = len(metrics_df[metrics_df['status'] == 'completed'])
        fail = len(metrics_df[metrics_df['status'] == 'failed'])
        print(f"BAMs processed: {ok} ok, {fail} failed")
        print(f"Avg time/BAM:   {metrics_df['duration'].mean():.2f}s")
        print(f"Total SNPs:     {metrics_df['snp_count'].sum()}")


if __name__ == "__main__":
    main()