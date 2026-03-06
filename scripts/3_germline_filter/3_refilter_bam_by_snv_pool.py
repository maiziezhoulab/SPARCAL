"""
SPARCAL Step 5: Filter BAMs by SNV Pools
==========================================
Filters per-spot BAM files to retain only reads overlapping SNV positions
from Beagle (defined/1kG variants) and Classifier (de novo variants).
Produces per-barcode filtered BAMs and per-barcode SNV VCFs.

Used by both germline and somatic pipelines (somatic skips the Classifier
inputs and only uses Beagle-defined variants).

Usage:
    python run_filter_bams_by_snv_pools.py --dataset DLPFC --section-id 151507
    python run_filter_bams_by_snv_pools.py --dataset P4_TUMOR --section-id 1
"""

import os
import sys
import gzip
import glob
import subprocess
import argparse
import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig


# ===========================================================================
# Global tool-path holder (set once in main, used by free functions)
# ===========================================================================
_TOOL_PATHS: Dict[str, str] = {}


def _init_tools(cfg: PipelineConfig):
    """Cache tool paths from config for free functions."""
    global _TOOL_PATHS
    _TOOL_PATHS = {
        'samtools': cfg.tool('samtools'),
        'bcftools': cfg.tool('bcftools'),
        'bgzip':    cfg.tool('bgzip'),
        'tabix':    cfg.tool('tabix'),
    }
    apps = cfg.apps_dir
    os.environ['PATH'] = f"{apps}:{os.environ.get('PATH', '')}"
    ld = os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['LD_LIBRARY_PATH'] = f"{apps}:{ld}" if ld else apps


# ===========================================================================
# Data classes
# ===========================================================================
@dataclass
class SNVInfo:
    chrom: str
    pos: int
    ref: str
    alt: str
    info: str = ""
    format_str: str = ""
    race: str = ""  # "defined" (Beagle/1kG) or "denovo" (Classifier)

    def __eq__(self, other):
        if isinstance(other, SNVInfo):
            return ((self.standardized_chrom, self.pos, self.ref, self.alt) ==
                    (other.standardized_chrom, other.pos, other.ref, other.alt))
        return False

    def __hash__(self):
        return hash((self.standardized_chrom, self.pos, self.ref, self.alt))

    @property
    def key(self) -> str:
        return f"{self.standardized_chrom}_{self.pos}_{self.ref}_{self.alt}"

    @property
    def standardized_chrom(self) -> str:
        return self.chrom.replace("chr", "")

    @classmethod
    def from_vcf_line(cls, line: str) -> 'SNVInfo':
        f = line.strip().split('\t')
        return cls(chrom=f[0], pos=int(f[1]), ref=f[3], alt=f[4],
                   info=f[7] if len(f) > 7 else "",
                   format_str=f[8] if len(f) > 8 else "")


# ===========================================================================
# Shell / BAM helpers
# ===========================================================================
def run_command(cmd: str, log_file: Optional[str] = None) -> bool:
    try:
        kw = dict(shell=True, check=True)
        if log_file:
            with open(log_file, 'a') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, **kw)
        else:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        return True
    except subprocess.CalledProcessError as e:
        if not log_file:
            print(f"Command failed: {cmd}\n  {e}")
        return False


def sort_bam_file(input_bam: str, output_bam: str) -> bool:
    temp_pfx = os.path.join(os.path.dirname(output_bam),
                            f"temp_{os.path.basename(output_bam).replace('.bam', '')}")
    return run_command(f"{_TOOL_PATHS['samtools']} sort -o {output_bam} -T {temp_pfx} {input_bam}")


def index_bam_file(bam_path: str) -> bool:
    return run_command(f"{_TOOL_PATHS['samtools']} index {bam_path}")


# ===========================================================================
# Per-chromosome BAM filtering
# ===========================================================================
def filter_bam_one_chrom(input_bam: str, chrom: str,
                         positions: List[int]) -> Tuple[List, List[int]]:
    """Filter reads overlapping sorted positions on one chromosome."""
    filtered_reads = []
    detected = set()
    start_j = 0

    try:
        with pysam.AlignmentFile(input_bam, "rb") as bam:
            for read in bam.fetch(chrom):
                if read.is_unmapped:
                    continue
                rs = read.reference_start + 1
                re_ = (read.reference_end + 1) if read.reference_end else rs

                while start_j < len(positions) and positions[start_j] < rs:
                    start_j += 1

                j = start_j
                while j < len(positions) and positions[j] <= re_:
                    if rs <= positions[j] <= re_:
                        filtered_reads.append(read)
                        detected.add(positions[j])
                        break
                    j += 1
    except Exception as e:
        print(f"Error on {chrom} in {input_bam}: {e}")

    return filtered_reads, list(detected)


def filter_bam_by_positions(input_bam: str, output_bam: str,
                            positions_by_chrom: Dict) -> Dict:
    """Filter BAM to reads overlapping SNV positions across all chroms."""
    try:
        temp_out = f"{output_bam}.unsorted"
        detected_snvs = set()

        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            refs = set(in_bam.references)
            with pysam.AlignmentFile(temp_out, "wb", header=in_bam.header) as out_bam:
                for chrom, pos_info_list in positions_by_chrom.items():
                    chr_w = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                    chr_wo = chrom.replace("chr", "")
                    bam_chrom = chr_w if chr_w in refs else (chr_wo if chr_wo in refs else None)
                    if not bam_chrom:
                        continue

                    positions = sorted([p for p, r, a in pos_info_list])
                    pos_map = {p: (r, a) for p, r, a in pos_info_list}

                    reads, det_pos = filter_bam_one_chrom(input_bam, bam_chrom, positions)
                    for p in det_pos:
                        r, a = pos_map[p]
                        detected_snvs.add((bam_chrom, p, r, a))
                    for rd in reads:
                        out_bam.write(rd)

        ok = sort_bam_file(temp_out, output_bam)
        if os.path.exists(temp_out):
            os.remove(temp_out)
        if not ok:
            return {'input_bam': input_bam, 'output_bam': output_bam,
                    'status': 'failed', 'error': 'sort failed', 'detected_snvs': []}

        index_bam_file(output_bam)
        return {'input_bam': input_bam, 'output_bam': output_bam,
                'status': 'completed', 'detected_snvs': list(detected_snvs)}
    except Exception as e:
        return {'input_bam': input_bam, 'output_bam': output_bam,
                'status': 'failed', 'error': str(e), 'detected_snvs': []}


def process_single_bam(bam_path: str, output_dir: str,
                       positions_by_chrom: Dict) -> Dict:
    bam_name = os.path.basename(bam_path)
    output_bam = os.path.join(output_dir, bam_name)
    return filter_bam_by_positions(bam_path, output_bam, positions_by_chrom)


# ===========================================================================
# Save per-barcode SNV VCFs
# ===========================================================================
def save_detected_snvs(output_dir: str, result: Dict,
                       snv_info_dict: Dict = None) -> bool:
    os.makedirs(output_dir, exist_ok=True)
    if result['status'] != 'completed' or not result.get('detected_snvs'):
        return False

    barcode = os.path.basename(result['input_bam']).replace('.bam', '')
    output_file = os.path.join(output_dir, f"{barcode}.vcf.gz")
    temp_vcf = os.path.join(output_dir, f"{barcode}.vcf")

    try:
        sorted_snvs = sorted(result['detected_snvs'], key=lambda x: (x[0], x[1]))
        with open(temp_vcf, 'w') as f:
            f.write("##fileformat=VCFv4.2\n")
            f.write("##source=SPARCAL_filter_bams_by_snv_pools\n")
            f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
            f.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
            f.write("##INFO=<ID=RACE,Number=1,Type=String,Description=\"SNV origin: defined or denovo\">\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{barcode}\n")

            for snv in sorted_snvs:
                if len(snv) < 4:
                    continue
                chrom, pos, ref, alt = snv[:4]
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"

                info_field = "."
                race = "unknown"
                if snv_info_dict:
                    key = (chrom.replace("chr", ""), pos, ref, alt)
                    if key in snv_info_dict:
                        si = snv_info_dict[key]
                        info_field = si.info if si.info else "."
                        race = si.race if si.race else "unknown"

                if info_field == ".":
                    info_field = f"RACE={race}"
                else:
                    info_field = f"{info_field};RACE={race}"

                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_field}\tGT\t./.\n")

        run_command(f"{_TOOL_PATHS['bgzip']} -f {temp_vcf}")
        run_command(f"{_TOOL_PATHS['bcftools']} index -t {output_file}")
        return True
    except Exception as e:
        print(f"Error saving SNVs for {barcode}: {e}")
        if os.path.exists(temp_vcf):
            os.remove(temp_vcf)
        return False


# ===========================================================================
# Parallel BAM filtering
# ===========================================================================
def filter_bams_parallel(input_bams: List[str], output_dir: str,
                         snvs: Set[SNVInfo], max_workers: int = 30,
                         snv_info_dict: Dict = None) -> List[Dict]:
    os.makedirs(output_dir, exist_ok=True)
    snv_vcf_dir = os.path.join(os.path.dirname(output_dir), "snv_vcf")
    os.makedirs(snv_vcf_dir, exist_ok=True)

    # Build positions_by_chrom
    positions_by_chrom: Dict[str, List[Tuple]] = defaultdict(list)
    for snv in snvs:
        positions_by_chrom[snv.standardized_chrom].append((snv.pos, snv.ref, snv.alt))
    for chrom in positions_by_chrom:
        positions_by_chrom[chrom].sort(key=lambda x: x[0])

    # Build snv_info_dict from snvs
    if snv_info_dict is None:
        snv_info_dict = {}
        for snv in snvs:
            snv_info_dict[(snv.standardized_chrom, snv.pos, snv.ref, snv.alt)] = snv

    # Save all_variants VCF
    all_var_vcf = os.path.join(output_dir, "all_input_variants.vcf")
    with open(all_var_vcf, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for chrom, plist in sorted(positions_by_chrom.items()):
            for pos, ref, alt in plist:
                key = (chrom, pos, ref, alt)
                info = snv_info_dict[key].info if key in snv_info_dict and snv_info_dict[key].info else "."
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info}\n")
    run_command(f"{_TOOL_PATHS['bgzip']} -f {all_var_vcf}")
    run_command(f"{_TOOL_PATHS['bcftools']} index -t {all_var_vcf}.gz")

    results = []
    saved = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_bam, bam, output_dir, positions_by_chrom): bam
            for bam in input_bams
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Filtering BAMs"):
            try:
                result = future.result()
                results.append(result)
                if save_detected_snvs(snv_vcf_dir, result, snv_info_dict):
                    saved += 1
            except Exception as e:
                bam = futures[future]
                results.append({'input_bam': bam, 'status': 'failed',
                                'error': str(e), 'detected_snvs': []})

    print(f"\nBAMs with saved SNV VCFs: {saved}")

    # Create summary
    create_all_variants_summary(output_dir, snv_vcf_dir)
    return results


def create_all_variants_summary(output_dir: str, snv_vcf_dir: str):
    """Aggregate per-barcode VCFs into a count-annotated summary."""
    summary = os.path.join(output_dir, "all_detected_variants_summary.vcf")
    summary_gz = summary + ".gz"

    all_variants = {}
    bc_count = 0
    for vcf_file in glob.glob(os.path.join(snv_vcf_dir, "*.vcf.gz")):
        bc_count += 1
        try:
            with gzip.open(vcf_file, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    key = (parts[0], parts[1], parts[3], parts[4])
                    if key not in all_variants:
                        all_variants[key] = {'count': 0,
                                             'info': parts[7] if len(parts) > 7 else "."}
                    all_variants[key]['count'] += 1
        except Exception:
            continue

    with open(summary, 'w') as f:
        f.write("##fileformat=VCFv4.2\n")
        f.write("##INFO=<ID=COUNT,Number=1,Type=Integer,Description=\"Barcode count\">\n")
        f.write("##INFO=<ID=FREQ,Number=1,Type=Float,Description=\"Barcode frequency\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for (c, p, r, a), d in sorted(all_variants.items(), key=lambda x: (x[0][0], int(x[0][1]))):
            freq = d['count'] / bc_count if bc_count else 0
            info = d['info']
            extra = f"COUNT={d['count']};FREQ={freq:.4f}"
            info = f"{info};{extra}" if info != "." else extra
            f.write(f"{c}\t{p}\t.\t{r}\t{a}\t.\tPASS\t{info}\n")

    run_command(f"{_TOOL_PATHS['bgzip']} -f {summary}")
    run_command(f"{_TOOL_PATHS['bcftools']} index -t {summary_gz}")
    print(f"Summary: {len(all_variants)} unique variants across {bc_count} barcodes")


def index_bams_in_directory(directory: str) -> List[Dict]:
    bam_files = glob.glob(os.path.join(directory, '*.bam'))
    results = []
    for bam in tqdm(bam_files, desc="Indexing BAMs"):
        ok = index_bam_file(bam)
        results.append({'input_bam': bam, 'status': 'completed' if ok else 'failed'})
    ok_n = sum(1 for r in results if r['status'] == 'completed')
    print(f"Indexed: {ok_n}/{len(results)}")
    return results


# ===========================================================================
# SNVMatrixGenerator — orchestrator class
# ===========================================================================
class SNVMatrixGenerator:
    """Collect SNV pools and filter per-spot BAMs accordingly."""

    def __init__(self, cfg: PipelineConfig, classifier: str = "neural_network",
                 min_af_threshold: float = 0.2):
        self.cfg = cfg
        self.classifier = classifier
        self.min_af_threshold = min_af_threshold
        self.setup_paths()

    def setup_paths(self):
        base = self.cfg.output_dir

        # Input VCFs
        self.beagle_vcf = os.path.join(base, "output_VCFs", "beagle",
                                        "all_filtered_in.vcf.gz")
        classifier_results = os.path.join(base, "output_VCFs", "Classifier", "results")
        self.classifier_homo_vcf = os.path.join(classifier_results,
                                                 f"{self.classifier}_homozygous.vcf.gz")
        self.classifier_hetero_vcf = os.path.join(classifier_results,
                                                   f"{self.classifier}_heterozygous.vcf.gz")

        # BAM directory
        self.bam_dir = os.path.join(self.cfg.bam_base_path, self.cfg.bam_pattern)

        # Output directories
        self.filtered_bam_dir = os.path.join(base, "output_VCFs", "BAM_filtered")
        self.log_dir = os.path.join(base, "logs", "BAM_filtered")
        os.makedirs(self.filtered_bam_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    # -------------------------------------------------------- genotype check
    @staticmethod
    def _is_valid_genotype(vcf_line: str) -> bool:
        try:
            f = vcf_line.strip().split('\t')
            fmt = f[8].split(':')
            samp = f[9].split(':')
            gt = samp[fmt.index('GT')]
            return gt in ('0/1', '1/1')
        except (ValueError, IndexError):
            return False

    def count_genotypes(self, vcf_path: str) -> Tuple[int, int]:
        n01 = n11 = 0
        if not os.path.exists(vcf_path):
            return n01, n11
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if self._is_valid_genotype(line):
                    flds = line.strip().split('\t')
                    fmt = flds[8].split(':')
                    samp = flds[9].split(':')
                    gt = samp[fmt.index('GT')]
                    if gt == '0/1':
                        n01 += 1
                    elif gt == '1/1':
                        n11 += 1
        return n01, n11

    # -------------------------------------------------------- collect SNVs
    def collect_snvs(self) -> Set[SNVInfo]:
        """Collect SNVs from Beagle + Classifier outputs."""
        combined = set()

        sources = {"Beagle": self.beagle_vcf}
        # Classifier VCFs only exist for germline pipeline
        if self.cfg.is_germline:
            sources[f"{self.classifier}_Hom"] = self.classifier_homo_vcf
            sources[f"{self.classifier}_Het"] = self.classifier_hetero_vcf

        for label, path in sources.items():
            if not os.path.exists(path):
                print(f"Warning: {label} VCF not found: {path}")
                continue

            race = "defined" if label == "Beagle" else "denovo"
            n01, n11 = self.count_genotypes(path)
            print(f"{label}: 0/1={n01}, 1/1={n11}, race={race}")

            with gzip.open(path, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    if self._is_valid_genotype(line):
                        snv = SNVInfo.from_vcf_line(line)
                        snv.race = race
                        combined.add(snv)

        print(f"Total unique SNVs: {len(combined)}")
        return combined

    # -------------------------------------------------------- filter
    def filter_bams(self, max_workers=30):
        snvs = self.collect_snvs()
        print(f"Filtering BAMs with {len(snvs)} SNVs...")

        search = self.bam_dir if '*' in self.bam_dir else os.path.join(self.bam_dir, '*.bam')
        bam_files = glob.glob(search)
        if not bam_files:
            raise FileNotFoundError(f"No BAMs at: {search}")
        print(f"Found {len(bam_files)} BAM files")

        results = filter_bams_parallel(bam_files, self.filtered_bam_dir, snvs, max_workers)

        completed = sum(1 for r in results if r['status'] == 'completed')
        with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs'))
        failed = sum(1 for r in results if r['status'] == 'failed')

        print(f"\nSummary: {completed} ok, {with_snvs} with SNVs, {failed} failed")
        return results

    def index_existing_bams(self):
        return index_bams_in_directory(self.filtered_bam_dir)


# ===========================================================================
# CLI
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 5: Filter BAMs by SNV Pools")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--section-id")
    parser.add_argument("--max-workers", type=int, default=30)
    parser.add_argument("--classifier", default="neural_network",
                        choices=["svm", "random_forest", "xgboost", "neural_network"])
    parser.add_argument("--index-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.section_id)
    _init_tools(cfg)

    print(f"Dataset:  {cfg.dataset_name}")
    print(f"Section:  {cfg.section_id}")
    print(f"Pipeline: {'germline' if cfg.is_germline else 'somatic'}")

    gen = SNVMatrixGenerator(cfg, classifier=args.classifier)

    if args.index_only:
        gen.index_existing_bams()
    else:
        results = gen.filter_bams(max_workers=args.max_workers)
        if any(r['status'] == 'failed' for r in results):
            exit(1)


if __name__ == "__main__":
    main()