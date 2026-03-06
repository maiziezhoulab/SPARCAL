"""
SPARCAL Step 2a: Beagle Genotype Phasing Pipeline
==================================================
Runs Beagle5.4 per chromosome using 1000 Genomes reference panels to
impute/phase genotypes from the mpileup VCF output. Produces:
  - Per-chromosome phased VCFs
  - all_filtered_in.vcf.gz  (variants present in 1kG → "defined")
  - all_filtered_out.vcf.gz (variants absent from 1kG → "denovo")

Usage:
    python run_beagle.py --dataset DLPFC --section_id 151507
    python run_beagle.py --dataset P4_TUMOR --section_id 1 --threads 24
    python run_beagle.py --dataset DCIS --section_id 1
"""

import os
import sys
import time
import gzip
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "config"))
from config_loader import load_config, PipelineConfig


# ---------------------------------------------------------------------------
# Beagle-specific defaults (non-path, non-dataset)
# ---------------------------------------------------------------------------
BEAGLE_DEFAULTS = {
    "THREADS": 24,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 0,
    "IMPUTE": False,
    "GPROBS": True,
}


class BeaglePipeline:
    """Runs Beagle genotype phasing per chromosome against 1kG panels."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.setup_paths()
        self.setup_environment()

    # ------------------------------------------------------------------ setup
    def setup_paths(self):
        """Create output directory structure."""
        base = self.cfg.output_dir
        self.output_dirs = {
            "input_vcf_dir":  os.path.join(base, "output_VCFs", "mpileup_multi_bam"),
            "output_vcf_dir": os.path.join(base, "output_VCFs", "beagle"),
            "log_dir":        os.path.join(base, "logs", "beagle"),
            "metrics_dir":    os.path.join(base, "metrics", "beagle"),
        }
        for d in self.output_dirs.values():
            os.makedirs(d, exist_ok=True)

    def setup_environment(self):
        """Prepend apps_dir to PATH / LD_LIBRARY_PATH."""
        apps = self.cfg.apps_dir
        os.environ['PATH'] = f"{apps}:{os.environ.get('PATH', '')}"
        ld = os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = f"{apps}:{ld}" if ld else apps

    # --------------------------------------------------------- 1kG reference
    def get_1000genome_reference(self, chromosome: str) -> str:
        """
        Resolve the 1000 Genomes reference VCF for a chromosome.

        For hg19 builds the {chrom} placeholder receives the bare number
        (e.g. "1"); for GRCh38 it keeps the original format from cfg.regions.
        """
        build = self.cfg.raw.get("reference", {}).get("build", "GRCh38")
        base = self.cfg.thousand_genome_base
        pattern = self.cfg.thousand_genome_pattern

        if build == "hg19":
            chrom = chromosome.replace("chr", "")
        else:
            chrom = chromosome

        ref_path = os.path.join(base, pattern.format(chrom=chrom))
        if not os.path.exists(ref_path):
            raise FileNotFoundError(
                f"1kG reference not found: {ref_path}\n"
                f"  build={build}  chromosome={chromosome}"
            )
        return ref_path

    # --------------------------------------------------------- VCF merge
    def merge_vcf_fields(self, original_vcf: str, beagle_vcf: str,
                         output_vcf: str):
        """Annotate Beagle output with FORMAT/INFO fields from the original."""
        bcftools = self.cfg.tool("bcftools")
        tabix = self.cfg.tool("tabix")

        for vcf in [original_vcf, beagle_vcf]:
            if not os.path.exists(vcf + '.tbi'):
                subprocess.run([tabix, '-p', 'vcf', vcf], check=True)

        cmd = [
            bcftools, 'annotate',
            '-a', original_vcf,
            '-c', ('INFO/DP,INFO/I16,INFO/QS,INFO/SGB,INFO/RPB,INFO/MQB,'
                   'INFO/MQSB,INFO/BQB,INFO/MQ0F,FORMAT/GQ,FORMAT/BAF,FORMAT/PL'),
            '-O', 'z',
            '-o', output_vcf,
            beagle_vcf,
        ]
        subprocess.run(cmd, check=True)
        subprocess.run([tabix, '-p', 'vcf', output_vcf], check=True)

    # --------------------------------------------------------- Beagle command
    def run_beagle_command(self, input_vcf: str, output_prefix: str,
                          chromosome: str, params: Dict, log_file: str):
        """Invoke Beagle JAR, then merge original fields back."""
        reference_panel = self.get_1000genome_reference(chromosome)
        beagle_output = f"{output_prefix}.temp.vcf.gz"

        java = os.path.join(self.cfg.apps_dir, self.cfg.java_path)
        jar = os.path.join(self.cfg.apps_dir, self.cfg.beagle_jar)

        cmd = [
            java,
            f"-Xmx{params['MEMORY']}",
            "-jar", jar,
            f"gl={input_vcf}",
            f"ref={reference_panel}",
            f"chrom={chromosome}",
            f"out={output_prefix}.temp",
            f"impute={'true' if params['IMPUTE'] else 'false'}",
            f"modelscale={params['MODEL_SCALE']}",
            f"nthreads={params['THREADS']}",
            f"gprobs={'true' if params['GPROBS'] else 'false'}",
            f"niterations={params['ITERATIONS']}",
        ]

        with open(log_file, 'w') as log:
            log.write(f"Command: {' '.join(cmd)}\n\n")
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)

        final_output = f"{output_prefix}.vcf.gz"
        self.merge_vcf_fields(input_vcf, beagle_output, final_output)

        # Cleanup temp
        if os.path.exists(beagle_output):
            os.remove(beagle_output)
        if os.path.exists(f"{beagle_output}.tbi"):
            os.remove(f"{beagle_output}.tbi")

    # ------------------------------------------------------- per-chromosome
    def process_chromosome(self, chromosome: str, params: Dict) -> Dict:
        """Process a single chromosome with Beagle."""
        start_time = time.time()
        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"],
                                 "merged_sorted_gt.vcf.gz")
        output_prefix = os.path.join(self.output_dirs["output_vcf_dir"],
                                     chromosome)
        log_file = os.path.join(self.output_dirs["log_dir"],
                                f"{chromosome}.log")

        try:
            self.run_beagle_command(input_vcf, output_prefix, chromosome,
                                   params, log_file)
            duration = time.time() - start_time
            beagle_out = f"{output_prefix}.vcf.gz"
            if not os.path.exists(beagle_out):
                raise FileNotFoundError(f"Expected output not found: {beagle_out}")
            return {"chromosome": chromosome, "duration": duration,
                    "status": "completed", "output_file": beagle_out}
        except Exception as e:
            return {"chromosome": chromosome, "duration": time.time() - start_time,
                    "status": "failed", "error": str(e)}

    # ------------------------------------------------------- variant sets
    def collect_passed_variants(self, beagle_vcf: str) -> Set[str]:
        """Collect CHROM_POS_REF_ALT keys from a Beagle output VCF."""
        passed = set()
        with gzip.open(beagle_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                passed.add(f"{fields[0]}_{fields[1]}_{fields[3]}_{fields[4]}")
        return passed

    def create_filtered_vcf(self, input_vcf: str, passed_variants: Set[str],
                            output_in: str, output_out: str):
        """
        Split original VCF into:
          - output_in:  variants present in 1kG (Beagle-genotyped → "defined")
          - output_out: variants absent from 1kG (→ "denovo")
        """
        bgzip = self.cfg.tool("bgzip")
        tabix = self.cfg.tool("tabix")

        temp_in = output_in.replace('.gz', '')
        temp_out = output_out.replace('.gz', '')

        with gzip.open(input_vcf, 'rt') as f_in, \
             open(temp_in, 'w') as f_pass, \
             open(temp_out, 'w') as f_fail:

            for line in f_in:
                if line.startswith('#'):
                    f_pass.write(line)
                    f_fail.write(line)
                    if line.startswith('#CHROM'):
                        break

            for line in f_in:
                fields = line.strip().split('\t')
                key = f"{fields[0]}_{fields[1]}_{fields[3]}_{fields[4]}"
                if key in passed_variants:
                    f_pass.write(line)
                else:
                    f_fail.write(line)

        for tmp, out in [(temp_in, output_in), (temp_out, output_out)]:
            subprocess.run([bgzip, '-f', tmp], check=True)
            subprocess.run([tabix, '-p', 'vcf', out], check=True)
            if os.path.exists(tmp):
                os.remove(tmp)

    # ------------------------------------------------------- main pipeline
    def run_pipeline(self, custom_params: Optional[Dict] = None) -> pd.DataFrame:
        """Run Beagle across all chromosomes, then split defined/denovo."""
        params = BEAGLE_DEFAULTS.copy()
        # Override with YAML beagle params
        beagle_yaml = self.cfg.raw.get("beagle", {})
        for k in ["threads", "memory", "model_scale", "iterations", "impute", "gprobs"]:
            if k in beagle_yaml:
                params[k.upper()] = beagle_yaml[k]
        if custom_params:
            params.update(custom_params)

        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"],
                                 "merged_sorted_gt.vcf.gz")
        if not os.path.exists(input_vcf):
            raise FileNotFoundError(f"Input VCF not found: {input_vcf}")

        regions = self.cfg.regions

        # Parallel chromosome processing
        results = []
        n_workers = min(params["THREADS"], len(regions))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self.process_chromosome, chrom, params): chrom
                for chrom in regions
            }
            with tqdm(total=len(regions), desc="Processing chromosomes") as pbar:
                for future in as_completed(futures):
                    chrom = futures[future]
                    try:
                        results.append(future.result())
                    except Exception as e:
                        results.append({"chromosome": chrom, "status": "failed",
                                        "error": str(e)})
                    pbar.update(1)

        metrics_df = pd.DataFrame(results)

        # If all chromosomes succeeded, create defined/denovo split
        n_ok = len(metrics_df[metrics_df['status'] == 'completed'])
        if n_ok == len(regions):
            print("\nCollecting passed variants across all chromosomes...")
            all_passed = set()
            for _, row in metrics_df[metrics_df['status'] == 'completed'].iterrows():
                all_passed.update(self.collect_passed_variants(row['output_file']))
            print(f"Total variants in 1kG (defined): {len(all_passed)}")

            out_in = os.path.join(self.output_dirs["output_vcf_dir"],
                                  "all_filtered_in.vcf.gz")
            out_out = os.path.join(self.output_dirs["output_vcf_dir"],
                                   "all_filtered_out.vcf.gz")
            self.create_filtered_vcf(input_vcf, all_passed, out_in, out_out)
            print(f"Defined VCF:  {out_in}")
            print(f"De novo VCF:  {out_out}")
        else:
            print(f"\nWarning: {len(regions) - n_ok} chromosomes failed — "
                  "skipping defined/denovo split")

        # Save metrics
        sid = f"_{self.cfg.section_id}" if self.cfg.section_id else ""
        metrics_file = os.path.join(
            self.output_dirs["metrics_dir"],
            f"{self.cfg.dataset_name}{sid}_beagle_metrics.csv"
        )
        metrics_df.to_csv(metrics_file, index=False)
        return metrics_df


# ========================== CLI =============================================

def main():
    parser = argparse.ArgumentParser(
        description="SPARCAL Step 2a: Beagle Genotype Phasing Pipeline")
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (must match a YAML config)")
    parser.add_argument("--section_id",
                        help="Section ID (required for multi-section datasets)")
    parser.add_argument("--threads", type=int, default=None,
                        help="Override number of Beagle threads")
    parser.add_argument("--memory", default=None,
                        help="Override Beagle memory (e.g. '20g')")
    args = parser.parse_args()

    cfg = load_config(args.dataset, args.section_id)
    print(f"Dataset:    {cfg.dataset_name}")
    print(f"Section:    {cfg.section_id}")
    print(f"Reference:  {cfg.raw.get('reference', {}).get('build', '?')}")
    print(f"Regions:    {cfg.regions[0]}..{cfg.regions[-1]}")
    print(f"1kG base:   {cfg.thousand_genome_base}")

    pipeline = BeaglePipeline(cfg)

    custom_params = {}
    if args.threads is not None:
        custom_params["THREADS"] = args.threads
    if args.memory is not None:
        custom_params["MEMORY"] = args.memory

    metrics = pipeline.run_pipeline(custom_params)

    # Summary
    print("\n--- Beagle Pipeline Summary ---")
    n_ok = len(metrics[metrics['status'] == 'completed'])
    n_fail = len(metrics[metrics['status'] == 'failed'])
    print(f"Chromosomes: {n_ok} completed, {n_fail} failed")
    if n_fail > 0:
        for _, row in metrics[metrics['status'] == 'failed'].iterrows():
            print(f"  FAILED {row['chromosome']}: {row.get('error', '?')}")
    if 'duration' in metrics.columns:
        print(f"Avg time/chrom: {metrics['duration'].mean():.2f}s")


if __name__ == "__main__":
    main()