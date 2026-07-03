#!/usr/bin/env python3
"""
Run Strelka2 for P4 and P6 tumor experiments with spatial transcriptomics data.

This script supports:
1. WES somatic variant calling using normal and tumor BAMs
2. Visium spatial transcriptomics variant calling with two options:
   a) Overall calling without barcode splitting
   b) Per-barcode calling with cell splitting (default)
"""

import os
import sys
import argparse
import subprocess
import glob
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import logging
import shutil
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger('run_strelka_spatial_p4p6')

# Define dataset directories
P4_DIRS = {
    "normal_wes": "/data/maiziezhou_lab/yuqi/snv_calling/data/P4_Normal_WES/P4_Normal_WES.bam",
    "tumor_wes": "/data/maiziezhou_lab/yuqi/snv_calling/data/P4_cSCC_WES/P4_cSCC_WES.bam",
    "visium_rep1": "/data/P4_Visium/spaceranger_align_rep1/P4_Tumor_output/outs/possorted_genome_bam.bam",
    "visium_rep2": "/data/P4_Visium/spaceranger_align_rep2/P4_Tumor_output/outs/possorted_genome_bam.bam"
}

P6_DIRS = {
    "normal_wes": "/data/P6_Normal_WES/P6_Normal_WES.bam",
    "tumor_wes": "/data/P6_cSCC_WES/P6_cSCC_WES.bam",
    "visium_rep1": "/data/P6_Visium/spaceranger_align_rep1/P6_Tumor_output/outs/possorted_genome_bam.bam",
    "visium_rep2": "/data/P6_Visium/spaceranger_align_rep2/P6_Tumor_output/outs/possorted_genome_bam.bam"
}

# Reference genome
REF_GENOME = "/data/maiziezhou_lab/shared/genomes/GRCh38/refdata-gex-GRCh38-2020-A/fasta/genome.fa"

# Strelka installation directory
STRELKA_DIR = "/data/maiziezhou_lab/yuqi/software/miniconda3/envs/snv_caller_new/share/strelka-2.9.10-0"


def check_bam_index(bam_file: str) -> bool:
    """
    Check if BAM index exists, create it if it doesn't.
    
    Args:
        bam_file: Path to BAM file
        
    Returns:
        True if successful, False otherwise
    """
    bai_file = f"{bam_file}.bai"
    if not os.path.exists(bai_file):
        logger.info(f"Creating index for {bam_file}")
        try:
            subprocess.run(["samtools", "index", bam_file], check=True)
            return os.path.exists(bai_file)
        except subprocess.CalledProcessError:
            logger.error(f"Failed to create index for {bam_file}")
            return False
    return True


def run_strelka_wes(normal_bam: str, tumor_bam: str, output_dir: str, ref_genome: str = REF_GENOME) -> bool:
    """
    Run Strelka2 for WES tumor/normal pair.
    
    Args:
        normal_bam: Path to normal BAM file
        tumor_bam: Path to tumor BAM file
        output_dir: Output directory for Strelka results
        ref_genome: Path to reference genome
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running Strelka2 for WES tumor/normal pair:\n  Normal: {normal_bam}\n  Tumor: {tumor_bam}")
    
    # Check BAM indexes
    if not check_bam_index(normal_bam) or not check_bam_index(tumor_bam):
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure Strelka
    configure_cmd = [
        f"{STRELKA_DIR}/bin/configureStrelkaSomaticWorkflow.py",
        f"--normalBam={normal_bam}",
        f"--tumorBam={tumor_bam}",
        f"--referenceFasta={ref_genome}",
        f"--runDir={output_dir}",
        "--exome"
    ]
    
    try:
        logger.info("Configuring Strelka2...")
        subprocess.run(configure_cmd, check=True)
        
        # Run Strelka
        run_cmd = [
            f"{output_dir}/runWorkflow.py",
            "-m", "local",
            "-j", "8"  # Use 8 parallel jobs
        ]
        
        logger.info("Running Strelka2...")
        subprocess.run(run_cmd, check=True)
        logger.info(f"Strelka2 WES analysis complete. Results in {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Strelka2 execution failed: {e}")
        return False


def run_strelka_visium_overall(normal_bam: str, visium_bam: str, output_dir: str, ref_genome: str = REF_GENOME) -> bool:
    """
    Run Strelka2 for overall Visium BAM without splitting by barcode.
    
    Args:
        normal_bam: Path to normal BAM file
        visium_bam: Path to Visium BAM file
        output_dir: Output directory for Strelka results
        ref_genome: Path to reference genome
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Running Strelka2 for overall Visium BAM:\n  Normal: {normal_bam}\n  Visium: {visium_bam}")
    
    # Check BAM indexes
    if not check_bam_index(normal_bam) or not check_bam_index(visium_bam):
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure Strelka
    configure_cmd = [
        f"{STRELKA_DIR}/bin/configureStrelkaSomaticWorkflow.py",
        f"--normalBam={normal_bam}",
        f"--tumorBam={visium_bam}",
        f"--referenceFasta={ref_genome}",
        f"--runDir={output_dir}",
        "--targeted"  # Use targeted mode for RNA-seq
    ]
    
    try:
        logger.info("Configuring Strelka2...")
        subprocess.run(configure_cmd, check=True)
        
        # Run Strelka
        run_cmd = [
            f"{output_dir}/runWorkflow.py",
            "-m", "local",
            "-j", "8"  # Use 8 parallel jobs
        ]
        
        logger.info("Running Strelka2...")
        subprocess.run(run_cmd, check=True)
        logger.info(f"Strelka2 Visium overall analysis complete. Results in {output_dir}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Strelka2 execution failed: {e}")
        return False


def extract_barcodes_from_bam(bam_file: str, output_file: str, min_reads: int = 100) -> List[str]:
    """
    Extract cell barcodes from BAM file and filter by minimum read count.
    
    Args:
        bam_file: Path to BAM file
        output_file: Output file to save barcodes
        min_reads: Minimum number of reads to keep a barcode
        
    Returns:
        List of filtered barcodes
    """
    logger.info(f"Extracting barcodes from {bam_file} (minimum {min_reads} reads)")
    
    # Extract barcodes using samtools
    cmd = [
        "samtools", "view", bam_file, "|",
        "grep", "-o", "CB:Z:[ACGT]\+-[0-9]\+", "|",
        "sed", "s/CB:Z://", "|",
        "sort", "|",
        "uniq", "-c", "|",
        "sort", "-nr", ">", output_file
    ]
    
    cmd_str = " ".join(cmd)
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        
        # Parse and filter barcodes
        barcodes = []
        with open(output_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    count = int(parts[0])
                    barcode = parts[1]
                    if count >= min_reads:
                        barcodes.append(barcode)
        
        logger.info(f"Found {len(barcodes)} barcodes with ≥ {min_reads} reads")
        return barcodes
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to extract barcodes: {e}")
        return []


def split_bam_by_barcode(bam_file: str, barcodes: List[str], output_dir: str) -> Dict[str, str]:
    """
    Split BAM file by cell barcode.
    
    Args:
        bam_file: Path to BAM file
        barcodes: List of cell barcodes
        output_dir: Directory to save split BAM files
        
    Returns:
        Dictionary mapping barcodes to output BAM paths
    """
    logger.info(f"Splitting {bam_file} by {len(barcodes)} barcodes")
    os.makedirs(output_dir, exist_ok=True)
    
    barcode_bams = {}
    
    for barcode in barcodes:
        output_bam = os.path.join(output_dir, f"{barcode}.bam")
        barcode_bams[barcode] = output_bam
        
        # Extract reads for specific barcode
        cmd = [
            "samtools", "view", "-h", bam_file, "|",
            "awk", f"'$0 ~ /^@/ || $0 ~ /CB:Z:{barcode}/'", "|",
            "samtools", "view", "-b", "-o", output_bam
        ]
        
        cmd_str = " ".join(cmd)
        try:
            logger.info(f"Extracting reads for barcode: {barcode}")
            subprocess.run(cmd_str, shell=True, check=True)
            check_bam_index(output_bam)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to split BAM for barcode {barcode}: {e}")
    
    return barcode_bams


def run_strelka_per_barcode_parallel(normal_bam: str, barcode_bams: Dict[str, str], 
                                    output_base_dir: str, ref_genome: str = REF_GENOME,
                                    max_workers: int = 4) -> Dict[str, bool]:
    """
    Run Strelka2 for each barcode-specific BAM against the normal BAM in parallel.
    
    Args:
        normal_bam: Path to normal BAM file
        barcode_bams: Dictionary mapping barcodes to BAM paths
        output_base_dir: Base directory for Strelka outputs
        ref_genome: Path to reference genome
        max_workers: Maximum number of parallel jobs
        
    Returns:
        Dictionary mapping barcodes to success status
    """
    results = {}
    
    logger.info(f"Processing {len(barcode_bams)} barcodes using {max_workers} parallel workers")
    
    # Process barcodes in parallel batches
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_barcode = {
            executor.submit(
                run_strelka_visium_overall, 
                normal_bam, 
                tumor_bam, 
                os.path.join(output_base_dir, barcode),
                ref_genome
            ): barcode 
            for barcode, tumor_bam in barcode_bams.items()
        }
        
        # Process results as they complete
        with tqdm(total=len(barcode_bams), desc="Processing barcodes") as pbar:
            for future in as_completed(future_to_barcode):
                barcode = future_to_barcode[future]
                try:
                    success = future.result()
                    results[barcode] = success
                    status = "completed" if success else "failed"
                    logger.info(f"Barcode {barcode} processing {status}")
                except Exception as e:
                    results[barcode] = False
                    logger.error(f"Error processing barcode {barcode}: {str(e)}")
                finally:
                    pbar.update(1)
    
    # Summarize results
    successes = sum(1 for success in results.values() if success)
    failures = len(results) - successes
    logger.info(f"Completed processing {len(results)} barcodes: {successes} succeeded, {failures} failed")
    
    return results


def compare_mpileup_strelka(project_id: str, barcode: str, output_dir: str):
    """
    Compare mpileup and Strelka variant calls for a given barcode.
    
    Args:
        project_id: Project ID (P4 or P6)
        barcode: Cell barcode
        output_dir: Output directory for comparison results
    """
    from scripts.tools.compare_mpileup_single_n_strelka import plot_venn_diagram, parse_vcf_with_genotypes
    
    logger.info(f"Comparing mpileup and Strelka for {project_id} barcode {barcode}")
    
    # Define paths
    base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling/data/{project_id}"
    mpileup_vcf = os.path.join(base_dir, f"output_VCFs/mpileup_single_bam/baseQ0mapQ0/{barcode}.vcf.gz")
    strelka_vcf = os.path.join(base_dir, f"output_VCFs/strelka/strelkaQ0/{barcode}/results/variants/variants.vcf.gz")
    
    # Check if files exist
    if not os.path.exists(mpileup_vcf):
        logger.error(f"Mpileup VCF not found: {mpileup_vcf}")
        return
        
    if not os.path.exists(strelka_vcf):
        logger.error(f"Strelka VCF not found: {strelka_vcf}")
        return
    
    # Parse VCFs with genotypes
    mpileup_genotypes = parse_vcf_with_genotypes(mpileup_vcf)
    strelka_genotypes = parse_vcf_with_genotypes(strelka_vcf)
    
    # Create output directory
    barcode_output_dir = os.path.join(output_dir, barcode)
    os.makedirs(barcode_output_dir, exist_ok=True)
    
    # Plot Venn diagrams for 0/1, 1/1, and both genotypes
    for gt in ['0/1', '1/1', 'both']:
        if gt == 'both':
            mpileup_set = {k for k, v in mpileup_genotypes.items() if v in {'0/1', '1/1'}}
            strelka_set = {k for k, v in strelka_genotypes.items() if v in {'0/1', '1/1'}}
            title = f"Variant Call Comparison for {barcode} (0/1 and 1/1)"
            output_path = os.path.join(barcode_output_dir, f"{barcode}_venn_both.png")
        else:
            mpileup_set = {k for k, v in mpileup_genotypes.items() if v == gt}
            strelka_set = {k for k, v in strelka_genotypes.items() if v == gt}
            title = f"Variant Call Comparison for {barcode} ({gt})"
            output_path = os.path.join(barcode_output_dir, f"{barcode}_venn_{gt.replace('/', '')}.png")
        
        plot_venn_diagram(mpileup_set, strelka_set, title, output_path)


def compare_mpileup_strelka_parallel(project_id: str, barcodes: List[str], output_dir: str, max_workers: int = 4):
    """
    Compare mpileup and Strelka variant calls for multiple barcodes in parallel.
    
    Args:
        project_id: Project ID (P4 or P6)
        barcodes: List of cell barcodes
        output_dir: Output directory for comparison results
        max_workers: Maximum number of parallel jobs
    """
    from scripts.tools.compare_mpileup_single_n_strelka import plot_venn_diagram, parse_vcf_with_genotypes
    
    logger.info(f"Comparing mpileup and Strelka for {len(barcodes)} barcodes in {project_id}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process barcodes in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        futures = [
            executor.submit(compare_mpileup_strelka, project_id, barcode, output_dir)
            for barcode in barcodes
        ]
        
        # Process results as they complete
        with tqdm(total=len(barcodes), desc="Comparing variants") as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error comparing variants: {str(e)}")
                finally:
                    pbar.update(1)


def main():
    parser = argparse.ArgumentParser(description='Run Strelka2 for P4 and P6 tumor experiments')
    parser.add_argument('--project', choices=['P4', 'P6'], required=True, help='Project ID (P4 or P6)')
    parser.add_argument('--output-dir', required=True, help='Base output directory')
    parser.add_argument('--mode', choices=['wes', 'visium-overall', 'visium-split'], default='visium-split',
                        help='Analysis mode (default: visium-split)')
    parser.add_argument('--rep', choices=['1', '2'], default='1', help='Visium replicate (1 or 2, default: 1)')
    parser.add_argument('--min-reads', type=int, default=500, help='Minimum reads per barcode (default: 500)')
    parser.add_argument('--compare', action='store_true', help='Compare with mpileup results')
    parser.add_argument('--barcodes', nargs='+', help='Specific barcodes to process (optional)')
    parser.add_argument('--parallel', type=int, default=4, help='Number of parallel jobs (default: 4)')
    
    args = parser.parse_args()
    
    # Get project directories
    dirs = P4_DIRS if args.project == 'P4' else P6_DIRS
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run analysis based on mode
    if args.mode == 'wes':
        output_dir = os.path.join(args.output_dir, f"{args.project}_WES")
        run_strelka_wes(dirs['normal_wes'], dirs['tumor_wes'], output_dir)
    
    elif args.mode == 'visium-overall':
        # Run analysis on overall Visium BAM
        visium_bam = dirs[f'visium_rep{args.rep}']
        output_dir = os.path.join(args.output_dir, f"{args.project}_Visium_rep{args.rep}_overall")
        run_strelka_visium_overall(dirs['normal_wes'], visium_bam, output_dir)
    
    elif args.mode == 'visium-split':
        # Split Visium BAM by barcode and run analysis
        visium_bam = dirs[f'visium_rep{args.rep}']
        base_output_dir = os.path.join(args.output_dir, f"{args.project}_Visium_rep{args.rep}_split")
        os.makedirs(base_output_dir, exist_ok=True)
        
        # Get barcodes
        if args.barcodes:
            barcodes = args.barcodes
            logger.info(f"Using {len(barcodes)} provided barcodes")
        else:
            barcodes_file = os.path.join(base_output_dir, "barcodes.txt")
            barcodes = extract_barcodes_from_bam(visium_bam, barcodes_file, args.min_reads)
        
        if not barcodes:
            logger.error("No barcodes found or provided. Exiting.")
            sys.exit(1)
        
        # Split BAM by barcode
        bam_split_dir = os.path.join(base_output_dir, "split_bams")
        barcode_bams = split_bam_by_barcode(visium_bam, barcodes, bam_split_dir)
        
        # Run Strelka for each barcode in parallel
        strelka_output_dir = os.path.join(base_output_dir, "strelka_results")
        run_strelka_per_barcode_parallel(dirs['normal_wes'], barcode_bams, strelka_output_dir, 
                                        REF_GENOME, args.parallel)
        
        # Compare with mpileup results if requested, also in parallel
        if args.compare:
            comparison_dir = os.path.join(base_output_dir, "comparisons")
            os.makedirs(comparison_dir, exist_ok=True)
            compare_mpileup_strelka_parallel(args.project, barcodes, comparison_dir, args.parallel)
    
    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()

# usage:
# python scripts/strelka/run_strelka_spatial_p4p6.py --project P4 --output-dir ./data --mode visium-split --rep 1 --min-reads 5 --compare --parallel 5