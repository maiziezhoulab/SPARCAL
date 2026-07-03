#!/usr/bin/env python3

"""
VCF Germline Overlap Analysis Script

This script analyzes the overlap between VCF files from SNV calling pipeline
and the gold standard GATK normal exome results.
"""

import os
import subprocess
import pandas as pd
import glob
from pathlib import Path
import logging
import argparse
from collections import defaultdict
import csv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_CONFIG = {
    "base_dir": "/data/maiziezhou_lab/yuqi/snv_calling",
    "gold_standard": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_exome_snps.vcf.gz",
    "output_dir": "/data/maiziezhou_lab/yuqi/snv_calling/germline_overlap_analysis",
    "quality_filter": "baseQ0mapQ0"
}

# Datasets to process
DATASETS = [
    "DLPFC/151507",
    "P4_tumor/1",
    "P6_tumor/1"
]

def run_command(cmd, check=True):
    """Run a shell command and return output"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=check, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {cmd}")
        logger.error(f"Error: {e}")
        logger.error(f"STDERR: {e.stderr}")
        if check:
            raise
        return None

def prepare_vcf(vcf_path):
    """Check and prepare VCF file (ensure it's compressed and indexed)"""
    if not os.path.exists(vcf_path):
        logger.warning(f"File not found: {vcf_path}")
        return None
    
    # Always uncompress the original file, and use bgzip to compress the file
    # uncompress the file
    if vcf_path.endswith('.gz'):
        logger.info(f"Uncompressing {vcf_path}")
        run_command(f"gunzip -f {vcf_path}")
        vcf_path = vcf_path.replace('.gz', '')
    
    # compress the file
    logger.info(f"Compressing {vcf_path}")
    run_command(f"bgzip -f {vcf_path}")
    vcf_path = f"{vcf_path}.gz"

    # if not vcf_path.endswith('.gz'):
    #     logger.info(f"Compressing {vcf_path}")
    #     run_command(f"bgzip -f {vcf_path}")
    #     vcf_path = f"{vcf_path}.gz"
    
    # Check if index exists
    if not os.path.exists(f"{vcf_path}.tbi"):
        logger.info(f"Indexing {vcf_path}")
        run_command(f"tabix -p vcf {vcf_path}")
    
    return vcf_path

def analyze_overlap(vcf_path, gold_standard, output_dir):
    """Run bcftools isec and get overlap count"""
    vcf_name = os.path.basename(vcf_path).replace('.vcf.gz', '')
    output_subdir = os.path.join(output_dir, f"{vcf_name}_overlap")
    
    logger.info(f"Analyzing overlap for {vcf_name}")
    
    # Run bcftools isec
    cmd = f"bcftools isec -n=2 -w1 -O z -p {output_subdir} {vcf_path} {gold_standard}"
    result = run_command(cmd, check=False)
    
    if result is None or result.returncode != 0:
        logger.error(f"Failed to analyze overlap for {vcf_name}")
        return vcf_name, 0
    
    # Get overlap count
    sites_file = os.path.join(output_subdir, "sites.txt")
    if os.path.exists(sites_file):
        with open(sites_file, 'r') as f:
            lines = f.readlines()
            overlap_count = len(lines)
        logger.info(f"{vcf_name}: {overlap_count} germline variants")
        return vcf_name, overlap_count
    else:
        logger.warning(f"No sites.txt file found for {vcf_name}")
        return vcf_name, 0

def split_svm2_predictions(svm2_pred_path, output_dir):
    """Split SVM2 predictions into high/low confidence files"""
    if not os.path.exists(svm2_pred_path):
        logger.warning(f"SVM2 predictions file not found: {svm2_pred_path}")
        return None, None
    
    # Check and prepare SVM2 predictions file
    prepared_vcf = prepare_vcf(svm2_pred_path)
    if prepared_vcf is None:
        return None, None
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract high confidence variants (SVM2_PRED=1)
    high_conf_path = os.path.join(output_dir, "high_confidence.vcf.gz")
    run_command(f'bcftools view -i \'INFO/SVM2_PRED="1"\' -O z -o {high_conf_path} {prepared_vcf}')
    run_command(f'tabix -p vcf {high_conf_path}')
    
    # Extract low confidence variants (SVM2_PRED=0)
    low_conf_path = os.path.join(output_dir, "low_confidence.vcf.gz")
    run_command(f'bcftools view -i \'INFO/SVM2_PRED="0"\' -O z -o {low_conf_path} {prepared_vcf}')
    run_command(f'tabix -p vcf {low_conf_path}')
    
    return high_conf_path, low_conf_path

def get_vcf_files(dataset_dir, quality_filter):
    """Get dictionary of VCF file paths for the dataset"""
    vcf_files = {
        "mpileup_output": os.path.join(dataset_dir, f"output_VCFs/mpileup_multi_bam/{quality_filter}/merged_sorted_gt.vcf.gz"),
        "beagle_filtered_in": os.path.join(dataset_dir, f"output_VCFs/beagle/{quality_filter}/all_filtered_in.vcf.gz"),
        "beagle_filtered_out": os.path.join(dataset_dir, f"output_VCFs/beagle/{quality_filter}/all_filtered_out.vcf.gz"),
        "seq_error": os.path.join(dataset_dir, f"output_VCFs/SeqErrModel/{quality_filter}/sequence_error.vcf.gz"),
        "seq_no_error": os.path.join(dataset_dir, f"output_VCFs/SeqErrModel/{quality_filter}/sequence_no_error.vcf.gz"),
        "svm1_high_conf": os.path.join(dataset_dir, f"output_VCFs/SVMModel/{quality_filter}/results/high_confidence.vcf.gz"),
        "svm1_low_conf": os.path.join(dataset_dir, f"output_VCFs/SVMModel/{quality_filter}/results/low_confidence.vcf.gz"),
        "svm2_predictions": os.path.join(dataset_dir, f"output_VCFs/SVM2Model/{quality_filter}/svm2_predictions.vcf.gz"),
    }
    
    # SVM2 split confidence files
    svm2_split_dir = os.path.join(dataset_dir, f"output_VCFs/SVM2Model/{quality_filter}/split_confidence")
    vcf_files["svm2_high_conf"] = os.path.join(svm2_split_dir, "high_confidence.vcf.gz")
    vcf_files["svm2_low_conf"] = os.path.join(svm2_split_dir, "low_confidence.vcf.gz")
    
    return vcf_files

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Analyze germline variant overlap with gold standard")
    parser.add_argument("--datasets", nargs="+", default=DATASETS, 
                       help="List of datasets to process (e.g., 'DLPFC/151507' 'P4_tumor/1')")
    parser.add_argument("--quality-filter", default=DEFAULT_CONFIG["quality_filter"],
                       help="Quality filter to use")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG["output_dir"],
                       help="Output directory for results")
    parser.add_argument("--gold-standard", default=DEFAULT_CONFIG["gold_standard"],
                       help="Path to gold standard VCF file")
    parser.add_argument("--base-dir", default=DEFAULT_CONFIG["base_dir"],
                       help="Base directory for SNV calling results")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize results dataframe
    results = []
    
    # Process each dataset
    for dataset in args.datasets:
        logger.info(f"Processing dataset: {dataset}")
        
        # Get dataset directory
        dataset_dir = os.path.join(args.base_dir, "data", dataset)
        
        # Get VCF files for this dataset
        vcf_files = get_vcf_files(dataset_dir, args.quality_filter)
        
        # Split SVM2 predictions if needed
        svm2_split_dir = os.path.join(dataset_dir, f"output_VCFs/SVM2Model/{args.quality_filter}/split_confidence")
        if os.path.exists(vcf_files["svm2_predictions"]) and not (
            os.path.exists(vcf_files["svm2_high_conf"]) and 
            os.path.exists(vcf_files["svm2_low_conf"])
        ):
            logger.info("Splitting SVM2 predictions into high/low confidence")
            split_svm2_predictions(vcf_files["svm2_predictions"], svm2_split_dir)
        
        # Process each VCF file
        dataset_results = []
        for vcf_name, vcf_path in vcf_files.items():
            # Check and prepare VCF file
            prepared_vcf = prepare_vcf(vcf_path)
            if prepared_vcf:
                # Run analysis
                file_name, overlap_count = analyze_overlap(prepared_vcf, args.gold_standard, args.output_dir)
                dataset_results.append({
                    "dataset": dataset,
                    "file_name": vcf_name,
                    "germline_overlap_count": overlap_count
                })
            else:
                logger.warning(f"Skipping {vcf_name}")
                dataset_results.append({
                    "dataset": dataset,
                    "file_name": vcf_name,
                    "germline_overlap_count": 0
                })
        
        # Add dataset results to overall results
        results.extend(dataset_results)
    
    # Create DataFrame and save results
    results_df = pd.DataFrame(results)
    results_csv = os.path.join(args.output_dir, "overlap_counts.csv")
    results_df.to_csv(results_csv, index=False)
    
    # Print summary
    logger.info("\nSummary of Germline Overlap Analysis:")
    logger.info("=======================================")
    
    # Print summary by dataset
    for dataset in args.datasets:
        dataset_df = results_df[results_df["dataset"] == dataset]
        logger.info(f"\nDataset: {dataset}")
        logger.info("-" * 50)
        for _, row in dataset_df.iterrows():
            logger.info(f"{row['file_name']}: {row['germline_overlap_count']} germline variants")
    
    logger.info(f"\nAnalysis complete. Results saved to {results_csv}")

if __name__ == "__main__":
    main()


# check for P4 sec 1
# python scripts/tools/overlap_with_groundtruth_check.py --datasets "P4_tumor/1" --quality-filter "baseQ0mapQ0" 