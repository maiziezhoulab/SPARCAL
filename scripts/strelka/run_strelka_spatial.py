#!/usr/bin/env python3
"""
Run Strelka on a spatial transcriptomics dataset (DLPFC) one BAM file at a time.
This script handles running Strelka in germline mode on multiple BAM files
from a spatial dataset, processing them in parallel batches.
"""

import os
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
import sys
from pathlib import Path

# Configuration adapted from mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
}

# Reference configurations from mpileup_pipeline.py
REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [str(i) for i in range(1, 23)]  # 1, 2, 3, ..., 22
    }
}

# Dataset Configurations
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "bam_pattern": "{section_id}/bam_bycell/*.bam",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True
    }
}

# Path to Strelka installation - update this to point to your installation
STRELKA_PATH = "/data/maiziezhou_lab/yuqi/snv_calling/strelka-2.9.2.centos6_x86_64"

def setup_environment():
    """Setup environment variables for library paths."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }


def get_bam_list(dataset_name, section_id, max_files=None):
    """Get list of BAM files from the dataset."""
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if dataset_config["has_sections"]:
        if not section_id:
            raise ValueError(f"Dataset {dataset_name} requires a section_id")
        bam_pattern = os.path.join(dataset_config["base_path"], 
                                 dataset_config["bam_pattern"].format(section_id=section_id))
    else:
        bam_pattern = os.path.join(dataset_config["base_path"], 
                                 dataset_config["bam_pattern"])
    
    bam_files = glob.glob(bam_pattern)
    
    if not bam_files:
        raise ValueError(f"No BAM files found at: {bam_pattern}")
    
    # Sort the BAMs for deterministic behavior
    bam_files.sort()
    
    # Limit to max_files if specified
    if max_files:
        bam_files = bam_files[:max_files]
    
    print(f"Found {len(bam_files)} BAM files")
    return bam_files


def setup_output_dirs(dataset_name, section_id, quality_filter):
    """Setup output directories for Strelka results."""
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    
    if dataset_config["has_sections"]:
        output_base = os.path.join(PATH_CONFIG["PROJECT_DIR"], 
                                dataset_config["output_dir"].format(section_id=section_id))
    else:
        output_base = os.path.join(PATH_CONFIG["PROJECT_DIR"], 
                                dataset_config["output_dir"])
    
    strelka_dir = os.path.join(output_base, "output_VCFs/strelka", quality_filter)
    log_dir = os.path.join(output_base, "logs/strelka", quality_filter)
    metrics_dir = os.path.join(output_base, "metrics/strelka", quality_filter)
    
    # Create directories
    for dir_path in [strelka_dir, log_dir, metrics_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    return {
        "strelka_dir": strelka_dir,
        "log_dir": log_dir,
        "metrics_dir": metrics_dir
    }


def run_strelka_on_bam(bam_file, reference_path, output_dirs, exome_flag=False, threads=1):
    """Run Strelka on a single BAM file."""
    bam_name = os.path.basename(bam_file).replace('.bam', '')
    run_dir = os.path.join(output_dirs["strelka_dir"], bam_name)
    log_file = os.path.join(output_dirs["log_dir"], f"{bam_name}.log")
    final_vcf = os.path.join(output_dirs["strelka_dir"], f"{bam_name}.vcf.gz")
    
    # Skip if output already exists
    if os.path.exists(final_vcf):
        print(f"Skipping {bam_name} - output already exists: {final_vcf}")
        return {
            "bam": bam_file,
            "status": "skipped",
            "output_vcf": final_vcf
        }
    
    # Start timing
    start_time = time.time()
    
    try:
        # Create the run directory
        os.makedirs(run_dir, exist_ok=True)
        
        # Configure Strelka
        config_cmd = [
            f"{STRELKA_PATH}/bin/configureStrelkaGermlineWorkflow.py",
            f"--bam", bam_file,
            f"--referenceFasta", reference_path,
            f"--runDir", run_dir
        ]
        
        # Add exome flag if requested
        if exome_flag:
            config_cmd.append("--exome")
        
        # Run configuration
        with open(log_file, 'w') as log:
            log.write(f"Configuration command: {' '.join(config_cmd)}\n\n")
            subprocess.run(config_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        
        # Run Strelka workflow
        run_cmd = [
            f"{run_dir}/runWorkflow.py",
            "-m", "local",
            "-j", str(threads)
        ]
        
        with open(log_file, 'a') as log:
            log.write(f"Execution command: {' '.join(run_cmd)}\n\n")
            subprocess.run(run_cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        
        # Copy the final VCF to the output directory
        variants_vcf = os.path.join(run_dir, "results", "variants.vcf.gz")
        subprocess.run(["cp", variants_vcf, final_vcf], check=True)
        subprocess.run(["cp", f"{variants_vcf}.tbi", f"{final_vcf}.tbi"], check=True)
        
        # Calculate duration
        duration = time.time() - start_time
        
        return {
            "bam": bam_file,
            "status": "completed",
            "duration": duration,
            "output_vcf": final_vcf
        }
        
    except Exception as e:
        duration = time.time() - start_time
        with open(log_file, 'a') as log:
            log.write(f"ERROR: {str(e)}\n")
        
        return {
            "bam": bam_file,
            "status": "failed",
            "duration": duration,
            "error": str(e)
        }


def run_strelka_parallel(bam_files, reference_path, output_dirs, exome_flag=False, max_workers=4, threads_per_worker=4):
    """Run Strelka on multiple BAM files in parallel."""
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit jobs
        future_to_bam = {
            executor.submit(
                run_strelka_on_bam, 
                bam, 
                reference_path, 
                output_dirs, 
                exome_flag, 
                threads_per_worker
            ): bam for bam in bam_files
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_bam), total=len(bam_files), desc="Processing BAMs"):
            bam = future_to_bam[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed {os.path.basename(bam)}: {result['status']}")
            except Exception as e:
                results.append({
                    "bam": bam,
                    "status": "error",
                    "error": str(e)
                })
                print(f"Error processing {os.path.basename(bam)}: {str(e)}")
    
    return results


def merge_vcfs(vcf_files, output_vcf, bcftools_path):
    """Merge multiple VCF files into one."""
    if not vcf_files:
        print("No VCF files to merge")
        return False
    
    cmd = [
        bcftools_path, "merge",
        "-o", output_vcf,
        "-O", "z"
    ] + vcf_files
    
    try:
        subprocess.run(cmd, check=True)
        subprocess.run([f"{PATH_CONFIG['APPS_DIR']}/tabix", "-p", "vcf", output_vcf], check=True)
        return True
    except Exception as e:
        print(f"Error merging VCFs: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run Strelka on DLPFC spatial dataset")
    parser.add_argument("--section_id", default="151507", help="Section ID for DLPFC dataset")
    parser.add_argument("--max_files", type=int, help="Maximum number of BAM files to process")
    parser.add_argument("--quality_filter", default="strelkaQ0", help="Quality filter directory name")
    parser.add_argument("--parallel", type=int, default=4, help="Number of parallel Strelka jobs")
    parser.add_argument("--threads", type=int, default=4, help="Threads per Strelka job")
    parser.add_argument("--exome", action="store_true", help="Use exome mode for Strelka")
    parser.add_argument("--merge", action="store_true", help="Merge resulting VCFs")
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check if Strelka is properly configured
    if not os.path.exists(STRELKA_PATH):
        print(f"ERROR: Strelka installation not found at {STRELKA_PATH}")
        print("Please update the STRELKA_PATH variable in the script to point to your Strelka installation")
        sys.exit(1)
    
    # Get reference path
    reference_path = REFERENCE_CONFIGS["DLPFC"]["path"]
    if not os.path.exists(reference_path):
        print(f"ERROR: Reference genome not found at {reference_path}")
        sys.exit(1)
    
    # Setup output directories
    output_dirs = setup_output_dirs("DLPFC", args.section_id, args.quality_filter)
    
    # Get BAM files
    bam_files = get_bam_list("DLPFC", args.section_id, args.max_files)
    
    # Run Strelka
    print(f"\nRunning Strelka on {len(bam_files)} BAM files in {args.parallel} parallel jobs")
    results = run_strelka_parallel(
        bam_files=bam_files,
        reference_path=reference_path,
        output_dirs=output_dirs,
        exome_flag=args.exome,
        max_workers=args.parallel,
        threads_per_worker=args.threads
    )
    
    # Print summary
    completed = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    
    print("\nStrelka Processing Summary:")
    print(f"Total BAMs processed: {len(results)}")
    print(f"Successfully completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Skipped (already processed): {skipped}")
    
    if failed > 0:
        print("\nFailed BAMs:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  {os.path.basename(result['bam'])}: {result.get('error', 'Unknown error')}")
    
    # Merge VCFs if requested
    if args.merge and completed > 0:
        print("\nMerging VCFs...")
        successful_vcfs = [r['output_vcf'] for r in results if r['status'] in ['completed', 'skipped']]
        
        if successful_vcfs:
            merged_vcf = os.path.join(output_dirs["strelka_dir"], "merged_variants.vcf.gz")
            success = merge_vcfs(successful_vcfs, merged_vcf, PATH_CONFIG["BCFTOOLS"])
            
            if success:
                print(f"Successfully merged VCFs to: {merged_vcf}")
            else:
                print("Failed to merge VCFs")
        else:
            print("No VCFs to merge")

    # Write summary to file
    summary_file = os.path.join(output_dirs["metrics_dir"], "strelka_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Strelka Processing Summary\n")
        f.write("=========================\n\n")
        f.write(f"Dataset: DLPFC\n")
        f.write(f"Section ID: {args.section_id}\n")
        f.write(f"Quality Filter: {args.quality_filter}\n\n")
        f.write(f"Total BAMs processed: {len(results)}\n")
        f.write(f"Successfully completed: {completed}\n")
        f.write(f"Failed: {failed}\n")
        f.write(f"Skipped (already processed): {skipped}\n\n")
        
        if failed > 0:
            f.write("Failed BAMs:\n")
            for result in results:
                if result['status'] == 'failed':
                    f.write(f"  {os.path.basename(result['bam'])}: {result.get('error', 'Unknown error')}\n")

    print(f"\nSummary written to: {summary_file}")
    print(f"Results directory: {output_dirs['strelka_dir']}")
    print(f"Logs directory: {output_dirs['log_dir']}")


if __name__ == "__main__":
    main()