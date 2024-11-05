import os
import time
import glob
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
from tqdm import tqdm

# Default Parameters
DEFAULT_PARAMS = {
    "MIN_BASE_QUALITY": 0,
    "MIN_MAPPING_QUALITY": 0,
    "MAX_DEPTH": 10000000,
    "THREADS": 100,
    "MAX_FILES": None,  # None means process all files
    "REGIONS": [f"{i}" for i in range(1, 23)] + ['X', 'Y']
}

# File Paths
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "REFERENCE_SEQ": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
    "BEDFILE": "/data/maiziezhou_lab/yuqi/snv_calling/data/reference/GRCh38.bed",
    "HEADER": "/data/maiziezhou_lab/yuqi/snv_calling/data/reference/header.txt",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip"
}

def setup_output_dirs(section_id: str, caller_type: str) -> Dict[str, str]:
    """Setup output directories for the pipeline."""
    base_dir = os.path.join(PATH_CONFIG["PROJECT_DIR"], "data/dlpfc", section_id)
    output_structure = {
        "vcf_dir": os.path.join(base_dir, "output_VCFs", caller_type),
        "log_dir": os.path.join(base_dir, "logs", caller_type),
        "metrics_dir": os.path.join(base_dir, "metrics", caller_type)
    }
    
    for dir_path in output_structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_structure

def count_snps(vcf_file: str) -> int:
    """Count the number of SNPs in a VCF file."""
    count = 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                count += 1
    return count

def setup_environment():
    """Setup environment variables for library paths."""
    # Add apps directory to PATH
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    
    # Add apps directory to LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

def create_regions_file(regions: List[str], output_dir: str) -> str:
    """Create a regions file for mpileup."""
    regions_file = os.path.join(output_dir, "regions.txt")
    with open(regions_file, 'w') as f:
        for region in regions:
            f.write(f"{region}\n")
    return regions_file

def process_region(region: str, bam_list: str, output_dirs: Dict[str, str], 
                  params: Dict) -> Dict:
    """Process a specific genomic region for multiple BAM files."""
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"region_{region}.vcf")
    log_file = os.path.join(output_dirs["log_dir"], f"region_{region}.log")
    
    start_time = time.time()
    
    # Set up environment
    env = setup_environment()
    
    mpileup_cmd = (
        f"{PATH_CONFIG['SAMTOOLS']} mpileup "
        f"-f {PATH_CONFIG['REFERENCE_SEQ']} "
        f"-b {bam_list} "
        f"-r {region} "
        f"-q {params['MIN_MAPPING_QUALITY']} "
        f"-Q {params['MIN_BASE_QUALITY']} "
        f"-d {params['MAX_DEPTH']} -v | "
        f"{PATH_CONFIG['BCFTOOLS']} view | "
        f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
        f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {PATH_CONFIG['REFERENCE_SEQ']} | "
        f"grep -v '<X>\|INDEL' > {output_vcf}"
    )
    
    with open(log_file, 'w') as log:
        run_command(mpileup_cmd, env=env, shell=True, stderr=log)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "region": region,
        "duration": duration,
        "snp_count": count_snps(output_vcf),
        "output_vcf": output_vcf
    }

def run_command(cmd: str, env: Dict = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with proper environment variables."""
    if env is None:
        env = os.environ.copy()
    else:
        env = {**os.environ.copy(), **env}
    
    return subprocess.run(cmd, env=env, **kwargs)

def run_mpileup(bam_file: str, output_dirs: Dict[str, str], params: Dict) -> Dict:
    """Run mpileup on a single BAM file."""
    basename = os.path.basename(bam_file).replace('.bam', '')
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"{basename}.vcf")
    log_file = os.path.join(output_dirs["log_dir"], f"{basename}.log")
    
    start_time = time.time()
    
    mpileup_cmd = (
        f"{PATH_CONFIG['SAMTOOLS']} mpileup "
        f"-f {PATH_CONFIG['REFERENCE_SEQ']} {bam_file} "
        f"-q {params['MIN_MAPPING_QUALITY']} "
        f"-Q {params['MIN_BASE_QUALITY']} "
        f"-d {params['MAX_DEPTH']} -v | "
        f"{PATH_CONFIG['BCFTOOLS']} view | "
        f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
        f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {PATH_CONFIG['REFERENCE_SEQ']} | "
        f"grep -v '<X>\|INDEL' > {output_vcf}"
    )
    
    # Set up environment
    env = setup_environment()
    
    with open(log_file, 'w') as log:
        run_command(mpileup_cmd, env=env, shell=True, stderr=log)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "bam_file": basename,
        "duration": duration,
        "snp_count": count_snps(output_vcf),
        "output_vcf": output_vcf
    }

def run_multi_bam_mpileup(bam_files: List[str], output_dirs: Dict[str, str], 
                         params: Dict) -> Tuple[pd.DataFrame, str]:
    """Run mpileup on multiple BAM files in parallel by region."""
    # Create BAM list file
    bam_list_file = os.path.join(output_dirs["log_dir"], "bam_list.txt")
    with open(bam_list_file, 'w') as f:
        for bam in bam_files:
            f.write(f"{bam}\n")
    
    # Process regions in parallel
    results = []
    regions = params.get('REGIONS', DEFAULT_PARAMS['REGIONS'])
    
    with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
        future_to_region = {
            executor.submit(process_region, region, bam_list_file, output_dirs, params): region
            for region in regions
        }
        
        # Setup progress bar
        pbar = tqdm(total=len(regions), desc="Processing regions")
        
        for future in as_completed(future_to_region):
            try:
                result = future.result()
                results.append(result)
                pbar.update(1)
            except Exception as e:
                region = future_to_region[future]
                print(f"Error processing region {region}: {str(e)}")
        
        pbar.close()
    
    # Merge VCF files from all regions
    merged_vcf = os.path.join(output_dirs["vcf_dir"], "merged_multi_bam.vcf")
    region_vcfs = [result["output_vcf"] for result in results]
    merge_vcfs(region_vcfs, merged_vcf)
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(results)
    metrics_file = os.path.join(output_dirs["metrics_dir"], "region_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"metrics df is: {metrics_df}")
    return metrics_df, merged_vcf

def run_custom_caller(bam_file: str, output_dirs: Dict[str, str], 
                     caller_script: str, params: Dict) -> Dict:
    """Run custom caller (old or new) on a single BAM file."""
    basename = os.path.basename(bam_file).replace('.bam', '')
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"{basename}.vcf")
    
    # Write header
    with open(output_vcf, 'w') as f:
        with open(PATH_CONFIG["HEADER"], 'r') as header:
            f.write(header.read())
    
    start_time = time.time()
    
    # Set up environment
    env = setup_environment()
    
    for chrom in range(1, 23):
        run_command([
            "python", caller_script,
            "--reference_seq", PATH_CONFIG["REFERENCE_SEQ"],
            "--chromosome", str(chrom),
            "--bamfile", bam_file,
            "--bedfile", PATH_CONFIG["BEDFILE"],
            "--header", PATH_CONFIG["HEADER"],
            "--out", output_vcf
        ], env=env)
    
    # Process X and Y chromosomes
    for chrom in ['X', 'Y']:
        run_command([
            "python", caller_script,
            "--reference_seq", PATH_CONFIG["REFERENCE_SEQ"],
            "--chromosome", chrom,
            "--bamfile", bam_file,
            "--bedfile", PATH_CONFIG["BEDFILE"],
            "--header", PATH_CONFIG["HEADER"],
            "--out", output_vcf
        ], env=env)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "bam_file": basename,
        "duration": duration,
        "snp_count": count_snps(output_vcf),
        "output_vcf": output_vcf
    }

def merge_vcfs(vcf_files: List[str], output_file: str):
    """Merge multiple VCF files into one."""
    # Write header from the first file
    with open(vcf_files[0], 'r') as first_file:
        with open(output_file, 'w') as out_file:
            for line in first_file:
                if line.startswith('#'):
                    out_file.write(line)
                else:
                    break
    
    # Concatenate variants from all files
    for vcf_file in vcf_files:
        with open(vcf_file, 'r') as f:
            with open(output_file, 'a') as out_file:
                for line in f:
                    if not line.startswith('#'):
                        out_file.write(line)

def run_pipeline(section_id: str, caller_type: str, custom_params: Dict = None,
                multi_bam_mode: bool = False):
    """Enhanced pipeline function supporting both single and multi-BAM processing."""
    params = DEFAULT_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    if multi_bam_mode:
        caller_type = "mpileup_multi_bam"
    output_dirs = setup_output_dirs(section_id, caller_type)
    
    bam_dir = f"/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section_id}/bam_bycell"
    bam_files = glob.glob(os.path.join(bam_dir, "*.bam"))
    
    if params["MAX_FILES"]:
        bam_files = bam_files[:params["MAX_FILES"]]
    print(multi_bam_mode, caller_type)

    if caller_type == "mpileup_multi_bam":
        print("Running multi-BAM mpileup pipeline...")
        results = run_multi_bam_mpileup(bam_files, output_dirs, params)
        return results
    else:
        # Original single-BAM processing logic
        results = []
        pbar = tqdm(total=len(bam_files), desc=f"Processing {caller_type}")
        
        with ThreadPoolExecutor(max_workers=params["THREADS"]) as executor:
            print(f"Running {caller_type} pipeline...")
            future_to_bam = {}
            
            for bam_file in bam_files:
                if caller_type == "mpileup":
                    future = executor.submit(run_mpileup, bam_file, output_dirs, params)
                else:
                    caller_script = os.path.join(
                        PATH_CONFIG["PROJECT_DIR"], 
                        "scripts/calling",
                        "self_caller.py" if caller_type == "old_caller" else "new_caller.py"
                    )
                    future = executor.submit(run_custom_caller, bam_file, output_dirs, caller_script, params)
                future_to_bam[future] = bam_file
            
            for future in as_completed(future_to_bam):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {future_to_bam[future]}: {str(e)}")
        
        pbar.close()
        
        metrics_df = pd.DataFrame(results)
        metrics_file = os.path.join(output_dirs["metrics_dir"], f"{section_id}_{caller_type}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
        return metrics_df


def main():
    parser = argparse.ArgumentParser(description="SNV Calling Pipeline")
    parser.add_argument("--section_id", required=True, help="Section ID")
    parser.add_argument("--caller_type", choices=["mpileup", "old_caller", "new_caller"],
                      required=True, help="Type of caller to use")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to use")
    parser.add_argument("--multi_bam", action="store_true", 
                      help="Process multiple BAM files together with region parallelization")
    parser.add_argument("--regions_file", help="File containing custom regions (optional)")
    
    args = parser.parse_args()
    
    # Setup environment variables
    env = setup_environment()
    print("Environment setup complete:")
    print(f"PATH: {env['PATH']}")
    print(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    
    custom_params = {
        "MAX_FILES": args.max_files,
        "THREADS": args.threads
    }
    
    # Load custom regions if provided
    if args.regions_file:
        with open(args.regions_file, 'r') as f:
            custom_params['REGIONS'] = [line.strip() for line in f]
    
    # Run pipeline
    result = run_pipeline(args.section_id, args.caller_type, custom_params, args.multi_bam)
    
    if args.multi_bam:
        metrics_df, merge_vcfs = result
        # print(metrics_df)
        print("\nMulti-BAM Pipeline Summary:")
        print(f"Total regions processed: {len(metrics_df)}")
        print(f"Average processing time per region: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
        print(f"Merged VCF file: {merge_vcfs}")
    else:
        metrics_df = result
        print("\nSingle-BAM Pipeline Summary:")
        print(f"Total files processed: {len(metrics_df)}")
        print(f"Average processing time: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
        print(f"Average SNPs per file: {metrics_df['snp_count'].mean():.2f}")

if __name__ == "__main__":
    main()


# Usage
# python run_slurm/dlpfc/run_caller_pipeline.py --section_id 151507 --caller_type mpileup --max_files 10
# python run_slurm/dlpfc/run_caller_pipeline.py --section_id 151507 --caller_type new_caller --max_files 10
# python run_slurm/dlpfc/run_caller_pipeline.py --section_id 151507 --caller_type mpileup --multi_bam --max_files 10