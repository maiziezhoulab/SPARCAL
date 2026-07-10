import os
import time
import glob
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Set
from tqdm import tqdm
import pandas as pd
import gzip
from pathlib import Path


# Default Parameters
DEFAULT_PARAMS = {
    "MIN_BASE_QUALITY": "5",
    "MIN_MAPPING_QUALITY": "5",
    "MAX_DEPTH": 10000000,
    "THREADS": 30,
    "MAX_FILES": None,
    "MIN_DEPTH": 0,
    "MIN_GQ": 8,
    "MIN_QUAL": 0,

}

# File Paths
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
}

# Reference configurations
REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [str(i) for i in range(1, 23)]  # 1, 2, 3, ..., 22
    },
    "FFPE_VISIUM": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-GRCh38-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    },
    "TUMOR":{
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22 
    }
}

# Opt-in flag to REGENERATE the pre-dedup DLPFC call set from the ORIGINAL
# non-dedup per-cell BAMs into a SEPARATE tree (data/dlpfc_prededup), leaving the
# post-dedup results under data/dlpfc untouched. Enable with env DLPFC_PREDEDUP=1.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"

# Dataset Configurations
# Dataset Configurations
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        # UMI-deduped per-cell BAMs (produced by run_slurm/dlpfc/0_umidedup_split_DLPFC.sh).
        # bam_base_path repoints ONLY the BAM glob to the project dir; base_path above
        # stays the read-only dataset for spatial/position files.
        # Non-dedup source (rollback): base_path/{section_id}/bam_bycell/*.bam
        # DLPFC_PREDEDUP=1 -> read the ORIGINAL non-dedup BAMs + write to data/dlpfc_prededup.
        "bam_base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD" if _PREDEDUP else "/data/maiziezhou_lab/leiy4/snv_calling",
        "bam_pattern": "{section_id}/bam_bycell/*.bam" if _PREDEDUP else "data/dlpfc/{section_id}/bam_bycell_dedup/*.bam",
        "output_dir": "data/dlpfc_prededup/{section_id}" if _PREDEDUP else "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True,
        "spatial_dir": "spatial",
        "position_file": "tissue_positions_list.csv",
        "in_tissue_column": 1  # Index of the column containing the in_tissue flag (0-based)
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}_hg19/P4_Tumor_output/outs/split_BAM/*.bam",
        "barcode_file": "spaceranger_align_rep{section_id}_hg19/Meta_Data/GSM4565823_barcodes.tsv.gz",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True,
        "spatial_dir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565823_P4_rep1_tissue_positions_list.csv",
            "2": "GSM4565824_P4_rep2_tissue_positions_list.csv"
        },
        "missing_tissue_file": "spaceranger_align_rep{section_id}_hg19/Meta_Data/missing_barcodes.txt"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}_hg19/P6_Tumor_output/outs/split_BAM/*.bam",
        "barcode_file": "spaceranger_align_rep{section_id}_hg19/Meta_Data/GSM4565825_barcodes.tsv.gz",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True,
        "spatial_dir": "Meta_Data",
        "position_file_patterns": {
            "1": "GSM4565825_P6_rep1_tissue_positions_list.csv",
            "2": "GSM4565826_P6_rep2_tissue_positions_list.csv"
        },
        "missing_tissue_file": "spaceranger_align_rep{section_id}_hg19/Meta_Data/missing_barcodes.txt"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "bam_pattern": "DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/split_BAM/*.bam",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC",
        "multiple_bams": True,
        "spatial_dir": "spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/spatial",
        "position_file": "tissue_positions.csv",
        "in_tissue_column": 1,
        "missing_tissue_file": "meta_data/dcis{section_id}/filtered_feature_bc_matrix/barcodes.tsv"
    },
    "OVAR_P5": {
        # Ovarian cancer patient (colleague-generated, GRCh38 with chr prefix).
        # Source outs/ is read-only; per-barcode split BAMs are written into OUR project
        # (data/ovar_p5/{section_id}/split_BAM), so base_path points at PROJECT_DIR for the
        # BAM glob while spatial files are read from the colleague's outs via spatial_base_path.
        "base_path": "/data/maiziezhou_lab/leiy4/snv_calling",
        "bam_pattern": "data/ovar_p5/{section_id}/split_BAM/*.bam",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "FFPE_VISIUM",   # GRCh38, chr prefix
        "multiple_bams": True,
        "spatial_base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "spatial_dir": "outs/spatial",
        "position_file": "tissue_positions_list.csv",   # no header, standard Visium V1
        "in_tissue_column": 1
    }
}

def setup_environment() -> Dict[str, str]:
    """Setup environment variables for library paths."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

def setup_output_dirs(output_base: str, params: Dict, call_mode: str) -> Dict[str, str]:
    """Setup output directories for the pipeline."""
    quality_filter = "baseQ" + params["MIN_BASE_QUALITY"] + "mapQ" + params["MIN_MAPPING_QUALITY"]
    
    # Base directory changes based on call mode
    base_dir = "mpileup_multi_bam" if call_mode == "multi" else "mpileup_single_bam"
    
    output_structure = {
        "vcf_dir": os.path.join(output_base, "output_VCFs", base_dir, quality_filter),
        "log_dir": os.path.join(output_base, "logs", base_dir, quality_filter),
        "metrics_dir": os.path.join(output_base, "metrics", base_dir, quality_filter)
    }
    
    for dir_path in output_structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_structure

# def run_command(cmd: str, env: Dict = None, **kwargs) -> int:
#     """Run a command with proper environment variables using os.system."""
#     if env is None:
#         env = os.environ.copy()
#     else:
#         env = {**os.environ.copy(), **env}
    
#     # Set environment variables
#     for key, value in env.items():
#         os.environ[key] = value
    
#     # Run command using os.system
#     return_code = os.system(cmd)
    
#     # Return the exit status
#     return return_code >> 8  # Convert exit status to proper return code

def run_command(cmd: str, env: Dict = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with proper environment variables."""
    if env is None:
        env = os.environ.copy()
    else:
        env = {**os.environ.copy(), **env}
    
    return subprocess.run(cmd, env=env, **kwargs)

def process_single_bam(bam_file: str, output_dirs: Dict[str, str], 
                      params: Dict, reference_path: str) -> Dict:
    """Process a single BAM file to generate one VCF."""
    # Create output name based on BAM filename
    bam_name = os.path.basename(bam_file).replace('.bam', '')
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"{bam_name}.vcf.gz")
    temp_vcf = os.path.join(output_dirs["vcf_dir"], f"{bam_name}_temp.vcf")
    log_file = os.path.join(output_dirs["log_dir"], f"{bam_name}.log")
    
    start_time = time.time()
    env = setup_environment()
    
    try:
        # Check if a position list file exists for this BAM
        snv_pos_file = os.path.join(
            os.path.dirname(output_dirs["vcf_dir"]), 
            "BAM_filtered", 
            params["quality_filter"],
            "snv_positions", 
            f"{bam_name}.txt"
        )
        
        # Add -l option if position file exists
        position_option = ""
        if os.path.exists(snv_pos_file) and os.path.getsize(snv_pos_file) > 0:
            position_option = f"-l {snv_pos_file}"
            with open(log_file, 'w') as log:
                log.write(f"Using position list file: {snv_pos_file}\n")
        
        # Process all regions in a single mpileup command
        mpileup_cmd = (
            f"{PATH_CONFIG['SAMTOOLS']} mpileup "
            f"-f {reference_path} "
            f"-q {params['MIN_MAPPING_QUALITY']} "
            f"-Q {params['MIN_BASE_QUALITY']} "
            f"-d {params['MAX_DEPTH']} -v "
            f"{position_option} "  # Add position list option if available
            f"{bam_file} | "
            f"{PATH_CONFIG['BCFTOOLS']} view | "
            f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
            f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {reference_path} | "
            f"grep -v '<X>\|INDEL' > {temp_vcf}"
        )
        
        with open(log_file, 'a') as log:
            log.write(f"Running command: {mpileup_cmd}\n")
            process = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
            if process.returncode != 0:
                raise Exception(f"Command failed with return code {process.returncode}")
        
        # Count SNPs
        snp_count = 0
        with open(temp_vcf, 'r') as f:
            for line in f:
                if not line.startswith('#'):
                    snp_count += 1
        
        # Compress and index final VCF
        run_command(f"{PATH_CONFIG['BGZIP']} -c {temp_vcf} > {output_vcf}", shell=True)
        run_command(['tabix', '-p', 'vcf', output_vcf])
        
        # Clean up temporary file
        if os.path.exists(temp_vcf):
            os.remove(temp_vcf)
        
        return {
            "bam": bam_file,
            "duration": time.time() - start_time,
            "snp_count": snp_count,
            "output_vcf": output_vcf,
            "used_positions_file": bool(position_option),
            "status": "completed"
        }
        
    except Exception as e:
        print(f"Error processing BAM {bam_file}: {str(e)}")
        return {
            "bam": bam_file,
            "duration": time.time() - start_time,
            "snp_count": 0,
            "status": "failed",
            "error": str(e)
        }

def process_single_bam_by_region(bam_file, output_dirs, params, reference_path):
    """Process a single BAM file by processing regions separately."""
    bam_name = os.path.basename(bam_file).replace('.bam', '')
    final_output_vcf = os.path.join(output_dirs["vcf_dir"], f"{bam_name}.vcf.gz")
    temp_dir = os.path.join(output_dirs["vcf_dir"], f"temp_{bam_name}")
    os.makedirs(temp_dir, exist_ok=True)
    
    start_time = time.time()
    region_results = []
    region_vcfs = []
    
    # Process each region separately
    for region in params['REGIONS']:
        region_output_vcf = os.path.join(temp_dir, f"{region}.vcf.gz")
        log_file = os.path.join(output_dirs["log_dir"], f"{bam_name}_{region}.log")
        
        try:
            # Run mpileup on this specific region
            mpileup_cmd = (
                f"{PATH_CONFIG['SAMTOOLS']} mpileup "
                f"-f {reference_path} "
                f"-r {region} "
                f"-q {params['MIN_MAPPING_QUALITY']} "
                f"-Q {params['MIN_BASE_QUALITY']} "
                f"-d {params['MAX_DEPTH']} -v "
                f"{bam_file} | "
                f"{PATH_CONFIG['BCFTOOLS']} view | "
                f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
                f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {reference_path} | "
                f"grep -v '<X>\|INDEL' | "
                f"{PATH_CONFIG['BGZIP']} -c > {region_output_vcf}"
            )

            with open(log_file, 'w') as log:
                env = setup_environment()
                process = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
                
            if process.returncode == 0 and os.path.exists(region_output_vcf) and os.path.getsize(region_output_vcf) > 0:
                # Index the VCF
                run_command(['tabix', '-p', 'vcf', region_output_vcf], check=True)
                
                # Count SNPs
                snp_count = 0
                with gzip.open(region_output_vcf, 'rt') as f:
                    for line in f:
                        if not line.startswith('#'):
                            snp_count += 1
                
                region_results.append({
                    "region": region,
                    "snp_count": snp_count, 
                    "status": "completed"
                })
                region_vcfs.append(region_output_vcf)
            else:
                # Create empty VCF with headers for failed regions
                with open(log_file, 'a') as log:
                    log.write(f"Failed to process region {region} - creating empty VCF\n")
                
                # Create minimal VCF header
                with gzip.open(region_output_vcf, 'wt') as f:
                    f.write("##fileformat=VCFv4.2\n")
                    f.write(f"##reference={reference_path}\n")
                    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
                
                run_command(['tabix', '-p', 'vcf', region_output_vcf], check=True)
                region_vcfs.append(region_output_vcf)
                region_results.append({
                    "region": region,
                    "snp_count": 0,
                    "status": "failed"
                })
                
        except Exception as e:
            print(f"Error processing region {region} for BAM {bam_name}: {str(e)}")
            region_results.append({
                "region": region,
                "snp_count": 0,
                "status": "failed",
                "error": str(e)
            })
    
    # Merge the regional VCFs
    if region_vcfs:
        try:
            # Create merged VCF with genotype inference
            merged_vcf = merge_vcfs(region_vcfs, temp_dir, params)
            # Move to final location
            shutil.move(merged_vcf, final_output_vcf)
            if os.path.exists(merged_vcf + '.tbi'):
                shutil.move(merged_vcf + '.tbi', final_output_vcf + '.tbi')
                
            total_duration = time.time() - start_time
            total_snps = sum(r["snp_count"] for r in region_results if r["status"] == "completed")
            
            # Clean up temp directory if successful
            shutil.rmtree(temp_dir)
            
            return {
                "bam": bam_file,
                "duration": total_duration,
                "snp_count": total_snps,
                "output_vcf": final_output_vcf,
                "status": "completed",
                "region_results": region_results
            }
        except Exception as e:
            return {
                "bam": bam_file,
                "duration": time.time() - start_time,
                "snp_count": 0,
                "status": "failed",
                "error": f"Error merging regional VCFs: {str(e)}",
                "region_results": region_results
            }
    else:
        return {
            "bam": bam_file,
            "duration": time.time() - start_time,
            "snp_count": 0,
            "status": "failed",
            "error": "Failed to process any regions successfully",
            "region_results": region_results
        }
    
def process_region(region: str, bam_input: str, output_dirs: Dict[str, str], 
                  params: Dict, reference_path: str, is_bam_list: bool = True) -> Dict:
    """Process a specific genomic region for BAM file(s)."""
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"region_{region}.vcf.gz")
    log_file = os.path.join(output_dirs["log_dir"], f"region_{region}.log")
    
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)
    start_time = time.time()
    env = setup_environment()
        
    try:
        # Construct mpileup command based on whether we're using a BAM list or single BAM
        if is_bam_list:
            bam_input_cmd = f"-b {bam_input}"
        else:
            bam_input_cmd = bam_input
            
        mpileup_cmd = (
            f"{PATH_CONFIG['SAMTOOLS']} mpileup "
            f"-f {reference_path} "
            f"-r {region} "
            f"-q {params['MIN_MAPPING_QUALITY']} "
            f"-Q {params['MIN_BASE_QUALITY']} "
            f"-d {params['MAX_DEPTH']} -v "
            f"{bam_input_cmd} | "  # Moved to end of mpileup options
            f"{PATH_CONFIG['BCFTOOLS']} view | "
            f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
            f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {reference_path} | "
            f"grep -v '<X>\|INDEL' | "
            f"{PATH_CONFIG['BGZIP']} -c > {output_vcf}"
        )

        with open(log_file, 'w') as log:
            process = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
            if process.returncode != 0:
                raise Exception(f"Command failed with return code {process.returncode}")

        snp_count = 0
        with gzip.open(output_vcf, 'rt') as f:
            for line in f:
                if not line.startswith('#'):
                    snp_count += 1
                    
        return {
            "region": region,
            "duration": time.time() - start_time,
            "snp_count": snp_count,
            "output_vcf": output_vcf
        }
        
    except Exception as e:
        print(f"Error processing region {region}: {str(e)}")
        return None

def load_in_tissue_barcodes(dataset_name: str, section_id: str) -> Set[str]:
    """
    Load barcodes that are within tissue boundaries.
    
    Args:
        dataset_name: Dataset name (e.g., DLPFC, P4_TUMOR, P6_TUMOR)
        section_id: Section ID
        
    Returns:
        Set of barcodes that are within tissue
    """
    in_tissue_barcodes = set()
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Determine the path to the tissue positions file
    if dataset_name == "DLPFC":
        # DLPFC-specific paths using updated config
        spatial_base_dir = os.path.join(dataset_config["base_path"], section_id)
        spatial_subdir = os.path.join(spatial_base_dir, dataset_config["spatial_dir"])
        position_file = os.path.join(spatial_subdir, dataset_config["position_file"])
        
        # Load the positions file
        try:
            # Format: barcode, in_tissue, array_row, array_col, pxl_row_in_fullres, pxl_col_in_fullres
            df = pd.read_csv(position_file, header=None)
            in_tissue_col = dataset_config.get("in_tissue_column", 1)  # Default to column 1
            
            for _, row in df.iterrows():
                barcode = row[0]
                in_tissue = int(row[in_tissue_col])
                if in_tissue == 1:  # Only consider spots that are in tissue
                    in_tissue_barcodes.add(barcode)
                    
            print(f"Loaded {len(in_tissue_barcodes)} in-tissue barcodes from tissue positions file")
        except Exception as e:
            print(f"Error loading DLPFC tissue positions: {str(e)}")
            print(f"Continuing with all BAM files for DLPFC as requested.")
            # Instead of raising an error, we'll return an empty set so all BAMs are used
            return set()
        
    elif dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
        # Handle P4/P6 tumor datasets using the new config entries
        if "missing_tissue_file" in dataset_config:
            # Check if an explicit out-of-tissue barcode file exists
            missing_barcodes_file = os.path.join(
                dataset_config["base_path"],
                dataset_config["missing_tissue_file"].format(section_id=section_id)
            )
            
            if os.path.exists(missing_barcodes_file):
                # Read the list of out-of-tissue barcodes
                out_tissue_barcodes = set()
                with open(missing_barcodes_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            out_tissue_barcodes.add(parts[0])  # First element is the barcode
                
                print(f"Loaded {len(out_tissue_barcodes)} out-of-tissue barcodes from missing_barcodes.txt")
                
                # Get the position file to read all barcodes
                position_file_pattern = dataset_config["position_file_patterns"][section_id]
                position_file = os.path.join(
                    dataset_config["base_path"],
                    f"spaceranger_align_rep{section_id}_hg19",
                    dataset_config["spatial_dir"],
                    position_file_pattern
                )
                
                try:
                    # Read all barcodes from positions file
                    df = pd.read_csv(position_file, header=None)
                    for _, row in df.iterrows():
                        barcode = row[0]
                        # Include if not in the out-of-tissue set
                        if barcode not in out_tissue_barcodes:
                            in_tissue_barcodes.add(barcode)
                except Exception as e:
                    print(f"Error loading {dataset_name} tissue positions: {str(e)}")
                    raise
            else:
                print(f"Warning: Missing tissue file not found: {missing_barcodes_file}")
                print(f"Using all spots for {dataset_name}.")
                
                # Fallback to using all barcodes from the barcode file
                barcode_file = os.path.join(
                    dataset_config["base_path"],
                    dataset_config["barcode_file"].format(section_id=section_id)
                )
                if os.path.exists(barcode_file):
                    barcodes = read_barcode_list(barcode_file)
                    in_tissue_barcodes = set(barcodes)
        else:
            print(f"Warning: No missing_tissue_file configuration for {dataset_name}. Using all spots.")
            # If no missing_tissue_file is configured, use all barcodes
            barcode_file = os.path.join(
                dataset_config["base_path"],
                dataset_config["barcode_file"].format(section_id=section_id)
            )
            if os.path.exists(barcode_file):
                barcodes = read_barcode_list(barcode_file)
                in_tissue_barcodes = set(barcodes)
    
    elif dataset_name == "DCIS":
        # Handle DCIS dataset - missing_tissue_file contains the whitelist of valid barcodes
        if "missing_tissue_file" in dataset_config:
            barcode_whitelist_file = os.path.join(
                dataset_config["base_path"],
                dataset_config["missing_tissue_file"].format(section_id=section_id)
            )
            
            if os.path.exists(barcode_whitelist_file):
                # Read the whitelist of valid barcodes
                barcodes = read_barcode_list(barcode_whitelist_file)
                in_tissue_barcodes = set(barcodes)
                print(f"Loaded {len(in_tissue_barcodes)} valid barcodes from barcode whitelist file")
            else:
                print(f"Warning: Barcode whitelist file not found: {barcode_whitelist_file}")
                print(f"Using all BAM files for DCIS.")
                return set()  # Return empty set to use all BAMs
        else:
            print(f"Warning: No missing_tissue_file configuration for DCIS. Using all spots.")
            return set()  # Return empty set to use all BAMs

    elif dataset_name == "OVAR_P5":
        # Standard headerless Visium tissue_positions_list.csv in the colleague's read-only outs.
        position_file = os.path.join(
            dataset_config["spatial_base_path"], section_id,
            dataset_config["spatial_dir"], dataset_config["position_file"]
        )
        try:
            df = pd.read_csv(position_file, header=None)
            in_tissue_col = dataset_config.get("in_tissue_column", 1)
            for _, row in df.iterrows():
                if int(row[in_tissue_col]) == 1:
                    in_tissue_barcodes.add(row[0])
            print(f"Loaded {len(in_tissue_barcodes)} in-tissue barcodes from {position_file}")
        except Exception as e:
            print(f"Error loading OVAR_P5 tissue positions: {str(e)}")
            return set()  # Return empty set to use all BAMs

    print(f"Loaded {len(in_tissue_barcodes)} in-tissue barcodes for {dataset_name} section {section_id}")
    return in_tissue_barcodes

def read_barcode_list(barcode_file: str) -> List[str]:
    """
    Read barcodes from a TSV file (compressed or uncompressed)
    
    Args:
        barcode_file: Path to the barcode TSV file
        
    Returns:
        List of barcodes
    """
    # Check if file is gzipped
    if barcode_file.endswith('.gz'):
        opener = gzip.open
        mode = 'rt'  # text mode for gzipped files
    else:
        opener = open
        mode = 'r'
    
    with opener(barcode_file, mode) as f:
        # Read barcodes, assuming they're in the first column
        barcodes = [line.strip().split('\t')[0] for line in f if line.strip()]
    
    return barcodes

def get_bam_list_for_tumor(base_path: str, barcode_file: str, section_id: str, bam_pattern: str) -> List[str]:
    """
    Generate list of BAM files based on barcodes from reference file
    
    Args:
        base_path: Base path to the dataset
        barcode_file: Path to the barcode reference file
        section_id: Section ID
        
    Returns:
        List of BAM file paths that exist
    """
    # Read barcodes
    barcodes = read_barcode_list(barcode_file)
    
    # Construct split BAM directory path
    # split_bam_dir = os.path.join(
    #     base_path,
    #     f"spaceranger_align_rep{section_id}",
    #     "P4_Tumor_output/outs/split_BAM"
    # )
    # Use bam pattern to find the split BAMs
    split_bam_dir = os.path.join(base_path, bam_pattern)

    
    # Generate BAM paths and filter for existing files
    bam_files = []
    for barcode in barcodes:
        bam_path = os.path.join(split_bam_dir, f"{barcode}.bam")
        if os.path.exists(bam_path):
            bam_files.append(bam_path)
        else:
            print(f"File not found: {bam_path}")
    
    return bam_files

def infer_gt_from_pl(input_vcf: str, output_vcf: str,
                     min_depth: int = 5,
                     min_gq: float = 7,
                     min_qual: int = 0):
    """
    Infer genotypes from PL fields and calculate BAF using I16 values.
    
    Args:
        input_vcf: Input VCF file (can be gzipped)
        output_vcf: Output VCF file path
        min_depth: Minimum read depth required
        min_gq: Minimum genotype quality required (difference between best and second-best PL)
        min_qual: Minimum variant quality required
    """
    def calculate_gq(pl_values):
        """Calculate genotype quality as difference between best and second-best PL."""
        sorted_pls = sorted(pl_values)
        return sorted_pls[1] - sorted_pls[0]  # Second best - best (lower is better for PL)
    
    def calculate_baf_from_i16(i16_values):
        """Calculate BAF using I16 field values."""
        try:
            ref_depth = i16_values[0] + i16_values[1]  # Total reference depth
            alt_depth = i16_values[2] + i16_values[3]  # Total alternate depth
            total_depth = ref_depth + alt_depth
            
            if total_depth == 0:
                return 0.0
                
            return alt_depth / total_depth
        except (IndexError, ZeroDivisionError):
            return 0.0

    def parse_i16(i16_str):
        """Parse I16 into numbers.

        Use float() first: at high depth bcftools writes the later I16 entries
        (sum-of-squared qualities) in scientific notation, e.g. '1.38392e+07'.
        int(x) cannot parse that — the old `[int(x) for x in ...]` raised
        ValueError on the whole list and silently fell back to [0]*16, zeroing
        BAF for every high-depth site (BAF=0 with DiscordantBAF). Only indices
        0-3 (ref/alt base counts) are used downstream; float is exact for those.
        """
        try:
            return [int(float(x)) for x in i16_str.split(',')]
        except (ValueError, AttributeError):
            return [0] * 16

    # Open files
    infile = gzip.open(input_vcf, 'rt') if input_vcf.endswith('.gz') else open(input_vcf, 'r')
    
    with infile, open(output_vcf, 'w') as f_out:
        # Process header
        for line in infile:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    # Add new FORMAT and FILTER fields
                    f_out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                    f_out.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality (difference between best and second-best PL)">\n')
                    f_out.write('##FORMAT=<ID=BAF,Number=1,Type=Float,Description="B-Allele Frequency calculated from I16 values">\n')
                    f_out.write('##FILTER=<ID=LowQual,Description="Low quality variant call">\n')
                    f_out.write('##FILTER=<ID=LowDP,Description="Low read depth">\n')
                    f_out.write('##FILTER=<ID=LowGQ,Description="Low genotype quality">\n')
                    f_out.write('##FILTER=<ID=DiscordantBAF,Description="BAF inconsistent with genotype">\n')
                f_out.write(line)
                continue
            
            # Process variant lines
            fields = line.strip().split('\t')
            
            # Extract INFO field values
            info_dict = dict(item.split('=') for item in fields[7].split(';') if '=' in item)
            
            # Get depth and I16 values
            depth = int(info_dict.get('DP', 0))
            i16_values = parse_i16(info_dict.get('I16', ''))
            
            # Parse FORMAT field
            format_field = fields[8]
            sample_field = fields[9]
            format_dict = dict(zip(format_field.split(':'), sample_field.split(':')))
            
            try:
                pl_values = [int(x) for x in format_dict.get('PL', '').split(',')]
                if len(pl_values) != 3:
                    continue
                    
                qual = float(fields[5]) if fields[5] != '.' else 0
                
                # Calculate BAF using I16 values
                baf = calculate_baf_from_i16(i16_values)
                
            except (ValueError, KeyError, IndexError):
                continue
            
            # Determine best genotype from PL values
            min_pl_index = pl_values.index(min(pl_values))
            inferred_gt = {0: '0/0', 1: '0/1', 2: '1/1'}[min_pl_index]
            
            if inferred_gt == '0/0':
                continue

            # Calculate genotype quality
            gq = calculate_gq(pl_values)
            
            # Validate BAF against genotype
            baf_valid = True
            if inferred_gt == '0/0' and baf > 0.15:
                baf_valid = False
            elif inferred_gt == '0/1' and (baf < 0.35 or baf > 0.65):
                baf_valid = False
            elif inferred_gt == '1/1' and baf < 0.85:
                baf_valid = False
            
            # Apply filters
            filters = []
            if depth < min_depth:
                filters.append('LowDP')
            if gq < min_gq:
                filters.append('LowGQ')
            if qual < min_qual:
                filters.append('LowQual')
            if not baf_valid:
                filters.append('DiscordantBAF')
            
            # Update FILTER field
            fields[6] = ';'.join(filters) if filters else 'PASS'
            
            # Update FORMAT and sample fields
            fields[8] = 'GT:GQ:BAF:' + format_field
            fields[9] = f"{inferred_gt}:{gq}:{baf:.3f}:" + sample_field
            
            # Write updated line
            f_out.write('\t'.join(fields) + '\n')

def merge_vcfs(vcf_files: List[str], output_dir: str, params: Dict) -> str:
    """Merge and process VCF files."""
    final_output_dir = os.path.join(output_dir)
    os.makedirs(final_output_dir, exist_ok=True)
    
    temp_merged = os.path.join(final_output_dir, "temp_merged.vcf")
    temp_gt_inferred = os.path.join(final_output_dir, "temp_gt_inferred.vcf")
    final_vcf = os.path.join(final_output_dir, "merged_sorted_gt.vcf")
    final_vcf_gz = f"{final_vcf}.gz"
    
    print(f"Merging VCF files into: {temp_merged}")
    
    # Merge VCF files
    with open(temp_merged, 'w') as out_file:
        # Write header from first file
        with gzip.open(vcf_files[0], 'rt') as first_file:
            for line in first_file:
                if line.startswith('#'):
                    out_file.write(line)
                else:
                        break
        
        # Concatenate variants
        for vcf_file in vcf_files:
            with gzip.open(vcf_file, 'rt') as infile:
                for line in infile:
                    if not line.startswith('#'):
                        out_file.write(line)
    
    # Infer genotypes
    infer_gt_from_pl(temp_merged, temp_gt_inferred, params['MIN_DEPTH'], params['MIN_GQ'], params['MIN_QUAL'])
    # Sort and process
    run_command(f"{PATH_CONFIG['BCFTOOLS']} sort {temp_gt_inferred} -o {final_vcf}", shell=True)
    run_command([PATH_CONFIG['BGZIP'], '-f', final_vcf])
    run_command(['tabix', '-p', 'vcf', final_vcf_gz])
    
    # Cleanup
    os.remove(temp_merged)
    
    return final_vcf_gz

def add_chr_prefix_to_vcf(input_vcf: str, output_vcf: str):
    """
    Add 'chr' prefix to chromosome numbers in VCF file.
    
    Args:
        input_vcf: Path to input VCF file (can be gzipped)
        output_vcf: Path to output VCF file (will be gzipped)
    """
    print(f"Adding chr prefix to {input_vcf}")
    temp_vcf = output_vcf.replace('.gz', '')
    
    with (gzip.open(input_vcf, 'rt') if input_vcf.endswith('.gz') else open(input_vcf, 'r')) as f_in, \
         open(temp_vcf, 'w') as f_out:
        
        for line in f_in:
            if line.startswith('#'):
                f_out.write(line)
                continue
                
            fields = line.strip().split('\t')
            # Add 'chr' prefix if it's a number or X/Y
            if fields[0].isdigit() or fields[0] in ['X', 'Y']:
                fields[0] = 'chr' + fields[0]
            
            f_out.write('\t'.join(fields) + '\n')
    
    # Compress the output file
    run_command(f"{PATH_CONFIG['BGZIP']} -f {temp_vcf}", shell=True)
    
    # Index the new VCF
    run_command(['tabix', '-p', 'vcf', output_vcf])
    
    # Remove temporary file if it exists
    if os.path.exists(temp_vcf):
        os.remove(temp_vcf)

def process_vcfs_for_chr_prefix(output_dirs: Dict[str, str], call_mode: str, 
                              dataset_name: str) -> None:
    """
    Process all VCF files in the output directory to add chr prefix if needed.
    
    Args:
        output_dirs: Dictionary of output directories
        call_mode: The calling mode ('single' or 'multi')
        dataset_name: Name of the dataset
    """
    # Get reference configuration for this dataset
    dataset_config = DATASET_CONFIGS[dataset_name]
    reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
    
    # Only proceed if chr_prefix is empty
    if reference_config['chr_prefix'] == '':
        vcf_dir = output_dirs['vcf_dir']
        
        if call_mode == 'multi':
            # For multi mode, only process the final merged VCF
            merged_vcf = os.path.join(vcf_dir, "merged_sorted_gt.vcf.gz")
            if os.path.exists(merged_vcf):
                chr_prefix_vcf = os.path.join(vcf_dir, "merged_sorted_gt_chr.vcf.gz")
                add_chr_prefix_to_vcf(merged_vcf, chr_prefix_vcf)
                # Replace original with chr-prefixed version
                os.rename(chr_prefix_vcf, merged_vcf)
                os.rename(chr_prefix_vcf + '.tbi', merged_vcf + '.tbi')
        else:
            # For single mode, process each individual VCF
            vcf_files = glob.glob(os.path.join(vcf_dir, "*.vcf.gz"))
            for vcf_file in vcf_files:
                if not vcf_file.endswith('.tbi'):  # Skip index files
                    chr_prefix_vcf = vcf_file.replace('.vcf.gz', '_chr.vcf.gz')
                    add_chr_prefix_to_vcf(vcf_file, chr_prefix_vcf)
                    # Replace original with chr-prefixed version
                    os.rename(chr_prefix_vcf, vcf_file)
                    os.rename(chr_prefix_vcf + '.tbi', vcf_file + '.tbi')

# 

def run_pipeline(dataset_name: str, section_id: str = None, custom_params: Dict = None,
                call_mode: str = "multi", filter_out_tissue: bool = False):
    """Run the SNV calling pipeline for a specific dataset."""
    params = DEFAULT_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    if not dataset_config:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Get reference configuration
    reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
    params['REGIONS'] = reference_config['regions']
    
    # Determine BAM file pattern and output directory.
    # bam_base_path (optional) decouples the BAM glob root from base_path so a
    # dataset can keep its (read-only) base_path for spatial/position files while
    # reading BAMs from elsewhere — e.g. DLPFC UMI-deduped BAMs under the project dir.
    bam_base_path = dataset_config.get("bam_base_path", dataset_config["base_path"])
    if dataset_config["has_sections"]:
        if not section_id:
            raise ValueError(f"Dataset {dataset_name} requires a section_id")
        bam_pattern = os.path.join(bam_base_path,
                                 dataset_config["bam_pattern"].format(section_id=section_id))
        output_base = os.path.join(PATH_CONFIG["PROJECT_DIR"],
                                dataset_config["output_dir"].format(section_id=section_id))
    else:
        bam_pattern = os.path.join(bam_base_path,
                                 dataset_config["bam_pattern"])
        output_base = os.path.join(PATH_CONFIG["PROJECT_DIR"],
                                dataset_config["output_dir"])
    
    # Setup directories with call mode
    output_dirs = setup_output_dirs(output_base, params, call_mode)
    
    # Get list of BAM files
    bam_files = glob.glob(bam_pattern)
    # print example of bam files
    print(f"Example BAM files: {bam_files[:5]}")
    if params["MAX_FILES"]:
        bam_files = bam_files[:params["MAX_FILES"]]
    
    if not bam_files:
        raise ValueError(f"No BAM files found at: {bam_pattern}")
    
    # Filter out-of-tissue BAM files if requested
    if filter_out_tissue:
        print(f"Filtering out-of-tissue spots...")
        in_tissue_barcodes = load_in_tissue_barcodes(dataset_name, section_id)
        
        # print several in-tissue barcodes
        print(f"Example in-tissue barcodes: {list(in_tissue_barcodes)[:5]}")
        # prnt example of barcodes
        print(f"Example read barcodes: {os.path.basename(bam_files[1])}")
        # Filter BAM files to include only in-tissue barcodes
        in_tissue_bam_files = []
        for bam_file in bam_files:
            barcode = os.path.basename(bam_file).replace('.bam', '')
            if barcode in in_tissue_barcodes:
                in_tissue_bam_files.append(bam_file)
        
        print(f"Filtered {len(bam_files) - len(in_tissue_bam_files)} out-of-tissue BAM files")
        print(f"Processing {len(in_tissue_bam_files)} in-tissue BAM files")
        
        # Use filtered BAM list
        bam_files = in_tissue_bam_files
        
        # Save the list of in-tissue barcodes for reference
        in_tissue_list_file = os.path.join(output_dirs["log_dir"], "in_tissue_barcodes.txt")
        with open(in_tissue_list_file, 'w') as f:
            for barcode in sorted(in_tissue_barcodes):
                f.write(f"{barcode}\n")
        print(f"Saved in-tissue barcode list to: {in_tissue_list_file}")
    else:
        print(f"Found {len(bam_files)} BAM files (no tissue filtering)")
    
    results = []
    if call_mode == "multi":
        # Original multi-BAM processing
        bam_list_file = os.path.join(output_dirs["log_dir"], "bam_list.txt")
        with open(bam_list_file, 'w') as f:
            for bam in bam_files:
                f.write(f"{bam}\n")
        
        # Process regions in parallel
        with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
            future_to_region = {
                executor.submit(
                    process_region, 
                    region, 
                    bam_list_file, 
                    output_dirs, 
                    params,
                    reference_config['path'],
                    True
                ): region
                for region in params['REGIONS']
            }
            
            with tqdm(total=len(params['REGIONS']), desc="Processing regions") as pbar:
                for future in as_completed(future_to_region):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
        
        # Merge VCFs for multi mode
        region_vcfs = [result["output_vcf"] for result in results]
        final_vcf = merge_vcfs(region_vcfs, output_dirs["vcf_dir"], params)
        
    else:  # single BAM processing
        # Process each BAM file in parallel
        with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
            future_to_bam = {
                executor.submit(
                    process_single_bam,
                    bam,
                    output_dirs,
                    params,
                    reference_config['path']
                ): bam
                for bam in bam_files
            }
            
            with tqdm(total=len(bam_files), desc="Processing BAM files") as pbar:
                for future in as_completed(future_to_bam):
                    result = future.result()
                    if result:
                        results.append(result)
                    pbar.update(1)
    
    # Add chr prefix if needed
    if reference_config['chr_prefix'] == '':
        process_vcfs_for_chr_prefix(output_dirs, call_mode, dataset_name)
    
    # Save metrics
    metrics_df = pd.DataFrame(results)
    metrics_file = os.path.join(output_dirs["metrics_dir"], "processing_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    
    return metrics_df

def validate_section_id(dataset_name: str, section_id: str) -> bool:
    """Validate section ID for datasets that require it."""
    dataset_config = DATASET_CONFIGS.get(dataset_name)
    if not dataset_config or not dataset_config.get("has_sections"):
        return True
    
    if "section_ids" in dataset_config:
        return section_id in dataset_config["section_ids"]
    return True  # For datasets like DLPFC that don't have predefined section IDs

def main():
    parser = argparse.ArgumentParser(description="Multi-BAM SNV Calling Pipeline")
    parser.add_argument("--dataset", choices=list(DATASET_CONFIGS.keys()), required=True,
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--max_files", type=int, help="Maximum number of BAM files to process")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to use")
    parser.add_argument("--regions_file", help="File containing custom regions")
    parser.add_argument("--base_quality", type=str, default="0", help="Minimum base quality")
    parser.add_argument("--mapping_quality", type=str, default="0", help="Minimum mapping quality")
    parser.add_argument("--call_mode", choices=["single", "multi"], default="multi",
                      help="Calling mode: process BAMs individually or together (default: multi)")
    parser.add_argument("--filter_out_tissue", action="store_true", 
                      help="Filter out spots outside of tissue boundaries")
    
    args = parser.parse_args()
    
    # Validate section ID if required
    if not validate_section_id(args.dataset, args.section_id):
        valid_sections = DATASET_CONFIGS[args.dataset].get("section_ids", [])
        raise ValueError(f"Invalid section_id for {args.dataset}. Valid section IDs are: {valid_sections}")
    
    # Setup environment
    env = setup_environment()
    print("Environment setup complete:")
    print(f"PATH: {env['PATH']}")
    print(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    
    custom_params = {
        "MAX_FILES": args.max_files,
        "THREADS": args.threads,
        "MIN_BASE_QUALITY": args.base_quality,
        "MIN_MAPPING_QUALITY": args.mapping_quality
    }
    
    if args.regions_file:
        with open(args.regions_file, 'r') as f:
            custom_params['REGIONS'] = [line.strip() for line in f]
    
    metrics_df = run_pipeline(args.dataset, args.section_id, custom_params, args.call_mode, args.filter_out_tissue)
    
    # Print summary information
    print("\nPipeline Summary:")
    if args.call_mode == "multi":
        print(f"Total regions processed: {len(metrics_df)}")
        print(f"Average processing time per region: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
    else:
        print(f"Total BAMs processed: {len(metrics_df)}")
        succeeded = len(metrics_df[metrics_df['status'] == 'completed'])
        failed = len(metrics_df[metrics_df['status'] == 'failed'])
        print(f"Successfully processed: {succeeded}")
        print(f"Failed: {failed}")
        print(f"Average processing time per BAM: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
        
        if failed > 0:
            print("\nFailed BAMs:")
            failed_bams = metrics_df[metrics_df['status'] == 'failed']
            for _, row in failed_bams.iterrows():
                print(f"  {os.path.basename(row['bam'])}: {row.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC:
# python mpileup_pipeline.py --dataset DLPFC --section_id 151507 --max_files 2
# For 10X_BC_6.5MM:
# python mpileup_pipeline.py --dataset 10X_BC_6.5MM --max_files 2
# For 10X_BC_FFPE:
# python mpileup_pipeline.py --dataset 10X_BC_FFPE --max_files 2
# For P4_TUMOR:
# python mpileup_pipeline.py --dataset P4_TUMOR --section_id 1
# For P6_TUMOR:
# python mpileup_pipeline.py --dataset P6_TUMOR --section_id 2


# Usage examples:
# For DLPFC multi:
# python scripts/calling/mpileup_pipeline.py --dataset DLPFC --section_id 151507 --max_files 2
# For DLPFC_SVM_FILTERED, run single mpileup, baseQ13mapQ20:
# python scripts/calling/mpileup_pipeline.py --dataset DLPFC_SVM_FILTERED --section_id 151507 --base_quality 13 --mapping_quality 20 --call_mode single --threads 30
# For 10X_BC_6.5MM:
# python scripts/calling/mpileup_pipeline.py --dataset 10X_BC_6.5MM --max_files 5
# For 10X_BC_FFPE:
# python scripts/calling/mpileup_pipeline.py --dataset 10X_BC_FFPE --max_files 2
# For P4_TUMOR:
# python scripts/calling/mpileup_pipeline.py --dataset P4_TUMOR --section_id 1 --filter_out_tissue
# For P6_TUMOR:
# python scripts/calling/mpileup_pipeline.py --dataset P6_TUMOR --section_id 2
# For dlpfc, 151510, multi, baseQ13mapQ20:
# python scripts/calling/mpileup_pipeline.py --dataset DLPFC --section_id 151510 --base_quality 13 --mapping_quality 20 --call_mode multi --threads 30