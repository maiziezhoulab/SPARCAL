#!/usr/bin/env python3
"""
Calculate overlap between experimental VCF and Reference (like GATK) ground truth VCFs.

This script compares a specified VCF file with reference VCFs,
handling proper compression and indexing if needed.

Usage:
    python calc_overlap.py --input-vcf path/to/input.vcf.gz --output-dir path/to/output
"""

import os
import sys
import gzip
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Path configurations
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/yuqi/snv_calling/apps/tabix"
}

# Ground truth VCF configurations
GATK_VCFS = {
    "P4_tumor_all": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_cSCC_WES/P4_cSCC_WES_gatk_snp_chr1_22.vcf.gz",
    "P4_tumor_exome": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_cSCC_WES/P4_cSCC_WES_gatk_exome_snps_chr1_22.vcf.gz",
    "P4_normal_all": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz",
    "P4_normal_exome": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_exome_snps_chr1_22.vcf.gz",
    # "P4_mutect2_all": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz",
    # "P4_mutect2_exome": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz"
}

# gnomAD VCF configurations
GNOMAD_VCFS_HG19 = {
    f"gnomad_hg19_chr{i}": f"/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/gnomAD/2.1.1/vcf/genomes_hg19/gnomad.genomes.r2.1.1.sites.{i}.vcf.bgz"
    for i in range(1, 23)
}

GNOMAD_VCFS_HG38 = {
    f"gnomad_hg38_chr{i}": f"/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/gnomAD/2.1.1/vcf/genomes_hg38/gnomad.genomes.r2.1.1.sites.{i}.liftover_grch38.vcf.bgz"
    for i in range(1, 23)
}

# Merge all reference VCFs
ALL_REFERENCE_VCFS = {**GATK_VCFS, **GNOMAD_VCFS_HG19, **GNOMAD_VCFS_HG38}

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

def run_command(cmd: str, check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return exit code, stdout, and stderr."""
    result = subprocess.run(
        cmd, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    if check and result.returncode != 0:
        print(f"Command failed: {cmd}")
        print(f"Exit code: {result.returncode}")
        print(f"stderr: {result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, output=result.stdout, stderr=result.stderr
        )
    
    return result.returncode, result.stdout, result.stderr

def is_bgzipped(file_path: str) -> bool:
    """Check if a file is properly bgzipped."""
    try:
        # Try to index with tabix - if it works, the file is bgzipped
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_index = os.path.join(temp_dir, "temp.tbi")
            cmd = f"{PATH_CONFIG['TABIX']} -fp vcf -o {temp_index} {file_path}"
            result = subprocess.run(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True, check=False
            )
            return result.returncode == 0
    except Exception:
        return False

def prepare_vcf(input_vcf: str, output_dir: str) -> str:
    """
    Prepare the input VCF by ensuring it's properly bgzipped and indexed.
    
    Args:
        input_vcf: Path to input VCF file
        output_dir: Directory to store processed files
        
    Returns:
        Path to prepared VCF file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine output file name
    base_name = os.path.basename(input_vcf)
    if base_name.endswith('.vcf.gz'):
        base_name = base_name[:-7]  # Remove .vcf.gz
    elif base_name.endswith('.vcf'):
        base_name = base_name[:-4]  # Remove .vcf
        
    prepared_vcf = os.path.join(output_dir, f"{base_name}_prepared.vcf.gz")
    
    # Check if input VCF exists
    if not os.path.exists(input_vcf):
        raise FileNotFoundError(f"Input VCF file not found: {input_vcf}")
    
    # Check if the file is already properly bgzipped
    if input_vcf.endswith('.vcf.gz') and is_bgzipped(input_vcf):
        print(f"Input VCF is already properly bgzipped: {input_vcf}")
        
        # If input and output would be the same, just copy it
        if os.path.abspath(input_vcf) != os.path.abspath(prepared_vcf):
            print(f"Copying to: {prepared_vcf}")
            shutil.copy2(input_vcf, prepared_vcf)
        else:
            prepared_vcf = input_vcf
            
        # Make sure it's indexed
        if not os.path.exists(prepared_vcf + '.tbi'):
            print(f"Indexing VCF: {prepared_vcf}")
            run_command(f"{PATH_CONFIG['TABIX']} -p vcf {prepared_vcf}")
            
    else:
        print(f"Processing VCF: {input_vcf}")
        
        # Temp file for decompressed content
        temp_vcf = os.path.join(output_dir, f"{base_name}_temp.vcf")
        
        try:
            # If input is gzipped, decompress it first
            if input_vcf.endswith('.gz'):
                print("Decompressing gzipped VCF...")
                with gzip.open(input_vcf, 'rt') as f_in, open(temp_vcf, 'w') as f_out:
                    for line in f_in:
                        f_out.write(line)
            else:
                # Just copy the file
                shutil.copy2(input_vcf, temp_vcf)
            
            # Compress with bgzip
            print(f"Compressing with bgzip to: {prepared_vcf}")
            run_command(f"{PATH_CONFIG['BGZIP']} -c {temp_vcf} > {prepared_vcf}")
            
            # Index with tabix
            print(f"Indexing VCF: {prepared_vcf}")
            run_command(f"{PATH_CONFIG['TABIX']} -p vcf {prepared_vcf}")
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_vcf):
                os.remove(temp_vcf)
    
    return prepared_vcf

def count_variants(vcf_path: str) -> int:
    """Count the number of variants in a VCF file."""
    count = 0
    
    try:
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if not line.startswith('#'):
                    count += 1
    except Exception as e:
        print(f"Error counting variants in {vcf_path}: {e}")
        
    return count

def compare_with_ref(input_vcf: str, ref_vcf: str, output_dir: str, ref_name: str) -> Tuple[str, int, int]:
    """
    Compare the input VCF with a reference VCF.
    
    Args:
        input_vcf: Path to the input VCF
        ref_vcf: Path to the reference VCF
        output_dir: Directory to store results
        ref_name: Name of the reference for output naming
        
    Returns:
        Tuple of (overlap_vcf_path, overlap_count, input_count)
    """
    # Create overlap directory
    overlap_dir = os.path.join(output_dir, f"overlap_{ref_name}")
    os.makedirs(overlap_dir, exist_ok=True)
    
    # Run bcftools isec to identify overlapping variants
    cmd = (
        f"{PATH_CONFIG['BCFTOOLS']} isec -n=2 -w1 -O z -p {overlap_dir} "
        f"{input_vcf} {ref_vcf}"
    )
    
    print(f"\nComparing with {ref_name}...")
    print(f"Command: {cmd}")
    
    try:
        run_command(cmd)
        
        # Check if overlap files were created
        overlap_vcf = os.path.join(overlap_dir, "0000.vcf.gz")
        if os.path.exists(overlap_vcf):
            # Count variants in original and overlap files
            input_count = count_variants(input_vcf)
            overlap_count = count_variants(overlap_vcf)
            
            print(f"\nOverlap Results with {ref_name}:")
            print(f"Total input variants: {input_count}")
            print(f"Variants also in Ref: {overlap_count}")
            print(f"Percentage overlap: {overlap_count/max(1, input_count)*100:.2f}%")
            
            # Write summary to file
            summary_file = os.path.join(overlap_dir, "summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Overlap Results with {ref_name}:\n")
                f.write(f"Total input variants: {input_count}\n")
                f.write(f"Variants also in Ref: {overlap_count}\n")
                f.write(f"Percentage overlap: {overlap_count/max(1, input_count)*100:.2f}%\n")
            
            # Index the overlap VCF
            if not os.path.exists(overlap_vcf + '.tbi'):
                run_command(f"{PATH_CONFIG['TABIX']} -p vcf {overlap_vcf}")
                
            return overlap_vcf, overlap_count, input_count
        else:
            print(f"Warning: No overlap file created in {overlap_dir}")
            return os.path.join(overlap_dir, "0000.vcf.gz"), 0, count_variants(input_vcf)
            
    except subprocess.CalledProcessError as e:
        print(f"Error comparing with {ref_name}: {e}")
        return None, 0, count_variants(input_vcf)

def write_combined_summary(output_dir: str, results: Dict[str, Dict[str, any]]):
    """Write a combined summary of all comparison results."""
    summary_file = os.path.join(output_dir, "combined_summary.txt")
    
    with open(summary_file, 'w') as f:
        f.write("Combined Overlap Results Summary\n")
        f.write("===============================\n\n")
        
        for ref_name, result in results.items():
            input_count = result.get('input_count', 0)
            overlap_count = result.get('overlap_count', 0)
            overlap_pct = overlap_count / max(1, input_count) * 100
            
            f.write(f"Comparison with {ref_name}:\n")
            f.write(f"  Total input variants: {input_count}\n")
            f.write(f"  Variants also in Ref: {overlap_count}\n")
            f.write(f"  Percentage overlap: {overlap_pct:.2f}%\n\n")
    
    print(f"\nCombined summary written to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate overlap between experimental VCF and reference VCFs")
        
    parser.add_argument("--input-vcf", required=True,
                    help="Path to input VCF file")
    parser.add_argument("--output-dir", default="overlap_results",
                    help="Directory to store output files (default: overlap_results)")
    parser.add_argument("--ref-type", choices=["gatk", "gnomad_hg19", "gnomad_hg38"],
                    default="gatk", help="Type of reference to compare against")
    parser.add_argument("--chr", type=str, help="Specific chromosome to compare (e.g., '1')")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check if output directory exists, create if not
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Prepare input VCF
    try:
        prepared_vcf = prepare_vcf(args.input_vcf, args.output_dir)
        print(f"Prepared VCF: {prepared_vcf}")
    except Exception as e:
        print(f"Error preparing input VCF: {e}")
        return 1
    
    refs_to_use = {}
    if args.ref_type == "gatk":
        refs_to_use = GATK_VCFS
    elif args.ref_type == "gnomad_hg19":
        if args.chr:
            refs_to_use = {f"gnomad_hg19_chr{args.chr}": GNOMAD_VCFS_HG19[f"gnomad_hg19_chr{args.chr}"]}
        else:
            refs_to_use = GNOMAD_VCFS_HG19
    elif args.ref_type == "gnomad_hg38":
        if args.chr:
            refs_to_use = {f"gnomad_hg38_chr{args.chr}": GNOMAD_VCFS_HG38[f"gnomad_hg38_chr{args.chr}"]}
        else:
            refs_to_use = GNOMAD_VCFS_HG38

    # Compare with each selected reference
    results = {}
    for ref_name, ref_vcf in refs_to_use.items():
        # Check if reference VCF exists
        if not os.path.exists(ref_vcf):
            print(f"Warning: Reference VCF not found: {ref_vcf}")
            continue
        
        # Compare with reference
        overlap_vcf, overlap_count, input_count = compare_with_ref(
            prepared_vcf, ref_vcf, args.output_dir, ref_name
        )
        
        # Store results
        results[ref_name] = {
            'overlap_vcf': overlap_vcf,
            'overlap_count': overlap_count,
            'input_count': input_count
        }
    
    # Write combined summary
    if results:
        write_combined_summary(args.output_dir, results)
        
    return 0

if __name__ == "__main__":
    sys.exit(main())

# compare with: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/SVMModel/baseQ0mapQ0/positive_training.vcf.gz

# compare with: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_high_confidence.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_high_confidence.vcf.gz

# compare with: /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_heterozygous.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/yuqi/snv_calling/data/P4_tumor/1/output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_heterozygous.vcf.gz

# compare with:     - /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.txt|.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.vcf.gz

# compare with     - /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.txt|.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.vcf.gz

# compare with     - /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.txt|.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/ambiguous/ambiguous_variants.vcf.gz

# compare with /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz
# python scripts/tools/calculate_overlap_vcfs.py --input-vcf /data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz