#!/usr/bin/env python3
"""
Extract SNP positions and allele frequencies from 1000 Genomes VCF files.

This script creates preprocessed text files of SNP positions and their allele frequencies
from 1000 Genomes VCF files, which can be quickly loaded by the validation script.
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from tqdm import tqdm

# Configuration dictionaries for 1000 Genomes data
THOUSAND_GENOME_CONFIGS = {
    "GRCh38": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "pattern": "CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    "hg19": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19/",
        "pattern": "hg19_chr{chrom}.vcf.gz"
    }
}

def extract_info_field(info_str: str, field_name: str):
    """Extract a specific field from VCF INFO column."""
    for field in info_str.split(';'):
        if field.startswith(f"{field_name}="):
            return field.split('=')[1]
    return None

def process_chromosome(genome_build: str, chrom: str, output_dir: str, 
                      af_threshold: float = None, keys_only: bool = False) -> Tuple[str, int, int]:
    """
    Process a single chromosome VCF file to extract SNPs and their allele frequencies.
    
    Args:
        genome_build: "GRCh38" or "hg19"
        chrom: Chromosome number or letter (X, Y)
        output_dir: Directory to write output file
        af_threshold: If provided, only include variants with AF >= threshold
        keys_only: If True, only extract chr_pos keys without AF values
        
    Returns:
        Tuple of (chrom, total_variants, common_variants)
    """
    # Get the appropriate file path
    config = THOUSAND_GENOME_CONFIGS[genome_build]
    vcf_path = os.path.join(
        config["base_path"],
        config["pattern"].format(chrom=chrom)
    )
    
    # Get appropriate output path
    chrom_str = chrom if chrom.startswith("chr") else f"chr{chrom}"
    if keys_only:
        output_file = os.path.join(output_dir, f"{chrom_str}_snps_keys.txt")
    else:
        output_file = os.path.join(output_dir, f"{chrom_str}_snps_with_af.txt")
    
    if not os.path.exists(vcf_path):
        print(f"Warning: VCF file not found: {vcf_path}")
        return chrom, 0, 0
    
    # Process the file
    total_variants = 0
    common_variants = 0
    
    with gzip.open(vcf_path, 'rt') as vcf_file, open(output_file, 'w') as out_file:
        for line in vcf_file:
            if line.startswith('#'):
                continue
                
            fields = line.strip().split('\t')
            
            # Get SNP position
            raw_chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            info = fields[7]
            
            # Skip non-SNP variants or multi-allelic sites
            if len(ref) != 1 or len(alt) != 1 or ',' in alt:
                continue
                
            # Standardize chromosome format to always include "chr" prefix
            std_chrom = raw_chrom if raw_chrom.startswith('chr') else f"chr{raw_chrom}"
            
            # Create SNP key
            key = f"{std_chrom}_{pos}"
            
            total_variants += 1
            
            # Extract AF if needed
            if not keys_only:
                af_str = extract_info_field(info, 'AF')
                if af_str:
                    af = float(af_str.split(',')[0])  # Use first AF for multi-allelic sites
                    
                    # Apply AF threshold if provided
                    if af_threshold is not None:
                        if af < af_threshold:
                            continue
                    
                    common_variants += 1
                    out_file.write(f"{key}\t{af}\n")
            else:
                # Write just the key
                out_file.write(f"{key}\n")
                common_variants += 1
    
    return chrom, total_variants, common_variants

def main():
    parser = argparse.ArgumentParser(
        description="Extract SNP positions and allele frequencies from 1000 Genomes VCF files"
    )
    parser.add_argument('--genome', choices=['GRCh38', 'hg19', 'both'], default='both',
                      help="Genome build to process (default: both)")
    parser.add_argument('--chromosome', 
                      help="Process specific chromosome (e.g., 1, 2, ..., X, Y)")
    parser.add_argument('--af-threshold', type=float,
                      help="Only include variants with AF >= threshold")
    parser.add_argument('--output-dir', 
                      default="/data/maiziezhou_lab/yuqi/snv_calling/data/1kG_positions",
                      help="Base directory for output files")
    parser.add_argument('--threads', type=int, default=8,
                      help="Number of parallel threads to use")
    parser.add_argument('--keys-only', action='store_true',
                      help="Only extract chromosome_position keys, not AFs")
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    genomes_to_process = []
    if args.genome == 'both':
        genomes_to_process = ['GRCh38', 'hg19']
    else:
        genomes_to_process = [args.genome]
    
    for genome_build in genomes_to_process:
        genome_output_dir = os.path.join(args.output_dir, genome_build)
        os.makedirs(genome_output_dir, exist_ok=True)
        
        print(f"Processing {genome_build} genome...")
        
        # Determine chromosomes to process
        if args.chromosome:
            chromosomes = [args.chromosome]
        else:
            # Process all numbered chromosomes plus X and Y
            chromosomes = [str(i) for i in range(1, 23)] + ['X', 'Y']
        
        results = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            futures = []
            for chrom in chromosomes:
                future = executor.submit(
                    process_chromosome,
                    genome_build,
                    chrom,
                    genome_output_dir,
                    args.af_threshold,
                    args.keys_only
                )
                futures.append((future, chrom))
            
            # Process results as they complete
            for future, chrom in tqdm(futures, desc=f"Processing {genome_build} chromosomes"):
                try:
                    chrom, total, common = future.result()
                    results.append((chrom, total, common))
                    print(f"  {genome_build} chromosome {chrom}: {common:,} variants processed out of {total:,}")
                except Exception as e:
                    print(f"  Error processing {genome_build} chromosome {chrom}: {str(e)}")
        
        # Create a combined file with all chromosomes
        print(f"Creating combined file for {genome_build}...")
        
        if args.keys_only:
            combined_file = os.path.join(genome_output_dir, "all_snps_keys.txt")
            cmd = f"cat {genome_output_dir}/chr*_snps_keys.txt > {combined_file}"
        else:
            combined_file = os.path.join(genome_output_dir, "all_snps_with_af.txt")
            cmd = f"cat {genome_output_dir}/chr*_snps_with_af.txt > {combined_file}"
            
        subprocess.run(cmd, shell=True, check=True)
        
        # Count lines in the combined file
        with open(combined_file, 'r') as f:
            line_count = sum(1 for _ in f)
        print(f"Total {genome_build} variants in combined file: {line_count:,}")

if __name__ == "__main__":
    main()


# hg38:
# python scripts/tools/extract_1kg_snps_with_af.py --genome GRCh38 --output-dir /data/maiziezhou_lab/yuqi/snv_calling/data/1kG_positions --threads 30 --af-threshold 0.01