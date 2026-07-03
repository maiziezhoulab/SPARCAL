#!/usr/bin/env python3
"""
Consecutive De Novo SNV Finder

This script identifies consecutive de novo SNVs without interspersed germline variants
to create a negative training set for SVM classification, following the approach in the
Monopogen paper.

Usage:
    python consecutive_denovo_finder.py --dataset DLPFC --section_id 151507 --quality_filter baseQ0mapQ0
                                       --min-consecutive 3
                                       --max-distance 1000
                                       --threads 8
"""

import os
import sys
import gzip
import argparse
import pysam
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm
from typing import Dict, List, Set, Tuple, Optional
import subprocess
import tempfile
import matplotlib.pyplot as plt
import numpy as np

# Path configurations - matches mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/yuqi/snv_calling/apps/tabix"
}

# Dataset Configurations
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
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

def run_command(command: str, check: bool = True) -> int:
    """Run a shell command and return the exit code."""
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, command, output=result.stdout, stderr=result.stderr
        )
    return result.returncode

def get_chromosomes_from_vcf(vcf_path: str) -> List[str]:
    """Extract list of chromosomes from a VCF file."""
    try:
        if not os.path.exists(vcf_path + '.tbi'):
            print(f"Indexing {vcf_path} as no index file found")
            run_command(f"{PATH_CONFIG['TABIX']} -p vcf {vcf_path}")
            
        # Use tabix to list contigs in the VCF file
        command = f"{PATH_CONFIG['TABIX']} -l {vcf_path}"
        result = subprocess.run(command, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              universal_newlines=True)
        
        chromosomes = result.stdout.strip().split('\n')
        return chromosomes
    except subprocess.CalledProcessError as e:
        print(f"Error getting chromosomes from VCF: {e}")
        print(f"stderr: {e.stderr}")
        # Try an alternative approach - actually parse some of the VCF
        try:
            with pysam.VariantFile(vcf_path) as vcf:
                return list(vcf.header.contigs)
        except Exception as e2:
            print(f"Alternative approach failed: {e2}")
            raise

def extract_positions(vcf_path: str, chromosome: str) -> Set[int]:
    """
    Extract SNV positions for a specific chromosome from a VCF file.
    
    Args:
        vcf_path: Path to the VCF file
        chromosome: Chromosome to extract SNVs from
        
    Returns:
        Set of positions (integers)
    """
    positions = set()
    
    try:
        with pysam.VariantFile(vcf_path) as vcf:
            for record in vcf.fetch(chromosome):
                # Only include SNVs (exclude indels, etc.)
                if len(record.ref) == 1 and len(record.alts[0]) == 1:
                    positions.add(record.pos)
    except ValueError:
        # The contig might not exist in this VCF
        print(f"Warning: Chromosome {chromosome} not found in {vcf_path}")
    except Exception as e:
        print(f"Error extracting positions from {vcf_path} for chromosome {chromosome}: {e}")
    
    return positions

def find_consecutive_denovo(denovo_positions, germline_positions, min_consecutive, max_distance):
    # Sort both position lists
    sorted_denovo = sorted(denovo_positions)
    sorted_germline = sorted(germline_positions)
    
    # Initialize variables
    current_run = []
    consecutive_ranges = []
    germline_idx = 0  # Current position in the germline array
    
    # Find potential runs
    for pos in sorted_denovo:
        # If we have a current run, check if we can add to it
        if current_run:
            last_pos = current_run[-1]
            
            # Check if this position is within max_distance
            if pos - last_pos <= max_distance:
                # Check if there are any germline variants between last_pos and pos
                has_germline_between = False
                while germline_idx < len(sorted_germline) and sorted_germline[germline_idx] <= pos:
                    if sorted_germline[germline_idx] > last_pos:
                        # Found a germline variant between last_pos and pos
                        has_germline_between = True
                        break
                    germline_idx += 1
                
                if has_germline_between:
                    # Found a germline in between - end the current run and check if it's valid
                    if len(current_run) >= min_consecutive:
                        consecutive_ranges.append((current_run[0], current_run[-1]))
                    # Start a new run with the current position
                    current_run = [pos]
                else:
                    # No germline in between, add to the current run
                    current_run.append(pos)
            else:
                # Too far away - end the current run and check if it's valid
                if len(current_run) >= min_consecutive:
                    consecutive_ranges.append((current_run[0], current_run[-1]))
                # Start a new run with the current position
                current_run = [pos]
        else:
            # Start a new run
            current_run = [pos]
    
    # Process the last run
    if len(current_run) >= min_consecutive:
        consecutive_ranges.append((current_run[0], current_run[-1]))
    
    return consecutive_ranges

def extract_variants_from_ranges(
    vcf_path: str, 
    chromosome: str, 
    ranges: List[Tuple[int, int]]
) -> List[Dict]:
    """
    Extract variants from specified ranges in a VCF file.
    
    Args:
        vcf_path: Path to the VCF file
        chromosome: Chromosome to extract variants from
        ranges: List of (start_pos, end_pos) tuples
        
    Returns:
        List of variant dictionaries with all relevant fields
    """
    variants = []
    
    if not ranges:
        return variants
    
    try:
        with pysam.VariantFile(vcf_path) as vcf:
            # Create a merged set of all positions in ranges
            positions_set = set()
            for start_pos, end_pos in ranges:
                positions_set.update(range(start_pos, end_pos + 1))
            
            # Fetch variants in chromosome and filter by position
            for record in vcf.fetch(chromosome):
                # Skip non-SNVs
                if len(record.ref) != 1 or len(record.alts[0]) != 1:
                    continue
                    
                if record.pos in positions_set:
                    # Extract all the fields from the VCF record
                    variant = {
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'id': record.id,
                        'ref': record.ref,
                        'alt': record.alts[0],  # Take first alt for simplicity
                        'qual': record.qual,
                        'filter': list(record.filter),
                        'info': dict(record.info),
                        'format': list(record.format),
                        'samples': dict(zip(record.samples.keys(), 
                                         [{key: record.samples[sample][key] for key in record.format} 
                                         for sample in record.samples]))
                    }
                    variants.append(variant)
    except Exception as e:
        print(f"Error extracting variants from {vcf_path} for chromosome {chromosome}: {e}")
    
    return variants

def extract_isolated_variants(
    vcf_path: str,
    chromosome: str,
    consecutive_positions: Set[int]
) -> List[Dict]:
    """
    Extract variants that are NOT part of consecutive runs.
    
    Args:
        vcf_path: Path to the VCF file
        chromosome: Chromosome to extract variants from
        consecutive_positions: Set of positions that are part of consecutive runs
        
    Returns:
        List of isolated variant dictionaries
    """
    isolated_variants = []
    
    try:
        with pysam.VariantFile(vcf_path) as vcf:
            # Fetch variants in chromosome
            for record in vcf.fetch(chromosome):
                # Skip non-SNVs
                if len(record.ref) != 1 or len(record.alts[0]) != 1:
                    continue
                
                # Only include if not in consecutive positions
                if record.pos not in consecutive_positions:
                    # Extract all the fields from the VCF record
                    variant = {
                        'chrom': record.chrom,
                        'pos': record.pos,
                        'id': record.id,
                        'ref': record.ref,
                        'alt': record.alts[0],  # Take first alt for simplicity
                        'qual': record.qual,
                        'filter': list(record.filter),
                        'info': dict(record.info),
                        'format': list(record.format),
                        'samples': dict(zip(record.samples.keys(), 
                                         [{key: record.samples[sample][key] for key in record.format} 
                                         for sample in record.samples]))
                    }
                    isolated_variants.append(variant)
    except Exception as e:
        print(f"Error extracting isolated variants from {vcf_path} for chromosome {chromosome}: {e}")
    
    return isolated_variants

def get_genotype_counts(variants: List[Dict]) -> Dict[str, int]:
    """
    Count the number of variants by genotype.
    
    Args:
        variants: List of variant dictionaries
        
    Returns:
        Dictionary mapping genotype to count
    """
    genotype_counts = {"0/1": 0, "1/1": 0, "other": 0}
    
    for variant in variants:
        # Get the sample data (assuming single-sample VCF)
        sample_key = list(variant['samples'].keys())[0]
        sample_data = variant['samples'][sample_key]
        
        # Extract genotype if available
        if 'GT' in sample_data:
            gt = sample_data['GT']
            if gt == (0, 1) or gt == "0/1":  # Handle both tuple and string formats
                genotype_counts["0/1"] += 1
            elif gt == (1, 1) or gt == "1/1":
                genotype_counts["1/1"] += 1
            else:
                genotype_counts["other"] += 1
        else:
            genotype_counts["other"] += 1
            
    return genotype_counts

def process_chromosome(
    chromosome: str,
    germline_vcf: str,
    denovo_vcf: str,
    min_consecutive: int,
    max_distance: int
) -> Tuple[str, Dict]:
    """
    Process a single chromosome to find consecutive and isolated de novo SNVs.
    
    Args:
        chromosome: Chromosome to process
        germline_vcf: Path to germline VCF file
        denovo_vcf: Path to de novo VCF file
        min_consecutive: Minimum number of consecutive de novo SNVs required
        max_distance: Maximum distance between consecutive SNVs to be considered consecutive
        
    Returns:
        Tuple of (chromosome, dict with consecutive and isolated variants)
    """
    try:
        print(f"Processing chromosome {chromosome}...")
        
        # Extract positions from both VCFs
        germline_positions = extract_positions(germline_vcf, chromosome)
        denovo_positions = extract_positions(denovo_vcf, chromosome)
        
        print(f"  Chromosome {chromosome}: {len(germline_positions)} germline and {len(denovo_positions)} de novo SNVs")
        
        # Find consecutive de novo SNVs
        consecutive_ranges = find_consecutive_denovo(
            denovo_positions, germline_positions, min_consecutive, max_distance
        )
        
        print(f"  Found {len(consecutive_ranges)} consecutive de novo ranges in chromosome {chromosome}")
        
        # Extract full variant information for these ranges
        consecutive_variants = extract_variants_from_ranges(denovo_vcf, chromosome, consecutive_ranges)
        
        # Get all consecutive positions
        consecutive_positions = set()
        for start_pos, end_pos in consecutive_ranges:
            consecutive_positions.update(range(start_pos, end_pos + 1))
        
        # Extract isolated variants that are not in consecutive runs
        isolated_variants = extract_isolated_variants(denovo_vcf, chromosome, consecutive_positions)
        
        print(f"  Extracted {len(consecutive_variants)} consecutive and {len(isolated_variants)} isolated variants in chromosome {chromosome}")
        
        # Get genotype counts for reporting
        consecutive_genotypes = get_genotype_counts(consecutive_variants)
        isolated_genotypes = get_genotype_counts(isolated_variants)
        
        return (chromosome, {
            'consecutive_variants': consecutive_variants,
            'isolated_variants': isolated_variants,
            'consecutive_genotypes': consecutive_genotypes,
            'isolated_genotypes': isolated_genotypes,
            'total_denovo': len(denovo_positions),
            'consecutive_count': len(consecutive_variants),
            'isolated_count': len(isolated_variants)
        })
    
    except Exception as e:
        print(f"Error processing chromosome {chromosome}: {e}")
        return (chromosome, {
            'consecutive_variants': [],
            'isolated_variants': [],
            'consecutive_genotypes': {"0/1": 0, "1/1": 0, "other": 0},
            'isolated_genotypes': {"0/1": 0, "1/1": 0, "other": 0},
            'total_denovo': 0,
            'consecutive_count': 0,
            'isolated_count': 0,
            'error': str(e)
        })

def write_variants_to_vcf(
    variants: List[Dict],
    template_vcf: str,
    output_vcf: str
):
    """
    Write variants to a new VCF file using a template for the header.
    
    Args:
        variants: List of variant dictionaries
        template_vcf: VCF file to use as template for header
        output_vcf: Path to write the output VCF
    """
    # Create a temporary uncompressed VCF
    temp_vcf = output_vcf.replace('.gz', '')
    
    try:
        # Copy header from template
        with pysam.VariantFile(template_vcf) as template:
            with pysam.VariantFile(temp_vcf, 'w', header=template.header) as out_vcf:
                # Sort variants by chromosome and position
                sorted_variants = sorted(variants, key=lambda v: (v['chrom'], v['pos']))
                
                # Write each variant
                for variant in sorted_variants:
                    # Create a new record
                    record = out_vcf.new_record(
                        contig=variant['chrom'],
                        start=variant['pos'] - 1,  # 0-based
                        alleles=(variant['ref'], variant['alt']),
                        id=variant['id'],
                        qual=variant['qual']
                    )
                    
                    # Set filters
                    if variant['filter']:
                        for filt in variant['filter']:
                            record.filter.add(filt)
                    
                    # Set INFO fields
                    for key, value in variant['info'].items():
                        record.info[key] = value
                    
                    # Set FORMAT and sample fields
                    for sample_name, sample_data in variant['samples'].items():
                        for key, value in sample_data.items():
                            record.samples[sample_name][key] = value
                    
                    out_vcf.write(record)
        
        # Compress and index the output VCF
        run_command(f"{PATH_CONFIG['BGZIP']} -f {temp_vcf}")
        run_command(f"{PATH_CONFIG['TABIX']} -p vcf {output_vcf}")
        
        print(f"Successfully wrote {len(variants)} variants to {output_vcf}")
        
    except Exception as e:
        print(f"Error writing variants to VCF: {e}")
        raise
    finally:
        # Clean up temporary file if it exists
        if os.path.exists(temp_vcf):
            os.remove(temp_vcf)

def visualize_chromosome_ratios(results: Dict, output_dir: str, dataset: str, section_id: str):
    """
    Create visualizations of the consecutive de novo ratios by chromosome.
    
    Args:
        results: Dictionary mapping chromosome to processing results
        output_dir: Directory to save visualizations
        dataset: Dataset name
        section_id: Section ID
    """
    # Extract data for plotting
    chromosomes = []
    consecutive_ratios = []
    isolated_ratios = []
    consecutive_het_ratios = []
    consecutive_hom_ratios = []
    
    for chrom in results:
        if chrom == 'combined':
            continue  # Skip combined results
            
        data = results[chrom]
        total_denovo = data['total_denovo']
        
        if total_denovo == 0:
            continue  # Skip chromosomes with no de novo variants
            
        chromosomes.append(chrom)
        
        # Calculate consecutive and isolated ratios
        consecutive_ratio = data['consecutive_count'] / total_denovo if total_denovo > 0 else 0
        isolated_ratio = data['isolated_count'] / total_denovo if total_denovo > 0 else 0
        
        consecutive_ratios.append(consecutive_ratio)
        isolated_ratios.append(isolated_ratio)
        
        # Calculate heterozygous and homozygous ratios within consecutive variants
        consecutive_total = data['consecutive_genotypes']['0/1'] + data['consecutive_genotypes']['1/1']
        het_ratio = data['consecutive_genotypes']['0/1'] / consecutive_total if consecutive_total > 0 else 0
        hom_ratio = data['consecutive_genotypes']['1/1'] / consecutive_total if consecutive_total > 0 else 0
        
        consecutive_het_ratios.append(het_ratio)
        consecutive_hom_ratios.append(hom_ratio)
    
    # Create directory for visualizations
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Plot 1: Consecutive vs Isolated Ratio by Chromosome
    plt.figure(figsize=(14, 8))
    x = np.arange(len(chromosomes))
    width = 0.35
    
    plt.bar(x - width/2, consecutive_ratios, width, label='Consecutive De Novo')
    plt.bar(x + width/2, isolated_ratios, width, label='Isolated De Novo')
    
    plt.xlabel('Chromosome')
    plt.ylabel('Ratio of Total De Novo SNVs')
    plt.title(f'Ratio of Consecutive vs Isolated De Novo SNVs by Chromosome\n{dataset} {section_id}')
    plt.xticks(x, chromosomes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(consecutive_ratios):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(isolated_ratios):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'consecutive_vs_isolated_ratio.png'), dpi=300)
    plt.close()
    
    # Plot 2: Heterozygous vs Homozygous Ratio in Consecutive De Novo SNVs
    plt.figure(figsize=(14, 8))
    
    plt.bar(x - width/2, consecutive_het_ratios, width, label='Heterozygous (0/1)')
    plt.bar(x + width/2, consecutive_hom_ratios, width, label='Homozygous (1/1)')
    
    plt.xlabel('Chromosome')
    plt.ylabel('Ratio within Consecutive De Novo SNVs')
    plt.title(f'Genotype Distribution in Consecutive De Novo SNVs by Chromosome\n{dataset} {section_id}')
    plt.xticks(x, chromosomes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(consecutive_het_ratios):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    for i, v in enumerate(consecutive_hom_ratios):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'consecutive_genotype_ratio.png'), dpi=300)
    plt.close()
    
    # Plot 3: Total consecutive and isolated counts by chromosome
    plt.figure(figsize=(14, 8))
    
    consecutive_counts = [results[chrom]['consecutive_count'] for chrom in chromosomes]
    isolated_counts = [results[chrom]['isolated_count'] for chrom in chromosomes]
    
    plt.bar(x - width/2, consecutive_counts, width, label='Consecutive De Novo')
    plt.bar(x + width/2, isolated_counts, width, label='Isolated De Novo')
    
    plt.xlabel('Chromosome')
    plt.ylabel('Number of SNVs')
    plt.title(f'Count of Consecutive vs Isolated De Novo SNVs by Chromosome\n{dataset} {section_id}')
    plt.xticks(x, chromosomes, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, v in enumerate(consecutive_counts):
        plt.text(i - width/2, v + 1, str(v), ha='center', fontsize=8)
    for i, v in enumerate(isolated_counts):
        plt.text(i + width/2, v + 1, str(v), ha='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'consecutive_vs_isolated_counts.png'), dpi=300)
    plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def setup_paths(dataset: str, section_id: str, quality_filter: str) -> Dict[str, str]:
    """
    Set up paths for VCF files based on dataset configuration.
    
    Args:
        dataset: Dataset name (e.g., 'DLPFC', 'P4_TUMOR', 'P6_TUMOR')
        section_id: Section ID (required for some datasets)
        quality_filter: Quality filter (e.g., 'baseQ0mapQ0')
        
    Returns:
        Dictionary of paths for germline, de novo, and output VCFs
    """
    # Validate dataset
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset}")
        
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[dataset]
    
    # Validate section ID if required
    if dataset_config["has_sections"] and not section_id:
        raise ValueError(f"Dataset {dataset} requires a section_id")
    
    # Check specific section IDs if defined
    if dataset_config.get("section_ids") and section_id not in dataset_config["section_ids"]:
        raise ValueError(f"Invalid section_id for {dataset}. Valid section IDs are: {dataset_config['section_ids']}")
    
    # Set base output directory
    if dataset_config["has_sections"]:
        output_base = os.path.join(
            PATH_CONFIG["PROJECT_DIR"],
            dataset_config["output_dir"].format(section_id=section_id)
        )
    else:
        output_base = os.path.join(
            PATH_CONFIG["PROJECT_DIR"],
            dataset_config["output_dir"]
        )
    
    # Set VCF paths
    germline_vcf = os.path.join(
        output_base, 
        "output_VCFs/beagle",
        quality_filter, 
        "all_filtered_in.vcf.gz"
    )
    
    denovo_vcf = os.path.join(
        output_base, 
        "output_VCFs/beagle",
        quality_filter, 
        "all_filtered_out.vcf.gz"
    )
    
    # Output directory for consecutive de novo SNVs
    output_dir = os.path.join(
        output_base, 
        "output_VCFs/SeqErrModel",
        quality_filter
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Output VCF paths for consecutive and isolated variants
    consecutive_vcf = os.path.join(
        output_dir,
        "consecutive_denovo.vcf.gz"
    )
    
    isolated_vcf = os.path.join(
        output_dir,
        "isolated_denovo.vcf.gz"
    )
    
    return {
        "germline_vcf": germline_vcf,
        "denovo_vcf": denovo_vcf,
        "consecutive_vcf": consecutive_vcf,
        "isolated_vcf": isolated_vcf,
        "output_dir": output_dir
    }

def compare_with_gatk_normal(output_vcf, gatk_normal_vcf):
    """
    Compare the identified consecutive de novo variants with GATK normal VCF
    to validate results.
    
    Args:
        output_vcf: Path to the output VCF with consecutive de novo variants
        gatk_normal_vcf: Path to the GATK normal VCF used as ground truth
    """
    # Setup environment for bcftools
    setup_environment()
    
    # Create output directory for intersection results
    overlap_dir = os.path.join(os.path.dirname(output_vcf), "overlap_denovo_gatk_normal")
    os.makedirs(overlap_dir, exist_ok=True)
    
    # Run bcftools isec to identify overlapping variants
    cmd = (
        f"{PATH_CONFIG['APPS_DIR']}/bcftools isec -n=2 -w1 -O z -p {overlap_dir} "
        f"{output_vcf} {gatk_normal_vcf}"
    )
    
    print(f"Comparing variants with GATK normal VCF...")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              universal_newlines=True)
        
        # Check if overlap files were created
        overlap_vcf = os.path.join(overlap_dir, "0000.vcf.gz")
        if os.path.exists(overlap_vcf):
            # Count variants in original and overlap files
            orig_count = count_variants(output_vcf)
            overlap_count = count_variants(overlap_vcf)
            
            print(f"\nValidation Results:")
            print(f"Total consecutive de novo variants: {orig_count}")
            print(f"Variants also in GATK normal: {overlap_count}")
            print(f"Percentage in GATK normal: {overlap_count/max(1, orig_count)*100:.2f}%")
            
            # Write summary to file
            summary_file = os.path.join(overlap_dir, "validation_summary.txt")
            with open(summary_file, 'w') as f:
                f.write(f"Validation Results:\n")
                f.write(f"Total de novo variants: {orig_count}\n")
                f.write(f"Variants also in GATK normal: {overlap_count}\n")
                f.write(f"Percentage in GATK normal: {overlap_count/max(1, orig_count)*100:.2f}%\n")
            
            return overlap_vcf, overlap_count, orig_count
            
    except subprocess.CalledProcessError as e:
        print(f"Error comparing with GATK normal: {e}")
        print(f"stderr: {e.stderr}")
        return None, 0, 0

def count_variants(vcf_path):
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

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Find consecutive de novo SNVs for negative training set")
    
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0",
                      help="Quality filter to use (default: baseQ0mapQ0)")
    parser.add_argument("--min-consecutive", type=int, default=7,
                      help="Minimum number of consecutive de novo SNVs required (default: 3)")
    parser.add_argument("--max-distance", type=int, default=700,
                      help="Maximum distance between SNVs to be considered consecutive (default: 1000)")
    parser.add_argument("--threads", type=int, default=24,
                      help="Number of threads to use (default: 8)")
    parser.add_argument("--gatk-normal-vcf", default="/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz",
                      help="Path to GATK normal VCF for validation")

    args = parser.parse_args()
    
    # Setup the environment
    setup_environment()
    
    # Setup paths based on dataset configuration
    try:
        paths = setup_paths(args.dataset, args.section_id, args.quality_filter)
        print(f"Processing dataset: {args.dataset}")
        print(f"Section ID: {args.section_id}")
        print(f"Quality filter: {args.quality_filter}")
        print(f"Germline VCF: {paths['germline_vcf']}")
        print(f"De novo VCF: {paths['denovo_vcf']}")
        print(f"Consecutive De Novo VCF: {paths['consecutive_vcf']}")
        print(f"Isolated De Novo VCF: {paths['isolated_vcf']}")
    except ValueError as e:
        print(f"Error setting up paths: {e}")
        sys.exit(1)
    
    # Validate input files
    for vcf_file in [paths['germline_vcf'], paths['denovo_vcf']]:
        if not os.path.exists(vcf_file):
            print(f"Error: VCF file not found: {vcf_file}")
            sys.exit(1)
    
    # Get list of chromosomes to process
    try:
        chromosomes = get_chromosomes_from_vcf(paths['denovo_vcf'])
        print(f"Found {len(chromosomes)} chromosomes: {', '.join(chromosomes[:5])}...")
    except Exception as e:
        print(f"Error getting chromosomes: {e}")
        sys.exit(1)
    
    # Process each chromosome in parallel
    all_consecutive_variants = []
    all_isolated_variants = []
    chromosome_results = {}
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                process_chromosome,
                chromosome,
                paths['germline_vcf'],
                paths['denovo_vcf'],
                args.min_consecutive,
                args.max_distance
            ): chromosome
            for chromosome in chromosomes
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing chromosomes"):
            chromosome = futures[future]
            try:
                chrom, result = future.result()
                all_consecutive_variants.extend(result['consecutive_variants'])
                all_isolated_variants.extend(result['isolated_variants'])
                chromosome_results[chrom] = result
                print(f"Completed chromosome {chromosome}: found {len(result['consecutive_variants'])} consecutive and {len(result['isolated_variants'])} isolated variants")
            except Exception as e:
                print(f"Error processing chromosome {chromosome}: {e}")
                chromosome_results[chromosome] = {
                    'consecutive_variants': [],
                    'isolated_variants': [],
                    'consecutive_genotypes': {"0/1": 0, "1/1": 0, "other": 0},
                    'isolated_genotypes': {"0/1": 0, "1/1": 0, "other": 0},
                    'total_denovo': 0,
                    'consecutive_count': 0,
                    'isolated_count': 0,
                    'error': str(e)
                }
    
    # Calculate combined statistics
    consecutive_0_1 = sum(result['consecutive_genotypes']['0/1'] for result in chromosome_results.values())
    consecutive_1_1 = sum(result['consecutive_genotypes']['1/1'] for result in chromosome_results.values())
    isolated_0_1 = sum(result['isolated_genotypes']['0/1'] for result in chromosome_results.values())
    isolated_1_1 = sum(result['isolated_genotypes']['1/1'] for result in chromosome_results.values())
    
    # Add combined statistics to results
    chromosome_results['combined'] = {
        'consecutive_count': len(all_consecutive_variants),
        'isolated_count': len(all_isolated_variants),
        'consecutive_genotypes': {
            '0/1': consecutive_0_1,
            '1/1': consecutive_1_1,
            'other': 0
        },
        'isolated_genotypes': {
            '0/1': isolated_0_1,
            '1/1': isolated_1_1,
            'other': 0
        },
        'total_denovo': len(all_consecutive_variants) + len(all_isolated_variants)
    }
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total consecutive de novo variants: {len(all_consecutive_variants)}")
    print(f"  Heterozygous (0/1): {consecutive_0_1}")
    print(f"  Homozygous (1/1): {consecutive_1_1}")
    print(f"Total isolated de novo variants: {len(all_isolated_variants)}")
    print(f"  Heterozygous (0/1): {isolated_0_1}")
    print(f"  Homozygous (1/1): {isolated_1_1}")
    
    # Write consecutive variants to output VCF
    if all_consecutive_variants:
        print(f"\nWriting {len(all_consecutive_variants)} consecutive de novo variants to {paths['consecutive_vcf']}")
        write_variants_to_vcf(all_consecutive_variants, paths['denovo_vcf'], paths['consecutive_vcf'])
        if args.gatk_normal_vcf:
            if os.path.exists(args.gatk_normal_vcf):
                compare_with_gatk_normal(paths['consecutive_vcf'], args.gatk_normal_vcf)
            else:
                print(f"Warning: GATK normal VCF not found: {args.gatk_normal_vcf}")
    else:
        print("\nNo consecutive de novo SNVs found! Check your parameters or input files.")
    
    # Write isolated variants to output VCF
    if all_isolated_variants:
        print(f"Writing {len(all_isolated_variants)} isolated de novo variants to {paths['isolated_vcf']}")
        write_variants_to_vcf(all_isolated_variants, paths['denovo_vcf'], paths['isolated_vcf'])
        if args.gatk_normal_vcf:
            if os.path.exists(args.gatk_normal_vcf):
                compare_with_gatk_normal(paths['isolated_vcf'], args.gatk_normal_vcf)
            else:
                print(f"Warning: GATK normal VCF not found: {args.gatk_normal_vcf}")
    else:
        print("No consecutive de novo SNVs found! Check your parameters or input files.")
    
    # Create visualizations
    visualize_chromosome_ratios(chromosome_results, paths['output_dir'], args.dataset, args.section_id)
    
    # Write summary to file
    summary_file = os.path.join(paths['output_dir'], 'denovo_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"De Novo SNV Analysis Summary\n")
        f.write(f"========================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Section ID: {args.section_id}\n")
        f.write(f"Quality Filter: {args.quality_filter}\n")
        f.write(f"Min Consecutive: {args.min_consecutive}\n")
        f.write(f"Max Distance: {args.max_distance}\n\n")
        
        f.write(f"Total consecutive de novo variants: {len(all_consecutive_variants)}\n")
        f.write(f"  Heterozygous (0/1): {consecutive_0_1}\n")
        f.write(f"  Homozygous (1/1): {consecutive_1_1}\n")
        f.write(f"Total isolated de novo variants: {len(all_isolated_variants)}\n")
        f.write(f"  Heterozygous (0/1): {isolated_0_1}\n")
        f.write(f"  Homozygous (1/1): {isolated_1_1}\n\n")
        
        f.write("Chromosome Statistics:\n")
        for chrom, result in sorted(chromosome_results.items()):
            if chrom == 'combined':
                continue
            f.write(f"  {chrom}:\n")
            f.write(f"    Total de novo SNVs: {result['total_denovo']}\n")
            f.write(f"    Consecutive SNVs: {result['consecutive_count']} ")
            f.write(f"(0/1: {result['consecutive_genotypes']['0/1']}, 1/1: {result['consecutive_genotypes']['1/1']})\n")
            f.write(f"    Isolated SNVs: {result['isolated_count']} ")
            f.write(f"(0/1: {result['isolated_genotypes']['0/1']}, 1/1: {result['isolated_genotypes']['1/1']})\n")
    
    print(f"\nAnalysis complete. Results saved to {paths['output_dir']}")
    print(f"Summary file: {summary_file}")

if __name__ == "__main__":
    main()

# P4 sec1 baseq0mapq0:
# python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ0mapQ0 --min-consecutive 3 --max-distance 2000 --threads 24