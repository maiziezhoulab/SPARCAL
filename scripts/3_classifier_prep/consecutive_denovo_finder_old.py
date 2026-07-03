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
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
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
        # Add to current run if it's close enough
        if not current_run or (pos - current_run[-1] <= max_distance):
            current_run.append(pos)
        else:
            # Process completed run
            if len(current_run) >= min_consecutive:
                start_pos = current_run[0]
                end_pos = current_run[-1]
                
                # Fast-forward through germline positions until we reach the region
                while germline_idx < len(sorted_germline) and sorted_germline[germline_idx] < start_pos:
                    germline_idx += 1
                
                # Check if any germline positions fall within this region
                has_germline = False
                temp_idx = germline_idx
                while temp_idx < len(sorted_germline) and sorted_germline[temp_idx] <= end_pos:
                    has_germline = True
                    break
                
                # Only add run if no germline variants in region
                if not has_germline:
                    consecutive_ranges.append((start_pos, end_pos))
            
            # Start a new run
            current_run = [pos]
    
    # Process the last run
    if len(current_run) >= min_consecutive:
        start_pos = current_run[0]
        end_pos = current_run[-1]
        
        # Check for germline variants in region (same as above)
        has_germline = False
        temp_idx = germline_idx
        while temp_idx < len(sorted_germline) and sorted_germline[temp_idx] <= end_pos:
            has_germline = True
            break
        
        if not has_germline:
            consecutive_ranges.append((start_pos, end_pos))
    
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

def process_chromosome(
    chromosome: str,
    germline_vcf: str,
    denovo_vcf: str,
    min_consecutive: int,
    max_distance: int
) -> Tuple[str, List[Dict]]:
    """
    Process a single chromosome to find consecutive de novo SNVs.
    
    Args:
        chromosome: Chromosome to process
        germline_vcf: Path to germline VCF file
        denovo_vcf: Path to de novo VCF file
        min_consecutive: Minimum number of consecutive de novo SNVs required
        max_distance: Maximum distance between consecutive SNVs to be considered consecutive
        
    Returns:
        Tuple of (chromosome, list of variant dictionaries)
    """
    try:
        print(f"Processing chromosome {chromosome}...")
        
        # Extract positions from both VCFs
        germline_positions = extract_positions(germline_vcf, chromosome)
        denovo_positions = list(extract_positions(denovo_vcf, chromosome))
        
        print(f"  Chromosome {chromosome}: {len(germline_positions)} germline and {len(denovo_positions)} de novo SNVs")
        
        # Find consecutive de novo SNVs
        consecutive_ranges = find_consecutive_denovo(
            denovo_positions, germline_positions, min_consecutive, max_distance
        )
        
        print(f"  Found {len(consecutive_ranges)} consecutive de novo ranges in chromosome {chromosome}")
        
        # Extract full variant information for these ranges
        variants = extract_variants_from_ranges(denovo_vcf, chromosome, consecutive_ranges)
        
        print(f"  Extracted {len(variants)} variants from consecutive ranges in chromosome {chromosome}")
        
        return (chromosome, variants)
    
    except Exception as e:
        print(f"Error processing chromosome {chromosome}: {e}")
        return (chromosome, [])

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
    
    output_vcf = os.path.join(
        output_dir,
        "consecutive_denovo.vcf.gz"
    )
    
    return {
        "germline_vcf": germline_vcf,
        "denovo_vcf": denovo_vcf,
        "output_vcf": output_vcf,
        "output_dir": output_dir
    }

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Find consecutive de novo SNVs for negative training set")
    
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--quality_filter", default="baseQ0mapQ0",
                      help="Quality filter to use (default: baseQ0mapQ0)")
    parser.add_argument("--min-consecutive", type=int, default=3,
                      help="Minimum number of consecutive de novo SNVs required (default: 3)")
    parser.add_argument("--max-distance", type=int, default=1000,
                      help="Maximum distance between SNVs to be considered consecutive (default: 1000)")
    parser.add_argument("--threads", type=int, default=24,
                      help="Number of threads to use (default: 8)")
    
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
        print(f"Output VCF: {paths['output_vcf']}")
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
    all_variants = []
    
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
                _, variants = future.result()
                all_variants.extend(variants)
                print(f"Completed chromosome {chromosome}: found {len(variants)} variants")
            except Exception as e:
                print(f"Error processing chromosome {chromosome}: {e}")
    
    # Write all variants to output VCF
    if all_variants:
        print(f"Writing {len(all_variants)} variants to {paths['output_vcf']}")
        write_variants_to_vcf(all_variants, paths['denovo_vcf'], paths['output_vcf'])
        print(f"Successfully created negative training set with {len(all_variants)} variants")
    else:
        print("No consecutive de novo SNVs found! Check your parameters or input files.")

if __name__ == "__main__":
    main()

# P4 sec1 baseq0mapq0:
# python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ0mapQ0