#!/usr/bin/env python3
"""
Validate Beagle output by comparing with 1000 Genomes allele frequencies.

This script analyzes variants called by Beagle to determine what proportion
are common variants (AF ≥ 0.01) in the 1000 Genomes Project data.
"""

import os
import sys
import gzip
import argparse
from pathlib import Path
import glob
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration dictionaries
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
    "TUMOR": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",
        "regions": [f"chr{i}" for i in range(1, 23)]
    }
}

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "10X_BC_6.5MM": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE",
        "output_dir": "data/10X_BC_6.5mm",
        "has_sections": False,
        "reference": "FFPE_VISIUM"
    },
    "10X_BC_FFPE": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_Ductal_Carcinoma_In_Situ_Invasive_Carcinoma_FFPE",
        "output_dir": "data/10X_BC_FFPE",
        "has_sections": False,
        "reference": "FFPE_VISIUM"
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


def extract_format_field(format_str: str, sample_str: str, field_name: str) -> Optional[str]:
    """Extract a specific field from VCF FORMAT column."""
    try:
        format_fields = format_str.split(':')
        if field_name not in format_fields:
            return None
        
        field_idx = format_fields.index(field_name)
        value_fields = sample_str.split(':')
        
        if field_idx >= len(value_fields):
            return None
            
        return value_fields[field_idx]
    except (ValueError, IndexError):
        return None


def extract_info_field(info_str: str, field_name: str) -> Optional[str]:
    """Extract a specific field from VCF INFO column."""
    for field in info_str.split(';'):
        if field.startswith(f"{field_name}="):
            return field.split('=')[1]
    return None


def parse_vcf_variants(vcf_path: str, require_pass: bool = True) -> Dict[str, Dict]:
    """
    Parse variants from a VCF file.
    
    Args:
        vcf_path: Path to the VCF file
        require_pass: Only include variants that PASS filters
        
    Returns:
        Dict mapping variant keys (CHROM_POS) to variant info
    """
    variants = {}
    
    # Check if file exists
    if not os.path.exists(vcf_path):
        print(f"WARNING: VCF file not found: {vcf_path}")
        return variants
    
    # Open file with gzip if it's compressed
    opener = gzip.open if vcf_path.endswith('.gz') else open
    mode = 'rt' if vcf_path.endswith('.gz') else 'r'
    
    with opener(vcf_path, mode) as f:
        for line in f:
            # Skip header lines
            if line.startswith('#'):
                continue
            
            # Parse variant line
            fields = line.strip().split('\t')
            if len(fields) < 10:  # Need FORMAT and sample fields
                continue
                
            chrom = fields[0]
            pos = fields[1]
            ref = fields[3]
            alt = fields[4]
            filter_status = fields[6]
            info = fields[7]
            format_str = fields[8]
            sample_str = fields[9]
            
            # Skip variants that don't PASS filters if required
            if require_pass and filter_status != "PASS":
                continue
            
            # Create variant key by CHROM_POS
            variant_key = f"{chrom}_{pos}"
            
            # Extract genotype
            gt = extract_format_field(format_str, sample_str, 'GT')
            if not gt or gt == '0/0':  # Skip homozygous reference
                continue
                
            # Skip complex variants (multiple ALTs)
            if ',' in alt:
                continue
            
            # Store variant info
            variants[variant_key] = {
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'gt': gt,
                'filter': filter_status,
                'info': info
            }
    
    return variants


def load_1kg_common_variants(dataset_name: str, regions: List[str], af_threshold: float = 0.01, 
                            num_threads: int = 4, verbose: bool = False) -> Dict[str, Set[str]]:
    """
    Load common variants (AF >= threshold) from 1000 Genomes VCF files.
    
    Args:
        dataset_name: Dataset name to determine genome build
        regions: List of chromosome names to process
        af_threshold: Threshold for common variant (AF ≥ threshold)
        num_threads: Number of parallel threads to use
        verbose: Print detailed debugging information
        
    Returns:
        Dict of chromosome -> set of positions (as strings) for common variants
    """
    dataset_config = DATASET_CONFIGS[dataset_name]
    reference_name = dataset_config['reference']
    
    # Select appropriate 1000 Genomes configuration
    genome_build = "hg19" if reference_name == "TUMOR" else "GRCh38"
    genome_config = THOUSAND_GENOME_CONFIGS[genome_build]
    
    print(f"Loading 1000 Genomes data for {genome_build} using {num_threads} threads...")
    print(f"Looking for variants with AF >= {af_threshold}...")
    
    # Function to process a single chromosome
    def process_chrom(region):
        common_positions = set()
        example_variants = []
        
        # Format chromosome name correctly for file access
        if genome_build == "hg19":
            # For hg19, remove 'chr' prefix as 1000G files use just numbers
            chrom = region.replace('chr', '')
        else:
            # For GRCh38, keep chromosome format as is
            chrom = region
            
        reference_pattern = os.path.join(
            genome_config["base_path"],
            genome_config["pattern"].format(chrom=chrom)
        )
        
        if not os.path.exists(reference_pattern):
            print(f"WARNING: 1000 Genome reference not found: {reference_pattern}")
            return region, common_positions, example_variants, 0, 0
        
        # Parse 1000G VCF
        try:
            variants_count = 0
            common_count = 0
            with gzip.open(reference_pattern, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    variants_count += 1
                    fields = line.strip().split('\t')
                    
                    # Standardize chromosome name to always have "chr" prefix
                    chrom_formatted = fields[0]
                    if not chrom_formatted.startswith('chr'):
                        chrom_formatted = f"chr{chrom_formatted}"
                        
                    pos = fields[1]
                    ref = fields[3]
                    alt = fields[4]
                    info = fields[7]
                    
                    # Extract basic AF from INFO field
                    af_str = None
                    for field in info.split(';'):
                        if field.startswith('AF='):
                            af_str = field.split('=')[1]
                            break
                    
                    if af_str:
                        try:
                            # Handle multi-allelic sites - just use the first AF
                            af_values = [float(x) for x in af_str.split(',')]
                            af = af_values[0]
                            
                            # Collect a few examples for debugging
                            if len(example_variants) < 3 and af >= af_threshold:
                                example_variants.append({
                                    'chrom': chrom_formatted,
                                    'pos': pos,
                                    'ref': ref,
                                    'alt': alt,
                                    'af': af,
                                    'info': info[:100] + '...' if len(info) > 100 else info
                                })

                            # Add common variants to the set
                            if af >= af_threshold:
                                # Store just the position as a string for efficient lookup
                                common_positions.add(pos)
                                common_count += 1
                                
                            if common_count % 50000 == 0 and variants_count > 0:
                                print(f"  Progress on {region}: {common_count:,} common variants found out of {variants_count:,} processed")
                        except ValueError:
                            # Skip if AF conversion fails
                            continue
            
            return region, common_positions, example_variants, variants_count, common_count
            
        except Exception as e:
            print(f"ERROR processing chromosome {region}: {str(e)}")
            return region, set(), [], 0, 0
    
    # Process chromosomes in parallel - one thread per chromosome
    common_variants_by_chrom = {}
    example_variants = []
    total_variants = 0
    total_common = 0
    
    # Adjust thread count if more threads than chromosomes are requested
    effective_threads = min(num_threads, len(regions))
    if effective_threads < num_threads:
        print(f"Note: Reducing thread count from {num_threads} to {effective_threads} (one per chromosome)")
    
    print(f"Processing {len(regions)} chromosomes with {effective_threads} threads...")
    
    with ThreadPoolExecutor(max_workers=effective_threads) as executor:
        # Submit all chromosome processing tasks
        future_to_chrom = {executor.submit(process_chrom, region): region for region in regions}
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_chrom), total=len(regions), desc="Processing chromosomes"):
            chrom = future_to_chrom[future]
            result_chrom, chrom_common_positions, chrom_examples, chrom_count, chrom_common_count = future.result()
            
            # Store common positions by chromosome for efficient lookup
            common_variants_by_chrom[result_chrom] = chrom_common_positions
            
            example_variants.extend(chrom_examples[:2])  # Only take first 2 examples per chromosome
            total_variants += chrom_count
            total_common += chrom_common_count
            
            print(f"  Chromosome {chrom} completed: {chrom_common_count:,} common positions of {chrom_count:,} total ({chrom_common_count/chrom_count*100:.2f}%)")
    
    # Display some example common variants
    if example_variants and verbose:
        print("\nExample common variants from 1000 Genomes:")
        for i, var in enumerate(example_variants[:5], 1):  # Show up to 5 examples
            print(f"Example {i}:")
            print(f"  Chromosome: {var['chrom']}")
            print(f"  Position: {var['pos']}")
            print(f"  Reference: {var['ref']} -> {var['alt']}")
            print(f"  AF: {var['af']}")
            print(f"  INFO: {var['info']}")
            print()
    
    total_positions = sum(len(positions) for positions in common_variants_by_chrom.values())
    print(f"Found {total_positions:,} common variant positions (AF >= {af_threshold}) out of {total_variants:,} total variants")
    print(f"Overall common variant percentage: {total_common/total_variants*100:.2f}%")
    
    return common_variants_by_chrom


def analyze_beagle_variants(dataset_name: str, section_id: str = None, 
                          quality_filter: str = "baseQ0mapQ0",
                          af_threshold: float = 0.01,
                          num_threads: int = 4,
                          verbose: bool = False) -> Dict:
    """
    Analyze Beagle variants to determine how many are common in 1000 Genomes.
    
    Args:
        dataset_name: Name of the dataset
        section_id: Section ID (if applicable)
        quality_filter: Quality filter used
        af_threshold: Threshold for common variant (AF ≥ threshold)
        num_threads: Number of parallel threads to use
        verbose: Print detailed debugging information
    
    Returns:
        Dict with analysis results
    """
    # Setup paths
    dataset_config = DATASET_CONFIGS[dataset_name]
    reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
    
    if dataset_config["has_sections"]:
        if not section_id:
            raise ValueError(f"Dataset {dataset_name} requires a section_id")
        data_path = dataset_config["output_dir"].format(section_id=section_id)
    else:
        data_path = dataset_config["output_dir"]
    
    base_dir = f"/data/maiziezhou_lab/yuqi/snv_calling"
    input_base = os.path.join(base_dir, data_path)
    
    # First, load common variants from 1000G - organized by chromosome
    common_variants_by_chrom = load_1kg_common_variants(
        dataset_name, 
        reference_config['regions'],
        af_threshold,
        num_threads,
        verbose
    )
    
    # Process each chromosome's Beagle output
    print(f"\nAnalyzing individual chromosome Beagle outputs...")
    
    # Setup output directory for VCF files
    beagle_output_dir = os.path.join(
        input_base, "output_VCFs/beagle",
        quality_filter
    )
    
    if not os.path.exists(beagle_output_dir):
        print(f"ERROR: Beagle output directory not found: {beagle_output_dir}")
        return {
            'dataset': dataset_name,
            'section_id': section_id,
            'quality_filter': quality_filter,
            'beagle_variants': 0,
            'common_in_1kg': 0,
            'common_ratio': 0,
            'af_threshold': af_threshold
        }
    
    # Function to process a single chromosome file using more efficient double-pointer approach
    def process_chrom_vcf(chrom):
        beagle_vcf = os.path.join(beagle_output_dir, f"{chrom}.vcf.gz")
        
        if not os.path.exists(beagle_vcf):
            print(f"WARNING: Beagle VCF not found for chromosome {chrom}: {beagle_vcf}")
            return chrom, 0, 0, 0
        
        # Get standardized chromosome name
        std_chrom = chrom
        if not std_chrom.startswith('chr'):
            std_chrom = f"chr{std_chrom}"
            
        # Get 1000G common positions for this chromosome
        # For chromosomes without "chr" prefix in keys, try both formats
        if std_chrom in common_variants_by_chrom:
            common_positions = common_variants_by_chrom[std_chrom]
        elif chrom in common_variants_by_chrom:  # Try without "chr" prefix
            common_positions = common_variants_by_chrom[chrom]
        else:
            print(f"WARNING: No common variant data for chromosome {chrom}")
            return chrom, 0, 0, 0
        
        # Process Beagle VCF
        beagle_total = 0
        beagle_common = 0
        
        # Extract positions from Beagle VCF file
        with gzip.open(beagle_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                    
                fields = line.strip().split('\t')
                
                try:
                    pos = fields[1]
                    beagle_total += 1
                    
                    # Check if position is in common variants
                    if pos in common_positions:
                        beagle_common += 1
                except Exception as e:
                    if verbose:
                        print(f"Warning: Error parsing line in {chrom}: {line.strip()}")
                        print(f"Error details: {str(e)}")
                    continue
        
        # Calculate common ratio
        common_ratio = beagle_common / beagle_total * 100 if beagle_total else 0
        
        return chrom, beagle_total, beagle_common, common_ratio
    
    # Process all chromosomes in parallel
    chrom_results = []
    total_variants = 0
    total_common = 0
    
    regions = reference_config['regions']
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_chrom = {executor.submit(process_chrom_vcf, chrom): chrom for chrom in regions}
        
        for future in tqdm(as_completed(future_to_chrom), total=len(regions), desc="Processing chromosomes"):
            chrom = future_to_chrom[future]
            try:
                result = future.result()
                chrom_results.append(result)
                chrom, n_variants, n_common, ratio = result
                total_variants += n_variants
                total_common += n_common
                print(f"  Chromosome {chrom}: {n_common:,} common variants of {n_variants:,} total ({ratio:.2f}%)")
            except Exception as e:
                print(f"Error processing chromosome {chrom}: {str(e)}")
    
    # Also check the combined "all_filtered_in.vcf.gz" file if it exists
    all_filtered_vcf = os.path.join(beagle_output_dir, "all_filtered_in.vcf.gz")
    combined_variants = 0
    combined_common = 0
    combined_ratio = 0
    
    if os.path.exists(all_filtered_vcf):
        print("\nValidating combined filtered variants using efficient double-pointer approach...")
        
        # Process the combined VCF file
        # We'll organize the common variants by chromosome for more efficient lookup
        beagle_counts_by_chrom = defaultdict(int)  # Total variants by chromosome
        common_counts_by_chrom = defaultdict(int)  # Common variants by chromosome
        
        with gzip.open(all_filtered_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                
                fields = line.strip().split('\t')
                
                try:
                    vcf_chrom = fields[0]
                    # Standardize chromosome name
                    if not vcf_chrom.startswith('chr'):
                        vcf_chrom = f"chr{vcf_chrom}"
                    
                    pos = fields[1]
                    
                    # Increment total count for this chromosome
                    beagle_counts_by_chrom[vcf_chrom] += 1
                    
                    # Check if this position is common in 1000G
                    if vcf_chrom in common_variants_by_chrom and pos in common_variants_by_chrom[vcf_chrom]:
                        common_counts_by_chrom[vcf_chrom] += 1
                except Exception as e:
                    if verbose:
                        print(f"Warning: Error parsing line: {line.strip()}")
                        print(f"Error details: {str(e)}")
                    continue
        
        # Calculate totals from combined file
        combined_variants = sum(beagle_counts_by_chrom.values())
        combined_common = sum(common_counts_by_chrom.values())
        combined_ratio = combined_common / combined_variants * 100 if combined_variants else 0
        
        print(f"\nCombined VCF results:")
        print(f"Total Beagle variants: {combined_variants:,}")
        print(f"Common variants in 1000G: {combined_common:,}")
        print(f"Common variant ratio: {combined_ratio:.2f}%")
        
        # Verify if combined matches sum of individual chromosomes
        if total_variants != combined_variants:
            print(f"WARNING: Sum of chromosome variants ({total_variants:,}) doesn't match combined file ({combined_variants:,})")
    else:
        print("\nNote: Combined filtered VCF not found. Using sum of chromosome results.")
        combined_variants = total_variants
        combined_common = total_common
        combined_ratio = combined_common / combined_variants * 100 if combined_variants else 0
    
    # Calculate overall results
    print("\nBeagle Variant Analysis Results (using chromosome-by-chromosome analysis):")
    print(f"Total Beagle variants: {total_variants:,}")
    print(f"Common variants in 1000G: {total_common:,}")
    overall_ratio = total_common / total_variants * 100 if total_variants else 0
    print(f"Common variant ratio: {overall_ratio:.2f}%")
    
    # Return results dictionary
    return {
        'dataset': dataset_name,
        'section_id': section_id,
        'quality_filter': quality_filter,
        'beagle_variants': total_variants,
        'common_in_1kg': total_common,
        'common_ratio': overall_ratio,
        'af_threshold': af_threshold,
        'chrom_results': chrom_results,
        'combined_variants': combined_variants,
        'combined_common': combined_common,
        'combined_ratio': combined_ratio,
        'common_variants_by_chrom': common_variants_by_chrom,  # Include for potential further analysis
    }


def plot_af_distribution(results: Dict, output_dir: str):
    """
    Plot the distribution of allele frequencies.
    
    Args:
        results: Results dictionary from analyze_beagle_output
        output_dir: Directory to save the plot
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Create histogram
    bins = np.logspace(-4, 0, 50)  # Log-scale bins from 0.0001 to 1
    plt.hist(results['af_values'], bins=bins, alpha=0.7)
    
    # Add vertical line at threshold
    plt.axvline(x=results['af_threshold'], color='r', linestyle='--', 
                label=f'Threshold (AF = {results["af_threshold"]})')
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add labels and title
    plt.xlabel('Allele Frequency (log scale)')
    plt.ylabel('Number of Variants')
    
    dataset_name = results['dataset']
    if results['section_id']:
        dataset_name += f" section {results['section_id']}"
        
    plt.title(f'1000 Genomes Allele Frequency Distribution of Beagle Variants\n{dataset_name}')
    
    # Add summary text
    summary = (
        f"Total Beagle variants: {results['beagle_variants']:,}\n"
        f"Matched with 1000G: {results['matched_with_1kg']:,} ({results['matched_with_1kg']/results['beagle_variants']*100:.1f}%)\n"
        f"Common variants (AF ≥ {results['af_threshold']}): {results['common_variants']:,} "
        f"({results['common_variant_percentage']:.1f}%)\n"
        f"Rare variants (AF < {results['af_threshold']}): {results['rare_variants']:,} "
        f"({results['rare_variant_percentage']:.1f}%)"
    )
    
    plt.annotate(summary, xy=(0.05, 0.95), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='top', ha='left', fontsize=10)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"{results['dataset']}_{results['section_id'] or 'nosection'}_af_distribution.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"AF distribution plot saved to: {output_file}")
    
    # Also create a cumulative distribution plot
    plt.figure(figsize=(12, 8))
    
    # Sort AF values
    sorted_afs = np.sort(results['af_values'])
    
    # Calculate cumulative distribution
    y = np.arange(1, len(sorted_afs) + 1) / len(sorted_afs)
    
    # Plot cumulative distribution
    plt.plot(sorted_afs, y)
    
    # Add vertical line at threshold
    plt.axvline(x=results['af_threshold'], color='r', linestyle='--', 
                label=f'Threshold (AF = {results["af_threshold"]})')
    
    # Set log scale for x-axis
    plt.xscale('log')
    
    # Add labels and title
    plt.xlabel('Allele Frequency (log scale)')
    plt.ylabel('Cumulative Proportion')
    plt.title(f'Cumulative Distribution of 1000 Genomes Allele Frequencies\n{dataset_name}')
    
    # Add summary text
    plt.annotate(summary, xy=(0.05, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 va='bottom', ha='left', fontsize=10)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, f"{results['dataset']}_{results['section_id'] or 'nosection'}_af_cumulative.png")
    plt.savefig(output_file, dpi=300)
    plt.close()
    
    print(f"Cumulative AF distribution plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Validate Beagle output with 1000 Genomes Project data')
    parser.add_argument('--dataset', required=True, choices=list(DATASET_CONFIGS.keys()),
                      help='Dataset to analyze')
    parser.add_argument('--section-id', 
                      help='Section ID (required for some datasets)')
    parser.add_argument('--quality-filter', default='baseQ0mapQ0',
                      help='Quality filter to use (default: baseQ0mapQ0)')
    parser.add_argument('--af-threshold', type=float, default=0.01,
                      help='Threshold for common variants (AF ≥ threshold, default: 0.01)')
    parser.add_argument('--output-dir', default='beagle_validation_results',
                      help='Directory to save output files')
    parser.add_argument('--threads', type=int, default=22,
                      help='Number of parallel threads to use (default: 22)')
    parser.add_argument('--verbose', action='store_true',
                      help='Print detailed debugging information')
    parser.add_argument('--use-precomputed-snps', action='store_true',
                      help='Use precomputed SNP positions instead of parsing VCFs')
    parser.add_argument('--chromosome', 
                      help='Process a specific chromosome only')
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
    
    # If specific chromosome is requested, modify regions
    if args.chromosome:
        reference_config = REFERENCE_CONFIGS[dataset_config['reference']]
        chr_format = args.chromosome
        if not chr_format.startswith('chr') and reference_config['chr_prefix'] == 'chr':
            chr_format = f"chr{chr_format}"
        elif chr_format.startswith('chr') and reference_config['chr_prefix'] == '':
            chr_format = chr_format[3:]  # Remove "chr" prefix
            
        # Check if chromosome is valid
        if chr_format not in reference_config['regions']:
            parser.error(f"Invalid chromosome: {args.chromosome}. Valid chromosomes for {args.dataset}: {reference_config['regions']}")
            
        print(f"Processing only chromosome {chr_format}")
        # Override regions with just the requested chromosome
        reference_config['regions'] = [chr_format]
    
    # Run analysis
    print(f"Analyzing dataset {args.dataset} with AF threshold {args.af_threshold}")
    results = analyze_beagle_variants(
        args.dataset, 
        args.section_id, 
        args.quality_filter, 
        args.af_threshold,
        args.threads,
        args.verbose
    )
    
    # Save results to file
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create summary file path
    if args.section_id:
        if args.chromosome:
            summary_file = os.path.join(args.output_dir, f"{args.dataset}_{args.section_id}_{args.chromosome}_summary.txt")
        else:
            summary_file = os.path.join(args.output_dir, f"{args.dataset}_{args.section_id}_summary.txt")
    else:
        if args.chromosome:
            summary_file = os.path.join(args.output_dir, f"{args.dataset}_{args.chromosome}_summary.txt")
        else:
            summary_file = os.path.join(args.output_dir, f"{args.dataset}_summary.txt")
    
    # Write summary
    with open(summary_file, 'w') as f:
        f.write(f"Beagle Validation with 1000 Genomes Project\n")
        f.write(f"=========================================\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        if args.section_id:
            f.write(f"Section ID: {args.section_id}\n")
        f.write(f"Quality filter: {args.quality_filter}\n")
        f.write(f"Common variant threshold: AF ≥ {args.af_threshold}\n")
        f.write(f"Threads used: {args.threads}\n\n")
        
        f.write(f"Total Beagle variants: {results['beagle_variants']:,}\n")
        f.write(f"Common variants in 1000G: {results['common_in_1kg']:,} ({results['common_in_1kg']/results['beagle_variants']*100:.1f}%)\n")
        f.write(f"Common variant ratio: {results['common_ratio']:.2f}%\n\n")
        
        f.write(f"Chromosome-by-chromosome breakdown:\n")
        f.write(f"----------------------------------\n")
        for chrom, variants, common, ratio in results['chrom_results']:
            if variants > 0:
                f.write(f"{chrom}: {common:,}/{variants:,} common variants ({ratio:.2f}%)\n")
        
        if 'combined_variants' in results:
            f.write(f"\nCombined VCF file (all_filtered_in.vcf.gz):\n")
            f.write(f"Total variants: {results['combined_variants']:,}\n")
            f.write(f"Common variants: {results['combined_common']:,} ({results['combined_common']/results['combined_variants']*100:.1f}%)\n")
            f.write(f"Common variant ratio: {results['combined_ratio']:.2f}%\n")
    
    print(f"Results summary saved to: {summary_file}")
    
    # If a specific chromosome was processed, also generate a CSV with variant keys
    if args.chromosome:
        # Extract the chromosome's common variant keys
        chrom_format = args.chromosome
        if not chrom_format.startswith('chr'):
            chrom_format = f"chr{chrom_format}"
            
        # Get beagle variants for this chromosome
        beagle_chrom_vcf = os.path.join(
            f"/data/maiziezhou_lab/yuqi/snv_calling",
            dataset_config["output_dir"].format(section_id=args.section_id) if dataset_config["has_sections"] else dataset_config["output_dir"],
            "output_VCFs/beagle",
            args.quality_filter,
            f"{chrom_format}.vcf.gz"
        )
        
        if os.path.exists(beagle_chrom_vcf):
            chrom_variants_file = os.path.join(args.output_dir, f"{args.dataset}_{args.section_id}_{args.chromosome}_variants.csv")
            
            # Use the optimized approach to process the chromosome
            common_positions = set()
            if chrom_format in results['common_variants_by_chrom']:
                common_positions = results['common_variants_by_chrom'][chrom_format]
            
            with gzip.open(beagle_chrom_vcf, 'rt') as vcf, open(chrom_variants_file, 'w') as out:
                out.write("chromosome,position,ref,alt,in_1000g\n")
                
                for line in vcf:
                    if line.startswith('#'):
                        continue
                        
                    fields = line.strip().split('\t')
                    
                    try:
                        vcf_chrom = fields[0]
                        if not vcf_chrom.startswith('chr'):
                            vcf_chrom = f"chr{vcf_chrom}"
                        
                        pos = fields[1]
                        ref = fields[3]
                        alt = fields[4]
                        
                        is_common = pos in common_positions
                        
                        out.write(f"{vcf_chrom},{pos},{ref},{alt},{is_common}\n")
                    except:
                        continue
                        
            print(f"Variant details for chromosome {args.chromosome} saved to: {chrom_variants_file}")

if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC with 8 threads:
# python scripts/tools/Beagle_1kG_validation.py --dataset DLPFC --section-id 151507 --threads 8

# For P4_TUMOR with 12 threads:
# python scripts/tools/Beagle_1kG_validation.py --dataset P4_TUMOR --section-id 1 

# For 10X_BC_FFPE with 16 threads:
# python scripts/tools/Beagle_1kG_validation.py --dataset 10X_BC_FFPE --threads 16
``` 