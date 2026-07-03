#!/usr/bin/env python3
"""
SPARCAL Spatial Variant Classifier - Main Execution Script
Processes per-spot VCFs and classifies variants as somatic vs germline
"""

import argparse
import os
import sys
import gzip
import yaml
import logging
from pathlib import Path
from collections import defaultdict
import pysam
import pandas as pd
import numpy as np

# Import SPARCAL modules
from sparcal_modules import (
    CNAProfile,
    SpotMetadata,
    SPARCALFeatureEngineer,
    BayesianClassifier
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('SPARCAL')


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='SPARCAL: Spatial Variant Classification with CNA integration'
    )
    
    parser.add_argument(
        '--dataset',
        required=True,
        help='Dataset name (e.g., P4_TUMOR, DCIS)'
    )
    
    parser.add_argument(
        '--section',
        required=True,
        help='Section ID (e.g., P4_sec1, DCIS1)'
    )
    
    parser.add_argument(
        '--quality_filter',
        default='baseQ0mapQ0',
        help='Quality filter directory name (default: baseQ0mapQ0)'
    )
    
    parser.add_argument(
        '--chromosome',
        required=True,
        help='Chromosome to process (e.g., chr17)'
    )
    
    parser.add_argument(
        '--config',
        default=None,
        help='Path to config YAML file (default: auto-detect in run_slurm/{dataset}/)'
    )
    
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Output directory (default: auto-generated)'
    )
    
    parser.add_argument(
        '--base_dir',
        default='/data/maiziezhou_lab/leiy4/snv_calling',
        help='Base directory for SNV calling'
    )
    
    parser.add_argument(
        '--calicost_base',
        default='/data/maiziezhou_lab/leiy4/CalicoST',
        help='Base directory for CalicoST outputs'
    )
    
    return parser.parse_args()


def load_config(args):
    """Load configuration file"""
    if args.config:
        config_path = args.config
    else:
        # Auto-detect config
        config_path = os.path.join(
            args.base_dir, 
            'configs',        # ← NEW: changed from 'run_slurm' to 'configs'
            args.dataset,
            args.section,     # ← NEW: added section subdirectory
            'sparcal_config.yaml'
        )
    
    logger.info(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.error(f"Expected location: {args.base_dir}/configs/{args.dataset}/{args.section}/sparcal_config.yaml")  # ← NEW: helpful error message
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_paths(args):
    """Setup all necessary file paths"""
    paths = {}
    
    # Input directory with per-spot VCFs
    paths['vcf_dir'] = os.path.join(
        args.base_dir,
        args.dataset,
        args.section,
        'output_VCFs',
        'spotprofiles',
        args.quality_filter,
        'vcf_by_spot'
    )
    
    # CalicoST outputs
    calicost_dir = os.path.join(
        args.calicost_base,
        args.section,
        'calicost',
        'clone3_rectangle0_w1.0'
    )
    
    paths['cna_seg_file'] = os.path.join(calicost_dir, 'cnv_seglevel.tsv')
    paths['clone_file'] = os.path.join(calicost_dir, 'clone_labels.tsv')
    
    # Try to find spatial directory (common locations)
    spatial_candidates = [
        os.path.join(args.calicost_base, args.section, 'spatial'),
        os.path.join(args.base_dir, args.dataset, args.section, 'spatial'),
        calicost_dir  # Sometimes in same directory
    ]
    
    paths['spatial_dir'] = None
    for candidate in spatial_candidates:
        if os.path.exists(candidate):
            paths['spatial_dir'] = candidate
            break
    
    if paths['spatial_dir'] is None:
        logger.warning("Could not find spatial coordinates directory")
        paths['spatial_dir'] = calicost_dir  # Use as fallback
    
    # Output directory
    if args.output_dir:
        paths['output_dir'] = args.output_dir
    else:
        paths['output_dir'] = os.path.join(
            args.base_dir,
            args.dataset,
            args.section,
            'output_VCFs',
            'spotprofiles',
            args.quality_filter,
            'spatial_filtered'
        )
    
    # Create output directory
    os.makedirs(paths['output_dir'], exist_ok=True)
    
    # Output files
    chrom = args.chromosome
    paths['output_vcf'] = os.path.join(paths['output_dir'], f'{chrom}.classified.vcf.gz')
    paths['output_features'] = os.path.join(paths['output_dir'], f'{chrom}.features.tsv.gz')
    paths['output_summary'] = os.path.join(paths['output_dir'], f'{chrom}.summary.txt')
    
    return paths


def get_vcf_list(vcf_dir):
    """Get list of all per-spot VCF files"""
    vcf_files = []
    for fname in os.listdir(vcf_dir):
        if fname.endswith('.vcf.gz'):
            vcf_files.append(os.path.join(vcf_dir, fname))
    
    logger.info(f"Found {len(vcf_files)} per-spot VCF files")
    return vcf_files


def parse_vcf_info(info_str):
    """Parse VCF INFO field into dictionary"""
    info_dict = {}
    for item in info_str.split(';'):
        if '=' in item:
            key, value = item.split('=', 1)
            info_dict[key] = value
        else:
            info_dict[item] = True
    return info_dict


def extract_vaf_from_vcf(vcf_file, chrom, pos, ref, alt):
    """
    Extract VAF and depth for a variant from a VCF file
    
    Returns:
        (depth, vaf, ad_ref, ad_alt) or None if variant not found
    """
    try:
        vcf = pysam.VariantFile(vcf_file)
        
        # Look for variant
        for record in vcf.fetch(chrom, pos-1, pos):
            if record.pos == pos and record.ref == ref and alt in record.alts:
                # Extract depth and VAF
                info = record.info
                
                # Try to get depth
                depth = info.get('DP', 0)
                
                # Try to get allelic depths
                if 'I16' in info:
                    # Format: ref_fwd, ref_rev, alt_fwd, alt_rev, ...
                    i16 = info['I16']
                    ad_ref = i16[0] + i16[1]
                    ad_alt = i16[2] + i16[3]
                    
                    if depth == 0:
                        depth = ad_ref + ad_alt
                    
                    if depth > 0:
                        vaf = ad_alt / depth
                    else:
                        vaf = 0
                    
                    return depth, vaf, ad_ref, ad_alt
                else:
                    # No allelic depth info
                    return depth, 0, 0, 0
        
        vcf.close()
        return None
        
    except Exception as e:
        logger.debug(f"Error reading {vcf_file}: {e}")
        return None


def collect_variant_data(chrom, pos, ref, alt, vcf_files, spot_metadata):
    """
    Collect variant data across all spots
    
    Returns:
        variant_data dict with 'spots' list
    """
    spots_data = []
    
    for vcf_file in vcf_files:
        # Extract barcode from filename
        barcode = os.path.basename(vcf_file).replace('.vcf.gz', '')
        
        # Extract variant info
        result = extract_vaf_from_vcf(vcf_file, chrom, pos, ref, alt)
        
        if result:
            depth, vaf, ad_ref, ad_alt = result
            
            # Only include if has some coverage
            if depth > 0:
                spots_data.append({
                    'barcode': barcode,
                    'depth': depth,
                    'vaf': vaf,
                    'ad_ref': ad_ref,
                    'ad_alt': ad_alt
                })
    
    variant_data = {
        'chrom': chrom,
        'pos': pos,
        'ref': ref,
        'alt': alt,
        'spots': spots_data
    }
    
    return variant_data


def get_all_variants_in_chromosome(vcf_files, chrom):
    """
    Get list of all unique variants in a chromosome across all VCFs
    
    Returns:
        List of (chrom, pos, ref, alt) tuples
    """
    variants = set()
    
    logger.info(f"Scanning VCF files for variants in {chrom}...")
    
    for i, vcf_file in enumerate(vcf_files):
        if i % 100 == 0:
            logger.info(f"Scanned {i}/{len(vcf_files)} VCF files, found {len(variants)} variants so far")
        
        try:
            vcf = pysam.VariantFile(vcf_file)
            
            for record in vcf.fetch(chrom):
                for alt in record.alts:
                    variants.add((record.chrom, record.pos, record.ref, alt))
            
            vcf.close()
            
        except Exception as e:
            logger.debug(f"Error reading {vcf_file}: {e}")
            continue
    
    logger.info(f"Found {len(variants)} unique variants in {chrom}")
    
    return sorted(list(variants))


def write_vcf_header(output_file, config):
    """Write VCF header with SPARCAL INFO fields"""
    header_lines = [
        "##fileformat=VCFv4.2",
        "##source=SPARCAL_v1.0",
        "##INFO=<ID=SPARCAL_CLASS,Number=1,Type=String,Description=\"SPARCAL classification: somatic, germline, or uncertain\">",
        "##INFO=<ID=SPARCAL_PROB_SOM,Number=1,Type=Float,Description=\"Probability variant is somatic\">",
        "##INFO=<ID=SPARCAL_PROB_GERM,Number=1,Type=Float,Description=\"Probability variant is germline\">",
        "##INFO=<ID=SPARCAL_LR,Number=1,Type=Float,Description=\"Likelihood ratio (somatic/germline)\">",
        "##INFO=<ID=SPARCAL_CONF,Number=1,Type=Float,Description=\"Classification confidence\">",
        "##INFO=<ID=CNA_STATE,Number=2,Type=Integer,Description=\"Copy number state (copy_A,copy_B)\">",
        "##INFO=<ID=CNA_CATEGORY,Number=1,Type=String,Description=\"CNA category\">",
        "##INFO=<ID=N_SPOTS_COV,Number=1,Type=Integer,Description=\"Number of spots with coverage\">",
        "##INFO=<ID=MEAN_VAF,Number=1,Type=Float,Description=\"Mean VAF across covered spots\">",
        "##INFO=<ID=VAF_PURITY_CORR,Number=1,Type=Float,Description=\"Pearson correlation between VAF and purity\">",
        "##INFO=<ID=MORANS_I,Number=1,Type=Float,Description=\"Moran's I spatial autocorrelation\">",
        "##FILTER=<ID=PASS,Description=\"Passed all filters\">",
        "##FILTER=<ID=SOMATIC,Description=\"Classified as somatic\">",
        "##FILTER=<ID=GERMLINE,Description=\"Classified as germline\">",
        "##FILTER=<ID=UNCERTAIN,Description=\"Classification uncertain\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    ]
    
    with gzip.open(output_file, 'wt') as f:
        f.write('\n'.join(header_lines) + '\n')


def write_vcf_record(f, variant_data, classification, features):
    """Write a single VCF record"""
    chrom = variant_data['chrom']
    pos = variant_data['pos']
    ref = variant_data['ref']
    alt = variant_data['alt']
    
    # Build INFO field
    info_fields = [
        f"SPARCAL_CLASS={classification['classification']}",
        f"SPARCAL_PROB_SOM={classification['prob_somatic']:.4f}",
        f"SPARCAL_PROB_GERM={classification['prob_germline']:.4f}",
        f"SPARCAL_LR={classification['likelihood_ratio']:.4f}",
        f"SPARCAL_CONF={classification['confidence']:.4f}",
        f"CNA_STATE={features['copy_A']},{features['copy_B']}",
        f"CNA_CATEGORY={features['cna_category']}",
        f"N_SPOTS_COV={features['n_covered_spots']}",
        f"MEAN_VAF={features['mean_vaf']:.4f}",
        f"VAF_PURITY_CORR={features.get('vaf_purity_pearson', 0):.4f}",
        f"MORANS_I={features.get('morans_i', 0):.4f}"
    ]
    
    info_str = ';'.join(info_fields)
    
    # FILTER field
    filter_field = classification['classification'].upper()
    
    # Write record
    record = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t{filter_field}\t{info_str}\n"
    f.write(record)


def write_features_tsv(output_file, all_features, all_classifications):
    """Write detailed features to TSV"""
    logger.info(f"Writing features TSV to {output_file}")
    
    rows = []
    for i, (variant_data, features, classification) in enumerate(
        zip(all_features['variants'], all_features['features'], all_classifications)
    ):
        row = {
            'chrom': variant_data['chrom'],
            'pos': variant_data['pos'],
            'ref': variant_data['ref'],
            'alt': variant_data['alt'],
            'classification': classification['classification'],
            'prob_somatic': classification['prob_somatic'],
            'prob_germline': classification['prob_germline'],
            'likelihood_ratio': classification['likelihood_ratio'],
            'confidence': classification['confidence']
        }
        
        # Add all features
        row.update(features)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, sep='\t', index=False, compression='gzip')
    
    logger.info(f"Wrote {len(rows)} variant features")


def write_summary_stats(output_file, all_classifications, config):
    """Write summary statistics"""
    logger.info(f"Writing summary to {output_file}")
    
    n_total = len(all_classifications)
    n_somatic = sum(1 for c in all_classifications if c['classification'] == 'somatic')
    n_germline = sum(1 for c in all_classifications if c['classification'] == 'germline')
    n_uncertain = sum(1 for c in all_classifications if c['classification'] == 'uncertain')
    
    prob_somatic_values = [c['prob_somatic'] for c in all_classifications]
    lr_values = [c['likelihood_ratio'] for c in all_classifications]
    
    with open(output_file, 'w') as f:
        f.write("SPARCAL Classification Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Classification Counts:\n")
        f.write(f"  Total variants: {n_total}\n")
        f.write(f"  Somatic: {n_somatic} ({100*n_somatic/n_total:.1f}%)\n")
        f.write(f"  Germline: {n_germline} ({100*n_germline/n_total:.1f}%)\n")
        f.write(f"  Uncertain: {n_uncertain} ({100*n_uncertain/n_total:.1f}%)\n\n")
        
        f.write("Probability Statistics:\n")
        f.write(f"  Mean P(somatic): {np.mean(prob_somatic_values):.3f}\n")
        f.write(f"  Median P(somatic): {np.median(prob_somatic_values):.3f}\n")
        f.write(f"  Std P(somatic): {np.std(prob_somatic_values):.3f}\n\n")
        
        f.write("Likelihood Ratio Statistics:\n")
        f.write(f"  Mean LR: {np.mean(lr_values):.3f}\n")
        f.write(f"  Median LR: {np.median(lr_values):.3f}\n")
        f.write(f"  High LR (>5): {sum(1 for lr in lr_values if lr > 5)}\n")
        f.write(f"  Low LR (<0.2): {sum(1 for lr in lr_values if lr < 0.2)}\n\n")
        
        f.write("Configuration Used:\n")
        f.write(f"  P(somatic) threshold: {config['classification']['prob_somatic_threshold']}\n")
        f.write(f"  P(germline) threshold: {config['classification']['prob_germline_threshold']}\n")
        f.write(f"  LR somatic threshold: {config['classification']['likelihood_ratio_somatic']}\n")
        f.write(f"  LR germline threshold: {config['classification']['likelihood_ratio_germline']}\n")
    
    logger.info("Summary statistics written")


def main():
    """Main execution"""
    args = parse_arguments()
    
    logger.info("=" * 70)
    logger.info("SPARCAL Spatial Variant Classifier")
    logger.info("=" * 70)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Section: {args.section}")
    logger.info(f"Chromosome: {args.chromosome}")
    
    # Load configuration
    config = load_config(args)
    logger.info("Configuration loaded")
    
    # Setup paths
    paths = setup_paths(args)
    logger.info(f"VCF directory: {paths['vcf_dir']}")
    logger.info(f"CNA seg file: {paths['cna_seg_file']}")
    logger.info(f"Clone file: {paths['clone_file']}")
    logger.info(f"Output directory: {paths['output_dir']}")
    
    # Validate inputs
    if not os.path.exists(paths['vcf_dir']):
        logger.error(f"VCF directory not found: {paths['vcf_dir']}")
        sys.exit(1)
    
    if not os.path.exists(paths['cna_seg_file']):
        logger.error(f"CNA seg file not found: {paths['cna_seg_file']}")
        sys.exit(1)
    
    if not os.path.exists(paths['clone_file']):
        logger.error(f"Clone file not found: {paths['clone_file']}")
        sys.exit(1)
    
    # Load CNA profile
    logger.info("Loading CNA profile...")
    cna_profile = CNAProfile(paths['cna_seg_file'])
    
    # Load spot metadata
    logger.info("Loading spot metadata...")
    spot_metadata = SpotMetadata(paths['clone_file'], paths['spatial_dir'])
    
    # Initialize models
    logger.info("Initializing classification models...")
    feature_engineer = SPARCALFeatureEngineer(config)
    classifier = BayesianClassifier(config)
    
    # Get list of VCF files
    vcf_files = get_vcf_list(paths['vcf_dir'])
    
    if len(vcf_files) == 0:
        logger.error("No VCF files found")
        sys.exit(1)
    
    # Get all variants in this chromosome
    variants = get_all_variants_in_chromosome(vcf_files, args.chromosome)
    
    if len(variants) == 0:
        logger.warning(f"No variants found in {args.chromosome}")
        # Write empty outputs
        write_vcf_header(paths['output_vcf'], config)
        with open(paths['output_summary'], 'w') as f:
            f.write(f"No variants found in {args.chromosome}\n")
        sys.exit(0)
    
    logger.info(f"Processing {len(variants)} variants...")
    
    # Process variants
    all_features_list = []
    all_classifications = []
    
    # Write VCF header
    write_vcf_header(paths['output_vcf'], config)
    
    # Open files for writing
    vcf_out = gzip.open(paths['output_vcf'], 'at')
    
    for i, (chrom, pos, ref, alt) in enumerate(variants):
        if i % 100 == 0:
            logger.info(f"Processed {i}/{len(variants)} variants")
        
        # Collect variant data across spots
        variant_data = collect_variant_data(chrom, pos, ref, alt, vcf_files, spot_metadata)
        
        # Check minimum coverage requirement
        if variant_data['spots'] and len(variant_data['spots']) >= config['coverage']['min_spots_with_coverage']:
            # Engineer features
            features = feature_engineer.engineer_all_features(
                variant_data, cna_profile, spot_metadata
            )
            
            # Classify
            classification = classifier.classify(features, return_details=False)
            
            # Write to VCF
            write_vcf_record(vcf_out, variant_data, classification, features)
            
            # Store for features TSV
            all_features_list.append({
                'variant': variant_data,
                'features': features,
                'classification': classification
            })
            
            all_classifications.append(classification)
    
    vcf_out.close()
    
    logger.info(f"Finished processing {len(variants)} variants")
    logger.info(f"Classified {len(all_classifications)} variants (passed coverage threshold)")
    
    # Write features TSV
    if config['output']['write_features_tsv'] and all_features_list:
        all_features_dict = {
            'variants': [item['variant'] for item in all_features_list],
            'features': [item['features'] for item in all_features_list],
        }
        write_features_tsv(paths['output_features'], all_features_dict, all_classifications)
    
    # Write summary statistics
    if config['output']['write_summary_stats'] and all_classifications:
        write_summary_stats(paths['output_summary'], all_classifications, config)
    
    # Index VCF
    logger.info("Indexing output VCF...")
    pysam.tabix_index(paths['output_vcf'], preset='vcf', force=True)
    
    logger.info("=" * 70)
    logger.info("SPARCAL classification complete!")
    logger.info(f"Output VCF: {paths['output_vcf']}")
    logger.info(f"Output features: {paths['output_features']}")
    logger.info(f"Output summary: {paths['output_summary']}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()