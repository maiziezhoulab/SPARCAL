#!/usr/bin/env python3
"""
Benchmark BAM filtering script for external variant calling models (Strelka, GATK, Monopogen).

This script adapts the refilter_bam approach to work with VCF files from other variant calling models.
Each model may have different naming conventions and file structures, so this script provides
flexible VCF path configuration.

Author: Co-developed for benchmarking study
Date: 2025
"""

import os
import sys
import glob
import gzip
import argparse
import pysam
from pathlib import Path
from typing import List, Set, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess

# Configuration for paths and environment
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
}

# Dataset configurations - reused from existing pipeline
DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/DLPFC_V2",
        "bam_pattern": "DLPFC_{section_id}/spaceranger_align/output/outs/split_BAM/*.bam",
        "barcode_file": "DLPFC_{section_id}/spaceranger_align/output/outs/filtered_feature_bc_matrix/barcodes.tsv.gz",
        "output_dir": "data/DLPFC/{section_id}",
        "has_sections": True,
        "section_ids": ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"],
        "reference": "DLPFC",
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P4_Tumor_output/outs/split_BAM/*.bam",
        "barcode_file": "spaceranger_align_rep{section_id}/Meta_Data/GSM4565823_barcodes.tsv.gz",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P6_Tumor_output/outs/split_BAM/*.bam",
        "barcode_file": "spaceranger_align_rep{section_id}/Meta_Data/GSM4565825_barcodes.tsv.gz",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
    }
}

@dataclass
class SNVInfo:
    """Data class to store SNV information."""
    chrom: str
    pos: int
    ref: str
    alt: str
    variant_type: str = "germline"  # germline or somatic
    
    def __eq__(self, other):
        if isinstance(other, SNVInfo):
            return (self.standardized_chrom, self.pos, self.ref, self.alt) == (other.standardized_chrom, other.pos, other.ref, other.alt)
        return False

    def __hash__(self):
        return hash((self.standardized_chrom, self.pos, self.ref, self.alt))

    @property
    def key(self) -> str:
        return f"{self.standardized_chrom}_{self.pos}_{self.ref}_{self.alt}"
    
    @property
    def standardized_chrom(self) -> str:
        """Remove 'chr' prefix if present for consistent comparison."""
        return self.chrom.replace("chr", "")
    
    @classmethod
    def from_vcf_line(cls, line: str, variant_type: str = "germline") -> 'SNVInfo':
        """Create SNVInfo from VCF line."""
        fields = line.strip().split('\t')
        return cls(
            chrom=fields[0],
            pos=int(fields[1]),
            ref=fields[3],
            alt=fields[4],
            variant_type=variant_type
        )

def setup_environment():
    """Setup environment variables for library paths."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path

def filter_bam_one_chrom(input_bam: str, chrom: str, positions: List[int]) -> Tuple[List[pysam.AlignedSegment], List[int]]:
    """
    Filter a BAM file for a single chromosome with optimized search algorithm.
    
    Args:
        input_bam: Path to input BAM file
        chrom: Chromosome name to filter
        positions: Sorted list of positions to filter for (1-based)
        
    Returns:
        Tuple of (filtered reads, detected positions)
    """
    filtered_reads = []
    detected_positions = set()
    start_j = 0
    
    try:
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            for read in in_bam.fetch(chrom):
                if read.is_unmapped:
                    continue
                    
                read_start = read.reference_start + 1
                read_end = read.reference_end + 1 if read.reference_end else read_start
                
                while start_j < len(positions) and positions[start_j] < read_start:
                    start_j += 1
                
                j = start_j
                while j < len(positions) and positions[j] <= read_end:
                    pos = positions[j]
                    if read_start <= pos <= read_end:
                        filtered_reads.append(read)
                        detected_positions.add(pos)
                        break
                    j += 1
    except Exception as e:
        print(f"Error processing chromosome {chrom} in {input_bam}: {str(e)}")
        
    return filtered_reads, list(detected_positions)

def filter_single_bam(args) -> Dict:
    """
    Filter a single BAM file for reads overlapping SNV positions.
    
    Args:
        args: Tuple of (input_bam, output_dir, snvs, barcode)
        
    Returns:
        Dictionary with processing results
    """
    input_bam, output_dir, snvs, barcode = args
    
    try:
        # Group SNVs by chromosome
        snvs_by_chrom = {}
        for snv in snvs:
            chrom = snv.standardized_chrom
            if chrom not in snvs_by_chrom:
                snvs_by_chrom[chrom] = []
            snvs_by_chrom[chrom].append(snv.pos)
        
        # Sort positions for each chromosome
        for chrom in snvs_by_chrom:
            snvs_by_chrom[chrom].sort()
        
        # Process each chromosome
        all_filtered_reads = []
        all_detected_positions = []
        
        for chrom, positions in snvs_by_chrom.items():
            filtered_reads, detected_positions = filter_bam_one_chrom(input_bam, chrom, positions)
            all_filtered_reads.extend(filtered_reads)
            all_detected_positions.extend([(chrom, pos) for pos in detected_positions])
        
        # Write filtered BAM
        output_bam = os.path.join(output_dir, f"{barcode}_filtered.bam")
        
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            with pysam.AlignmentFile(output_bam, "wb", template=in_bam) as out_bam:
                for read in all_filtered_reads:
                    out_bam.write(read)
        
        # Index the filtered BAM
        pysam.index(output_bam)
        
        # Save detected SNV positions to text file
        snv_txt_dir = os.path.join(os.path.dirname(output_dir), "snv_positions")
        os.makedirs(snv_txt_dir, exist_ok=True)
        snv_txt_file = os.path.join(snv_txt_dir, f"{barcode}_snv_positions.txt")
        
        with open(snv_txt_file, 'w') as f:
            f.write("chromosome\tposition\n")
            for chrom, pos in sorted(all_detected_positions):
                f.write(f"{chrom}\t{pos}\n")
        
        return {
            'status': 'completed',
            'input_bam': input_bam,
            'output_bam': output_bam,
            'detected_snvs': len(all_detected_positions),
            'total_reads': len(all_filtered_reads)
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'input_bam': input_bam,
            'error': str(e)
        }

def filter_bams_parallel(input_bams: List[str], output_dir: str, snvs: Set[SNVInfo], max_workers: int = 30) -> List[Dict]:
    """Filter BAM files in parallel."""
    
    # Extract barcode from BAM filename
    def extract_barcode(bam_path):
        basename = os.path.basename(bam_path)
        return basename.replace('.bam', '').replace('_possorted', '').replace('_sorted', '')
    
    # Prepare arguments for parallel processing
    args_list = []
    for bam_file in input_bams:
        barcode = extract_barcode(bam_file)
        args_list.append((bam_file, output_dir, snvs, barcode))
    
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_bam = {executor.submit(filter_single_bam, args): args[0] for args in args_list}
        
        for future in as_completed(future_to_bam):
            bam_file = future_to_bam[future]
            try:
                result = future.result()
                results.append(result)
                
                if result['status'] == 'completed':
                    print(f"✓ {os.path.basename(bam_file)}: {result['detected_snvs']} SNVs, {result['total_reads']} reads")
                else:
                    print(f"✗ {os.path.basename(bam_file)}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"✗ {os.path.basename(bam_file)}: Exception - {str(e)}")
                results.append({
                    'status': 'failed',
                    'input_bam': bam_file,
                    'error': str(e)
                })
    
    return results

class BenchmarkBAMFilter:
    """Main class for filtering BAMs using external model VCF files."""
    
    def __init__(self, dataset_name: str, section_id: str, model_name: str, 
                 vcf_paths: Dict[str, str], quality_filter: str = "baseQ0mapQ0"):
        """
        Initialize the benchmark BAM filter.
        
        Args:
            dataset_name: Name of the dataset (DLPFC, P4_TUMOR, P6_TUMOR)
            section_id: Section ID for the dataset
            model_name: Name of the benchmarking model (strelka, gatk, monopogen)
            vcf_paths: Dictionary with VCF file paths, e.g.:
                      {'germline': '/path/to/germline.vcf.gz', 'somatic': '/path/to/somatic.vcf.gz'}
            quality_filter: Quality filter string
        """
        self.dataset_name = dataset_name.upper()
        self.section_id = section_id
        self.model_name = model_name.lower()
        self.vcf_paths = vcf_paths
        self.quality_filter = quality_filter
        self.base_dir = PATH_CONFIG["PROJECT_DIR"]
        
        self.validate_dataset_config()
        self.setup_paths()
        setup_environment()
    
    def validate_dataset_config(self):
        """Validate dataset configuration and section ID."""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. "
                           f"Available: {list(DATASET_CONFIGS.keys())}")
        
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        if dataset_config["has_sections"]:
            if not self.section_id:
                raise ValueError(f"Dataset {self.dataset_name} requires a section_id")
            if "section_ids" in dataset_config:
                if self.section_id not in dataset_config["section_ids"]:
                    raise ValueError(f"Invalid section_id for {self.dataset_name}. "
                                   f"Valid section IDs are: {dataset_config['section_ids']}")
    
    def setup_paths(self):
        """Setup input and output paths based on dataset configuration."""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        
        # Determine base paths
        if dataset_config["has_sections"]:
            data_path = dataset_config["output_dir"].format(section_id=self.section_id)
        else:
            data_path = dataset_config["output_dir"]
            
        input_base = os.path.join(self.base_dir, data_path)
        
        # BAM directory based on dataset configuration
        if dataset_config["has_sections"]:
            self.bam_dir = os.path.join(
                dataset_config["base_path"],
                dataset_config["bam_pattern"].format(section_id=self.section_id)
            )
        else:
            self.bam_dir = os.path.join(
                dataset_config["base_path"],
                dataset_config["bam_pattern"]
            )
        
        # Output directory for filtered BAMs - organized by model
        self.filtered_bam_dir = os.path.join(
            input_base, 
            f"output_VCFs/BAM_filtered_{self.model_name}",
            self.quality_filter
        )
        os.makedirs(self.filtered_bam_dir, exist_ok=True)
        
        # Log directory
        self.log_dir = os.path.join(
            input_base,
            f"logs/BAM_filtered_{self.model_name}",
            self.quality_filter
        )
        os.makedirs(self.log_dir, exist_ok=True)
    
    def is_valid_variant(self, vcf_line: str) -> bool:
        """
        Check if a VCF line represents a valid variant.
        This method can be customized for different variant calling models.
        """
        try:
            fields = vcf_line.strip().split('\t')
            if len(fields) < 5:
                return False
                
            # Basic checks
            ref, alt = fields[3], fields[4]
            
            # Skip if REF or ALT contains 'N'
            if 'N' in ref or 'N' in alt:
                return False
            
            # For models with genotype information
            if len(fields) > 9:
                format_fields = fields[8].split(':')
                sample_fields = fields[9].split(':')
                
                if 'GT' in format_fields:
                    gt_idx = format_fields.index('GT')
                    gt = sample_fields[gt_idx]
                    # Only include non-reference genotypes
                    return gt in ['0/1', '1/1', '1/0']
            
            # If no genotype info, include all variants
            return True
            
        except (ValueError, IndexError):
            return False
    
    def collect_snvs_from_vcf(self, vcf_path: str, variant_type: str = "germline") -> Set[SNVInfo]:
        """Collect SNVs from a single VCF file."""
        snvs = set()
        
        if not os.path.exists(vcf_path):
            print(f"Warning: VCF file not found: {vcf_path}")
            return snvs
        
        print(f"Collecting {variant_type} SNVs from: {vcf_path}")
        
        try:
            # Handle both gzipped and uncompressed VCF files
            if vcf_path.endswith('.gz'):
                file_handle = gzip.open(vcf_path, 'rt')
            else:
                file_handle = open(vcf_path, 'r')
            
            with file_handle as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    if self.is_valid_variant(line):
                        try:
                            snv = SNVInfo.from_vcf_line(line, variant_type)
                            # Only include SNPs (single nucleotide variants)
                            if len(snv.ref) == 1 and len(snv.alt) == 1:
                                snvs.add(snv)
                        except (ValueError, IndexError) as e:
                            continue
            
            print(f"Collected {len(snvs)} {variant_type} SNVs")
            return snvs
            
        except Exception as e:
            print(f"Error reading VCF file {vcf_path}: {e}")
            return snvs
    
    def collect_all_snvs(self) -> Set[SNVInfo]:
        """Collect all SNVs from provided VCF files."""
        all_snvs = set()
        
        for variant_type, vcf_path in self.vcf_paths.items():
            if vcf_path:  # Check if path is provided
                snvs = self.collect_snvs_from_vcf(vcf_path, variant_type)
                all_snvs.update(snvs)
        
        print(f"Total unique SNVs collected: {len(all_snvs)}")
        return all_snvs
    
    def filter_bams(self, max_workers: int = 30):
        """Filter BAM files based on SNV positions."""
        print(f"Starting BAM filtering for {self.model_name} model...")
        
        # Collect SNVs from VCF files
        snvs = self.collect_all_snvs()
        if not snvs:
            print("No SNVs found to filter on. Exiting.")
            return []
        
        # Get list of BAM files
        if '*' not in self.bam_dir:
            search_pattern = os.path.join(self.bam_dir, '*.bam')
        else:
            search_pattern = self.bam_dir
        
        bam_files = glob.glob(search_pattern)
        if not bam_files:
            raise FileNotFoundError(f"No BAM files found at: {search_pattern}")
        
        print(f"Found {len(bam_files)} BAM files to process")
        
        # Filter BAMs in parallel
        results = filter_bams_parallel(
            input_bams=bam_files,
            output_dir=self.filtered_bam_dir,
            snvs=snvs,
            max_workers=max_workers
        )
        
        # Print summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs', 0) > 0)
        total_snvs = sum(r.get('detected_snvs', 0) for r in results if r['status'] == 'completed')
        total_reads = sum(r.get('total_reads', 0) for r in results if r['status'] == 'completed')
        
        print(f"\n{self.model_name.upper()} BAM Filtering Summary:")
        print(f"Total BAMs processed: {len(results)}")
        print(f"Successfully filtered: {completed}")
        print(f"Failed: {failed}")
        print(f"BAMs with detected SNVs: {with_snvs}")
        print(f"Total SNVs detected: {total_snvs}")
        print(f"Total filtered reads: {total_reads}")
        
        if failed > 0:
            print("\nFailed BAMs:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  {os.path.basename(result['input_bam'])}: {result.get('error', 'Unknown error')}")
        
        print(f"\nFiltered BAM files are located in: {self.filtered_bam_dir}")
        
        # Create summary report
        self.create_summary_report(results, snvs)
        
        return results
    
    def create_summary_report(self, results: List[Dict], snvs: Set[SNVInfo]):
        """Create a summary report of the filtering results."""
        report_file = os.path.join(self.log_dir, f"{self.model_name}_filtering_summary.txt")
        
        with open(report_file, 'w') as f:
            f.write(f"BAM Filtering Summary for {self.model_name.upper()} Model\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n")
            f.write(f"Processing Date: {subprocess.check_output(['date']).decode().strip()}\n\n")
            
            f.write("VCF Files Used:\n")
            for variant_type, vcf_path in self.vcf_paths.items():
                f.write(f"  {variant_type}: {vcf_path}\n")
            f.write(f"Total unique SNVs: {len(snvs)}\n\n")
            
            # Summarize by variant type
            germline_snvs = sum(1 for snv in snvs if snv.variant_type == "germline")
            somatic_snvs = sum(1 for snv in snvs if snv.variant_type == "somatic")
            f.write(f"Germline SNVs: {germline_snvs}\n")
            f.write(f"Somatic SNVs: {somatic_snvs}\n\n")
            
            # Results summary
            completed = sum(1 for r in results if r['status'] == 'completed')
            failed = sum(1 for r in results if r['status'] == 'failed')
            with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs', 0) > 0)
            total_snvs = sum(r.get('detected_snvs', 0) for r in results if r['status'] == 'completed')
            total_reads = sum(r.get('total_reads', 0) for r in results if r['status'] == 'completed')
            
            f.write("Processing Results:\n")
            f.write(f"Total BAMs processed: {len(results)}\n")
            f.write(f"Successfully filtered: {completed}\n")
            f.write(f"Failed: {failed}\n")
            f.write(f"BAMs with detected SNVs: {with_snvs}\n")
            f.write(f"Total SNVs detected across all BAMs: {total_snvs}\n")
            f.write(f"Total filtered reads: {total_reads}\n\n")
            
            # Individual BAM results
            f.write("Individual BAM Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Barcode':<20} {'Status':<12} {'SNVs':<8} {'Reads':<10} {'Output BAM':<30}\n")
            f.write("-" * 80 + "\n")
            
            for result in sorted(results, key=lambda x: os.path.basename(x['input_bam'])):
                barcode = os.path.basename(result['input_bam']).replace('.bam', '')
                status = result['status']
                snv_count = result.get('detected_snvs', 0) if status == 'completed' else 'N/A'
                read_count = result.get('total_reads', 0) if status == 'completed' else 'N/A'
                output_bam = os.path.basename(result.get('output_bam', 'N/A'))
                
                f.write(f"{barcode:<20} {status:<12} {snv_count:<8} {read_count:<10} {output_bam:<30}\n")
        
        print(f"Summary report saved to: {report_file}")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Filter BAM files using variants from external models (Strelka, GATK, Monopogen)"
    )
    
    parser.add_argument("--dataset", required=True, 
                       choices=list(DATASET_CONFIGS.keys()),
                       help="Dataset name")
    
    parser.add_argument("--section-id", 
                       help="Section ID (required for datasets with sections)")
    
    parser.add_argument("--model", required=True,
                       choices=["strelka", "gatk", "monopogen"],
                       help="Benchmarking model name")
    
    parser.add_argument("--germline-vcf", 
                       help="Path to germline variants VCF file (.vcf.gz)")
    
    parser.add_argument("--somatic-vcf", 
                       help="Path to somatic variants VCF file (.vcf.gz)")
    
    parser.add_argument("--merged-vcf", 
                       help="Path to merged variants VCF file (if germline and somatic are combined)")
    
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                       help="Quality filter string (default: baseQ0mapQ0)")
    
    parser.add_argument("--max-workers", type=int, default=30,
                       help="Maximum number of parallel workers (default: 30)")
    
    args = parser.parse_args()
    
    # Validate VCF file inputs
    vcf_paths = {}
    if args.merged_vcf:
        if not os.path.exists(args.merged_vcf):
            parser.error(f"Merged VCF file not found: {args.merged_vcf}")
        vcf_paths["merged"] = args.merged_vcf
    else:
        if args.germline_vcf:
            if not os.path.exists(args.germline_vcf):
                parser.error(f"Germline VCF file not found: {args.germline_vcf}")
            vcf_paths["germline"] = args.germline_vcf
        
        if args.somatic_vcf:
            if not os.path.exists(args.somatic_vcf):
                parser.error(f"Somatic VCF file not found: {args.somatic_vcf}")
            vcf_paths["somatic"] = args.somatic_vcf
    
    if not vcf_paths:
        parser.error("At least one VCF file must be provided (--germline-vcf, --somatic-vcf, or --merged-vcf)")
    
    # Dataset configuration validation
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"]:
        if not args.section_id:
            parser.error(f"Dataset {args.dataset} requires --section-id")
        if "section_ids" in dataset_config:
            if args.section_id not in dataset_config["section_ids"]:
                parser.error(f"Invalid section_id. Valid values: {dataset_config['section_ids']}")
    
    # Print configuration
    print("\nBenchmark BAM Filtering Configuration:")
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Model: {args.model}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Max Workers: {args.max_workers}")
    print("VCF Files:")
    for variant_type, vcf_path in vcf_paths.items():
        print(f"  {variant_type}: {vcf_path}")
    print("\n")
    
    # Initialize and run the filter
    try:
        filter_obj = BenchmarkBAMFilter(
            dataset_name=args.dataset,
            section_id=args.section_id,
            model_name=args.model,
            vcf_paths=vcf_paths,
            quality_filter=args.quality_filter
        )
        
        results = filter_obj.filter_bams(max_workers=args.max_workers)
        
        # Exit with error if any BAMs failed
        if any(r['status'] == 'failed' for r in results):
            print("Some BAM files failed to process. Check the summary report for details.")
            sys.exit(1)
        else:
            print("All BAM files processed successfully!")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

# Usage examples:
# 
# For Strelka with separate germline and somatic VCFs:
# python run_benchmark_models_bam_filter.py --dataset DLPFC --section-id 151508 --model strelka \
#   --germline-vcf /path/to/strelka/germline.vcf.gz --somatic-vcf /path/to/strelka/somatic.vcf.gz \
#   --quality-filter baseQ0mapQ0 --max-workers 30
#
# For GATK with merged VCF:
# python run_benchmark_models_bam_filter.py --dataset DLPFC --section-id 151508 --model gatk \
#   --merged-vcf /path/to/gatk/merged_variants.vcf.gz --quality-filter baseQ0mapQ0
#
# For Monopogen with germline only:
# python run_benchmark_models_bam_filter.py --dataset P4_TUMOR --section-id 1 --model monopogen \
#   --germline-vcf /path/to/monopogen/germline.vcf.gz --quality-filter baseQ0mapQ0