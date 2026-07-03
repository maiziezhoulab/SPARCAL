import os
import gzip
import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import glob
import subprocess
import tempfile

# Modify the save_detected_snvs function to save VCF files instead of TXT files
def save_detected_snvs(output_dir: str, result: Dict, snv_info_dict: Dict = None):
    """
    Save detected SNVs to a VCF file named after the BAM file.
    
    Args:
        output_dir: Directory to save the file
        result: Result dictionary from filter_bam_by_positions
        snv_info_dict: Dictionary mapping (chrom, pos, ref, alt) to SNVInfo objects
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    if result['status'] != 'completed' or not result.get('detected_snvs'):
        # Return False to indicate no SNVs were saved
        return False
        
    # Get barcode name from BAM filename
    bam_name = os.path.basename(result['input_bam'])
    barcode = bam_name.replace('.bam', '')
    output_file = os.path.join(output_dir, f"{barcode}.vcf.gz")
    temp_vcf = os.path.join(output_dir, f"{barcode}.vcf")
    
    try:
        # Sort detected SNVs by chromosome and position
        sorted_snvs = sorted(result['detected_snvs'], key=lambda x: (x[0], x[1]))
        
        # Write VCF file
        with open(temp_vcf, 'w') as f:
            # Write VCF header
            f.write("##fileformat=VCFv4.2\n")
            f.write("##source=SNVMatrixGenerator_filter_bams_by_snv_pools\n")
            f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
            f.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write("##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths for the ref and alt alleles\">\n")
            f.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n")
            f.write("##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n")
            f.write("##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods\">\n")
            f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{barcode}\n")
            
            # Write variant records
            for snv_tuple in sorted_snvs:
                if len(snv_tuple) >= 4:
                    chrom, pos, ref, alt = snv_tuple[:4]
                    
                    if not chrom.startswith("chr"):
                        chrom = f"chr{chrom}"
                    # Try to get additional info from snv_info_dict if provided
                    info_field = "."
                    format_field = "GT"
                    sample_field = "./."
                    
                    if snv_info_dict:
                        # Create a standardized key for lookup
                        standardized_chrom = chrom.replace("chr", "")
                        snv_key = (standardized_chrom, pos, ref, alt)
                        
                        if snv_key in snv_info_dict:
                            snv_info = snv_info_dict[snv_key]
                            info_field = snv_info.info if snv_info.info else "."
                            format_field = snv_info.format_str if snv_info.format_str else "GT"
                            # Default sample field - this would come from the original VCF
                            sample_field = "./."
                    
                    # Write the VCF line
                    f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_field}\t{format_field}\t{sample_field}\n")
        
        # Compress the VCF file using bgzip
        compress_cmd = f"{PATH_CONFIG['BGZIP']} -f {temp_vcf}"
        compress_result = run_command(compress_cmd)
        
        if not compress_result:
            print(f"Warning: Failed to compress VCF for {barcode}, keeping uncompressed")
            return False
            
        # Index the compressed VCF using tabix
        index_cmd = f"{PATH_CONFIG['BCFTOOLS']} index -t {output_file}"
        index_result = run_command(index_cmd)
        
        if not index_result:
            print(f"Warning: Failed to index VCF for {barcode}")
            # Still return True as the VCF was created successfully
        
        return True
    except Exception as e:
        print(f"Error saving SNVs for {bam_name}: {str(e)}")
        # Clean up temp file if it exists
        if os.path.exists(temp_vcf):
            os.remove(temp_vcf)
        return False

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
        "bam_pattern": "{section_id}/bam_bycell/*.bam",
        "output_dir": "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC",
        "multiple_bams": True
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}_hg19/P4_Tumor_output/outs/split_BAM/",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}_hg19/P6_Tumor_output/outs/split_BAM/",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    }
}

# Path configuration matching mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
}

@dataclass
class SNVInfo:
    chrom: str
    pos: int
    ref: str
    alt: str
    info: str = ""
    format_str: str = ""
    
    def __eq__(self, other):
        if isinstance(other, SNVInfo):
            # Compare with standardized chromosome names
            return (self.standardized_chrom, self.pos, self.ref, self.alt) == (other.standardized_chrom, other.pos, other.ref, other.alt)
        return False

    def __hash__(self):
        # Use standardized chromosome name for hashing
        return hash((self.standardized_chrom, self.pos, self.ref, self.alt))

    @property
    def key(self) -> str:
        # Use standardized chromosome name for key
        return f"{self.standardized_chrom}_{self.pos}_{self.ref}_{self.alt}"
    
    @property
    def standardized_chrom(self) -> str:
        # Remove 'chr' prefix if present for consistent comparison
        return self.chrom.replace("chr", "")
    
    @classmethod
    def from_vcf_line(cls, line: str) -> 'SNVInfo':
        fields = line.strip().split('\t')
        return cls(
            chrom=fields[0],
            pos=int(fields[1]),
            ref=fields[3],
            alt=fields[4],
            info=fields[7],
            format_str=fields[8]
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
    detected_positions = set()  # Track which positions we actually find
    start_j = 0  # Starting position in the sorted positions list
    
    try:
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            # Use fetch to efficiently get reads for this chromosome
            for read in in_bam.fetch(chrom):
                # Skip unmapped reads
                if read.is_unmapped:
                    continue
                    
                # Get read coordinates (1-based)
                read_start = read.reference_start + 1
                read_end = read.reference_end + 1 if read.reference_end else read_start
                
                # Skip positions we've already passed
                while start_j < len(positions) and positions[start_j] < read_start:
                    start_j += 1
                
                # Check for overlap with any position
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

def create_all_variants_summary(output_dir: str):
    """
    Create a summary VCF file with all detected variants across all barcodes.
    
    Args:
        output_dir: Directory containing SNV VCF files
    """
    snv_vcf_dir = os.path.join(output_dir, "snv_vcf")
    summary_file = os.path.join(output_dir, "all_detected_variants_summary.vcf")
    summary_file_gz = summary_file + ".gz"
    
    # Dictionary to track variants and their counts
    all_variants = {}  # (chrom, pos, ref, alt) -> (count, info, format)
    barcode_count = 0
    
    # Process each barcode VCF file
    for vcf_file in glob.glob(os.path.join(snv_vcf_dir, "*.vcf.gz")):
        barcode_count += 1
        try:
            with gzip.open(vcf_file, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.strip().split('\t')
                    if len(parts) < 5:
                        continue
                    
                    chrom, pos, id_field, ref, alt = parts[:5]
                    info_field = parts[7] if len(parts) > 7 else "."
                    format_field = parts[8] if len(parts) > 8 else "GT"
                    
                    variant_key = (chrom, pos, ref, alt)
                    if variant_key not in all_variants:
                        all_variants[variant_key] = {
                            'count': 0, 
                            'info': info_field,
                            'format': format_field
                        }
                    all_variants[variant_key]['count'] += 1
        except Exception as e:
            print(f"Warning: Failed to read {vcf_file}: {str(e)}")
            continue
    
    # Write summary VCF file
    with open(summary_file, 'w') as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=SNVMatrixGenerator_all_variants_summary\n")
        f.write("##INFO=<ID=COUNT,Number=1,Type=Integer,Description=\"Number of barcodes with this variant\">\n")
        f.write("##INFO=<ID=FREQ,Number=1,Type=Float,Description=\"Frequency of variant across barcodes\">\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        # Sort by chromosome and position
        sorted_variants = sorted(all_variants.items(), key=lambda x: (x[0][0], int(x[0][1])))
        
        for (chrom, pos, ref, alt), var_data in sorted_variants:
            count = var_data['count']
            frequency = count / barcode_count if barcode_count > 0 else 0
            original_info = var_data['info']
            
            # Build INFO field with COUNT and FREQ
            if original_info != ".":
                info_field = f"{original_info};COUNT={count};FREQ={frequency:.4f}"
            else:
                info_field = f"COUNT={count};FREQ={frequency:.4f}"
            
            f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_field}\n")
    
    # Compress and index the summary VCF
    compress_cmd = f"{PATH_CONFIG['BGZIP']} -f {summary_file}"
    if run_command(compress_cmd):
        index_cmd = f"{PATH_CONFIG['BCFTOOLS']} index -t {summary_file_gz}"
        run_command(index_cmd)
    
    print(f"Created summary of all detected variants: {summary_file_gz}")
    print(f"Total unique variants: {len(all_variants)}")
    print(f"Total barcodes processed: {barcode_count}")
    
    return summary_file_gz

def run_command(cmd: str, log_file: Optional[str] = None) -> bool:
    """
    Run a shell command with logging.
    
    Args:
        cmd: Command to run
        log_file: Optional path to log file
        
    Returns:
        True if command succeeded, False otherwise
    """
    try:
        if log_file:
            with open(log_file, 'a') as log:
                result = subprocess.run(
                    cmd, 
                    shell=True, 
                    stdout=log, 
                    stderr=subprocess.STDOUT,
                    check=True
                )
        else:
            result = subprocess.run(
                cmd, 
                shell=True, 
                stderr=subprocess.PIPE,
                stdout=subprocess.PIPE,
                check=True
            )
        return True
    except subprocess.CalledProcessError as e:
        if not log_file:
            print(f"Command failed: {cmd}")
            print(f"Error: {e}")
            if e.stderr:
                print(f"stderr: {e.stderr.decode('utf-8')}")
        return False

def sort_bam_file(input_bam: str, output_bam: str) -> bool:
    """
    Sort a BAM file using samtools.
    
    Args:
        input_bam: Path to input BAM file
        output_bam: Path to output sorted BAM file
        
    Returns:
        True if sorting succeeded, False otherwise
    """
    # Create a temporary file for the sorted BAM
    temp_dir = os.path.dirname(output_bam)
    temp_prefix = os.path.join(temp_dir, f"temp_{os.path.basename(output_bam).replace('.bam', '')}")
    
    # Run sorting command
    cmd = f"{PATH_CONFIG['SAMTOOLS']} sort -o {output_bam} -T {temp_prefix} {input_bam}"
    return run_command(cmd)

def index_bam_file(bam_path: str) -> bool:
    """
    Index a BAM file using samtools.
    
    Args:
        bam_path: Path to BAM file to index
        
    Returns:
        True if indexing succeeded, False otherwise
    """
    cmd = f"{PATH_CONFIG['SAMTOOLS']} index {bam_path}"
    return run_command(cmd)

def filter_bam_by_positions(input_bam: str, output_bam: str, positions_by_chrom: Dict[str, List[Tuple]]) -> Dict:
    """
    Filter BAM file to only keep reads that overlap with SNV positions.
    
    Args:
        input_bam: Path to input BAM file
        output_bam: Path to output filtered BAM file
        positions_by_chrom: Dictionary of chromosome -> list of (position, ref, alt) tuples
        
    Returns:
        Dictionary with status information including detected SNVs
    """
    try:
        # Create temporary unsorted output file
        temp_output = f"{output_bam}.unsorted"
        detected_snvs = set()  # Set to store detected SNVs (chrom, pos, ref, alt)
        
        # Process one chromosome at a time
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            # Check which chromosomes are actually in the BAM file
            bam_references = set(in_bam.references)
            
            # Open output file
            with pysam.AlignmentFile(temp_output, "wb", header=in_bam.header) as out_bam:
                # Process each chromosome
                for chrom, pos_info_list in positions_by_chrom.items():
                    # Handle chromosome naming differences
                    chrom_with_prefix = f"chr{chrom}" if not chrom.startswith("chr") else chrom
                    chrom_without_prefix = chrom.replace("chr", "")
                    
                    # Find the correct chromosome name in the BAM
                    if chrom_with_prefix in bam_references:
                        bam_chrom = chrom_with_prefix
                    elif chrom_without_prefix in bam_references:
                        bam_chrom = chrom_without_prefix
                    else:
                        # Skip chromosomes not in the BAM
                        continue
                    
                    # Extract just positions for filtering
                    positions = [pos for pos, ref, alt in pos_info_list]
                    pos_to_info = {pos: (ref, alt) for pos, ref, alt in pos_info_list}
                    
                    # Filter reads for this chromosome
                    filtered_reads, detected_positions = filter_bam_one_chrom(input_bam, bam_chrom, positions)
                    
                    # Add detected positions to our set with correct chromosome name and ref/alt info
                    for pos in detected_positions:
                        ref, alt = pos_to_info[pos]
                        detected_snvs.add((bam_chrom, pos, ref, alt))
                    
                    # Write filtered reads
                    for read in filtered_reads:
                        out_bam.write(read)
        
        # Sort the output BAM file
        sorting_success = sort_bam_file(temp_output, output_bam)
        
        # Clean up temporary file
        if os.path.exists(temp_output):
            os.remove(temp_output)
        
        if not sorting_success:
            return {
                'input_bam': input_bam,
                'output_bam': output_bam,
                'error': "Failed to sort output BAM file",
                'status': 'failed',
                'detected_snvs': []
            }
        
        # Index the sorted output BAM file
        indexing_success = index_bam_file(output_bam)
        
        if not indexing_success:
            return {
                'input_bam': input_bam,
                'output_bam': output_bam,
                'error': "Failed to index output BAM file",
                'status': 'failed',
                'detected_snvs': []
            }
        
        return {
            'input_bam': input_bam,
            'output_bam': output_bam,
            'status': 'completed',
            'detected_snvs': list(detected_snvs)  # Convert set to list for returning
        }
        
    except Exception as e:
        return {
            'input_bam': input_bam,
            'output_bam': output_bam,
            'error': str(e),
            'status': 'failed',
            'detected_snvs': []
        }

def process_single_bam(input_bam: str, output_dir: str, positions_by_chrom: Dict[str, List[int]]) -> Dict:
    """
    Process a single BAM file: filter by SNV positions, sort, and index.
    
    Args:
        input_bam: Path to input BAM file
        output_dir: Directory for output BAM file
        positions_by_chrom: Dictionary of chromosome -> sorted list of positions
        
    Returns:
        Dictionary with status information
    """
    try:
        # Create output BAM path
        bam_name = os.path.basename(input_bam)
        output_bam = os.path.join(output_dir, bam_name)
        
        # Filter BAM file
        result = filter_bam_by_positions(input_bam, output_bam, positions_by_chrom)
        
        return result
        
    except Exception as e:
        return {
            'input_bam': input_bam,
            'error': str(e),
            'status': 'failed'
        }

def filter_bams_parallel(input_bams: List[str], output_dir: str, snvs: Set[SNVInfo], 
                      max_workers: int = 30) -> List[Dict]:
    """
    Filter multiple BAM files in parallel.
    
    Args:
        input_bams: List of paths to input BAM files
        output_dir: Directory for output BAM files
        snvs: Set of SNVInfo objects defining positions to keep
        max_workers: Maximum number of parallel workers
        
    Returns:
        List of dictionaries with status information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create and ensure the SNV VCF directory exists
    snv_vcf_dir = os.path.join(output_dir, "snv_vcf")
    os.makedirs(snv_vcf_dir, exist_ok=True)
    print(f"SNV VCF files will be saved to: {snv_vcf_dir}")
    
    results = []
    
    # Create position dictionary once for all BAM files
    print("Building SNV position dictionary...")
    positions_by_chrom = defaultdict(list)
    snv_info_dict = {}  # Dictionary to store SNVInfo objects for later retrieval
    
    for snv in snvs:
        # Use standardized chromosome name
        chrom = snv.standardized_chrom
        # Include ref and alt information
        positions_by_chrom[chrom].append((snv.pos, snv.ref, snv.alt))
        # Store the SNVInfo object for later use
        snv_key = (chrom, snv.pos, snv.ref, snv.alt)
        snv_info_dict[snv_key] = snv
    
    # Sort positions for each chromosome
    for chrom in positions_by_chrom:
        positions_by_chrom[chrom] = sorted(positions_by_chrom[chrom], key=lambda x: x[0])
        print(f"Chromosome {chrom}: {len(positions_by_chrom[chrom])} positions")
    
    print(f"Position dictionary built for {len(positions_by_chrom)} chromosomes")
    
    # Create a summary file with all variants in VCF format
    all_variants_vcf = os.path.join(output_dir, "all_variants.vcf")
    with open(all_variants_vcf, 'w') as f:
        # Write VCF header
        f.write("##fileformat=VCFv4.2\n")
        f.write("##source=SNVMatrixGenerator_input_variants\n")
        f.write("##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">\n")
        f.write("##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Frequency\">\n")
        f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        
        for chrom, pos_info_list in sorted(positions_by_chrom.items()):
            for pos, ref, alt in pos_info_list:
                snv_key = (chrom, pos, ref, alt)
                info_field = snv_info_dict[snv_key].info if snv_key in snv_info_dict and snv_info_dict[snv_key].info else "."
                f.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t{info_field}\n")
    
    # Compress and index the all_variants VCF
    compress_cmd = f"{PATH_CONFIG['BGZIP']} -f {all_variants_vcf}"
    if run_command(compress_cmd):
        all_variants_vcf_gz = all_variants_vcf + ".gz"
        index_cmd = f"{PATH_CONFIG['BCFTOOLS']} index -t {all_variants_vcf_gz}"
        run_command(index_cmd)
        print(f"Created input variants VCF: {all_variants_vcf_gz}")
    else:
        print(f"Warning: Failed to compress {all_variants_vcf}")

    
    # Sample a small subset of BAMs for initial testing
    if len(input_bams) > 10000:
        print("Large number of BAMs detected. Processing a sample of 5 first...")
        sample_bams = input_bams[:5]
        
        # Process sample BAMs sequentially for better debugging
        for bam in sample_bams:
            result = process_single_bam(bam, output_dir, positions_by_chrom)
            print(f"Sample result for {os.path.basename(bam)}: {result['status']}")
            print(f"  Detected SNVs: {len(result.get('detected_snvs', []))}")
            if result.get('detected_snvs'):
                print(f"  First few SNVs: {result['detected_snvs'][:3]}")
                # Test saving SNVs for the first sample
                save_success = save_detected_snvs(snv_vcf_dir, result, snv_info_dict)
                print(f"  Saved SNVs successfully: {save_success}")
                if not save_success:
                    print(f"  Failed to save SNVs to {snv_vcf_dir}")
    
    saved_bam_count = 0
    print(f"Starting parallel processing of {len(input_bams)} BAM files...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all BAM processing jobs
        future_to_bam = {
            executor.submit(process_single_bam, bam, output_dir, positions_by_chrom): bam 
            for bam in input_bams
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_bam), 
                         total=len(future_to_bam),
                         desc="Filtering BAM files"):
            bam = future_to_bam[future]
            try:
                result = future.result()
                results.append(result)
                
                # Save detected SNVs to barcode.vcf.gz file
                if save_detected_snvs(snv_vcf_dir, result, snv_info_dict):
                    saved_bam_count += 1
                    if saved_bam_count % 100 == 0:
                        print(f"Saved {saved_bam_count} BAM files with detected SNVs so far")
            except KeyboardInterrupt:
                print("KeyboardInterrupt: Stopping processing")
                print(f"Saved {saved_bam_count} BAM files with detected SNVs")
                break
            except Exception as e:
                print(f"Error processing BAM {os.path.basename(bam)}: {str(e)}")
                results.append({
                    'input_bam': bam,
                    'error': str(e),
                    'status': 'failed',
                    'detected_snvs': []
                })
    
    # Print statistics about detected SNVs
    bams_with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs'))
    print(f"\nBAMs with detected SNVs: {bams_with_snvs} out of {len(results)}")
    if len(results) > 0:
        print(f"Percentage with SNVs: {bams_with_snvs/len(results)*100:.2f}%")
        print(f"SNV VCF files saved: {saved_bam_count}")
    
    # Verify SNV VCF files were created
    if saved_bam_count > 0:
        snv_files = glob.glob(os.path.join(snv_vcf_dir, "*.vcf.gz"))
        print(f"Found {len(snv_files)} SNV VCF files in {snv_vcf_dir}")
        if len(snv_files) != saved_bam_count:
            print("Warning: Number of SNV VCF files doesn't match saved_bam_count")
    
    summary_file = create_all_variants_summary(output_dir)
    print(f"All variants summary saved to: {summary_file}")
    
    return results

def index_bams_in_directory(directory: str) -> List[Dict]:
    """
    Index all BAM files in the specified directory.
    
    Args:
        directory: Directory containing BAM files to index
        
    Returns:
        List of dictionaries with status information
    """
    # Find all BAM files
    bam_files = glob.glob(os.path.join(directory, '*.bam'))
    if not bam_files:
        print(f"No BAM files found in {directory}")
        return []
    
    print(f"Found {len(bam_files)} BAM files for indexing")
    results = []
    
    # Process files with progress bar
    for bam in tqdm(bam_files, desc="Indexing BAM files"):
        success = index_bam_file(bam)
        results.append({
            'input_bam': bam,
            'status': 'completed' if success else 'failed',
            'error': None if success else "Indexing failed"
        })
    
    # Summarize results
    successful = sum(1 for r in results if r['status'] == 'completed')
    failed = sum(1 for r in results if r['status'] == 'failed')
    
    print(f"\nIndexing completed:")
    print(f"  Successfully indexed: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result['status'] == 'failed':
                print(f"  {os.path.basename(result['input_bam'])}")
    
    return results

class SNVMatrixGenerator:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0",
                section_id: str = None, use_binary: bool = False,
                min_af_threshold: float = 0.2, classifier: str = "neural_network"):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.use_binary = use_binary
        self.min_af_threshold = min_af_threshold
        self.classifier = classifier  # Store the classifier type
        self.base_dir = "/data/maiziezhou_lab/leiy4/snv_calling"
        
        # Position dictionary that will be built once and reused
        self.positions_by_chrom = None
        
        self.validate_dataset_config()
        self.setup_paths()
        
    def validate_dataset_config(self):
        """Validate dataset configuration and section ID if required."""
        if self.dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
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
        
        # Input paths
        self.beagle_vcf = os.path.join(
            input_base, "output_VCFs/beagle",
            self.quality_filter, "all_filtered_in.vcf.gz"
        )
        # self.svm_vcf = os.path.join(
        #     input_base, "output_VCFs/SVMModel",
        #     self.quality_filter, "results/high_confidence.vcf.gz"
        # )
        # Add new paths for supplementary model outputs
        # In setup_paths() method       
        self.classifier_dir = os.path.join(
            input_base, "output_VCFs/Classifier",
            self.quality_filter
        )
        self.model_dir = os.path.join(self.classifier_dir, "results")
        self.classifier_homo_vcf = os.path.join(self.model_dir, f"{self.classifier}_homozygous.vcf.gz")
        self.classifier_hetero_vcf = os.path.join(self.model_dir, f"{self.classifier}_heterozygous.vcf.gz")
        
        # BAM directory based on dataset configuration
        if dataset_config["has_sections"]:
            if self.dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
                # For tumor datasets with specific BAM pattern
                self.bam_dir = os.path.join(
                    dataset_config["base_path"],
                    dataset_config["bam_pattern"].format(section_id=self.section_id)
                )
            else:
                self.bam_dir = os.path.join(
                    dataset_config["base_path"],
                    dataset_config["bam_pattern"].format(section_id=self.section_id)
                )
        else:
            self.bam_dir = os.path.join(
                dataset_config["base_path"],
                dataset_config["bam_pattern"]
            )
        
        # Output directory for filtered BAMs
        self.filtered_bam_dir = os.path.join(
            input_base, 
            "output_VCFs/BAM_filtered",
            self.quality_filter
        )
        os.makedirs(self.filtered_bam_dir, exist_ok=True)
        
        # Log directory
        self.log_dir = os.path.join(
            input_base,
            "logs/BAM_filtered",
            self.quality_filter
        )
        os.makedirs(self.log_dir, exist_ok=True)

    def count_genotypes(self, vcf_path: str) -> Tuple[int, int]:
        """Count the number of 0/1 and 1/1 genotypes in a VCF file."""
        count_0_1 = 0
        count_1_1 = 0
        
        if not os.path.exists(vcf_path):
            print(f"Warning: VCF file not found: {vcf_path}")
            return count_0_1, count_1_1
            
        with gzip.open(vcf_path, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                if self._is_valid_genotype(line):
                    fields = line.strip().split('\t')
                    format_fields = fields[8].split(':')
                    sample_fields = fields[9].split(':')
                    gt_idx = format_fields.index('GT')
                    gt = sample_fields[gt_idx]
                    if gt == '0/1':
                        count_0_1 += 1
                    elif gt == '1/1':
                        count_1_1 += 1
        return count_0_1, count_1_1

    def collect_snvs(self) -> Set[SNVInfo]:
        """Collect SNVs from both Beagle and Classifier outputs."""
        beagle_snvs = set()
        classifier_snvs = set()
        combined_snvs = set()
        heterozygous_count = 0
        homozygous_count = 0
        # Check if classifier VCF files exist
        vcf_files = {
            f"{self.classifier.upper()} Homozygous": self.classifier_homo_vcf,
            f"{self.classifier.upper()} Heterozygous": self.classifier_hetero_vcf,
            # beagle vcf
            "Beagle": self.beagle_vcf
        }

        for source, path in vcf_files.items():
            if not os.path.exists(path):
                print(f"Warning: {source} VCF file not found: {path}")
                continue
                
            # Count genotypes
            count_0_1, count_1_1 = self.count_genotypes(path)
            if source == f"{self.classifier.upper()} Homozygous":
                homozygous_count += count_0_1 + count_1_1
            elif source == f"{self.classifier.upper()} Heterozygous":
                heterozygous_count += count_0_1 + count_1_1
            else:
                heterozygous_count += count_0_1
                homozygous_count += count_1_1

            print(f"{source} VCF - 0/1: {count_0_1}, 1/1: {count_1_1}")
            
            # Collect SNVs
            source_snvs = set()
            with gzip.open(path, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    if self._is_valid_genotype(line):
                        snv = SNVInfo.from_vcf_line(line)
                        source_snvs.add(snv)
                        combined_snvs.add(snv)
            
            # Print sample keys for debugging
            print(f"SNVs collected from {source}: {len(source_snvs)}")
            sample_size = min(5, len(source_snvs))
            if sample_size > 0:
                print(f"Sample SNV keys from {source}:")
                sample_snvs = list(source_snvs)[:sample_size]
                for snv in sample_snvs:
                    print(f"  Original chrom: {snv.chrom}, Standardized: {snv.standardized_chrom}, Key: {snv.key}")
            
            # Store in source-specific set
            if source == "Beagle":
                beagle_snvs = source_snvs
            else:
                classifier_snvs = source_snvs
        
        # Print set overlap statistics for debugging
        if beagle_snvs and classifier_snvs:
            overlap = beagle_snvs.intersection(classifier_snvs)
            print(f"\nOverlap between Beagle and Classifier SNVs: {len(overlap)}")
            print(f"Classifier unique to Beagle: {len(beagle_snvs - overlap)}")
            print(f"SNVs unique to Classifier: {len(classifier_snvs - overlap)}")
        
        print(f"Total unique SNVs after combining sources: {len(combined_snvs)}")
        print(f"Total Heterozygous SNVs count: {heterozygous_count}")
        print(f"Total Homozygous SNVs count: {homozygous_count}")
        
        # Print chromosome distribution
        chrom_counts = {}
        for snv in combined_snvs:
            chrom = snv.standardized_chrom
            chrom_counts[chrom] = chrom_counts.get(chrom, 0) + 1
        
        print("\nChromosome distribution:")
        for chrom, count in sorted(chrom_counts.items()):
            print(f"  {chrom}: {count} SNVs")
            
        return combined_snvs

    def _is_valid_genotype(self, vcf_line: str) -> bool:
        """Check if genotype is 0/1 or 1/1."""
        try:
            fields = vcf_line.strip().split('\t')
            format_fields = fields[8].split(':')
            sample_fields = fields[9].split(':')
            gt_idx = format_fields.index('GT')
            gt = sample_fields[gt_idx]
            return gt in ['0/1', '1/1']
        except (ValueError, IndexError):
            return False

    def filter_bams(self, max_workers=30):
        """Filter BAM files based on SNV positions."""
        print("Collecting SNVs from VCF files...")
        snvs = self.collect_snvs()
        print(f"Found {len(snvs)} SNVs to use for filtering")

        # Get list of BAM files
        if '*' not in self.bam_dir:
            search_pattern = os.path.join(self.bam_dir, '*.bam')
        else:
            search_pattern = self.bam_dir
        
        bam_files = glob.glob(search_pattern)
        if not bam_files:
            raise FileNotFoundError(f"No BAM files found at: {search_pattern}")
        
        print(f"Found {len(bam_files)} BAM files to process")
        
        # Setup environment for samtools
        setup_environment()
        
        # Create output directories
        os.makedirs(self.filtered_bam_dir, exist_ok=True)
        
        # Create a directory for SNV VCF files
        snv_vcf_dir = os.path.join(os.path.dirname(self.filtered_bam_dir), "snv_vcf")
        os.makedirs(snv_vcf_dir, exist_ok=True)
        print(f"SNV VCF files will be saved to: {snv_vcf_dir}")
        
        # Filter BAMs in parallel
        results = filter_bams_parallel(
            input_bams=bam_files,
            output_dir=self.filtered_bam_dir,
            snvs=snvs,
            max_workers=max_workers
        )
        
        # Additional verification - check if SNV VCF files were created
        snv_files = glob.glob(os.path.join(snv_vcf_dir, "*.vcf.gz"))
        print(f"Found {len(snv_files)} SNV VCF files after processing")
        
        # Print summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs'))
        
        print("\nBAM Filtering Summary:")
        print(f"Total BAMs processed: {len(results)}")
        print(f"Successfully filtered: {completed}")
        print(f"Failed: {failed}")
        print(f"BAMs with detected SNVs: {with_snvs}")
        
        if failed > 0:
            print("\nFailed BAMs:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  {os.path.basename(result['input_bam'])}: {result.get('error', 'Unknown error')}")
        
        print(f"\nFiltered BAM files are located in: {self.filtered_bam_dir}")
        print(f"SNV VCF files are located in: {snv_vcf_dir}")
        
        # Write summary report
        summary_file = os.path.join(self.filtered_bam_dir, "filtering_summary.txt")
        with open(summary_file, 'w') as f:
            f.write("BAM Filtering Summary\n")
            f.write("===================\n\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            if self.section_id:
                f.write(f"Section ID: {self.section_id}\n")
            f.write(f"Quality Filter: {self.quality_filter}\n\n")
            f.write(f"Total SNVs used for filtering: {len(snvs)}\n")
            f.write(f"Total BAMs processed: {len(results)}\n")
            f.write(f"Successfully filtered: {completed}\n")
            f.write(f"BAMs with detected SNVs: {with_snvs}\n")
            f.write(f"Failed: {failed}\n\n")
            
            if failed > 0:
                f.write("Failed BAMs:\n")
                for result in results:
                    if result['status'] == 'failed':
                        f.write(f"  {os.path.basename(result['input_bam'])}: {result.get('error', 'Unknown error')}\n")
        
        return results
        
    def index_existing_bams(self):
        """Index existing BAM files in the filtered directory."""
        print(f"Indexing existing BAM files in {self.filtered_bam_dir}")
        
        # Setup environment for samtools
        setup_environment()
        
        # Index BAMs
        results = index_bams_in_directory(self.filtered_bam_dir)
        
        # Update summary file with indexing results
        summary_file = os.path.join(self.filtered_bam_dir, "filtering_summary.txt")
        with open(summary_file, 'a') as f:
            f.write("\n\nIndexing Summary (Latest Run)\n")
            f.write("==========================\n")
            f.write(f"Classifier: {self.classifier}\n")
            f.write(f"Total BAMs indexed: {len(results)}\n")
            f.write(f"Successfully indexed: {sum(1 for r in results if r['status'] == 'completed')}\n")
            f.write(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}\n")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Filter and index BAM files based on SNV positions")
    
    # Required arguments
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    
    # Optional arguments with defaults
    parser.add_argument("--section-id", 
                      help="Section ID (required for some datasets)")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter to use (default: baseQ0mapQ0)")
    parser.add_argument("--max-workers", type=int, default=30,
                      help="Maximum number of parallel workers (default: 30)")
    parser.add_argument("--min-af-threshold", type=float, default=0.2,
                      help="Minimum allele frequency threshold for binary mode (default: 0.2)")
    parser.add_argument("--index-only", action="store_true",
                      help="Only index existing BAM files without filtering")
    parser.add_argument("--classifier", default="neural_network", 
                  choices=["svm", "random_forest", "xgboost", "neural_network"],
                  help="Type of classifier to use for SNV filtering (default: neural_network)")                  
    
    args = parser.parse_args()
    
    # Validate section ID requirement
    dataset_config = DATASET_CONFIGS[args.dataset]
    if dataset_config["has_sections"] and not args.section_id:
        if "section_ids" in dataset_config:
            valid_sections = dataset_config["section_ids"]
            parser.error(f"Dataset {args.dataset} requires --section-id. Valid values: {valid_sections}")
        else:
            parser.error(f"Dataset {args.dataset} requires --section-id")
            
    # Print configuration
    print("\nBAM Processing Configuration:")
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Max Workers: {args.max_workers}")
    if args.index_only:
        print(f"Mode: Index existing BAM files only")
    else:
        print(f"Mode: Filter and index BAM files")
    print("\n")
    
    # Initialize generator
    generator = SNVMatrixGenerator(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        min_af_threshold=args.min_af_threshold,
        classifier=args.classifier
    )
    
    if args.index_only:
        # Only index existing BAMs
        results = generator.index_existing_bams()
    else:
        # Filter and index BAMs
        results = generator.filter_bams(max_workers=args.max_workers)
    
    # Exit with error if any BAMs failed
    if any(r['status'] == 'failed' for r in results):
        exit(1)
    
if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC :
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset DLPFC --section-id 151508 --quality-filter baseQ0mapQ0 --classifier neural_network

# For DLPFC (b0m0):
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset DLPFC --section-id 151669 --quality-filter baseQ0mapQ0 --classifier neural_network

# For P4_TUMOR:
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset P4_TUMOR --section-id 1 --quality-filter baseQ0mapQ0 --classifier neural_network

# For P6_TUMOR:
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset P6_TUMOR --section-id 1 --quality-filter baseQ0mapQ0