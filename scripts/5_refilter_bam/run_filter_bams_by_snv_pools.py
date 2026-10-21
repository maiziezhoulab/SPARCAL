import os
import gzip
import json
import hashlib
import math
import re
import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import glob
import subprocess
import tempfile
from datetime import datetime, timezone


SUPPORTED_BEAGLE_ROUTER_SCHEMAS = frozenset({
    "sparcal_beagle_evidence_v2_1",
    "sparcal_beagle_evidence_v2_2",
})
BEAGLE_ROUTE_VCF_NAMES = (
    "all_beagle_nonref_highconf.vcf.gz",
    "all_beagle_ref_highconf.vcf.gz",
    "all_beagle_uncertain.vcf.gz",
    "all_panel_absent.vcf.gz",
)
BEAGLE_COMPATIBILITY_VCF_NAMES = (
    "all_filtered_in.vcf.gz",
    "all_filtered_out.vcf.gz",
    "all_panel_discordant.vcf.gz",
)
PANEL_PRESENT_SCHEMA = "sparcal_beagle_evidence_v2_2"
PANEL_PRESENT_VCF_NAME = "all_panel_present.vcf.gz"
SPOTPROFILE_SCHEMA = "sparcal_spotprofiles_v1"
PRODUCTION_MIN_ANCHOR_PROBABILITY = 0.90
ROUTE_TO_FILENAME = {
    "beagle_nonref_highconf": "all_beagle_nonref_highconf.vcf.gz",
    "beagle_ref_highconf": "all_beagle_ref_highconf.vcf.gz",
    "beagle_uncertain": "all_beagle_uncertain.vcf.gz",
    "panel_absent": "all_panel_absent.vcf.gz",
}


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
            f.write("##INFO=<ID=RACE,Number=1,Type=String,Description=\"SNV origin: defined (panel-concordant), panel_discordant, or denovo (panel-absent)\">\n")
            f.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            f.write("##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths for the ref and alt alleles\">\n")
            f.write("##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">\n")
            f.write("##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Genotype Quality\">\n")
            f.write("##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods\">\n")
            f.write(f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{barcode}\n")
            
            # Per-spot allele depths from the BAM scan (chrom,pos,ref,alt)->(ref_ct, alt_ct)
            allele_counts = result.get('allele_counts', {})

            # Write variant records
            for snv_tuple in sorted_snvs:
                if len(snv_tuple) >= 4:
                    chrom, pos, ref, alt = snv_tuple[:4]
                    ac_key = (chrom, pos, ref, alt)   # key as stored (pre chr-prefix)

                    if not chrom.startswith("chr"):
                        chrom = f"chr{chrom}"
                    # Try to get additional info from snv_info_dict if provided
                    info_field = "."
                    race = "unknown"

                    if snv_info_dict:
                        # Create a standardized key for lookup
                        standardized_chrom = chrom.replace("chr", "")
                        snv_key = (standardized_chrom, pos, ref, alt)

                        if snv_key in snv_info_dict:
                            snv_info = snv_info_dict[snv_key]
                            info_field = snv_info.info if snv_info.info else "."
                            race = snv_info.race if snv_info.race else "unknown"

                    # Add RACE to INFO field (INFO stays pseudobulk-derived; AD below is per-spot)
                    if info_field == ".":
                        info_field = f"RACE={race}"
                    else:
                        info_field = f"{info_field};RACE={race}"

                    # Per-spot FORMAT: real GT:AD:DP from this spot's reads (was "./.")
                    rc, alt_c = allele_counts.get(ac_key, (None, None))
                    if rc is None:
                        format_field, sample_field = "GT", "./."
                    else:
                        dp = rc + alt_c
                        gt = "0/0" if alt_c == 0 else ("1/1" if rc == 0 else "0/1")
                        format_field = "GT:AD:DP"
                        sample_field = f"{gt}:{rc},{alt_c}:{dp}"

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
            print(f"Error: Failed to index VCF for {barcode}")
            return False

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

# DLPFC_PREDEDUP=1 -> read the ORIGINAL non-dedup per-cell BAMs and write to a
# SEPARATE tree (data/dlpfc_prededup) so the post-dedup results under data/dlpfc
# are left untouched. Default: unchanged.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        # UMI-deduped per-cell BAMs (run_slurm/dlpfc/0_umidedup_split_DLPFC.sh).
        # Non-dedup source (rollback): base_path/{section_id}/bam_bycell/*.bam
        "bam_base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD" if _PREDEDUP else "/data/maiziezhou_lab/leiy4/snv_calling",
        "bam_pattern": "{section_id}/bam_bycell/*.bam" if _PREDEDUP else "data/dlpfc/{section_id}/bam_bycell_dedup/*.bam",
        "output_dir": "data/dlpfc_prededup/{section_id}" if _PREDEDUP else "data/dlpfc/{section_id}",
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
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "bam_pattern": "DCIS{section_id}/spaceranger_align_DCIS{section_id}_hg38/DCIS{section_id}_output/outs/split_BAM/",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "FFPE_VISIUM",  # Uses GRCh38 with chr prefix
        "multiple_bams": True
    },
    "OVAR_P5": {
        # Per-barcode split BAMs live in OUR project (colleague outs/ is read-only),
        # so base_path is PROJECT_DIR and bam_pattern points at data/ovar_p5/.../split_BAM.
        "base_path": "/data/maiziezhou_lab/leiy4/snv_calling",
        # Rollback (July, pre-dedup): data/ovar_p5/{section_id}/split_BAM/
        "bam_pattern": "data/ovar_p5/{section_id}/split_BAM_dedup/",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "FFPE_VISIUM",  # GRCh38 with chr prefix
        "multiple_bams": True
    },
    "NCCE": {
        # Split BAMs are generated under our project from read-only SpaceRanger outs.
        "base_path": "/data/maiziezhou_lab/leiy4/snv_calling",
        "bam_pattern": "data/ncce/{section_id}/split_BAM/",
        "output_dir": "data/ncce/{section_id}",
        "has_sections": True,
        "section_ids": ["6A", "6B", "6C", "8A", "8B", "8C", "10A", "10B", "10C", "11A", "11B", "11C", "14A", "14B", "14C"],
        "reference": "FFPE_VISIUM",
        "multiple_bams": True
    }
}
# For reference, TUMOR means hg19, FFPE means hg38 with 'chr', DLPFC is hg38 without 'chr'

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
    race: str = ""  # "defined" (from Beagle) or "denovo" (from Classifier)
    
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

def filter_bam_one_chrom(
    input_bam: str,
    chrom: str,
    alleles_by_pos: Dict[int, List[Tuple[str, str]]],
) -> Tuple[
    List[pysam.AlignedSegment],
    Set[Tuple[int, str, str]],
    Dict[Tuple[int, str, str], List[int]],
]:
    """
    Filter one chromosome and count every exact REF/ALT allele independently.

    Args:
        input_bam: Path to input BAM file
        chrom: Chromosome name to filter
        alleles_by_pos: ``position -> [(ref, alt), ...]`` for every exact allele
            at that 1-based position.

    Returns:
        ``(filtered_reads, detected_alleles, allele_counts)`` where both allele
        collections retain ``(position, ref, alt)``. A read aligned at a shared
        position contributes independently to every candidate allele there.

    Note: a read is now recorded against EVERY SNV position it covers (the old
    code `break`-ed after the leftmost covered SNV, so reads spanning multiple
    SNVs registered only the first — under-detecting in dense regions). This
    fixes that and is required for correct per-allele counts.
    """
    normalized_alleles = {
        int(pos): tuple(sorted(set(alleles)))
        for pos, alleles in alleles_by_pos.items()
    }
    positions = sorted(normalized_alleles)
    filtered_reads = []
    detected_alleles: Set[Tuple[int, str, str]] = set()
    allele_counts: Dict[Tuple[int, str, str], List[int]] = {}
    start_j = 0  # Starting position in the sorted positions list

    try:
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            for read in in_bam.fetch(chrom):
                if read.is_unmapped:
                    continue

                read_start = read.reference_start + 1
                # pysam reference_end is 0-based exclusive; numerically it is
                # the last covered coordinate in a 1-based inclusive system.
                read_end = read.reference_end if read.reference_end is not None else read_start

                while start_j < len(positions) and positions[start_j] < read_start:
                    start_j += 1

                covered = []
                j = start_j
                while j < len(positions) and positions[j] <= read_end:
                    pos = positions[j]
                    if read_start <= pos <= read_end:
                        covered.append(pos)
                    j += 1

                if not covered:
                    continue
                filtered_reads.append(read)

                if read.query_sequence is None:
                    continue
                seq = read.query_sequence
                refpos = read.get_reference_positions(full_length=True)
                ref2q = {rp: qp for qp, rp in enumerate(refpos) if rp is not None}
                for pos in covered:
                    qp = ref2q.get(pos - 1)
                    if qp is None:
                        continue
                    base = seq[qp].upper()
                    for ref, alt in normalized_alleles[pos]:
                        key = (pos, ref, alt)
                        detected_alleles.add(key)
                        ac = allele_counts.setdefault(key, [0, 0])
                        if base == alt:
                            ac[1] += 1
                        elif base == ref:
                            ac[0] += 1
    except Exception as e:
        raise RuntimeError(
            f"failed processing chromosome {chrom} in {input_bam}: {e}"
        ) from e

    return filtered_reads, detected_alleles, allele_counts

def create_all_variants_summary(merged_output_dir: str, vcf_output_dir: str):
    """
    Create a summary VCF file with all detected variants across all barcodes.
    
    Args:
        merged_output_dir: Directory to save the merged summary file (root level)
        vcf_output_dir: Directory containing per-spot SNV VCF files (vcf_by_spot)
    """
    summary_file = os.path.join(merged_output_dir, "all_detected_variants_summary.vcf")
    summary_file_gz = summary_file + ".gz"
    
    # Dictionary to track variants and their counts
    all_variants = {}  # (chrom, pos, ref, alt) -> (count, info, format)
    barcode_count = 0
    
    # Process each barcode VCF file
    for vcf_file in sorted(glob.glob(os.path.join(vcf_output_dir, "*.vcf.gz"))):
        barcode_count += 1
        try:
            with gzip.open(vcf_file, 'rt') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) != 10:
                        raise ValueError(
                            f"malformed single-sample VCF record with {len(parts)} fields"
                        )

                    chrom, pos, id_field, ref, alt = parts[:5]
                    try:
                        pos_int = int(pos)
                    except ValueError as error:
                        raise ValueError(f"invalid POS {pos!r}") from error
                    if pos_int <= 0 or chrom in {'', '.'} or ref in {'', '.'} or alt in {'', '.'}:
                        raise ValueError("invalid allele fields")
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
            raise ValueError(f"Failed to read per-spot VCF {vcf_file}: {e}") from e
    
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
    if not run_command(compress_cmd):
        raise RuntimeError(f"Failed to compress summary VCF: {summary_file}")
    index_cmd = f"{PATH_CONFIG['BCFTOOLS']} index -t {summary_file_gz}"
    if not run_command(index_cmd):
        raise RuntimeError(f"Failed to index summary VCF: {summary_file_gz}")

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
        all_allele_counts = {}  # (chrom, pos, ref, alt) -> (ref_count, alt_count)
        
        # Process one chromosome at a time
        with pysam.AlignmentFile(input_bam, "rb") as in_bam:
            # Check which chromosomes are actually in the BAM file
            bam_references = set(in_bam.references)
            
            # Open output file
            with pysam.AlignmentFile(temp_output, "wb", header=in_bam.header) as out_bam:
                # Process each chromosome
                for chrom, pos_info_list in sorted(positions_by_chrom.items()):
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
                    
                    alleles_by_pos = defaultdict(list)
                    for pos, ref, alt in pos_info_list:
                        alleles_by_pos[pos].append((ref, alt))

                    # Filter reads and tally every exact allele independently.
                    filtered_reads, detected_alleles, allele_counts = filter_bam_one_chrom(
                        input_bam, bam_chrom, alleles_by_pos
                    )

                    for pos, ref, alt in detected_alleles:
                        detected_snvs.add((bam_chrom, pos, ref, alt))
                        rc, ac = allele_counts.get((pos, ref, alt), [0, 0])
                        all_allele_counts[(bam_chrom, pos, ref, alt)] = (rc, ac)
                    
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
            'detected_snvs': list(detected_snvs),  # Convert set to list for returning
            'allele_counts': all_allele_counts     # (chrom,pos,ref,alt) -> (ref_count, alt_count)
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
                      max_workers: int = 30, vcf_output_dir: str = None, 
                      merged_output_dir: str = None) -> List[Dict]:
    """
    Filter multiple BAM files in parallel.
    
    Args:
        input_bams: List of paths to input BAM files
        output_dir: Directory for output BAM files (bam_by_spot)
        snvs: Set of SNVInfo objects defining positions to keep
        max_workers: Maximum number of parallel workers
        vcf_output_dir: Directory for per-spot VCF files (vcf_by_spot)
        merged_output_dir: Directory for merged summary files (root level)
        
    Returns:
        List of dictionaries with status information
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Use provided directories or default to subdirectories of output_dir
    if vcf_output_dir is None:
        vcf_output_dir = os.path.join(output_dir, "vcf_by_spot")
    if merged_output_dir is None:
        merged_output_dir = os.path.dirname(output_dir)
    
    os.makedirs(vcf_output_dir, exist_ok=True)
    os.makedirs(merged_output_dir, exist_ok=True)
    
    print(f"Per-spot BAM files will be saved to: {output_dir}")
    print(f"Per-spot VCF files will be saved to: {vcf_output_dir}")
    print(f"Merged files will be saved to: {merged_output_dir}")
    
    results = []
    
    # Create position dictionary once for all BAM files
    print("Building SNV position dictionary...")
    positions_by_chrom = defaultdict(list)
    snv_info_dict = {}  # Dictionary to store SNVInfo objects for later retrieval
    
    ordered_snvs = sorted(
        snvs, key=lambda snv: (snv.standardized_chrom, snv.pos, snv.ref, snv.alt)
    )
    for snv in ordered_snvs:
        # Use standardized chromosome name
        chrom = snv.standardized_chrom
        # Include ref and alt information
        positions_by_chrom[chrom].append((snv.pos, snv.ref, snv.alt))
        # Store the SNVInfo object for later use
        snv_key = (chrom, snv.pos, snv.ref, snv.alt)
        snv_info_dict[snv_key] = snv
    
    # Sort positions for each chromosome
    for chrom in positions_by_chrom:
        positions_by_chrom[chrom] = sorted(
            set(positions_by_chrom[chrom]), key=lambda item: (item[0], item[1], item[2])
        )
        print(f"Chromosome {chrom}: {len(positions_by_chrom[chrom])} positions")
    
    print(f"Position dictionary built for {len(positions_by_chrom)} chromosomes")
    
    # Create a summary file with all variants in VCF format (saved to merged directory)
    all_variants_vcf = os.path.join(merged_output_dir, "all_variants.vcf")
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
    if not run_command(compress_cmd):
        raise RuntimeError(f"Failed to compress input-variant VCF: {all_variants_vcf}")
    all_variants_vcf_gz = all_variants_vcf + ".gz"
    index_cmd = f"{PATH_CONFIG['BCFTOOLS']} index -t {all_variants_vcf_gz}"
    if not run_command(index_cmd):
        raise RuntimeError(f"Failed to index input-variant VCF: {all_variants_vcf_gz}")
    print(f"Created input variants VCF: {all_variants_vcf_gz}")

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
                
                # Save detected SNVs to barcode.vcf.gz file. A completed BAM
                # with detected variants is not successful unless its VCF and
                # tabix index were both written.
                if result['status'] == 'completed' and result.get('detected_snvs'):
                    result['vcf_saved'] = save_detected_snvs(
                        vcf_output_dir, result, snv_info_dict
                    )
                    if result['vcf_saved']:
                        saved_bam_count += 1
                        if saved_bam_count % 100 == 0:
                            print(
                                f"Saved {saved_bam_count} BAM files with detected SNVs so far"
                            )
                    else:
                        result['status'] = 'failed'
                        result['error'] = 'Failed to write/index per-spot VCF'
                else:
                    result['vcf_saved'] = None
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
    
    # Verify the fresh directory contains exactly the VCFs reported saved.
    snv_files = sorted(glob.glob(os.path.join(vcf_output_dir, "*.vcf.gz")))
    print(f"Found {len(snv_files)} SNV VCF files in {vcf_output_dir}")
    if len(snv_files) != saved_bam_count:
        raise RuntimeError(
            "Per-spot VCF count disagrees with successful writes: "
            f"{len(snv_files)} != {saved_bam_count}"
        )
    missing_indices = [path + ".tbi" for path in snv_files if not os.path.isfile(path + ".tbi")]
    if missing_indices:
        raise RuntimeError(
            f"Missing tabix index for {len(missing_indices)} per-spot VCF(s); "
            f"first={missing_indices[0]}"
        )
    
    summary_file = create_all_variants_summary(merged_output_dir, vcf_output_dir)
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
                min_af_threshold: float = 0.2, classifier: str = "neural_network",
                section_root: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.use_binary = use_binary
        self.min_af_threshold = min_af_threshold
        self.classifier = classifier  # Store the classifier type
        self.section_root = os.path.abspath(section_root) if section_root else None
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
            
        input_base = self.section_root or os.path.join(self.base_dir, data_path)
        
        # Input paths
        self.beagle_bundle_dir = os.path.join(
            input_base, "output_VCFs/beagle", self.quality_filter)
        self.beagle_vcf = os.path.join(self.beagle_bundle_dir, "all_filtered_in.vcf.gz")
        self.panel_discordant_vcf = os.path.join(self.beagle_bundle_dir, "all_panel_discordant.vcf.gz")
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
        
        # BAM directory based on dataset configuration.
        # bam_base_path (optional) decouples the BAM root from base_path so DLPFC can
        # read its UMI-deduped BAMs from the project dir (see mpileup_pipeline.py).
        bam_base_path = dataset_config.get("bam_base_path", dataset_config["base_path"])
        if dataset_config["has_sections"]:
            if self.dataset_name in ["P4_TUMOR", "P6_TUMOR"]:
                # For tumor datasets with specific BAM pattern
                self.bam_dir = os.path.join(
                    bam_base_path,
                    dataset_config["bam_pattern"].format(section_id=self.section_id)
                )
            else:
                self.bam_dir = os.path.join(
                    bam_base_path,
                    dataset_config["bam_pattern"].format(section_id=self.section_id)
                )
        else:
            self.bam_dir = os.path.join(
                bam_base_path,
                dataset_config["bam_pattern"]
            )
        
        # Output directory for filtered BAMs (now using spotprofiles)
        self.filtered_bam_dir = os.path.join(
            input_base, 
            "output_VCFs/spotprofiles",
            self.quality_filter
        )
        
        # Create subdirectories for per-spot outputs
        self.bam_by_spot_dir = os.path.join(self.filtered_bam_dir, "bam_by_spot")
        self.vcf_by_spot_dir = os.path.join(self.filtered_bam_dir, "vcf_by_spot")
        
        # Log directory
        self.log_dir = os.path.join(
            input_base,
            "logs/spotprofiles",
            self.quality_filter
        )

    @staticmethod
    def _parse_router_header(line: str, vcf_path: Path) -> dict:
        """Parse the versioned router meta-header without trusting field order."""
        match = re.fullmatch(r"##SPARCAL_BEAGLE_ROUTER=<(.+)>", line.rstrip("\n"))
        if match is None:
            raise ValueError(f"malformed SPARCAL Beagle router header: {vcf_path}")
        values = {}
        for field in re.finditer(r'(\w+)=("[^"]*"|[^,>]*)', match.group(1)):
            value = field.group(2)
            values[field.group(1)] = value[1:-1] if value.startswith('"') else value
        required = {"Schema", "AnchorScore", "MinAnchorProbability", "HardAnchor"}
        missing = sorted(required - set(values))
        if missing:
            raise ValueError(
                f"router header in {vcf_path} is missing: {', '.join(missing)}"
            )
        try:
            values["MinAnchorProbability"] = float(values["MinAnchorProbability"])
        except ValueError as error:
            raise ValueError(
                f"invalid MinAnchorProbability in router header: {vcf_path}"
            ) from error
        return values

    @classmethod
    def _scan_routed_vcf(
        cls,
        vcf_path: Path,
        expected_schema: str,
        allowed_routes: Set[str],
    ) -> Tuple[Set[Tuple[str, int, str, str]], Dict[str, int]]:
        """Read one routed VCF completely and return exact records by route."""
        router_contract = None
        saw_chrom_header = False
        keys = set()
        route_counts = defaultdict(int)
        try:
            with gzip.open(vcf_path, "rt") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if line.startswith("##SPARCAL_BEAGLE_ROUTER=<"):
                        if router_contract is not None:
                            raise ValueError(
                                f"duplicate router header in {vcf_path}:{line_number}"
                            )
                        router_contract = cls._parse_router_header(line, vcf_path)
                        continue
                    if line.startswith("#CHROM"):
                        fields = line.rstrip("\n").split("\t")
                        if len(fields) != 10:
                            raise ValueError(
                                f"{vcf_path}:{line_number}: expected exactly one sample"
                            )
                        saw_chrom_header = True
                        continue
                    if line.startswith("#"):
                        continue
                    if not saw_chrom_header:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: data before #CHROM header"
                        )
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 10:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: malformed VCF record with "
                            f"{len(fields)} fields"
                        )
                    try:
                        pos = int(fields[1])
                    except ValueError as error:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: invalid POS {fields[1]!r}"
                        ) from error
                    if pos <= 0 or fields[0] in {"", "."} or fields[3] in {"", "."} \
                            or fields[4] in {"", "."}:
                        raise ValueError(f"{vcf_path}:{line_number}: invalid exact allele")
                    key = (fields[0], pos, fields[3], fields[4])
                    if key in keys:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: duplicate exact allele {key}"
                        )
                    info = {}
                    for item in fields[7].split(";"):
                        if "=" in item:
                            name, value = item.split("=", 1)
                            info[name] = value
                    route = info.get("BEAGLE_ROUTE")
                    if route not in allowed_routes:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: route {route!r} not in "
                            f"{sorted(allowed_routes)}"
                        )
                    if info.get("BEAGLE_EVIDENCE_SCHEMA") != expected_schema:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: evidence schema disagrees "
                            "with manifest"
                        )
                    keys.add(key)
                    route_counts[route] += 1
        except (OSError, EOFError, UnicodeError) as error:
            raise ValueError(f"unreadable routed VCF: {vcf_path}: {error}") from error
        if router_contract is None:
            raise ValueError(f"missing SPARCAL Beagle router header: {vcf_path}")
        if not saw_chrom_header:
            raise ValueError(f"missing #CHROM header: {vcf_path}")
        if (
            router_contract["Schema"] != expected_schema
            or router_contract["AnchorScore"] != "p_nonref"
            or not math.isclose(
                router_contract["MinAnchorProbability"],
                PRODUCTION_MIN_ANCHOR_PROBABILITY,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or router_contract["HardAnchor"] != "beagle_nonref_highconf"
        ):
            raise ValueError(
                f"routed VCF contract disagrees with production manifest: {vcf_path}"
            )
        return keys, dict(route_counts)

    @classmethod
    def _read_router_contract(cls, vcf_path: Path) -> Tuple[str, str, float]:
        """Return schema, score and probability from a routed VCF header."""
        try:
            with gzip.open(vcf_path, "rt") as handle:
                for line in handle:
                    if line.startswith("##SPARCAL_BEAGLE_ROUTER=<"):
                        values = cls._parse_router_header(line, vcf_path)
                        return (
                            values["Schema"], values["AnchorScore"],
                            values["MinAnchorProbability"],
                        )
                    if line.startswith("#CHROM"):
                        break
        except (OSError, EOFError, UnicodeError) as error:
            raise ValueError(f"unreadable routed VCF: {vcf_path}: {error}") from error
        raise ValueError(f"missing SPARCAL Beagle router header: {vcf_path}")

    @staticmethod
    def _validate_router_summary(bundle: Path, manifest: dict) -> None:
        summary_path = bundle / "beagle_evidence_summary.tsv"
        expected_header = [
            "schema", "route", "count", "anchor_score",
            "min_anchor_probability", "hard_anchor",
        ]
        observed = {}
        try:
            with summary_path.open() as handle:
                header = handle.readline().rstrip("\n").split("\t")
                if header != expected_header:
                    raise ValueError(f"invalid router summary header: {summary_path}")
                for line_number, line in enumerate(handle, start=2):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != len(expected_header):
                        raise ValueError(
                            f"{summary_path}:{line_number}: malformed summary row"
                        )
                    schema, route, count, score, probability, hard_anchor = fields
                    if route in observed:
                        raise ValueError(
                            f"{summary_path}:{line_number}: duplicate route {route}"
                        )
                    try:
                        count = int(count)
                        probability = float(probability)
                    except ValueError as error:
                        raise ValueError(
                            f"{summary_path}:{line_number}: non-numeric count/probability"
                        ) from error
                    if (
                        schema != manifest["schema"]
                        or score != "p_nonref"
                        or not math.isclose(
                            probability, PRODUCTION_MIN_ANCHOR_PROBABILITY,
                            rel_tol=0.0, abs_tol=1e-12,
                        )
                    ):
                        raise ValueError(
                            f"{summary_path}:{line_number}: non-production router contract"
                        )
                    expected_hard = "1" if route == "beagle_nonref_highconf" else (
                        "." if route == "total" else "0"
                    )
                    if hard_anchor != expected_hard:
                        raise ValueError(
                            f"{summary_path}:{line_number}: invalid hard-anchor flag"
                        )
                    observed[route] = count
        except OSError as error:
            raise ValueError(f"unreadable router summary: {summary_path}: {error}") from error
        expected = {
            route: int(manifest["counts"][route]) for route in ROUTE_TO_FILENAME
        }
        expected["total"] = int(manifest["counts"]["total"])
        if observed != expected:
            raise ValueError(
                f"router summary counts disagree with manifest: {observed} != {expected}"
            )

    @staticmethod
    def _validate_router_audit(bundle: Path, manifest: dict) -> None:
        audit_path = bundle / "beagle_evidence_audit.tsv.gz"
        expected_columns = [
            "schema", "chrom", "pos", "ref", "alt", "raw_gt", "beagle_gt",
            "beagle_gp", "beagle_called_gp", "beagle_p_nonref",
            "beagle_gp_entropy", "beagle_gp_margin", "beagle_af",
            "gt_transition", "route", "hard_anchor",
        ]
        route_counts = defaultdict(int)
        try:
            with gzip.open(audit_path, "rt") as handle:
                header = handle.readline().rstrip("\n").split("\t")
                if header != expected_columns:
                    raise ValueError(f"invalid router audit header: {audit_path}")
                for line_number, line in enumerate(handle, start=2):
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != len(expected_columns):
                        raise ValueError(
                            f"{audit_path}:{line_number}: malformed audit row"
                        )
                    route = fields[14]
                    if route not in ROUTE_TO_FILENAME:
                        raise ValueError(
                            f"{audit_path}:{line_number}: unknown route {route!r}"
                        )
                    if fields[0] != manifest["schema"]:
                        raise ValueError(
                            f"{audit_path}:{line_number}: schema disagrees with manifest"
                        )
                    route_counts[route] += 1
        except (OSError, EOFError, UnicodeError) as error:
            raise ValueError(f"unreadable router audit: {audit_path}: {error}") from error
        expected = {
            route: int(manifest["counts"][route]) for route in ROUTE_TO_FILENAME
        }
        if dict(route_counts) != {route: count for route, count in expected.items() if count}:
            raise ValueError(
                f"router audit counts disagree with manifest: {dict(route_counts)} != {expected}"
            )

    def validate_beagle_route_bundle(self) -> dict:
        """Validate every record in the production P(nonref)>=0.90 route bundle."""
        bundle = Path(self.beagle_bundle_dir)
        vcf_paths = [
            bundle / name
            for name in (*BEAGLE_ROUTE_VCF_NAMES, *BEAGLE_COMPATIBILITY_VCF_NAMES)
        ]
        required = [
            *vcf_paths,
            *(Path(str(path) + ".tbi") for path in vcf_paths),
            bundle / "beagle_evidence_audit.tsv.gz",
            bundle / "beagle_evidence_summary.tsv",
            bundle / "MANIFEST.json",
            bundle / "RUN_COMPLETE",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"incomplete Beagle four-route bundle under {bundle}; missing: {preview}"
            )
        if (bundle / "RUN_COMPLETE").read_text().strip() != "complete":
            raise ValueError(f"invalid Beagle RUN_COMPLETE sentinel: {bundle}")

        try:
            manifest = json.loads((bundle / "MANIFEST.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid Beagle route manifest under {bundle}: {error}") from error
        schema = manifest.get("schema")
        if schema not in SUPPORTED_BEAGLE_ROUTER_SCHEMAS:
            raise ValueError(f"unsupported Beagle router schema {schema!r} under {bundle}")
        if schema == PANEL_PRESENT_SCHEMA:
            panel_present_path = bundle / PANEL_PRESENT_VCF_NAME
            panel_present_required = (
                panel_present_path, Path(str(panel_present_path) + ".tbi"),
            )
            panel_present_missing = [
                str(path) for path in panel_present_required if not path.is_file()
            ]
            if panel_present_missing:
                raise FileNotFoundError(
                    f"incomplete Beagle panel-present compatibility view under "
                    f"{bundle}; missing: {', '.join(panel_present_missing)}"
                )
        if manifest.get("anchor_score") != "p_nonref":
            raise ValueError(f"Beagle bundle is not P(nonref)-routed: {bundle}")
        try:
            min_anchor_probability = float(manifest["min_anchor_probability"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Beagle manifest has no numeric min_anchor_probability: {bundle}"
            ) from error
        if not math.isclose(
            min_anchor_probability, PRODUCTION_MIN_ANCHOR_PROBABILITY,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(
                f"Beagle bundle threshold must be P(nonref)>=0.90, found "
                f"{min_anchor_probability:g}: {bundle}"
            )
        if manifest.get("hard_anchor_route") != "beagle_nonref_highconf":
            raise ValueError(f"unexpected Beagle hard-anchor route under {bundle}")
        if manifest.get("production_inputs_include_wes") is not False:
            raise ValueError(f"Beagle bundle does not declare WES-free production input: {bundle}")

        counts = manifest.get("counts")
        if not isinstance(counts, dict):
            raise ValueError(f"Beagle manifest has no route counts: {bundle}")
        try:
            expected_counts = {
                route: int(counts[route]) for route in ROUTE_TO_FILENAME
            }
            total = int(counts["total"])
            raw_record_count = int(manifest["raw_record_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid Beagle manifest route counts: {bundle}") from error
        if any(count < 0 for count in expected_counts.values()) or total < 0:
            raise ValueError(f"negative Beagle manifest count under {bundle}")
        if sum(expected_counts.values()) != total or raw_record_count != total:
            raise ValueError(
                "Beagle manifest route counts do not conserve raw records: "
                f"routes={sum(expected_counts.values())}, total={total}, "
                f"raw={raw_record_count}"
            )

        route_sets = {}
        for route, filename in ROUTE_TO_FILENAME.items():
            keys, observed_routes = self._scan_routed_vcf(
                bundle / filename, schema, {route}
            )
            if observed_routes.get(route, 0) != expected_counts[route]:
                raise ValueError(
                    f"actual {route} VCF count disagrees with manifest: "
                    f"{len(keys)} != {expected_counts[route]}"
                )
            route_sets[route] = keys
        routes = list(route_sets)
        for index, left in enumerate(routes):
            for right in routes[index + 1:]:
                overlap = route_sets[left] & route_sets[right]
                if overlap:
                    raise ValueError(
                        f"Beagle route VCFs are not disjoint ({left}, {right}); "
                        f"example={next(iter(overlap))}"
                    )
        route_union = set().union(*route_sets.values())
        if len(route_union) != total:
            raise ValueError(
                f"actual Beagle route union does not conserve manifest total: "
                f"{len(route_union)} != {total}"
            )

        compatibility_expectations = {
            "all_filtered_in.vcf.gz": route_sets["beagle_nonref_highconf"],
            "all_filtered_out.vcf.gz": (
                route_sets["beagle_ref_highconf"]
                | route_sets["beagle_uncertain"]
                | route_sets["panel_absent"]
            ),
            "all_panel_discordant.vcf.gz": (
                route_sets["beagle_ref_highconf"]
                | route_sets["beagle_uncertain"]
            ),
        }
        compatibility_routes = {
            "all_filtered_in.vcf.gz": {"beagle_nonref_highconf"},
            "all_filtered_out.vcf.gz": {
                "beagle_ref_highconf", "beagle_uncertain", "panel_absent",
            },
            "all_panel_discordant.vcf.gz": {
                "beagle_ref_highconf", "beagle_uncertain",
            },
        }
        if schema == PANEL_PRESENT_SCHEMA:
            panel_present_routes = {
                "beagle_nonref_highconf",
                "beagle_ref_highconf",
                "beagle_uncertain",
            }
            compatibility_expectations[PANEL_PRESENT_VCF_NAME] = (
                route_sets["beagle_nonref_highconf"]
                | route_sets["beagle_ref_highconf"]
                | route_sets["beagle_uncertain"]
            )
            compatibility_routes[PANEL_PRESENT_VCF_NAME] = panel_present_routes
            declared_compatibility = manifest.get("compatibility")
            if (
                not isinstance(declared_compatibility, dict)
                or PANEL_PRESENT_VCF_NAME not in declared_compatibility
            ):
                raise ValueError(
                    f"Beagle v2.2 manifest does not declare {PANEL_PRESENT_VCF_NAME}: "
                    f"{bundle}"
                )
        for filename, expected_keys in compatibility_expectations.items():
            observed_keys, _ = self._scan_routed_vcf(
                bundle / filename, schema, compatibility_routes[filename]
            )
            if observed_keys != expected_keys:
                missing_keys = expected_keys - observed_keys
                extra_keys = observed_keys - expected_keys
                raise ValueError(
                    f"compatibility VCF {filename} is not the declared route union; "
                    f"missing={len(missing_keys)}, extra={len(extra_keys)}"
                )

        self._validate_router_summary(bundle, manifest)
        self._validate_router_audit(bundle, manifest)
        return manifest

    def preflight_output_paths(self) -> None:
        """Refuse a filtering run before any existing output can be overwritten."""
        conflicts = [
            path for path in (Path(self.filtered_bam_dir), Path(self.log_dir))
            if path.exists()
        ]
        if conflicts:
            raise FileExistsError(
                "refusing to overwrite existing spot-profile output: "
                + ", ".join(str(path) for path in conflicts)
            )

    def initialize_output_paths(self) -> None:
        """Create new output leaves after input validation and preflight succeed."""
        Path(self.filtered_bam_dir).mkdir(parents=True, exist_ok=False)
        Path(self.bam_by_spot_dir).mkdir(exist_ok=False)
        Path(self.vcf_by_spot_dir).mkdir(exist_ok=False)
        Path(self.log_dir).mkdir(parents=True, exist_ok=False)

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def write_completion_manifest(
        self,
        input_bams: List[str],
        results: List[Dict],
        snv_count: int,
    ) -> dict:
        """Write the immutable Stage 5 contract, with RUN_COMPLETE last."""
        failed = [result for result in results if result.get("status") != "completed"]
        if failed or len(results) != len(input_bams):
            raise RuntimeError(
                "cannot mark spot-profile run complete: "
                f"results={len(results)}/{len(input_bams)}, failed={len(failed)}"
            )
        output_bams = sorted(Path(self.bam_by_spot_dir).glob("*.bam"))
        expected_bam_names = sorted(Path(path).name for path in input_bams)
        if sorted(path.name for path in output_bams) != expected_bam_names:
            raise RuntimeError("spot-profile BAM outputs do not match input BAM basenames")
        missing_bam_indices = [
            str(path) + ".bai" for path in output_bams
            if not Path(str(path) + ".bai").is_file()
        ]
        if missing_bam_indices:
            raise RuntimeError(
                f"missing BAM indices; first={missing_bam_indices[0]}"
            )

        per_spot_vcfs = sorted(Path(self.vcf_by_spot_dir).glob("*.vcf.gz"))
        expected_vcf_names = sorted(
            Path(result["input_bam"]).stem + ".vcf.gz"
            for result in results if result.get("detected_snvs")
        )
        if [path.name for path in per_spot_vcfs] != expected_vcf_names:
            raise RuntimeError("per-spot VCF outputs do not match detected-SNV results")
        missing_vcf_indices = [
            str(path) + ".tbi" for path in per_spot_vcfs
            if not Path(str(path) + ".tbi").is_file()
        ]
        if missing_vcf_indices:
            raise RuntimeError(
                f"missing per-spot VCF indices; first={missing_vcf_indices[0]}"
            )
        required_aggregates = [
            Path(self.filtered_bam_dir) / "all_variants.vcf.gz",
            Path(self.filtered_bam_dir) / "all_variants.vcf.gz.tbi",
            Path(self.filtered_bam_dir) / "all_detected_variants_summary.vcf.gz",
            Path(self.filtered_bam_dir) / "all_detected_variants_summary.vcf.gz.tbi",
            Path(self.filtered_bam_dir) / "filtering_summary.txt",
        ]
        missing_aggregates = [str(path) for path in required_aggregates if not path.is_file()]
        if missing_aggregates:
            raise RuntimeError(
                "missing required spot-profile aggregate output(s): "
                + ", ".join(missing_aggregates)
            )

        beagle_manifest_path = Path(self.beagle_bundle_dir) / "MANIFEST.json"
        manifest = {
            "schema": SPOTPROFILE_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "complete",
            "dataset": self.dataset_name,
            "section_id": self.section_id,
            "quality_filter": self.quality_filter,
            "classifier": self.classifier,
            "input_bam_count": len(input_bams),
            "completed_bam_count": len(results),
            "failed_bam_count": 0,
            "candidate_exact_allele_count": int(snv_count),
            "bams_with_detected_snvs": len(per_spot_vcfs),
            "per_spot_vcf_count": len(per_spot_vcfs),
            "per_spot_vcfs": [path.name for path in per_spot_vcfs],
            "per_spot_vcf_indices_complete": True,
            "output_bam_count": len(output_bams),
            "output_bam_indices_complete": True,
            "beagle_route_schema": self.beagle_manifest["schema"],
            "beagle_anchor_score": self.beagle_manifest["anchor_score"],
            "beagle_min_anchor_probability": self.beagle_manifest[
                "min_anchor_probability"
            ],
            "beagle_manifest": str(beagle_manifest_path.resolve()),
            "beagle_manifest_sha256": self._sha256_file(beagle_manifest_path),
            "classifier_inputs": [
                {
                    "path": str(Path(path).resolve()),
                    "sha256": self._sha256_file(Path(path)),
                }
                for path in (self.classifier_homo_vcf, self.classifier_hetero_vcf)
            ],
        }
        manifest_path = Path(self.filtered_bam_dir) / "MANIFEST.json"
        complete_path = Path(self.filtered_bam_dir) / "RUN_COMPLETE"
        with manifest_path.open("x") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with complete_path.open("x") as handle:
            handle.write("complete\n")
        return manifest

    @staticmethod
    def _normalized_genotype(vcf_line: str) -> Optional[Tuple[int, ...]]:
        """Return a phase-insensitive called genotype, or None for no-call."""
        fields = vcf_line.rstrip("\n").split("\t")
        if len(fields) != 10:
            raise ValueError(f"malformed single-sample VCF record ({len(fields)} fields)")
        format_fields = fields[8].split(":")
        sample_fields = fields[9].split(":")
        if "GT" not in format_fields:
            raise ValueError("single-sample VCF record has no GT field")
        gt_index = format_fields.index("GT")
        if gt_index >= len(sample_fields):
            raise ValueError("single-sample VCF record has no GT sample value")
        tokens = re.split(r"[/|]", sample_fields[gt_index])
        if not tokens or any(token in {"", "."} for token in tokens):
            return None
        try:
            alleles = tuple(sorted(int(token) for token in tokens))
        except ValueError as error:
            raise ValueError(f"invalid GT value {sample_fields[gt_index]!r}") from error
        if any(allele < 0 for allele in alleles):
            raise ValueError(f"invalid GT value {sample_fields[gt_index]!r}")
        return alleles

    def count_genotypes(self, vcf_path: str) -> Tuple[int, int]:
        """Count phase-insensitive heterozygous and hom-alt genotypes."""
        count_het = 0
        count_hom_alt = 0
        saw_header = False
        with gzip.open(vcf_path, "rt") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.startswith("#CHROM"):
                    if len(line.rstrip("\n").split("\t")) != 10:
                        raise ValueError(
                            f"{vcf_path}:{line_number}: expected exactly one sample"
                        )
                    saw_header = True
                    continue
                if line.startswith("#"):
                    continue
                try:
                    genotype = self._normalized_genotype(line)
                except ValueError as error:
                    raise ValueError(f"{vcf_path}:{line_number}: {error}") from error
                if genotype is None or not any(allele > 0 for allele in genotype):
                    continue
                if len(set(genotype)) > 1:
                    count_het += 1
                elif genotype[0] > 0:
                    count_hom_alt += 1
        if not saw_header:
            raise ValueError(f"missing #CHROM header: {vcf_path}")
        return count_het, count_hom_alt

    def collect_snvs(self) -> Set[SNVInfo]:
        """Collect panel anchors, panel-discordant candidates, and classifier calls."""
        beagle_snvs = set()
        self.beagle_manifest = self.validate_beagle_route_bundle()
        classifier_snvs = set()
        combined_snvs = set()
        heterozygous_count = 0
        homozygous_count = 0
        # Check if classifier VCF files exist
        vcf_files = {
            f"{self.classifier.upper()} Homozygous": self.classifier_homo_vcf,
            f"{self.classifier.upper()} Heterozygous": self.classifier_hetero_vcf,
            # Panel-discordant calls bypass the Beagle-trained error/genotype
            # filters and proceed directly to spatial classification.
            "Panel-discordant candidates": self.panel_discordant_vcf,
            "Beagle": self.beagle_vcf
        }

        # Stage 4 always emits both class VCFs, even when a class is empty.
        # Therefore absence (or a missing index) is an incomplete upstream run,
        # not an optional zero-call case. The three routed files were already
        # validated as part of the Beagle bundle above.
        classifier_paths = (self.classifier_homo_vcf, self.classifier_hetero_vcf)
        missing_classifier = [
            required
            for path in classifier_paths
            for required in (path, path + ".tbi")
            if not os.path.isfile(required)
        ]
        if missing_classifier:
            raise FileNotFoundError(
                "incomplete required classifier outputs; missing: "
                + ", ".join(missing_classifier)
            )

        authoritative_sources = {"Beagle", "Panel-discordant candidates"}
        for source, path in vcf_files.items():
                
            # Panel-concordant Beagle anchors remain defined. Classifier
            # candidates preserve whether they overlapped the panel but
            # disagreed with Beagle, or were absent from the panel.
            race = (
                "defined" if source == "Beagle"
                else "panel_discordant" if source == "Panel-discordant candidates"
                else "denovo"
            )
            
            # Count genotypes
            count_0_1, count_1_1 = self.count_genotypes(path)
            if source == f"{self.classifier.upper()} Homozygous":
                homozygous_count += count_0_1 + count_1_1
            elif source == f"{self.classifier.upper()} Heterozygous":
                heterozygous_count += count_0_1 + count_1_1
            else:
                heterozygous_count += count_0_1
                homozygous_count += count_1_1

            print(f"{source} VCF - 0/1: {count_0_1}, 1/1: {count_1_1}, Race: {race}")
            
            # Collect SNVs with race information
            source_snvs = set()
            saw_header = False
            with gzip.open(path, 'rt') as f:
                for line_number, line in enumerate(f, start=1):
                    if line.startswith('#CHROM'):
                        saw_header = True
                        continue
                    if line.startswith('#'):
                        continue
                    try:
                        genotype_is_nonref = self._is_valid_genotype(line)
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
                    # Routed records are authoritative evidence categories; their
                    # raw FORMAT genotype was deliberately preserved for audit and
                    # must not veto the Beagle route. Classifier records retain a
                    # phase-insensitive non-reference GT gate.
                    if source in authoritative_sources or genotype_is_nonref:
                        snv = SNVInfo.from_vcf_line(line)
                        if source == "Beagle":
                            snv.race = "defined"
                        elif source == "Panel-discordant candidates":
                            # This compatibility VCF is exactly the union of
                            # Beagle ref-high-confidence and uncertain routes.
                            snv.race = "panel_discordant"
                        else:
                            beagle_route = next(
                                (field.split("=", 1)[1] for field in snv.info.split(";")
                                 if field.startswith("BEAGLE_ROUTE=")),
                                None,
                            )
                            panel_status = next(
                                (field.split("=", 1)[1] for field in snv.info.split(";")
                                 if field.startswith("PANEL_STATUS=")),
                                None,
                            )
                            snv.race = (
                                "panel_discordant"
                                if (
                                    beagle_route in {"beagle_ref_highconf", "beagle_uncertain"}
                                    or panel_status == "panel_discordant"
                                )
                                else "denovo"
                            )
                        source_snvs.add(snv)
                        combined_snvs.add(snv)
            if not saw_header:
                raise ValueError(f"missing #CHROM header: {path}")

            # Print sample keys for debugging
            print(f"SNVs collected from {source}: {len(source_snvs)}")
            sample_size = min(5, len(source_snvs))
            if sample_size > 0:
                print(f"Sample SNV keys from {source}:")
                sample_snvs = sorted(
                    source_snvs,
                    key=lambda snv: (
                        snv.standardized_chrom, snv.pos, snv.ref, snv.alt
                    ),
                )[:sample_size]
                for snv in sample_snvs:
                    print(f"  Original chrom: {snv.chrom}, Standardized: {snv.standardized_chrom}, Key: {snv.key}, Race: {snv.race}")
            
            # Store in source-specific set
            if source == "Beagle":
                beagle_snvs = source_snvs
            else:
                classifier_snvs.update(source_snvs)
        
        # Print set overlap statistics for debugging
        if beagle_snvs and classifier_snvs:
            overlap = beagle_snvs.intersection(classifier_snvs)
            print(f"\nOverlap between Beagle and Classifier SNVs: {len(overlap)}")
            print(f"Classifier unique to Beagle: {len(beagle_snvs - overlap)}")
            print(f"SNVs unique to Classifier: {len(classifier_snvs - overlap)}")
        
        print(f"Total unique SNVs after combining sources: {len(combined_snvs)}")
        print(f"Total Heterozygous SNVs count: {heterozygous_count}")
        print(f"Total Homozygous SNVs count: {homozygous_count}")
        
        # Print race distribution
        race_counts = {"defined": 0, "panel_discordant": 0, "denovo": 0}
        for snv in combined_snvs:
            race_counts[snv.race] = race_counts.get(snv.race, 0) + 1
        print("Race distribution - " + ", ".join(
            f"{race}: {count}" for race, count in race_counts.items()))
        
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
        """Return True for any called, phase-insensitive non-reference GT."""
        genotype = self._normalized_genotype(vcf_line)
        return genotype is not None and any(allele > 0 for allele in genotype)

    def filter_bams(self, max_workers=30):
        """Filter BAM files based on SNV positions."""
        print("Collecting SNVs from VCF files...")
        self.preflight_output_paths()
        snvs = self.collect_snvs()
        print(f"Found {len(snvs)} SNVs to use for filtering")
        if not snvs:
            raise ValueError("no valid SNVs were found in the routed/classified inputs")

        # Get list of BAM files
        if '*' not in self.bam_dir:
            search_pattern = os.path.join(self.bam_dir, '*.bam')
        else:
            search_pattern = self.bam_dir
        
        bam_files = sorted(glob.glob(search_pattern))
        if not bam_files:
            raise FileNotFoundError(f"No BAM files found at: {search_pattern}")
        bam_basenames = [os.path.basename(path) for path in bam_files]
        if len(bam_basenames) != len(set(bam_basenames)):
            raise ValueError("input BAM basenames are not unique; outputs would collide")

        print(f"Found {len(bam_files)} BAM files to process")
        
        # Setup environment for samtools
        setup_environment()
        self.initialize_output_paths()

        print(f"Per-spot BAM files will be saved to: {self.bam_by_spot_dir}")
        print(f"Per-spot VCF files will be saved to: {self.vcf_by_spot_dir}")
        print(f"Merged files will be saved to: {self.filtered_bam_dir}")
        
        # Filter BAMs in parallel
        results = filter_bams_parallel(
            input_bams=bam_files,
            output_dir=self.bam_by_spot_dir,  # BAMs go to bam_by_spot
            snvs=snvs,
            max_workers=max_workers,
            vcf_output_dir=self.vcf_by_spot_dir,  # VCFs go to vcf_by_spot
            merged_output_dir=self.filtered_bam_dir  # Merged files at root
        )
        
        # Verify VCF files were created
        snv_files = glob.glob(os.path.join(self.vcf_by_spot_dir, "*.vcf.gz"))
        print(f"Found {len(snv_files)} SNV VCF files after processing")
        
        # Print summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        with_snvs = sum(1 for r in results if r['status'] == 'completed' and r.get('detected_snvs'))
        if len(results) != len(bam_files):
            raise RuntimeError(
                f"BAM result count mismatch: {len(results)} != {len(bam_files)}"
            )

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
        
        print(f"\nFiltered BAM files are located in: {self.bam_by_spot_dir}")
        print(f"SNV VCF files are located in: {self.vcf_by_spot_dir}")
        print(f"Merged summary files are located in: {self.filtered_bam_dir}")
        
        # Write summary report to root directory
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
            f.write(f"Output Structure:\n")
            f.write(f"  Per-spot BAM files: {self.bam_by_spot_dir}\n")
            f.write(f"  Per-spot VCF files: {self.vcf_by_spot_dir}\n")
            f.write(f"  Merged summary files: {self.filtered_bam_dir}\n\n")
            
            if failed > 0:
                f.write("Failed BAMs:\n")
                for result in results:
                    if result['status'] == 'failed':
                        f.write(f"  {os.path.basename(result['input_bam'])}: {result.get('error', 'Unknown error')}\n")

        if failed:
            raise RuntimeError(
                f"spot-profile filtering failed for {failed}/{len(results)} BAMs; "
                "RUN_COMPLETE was not written"
            )
        self.write_completion_manifest(bam_files, results, len(snvs))
        return results
        
    def index_existing_bams(self):
        """Index existing BAM files in the bam_by_spot directory."""
        print(f"Indexing existing BAM files in {self.bam_by_spot_dir}")
        
        # Setup environment for samtools
        setup_environment()
        
        # Index BAMs
        results = index_bams_in_directory(self.bam_by_spot_dir)
        
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
    parser.add_argument("--section-root", default=None,
                  help="Override the section output root for an isolated run")
    
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
        classifier=args.classifier,
        section_root=args.section_root
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

# For DCIS:
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset DCIS --section-id 1 --quality-filter baseQ0mapQ0 --classifier neural_network
# python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset DCIS --section-id 2 --quality-filter baseQ0mapQ0 --classifier neural_network

# Output Structure (NEW):
# data/{dataset}/{section_id}/output_VCFs/spotprofiles/{quality_filter}/
# ├── bam_by_spot/                                # Per-spot BAM files
# │   ├── AAACAACGAATAGTTC-1.bam
# │   ├── AAACAACGAATAGTTC-1.bam.bai
# │   └── ...
# ├── vcf_by_spot/                                # Per-spot VCF files
# │   ├── AAACAACGAATAGTTC-1.vcf.gz
# │   ├── AAACAACGAATAGTTC-1.vcf.gz.tbi
# │   └── ...
# ├── all_variants.vcf.gz                         # All input SNVs from pool
# ├── all_variants.vcf.gz.tbi
# ├── all_detected_variants_summary.vcf.gz        # Detected SNVs with COUNT/FREQ
# ├── all_detected_variants_summary.vcf.gz.tbi
# └── filtering_summary.txt                       # Summary report
