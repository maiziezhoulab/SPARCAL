import os
import time
import glob
import argparse
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Set, Optional
from tqdm import tqdm
from collections import defaultdict
import gzip

# Default Parameters
DEFAULT_PARAMS = {
    "MIN_BASE_QUALITY": 0,
    "MIN_MAPPING_QUALITY": 0,
    "MAX_DEPTH": 10000000,
    "THREADS": 30,
    "MAX_FILES": None,  # None means process all files
    "REGIONS": [f"{i}" for i in range(1, 23)], # + ['X', 'Y']
}

# File Paths
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "REFERENCE_SEQ": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
    "BEDFILE": "/data/maiziezhou_lab/yuqi/snv_calling/data/reference/GRCh38.bed",
    "HEADER": "/data/maiziezhou_lab/yuqi/snv_calling/data/reference/header.txt",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools",
    "BCFTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bcftools",
    "BGZIP": "/data/maiziezhou_lab/yuqi/snv_calling/apps/bgzip",
    "BAMDIR": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
    # "BAMDIR": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_bam_processed"   # DLPFC spatial is original, bam_processed used barcode as read group.
}

def setup_output_dirs(section_id: str, caller_type: str) -> Dict[str, str]:
    """Setup output directories for the pipeline."""
    base_dir = os.path.join(PATH_CONFIG["PROJECT_DIR"], "data/dlpfc", section_id)
    output_structure = {
        "vcf_dir": os.path.join(base_dir, "output_VCFs", caller_type),
        "log_dir": os.path.join(base_dir, "logs", caller_type),
        "metrics_dir": os.path.join(base_dir, "metrics", caller_type)
    }
    
    for dir_path in output_structure.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return output_structure

def count_snps(vcf_file: str) -> int:
    """Count the number of SNPs in a VCF file."""
    count = 0
    with open(vcf_file, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                count += 1
    return count

def setup_environment():
    """Setup environment variables for library paths."""
    # Add apps directory to PATH
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    
    # Add apps directory to LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

def create_regions_file(regions: List[str], output_dir: str) -> str:
    """Create a regions file for mpileup."""
    regions_file = os.path.join(output_dir, "regions.txt")
    with open(regions_file, 'w') as f:
        for region in regions:
            f.write(f"{region}\n")
    return regions_file

def run_command(cmd: str, env: Dict = None, **kwargs) -> subprocess.CompletedProcess:
    """Run a command with proper environment variables."""
    if env is None:
        env = os.environ.copy()
    else:
        env = {**os.environ.copy(), **env}
     
    return subprocess.run(cmd, env=env, **kwargs)

def process_region(region: str, bam_list: str, output_dirs: Dict[str, str], params: Dict) -> Dict:
    """Process a specific genomic region for multiple BAM files."""
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"region_{region}.vcf.gz")
    log_file = os.path.join(output_dirs["log_dir"], f"region_{region}.log")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_vcf), exist_ok=True)
    
    start_time = time.time()
    
    # Set up environment
    env = setup_environment()
    
    try:
        mpileup_cmd = (
            f"{PATH_CONFIG['SAMTOOLS']} mpileup "
            f"-f {PATH_CONFIG['REFERENCE_SEQ']} "
            f"-b {bam_list} "
            f"-r {region} "
            f"-q {params['MIN_MAPPING_QUALITY']} "
            f"-Q {params['MIN_BASE_QUALITY']} "
            f"-d {params['MAX_DEPTH']} -v | "
            f"{PATH_CONFIG['BCFTOOLS']} view | "
            f"{PATH_CONFIG['BCFTOOLS']} filter -e 'REF !~ \"^[ATGC]$\"' | "
            f"{PATH_CONFIG['BCFTOOLS']} norm -m-both -f {PATH_CONFIG['REFERENCE_SEQ']} | "
            f"grep -v '<X>\|INDEL' | "
            f"{PATH_CONFIG['BGZIP']} -c > {output_vcf}"
        )

        with open(log_file, 'w') as log:
            process = run_command(mpileup_cmd, env=env, shell=True, stderr=log)
            if process.returncode != 0:
                raise Exception(f"Command failed with return code {process.returncode}")

        # Verify the file was created and is not empty
        if not os.path.exists(output_vcf):
            raise Exception("Output VCF file was not created")
            
        # Count SNPs using gzip
        snp_count = 0
        with gzip.open(output_vcf, 'rt') as f:
            for line in f:
                if not line.startswith('#'):
                    snp_count += 1
                    
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "region": region,
            "duration": duration,
            "snp_count": snp_count,
            "output_vcf": output_vcf
        }
        
    except Exception as e:
        print(f"Error processing region {region}: {str(e)}")
        return None

def run_multi_bam_mpileup(bam_files: List[str], output_dirs: Dict[str, str], 
                         params: Dict) -> Tuple[pd.DataFrame, str]:
    """Run mpileup on multiple BAM files in parallel by region."""
    # Create BAM list file
    bam_list_file = os.path.join(output_dirs["log_dir"], "bam_list.txt")
    with open(bam_list_file, 'w') as f:
        for bam in bam_files:
            f.write(f"{bam}\n")
     
    print("BAM list file:")
    with open(bam_list_file, 'r') as f:
        print(f.read())

    # Process regions in parallel
    results = []
    regions = params.get('REGIONS', DEFAULT_PARAMS['REGIONS'])
    
    with ThreadPoolExecutor(max_workers=params['THREADS']) as executor:
        future_to_region = {
            executor.submit(process_region, region, bam_list_file, output_dirs, params): region
            for region in regions
        }
        
        pbar = tqdm(total=len(regions), desc="Processing regions")
        
        for future in as_completed(future_to_region):
            try:
                result = future.result()
                results.append(result)
                pbar.update(1)
            except Exception as e:
                region = future_to_region[future]
                print(f"Error processing region {region}: {str(e)}")
        
        pbar.close()
    
    # Merge, sort, and process VCF files
    region_vcfs = [result["output_vcf"] for result in results]
    print("Merging, sorting, and processing VCF files...")
    final_vcf = merge_vcfs(region_vcfs, output_dirs["vcf_dir"], params)
    print(f"Final processed VCF (with genotypes) available at: {final_vcf}")
    
    # Create metrics DataFrame
    metrics_df = pd.DataFrame(results)
    metrics_file = os.path.join(output_dirs["metrics_dir"], "region_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to: {metrics_file}")
    
    return metrics_df, final_vcf

def distribute_variants_to_bams(vcf_dir: str, region: str):
    """
    Distribute variants to BAM-specific VCF files based on tracking information.
    
    Args:
        vcf_dir: Directory containing VCF files
        region: Chromosome region being processed
    """
    # Read the region's VCF and tracking information
    region_vcf = os.path.join(vcf_dir, f"region_{region}.vcf")
    track_file = os.path.join(vcf_dir, f"region_{region}_bam_track.txt")
    
    if not (os.path.exists(region_vcf) and os.path.exists(track_file)):
        return
    
    # Create a memory-efficient iterator for the tracking file
    def track_iterator(track_file):
        with open(track_file, 'r') as f:
            for line in f:
                yield line.strip().split()
    
    # Process variants in chunks to manage memory
    chunk_size = 1000  # Adjust based on available memory
    current_chunk = []
    current_pos = None
    
    for track_entry in track_iterator(track_file):
        pos = int(track_entry[1])
        if current_pos != pos and current_chunk:
            # Process the current chunk
            process_variant_chunk(current_chunk, vcf_dir, region)
            current_chunk = []
        current_pos = pos
        current_chunk.append(track_entry)
        
        if len(current_chunk) >= chunk_size:
            process_variant_chunk(current_chunk, vcf_dir, region)
            current_chunk = []
    
    # Process any remaining variants
    if current_chunk:
        process_variant_chunk(current_chunk, vcf_dir, region)

def process_variant_chunk(chunk: List[List[str]], vcf_dir: str, region: str):
    """
    Process a chunk of variants and distribute them to BAM-specific VCF files.
    
    Args:
        chunk: List of variant tracking entries
        vcf_dir: Directory containing VCF files
        region: Chromosome region being processed
    """
    # Group variants by BAM file
    bam_variants = defaultdict(list)
    for entry in chunk:
        bam_name = entry[6].split(',')[0]  # First QNAME is the BAM identifier
        bam_variants[bam_name].append(entry)
    
    # Write variants to BAM-specific VCF files
    for bam_name, variants in bam_variants.items():
        bam_vcf = os.path.join(vcf_dir, f"{bam_name}_{region}.vcf")
        with open(bam_vcf, 'a') as f:
            for var in variants:
                # Format variant line for VCF
                vcf_line = format_vcf_line(var)
                if vcf_line:
                    f.write(vcf_line + '\n')

def format_vcf_line(variant_data: List[str]) -> Optional[str]:
    """
    Format variant data as a VCF line.
    
    Args:
        variant_data: List containing variant information
    
    Returns:
        Formatted VCF line or None if invalid
    """
    try:
        chrom, pos, ref, alt, info = variant_data[:5]
        return f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t20\tPASS\tSVTYPE=SNV"
    except Exception:
        return None
    
def run_custom_caller(bam_file: str, output_dirs: Dict[str, str], 
                     caller_script: str, params: Dict) -> Dict:
    """Run custom caller (old or new) on a single BAM file."""
    basename = os.path.basename(bam_file).replace('.bam', '')
    output_vcf = os.path.join(output_dirs["vcf_dir"], f"{basename}.vcf")
    
    # Write header
    with open(output_vcf, 'w') as f:
        with open(PATH_CONFIG["HEADER"], 'r') as header:
            f.write(header.read())
    
    start_time = time.time()
    
    # Set up environment
    env = setup_environment()
    
    for chrom in range(1, 23):
        run_command([
            "python", caller_script,
            "--reference_seq", PATH_CONFIG["REFERENCE_SEQ"],
            "--chromosome", str(chrom),
            "--bamfile", bam_file,
            "--bedfile", PATH_CONFIG["BEDFILE"],
            "--header", PATH_CONFIG["HEADER"],
            "--out", output_vcf,
            "--min_base_quality", str(params["MIN_BASE_QUALITY"]),
            "--min_mapping_quality", str(params["MIN_MAPPING_QUALITY"]),

        ], env=env)

    # Process X and Y chromosomes
    # for chrom in ['X', 'Y']:
    #     run_command([
    #         "python", caller_script,
    #         "--reference_seq", PATH_CONFIG["REFERENCE_SEQ"],
    #         "--chromosome", chrom,
    #         "--bamfile", bam_file,
    #         "--bedfile", PATH_CONFIG["BEDFILE"],
    #         "--header", PATH_CONFIG["HEADER"],
    #         "--out", output_vcf,
    #         "--min_base_quality", str(params["MIN_BASE_QUALITY"]),
    #         "--min_mapping_quality", str(params["MIN_MAPPING_QUALITY"]),
    #     ], env=env)
    
    end_time = time.time()
    duration = end_time - start_time
    
    return {
        "bam_file": basename,
        "duration": duration,
        "snp_count": count_snps(output_vcf),
        "output_vcf": output_vcf
    }

def convert_and_process_vcf(input_vcf: str, output_vcf: str) -> str:
    """
    Convert chromosome format, sort, compress and index VCF file.
    
    Args:
        input_vcf: Path to input VCF file
        output_vcf: Path to output VCF file
    
    Returns:
        Path to the processed (compressed) VCF file
    """
    print(f"Processing VCF file: {input_vcf}")
    
    # Store header and data lines
    header_lines = []
    data_lines = []
    chromosomes = set()
    
    # Read and process VCF
    print("Reading and processing VCF file...")
    with open(input_vcf, 'r') as infile:
        for line in infile:
            if line.startswith('#'):
                if not line.startswith('##contig='):
                    header_lines.append(line)
            else:
                fields = line.strip().split('\t')
                # Convert chromosome name
                if not fields[0].startswith('chr'):
                    if fields[0] == 'MT':
                        fields[0] = 'chrM'
                    else:
                        fields[0] = f'chr{fields[0]}'
                chromosomes.add(fields[0])
                # Store chromosome, position, and full line for sorting
                pos = int(fields[1])
                data_lines.append((fields[0], pos, fields))

    # Sort data lines
    print("Sorting data...")
    data_lines.sort(key=lambda x: (x[0], x[1]))  # Sort by chromosome then position

    # Write sorted output
    print("Writing sorted output...")
    with open(output_vcf, 'w') as outfile:
        # Write original headers except contig lines
        for line in header_lines[:-1]:
            outfile.write(line)
            
        # Write contig lines
        for chrom in sorted(chromosomes):
            outfile.write(f'##contig=<ID={chrom}>\n')
            
        # Write #CHROM line
        outfile.write(header_lines[-1])
        
        # Write sorted data
        for _, _, fields in data_lines:
            outfile.write('\t'.join(fields) + '\n')

    # Compress with bgzip
    compressed_vcf = f"{output_vcf}.gz"
    try:
        run_command(['bgzip', '-f', output_vcf])
        # Index with tabix
        run_command(['tabix', '-f', '-p', 'vcf', compressed_vcf])
        print(f"Successfully compressed and indexed: {compressed_vcf}")
    except subprocess.CalledProcessError as e:
        print(f"Error during compression/indexing: {str(e)}")
        raise

    return compressed_vcf

def parse_i16(i16_str: str) -> List[int]:
    """Parse I16 field from VCF INFO column."""
    try:
        # Remove 'I16=' prefix and split by comma
        i16_values = [int(x) for x in i16_str.replace('I16=', '').split(',')]
        return i16_values
    except (ValueError, IndexError):
        return [0] * 16

def calculate_baf_from_i16(i16_values: List[int]) -> float:
    """
    Calculate BAF using the first 4 values from I16 field.
    I16[0:4] contains:
    [0] - reference forward Q13 bases
    [1] - reference reverse Q13 bases
    [2] - alternate forward Q13 bases
    [3] - alternate reverse Q13 bases
    """
    try:
        ref_depth = i16_values[0] + i16_values[1]  # Total reference depth
        alt_depth = i16_values[2] + i16_values[3]  # Total alternate depth
        total_depth = ref_depth + alt_depth
        
        if total_depth == 0:
            return 0.0
            
        return alt_depth / total_depth
    except (IndexError, ZeroDivisionError):
        return 0.0

def calculate_genotype_quality(pl_values: List[int]) -> float:
    """
    Calculate genotype quality score based on PL values.
    Uses difference between best and second-best PL scores.
    """
    if len(pl_values) < 2:
        return 0.0
        
    sorted_pl = sorted(pl_values)
    best_pl = sorted_pl[0]
    second_best_pl = sorted_pl[1]
    
    # Convert PL difference to probability
    pl_diff = second_best_pl - best_pl
    quality = 1 - (10 ** (-pl_diff/10))
    
    return quality

def infer_gt_from_pl(input_vcf: str, output_vcf: str, 
                     min_depth: int = 10,
                     min_gq: float = 0.8,
                     min_qual: int = 20):
    """
    Enhanced genotype inference with BAF calculation from I16 field.
    
    Args:
        input_vcf: Input VCF file (can be gzipped)
        output_vcf: Output VCF file path
        min_depth: Minimum read depth required
        min_gq: Minimum genotype quality required
        min_qual: Minimum variant quality required
    """
    # Open files
    infile = gzip.open(input_vcf, 'rt') if input_vcf.endswith('.gz') else open(input_vcf, 'r')
    
    with infile, open(output_vcf, 'w') as f_out:
        # Process header
        for line in infile:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    # Add new FORMAT fields before #CHROM line
                    f_out.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
                    f_out.write('##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">\n')
                    f_out.write('##FORMAT=<ID=BAF,Number=1,Type=Float,Description="B-Allele Frequency">\n')
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
            i16_values = parse_i16(info_dict.get('I16', '0,0,0,0'))
            
            # Calculate BAF from I16
            baf = calculate_baf_from_i16(i16_values)
            
            # Parse PL values
            format_field = fields[8]
            sample_field = fields[9]
            format_values = sample_field.split(':')
            pl_idx = format_field.split(':').index('PL')
            
            try:
                pl_values = [int(x) for x in format_values[pl_idx].split(',')]
                qual = float(fields[5]) if fields[5] != '.' else 0
            except (ValueError, IndexError):
                continue
            
            # Calculate genotype quality
            gq = calculate_genotype_quality(pl_values)
            
            # Determine best genotype from PL values
            min_pl_index = pl_values.index(min(pl_values))
            inferred_gt = {0: '0/0', 1: '0/1', 2: '1/1'}[min_pl_index]
            
            # Expected BAF ranges for validation
            expected_baf = {
                '0/0': (0.0, 0.15),    # Homozygous reference
                '0/1': (0.35, 0.65),   # Heterozygous
                '1/1': (0.85, 1.0)     # Homozygous alternate
            }
            
            # Apply filters
            filters = []
            if depth < min_depth:
                filters.append('LowDP')
            if gq < min_gq:
                filters.append('LowGQ')
            if qual < min_qual:
                filters.append('LowQual')
                
            # Add BAF validation
            baf_range = expected_baf[inferred_gt]
            if not (baf_range[0] <= baf <= baf_range[1]):
                if depth >= min_depth:  # Only add filter if we had sufficient depth
                    filters.append('DiscordantBAF')
            
            # Update FILTER field
            fields[6] = ';'.join(filters) if filters else 'PASS'
            
            # Update FORMAT and sample fields
            fields[8] = 'GT:GQ:BAF:' + format_field
            gq_int = int(gq * 100)  # Convert GQ to integer percentage
            fields[9] = f"{inferred_gt}:{gq_int}:{baf:.3f}:" + sample_field
            
            # Write updated line
            f_out.write('\t'.join(fields) + '\n')


def merge_vcfs(vcf_files: List[str], output_dir: str, params: Dict) -> str:
    """
    Merge multiple VCF files, process them, and infer genotypes.
    
    Args:
        vcf_files: List of VCF files to merge
        output_dir: Directory for output files
        params: Pipeline parameters including quality scores
    
    Returns:
        Path to the final processed (compressed) VCF file with genotypes
    """
    # Create output directory with quality parameters
    final_output_dir = os.path.join(output_dir, f"baseQ{params['MIN_BASE_QUALITY']}mapQ{params['MIN_MAPPING_QUALITY']}")
    os.makedirs(final_output_dir, exist_ok=True)
    
    # Define output files
    temp_merged = os.path.join(final_output_dir, "temp_merged.vcf")
    sorted_vcf = os.path.join(final_output_dir, "merged_sorted.vcf")
    sorted_vcf_gz = os.path.join(final_output_dir, "merged_sorted.vcf.gz")
    final_vcf = os.path.join(final_output_dir, "merged_sorted_gt.vcf")
    final_vcf_gz = os.path.join(final_output_dir, "merged_sorted_gt.vcf.gz")
    
    print(f"Merging VCF files into: {temp_merged}")
    
    # Merge VCF files
    with open(temp_merged, 'w') as out_file:
        # Write header from first file
        first_file = gzip.open(vcf_files[0], 'rt') if vcf_files[0].endswith('.gz') else open(vcf_files[0], 'r')
        with first_file:
            for line in first_file:
                if line.startswith('#'):
                    out_file.write(line)
                else:
                    break
        
        # Concatenate variants from all files
        for vcf_file in vcf_files:
            infile = gzip.open(vcf_file, 'rt') if vcf_file.endswith('.gz') else open(vcf_file, 'r')
            with infile:
                for line in infile:
                    if not line.startswith('#'):
                        out_file.write(line)
    
    # Process and sort the merged file
    print(f"Converting and processing merged VCF...")
    sorted_compressed_vcf = convert_and_process_vcf(temp_merged, sorted_vcf)
    
    # Infer genotypes
    print("Inferring genotypes from PL scores...")
    infer_gt_from_pl(sorted_compressed_vcf, final_vcf)
    
    # Compress and index the final VCF
    print("Compressing and indexing final VCF...")
    run_command(['bgzip', '-f', final_vcf])
    run_command(['tabix', '-p', 'vcf', final_vcf_gz])
    
    # Clean up temporary files
    print("Cleaning up temporary files...")
    for temp_file in [temp_merged, sorted_vcf, final_vcf]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except OSError as e:
            print(f"Warning: Could not remove {temp_file}: {e}")
    
    return final_vcf_gz

def run_pipeline(section_id: str, caller_type: str, custom_params: Dict = None,
                multi_bam_mode: bool = False):
    """Enhanced pipeline function supporting both single and multi-BAM processing."""
    params = DEFAULT_PARAMS.copy()
    if custom_params:
        params.update(custom_params)
    
    if multi_bam_mode:
        caller_type = "mpileup_multi_bam"
    output_dirs = setup_output_dirs(section_id, caller_type)
    
    # bam_dir = f"/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section_id}/bam_bycell"
    # use BAMDIR to get the bam_dir
    bam_dir = os.path.join(PATH_CONFIG["BAMDIR"], section_id, "bam_bycell")
    bam_files = glob.glob(os.path.join(bam_dir, "*.bam"))
    
    if params["MAX_FILES"]:
        bam_files = bam_files[:params["MAX_FILES"]]
    print(multi_bam_mode, caller_type)

    if caller_type == "mpileup_multi_bam":
        print("Running multi-BAM mpileup pipeline...")
        results = run_multi_bam_mpileup(bam_files, output_dirs, params)
        return results
    else:
        # Original single-BAM processing logic
        results = []
        pbar = tqdm(total=len(bam_files), desc=f"Processing {caller_type}")
        
        with ThreadPoolExecutor(max_workers=params["THREADS"]) as executor:
            print(f"Running {caller_type} pipeline...")
            future_to_bam = {}
            
            for bam_file in bam_files:
                if caller_type == "mpileup":
                    future = executor.submit(run_mpileup, bam_file, output_dirs, params)
                else:
                    caller_script = os.path.join(
                        PATH_CONFIG["PROJECT_DIR"], 
                        "scripts/calling",
                        "self_caller.py" if caller_type == "old_caller" else "new_caller.py"
                    )
                    future = executor.submit(run_custom_caller, bam_file, output_dirs, caller_script, params)
                future_to_bam[future] = bam_file
            
            for future in as_completed(future_to_bam):
                try:
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                except Exception as e:
                    print(f"Error processing {future_to_bam[future]}: {str(e)}")
        
        pbar.close()
        
        metrics_df = pd.DataFrame(results)
        metrics_file = os.path.join(output_dirs["metrics_dir"], f"{section_id}_{caller_type}_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Metrics saved to {metrics_file}")
        return metrics_df


def main():
    parser = argparse.ArgumentParser(description="SNV Calling Pipeline")
    parser.add_argument("--section_id", required=True, help="Section ID")
    parser.add_argument("--caller_type", choices=["mpileup", "old_caller", "new_caller"],
                      required=True, help="Type of caller to use")
    parser.add_argument("--max_files", type=int, help="Maximum number of files to process")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads to use")
    parser.add_argument("--multi_bam", action="store_true", 
                      help="Process multiple BAM files together with region parallelization")
    parser.add_argument("--regions_file", help="File containing custom regions (optional)")
    
    args = parser.parse_args()
    
    # Setup environment variables
    env = setup_environment()
    print("Environment setup complete:")
    print(f"PATH: {env['PATH']}")
    print(f"LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
    
    custom_params = {
        "MAX_FILES": args.max_files,
        "THREADS": args.threads
    }
    
    # Load custom regions if provided
    if args.regions_file:
        with open(args.regions_file, 'r') as f:
            custom_params['REGIONS'] = [line.strip() for line in f]
    
    # Run pipeline
    result = run_pipeline(args.section_id, args.caller_type, custom_params, args.multi_bam)
    
    if args.multi_bam:
        metrics_df, merge_vcfs = result
        # print(metrics_df)
        print("\nMulti-BAM Pipeline Summary:")
        print(f"Total regions processed: {len(metrics_df)}")
        print(f"Average processing time per region: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
        print(f"Merged VCF file: {merge_vcfs}")
    else:
        metrics_df = result
        print("\nSingle-BAM Pipeline Summary:")
        print(f"Total files processed: {len(metrics_df)}")
        print(f"Average processing time: {metrics_df['duration'].mean():.2f} seconds")
        print(f"Total SNPs found: {metrics_df['snp_count'].sum()}")
        print(f"Average SNPs per file: {metrics_df['snp_count'].mean():.2f}")

if __name__ == "__main__":
    main()


# Usage
# python run_slurm/dlpfc/run_caller_pipeline.py --section_id 151507 --caller_type mpileup --max_files 10
# python run_slurm/dlpfc/run_caller_pipeline.py --section_id 151507 --caller_type new_caller --max_files 10

# This one works for mpileup:
# python scripts/calling/all_caller_pipeline.py --section_id 151507 --caller_type mpileup --multi_bam --max_files 10