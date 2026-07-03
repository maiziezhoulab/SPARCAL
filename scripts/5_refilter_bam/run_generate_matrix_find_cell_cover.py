import os
import gzip
import pysam
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import glob
import subprocess

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
    "10X_BC_6.5MM": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE",
        "bam_pattern": "split_by_cell/BAMs/*.bam",
        "output_dir": "data/10X_BC_6.5mm",
        "has_sections": False,
        "reference": "FFPE_VISIUM",
        "multiple_bams": True
    },
    "10X_BC_FFPE": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_Ductal_Carcinoma_In_Situ_Invasive_Carcinoma_FFPE",
        "bam_pattern": "split_by_cell/bam_bycell/*.bam",
        "output_dir": "data/10X_BC_FFPE",
        "has_sections": False,
        "reference": "FFPE_VISIUM",
        "multiple_bams": True
    },
    "P4_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P4_Tumor_output/outs/split_BAM/",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    },
    "P6_TUMOR": {
        "base_path": "/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "bam_pattern": "spaceranger_align_rep{section_id}/P6_Tumor_output/outs/split_BAM/",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR",
        "multiple_bams": True
    }
}

# Path configuration matching mpileup_pipeline.py
PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/yuqi/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/yuqi/snv_calling/apps",
    "SAMTOOLS": "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools"
}

def setup_environment():
    """Setup environment variables for library paths."""
    os.environ['PATH'] = f"{PATH_CONFIG['APPS_DIR']}:{os.environ.get('PATH', '')}"
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    new_ld_path = f"{PATH_CONFIG['APPS_DIR']}:{current_ld_path}" if current_ld_path else PATH_CONFIG['APPS_DIR']
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    return {
        'PATH': os.environ['PATH'],
        'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']
    }

@dataclass
class SNVInfo:
    chrom: str
    pos: int
    ref: str
    alt: str
    info: str
    format_str: str
    
    def __eq__(self, other):
        if isinstance(other, SNVInfo):
            return (self.chrom, self.pos, self.ref, self.alt) == (other.chrom, other.pos, other.ref, other.alt)
        return False

    def __hash__(self):
        return hash((self.chrom, self.pos, self.ref, self.alt))

    @property
    def key(self) -> str:
        return f"{self.chrom}_{self.pos}_{self.ref}_{self.alt}"
    
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

def filter_bam_one_chrom(input_bam, chrom, poss):
    # Open input and output BAM files
    reads =[]
    start_j = 0
    with pysam.AlignmentFile(input_bam, "rb") as in_bam:
        # Create output BAM with same header as input
        # with pysam.AlignmentFile(output_bam, "wb", header=in_bam.header) as out_bam:
            # Process each read in the input BAM
        for read in in_bam.fetch(chrom):
            # Skip unmapped reads
            if read.is_unmapped:
                continue
                
            # Get chromosome name without 'chr' prefix
            chrom = read.reference_name.replace('chr', '')
            
            # Check if any SNV position is within this read's range
            # if chrom in positions_by_chrom:
            read_start = read.reference_start + 1  # Convert to 1-based position
            read_end = read.reference_end + 1 if read.reference_end else read_start
            
                # Check if any SNV position overlaps with read range

                #------------------------ N by N
            for j in range(start_j, len(poss)):
                pos = poss[j]
                if read_start <= pos <= read_end:
                    # out_bam.write(read)
                    reads.append(read)
                    start_j = j
                    break
                if pos >= read_end:
                    start_j = j
                    break
    return reads


def filter_bam_by_positions(input_bam: str, output_bam: str, snvs: Set[SNVInfo]) -> None:
    """
    Filter BAM file to only keep reads that overlap with SNV positions.
    """
    # Create position dictionary for faster lookup
    positions_by_chrom = defaultdict(list)
    for snv in snvs:
        # Handle chromosome naming with/without 'chr' prefix
        chrom = snv.chrom.replace('chr', '')
        positions_by_chrom[chrom].append(snv.pos)
    reads = []
    for chrom in positions_by_chrom:
        positions_by_chrom[chrom] = sorted(positions_by_chrom[chrom])
        poss = positions_by_chrom[chrom]

        reads.extend(filter_bam_one_chrom(input_bam, chrom, poss))
    with pysam.AlignmentFile(input_bam, "rb") as in_bam:
        with pysam.AlignmentFile(output_bam, "wb", header=in_bam.header) as out_bam:
            for read in reads:
                out_bam.write(read)
    

def index_bam_file(bam_path):
    """Index a BAM file using the correct samtools version."""
    env = setup_environment()
    try:
        cmd = [PATH_CONFIG['SAMTOOLS'], 'index', bam_path]
        result = subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return {'status': 'success', 'bam': bam_path}
    except subprocess.CalledProcessError as e:
        return {'status': 'error', 'bam': bam_path, 'error': str(e), 'stderr': e.stderr.decode()}

def index_bams_in_directory(directory, num_workers=30):
    """Index all BAM files in the specified directory."""
    # Find all BAM files
    bam_files = glob.glob(os.path.join(directory, '*.bam'))
    if not bam_files:
        print(f"No BAM files found in {directory}")
        return []
    
    print(f"Found {len(bam_files)} BAM files in {directory}")
    results = []
    
    # Process in parallel with progress bar
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_bam = {executor.submit(index_bam_file, bam): bam for bam in bam_files}
        
        with tqdm(total=len(bam_files), desc="Indexing BAM files") as pbar:
            for future in as_completed(future_to_bam):
                bam = future_to_bam[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append({'status': 'error', 'bam': bam, 'error': str(e)})
                pbar.update(1)
    
    # Summarize results
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = sum(1 for r in results if r['status'] == 'error')
    
    print(f"\nIndexing completed:")
    print(f"  Successfully indexed: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for result in results:
            if result['status'] == 'error':
                print(f"  {os.path.basename(result['bam'])}: {result['error']}")
    
    return results

def process_single_bam(input_bam: str, output_dir: str, snvs: Set[SNVInfo]) -> Dict:
    """Process a single BAM file: filter by SNV positions and index."""
    try:
        # Create output BAM path
        bam_name = os.path.basename(input_bam)
        output_bam = os.path.join(output_dir, bam_name)
        
        # Filter BAM file
        filter_bam_by_positions(input_bam, output_bam, snvs)
        
        # Index filtered BAM
        index_bam_file(output_bam)
        
        return {
            'input_bam': input_bam,
            'output_bam': output_bam,
            'status': 'completed'
        }
        
    except Exception as e:
        return {
            'input_bam': input_bam,
            'error': str(e),
            'status': 'failed'
        }

def filter_bams_parallel(input_bams: List[str], output_dir: str, snvs: Set[SNVInfo], 
                        max_workers: int = 30) -> List[Dict]:
    """Filter multiple BAM files in parallel."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all BAM processing jobs
        future_to_bam = {
            executor.submit(process_single_bam, bam, output_dir, snvs): bam 
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
            except Exception as e:
                results.append({
                    'input_bam': bam,
                    'error': str(e),
                    'status': 'failed'
                })
    
    return results

class SNVMatrixGenerator:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0",
                section_id: str = None, use_binary: bool = False,
                min_af_threshold: float = 0.2):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.use_binary = use_binary
        self.min_af_threshold = min_af_threshold
        self.base_dir = "/data/maiziezhou_lab/yuqi/snv_calling"
        
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
        self.svm_vcf = os.path.join(
            input_base, "output_VCFs/SVMModel",
            self.quality_filter, "results/high_confidence.vcf.gz"
        )
        
        # BAM directory based on dataset configuration
        if dataset_config["has_sections"]:
            self.bam_dir = os.path.join(
                dataset_config["base_path"],
                str(self.section_id),
                "bam_bycell"
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

    def count_genotypes(self, vcf_path: str) -> Tuple[int, int]:
        """Count the number of 0/1 and 1/1 genotypes in a VCF file."""
        count_0_1 = 0
        count_1_1 = 0
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
        """Collect SNVs from both Beagle and SVM outputs."""
        snvs = set()
        
        # Process Beagle VCF
        count_0_1, count_1_1 = self.count_genotypes(self.beagle_vcf)
        print(f"Beagle VCF - 0/1: {count_0_1}, 1/1: {count_1_1}")
        with gzip.open(self.beagle_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                snv = SNVInfo.from_vcf_line(line)
                if self._is_valid_genotype(line):
                    snvs.add(snv)
        print(f"Total SNVs collected: {len(snvs)}")
        # Process SVM VCF
        count_0_1, count_1_1 = self.count_genotypes(self.svm_vcf)
        print(f"SVM VCF - 0/1: {count_0_1}, 1/1: {count_1_1}")
        with gzip.open(self.svm_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                snv = SNVInfo.from_vcf_line(line)
                if self._is_valid_genotype(line):
                    snvs.add(snv)
        
        print(f"Total SNVs collected: {len(snvs)}")
        # Handle chromosome naming
        for snv in snvs:
            if snv.chrom.startswith("chr"):
                snv.chrom = snv.chrom[3:]
        return snvs

    def _is_valid_genotype(self, vcf_line: str) -> bool:
        """Check if genotype is 0/1 or 1/1."""
        fields = vcf_line.strip().split('\t')
        format_fields = fields[8].split(':')
        sample_fields = fields[9].split(':')
        gt_idx = format_fields.index('GT')
        gt = sample_fields[gt_idx]
        return gt in ['0/1', '1/1']

    def filter_bams(self):
        """Filter BAM files based on SNV positions."""
        print("Collecting SNVs from VCF files...")
        snvs = self.collect_snvs()
        print(f"Found {len(snvs)} SNVs to use for filtering")

        # Get list of BAM files
        if '*' not in self.bam_dir:
            self.bam_dir = os.path.join(self.bam_dir, '*.bam')
        
        bam_files = glob.glob(self.bam_dir)
        if not bam_files:
            raise FileNotFoundError(f"No BAM files found at: {self.bam_dir}")
        
        print(f"Found {len(bam_files)} BAM files to process")
        
        # Filter BAMs in parallel
        results = filter_bams_parallel(
            input_bams=bam_files,
            output_dir=self.filtered_bam_dir,
            snvs=snvs
        )
        
        # Print summary
        completed = sum(1 for r in results if r['status'] == 'completed')
        failed = sum(1 for r in results if r['status'] == 'failed')
        
        print("\nBAM Filtering Summary:")
        print(f"Total BAMs processed: {len(results)}")
        print(f"Successfully filtered: {completed}")
        print(f"Failed: {failed}")
        
        if failed > 0:
            print("\nFailed BAMs:")
            for result in results:
                if result['status'] == 'failed':
                    print(f"  {os.path.basename(result['input_bam'])}: {result['error']}")
        
        print(f"\nFiltered BAM files are located in: {self.filtered_bam_dir}")
        
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
            f.write(f"Failed: {failed}\n\n")
            
            if failed > 0:
                f.write("Failed BAMs:\n")
                for result in results:
                    if result['status'] == 'failed':
                        f.write(f"  {os.path.basename(result['input_bam'])}: {result['error']}\n")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Filter BAM files based on SNV positions")
    
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
    print("\nBAM Filtering Configuration:")
    print(f"Dataset: {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Quality Filter: {args.quality_filter}")
    print(f"Max Workers: {args.max_workers}")
    print("\n")
    
    # Initialize and run generator
    generator = SNVMatrixGenerator(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id,
        min_af_threshold=args.min_af_threshold
    )
    
    # Filter BAMs
    results = generator.filter_bams()
    
    # Exit with error if any BAMs failed
    if any(r['status'] == 'failed' for r in results):
        exit(1)
    
if __name__ == "__main__":
    main()

# Usage examples:
# For DLPFC:
# python scripts/postprocess/run_generate_matrix_find_cell_cover.py --dataset DLPFC --section-id 151507 --quality-filter baseQ13mapQ20

# For P4_TUMOR:
# python filter_bams.py --dataset P4_TUMOR --section-id 1 --quality-filter baseQ0mapQ0

# For 10X_BC_FFPE:
# python filter_bams.py --dataset 10X_BC_FFPE --quality-filter baseQ0mapQ0