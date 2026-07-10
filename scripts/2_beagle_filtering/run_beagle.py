import os
import time
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set
from tqdm import tqdm
import gzip
import argparse

# Configuration dictionaries from mpileup_pipeline.py
REFERENCE_CONFIGS = {
    "DLPFC": {
        "path": "/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa",
        "chr_prefix": "",  # No "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    },
    "TUMOR": {
        "path": "/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/fasta/genome.fa",
        "chr_prefix": "chr",  # Has "chr" prefix
        "regions": [f"chr{i}" for i in range(1, 23)]  # chr1, chr2, chr3, ..., chr22
    }
}

# DLPFC_PREDEDUP=1 -> write to a SEPARATE tree (data/dlpfc_prededup) so the
# post-dedup results under data/dlpfc are left untouched. Default: unchanged.
_PREDEDUP = os.environ.get("DLPFC_PREDEDUP") == "1"

DATASET_CONFIGS = {
    "DLPFC": {
        "base_path": "/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD",
        "output_dir": "data/dlpfc_prededup/{section_id}" if _PREDEDUP else "data/dlpfc/{section_id}",
        "has_sections": True,
        "reference": "DLPFC"
    },
    "P4_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium",
        "output_dir": "data/P4_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "P6_TUMOR": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium",
        "output_dir": "data/P6_tumor/{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "TUMOR"
    },
    "DCIS": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium",
        "output_dir": "data/dcis{section_id}",
        "has_sections": True,
        "section_ids": ["1", "2"],
        "reference": "DLPFC"
    },
    "OVAR_P5": {
        # GRCh38, chr prefix. Beagle: genome_build=GRCh38 (reference != TUMOR),
        # regions chr1..chr22 (DLPFC entry) — matches the chr-prefixed mpileup VCFs.
        "base_path": "/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs",
        "output_dir": "data/ovar_p5/{section_id}",
        "has_sections": True,
        "section_ids": ["P5_sr13"],
        "reference": "DLPFC"
    }
}

THOUSAND_GENOME_CONFIGS = {
    "GRCh38": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38",
        "pattern": "CCDG_14151_B01_GRM_WGS_2020-08-05_{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
    },
    # example of 1kG for chrom 1:
    # /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/CCDG_14151_B01_GRM_WGS_2020-08-05_1.filtered.shapeit2-duohmm-phased.vcf.gz
    "hg19": {
        "base_path": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19/",
        "pattern": "hg19_chr{chrom}.vcf.gz"
    }
}

# Default Parameters
DEFAULT_PARAMS = {
    "THREADS": 24,
    "MEMORY": "10g",
    "MODEL_SCALE": 2,
    "ITERATIONS": 0,
    "IMPUTE": False,
    "GPROBS": True
}

PATH_CONFIG = {
    "PROJECT_DIR": "/data/maiziezhou_lab/leiy4/snv_calling",
    "APPS_DIR": "/data/maiziezhou_lab/leiy4/snv_calling/apps",
    "BEAGLE_JAR": "beagle.27Jul16.86a.jar",
    "JAVA": "src/jdk-11.0.2/bin/java",
    "THOUSAND_GENOME_DIR": "/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38/",
    "BGZIP": "/data/maiziezhou_lab/leiy4/snv_calling/apps/bgzip",
    "TABIX": "/data/maiziezhou_lab/leiy4/snv_calling/apps/tabix"
}

class BeaglePipeline:
    def __init__(self, dataset_name: str, quality_filter: str = "baseQ0mapQ0", section_id: str = None):
        self.dataset_name = dataset_name
        self.quality_filter = quality_filter
        self.section_id = section_id
        self.validate_dataset_config()
        self.setup_paths()
        self.setup_environment()
        
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
        """Setup paths based on dataset configuration."""
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        reference_config = REFERENCE_CONFIGS[dataset_config["reference"]]
        
        # Set chromosome format and regions
        self.chr_prefix = reference_config["chr_prefix"]
        self.regions = reference_config["regions"]
        
        # Set output directory
        if dataset_config["has_sections"]:
            self.output_base = os.path.join(
                PATH_CONFIG["PROJECT_DIR"],
                dataset_config["output_dir"].format(section_id=self.section_id)
            )
        else:
            self.output_base = os.path.join(
                PATH_CONFIG["PROJECT_DIR"],
                dataset_config["output_dir"]
            )
        
        # Setup output directories with quality filter
        self.output_dirs = {
            "input_vcf_dir": os.path.join(self.output_base, "output_VCFs/mpileup_multi_bam", 
                                        self.quality_filter),
            "output_vcf_dir": os.path.join(self.output_base, "output_VCFs/beagle", 
                                         self.quality_filter),
            "log_dir": os.path.join(self.output_base, "logs/beagle", 
                                  self.quality_filter),
            "metrics_dir": os.path.join(self.output_base, "metrics/beagle", 
                                     self.quality_filter)
        }
        
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
    
    def setup_environment(self):
        """Setup environment variables for the pipeline."""
        apps_dir = PATH_CONFIG['APPS_DIR']
        os.environ['PATH'] = f"{apps_dir}:{os.environ.get('PATH', '')}"
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        new_ld_path = f"{apps_dir}:{current_ld_path}" if current_ld_path else apps_dir
        os.environ['LD_LIBRARY_PATH'] = new_ld_path

    def get_1000genome_reference(self, chromosome: str) -> str:
        """Get the appropriate 1000 Genome reference file for a chromosome."""
        # Determine genome build based on dataset
        dataset_config = DATASET_CONFIGS[self.dataset_name]
        reference_name = dataset_config['reference']
        
        # Select appropriate 1000 Genomes configuration
        genome_build = "hg19" if reference_name == "TUMOR" else "GRCh38"
        genome_config = THOUSAND_GENOME_CONFIGS[genome_build]
        
        # Format chromosome name correctly
        if genome_build == "hg19":
            # For hg19, remove 'chr' prefix as 1000G files use just numbers
            chrom = chromosome.replace('chr', '')
        else:
            # For GRCh38, keep chromosome format as is
            chrom = chromosome
            
        reference_pattern = os.path.join(
            genome_config["base_path"],
            genome_config["pattern"].format(chrom=chrom)
        )
        
        if not os.path.exists(reference_pattern):
            raise FileNotFoundError(
                f"1000 Genome reference not found: {reference_pattern}\n"
                f"Genome build: {genome_build}, Chromosome: {chromosome}"
            )
        
        return reference_pattern

    def merge_vcf_fields(self, original_vcf: str, beagle_vcf: str, output_vcf: str):
        """Merge the FORMAT and INFO fields from original VCF with Beagle's output."""
        # Ensure both VCFs are indexed
        for vcf in [original_vcf, beagle_vcf]:
            if not os.path.exists(vcf + '.tbi'):
                subprocess.run(['tabix', '-p', 'vcf', vcf], check=True)
        
        # Prepare bcftools command
        cmd = [
            'bcftools', 'annotate',
            '-a', original_vcf,
            '-c', 'INFO/DP,INFO/I16,INFO/QS,INFO/SGB,INFO/RPB,INFO/MQB,INFO/MQSB,INFO/BQB,INFO/MQ0F,FORMAT/GQ,FORMAT/BAF,FORMAT/PL',
            '-O', 'z',
            '-o', output_vcf,
            beagle_vcf
        ]
        
        subprocess.run(cmd, check=True)
        subprocess.run(['tabix', '-p', 'vcf', output_vcf], check=True)

    def run_beagle_command(self, input_vcf: str, output_prefix: str, 
                          chromosome: str, params: Dict, log_file: str):
        """Run Beagle with specified parameters."""
        reference_panel = self.get_1000genome_reference(chromosome)
        beagle_output = f"{output_prefix}.temp.vcf.gz"
        
        cmd = [
            os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["JAVA"]),
            f"-Xmx{params['MEMORY']}",
            "-jar",
            os.path.join(PATH_CONFIG["APPS_DIR"], PATH_CONFIG["BEAGLE_JAR"]),
            f"gl={input_vcf}",
            f"ref={reference_panel}",
            f"chrom={chromosome}",
            f"out={output_prefix}.temp",
            f"impute={'true' if params['IMPUTE'] else 'false'}",
            f"modelscale={params['MODEL_SCALE']}",
            f"nthreads={params['THREADS']}",
            f"gprobs={'true' if params['GPROBS'] else 'false'}",
            f"niterations={params['ITERATIONS']}"
        ]
        
        with open(log_file, 'w') as log:
            log.write(f"Command: {' '.join(cmd)}\n\n")
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        
        # Merge original fields with Beagle output
        final_output = f"{output_prefix}.vcf.gz"
        self.merge_vcf_fields(input_vcf, beagle_output, final_output)
        
        # Cleanup temporary file
        os.remove(beagle_output)
        if os.path.exists(f"{beagle_output}.tbi"):
            os.remove(f"{beagle_output}.tbi")

    def process_chromosome(self, chromosome: str, params: Dict) -> Dict:
        """Process a single chromosome with Beagle."""
        start_time = time.time()
        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"], "merged_sorted_gt.vcf.gz")
        output_prefix = os.path.join(self.output_dirs["output_vcf_dir"], f"{chromosome}")
        log_file = os.path.join(self.output_dirs["log_dir"], f"{chromosome}.log")
        
        try:
            self.run_beagle_command(input_vcf, output_prefix, chromosome, params, log_file)
            duration = time.time() - start_time
            
            beagle_output = f"{output_prefix}.vcf.gz"
            if not os.path.exists(beagle_output):
                raise FileNotFoundError(f"Expected output file not found: {beagle_output}")
            
            return {
                "chromosome": chromosome,
                "duration": duration,
                "status": "completed",
                "output_file": beagle_output
            }
            
        except Exception as e:
            return {
                "chromosome": chromosome,
                "duration": time.time() - start_time,
                "status": "failed",
                "error": str(e)
            }

    def run_pipeline(self, custom_params: Optional[Dict] = None) -> pd.DataFrame:
        """Run the complete Beagle pipeline."""
        params = DEFAULT_PARAMS.copy()
        if custom_params:
            params.update(custom_params)
        
        input_vcf = os.path.join(self.output_dirs["input_vcf_dir"], "merged_sorted_gt.vcf.gz")
        if not os.path.exists(input_vcf):
            raise FileNotFoundError(f"Input VCF not found: {input_vcf}")
        
        # Process chromosomes in parallel
        results = []
        with ThreadPoolExecutor(max_workers=min(params["THREADS"], len(self.regions))) as executor:
            future_to_chrom = {
                executor.submit(
                    self.process_chromosome,
                    chrom,
                    params
                ): chrom for chrom in self.regions
            }
            
            with tqdm(total=len(self.regions), desc="Processing chromosomes") as pbar:
                for future in as_completed(future_to_chrom):
                    chrom = future_to_chrom[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "chromosome": chrom,
                            "status": "failed",
                            "error": str(e)
                        })
                    pbar.update(1)
        
        # Create metrics DataFrame
        metrics_df = pd.DataFrame(results)
        
        # Only proceed with filtering if all chromosomes completed successfully
        if len(metrics_df[metrics_df['status'] == 'completed']) == len(self.regions):
            print("\nCollecting passed variants across all chromosomes...")
            all_passed_variants = set()
            for _, row in metrics_df[metrics_df['status'] == 'completed'].iterrows():
                passed_variants = self.collect_passed_variants(row['output_file'])
                all_passed_variants.update(passed_variants)
            print(f"Total passed variants: {len(all_passed_variants)}")
            
            print("Creating combined filtered and filtered out VCF...")
            output_filtered_in = os.path.join(self.output_dirs["output_vcf_dir"], "all_filtered_in.vcf.gz")
            output_filtered_out = os.path.join(self.output_dirs["output_vcf_dir"], "all_filtered_out.vcf.gz")
            self.create_filtered_vcf(input_vcf, all_passed_variants, output_filtered_in, output_filtered_out)
            # print(f"Total output_filtered
            print(f"Combined filtered VCF created: {output_filtered_in}")
            print(f"Combined filtered out VCF created: {output_filtered_out}")
        
        # Save metrics
        metrics_file = os.path.join(self.output_dirs["metrics_dir"], 
                                  f"{self.dataset_name}_beagle_metrics.csv")
        if self.section_id:
            metrics_file = os.path.join(self.output_dirs["metrics_dir"],
                                      f"{self.dataset_name}_{self.section_id}_beagle_metrics.csv")
        metrics_df.to_csv(metrics_file, index=False)
        
        return metrics_df

    def collect_passed_variants(self, beagle_vcf: str) -> Set[str]:
        """Collect variants that passed Beagle processing."""
        passed_positions = set()
        with gzip.open(beagle_vcf, 'rt') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.strip().split('\t')
                passed_positions.add(f"{fields[0]}_{fields[1]}_{fields[3]}_{fields[4]}")
        return passed_positions

    def create_filtered_vcf(self, input_vcf: str, passed_variants: Set[str], output_filtered_in: str, output_filtered_out: str):
        """Create VCF files containing variants that passed/failed Beagle."""
        # Create temporary uncompressed files
        temp_filtered_in = output_filtered_in.replace('.gz', '')
        temp_filtered_out = output_filtered_out.replace('.gz', '')
        
        # Process the VCF file
        with gzip.open(input_vcf, 'rt') as f_in, \
            open(temp_filtered_in, 'w') as f_filtered_in, \
            open(temp_filtered_out, 'w') as f_filtered_out:
            
            # Copy header
            for line in f_in:
                if line.startswith('#'):
                    f_filtered_in.write(line)
                    f_filtered_out.write(line)
                    if line.startswith('#CHROM'):
                        break
            
            # Write variants based on whether they passed Beagle
            for line in f_in:
                fields = line.strip().split('\t')
                pos_key = f"{fields[0]}_{fields[1]}_{fields[3]}_{fields[4]}"
                
                if pos_key in passed_variants:
                    f_filtered_in.write(line)
                else:
                    f_filtered_out.write(line)
        
        # Compress with bgzip
        for temp_file, output_file in [(temp_filtered_in, output_filtered_in), 
                                    (temp_filtered_out, output_filtered_out)]:
            subprocess.run([PATH_CONFIG['BGZIP'], '-f', temp_file], check=True)
            
            # Index with tabix
            subprocess.run([PATH_CONFIG['TABIX'], '-p', 'vcf', output_file], check=True)
            
            # Clean up temporary files
            if os.path.exists(temp_file):
                os.remove(temp_file)

def main():
    parser = argparse.ArgumentParser(description="Beagle Pipeline for SNV Processing")
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIGS.keys()),
                      help="Dataset to process")
    parser.add_argument("--section_id", help="Section ID (required for some datasets)")
    parser.add_argument("--threads", type=int, default=DEFAULT_PARAMS["THREADS"],
                      help="Number of threads to use")
    parser.add_argument("--memory", default=DEFAULT_PARAMS["MEMORY"],
                      help="Memory allocation for Beagle (e.g., '20g')")
    parser.add_argument("--quality-filter", default="baseQ0mapQ0",
                      help="Quality filter directory (default: baseQ0mapQ0)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = BeaglePipeline(
        dataset_name=args.dataset,
        quality_filter=args.quality_filter,
        section_id=args.section_id
    )
    
    # Set custom parameters
    custom_params = {
        "THREADS": args.threads,
        "MEMORY": args.memory
    }
    
    # Run pipeline
    print(f"\nStarting Beagle pipeline for {args.dataset}")
    if args.section_id:
        print(f"Section ID: {args.section_id}")
    print(f"Using chromosome format: {pipeline.chr_prefix}N")
    print(f"Quality filter: {args.quality_filter}")
    
    result_df = pipeline.run_pipeline(custom_params=custom_params)
    
    # Print summary
    print("\nBeagle Pipeline Summary:")
    print(f"Total chromosomes processed: {len(result_df)}")
    print(f"Successful runs: {len(result_df[result_df['status'] == 'completed'])}")
    print(f"Failed runs: {len(result_df[result_df['status'] == 'failed'])}")
    if len(result_df[result_df['status'] == 'failed']) > 0:
        print("\nFailed chromosomes:")
        failed_df = result_df[result_df['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"Chromosome {row['chromosome']}: {row['error']}")
    if 'duration' in result_df.columns:
        print(f"Average processing time: {result_df['duration'].mean():.2f} seconds")

if __name__ == "__main__":
    main()

# usage on P4_tumor, baseQ0mapQ0
# python scripts/filtering/run_beagle.py --dataset P4_TUMOR --section_id 1 --quality-filter baseQ0mapQ0

# Run beagle on DLPFC 151507
# python scripts/filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ0mapQ0 --section_id 151507