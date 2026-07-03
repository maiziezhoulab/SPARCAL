import os
import subprocess
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Configuration
SAMTOOLS = "/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools"
DATASETS = {
    "10X_BC_6.5MM": {
        "input_dir": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE/split_by_cell/BAMs",
        "output_dir": "/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE/split_by_cell/BAMs_no_chr_prefix"
    }
    # Add other datasets here if needed
}

def run_command(cmd, **kwargs):
    """Run a shell command."""
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              **kwargs)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {cmd}")
        print(f"Error message: {e.stderr.decode()}")
        return False

def create_header_map():
    """Create a header mapping file for chromosome name conversion."""
    header_content = []
    # Add standard chromosomes (1-22)
    for i in range(1, 23):
        header_content.append(f"chr{i}\t{i}")
    # Add sex chromosomes if needed
    # header_content.extend(["chrX\tX", "chrY\tY", "chrM\tMT"])
    return "\n".join(header_content)

def process_bam(input_bam, output_dir, header_file):
    """Process a single BAM file to standardize chromosome names."""
    try:
        basename = os.path.basename(input_bam)
        output_bam = os.path.join(output_dir, basename)
        
        # Create BAM index if it doesn't exist
        if not os.path.exists(input_bam + '.bai'):
            print(f"Creating index for {basename}")
            cmd = f"{SAMTOOLS} index {input_bam}"
            if not run_command(cmd):
                return False

        # Use reheader to change chromosome names
        cmd = f"{SAMTOOLS} reheader -c 'grep -v \"^@SQ\" || cat' {input_bam} | "
        cmd += f"{SAMTOOLS} view -H | grep \"^@SQ\" | "
        cmd += f"awk '{{split($2,a,\":\"); chr=a[2]; gsub(\"chr\",\"\",chr); "
        cmd += f"print $1 \"\\tSN:\" chr \"\\t\" $3}}' | "
        cmd += f"{SAMTOOLS} reheader - {input_bam} > {output_bam}"

        if not run_command(cmd):
            return False

        # Create index for the new BAM
        cmd = f"{SAMTOOLS} index {output_bam}"
        if not run_command(cmd):
            return False

        return True

    except Exception as e:
        print(f"Error processing {input_bam}: {str(e)}")
        return False

def process_dataset(dataset_name):
    """Process all BAM files in a dataset."""
    dataset_config = DATASETS.get(dataset_name)
    if not dataset_config:
        print(f"Dataset {dataset_name} not found in configuration")
        return

    input_dir = dataset_config["input_dir"]
    output_dir = dataset_config["output_dir"]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of BAM files
    bam_files = glob.glob(os.path.join(input_dir, "*.bam"))
    if not bam_files:
        print(f"No BAM files found in {input_dir}")
        return

    print(f"Found {len(bam_files)} BAM files to process")

    # Create temporary header mapping file
    header_file = os.path.join(output_dir, "header_map.txt")
    with open(header_file, 'w') as f:
        f.write(create_header_map())

    # Process BAM files in parallel
    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        for bam_file in bam_files:
            future = executor.submit(process_bam, bam_file, output_dir, header_file)
            futures.append(future)

        # Monitor progress
        with tqdm(total=len(bam_files), desc="Processing BAM files") as pbar:
            for future in futures:
                future.result()
                pbar.update(1)

    # Clean up
    os.remove(header_file)
    print(f"\nProcessing complete. Standardized BAMs are in {output_dir}")

def main():
    """Main function to run the standardization process."""
    print("Starting BAM chromosome name standardization...")
    for dataset_name in DATASETS:
        print(f"\nProcessing dataset: {dataset_name}")
        process_dataset(dataset_name)

if __name__ == "__main__":
    main()

# Run the script to standardize chromosome names in BAM files
# python scripts/tools/standardize-bams.py