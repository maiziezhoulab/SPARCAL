import os
import glob
import subprocess
import argparse
from tqdm import tqdm

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

def index_bam_file(bam_path):
    """Index a BAM file using the correct samtools version."""
    try:
        # Use os.system() instead of subprocess to avoid argument list issues
        cmd = f"{PATH_CONFIG['SAMTOOLS']} index '{bam_path}'"
        status = os.system(cmd)
        if status != 0:
            return {'status': 'error', 'bam': bam_path, 'error': f"Command failed with status {status}"}
        return {'status': 'success', 'bam': bam_path}
    except Exception as e:
        return {'status': 'error', 'bam': bam_path, 'error': str(e)}

def index_bams_in_directory(directory):
    """Index all BAM files in the specified directory sequentially."""
    # Find all BAM files
    bam_files = glob.glob(os.path.join(directory, '*.bam'))
    if not bam_files:
        print(f"No BAM files found in {directory}")
        return []
    
    print(f"Found {len(bam_files)} BAM files in {directory}")
    results = []
    
    # Process files sequentially with progress bar
    for bam in tqdm(bam_files, desc="Indexing BAM files"):
        result = index_bam_file(bam)
        results.append(result)
    
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

def main():
    parser = argparse.ArgumentParser(description="Index filtered BAM files using correct samtools version")
    parser.add_argument("--directory", required=True, help="Directory containing BAM files to index")
    args = parser.parse_args()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        return 1
    
    # Print configuration
    print(f"BAM Indexing Configuration:")
    print(f"  Directory: {args.directory}")
    print(f"  Samtools path: {PATH_CONFIG['SAMTOOLS']}")
    
    # Setup environment
    setup_environment()
    
    # Run indexing
    results = index_bams_in_directory(args.directory)
    
    # Return non-zero exit code if any indexing failed
    if any(r['status'] == 'error' for r in results):
        return 1
    return 0

if __name__ == "__main__":
    exit(main())

# Run the script with the following command:
# python scripts/tools/index_filtered_bams.py --directory /data/maiziezhou_lab/yuqi/snv_calling/data/dlpfc/151507/output_VCFs/BAM_filtered/baseQ13mapQ20