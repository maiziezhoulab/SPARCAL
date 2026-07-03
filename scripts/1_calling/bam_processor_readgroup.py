import pysam
import os
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def update_bam_readgroups(input_bam: str, output_bam: str) -> None:
    """
    Update read groups in a BAM file to use barcode as sample name.
    
    Args:
        input_bam: Path to input BAM file
        output_bam: Path to output BAM file
    """
    # Open input BAM file
    with pysam.AlignmentFile(input_bam, "rb") as inbam:
        # Get header from input BAM
        header = inbam.header.to_dict()
        
        # Clear existing read groups
        if 'RG' in header:
            header['RG'] = []
            
        # Extract barcode from filename
        barcode = os.path.basename(input_bam).replace('.bam', '')
        
        # Add new read group with barcode as sample name
        header['RG'].append({
            'ID': barcode,
            'SM': barcode,
            'LB': 'lib1',
            'PL': 'ILLUMINA'
        })
        
        # Open output BAM file
        with pysam.AlignmentFile(output_bam, "wb", header=header) as outbam:
            # Iterate through reads
            for read in inbam:
                # Update read group for each read
                read.set_tag('RG', barcode)
                outbam.write(read)
    
    # Index the output BAM file
    pysam.index(output_bam)

def process_sample_bams(input_dir: str, output_dir: str, threads: int = 30) -> List[str]:
    """
    Process all BAM files in a directory to update their read groups.
    
    Args:
        input_dir: Directory containing input BAM files
        output_dir: Directory for processed BAM files
        threads: Number of parallel threads to use
        
    Returns:
        List of processed BAM file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    bam_files = [f for f in os.listdir(input_dir) if f.endswith('.bam')]
    processed_bams = []
    
    def process_single_bam(bam_file: str) -> str:
        input_path = os.path.join(input_dir, bam_file)
        output_path = os.path.join(output_dir, bam_file)
        update_bam_readgroups(input_path, output_path)
        return output_path
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = []
        for bam_file in bam_files:
            future = executor.submit(process_single_bam, bam_file)
            futures.append(future)
            
        # Show progress bar
        for future in tqdm(futures, desc="Processing BAM files"):
            processed_bams.append(future.result())
            
    return processed_bams

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Update BAM files with barcode read groups")
    parser.add_argument("--section_id", required=True, help="Section ID")
    parser.add_argument("--input_dir", help="Input BAM directory")
    parser.add_argument("--output_dir", help="Output BAM directory")
    parser.add_argument("--threads", type=int, default=30, help="Number of threads")
    
    args = parser.parse_args()
    
    # Set default directories if not provided
    if not args.input_dir:
        args.input_dir = f"/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{args.section_id}/bam_bycell"
    if not args.output_dir:
        args.output_dir = f"/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{args.section_id}/bam_bycell_processed"
    
    processed_bams = process_sample_bams(args.input_dir, args.output_dir, args.threads)
    print(f"Processed {len(processed_bams)} BAM files")
    print(f"Output files are in: {args.output_dir}")

if __name__ == "__main__":
    main()


    # python bam_processor_readgroup.py --section_id 151507 --input_dir /data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/151507/bam_bycell --output_dir /data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_bam_processed/151507/bam_bycell