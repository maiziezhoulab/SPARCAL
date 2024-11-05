import os
import random
import time
import subprocess
import argparse

def sample_bam_files(bam_dir, sample_size):
    all_bam_files = [f for f in os.listdir(bam_dir) if f.endswith('.bam')]
    return random.sample(all_bam_files, min(sample_size, len(all_bam_files)))

def run_self_caller(script_path, reference_seq, chromosome, bamfile, bedfile, header, out):
    start_time = time.time()
    
    command = [
        "python", script_path,
        "--reference_seq", reference_seq,
        "--chromosome", chromosome,
        "--bamfile", bamfile,
        "--bedfile", bedfile,
        "--header", header,
        "--out", out
    ]
    
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return execution_time, stdout, stderr

def main():
    parser = argparse.ArgumentParser(description="Benchmark self_caller.py")
    parser.add_argument("--bam_dir", required=True, help="Directory containing BAM files")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of BAM files to sample")
    parser.add_argument("--script_path", required=True, help="Path to self_caller.py")
    parser.add_argument("--reference_seq", required=True, help="Path to reference sequence")
    parser.add_argument("--bedfile", required=True, help="Path to BED file")
    parser.add_argument("--header", required=True, help="Path to header file")
    parser.add_argument("--output_dir", required=True, help="Directory to store output VCF files")
    args = parser.parse_args()

    sampled_bams = sample_bam_files(args.bam_dir, args.sample_size)
    
    total_time = 0
    for bam_file in sampled_bams:
        print(f"Processing {bam_file}")
        bamfile_path = os.path.join(args.bam_dir, bam_file)
        output_path = os.path.join(args.output_dir, f"{os.path.splitext(bam_file)[0]}.vcf")
        
        # Process each chromosome
        for chromosome in list(range(1, 23)) + ['X', 'Y']:
            execution_time, stdout, stderr = run_self_caller(
                args.script_path, args.reference_seq, str(chromosome),
                bamfile_path, args.bedfile, args.header, output_path
            )
            total_time += execution_time
            print(f"  Chromosome {chromosome} processed in {execution_time:.2f} seconds")
            
            if stderr:
                print(f"  Error: {stderr.decode('utf-8')}")
        
        print(f"Finished processing {bam_file}")
        print("------------------------")
    
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Average time per BAM file: {total_time / len(sampled_bams):.2f} seconds")

if __name__ == "__main__":
    main()


# python benchmark_self_caller.py \
#     --bam_dir /path/to/bam/files \
#     --sample_size 5 \
#     --script_path /path/to/self_caller.py \
#     --reference_seq /path/to/reference/sequence \
#     --bedfile /path/to/bedfile \
#     --header /path/to/header \
#     --output_dir /path/to/output/directory
