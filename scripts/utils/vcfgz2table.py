import pandas as pd
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import glob
import os

def count_lines(vcf_file):
    result = subprocess.run(['zgrep', '-cv', '^#', vcf_file], stdout=subprocess.PIPE)
    return int(result.stdout.strip())

def process_line(line):
    parts = line.strip().split('\t')
    chrom = parts[0]
    pos = parts[1]
    ref = parts[3]
    alt = parts[4]
    
    if len(ref) == 1 and len(alt) == 1:  # Filter out non-SNVs
        return [chrom, pos, ref, alt]
    return None

def process_vcf(vcf_file, output_file, num_threads=4):
    rows = []
    lines = []

    # total_lines = count_lines(vcf_file)

    with gzip.open(vcf_file, 'rt') as f:
        print(f"Opening {vcf_file}")
        for line in f:
            if not line.startswith('#'):
                lines.append(line)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        print("Processing rows")
        futures = {executor.submit(process_line, line): line for line in lines}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    rows.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    df = pd.DataFrame(rows, columns=["CHROM", "POS", "REF", "ALT"])
    print(f"Saving into {output_file}")
    df.to_csv(output_file, index=False)

# Usage
input_path = '/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome/'
output_path  = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/reference/1000Genome/'

def collect_file_names(folder):
    pattern = os.path.join(folder, 'ALL.chr[0-9XY]*.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz')
    file_names = glob.glob(pattern)
    return file_names

file_names = collect_file_names(input_path)

for path in file_names:
    output_file = output_path + os.path.basename(path).replace('.vcf.gz', '.csv')
    process_vcf(path, output_file, num_threads=40)
