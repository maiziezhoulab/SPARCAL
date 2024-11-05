import pandas as pd
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import subprocess

def count_lines(vcf_file):
    result = subprocess.run(['zgrep', '-cv', '^#', vcf_file], stdout=subprocess.PIPE)
    return int(result.stdout.strip())

def process_line(line):
    parts = line.strip().split('\t')
    chrom = parts[0]
    pos = parts[1]
    ref = parts[3]
    alt = parts[4]
    identifier = f"{chrom}_{pos}_{ref}_{alt}"
    genotypes = parts[9:]
    row = []
    for genotype in genotypes:
        alleles = genotype.split('|')
        row.extend(alleles)
    return [identifier] + row

def process_vcf(vcf_file, output_file, num_threads=4):
    rows = []
    columns = []
    lines = []
    print("Counting lines")
    total_lines = count_lines(vcf_file)

    with gzip.open(vcf_file, 'rt') as f:
        print(f"Opening {vcf_file}")
        for line in tqdm(f, total=total_lines, desc="Reading VCF", unit="line"):
            if line.startswith('#CHROM'):
                header = line.strip().split('\t')
                samples = header[9:]
                for sample in samples:
                    columns.append(f"{sample}_1")
                    columns.append(f"{sample}_2")
            elif not line.startswith('#'):
                lines.append(line)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        print("Processing rows")
        futures = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Appending rows", unit="row"):
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    df = pd.DataFrame(rows, columns=["identifier"] + columns)
    df.set_index("identifier", inplace=True)
    print(f"Saving into {output_file}")
    df.to_pickle(output_file)

# Usage
root = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/testing/'
vcf_file = root + 'ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.vcf.gz'
output_file = root + 'ALL.chr1.phase3_shapeit2_mvncall_integrated_v5a.20130502.genotypes.pkl'
process_vcf(vcf_file, output_file, num_threads=40)
