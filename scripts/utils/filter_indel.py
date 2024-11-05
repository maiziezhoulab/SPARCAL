import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

def read_vcf(path):
    # Read a VCF file, preserving header lines starting with '##'
    headers = []
    lines = []
    columns = None
    
    with open(path, 'r') as file:
        for line in file:
            if line.startswith('##'):
                headers.append(line.strip())
            elif line.startswith('#') and columns is None:
                columns = line.strip().split('\t')
            else:
                lines.append(line.strip().split('\t'))

    if columns is None:
        raise ValueError(f"No column definition found in VCF file: {path}")
    
    return headers, pd.DataFrame(lines, columns=columns)

def write_vcf(headers, df, path):
    # Write the DataFrame to a VCF file, including original headers
    with open(path, 'w') as file:
        for header in headers:
            file.write(header + '\n')
        df.to_csv(file, sep='\t', index=False, mode='a')

def filter_snvs(df):
    # Filter out SNVs where REF or ALT lengths are more than 1
    return df[(df['REF'].apply(len) == 1) & (df['ALT'].apply(lambda x: len(x) == 1))]

def process_vcf(barcode, input_vcf_dir, output_vcf_dir):
    print(f"Processing {barcode}...")
    input_vcf_path = os.path.join(input_vcf_dir, f'{barcode}/new_rg.vcf')
    output_vcf_path = os.path.join(output_vcf_dir, f'{barcode}.vcf')

    if not os.path.exists(input_vcf_path):
        print(f"Input VCF file for barcode {barcode} not found.")
        return

    try:
        # Load the VCF into a DataFrame, including headers
        headers, df_vcf = read_vcf(input_vcf_path)

        # Filter SNVs with REF or ALT longer than 1
        df_vcf_filtered = filter_snvs(df_vcf)

        # Write the filtered DataFrame to a new VCF file, including original headers
        write_vcf(headers, df_vcf_filtered, output_vcf_path)

        print(f"Filtered VCF for {barcode} has been written to {output_vcf_path}.")
    except Exception as e:
        print(f"Error processing VCF for barcode {barcode}: {e}")

def main(number):
    # input_vcf_dir = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{number}/gatk/output_VCFs/raw/0"
    # output_vcf_dir = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{number}/gatk/output_VCFs/unfiltered/0"
    input_vcf_dir = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/V1_Mouse_Brain_Sagittal_Anterior_Section_2/gatk/output_VCFs/raw/0"
    output_vcf_dir = f"/data/maiziezhou_lab/hanliu/projects/snv_call/data/V1_Mouse_Brain_Sagittal_Anterior_Section_2/gatk/output_VCFs/unfiltered/0"

    # Create the output directory if it does not exist
    os.makedirs(output_vcf_dir, exist_ok=True)

    # Get the list of barcodes from the directory
    barcodes = [d for d in os.listdir(input_vcf_dir) if os.path.isdir(os.path.join(input_vcf_dir, d))]

    # Use ThreadPoolExecutor to process multiple barcodes concurrently
    max_threads = 40
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_vcf, barcode, input_vcf_dir, output_vcf_dir) for barcode in barcodes]
        
        for future in as_completed(futures):
            future.result()  # This will re-raise any exception that occurred in the thread

    print("Processing complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <number>")
        sys.exit(1)
    
    number = sys.argv[1]
    main(number)
