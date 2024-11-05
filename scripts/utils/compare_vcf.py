import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_vcf(path):
    # Read a VCF file and ignore header lines starting with '##'
    with open(path, 'r') as file:
        lines = [line.strip() for line in file if not line.startswith('##')]
    return pd.DataFrame([line.split('\t') for line in lines[1:]], columns=lines[0].split('\t'))

def filter_snvs(df):
    # Assumes that REF and ALT columns exist and filters for SNVs
    return df[(df['REF'].apply(len) == 1) & (df['ALT'].apply(lambda x: len(x) == 1))]

def compare_vcfs(df1, df2, df_filtered, output_file):
    # Create keys to merge on
    df1['key'] = df1['#CHROM'] + '_' + df1['POS']
    df2['key'] = df2['#CHROM'] + '_' + df2['POS']
    df_filtered['key'] = df_filtered['#CHROM'] + '_' + df_filtered['POS']
    
    # Find SNVs that exist in df1 but not in df2
    df1_unique = df1[~df1['key'].isin(df2['key'])]
    
    # Merge DataFrames on key to find common SNVs
    merged = pd.merge(df1, df2, on='key', suffixes=('_1', '_2'))
    
    # Merge DataFrames on key to find common SNVs with filtered
    merged_filtered = pd.merge(df1, df_filtered, on='key', suffixes=('_1', '_filtered'))
    
    # Count common SNVs by chromosome
    common_snvs_by_chromo = merged['#CHROM_1'].value_counts()
    common_filtered_snvs_by_chromo = merged_filtered['#CHROM_1'].value_counts()
    
    # Calculate the ratio of common SNVs to all SNVs in df1 by chromosome
    gatk_counts = df1['#CHROM'].value_counts()
    self_counts = df2['#CHROM'].value_counts()
    filtered_counts = df_filtered['#CHROM'].value_counts()

    def chromo_sort_key(chromo):
        # Attempt to convert chromosome to integer for numeric sorting. Non-numeric chromosomes go last.
        try:
            return int(chromo)
        except ValueError:
            return float('inf')  # Ensures non-numeric chromosomes (X, Y) sort last

    chromos = sorted(set(gatk_counts.index).union(self_counts.index).union(filtered_counts.index), key=chromo_sort_key)

    gatk_counts = gatk_counts.reindex(chromos, fill_value=0)
    self_counts = self_counts.reindex(chromos, fill_value=0)
    filtered_counts = filtered_counts.reindex(chromos, fill_value=0)
    common_snvs_by_chromo = common_snvs_by_chromo.reindex(chromos, fill_value=0)
    common_filtered_snvs_by_chromo = common_filtered_snvs_by_chromo.reindex(chromos, fill_value=0)

    snv_ratio_by_chromo = (common_snvs_by_chromo / gatk_counts).fillna(0) * 100
    snv_ratio_by_chromo = snv_ratio_by_chromo.replace(0, 100).round().astype(int)  # Round percentages to integer
    missing_snv = gatk_counts - common_snvs_by_chromo
    
    # Calculate self-called - filtered
    self_called_minus_filtered = self_counts - filtered_counts
    
    # Calculate percentage(filter/gatk) and missing-filter
    percentage_filter_gatk = (common_filtered_snvs_by_chromo / gatk_counts).fillna(0) * 100
    percentage_filter_gatk[gatk_counts == 0] = 100  # If GATK count is 0, set percentage to 100%
    percentage_filter_gatk = percentage_filter_gatk.round().astype(int)  # Round percentages to integer
    missing_filter = gatk_counts - common_filtered_snvs_by_chromo
    
    # Calculate percentage(selfCalled/gatk) and self-miss
    percentage_self_gatk = (common_snvs_by_chromo / gatk_counts).fillna(0) * 100
    percentage_self_gatk[gatk_counts == 0] = 100  # If selfCalled count is 0, set percentage to 100%
    percentage_self_gatk = percentage_self_gatk.round().astype(int)  # Round percentages to integer
    missing_self = gatk_counts - common_snvs_by_chromo

    # Prepare table
    table = pd.DataFrame({
        'chromo': chromos,
        'gatk_snv': gatk_counts,
        'selfCalled_snv': self_counts,
        'common_self': common_snvs_by_chromo,
        'percentage(selfCalled/gatk)': percentage_self_gatk,
        'self-miss': missing_self,
        'after-filter': filtered_counts,
        'common-filter': common_filtered_snvs_by_chromo,
        'filtered': self_called_minus_filtered,
        'percentage(filter/gatk)': percentage_filter_gatk,
        'filtered-miss': missing_filter
    }).fillna(0)

    return table

def process_barcode(barcode, root_dir, merge, output_dir):
    print(f"Processing {barcode}...")

    # Define the paths for the VCF files
    gatk_path = os.path.join(root_dir, f'output_VCFs/gatk_filtered/0/{barcode}{merge}.vcf')
    self_called_path = os.path.join(root_dir, f'output_VCFs/selfCalled/0/{barcode}{merge}.vcf')
    filtered_path = os.path.join(root_dir, f'results/0/filtered/1/{barcode}{merge}.vcf')
    
    # Check if the files exist
    if not os.path.exists(gatk_path):
        print(f"File not found: {gatk_path}")
        return
    if not os.path.exists(self_called_path):
        print(f"File not found: {self_called_path}")
        return
    if not os.path.exists(filtered_path):
        print(f"File not found: {filtered_path}")
        return

    # Read the VCF files
    df_small = read_vcf(gatk_path)
    df_large = read_vcf(self_called_path)
    df_filtered = read_vcf(filtered_path)
    
    # Filter for SNVs
    df_small_snvs = filter_snvs(df_small)
    df_large_snvs = filter_snvs(df_large)
    df_filtered_snvs = filter_snvs(df_filtered)

    # Specify the output file for unique SNVs
    unique_snv_output_path = os.path.join(root_dir, f'{barcode}_different_snvs.txt')

    # Compare VCFs and create a summary table
    summary_table = compare_vcfs(df_small_snvs, df_large_snvs, df_filtered_snvs, unique_snv_output_path)

    # Save the table to the output directory
    summary_output_path = os.path.join(output_dir, f'summary_{barcode}.csv')
    # summary_table.to_csv(summary_output_path, sep='\t', index=False)
    
    return barcode, summary_table

merge = ''
root_dir = "/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/151509/"
output_dir = "/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/151509/results/0/comparison"

# Create the output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get the list of barcodes from the directory
input_vcf_dir = "/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/151509/results/0/filtered/1"
barcodes = [os.path.splitext(f)[0] for f in os.listdir(input_vcf_dir) if f.endswith('.vcf')]
# barcodes = ['AAAGGCTACGGACCAT-1']
all_summaries = []

# Use ThreadPoolExecutor to process multiple barcodes concurrently
max_threads = 40
with ThreadPoolExecutor(max_workers=max_threads) as executor:
    futures = [executor.submit(process_barcode, barcode, root_dir, merge, output_dir) for barcode in barcodes]
    
    for future in as_completed(futures):
        try:
            barcode, summary_table = future.result()
            print(f'{barcode} Summary Table:')
            print(summary_table)
            all_summaries.append((barcode, summary_table))
        except Exception as e:
            print(f"Error processing barcode: {e}")

# Save the combined summary tables to a text file
combined_summary_output_path = os.path.join(output_dir, 'combined_summary.txt')
with open(combined_summary_output_path, 'w') as f:
    for barcode, summary_table in all_summaries:
        f.write(f"Here is the summary of {barcode}\n")
        summary_table.to_csv(f, sep='\t', index=False)
        f.write("\n")

print(f'Combined summary table saved to {combined_summary_output_path}')
