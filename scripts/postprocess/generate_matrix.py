import pandas as pd
import os
import glob
import configparser
import argparse
from io import StringIO

# Argument parser setup
parser = argparse.ArgumentParser(description='Process VCF files.')
parser.add_argument('id', type=str, help='ID for processing')
args = parser.parse_args()

# Read the config file
config_file_path = f'{args.id}'
print(config_file_path)
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file_path)

# Paths from config
NUM = int(config['DEFAULT']['NEIGHBOR_LIM'])
VCF_PATH = config['PROCESS']['OUT_VCF_PATH']
PROCESSED_TABLES_PATH = config['PROCESS']['processed_VCF_TABLES_PATH']
UNION_SET_PATH = config['RESULT']['UNION_SET_PATH']
MATRIX_OUTPUT_PATH = config['RESULT']['MATRIX_PATH']
TIS_POS = config['INPUT']['TISSUE_POS_PATH']

# Step 1 & 2: Convert VCFs to tables and process the tables
def vcf_to_dataframe(vcf_filename):
    try:
        with open(vcf_filename, 'r') as f:
            lines = [line for line in f if not line.startswith('##')]
    except FileNotFoundError:
        print(f"The file '{vcf_filename}' does not exist.")
        return pd.DataFrame()  # return an empty dataframe
        
    if not lines:
        print(f"The VCF file '{vcf_filename}' has no content or only contains meta-information lines.")
        return pd.DataFrame()  # return an empty dataframe

    df = pd.read_csv(
        StringIO(''.join(lines)),
        dtype={'#CHROM': str, 'POS': int, 'ID': str, 'REF': str, 'ALT': str,
               'QUAL': str, 'FILTER': str, 'INFO': str},
        sep='\t'
    ).rename(columns={'#CHROM': 'CHROM'})
    
    return df

def process_vcf_tables(df):
    if 'Sample' in df.columns:
        df['20'] = df['Sample'].str.split(':').str[0]
    elif '20' in df.columns:
        df['20'] = df['20'].str.split(':').str[0]

    replacements = {'1/1': '0/1', '1/0': '0/1', '1/2': '0/1'}
    df['20'] = df['20'].replace(replacements)
    df.drop_duplicates(inplace=True)
    
    return df

def step1_and_2_vcf_to_processed_tables():
    df1 = pd.read_csv(TIS_POS)
    df1 = df1[df1['in_tissue'] == 1]
    barcodes = df1["barcode"].tolist()
    
    for barcode in barcodes:
        vcf_filename = os.path.join(VCF_PATH, f'{barcode}.vcf')
        df = vcf_to_dataframe(vcf_filename)
        if not df.empty:
            processed_df = process_vcf_tables(df)
            processed_df.to_csv(os.path.join(PROCESSED_TABLES_PATH, f'processed_{barcode}.csv'), index=False)

# Step 3: Generate union set
def generate_union_set():
    all_files = glob.glob(os.path.join(PROCESSED_TABLES_PATH, "*.csv"))
    list_of_dfs = [pd.read_csv(file, index_col=None, header=0) for file in all_files]
    merged_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)

    merged_df['CHROM'] = merged_df['CHROM'].astype(str)
    merged_df.drop_duplicates(subset=['CHROM', 'POS'], inplace=True)
    merged_df = merged_df[(merged_df['REF'].apply(len) == 1) & (merged_df['ALT'].apply(len) == 1)]

    def custom_chrom_sort(chrom):
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        if chrom == 'X':
            return 23
        if chrom == 'Y':
            return 24
        if chrom == 'M':
            return 25
        return int(chrom)

    merged_df['CHROM_sort'] = merged_df['CHROM'].apply(custom_chrom_sort)
    merged_df = merged_df.sort_values(by='CHROM_sort').drop('CHROM_sort', axis=1)
    merged_df.to_csv(UNION_SET_PATH, index=False)
    return merged_df

# Step 4: Generate matrix
def merge_dfs(union_df, cell_df):
    for col in ['CHROM', 'POS', 'REF', 'ALT']:
        union_df[col] = union_df[col].astype(str)
        cell_df[col] = cell_df[col].astype(str)

    merged = pd.merge(union_df, cell_df, on=['CHROM', 'POS', 'REF', 'ALT'], how='left', indicator=True)
    match_list = (merged['_merge'] == 'both').astype(int).tolist()
    return match_list

def generate_matrix(union_df):
    filenames = [f for f in os.listdir(PROCESSED_TABLES_PATH) if os.path.isfile(os.path.join(PROCESSED_TABLES_PATH, f))]
    results = []
    for filename in filenames:
        cell_df = pd.read_csv(os.path.join(PROCESSED_TABLES_PATH, filename))
        results.append(merge_dfs(union_df, cell_df))

    cols = ["chr{}_{}".format(row["CHROM"], row["POS"]) for _, row in union_df.iterrows()]
    rows = [x.split(".")[0] for x in filenames]

    df = pd.DataFrame(results, index=rows, columns=cols)
    sparse_df = df.astype(pd.SparseDtype("int", 0))
    sparse_df.to_pickle(MATRIX_OUTPUT_PATH)

# Main workflow
def main():
    print("Step 1 & 2: Converting VCFs to tables and processing the tables")
    step1_and_2_vcf_to_processed_tables()
    
    print("Step 3: Generating union set")
    union_df = generate_union_set()

    print("Step 4: Generating matrix")
    generate_matrix(union_df)

    print(f"Workflow completed for ID {args.id}")

if __name__ == "__main__":
    main()
