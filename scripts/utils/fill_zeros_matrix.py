import os
import pandas as pd
import configparser
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process VCF files.')
parser.add_argument('id', type=str, help='ID for processing')
args = parser.parse_args()

# Use the provided ID
config_file_path = f'{args.id}.ini'

# Rest of your code...
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file_path)

TISSUE_POS_PATH  = config['INPUT']['TISSUE_POS_PATH']
MATRIX_PATH = config['RESULT']['MATRIX_PATH']
MODIFIED_MATRIX_PATH = config['RESULT']['MODIFIED_MATRIX_PATH']

def filling_zero_matrix():
    df = pd.read_pickle(MATRIX_PATH)
    print(df.head())
    # Modify the index to remove the "processed_" prefix and "_merge" suffix
    df.index = df.index.str.replace("processed_", "").str.replace("_merge", "")
    print(f'extracted_vcf_barcodes: {len(df)}')

    column_names = ['barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']
    data = pd.read_csv(TISSUE_POS_PATH, names = column_names)
    in_tissue_barcodes = data[data["in_tissue"] == '1']["barcode"].tolist()
    print(f'in_tissue_barcodes: {len(in_tissue_barcodes)}')
    df_barcodes = set(df.index)
    # Step 2: Find the barcodes that are in barcode_list but not in df_barcodes
    missing_barcodes = [barcode for barcode in in_tissue_barcodes if barcode not in df_barcodes]

    # Step 3: Create a new DataFrame with these missing barcodes and columns filled with 0s
    missing_df = pd.DataFrame(0, index=missing_barcodes, columns=df.columns)
    # print(len(missing_df))
    # Step 4: Append this new DataFrame to the original DataFrame
    df = pd.concat([df, missing_df])
    # print(df.head())
    df.to_pickle(MODIFIED_MATRIX_PATH)

filling_zero_matrix()
# print(f"filling_zero_matrix is done for ID {args.id}")