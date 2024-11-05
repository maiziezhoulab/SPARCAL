# root = "/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/scripts/snv_call/data/DLPFC/raw_VCFs/"
# # merge = '_merge'
# merge = ''
# # List of VCF file names
# vcf_names = ['AGGTTGAGGCACGCTT-1', 'TCCCGTCAGTCCCGCA-1', 'CCTCTAATCTGCCAAG-1', 'AACGATATGTCAACTG-1', 'TGCAGGATCGGCAAAG-1', 'CAGCCTCCTGCAGAGG-1', 'TCCGATGACTGAGCTC-1']
# vcf_files = [root + v + f'{merge}.vcf' for v in vcf_names]
# import itertools
# from tabulate import tabulate

# # Function to read SNVs (CHROM and POS) from a VCF file
# def read_snvs(vcf_path):
#     snvs = set()
#     with open(vcf_path, 'r') as file:
#         for line in file:
#             if line.startswith('#'):
#                 continue  # Skip header lines
#             parts = line.strip().split()
#             chrom = parts[0]
#             pos = parts[1]
#             snvs.add((chrom, pos))
#     return snvs

# # Read SNVs from each file
# snvs_by_file = {vcf_name: read_snvs(vcf_path) for vcf_name, vcf_path in zip(vcf_names, vcf_files)}

# # Create a matrix to hold the number of common SNVs between each pair of files
# matrix = [[0] * len(vcf_files) for _ in range(len(vcf_files))]

# for i, vcf1 in enumerate(vcf_names):
#     for j, vcf2 in enumerate(vcf_names):
#         if i != j:
#             matrix[i][j] = len(snvs_by_file[vcf1].intersection(snvs_by_file[vcf2]))
#         else:
#             matrix[i][j] = len(snvs_by_file[vcf1])  # Total SNVs in the file itself for diagonal

# # Print the results as a table
# print(tabulate(matrix, headers=vcf_names, showindex=vcf_names, tablefmt="grid"))

import json
import itertools
from tabulate import tabulate

# Function to read SNVs (CHROM and POS) from a VCF file
def read_snvs(vcf_path):
    snvs = set()
    with open(vcf_path, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            parts = line.strip().split()
            chrom = parts[0]
            pos = parts[1]
            snvs.add((chrom, pos))
    return snvs

# Function to process key barcodes and their neighbors
def process_key_barcodes(json_path, root):
    # Load the key barcodes and neighbors from the JSON file
    with open(json_path, 'r') as json_file:
        barcode_dict = json.load(json_file)

    for key_barcode, neighbors in barcode_dict.items():
        # Include the key barcode itself in the list of VCF names
        vcf_names = [key_barcode] + neighbors
        vcf_files = [root + v + '.vcf' for v in vcf_names]

        # Read SNVs from each file
        snvs_by_file = {vcf_name: read_snvs(vcf_path) for vcf_name, vcf_path in zip(vcf_names, vcf_files)}

        # Create a matrix to hold the number of common SNVs between each pair of files
        matrix = [[0] * len(vcf_files) for _ in range(len(vcf_files))]

        for i, vcf1 in enumerate(vcf_names):
            for j, vcf2 in enumerate(vcf_names):
                if i != j:
                    matrix[i][j] = len(snvs_by_file[vcf1].intersection(snvs_by_file[vcf2]))
                else:
                    matrix[i][j] = len(snvs_by_file[vcf1])  # Total SNVs in the file itself for diagonal

        # Print the results as a table
        print(f"Results for key barcode: {key_barcode}")
        print(tabulate(matrix, headers=vcf_names, showindex=vcf_names, tablefmt="grid"))

# Define the root directory and the path to the JSON file
root = "/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/scripts/snv_call/data/DLPFC/raw_VCFs/"
json_path = "/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/151509/neighbors.json"

# Process key barcodes and their neighbors
process_key_barcodes(json_path, root)
