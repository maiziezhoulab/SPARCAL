import json
import sys
import os
import pandas as pd
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Converts a text file to a JSON file
def convert_txt_to_json(txt_file_path, json_file_path):
    data_dict = {}
    with open(txt_file_path, 'r') as file:
        for line in file:
            key, value = line.split(" : ", 1)
            value = value.strip()
            value = eval(value)
            data_dict[key] = value
    with open(json_file_path, 'w') as json_file:
        json.dump(data_dict, json_file, indent=4)
    print("Data has been successfully converted to JSON format.")

# Extracts SNVs from a VCF file
def extract_snvs(filename):
    snvs = set()
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            chrom = parts[0]
            pos = parts[1]
            snvs.add((chrom, pos))
    return snvs

# Processes target cells and their neighbors to compute SNV sharing
def process_target_cell(target_cell, neighbors, root, result_path):
    snvs_by_file = {}
    relevant_cells = [target_cell] + neighbors
    vcf_files = [os.path.join(root, cell + ".vcf") for cell in relevant_cells if os.path.exists(os.path.join(root, cell + ".vcf"))]
    for file in vcf_files:
        snvs = extract_snvs(file)
        snvs_by_file[os.path.basename(file)[:-4]] = snvs
    target_file_path = os.path.join(root, f"{target_cell}.vcf")
    target_snvs = extract_snvs(target_file_path) if os.path.exists(target_file_path) else set()
    columns = [f"{chrom}_{pos}" for chrom, pos in sorted(target_snvs, key=lambda x: (x[0], int(x[1])))]
    df = pd.DataFrame(0, index=snvs_by_file.keys(), columns=columns)
    for file, snvs in snvs_by_file.items():
        for snv in target_snvs:
            col_name = f"{snv[0]}_{snv[1]}"
            if snv in snvs:
                df.at[file, col_name] = 1
    snv_sharing_count = df.sum(axis=0)
    snv_sharing_count -= 1
    snv_sharing_count.to_csv(os.path.join(result_path, f'tmp/sum/sum_{target_cell}.csv'), header=['Count'])
    df.to_csv(os.path.join(result_path, f"tmp/matrix/sharing_with_{target_cell}.csv"), index=True)

# Main function to read neighbors and process them
def process_neighbors(root, result_path, neighbors_file):
    with open(neighbors_file, 'r') as file:
        neighbors_info = json.load(file)
    max_threads = 40
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [executor.submit(process_target_cell, target_cell, neighbors, root, result_path) for target_cell, neighbors in neighbors_info.items()]
        for future in as_completed(futures):
            future.result()
    print("SNV processing complete.")

# Filters VCF file based on a threshold and SNV sharing
def read_csv_filter(csv_file, threshold):
    filtered_positions = []
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            chrom_pos, count = row
            count = int(count)
            if count >= threshold:  # Change the condition to 'at least' the threshold
                chrom, pos = chrom_pos.split('_')
                filtered_positions.append([chrom, pos])
    return filtered_positions

def filter_vcf(vcf_file, filtered_positions):
    with open(vcf_file, 'r') as f:
        lines = f.readlines()
    
    header_lines = [line for line in lines if line.startswith('##')]
    column_header = [line for line in lines if line.startswith('#CHROM')][0]
    snv_lines = [line for line in lines if not line.startswith('#')]

    filtered_snv_lines = []
    for line in snv_lines:
        parts = line.strip().split('\t')
        chrom, pos = parts[0], parts[1]
        if [chrom, pos] in filtered_positions:
            filtered_snv_lines.append(line)
    
    return header_lines, column_header, filtered_snv_lines

def write_filtered_vcf(output_vcf, header_lines, column_header, filtered_snv_lines):
    with open(output_vcf, 'w') as f:
        for line in header_lines:
            f.write(line)
        f.write(column_header)
        for line in filtered_snv_lines:
            f.write(line)

def filter_snv_by_neighbor(sum_file, input_vcf, output_vcf, threshold):
    filtered_positions = read_csv_filter(sum_file, threshold)
    header_lines, column_header, filtered_snv_lines = filter_vcf(input_vcf, filtered_positions)
    write_filtered_vcf(output_vcf, header_lines, column_header, filtered_snv_lines)

# Main function to integrate the full process
def main(input_path, result_path, neighbor_data_dir, threshold):
    # Step 1: Convert neighbors.txt to neighbors.json
    txt_file_path = os.path.join(neighbor_data_dir, 'neighbors.txt')
    json_file_path = os.path.join(neighbor_data_dir, 'neighbors.json')
    convert_txt_to_json(txt_file_path, json_file_path)

    # Step 2: Process SNVs
    process_neighbors(input_path, result_path, json_file_path)

    # Step 3: Filter SNVs based on sharing count
    sum_dir = os.path.join(result_path, "tmp/sum")

    sum_files = [os.path.join(sum_dir, f) for f in os.listdir(sum_dir) if f.startswith('sum_')]
    max_jobs = 40

    with ThreadPoolExecutor(max_workers=max_jobs) as executor:
        futures = []
        for sum_file in sum_files:
            barcode = os.path.basename(sum_file).replace('sum_', '').replace('.csv', '')
            input_vcf = os.path.join(input_path, f"{barcode}.vcf")
            output_vcf = os.path.join(result_path, f"{barcode}.vcf")
            if os.path.exists(input_vcf):
                futures.append(executor.submit(filter_snv_by_neighbor, sum_file, input_vcf, output_vcf, threshold))
        for future in as_completed(futures):
            future.result()

    print("Filtering SNVs by neighbor is complete.")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python unified_script.py <input_path> <output_path> <neighbor_data_dir> <threshold>")
        sys.exit(1)

    input_path = sys.argv[1]
    result_path = sys.argv[2]
    neighbor_data_dir = sys.argv[3]
    threshold = int(sys.argv[4])

    main(input_path, result_path, neighbor_data_dir, threshold)
