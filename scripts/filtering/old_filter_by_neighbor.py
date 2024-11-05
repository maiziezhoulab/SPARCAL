import csv
import sys

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

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python filter_snv_by_neighbor_snvs.py <threshold> <snv_sum.csv> <input.vcf> <output.vcf>")
        sys.exit(1)
    
    threshold = int(sys.argv[1])
    csv_file = sys.argv[2]
    vcf_file = sys.argv[3]
    output_vcf = sys.argv[4]

    filtered_positions = read_csv_filter(csv_file, threshold)
    header_lines, column_header, filtered_snv_lines = filter_vcf(vcf_file, filtered_positions)
    write_filtered_vcf(output_vcf, header_lines, column_header, filtered_snv_lines)

