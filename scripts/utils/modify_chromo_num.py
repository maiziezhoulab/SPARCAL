import pandas as pd

def read_vcf(path):
    # Read a VCF file, preserving header lines starting with '##'
    with open(path, 'r') as file:
        headers = []
        lines = []
        for line in file:
            if line.startswith('##'):
                headers.append(line.strip())
            elif line.startswith('#') and not lines:
                columns = line.strip().split('\t')
            else:
                lines.append(line.strip().split('\t'))
    return headers, pd.DataFrame(lines, columns=columns)

def write_vcf(headers, df, path):
    # Write the DataFrame to a VCF file, including original headers
    with open(path, 'w') as file:
        for header in headers:
            file.write(header + '\n')
        df.to_csv(file, sep='\t', index=False, mode='a')

def convert_chromo_numbers(df):
    # Dictionary mapping old chromosome numbers to new ones
    chromo_map = {'2': '22', '22': '21', '21': '20', '20': '2'}
    df['#CHROM'] = df['#CHROM'].map(chromo_map).fillna(df['#CHROM'])
    return df

# Path to the original VCF
root_dir = "/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/scripts/snv_call/data/DLPFC/output_VCFs"
input_vcf_path = root_dir + '/TCCCGTCAGTCCCGCA-1_merge.vcf'
# Path for the new VCF
output_vcf_path = root_dir + '/modified_TCCCGTCAGTCCCGCA-1_merge.vcf'

# Load the VCF into a DataFrame, including headers
headers, df_vcf = read_vcf(input_vcf_path)

# Convert chromosome numbers
df_vcf_modified = convert_chromo_numbers(df_vcf)

# Write the modified DataFrame to a new VCF file, including original headers
write_vcf(headers, df_vcf_modified, output_vcf_path)

print(f"Modified VCF has been written to {output_vcf_path}.")
