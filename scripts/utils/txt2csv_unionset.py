import pandas as pd
import os

# List of IDs
ids = [151507, 151508, 151509, 151510, 151669, 151670, 151671, 151672, 151673, 151674, 151675, 151676]

# Base directory
base_dir = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC'

# Loop through each ID
for id in ids:
    # Construct the file path
    input_file_path = f'{base_dir}/{id}/self/results/filtered_by_1000Genome/0/{id}_gatk_ref_on.txt'
    
    # Check if the input file exists
    if os.path.exists(input_file_path):
        # Read the text file into a pandas DataFrame
        df = pd.read_csv(input_file_path, sep='\t')
        
        # Keep only the required columns: CHROM, POS, REF, ALT.x, and 20
        df_filtered = df[['CHROM', 'POS', 'REF', 'ALT.x', '20']]
        
        # Rename ALT.x to ALT
        df_filtered.rename(columns={'ALT.x': 'ALT'}, inplace=True)
        
        # Construct the output file path (same directory as input)
        output_file_path = os.path.join(os.path.dirname(input_file_path), f'union_set.csv')
        
        # Save the filtered DataFrame to a CSV file
        df_filtered.to_csv(output_file_path, index=False, sep=',')
        
        print(f"CSV file has been created successfully for ID {id}: {output_file_path}")
    else:
        print(f"File does not exist for ID {id}: {input_file_path}")
