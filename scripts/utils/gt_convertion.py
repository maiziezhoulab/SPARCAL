import pandas as pd
import os

# Define the file paths
input_file_path = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/V1_Mouse_Brain_Sagittal_Anterior_Section_2/tissue_positions_list_GTs.txt'
output_file_path = '/data/maiziezhou_lab/hanliu/projects/snv_call/data/V1_Mouse_Brain_Sagittal_Anterior_Section_2/processed_tissue_positions_list_GTs.csv'

# Load the file into a pandas DataFrame
df = pd.read_csv(input_file_path, sep='\t')

# Drop the specified columns
columns_to_remove = ['ground_truth_1', 'ground_truth_2', 'ground_truth_3', 'ground_truth_3_group', 'ground_truth_4']
df.drop(columns=columns_to_remove, inplace=True)

# Check if 'ground_truth' column has exactly 52 unique values
unique_clusters = df['ground_truth'].unique()
if len(unique_clusters) != 52:
    print(f"Error: 'ground_truth' column contains {len(unique_clusters)} unique clusters instead of 52.")
else:
    print("The 'ground_truth' column contains exactly 52 unique clusters.")
    
    # Create a mapping from unique string values to numbers
    ground_truth_mapping = {value: i for i, value in enumerate(unique_clusters)}

    # Replace the ground_truth column with the mapped numbers
    df['ground_truth'] = df['ground_truth'].map(ground_truth_mapping)

    # Save the DataFrame to a new file as a well-formatted CSV
    df.to_csv(output_file_path, index=False)

    print(f"Processed file has been saved to: {output_file_path}")
