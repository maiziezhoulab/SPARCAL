import pandas as pd

def process_csv(file_path):
    # Define the column names
    column_names = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']

    # Read the CSV file
    try:
        df = pd.read_csv(file_path, header=None)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Check if the first row contains the expected header names
    if df.iloc[0].isin(column_names).all():
        df.columns = df.iloc[0]
        df = df[1:]  # Remove the header row from the data
    else:
        df.columns = column_names

    # Ensure the 'in_tissue' column is of integer type
    df['in_tissue'] = df['in_tissue'].astype(int)

    # Count the number of rows where 'in_tissue' is 1
    in_tissue_count = df[df['in_tissue'] == 1].shape[0]

    # Print the total number of rows and the number of rows in tissue
    total_rows = df.shape[0]
    print(f"Total number of rows: {total_rows}")
    print(f"Number of rows in tissue: {in_tissue_count}")

# Example usage
file_path = '/data/maiziezhou_lab/Datasets/ST_datasets/Mouse_Brain/Mouse_Brain_Anterior/spatial/tissue_positions_list.csv'
process_csv(file_path)
