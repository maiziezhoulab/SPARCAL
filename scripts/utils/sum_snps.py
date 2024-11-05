import pandas as pd
import os 
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

NUM = int(config['DEFAULT']['NEIGHBOR_LIM'])
INPUT_DATA_PATH = config['PROCESS']['VCF_TABLES_PATH']
OUTPUT_DATA_PATH = config['RESULT']['SUM_SNP_PATH']

def get_filenames(directory_path):
    file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    return file_names

def main():
    filenames = get_filenames(INPUT_DATA_PATH)
    num_rows = []

    # Open the output file in write mode to clear existing contents, then close it
    open(OUTPUT_DATA_PATH, "w").close()

    # Process each file and append to the output file
    for filename in filenames:
        try:
            df = pd.read_csv(INPUT_DATA_PATH + '/' + filename)
            num_rows += [len(df)]
            with open(OUTPUT_DATA_PATH, "a") as file:
                file.write(filename.split(".")[0] + ": " + str(len(df)) + "\n")
        except pd.errors.EmptyDataError:
            print(f"The file '{filename}' is empty.")
            num_rows += [0]
            with open(OUTPUT_DATA_PATH, "a") as file:
                file.write(filename.split(".")[0] + ": 0" + "\n")
    return num_rows

num_rows = main()
print(max(num_rows), min(num_rows))