import pandas as pd
from sklearn.cluster import KMeans
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

# INPUT_DATA_PATH = config['RESULT']['MODIFIED_MATRIX_PATH']
INPUT_DATA_PATH = config['RESULT']['MATRIX_PATH']
OUTPUT_DATA_PATH = config['RESULT']['CLUSTER_PATH']
NUM_CLUSTER = config['DEFAULT']['NUM_CLUSTER']

def main():
    df = pd.read_pickle(INPUT_DATA_PATH)
    kmeans = KMeans(n_clusters= int(NUM_CLUSTER), init='k-means++', n_init=10, max_iter=300, random_state=0).fit(df)
    df['Cluster'] = kmeans.labels_

    # Create a new dataframe with only original index and cluster labels
    output_df = df.reset_index()[['index', 'Cluster']].rename(columns={'index': 'barcode', 'Cluster': 'cluster'})
    # print(output_df.head(5))
    output_df['barcode'] = [v.split("_")[1] for v in output_df['barcode']]
    # # Save the new dataframe to a CSV file
    output_df.to_csv(OUTPUT_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
