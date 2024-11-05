from sklearn.metrics import adjusted_rand_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process VCF files.')
parser.add_argument('id', type=str, help='ID for processing')
args = parser.parse_args()
# Rest of your code...
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(f'{args.id}.ini')

# ID = 'mouse'
# column_names = ['barcode','in_tissue','array_row','array_col','gold_cluster']
ID = config['DEFAULT']['spatialID']
column_names = ['barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres','gold_cluster']
df1 = pd.read_csv(config['DEFAULT']['GOLD_ClUSTER'],names = column_names)
df2 = pd.read_csv(config['RESULT']['CLUSTER_PATH'])

df1 = df1.astype(str)
df2 = df2.astype(str)
# print(df1.head())
# print(df2.head())


# Merge the two dataframes on the barcode column
merged_df = pd.merge(df1, df2, on="barcode")
# print(merged_df)
# Extract the cluster labels from the merged dataframe
labels1 = merged_df["gold_cluster"]
labels2 = merged_df["cluster"]

# Calculate the Adjusted Rand Index (ARI)
ari = adjusted_rand_score(labels1, labels2)

print(f"Adjusted Rand Index (ARI): {ID} {ari}")
