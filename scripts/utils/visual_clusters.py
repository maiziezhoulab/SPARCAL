import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# TBSP = config['DEFAULT']['TBSP']
DATA_SOURCE = config['DEFAULT']['spatialID']
FILTER = config['DEFAULT']['FILTER']
CLUSTER_DATA = config['RESULT']['CLUSTER_PATH']
OUTPUT_PATH = config['RESULT']['VISUAL_PATH']
NUM_CLUSTER = config['DEFAULT']['NUM_CLUSTER']
TIS_POS = config['INPUT']['TISSUE_POS_PATH']
COLOR = 'tab10'  # bright, Paired
ari = '0.1573'
title = f'{DATA_SOURCE}_{FILTER} ARI={ari}'
def make_plot(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('lightgray')
    sns.scatterplot(x=df['pxl_col_in_fullres'], y=df['pxl_row_in_fullres'], hue=df['cluster'], palette=COLOR, s=50, ax=ax)
    
    # Invert the y-axis
    ax.invert_yaxis()

    # Set the title with larger text size
    plt.title(title, fontsize=20)

    # Remove x and y labels
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # Customize legend with larger text size
    legend = ax.legend(title='Cluster', title_fontsize='13', fontsize='13')

    # Save the plot as a PDF
    plt.savefig(OUTPUT_PATH + f'_ARI={ari}.pdf', format='pdf', dpi=1000)

def main():
    # column_names = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
    # df1 = pd.read_csv(TIS_POS, names=column_names)
    df1 = pd.read_csv(TIS_POS)
    df1 = df1[df1['in_tissue'] == 1]
    df2 = pd.read_csv(CLUSTER_DATA)
    merged_df = df1.merge(df2, on='barcode', how='inner')  # 'inner' means we only keep rows with matching barcodes in both dataframes
    print(len(merged_df))
    make_plot(merged_df)

if __name__ == "__main__":
    main()
