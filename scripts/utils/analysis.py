import matplotlib.pyplot as plt
import configparser
import argparse
import pandas as pd

# Set up argument parser
parser = argparse.ArgumentParser(description='Process VCF files.')
parser.add_argument('id', type=str, help='ID for processing')
args = parser.parse_args()

# Use the provided ID
config_file_path = f'{args.id}.ini'
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(config_file_path)

NUM = int(config['DEFAULT']['NEIGHBOR_LIM'])
SUM_SNP_PATH = config['RESULT']['SUM_SNP_PATH']
OUTPUT_FIG_PATH = config['RESULT']['histogram_PATH']
UNION_SET_PATH = config['RESULT']['UNION_SET_PATH']

# Initialize an empty list to store numbers
numbers = []

# Open the file in read mode
with open(SUM_SNP_PATH, "r") as file:
    for line in file:
        # Split the line at the colon and extract the second part
        num = int(line.split(":")[1].strip())
        numbers.append(num)

non_zeros = [v for v in numbers if v > 0]
print(len(non_zeros))

print(f"Analysis Result for ID {args.id}")
print("total number of spots: ", len(numbers))
print("the number of 0's: ", numbers.count(0))
print("max: ", max(numbers))
print("min: ", min(non_zeros))
print("average: ", sum(non_zeros)/len(non_zeros))

df = pd.read_csv(UNION_SET_PATH)
print(f"Union Set size: {len(df)}")

# Create a histogram
plt.hist(non_zeros, bins=20, edgecolor='black', alpha=0.7)

# Add title and labels
plt.title("Histogram of snps (" + str(NUM) +") neighbors")
plt.xlabel("the number of snps")
plt.ylabel("number")

# Show the plot
plt.savefig(OUTPUT_FIG_PATH, dpi=1000, bbox_inches='tight')