'''
    This is the main file of the ST-SNV calling Pipeline
'''

import os
import numpy as np
import pandas as pd
import subprocess
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import configparser
import argparse
import sys

parser = argparse.ArgumentParser(description="A sample script to demonstrate command line arguments.")
parser.add_argument('-c', '--config', help='config', required=True)
args = parser.parse_args()
print(args.config)
if not os.path.exists(args.config):
    print("Error: The specified config file does not exist at", args.config)
    sys.exit(1)
else:
    print("Read in the configs at", args.config)

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read(args.config)
# reference genome.fa
REFERENCE = config['DEFAULT']['REFERENCE']
SH_PATH = config['DEFAULT']['SH_PATH']
# variablle settings for radius and the number of neighbors
NEIGHBOR_LIM = int(config['DEFAULT']['NEIGHBOR_LIM'])
RAD_LIM = int(config['DEFAULT']['RAD_LIM'])
LOCAL_DATA_PATH = config['DEFAULT']['LOCAL_DATA_PATH']
# inputs: bams, tissue positions
BAM_PATH = config['INPUT']['BAM_PATH']
TISSUE_POS_PATH = config['INPUT']['TISSUE_POS_PATH']
# outputs: pixel_file, merged_bams, vcfs
NEIGHBOR_PIXEL_PATH = config['PROCESS']['NEIGHBOR_PIXEL_PATH']
MERGED_BAM_PATH = config['PROCESS']['MERGED_BAM_PATH']
OUT_VCF_PATH = config['PROCESS']['OUT_VCF_PATH']

def cal_euclidian_pixel(rad_lim, neighbor_lim, path):
    column_names = ['barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']
    data = pd.read_csv(path, names = column_names)
    # data = pd.read_csv(path)
    names = data['barcode'].values
    col = data['pxl_col_in_fullres'].values
    row = data['pxl_row_in_fullres'].values
    flag = data['in_tissue'].values
    # create data phrame for the final output
    edges = 0
    all_points = np.column_stack((row, col))
    df_names = ['Center', 'Neighbors', 'Distance']
    df = pd.DataFrame(columns = df_names)
    # initialize parameters to calculate number of edge cells
    num_neighbors = 0
    num = 0

    with open(LOCAL_DATA_PATH + '/neighbors.txt', 'a') as f:
        for i in range(len(names)):
        # only enter calculation if the cell is in tissue
            if(flag[i] == 1):
                num += 1
                point = [row[i], col[i]]
                # calculate distances from all points
                dist = np.linalg.norm(all_points - point, axis = 1)
                # find all indices where the dist from current point to that cell is within the set limits
                idx = np.where((0.0 < dist) & (dist < rad_lim) & (flag == 1))[0]
                neighbors = [names[j] for j in idx]
                neighbor_dist = [dist[j] for j in idx]
                # if the point does not contain as much neighbors as we hoped, we say it is an "edge" cell
                num_neighbors += len(neighbors)
                if(len(neighbors) < neighbor_lim):
                    edges += 1
                
                # optional testing, if need to test radius
                if(len(neighbors) > neighbor_lim):
                    print('wrong')

                new_data = {'Center': names[i], 'Neighbors': neighbors, 'Distance': neighbor_dist}
                new_data = pd.DataFrame(new_data)
                df = pd.concat([df, new_data], ignore_index = True)
                # print(names[i], ":", neighbors)
                f.write(f"{names[i]} : {neighbors}\n")
        # final summary
        avg_neigh = num_neighbors / num
        print("Total of " + str(num) + ' center cells. Detected ' + str(edges) + ' edges.')
        print('Average number of neighbors per cell is ' + str(avg_neigh) + ' neighbors.')
        df.to_csv(NEIGHBOR_PIXEL_PATH, index = False)

def merge_bam_files(bam_files, prev):
    name = MERGED_BAM_PATH + '/' + prev + '_merge.bam'
    # merge BAM files
    merge_command = ['samtools', 'merge', '-o', name] + bam_files
    # print(merge_command)
    subprocess.run(merge_command)
    os.chmod(name, 0o700)
    # index merged BAM file
    index_command = ['samtools', 'index', name]
    subprocess.run(index_command)

def merge_bam(neighbor_fileName, path):
    data = pd.read_csv(neighbor_fileName)
    center = data['Center'].values
    neighbors = data['Neighbors'].values

    bam_files_map = {}
    for val, neigh in zip(center, neighbors):
        if val not in bam_files_map:
            bam_files_map[val] = []
        bam_files_map[val].append(path + '/' + neigh + '.bam')

    with ThreadPoolExecutor(max_workers=30) as executor:
        for center, bam_files in bam_files_map.items():
            executor.submit(merge_bam_files, bam_files, center)

def sort_bams(directory):
    # Get list of all files in the directory
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    
    # Sort files by size
    sorted_files = sorted(files, key=lambda x: os.path.getsize(os.path.join(directory, x)))
    sorted_files = [f for f in sorted_files if f.endswith('.bam')]
    
    return sorted_files

def process_bam_file(bam_file):
    command = f'sh {SH_PATH}/bam2vcf.sh {MERGED_BAM_PATH}/{bam_file} {OUT_VCF_PATH} {REFERENCE}'
    print(command)
    subprocess.run(command, shell=True, check=True)

def main():

    # # calculate the distance
    cal_euclidian_pixel(RAD_LIM, NEIGHBOR_LIM, TISSUE_POS_PATH)
    
    # # merge the bams
    # merge_bam(NEIGHBOR_PIXEL_PATH, BAM_PATH)

    # sorted_bam_files = sort_bams(MERGED_BAM_PATH)
    # # convert bam to vcf
    # MAX_THREADS = 30 
    # with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
    #     executor.map(process_bam_file, sorted_bam_files)

if __name__ == "__main__":
    main()


    