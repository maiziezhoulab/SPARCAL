import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TISSUE_POS_PATH = '/data/maiziezhou_lab/Datasets/ST_datasets/10x_BC_6.5mm_Visium_CytAssist_FFPE/spatial/tissue_positions.csv'
# TISSUE_POS_PATH = '/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/151507/tissue_positions.csv'
# OUTPUT_PATH = '/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/data/DLPFC_spatialLIBD/151507/preprocessing/neighbor_rad_relation.csv'

import configparser

config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
config.read('/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/configs/Mouse_Brain/mb.ini')
# config.read('/data/maiziezhou_lab/hanliu/projects/ST-SNV-Calling/configs/spatialLIBD/151674.ini')

TISSUE_POS_PATH = config['INPUT']['TISSUE_POS_PATH']
OUTPUT_PATH = config['PROCESS']['NEIGHBOR_RELATION']



def cal_euclidian_pixel(rad_lim, neighbor_lim, path):
    column_names = ['barcode','in_tissue','array_row','array_col','pxl_row_in_fullres','pxl_col_in_fullres']
    data = pd.read_csv(path, names = column_names)
    names = data['barcode'].values
    col = data['pxl_col_in_fullres'].values
    row = data['pxl_row_in_fullres'].values
    flag = data['in_tissue'].values
    # create data phrame for the final output
    edges = 0
    anti_edges = 0

    all_points = np.column_stack((row, col))
    df_names = ['Center', 'Neighbors', 'Distance']
    df = pd.DataFrame(columns = df_names)
    # initialize parameters to calculate number of edge cells
    num_neighbors = 0
    num = 0

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

            # print(names[i], neighbors)

            neighbor_dist = [dist[j] for j in idx]
            # if the point does not contain as much neighbors as we hoped, we say it is an "edge" cell
            num_neighbors += len(neighbors)
            if(len(neighbors) < neighbor_lim):
                edges += 1

            # optional testing, if need to test radius
            if(len(neighbors) > neighbor_lim):
                anti_edges += 1
                # print('wrong')

            new_data = {'Center': names[i], 'Neighbors': neighbors, 'Distance': neighbor_dist}
            new_data = pd.DataFrame(new_data)
            df = pd.concat([df, new_data], ignore_index = True)
    # final summary
    avg_neigh = num_neighbors / num
    # print("Total of " + str(num) + ' center cells. Detected ' + str(edges) + ' edges.')
    # print('Average number of neighbors per cell is ' + str(avg_neigh) + ' neighbors.')
    # df.to_csv('Neighbors_pixels_revised_test.csv', index = False)
    return [edges, anti_edges, avg_neigh]

def distance_numNeighbors():
    path = TISSUE_POS_PATH
    rad_lims = list(range(150, 501, 50))
    neighbor_lims = list(range(6, 20, 1))
    df = pd.DataFrame(index=neighbor_lims, columns=rad_lims)
    neighbor_lim = 5
    for rad_lim in rad_lims:
        for neighbor_lim in neighbor_lims:
            print(neighbor_lim, rad_lim)
            df.at[neighbor_lim, rad_lim] = cal_euclidian_pixel(rad_lim, neighbor_lim, path)
            df.to_csv(OUTPUT_PATH)
    
distance_numNeighbors()
# cal_euclidian_pixel(620, 15, TISSUE_POS_PATH)

