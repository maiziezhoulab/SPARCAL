#!/bin/bash

# Variables
METHOD=gatk
REMOTE_BASE_PATH="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC"  # Base path on the server
REMOTE_COPY_PATH="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/${METHOD}_filtered_by_1000Genome_by_neighbor_1"  # New directory on the server for copied files

# Array of IDs
ids=("151507" "151508" "151509" "151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676")

# Create a new directory on the server to store all copied files
mkdir -p $REMOTE_COPY_PATH

# Loop through each ID and copy the corresponding files to the new directory on the server
for id in "${ids[@]}"; do
    REMOTE_DIR="${REMOTE_BASE_PATH}/${id}/${METHOD}/results/filtered_by_1000Genome_by_neighbor_1/0"
    
    # Find and copy files that contain the pattern to the new directory on the server
    find $REMOTE_DIR -type f -name '*filtered_by_1000Genome_by_neighbor_1_ARI*' -exec cp {} $REMOTE_COPY_PATH/ \;
    
    # Check if the copy command was successful
    if [ $? -eq 0 ]; then
        echo "Files for ID $id copied successfully to $REMOTE_COPY_PATH"
    else
        echo "File copy for ID $id failed"
    fi
done
