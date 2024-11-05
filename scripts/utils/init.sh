#!/bin/bash

# Define the range of IDs
ids=("151507" "151508" "151509" "151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676")

dir=/data/maiziezhou_lab/hanliu/projects/snv_calling/data/dlpfc
# Loop through each ID and create the required folders
for id in "${ids[@]}"; do
  mkdir -p $dir/$id/output_VCFs
  mkdir -p $dir/$id/processed_data
  mkdir -p $dir/$id/results
done

echo "Folders created successfully."
