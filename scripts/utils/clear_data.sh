#!/bin/bash

# Define the base directory path
base_dir="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC"

# Define the IDs to process
ids=("151510" "151669" "151670" "151671" "151672" "151673" "151674" "151675" "151676")

# Loop over each ID
for id in "${ids[@]}"; do
    # Define paths for processed_vcf_tables and vcf_tables
    processed_vcf_path="${base_dir}/${id}/self/processed_data/processed_vcf_tables/filtered_by_1000Genome/0/"
    vcf_tables_path="${base_dir}/${id}/self/processed_data/vcf_tables/filtered_by_1000Genome/0/"

    # Remove and recreate the processed_vcf_tables directory
    rm -r $processed_vcf_path
    mkdir -p $processed_vcf_path

    # Remove and recreate the vcf_tables directory
    rm -r $vcf_tables_path
    mkdir -p $vcf_tables_path

    echo "Processed for ID: ${id}"
done

echo "All specified IDs have been processed."
