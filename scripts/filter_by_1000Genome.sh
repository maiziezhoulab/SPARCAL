#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 -i <id_list> -m <method> -p <python_script> -r <reference_base_path> -v <vcf_base_path> -o <output_base_path>"
  echo "  -i  Comma-separated list of IDs (e.g., '151507,151508,151509')"
  echo "  -m  SNV calling method (e.g., 'gatk' or 'self')"
  echo "  -p  Path to the Python filtering script (e.g., 'filter_by_1000Genome.py')"
  echo "  -r  Base path for reference files (e.g., '/data/.../results')"
  echo "  -v  Base path for VCF directories (e.g., '/data/.../output_VCFs/unfiltered')"
  echo "  -o  Base path for output directories (e.g., '/data/.../output_VCFs/filtered_by_1000Genome')"
  exit 1
}

# Parse input arguments
while getopts "i:m:p:r:v:o:" opt; do
  case $opt in
    i) id_list="$OPTARG" ;;
    m) method="$OPTARG" ;;
    p) python_script="$OPTARG" ;;
    r) reference_base_path="$OPTARG" ;;
    v) vcf_base_path="$OPTARG" ;;
    o) output_base_path="$OPTARG" ;;
    *) usage ;;
  esac
done

# Check if all required arguments are provided
if [ -z "$id_list" ] || [ -z "$method" ] || [ -z "$python_script" ] || [ -z "$reference_base_path" ] || [ -z "$vcf_base_path" ] || [ -z "$output_base_path" ]; then
  usage
fi

# Convert comma-separated ID list to an array
IFS=',' read -r -a IDS <<< "$id_list"

# Function to process a single ID
process_id() {
    local ID=$1
    local METHOD=$2
    local PYTHON_SCRIPT=$3
    local VCF_BASE_PATH=$4
    local OUTPUT_BASE_PATH=$5

    echo "${ID} starts filtering."

    REFERENCE_BASE_PATH="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/{ID}/${METHOD}/results/filtered_by_1000Genome" \
    # Define the paths based on the current ID and input base paths
    REFERENCE_TXT="${REFERENCE_BASE_PATH}/${ID}_${METHOD}_ref_on.txt"
    VCF_DIRECTORY="${VCF_BASE_PATH}/${ID}/0"
    OUTPUT_DIRECTORY="${OUTPUT_BASE_PATH}/${ID}/0"

    # Create the output directory if it does not exist
    mkdir -p ${OUTPUT_DIRECTORY}

    # Run the Python script
    python3 ${PYTHON_SCRIPT}/filter_by_1000Genome.py ${REFERENCE_TXT} ${VCF_DIRECTORY} ${OUTPUT_DIRECTORY}

    echo "Filtering 1000 Genome for ID ${ID} completed."
}

# Export the function so it's available to parallel
export -f process_id

# Use GNU Parallel to run the process_id function for each ID in parallel
parallel process_id ::: "${IDS[@]}" ::: "$method" ::: "$python_script" ::: "$vcf_base_path" ::: "$output_base_path"

echo "All IDs processed."

