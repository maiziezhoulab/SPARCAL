#!/bin/bash

# Function to display usage
usage() {
  echo "Usage: $0 -i <original_ini_file> -o <output_directory> -s <SPATIALID> -r <RAD_LIM> -n <NEIGHBOR_LIM> -c <NUM_CLUSTER> -t <SNV_TOOL> -f <FILTER>"
  exit 1
}

# Parse input arguments
while getopts "i:o:s:r:n:c:t:f:d:" opt; do
  case $opt in
    i) original_ini_file="$OPTARG" ;;
    o) output_directory="$OPTARG" ;;
    s) spatialid="$OPTARG" ;;
    r) rad_lim="$OPTARG" ;;
    n) neighbor_lim="$OPTARG" ;;
    c) num_cluster="$OPTARG" ;;
    t) snv_tool="$OPTARG" ;;
    f) filter="$OPTARG" ;;
    d) threshold="$OPTARG" ;;
    *) usage ;;
  esac
done

# Check if all required arguments are provided
if [ -z "$original_ini_file" ] || [ -z "$output_directory" ] || [ -z "$spatialid" ] || [ -z "$rad_lim" ] || [ -z "$neighbor_lim" ] || [ -z "$num_cluster" ] || [ -z "$snv_tool" ] || [ -z "$filter" ] || [ -z "$threshold" ]; then
  usage
fi

# Define the new config file path
new_config_file="$output_directory/${spatialid}_${snv_tool}_${neighbor_lim}_${filter}.ini"

# Create the output directory if it does not exist
mkdir -p "$output_directory"

# Copy the original ini file to the new configuration file
cp "$original_ini_file" "$new_config_file"

# Function to add or update parameters in the DEFAULT section
add_or_update_param() {
  local param_name="$1"
  local param_value="$2"
  local file="$3"

  # Check if the parameter exists in the DEFAULT section
  if grep -q "^\[DEFAULT\]" "$file" && grep -q "^$param_name" "$file"; then
    # If the parameter exists, update it
    sed -i "/^\[DEFAULT\]/,/^\[/ s/^$param_name *=.*/$param_name = $param_value/" "$file"
  else
    # If the parameter does not exist, add it under the DEFAULT section
    sed -i "/^\[DEFAULT\]/a $param_name = $param_value" "$file"
  fi
}

# Add or update each parameter in the DEFAULT section
add_or_update_param "SPATIALID" "$spatialid" "$new_config_file"
add_or_update_param "RAD_LIM" "$rad_lim" "$new_config_file"
add_or_update_param "NEIGHBOR_LIM" "$neighbor_lim" "$new_config_file"
add_or_update_param "NUM_CLUSTER" "$num_cluster" "$new_config_file"
add_or_update_param "SNV_TOOL" "$snv_tool" "$new_config_file"
add_or_update_param "FILTER" "$filter" "$new_config_file"
add_or_update_param "THRESHOLD" "$threshold" "$new_config_file"

echo "New configuration file created at: $new_config_file"
