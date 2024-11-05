#!/bin/bash

# METHOD = 'self'
# Directories
INPUT_DIR="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/self_filtered_by_1000Genome_by_neighbor_1"  # Directory containing PDF files
OUTPUT_DIR="/data/maiziezhou_lab/hanliu/projects/snv_call/data/DLPFC/self_filtered_by_1000Genome_by_neighbor_1_png"  # Directory to save PNG files

# Check if ImageMagick's convert command is available
if ! command -v convert &> /dev/null
then
    echo "ImageMagick's 'convert' command not found. Please install ImageMagick first."
    exit 1
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Input directory $INPUT_DIR does not exist. Please create it and place your PDF files there."
    exit 1
fi

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all PDF files in the input directory
for pdf_file in "$INPUT_DIR"/*.pdf; do
    # Get the base name of the file without the extension
    base_name=$(basename "$pdf_file" .pdf)

    # Convert PDF to PNG
    convert -density 300 "$pdf_file" "$OUTPUT_DIR/${base_name}.png"

    # Check if the conversion was successful
    if [ $? -eq 0 ]; then
        echo "Converted $pdf_file to $OUTPUT_DIR/${base_name}.png"
    else
        echo "Failed to convert $pdf_file"
    fi
done
