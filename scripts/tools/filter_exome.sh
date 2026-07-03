#!/bin/bash

# Script to filter VCF file by BED regions and index the result
# Usage: ./filter_vcf.sh <input_vcf> [bed_file]

# Default parameters
DEFAULT_BED="/data/maiziezhou_lab/Softwares/refdata-hg19-2.1.0/regions/TruSeq_Exome_TargetedRegions_v1.2_hg19.bed"

# Check if at least input VCF is provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <input_vcf> [bed_file]"
    echo "Example: $0 input.vcf.gz"
    echo "Example: $0 input.vcf.gz custom_regions.bed"
    echo ""
    echo "Output will be automatically named based on input VCF"
    echo "Example: input.vcf.gz -> input.exome.vcf.gz"
    exit 1
fi

# Assign arguments to variables with defaults
INPUT_VCF="$1"
BED_FILE="${2:-$DEFAULT_BED}"

# Auto-generate output filename based on input VCF
# Remove .vcf.gz extension, add .exome.vcf.gz
BASENAME=$(basename "$INPUT_VCF" .vcf.gz)
DIRNAME=$(dirname "$INPUT_VCF")
OUTPUT_VCF="${DIRNAME}/${BASENAME}.exome.vcf.gz"

# Check if input files exist
if [ ! -f "$INPUT_VCF" ]; then
    echo "Error: Input VCF file '$INPUT_VCF' not found!"
    exit 1
fi

if [ ! -f "$BED_FILE" ]; then
    echo "Error: BED file '$BED_FILE' not found!"
    exit 1
fi

# Check if output directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_VCF")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Error: Output directory '$OUTPUT_DIR' does not exist!"
    exit 1
fi

echo "Starting VCF filtering..."
echo "Input VCF: $INPUT_VCF"
echo "BED file: $BED_FILE"
echo "Output VCF: $OUTPUT_VCF"

# Filter VCF by BED regions
bcftools view -R "$BED_FILE" \
  -O z -o "$OUTPUT_VCF" \
  "$INPUT_VCF"

# Check if bcftools command was successful
if [ $? -ne 0 ]; then
    echo "Error: bcftools view failed!"
    exit 1
fi

echo "Filtering complete. Indexing output..."

# Index the output VCF
bcftools index -t "$OUTPUT_VCF"

# Check if indexing was successful
if [ $? -ne 0 ]; then
    echo "Error: bcftools index failed!"
    exit 1
fi

echo "Done! Filtered and indexed VCF created: $OUTPUT_VCF"
echo "Index file created: ${OUTPUT_VCF}.tbi"