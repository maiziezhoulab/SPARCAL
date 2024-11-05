#!/bin/bash

# Ensure two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <bam_dir> <vcf_dir>"
    exit 1
fi

# Input arguments
BAM_DIR="$1"
VCF_DIR="$2"

# Define other variables
REFERENCE_SEQ="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BEDFILE="/data/maiziezhou_lab/hanliu/projects/snv_call/data/reference/GRCh38.bed"
HEADER="/data/maiziezhou_lab/hanliu/projects/snv_call/data/reference/header.txt"
SNV_CALL_PATH="/data/maiziezhou_lab/hanliu/projects/snv_call"  # Add the path to the directory containing snv_call.py

# Ensure the output directory exists
mkdir -p "$VCF_DIR"

# Function to process each BAM file
process_bam() {
    local BAMFILE="$1"
    local BASENAME=$(basename "$BAMFILE" .bam)
    local OUTPUT_PATH="$VCF_DIR/${BASENAME}.vcf"

    # Write the header to the output VCF
    cat "$HEADER" > "$OUTPUT_PATH"

    # Loop through chromosomes 1-22, X, Y
    for CHROMOSOME in {1..22} X Y; do
        echo "Processing $BAMFILE chromosome: chr$CHROMOSOME"
        python "$SNV_CALL_PATH/snv_call.py" --reference_seq "$REFERENCE_SEQ" \
                                            --chromosome "$CHROMOSOME" \
                                            --bamfile "$BAMFILE" \
                                            --bedfile "$BEDFILE" \
                                            --header "$HEADER" \
                                            --out "$OUTPUT_PATH"
        echo "Completed chromosome: chr$CHROMOSOME for $BAMFILE"
    done
}

# Export the function so it can be used by parallel
export -f process_bam

# Get a list of all BAM files
BAMFILES=("$BAM_DIR"/*.bam)

# Define the batch size
BATCH_SIZE=30

# Loop through the BAM files in batches
for ((i=0; i<${#BAMFILES[@]}; i+=BATCH_SIZE)); do
    for ((j=i; j<i+BATCH_SIZE && j<${#BAMFILES[@]}; j++)); do
        process_bam "${BAMFILES[j]}" &
    done
    wait
done

echo "All BAM files have been processed."
