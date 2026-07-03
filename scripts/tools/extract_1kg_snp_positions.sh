#!/bin/bash

# Extract SNP positions from 1000 Genomes VCF files for both GRCh38 and hg19
# and store them in easily loadable text files for faster access.

# Output directories
OUTPUT_DIR="/data/maiziezhou_lab/yuqi/snv_calling/data/1kG_positions"
OUTPUT_GRCh38="$OUTPUT_DIR/GRCh38"
OUTPUT_hg19="$OUTPUT_DIR/hg19"

# Create directories
mkdir -p "$OUTPUT_GRCh38"
mkdir -p "$OUTPUT_hg19"

# Source data paths
GRCh38_PATH="/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_GRCh38"
hg19_PATH="/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/1000Genome_hg19"

# Pattern for file names
GRCh38_PATTERN="CCDG_14151_B01_GRM_WGS_2020-08-05_chr{chrom}.filtered.shapeit2-duohmm-phased.vcf.gz"
hg19_PATTERN="hg19_chr{chrom}.vcf.gz"

# Extract SNP positions from a GRCh38 VCF file
extract_grch38_snps() {
    local chrom="$1"
    local input_file="${GRCh38_PATH}/$(echo $GRCh38_PATTERN | sed "s/{chrom}/$chrom/g")"
    local output_file="${OUTPUT_GRCh38}/chr${chrom}_snps_keys.txt"
    
    echo "Processing GRCh38 chromosome ${chrom}..."
    
    if [ ! -f "$input_file" ]; then
        echo "Error: VCF file not found: $input_file"
        return 1
    fi
    
    # Extract chromosome and position, then format as "chr{chrom}_{pos}" to match your validation script's format
    bcftools view -v snps "$input_file" | grep -v "^#" | awk -v chr="chr$chrom" '{print chr"_"$2}' > "$output_file"
    
    echo "Extracted $(wc -l < "$output_file") SNP positions from GRCh38 chromosome ${chrom} to $output_file"
}

# Extract SNP positions from an hg19 VCF file
extract_hg19_snps() {
    local chrom="$1"
    local input_file="${hg19_PATH}/$(echo $hg19_PATTERN | sed "s/{chrom}/$chrom/g")"
    local output_file="${OUTPUT_hg19}/chr${chrom}_snps_keys.txt"
    
    echo "Processing hg19 chromosome ${chrom}..."
    
    if [ ! -f "$input_file" ]; then
        echo "Error: VCF file not found: $input_file"
        return 1
    fi
    
    # Extract chromosome and position, already formatted with "chr" prefix in hg19 files
    bcftools view -v snps "$input_file" | grep -v "^#" | awk -v chr="chr$chrom" '{print chr"_"$2}' > "$output_file"
    
    echo "Extracted $(wc -l < "$output_file") SNP positions from hg19 chromosome ${chrom} to $output_file"
}

# Process all chromosomes
process_all_chromosomes() {
    for i in {1..22}; do
        # Process GRCh38
        extract_grch38_snps "$i"
        
        # Process hg19
        extract_hg19_snps "$i"
    done
    
    # Also process X and Y chromosomes if available
    for chr in X Y; do
        if [ -f "${GRCh38_PATH}/$(echo $GRCh38_PATTERN | sed "s/{chrom}/$chr/g")" ]; then
            extract_grch38_snps "$chr"
        fi
        if [ -f "${hg19_PATH}/$(echo $hg19_PATTERN | sed "s/{chrom}/$chr/g")" ]; then
            extract_hg19_snps "$chr"
        fi
    done
}

# Process just one chromosome
process_single_chromosome() {
    local genome="$1"
    local chrom="$2"
    
    if [ "$genome" == "GRCh38" ]; then
        extract_grch38_snps "$chrom"
    elif [ "$genome" == "hg19" ]; then
        extract_hg19_snps "$chrom"
    else
        echo "Invalid genome: $genome. Use GRCh38 or hg19."
        exit 1
    fi
}

# Check if we're running for all chromosomes or a specific one
if [ $# -eq 0 ]; then
    echo "Processing all chromosomes for both GRCh38 and hg19..."
    process_all_chromosomes
elif [ $# -eq 2 ]; then
    echo "Processing chromosome $2 for $1..."
    process_single_chromosome "$1" "$2"
else
    echo "Usage: $0 [genome chrom]"
    echo "  With no arguments: process all chromosomes for both genomes"
    echo "  With arguments: specify genome (GRCh38 or hg19) and chromosome number"
    echo "  Example: $0 GRCh38 1"
    exit 1
fi

# Create combined files with all positions
if [ $# -eq 0 ] || ([ $# -eq 2 ] && [ "$1" == "GRCh38" ]); then
    echo "Creating combined GRCh38 SNP positions file..."
    cat "${OUTPUT_GRCh38}"/chr*_snps_keys.txt > "${OUTPUT_GRCh38}/all_snps_keys.txt"
    echo "Total GRCh38 SNP positions: $(wc -l < "${OUTPUT_GRCh38}/all_snps_keys.txt")"
fi

if [ $# -eq 0 ] || ([ $# -eq 2 ] && [ "$1" == "hg19" ]); then
    echo "Creating combined hg19 SNP positions file..."
    cat "${OUTPUT_hg19}"/chr*_snps_keys.txt > "${OUTPUT_hg19}/all_snps_keys.txt"
    echo "Total hg19 SNP positions: $(wc -l < "${OUTPUT_hg19}/all_snps_keys.txt")"
fi

echo "Done!"
