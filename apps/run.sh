#!/bin/bash

# Variables (modify these as needed)
appdir=/data/maiziezhou_lab/yuqi/snv_calling/apps
SAMTOOLS="$appdir/samtools"               # Path to samtools binary (or just 'samtools' if in PATH)
BCFTOOLS="$appdir/bcftools"               # Path to bcftools binary (or just 'bcftools' if in PATH)
BGZIP="$appdir/bgzip"                     # Path to bgzip binary (or just 'bgzip' if in PATH)
BAM_FILTER="./bamlist"       # List of BAM files (one per line)
REFERENCE="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"  # Reference genome path
OUT_PREFIX="CATATTATTTGCCCTA"          # Region or job identifier
OUT_DIR="out_mapq0"        # Output directory for VCF files
export PATH="/data/maiziezhou_lab/yuqi/snv_calling/apps:$PATH"
export LD_LIBRARY_PATH="/data/maiziezhou_lab/yuqi/snv_calling/apps:$LD_LIBRARY_PATH"
# echo PATH=$PATH
# Ensure the output directory exists
mkdir -p "${OUT_DIR}/germline"

# Command construction
CMD="${SAMTOOLS} mpileup -b ${BAM_FILTER} -f ${REFERENCE} -q 0 -Q 0 --incl-flags 0 --excl-flags 0 -t DP -d 10000000 -v"
CMD+=" | ${BCFTOOLS} view"
CMD+=" | ${BCFTOOLS} filter -e 'REF !~ \"^[ATGC]$\"'"
CMD+=" | ${BCFTOOLS} norm -m-both -f ${REFERENCE}"
CMD+=" | grep -v '<X>\|INDEL'"
CMD+=" > ${OUT_DIR}/germline/${OUT_PREFIX}.gl.vcf"

# Execute the command
echo "Running: ${CMD}"
eval "${CMD}"

