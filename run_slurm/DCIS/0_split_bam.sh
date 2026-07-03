#!/bin/bash

#SBATCH --job-name=split_bam_DCIS
#SBATCH --output=slurm_output/DCIS/split_bam_DCIS_%a.out
#SBATCH --error=slurm_output/DCIS/split_bam_DCIS_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=400GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --array=1-2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

# Define paths
BASE_PATH="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium"
META_PATH="${BASE_PATH}/meta_data"
INPUT_BAM="${BASE_PATH}/DCIS${SECTION_ID}/spaceranger_align_DCIS${SECTION_ID}_hg38/DCIS${SECTION_ID}_output/outs/possorted_genome_bam.bam"
OUTPUT_DIR="${BASE_PATH}/DCIS${SECTION_ID}/spaceranger_align_DCIS${SECTION_ID}_hg38/DCIS${SECTION_ID}_output/outs/split_BAM"
BARCODE_FILE="${META_PATH}/dcis${SECTION_ID}/filtered_feature_bc_matrix/barcodes.tsv"
SPLIT_SCRIPT="/data/maiziezhou_lab/CanLuo/ST_SNV/Scripts/split_bam.py"

echo "Processing DCIS${SECTION_ID}"
echo "Input BAM: ${INPUT_BAM}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Barcode file: ${BARCODE_FILE}"
echo "Split script: ${SPLIT_SCRIPT}"

# Check if input files exist
if [ ! -f "${INPUT_BAM}" ]; then
    echo "ERROR: Input BAM file not found: ${INPUT_BAM}"
    exit 1
fi

if [ ! -f "${BARCODE_FILE}" ]; then
    echo "ERROR: Barcode file not found: ${BARCODE_FILE}"
    exit 1
fi

if [ ! -f "${SPLIT_SCRIPT}" ]; then
    echo "ERROR: Split script not found: ${SPLIT_SCRIPT}"
    exit 1
fi

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Split BAM by cell barcode using CanLuo's script
echo "Splitting BAM file by cell barcode (CB tag)..."
python3 ${SPLIT_SCRIPT} \
    -b ${INPUT_BAM} \
    -o ${OUTPUT_DIR} \
    -t 10 \
    -bc ${BARCODE_FILE}

# Verify split was successful
if [ $? -eq 0 ]; then
    # Count the number of split BAM files
    NUM_BAMS=$(ls ${OUTPUT_DIR}/*.bam 2>/dev/null | wc -l)
    echo "Successfully created ${NUM_BAMS} BAM files"
    
    # Check if index files exist
    NUM_INDEX=$(ls ${OUTPUT_DIR}/*.bai 2>/dev/null | wc -l)
    if [ ${NUM_INDEX} -gt 0 ]; then
        echo "Found ${NUM_INDEX} index files"
    else
        echo "Note: No .bai index files found (may be indexed as .bam.bai)"
        NUM_INDEX=$(ls ${OUTPUT_DIR}/*.bam.bai 2>/dev/null | wc -l)
        echo "Found ${NUM_INDEX} .bam.bai index files"
    fi
    
    echo "BAM splitting complete for DCIS${SECTION_ID}"
else
    echo "ERROR: BAM splitting failed for DCIS${SECTION_ID}"
    exit 1
fi

echo "End time: $(date)"