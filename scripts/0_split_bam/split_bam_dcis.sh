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
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --array=1-2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

# Define paths
BASE_PATH="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium"
INPUT_BAM="${BASE_PATH}/DCIS${SECTION_ID}/spaceranger_align_DCIS${SECTION_ID}_hg38/DCIS${SECTION_ID}_output/outs/possorted_genome_bam.bam"
OUTPUT_DIR="${BASE_PATH}/DCIS${SECTION_ID}/spaceranger_align_DCIS${SECTION_ID}_hg38/DCIS${SECTION_ID}_output/outs/split_BAM"
SAMTOOLS="/data/maiziezhou_lab/yuqi/snv_calling/apps/samtools"

echo "Processing DCIS${SECTION_ID}"
echo "Input BAM: ${INPUT_BAM}"
echo "Output directory: ${OUTPUT_DIR}"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Check if input BAM exists
if [ ! -f "${INPUT_BAM}" ]; then
    echo "ERROR: Input BAM file not found: ${INPUT_BAM}"
    exit 1
fi

# Split BAM by cell barcode (CB tag)
echo "Splitting BAM file by cell barcode..."
${SAMTOOLS} split -@ 8 -f "${OUTPUT_DIR}/%!.bam" -u "${OUTPUT_DIR}/unknown.bam" ${INPUT_BAM}

# Count the number of split BAM files
NUM_BAMS=$(ls ${OUTPUT_DIR}/*.bam 2>/dev/null | wc -l)
echo "Created ${NUM_BAMS} BAM files"

# Index each split BAM file
echo "Indexing split BAM files..."
for bam in ${OUTPUT_DIR}/*.bam; do
    if [ -f "$bam" ]; then
        ${SAMTOOLS} index $bam
    fi
done

echo "BAM splitting complete for DCIS${SECTION_ID}"
echo "End time: $(date)"