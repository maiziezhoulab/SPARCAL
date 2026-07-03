#!/bin/bash
#SBATCH --job-name=sparcal_filter
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/%x_%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/%x_%A_%a.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1-22

#================================================================
# SPARCAL Spatial Variant Classifier - SLURM Submission Script
# Classifies variants using CNA-aware spatial patterns
#================================================================

# Activate conda environment
source ~/.bashrc
conda activate snv_caller

# Set dataset and section
DATASET="P6_tumor"  # Change this: P4_TUMOR, P6_TUMOR, DCIS, etc.
SECTION="1"    # Change this: P4_sec1, P6_sec1, DCIS1, etc.
QUALITY_FILTER="baseQ0mapQ0"

# Base directories
BASE_DIR="/data/maiziezhou_lab/leiy4/snv_calling"
SCRIPT_DIR="${BASE_DIR}/scripts/6_sparcal_filter" 
CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"

# Get chromosome from array task ID
CHROMS=(chr1 chr2 chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22)
CHROM=${CHROMS[$SLURM_ARRAY_TASK_ID-1]}

echo "=========================================="
echo "SPARCAL Spatial Filter - Chromosome ${CHROM}"
echo "=========================================="
echo "Dataset: ${DATASET}"
echo "Section: ${SECTION}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "=========================================="

# Run SPARCAL classifier
python ${SCRIPT_DIR}/run_sparcal_classifier.py \
    --dataset ${DATASET} \
    --section ${SECTION} \
    --quality_filter ${QUALITY_FILTER} \
    --chromosome ${CHROM} \
    --base_dir ${BASE_DIR} \
    --calicost_base ${CALICOST_BASE}

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "SPARCAL completed successfully for ${CHROM}"
    echo "End time: $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: SPARCAL failed for ${CHROM}"
    echo "End time: $(date)"
    echo "=========================================="
    exit 1
fi