#!/bin/bash

#SBATCH --job-name=seq_error_model_DLPFC
#SBATCH --output=slurm_output/DLPFC/baseQ0mapQ0/seq_error_model_DLPFC.out
#SBATCH --error=slurm_output/DLPFC/baseQ0mapQ0/seq_error_model_DLPFC.err
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# Set base quality and mapping quality
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Make sure the output directory exists
mkdir -p slurm_output/DLPFC/${QUALITY_FILTER}

for SECTION_ID in  {151507..151510}; do
    echo "==============================================="
    echo "Processing section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Run the sequence_error_model.py script for each section
    python scripts/3_classifier_prep/run_sequence_error_model.py \
        --dataset DLPFC \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER}
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "Section ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to process section ${SECTION_ID}"
    fi
    
    echo "End time for section ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""
done

# Run the sequence error model analysis for each section
# for SECTION_ID in  {151669..151676}; do
#     echo "==============================================="
#     echo "Processing section: ${SECTION_ID}"
#     echo "Quality filter: ${QUALITY_FILTER}"
#     echo "Start time: $(date)"
    
#     # Run the sequence_error_model.py script for each section
#     python scripts/3_classifier_prep/run_sequence_error_model.py \
#         --dataset DLPFC \
#         --section_id ${SECTION_ID} \
#         --quality_filter ${QUALITY_FILTER}
    
#     # Check if the script ran successfully
#     if [ $? -eq 0 ]; then
#         echo "Section ${SECTION_ID} completed successfully"
#     else
#         echo "ERROR: Failed to process section ${SECTION_ID}"
#     fi
    
#     echo "End time for section ${SECTION_ID}: $(date)"
#     echo "==============================================="
#     echo ""
# done

echo "All sections processed"
echo "End time: $(date)"