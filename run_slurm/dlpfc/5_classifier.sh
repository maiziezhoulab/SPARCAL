#!/bin/bash

#SBATCH --job-name=nn_classifier_DLPFC
#SBATCH --output=slurm_output/DLPFC/baseQ0mapQ0/nn_classifier_DLPFC.out
#SBATCH --error=slurm_output/DLPFC/baseQ0mapQ0/nn_classifier_DLPFC.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
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

# Process sections 151507-151510
echo "Processing first group of sections: 151507-151510"
for SECTION_ID in {151507..151510}; do
    echo "==============================================="
    echo "Processing section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Run the neural network classifier for this section
    python scripts/4_classifier/run_supplimentary_models.py \
        --dataset DLPFC \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --model-type neural_network \
        --max-training-samples 90000
    
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

# Process sections 151669-151672
echo "Processing second group of sections: 151669-151672"
for SECTION_ID in {151669..151672}; do
    echo "==============================================="
    echo "Processing section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    # Run the neural network classifier for this section
    python scripts/4_classifier/run_supplimentary_models.py \
        --dataset DLPFC \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --model-type neural_network \
        --max-training-samples 90000
    
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

# Process sections 151673-151676
echo "Processing third group of sections: 151673-151676"
for SECTION_ID in {151673..151676}; do
    echo "==============================================="
    echo "Processing section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Run the neural network classifier for this section
    python scripts/4_classifier/run_supplimentary_models.py \
        --dataset DLPFC \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --model-type neural_network \
        --max-training-samples 90000
    
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

echo "All sections processed"
echo "End time: $(date)"