#!/bin/bash

#SBATCH --job-name=svm_hetero_DLPFC
#SBATCH --output=slurm_output/DLPFC/baseQ0mapQ0/svm_hetero_DLPFC.out
#SBATCH --error=slurm_output/DLPFC/baseQ0mapQ0/svm_hetero_DLPFC.err
#SBATCH --time=12:00:00
#SBATCH --account=cgw_maizie2
#SBATCH --partition=cgw-maizie2
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

# Run the SVM heterozygous finding analysis for each section
for SECTION_ID in {151507..151510} {151669..151676}; do
    echo "==============================================="
    echo "Processing section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Run the svm_hetero_finding.py script for each section
    python scripts/postprocess/run_svm_hetero_finding.py \
        --dataset DLPFC \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER}
    
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