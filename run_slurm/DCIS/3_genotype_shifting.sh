#!/bin/bash

#SBATCH --job-name=genotype_shifting_DCIS
#SBATCH --output=slurm_output/DCIS/genotype_shifting_DCIS_%a.out
#SBATCH --error=slurm_output/DCIS/genotype_shifting_DCIS_%a.err
#SBATCH --time=4:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --array=1-2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load required modules
source activate snv_caller

# Set quality parameters
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

echo "==============================================="
echo "Processing DCIS section: ${SECTION_ID}"
echo "Quality filter: ${QUALITY_FILTER}"
echo "Start time: $(date)"

# Run the genotype shifting analysis
python /data/maiziezhou_lab/leiy4/snv_calling/scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset DCIS \
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