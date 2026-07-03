#!/bin/bash

#SBATCH --job-name=mpileup_pipeline_DCIS
#SBATCH --output=slurm_output/DCIS/mpileup_pipeline_DCIS_%a.out
#SBATCH --error=slurm_output/DCIS/mpileup_pipeline_DCIS_%a.err
#SBATCH --time=60:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --array=1-2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# Set quality parameters
BASE_QUAL=0
MAP_QUAL=0

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

echo "Processing DCIS section: ${SECTION_ID}"
echo "Base Quality: ${BASE_QUAL}, Mapping Quality: ${MAP_QUAL}"

# Run the pipeline for DCIS
python /data/maiziezhou_lab/leiy4/snv_calling/scripts/1_calling/mpileup_pipeline.py \
    --dataset DCIS \
    --section_id ${SECTION_ID} \
    --base_quality ${BASE_QUAL} \
    --mapping_quality ${MAP_QUAL} \
    --call_mode multi \
    --threads 30 \
    --filter_out_tissue

echo "End time: $(date)"