#!/bin/bash

#SBATCH --job-name=beagle_pipeline_DCIS
#SBATCH --output=slurm_output/DCIS/beagle_pipeline_DCIS_%a.out
#SBATCH --error=slurm_output/DCIS/beagle_pipeline_DCIS_%a.err
#SBATCH --time=5:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=500GB
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

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

echo "Processing DCIS section: ${SECTION_ID}"
echo "Quality filter: baseQ${BASEQ}mapQ${MAPQ}"

# Run the Beagle pipeline for DCIS
python /data/maiziezhou_lab/leiy4/snv_calling/scripts/2_beagle_filtering/run_beagle.py \
    --dataset DCIS \
    --section_id ${SECTION_ID} \
    --quality-filter baseQ${BASEQ}mapQ${MAPQ} \
    --threads 30 \
    --memory 200g

echo "End time: $(date)"

# The command for DCIS section 1 with baseQ0mapQ0 filter:
# python /data/maiziezhou_lab/leiy4/snv_calling/scripts/2_beagle_filtering/run_beagle.py --dataset DCIS --section_id 1 --quality-filter baseQ0mapQ0 --threads 30 --memory 200g