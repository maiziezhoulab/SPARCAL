#!/bin/bash

#SBATCH --job-name=mpileup_pipeline_DLPFC
#SBATCH --output=slurm_output/DLPFC/mpileup_pipeline_DLPFC_sec1.out
#SBATCH --error=slurm_output/DLPFC/mpileup_pipeline_DLPFC_sec1.err
#SBATCH --time=60:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

BASE_QUAL = 0
MAP_QUAL = 0

# Run the pipeline for DLPFC section 151507-151510, 151669-151676 , base quality 0, mapping quality 0
# for SECTION_ID in {151507..151510}; do
#     python scripts/1_calling/mpileup_pipeline.py --dataset DLPFC --section_id ${SECTION_ID} --base_quality ${BASE_QUAL} --mapping_quality ${MAP_QUAL}
# done

# for SECTION_ID in {151669..151672}; do
#     python scripts/1_calling/mpileup_pipeline.py --dataset DLPFC --section_id ${SECTION_ID} --base_quality ${BASE_QUAL} --mapping_quality ${MAP_QUAL}
# done

# for SECTION_ID in {151673..151676}; do
#     python scripts/1_calling/mpileup_pipeline.py --dataset DLPFC --section_id ${SECTION_ID} --base_quality ${BASE_QUAL} --mapping_quality ${MAP_QUAL}
# done

echo "End time: $(date)"