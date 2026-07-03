#!/bin/bash

#SBATCH --job-name=mpileup_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/mpileup_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/mpileup_OVAR_P5.err
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

source activate snv_caller

BASE_QUAL=0
MAP_QUAL=0
SECTION_ID=P5_sr13

echo "Processing OVAR_P5 section: ${SECTION_ID}"
echo "Base Quality: ${BASE_QUAL}, Mapping Quality: ${MAP_QUAL}"

python scripts/1_calling/mpileup_pipeline.py \
    --dataset OVAR_P5 \
    --section_id ${SECTION_ID} \
    --base_quality ${BASE_QUAL} \
    --mapping_quality ${MAP_QUAL} \
    --call_mode multi \
    --threads 30 \
    --filter_out_tissue

echo "End time: $(date)"
