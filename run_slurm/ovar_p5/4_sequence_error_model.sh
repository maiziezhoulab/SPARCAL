#!/bin/bash

#SBATCH --job-name=seq_error_model_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/seq_error_model_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/seq_error_model_OVAR_P5.err
#SBATCH --time=2:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=P5_sr13

echo "Processing OVAR_P5 section: ${SECTION_ID} | ${QUALITY_FILTER}"

python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset OVAR_P5 \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER}

echo "End time: $(date)"
