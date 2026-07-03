#!/bin/bash

#SBATCH --job-name=genotype_shifting_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/genotype_shifting_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/genotype_shifting_OVAR_P5.err
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

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=P5_sr13

echo "Processing OVAR_P5 section: ${SECTION_ID} | ${QUALITY_FILTER}"

python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset OVAR_P5 \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER}

echo "End time: $(date)"
