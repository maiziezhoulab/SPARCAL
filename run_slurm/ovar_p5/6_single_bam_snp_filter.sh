#!/bin/bash

#SBATCH --job-name=filter_bams_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/baseQ0mapQ0/filter_single_bams_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/baseQ0mapQ0/filter_single_bams_OVAR_P5.err
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=P5_sr13

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

mkdir -p slurm_output/OVAR_P5/${QUALITY_FILTER}

echo "Processing OVAR_P5 section: ${SECTION_ID} | ${QUALITY_FILTER}"

# NOTE: this script uses --section-id (hyphen), matching the DCIS step-6 runner.
python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset OVAR_P5 \
    --section-id ${SECTION_ID} \
    --quality-filter ${QUALITY_FILTER} \
    --max-workers 30 \
    --classifier neural_network

echo "End time: $(date)"
