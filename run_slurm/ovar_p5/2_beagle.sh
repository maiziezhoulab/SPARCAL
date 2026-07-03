#!/bin/bash

#SBATCH --job-name=beagle_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/beagle_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/beagle_OVAR_P5.err
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

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

BASEQ=0
MAPQ=0
SECTION_ID=P5_sr13

echo "Processing OVAR_P5 section: ${SECTION_ID}"
echo "Quality filter: baseQ${BASEQ}mapQ${MAPQ}"

python scripts/2_beagle_filtering/run_beagle.py \
    --dataset OVAR_P5 \
    --section_id ${SECTION_ID} \
    --quality-filter baseQ${BASEQ}mapQ${MAPQ} \
    --threads 30 \
    --memory 200g

echo "End time: $(date)"
