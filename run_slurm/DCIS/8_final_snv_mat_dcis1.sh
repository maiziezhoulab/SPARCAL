#!/bin/bash

#SBATCH --job-name=final_snv_mat_DCIS1
#SBATCH --output=slurm_output/DCIS/baseQ0mapQ0/final_snv_mat_dcis1.out
#SBATCH --error=slurm_output/DCIS/baseQ0mapQ0/final_snv_mat_dcis1.err
#SBATCH --time=06:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=dcis1   # resolves to data/dcis1/

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

mkdir -p slurm_output/DCIS/${QUALITY_FILTER}

echo "Generating final SNV matrices for DCIS section ${SECTION_ID} (${QUALITY_FILTER})"

python scripts/postprocess/final_snv_mat.py \
    --dataset DCIS \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER}

if [ $? -eq 0 ]; then
    echo "SUCCESS: final SNV matrices for DCIS section ${SECTION_ID}"
else
    echo "ERROR: final SNV matrix generation failed for DCIS section ${SECTION_ID}"
    exit 1
fi

echo "End time: $(date)"
