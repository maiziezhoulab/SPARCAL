#!/bin/bash

#SBATCH --job-name=final_snv_mat_P6
#SBATCH --output=slurm_output/P6_TUMOR/baseQ0mapQ0/final_snv_mat_P6.out
#SBATCH --error=slurm_output/P6_TUMOR/baseQ0mapQ0/final_snv_mat_P6.err
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

mkdir -p slurm_output/P6_TUMOR/${QUALITY_FILTER}

echo "Generating final SNV matrices for P6_TUMOR section ${SECTION_ID} (${QUALITY_FILTER})"

python scripts/postprocess/final_snv_mat.py \
    --dataset P6_TUMOR \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER}

if [ $? -eq 0 ]; then
    echo "SUCCESS: final SNV matrices for P6_TUMOR section ${SECTION_ID}"
else
    echo "ERROR: final SNV matrix generation failed for P6_TUMOR section ${SECTION_ID}"
    exit 1
fi

echo "End time: $(date)"
