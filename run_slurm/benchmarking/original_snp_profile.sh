#!/bin/bash
#SBATCH --job-name=orig_snp_profile
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --time=24:00:00

#SBATCH --output=slurm_output/original_snp_profile/baseQ0mapQ0/original_snp_profile-%A_%a.out
#SBATCH --error=slurm_output/original_snp_profile/baseQ0mapQ0/original_snp_profile-%A_%a.err

#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int

# Array task mapping:
#   0  P4_TUMOR  section 1  (BAM pysam scan, ~8 GB BAM)
#   1  P6_TUMOR  section 1  (BAM pysam scan, ~7 GB BAM)
#   2  DCIS      section 1  (VCF split, multi-sample)
#   3  DCIS      section 2  (VCF split, multi-sample)

declare -A DATASETS=(  [0]="P4_TUMOR" [1]="P6_TUMOR" [2]="DCIS" [3]="DCIS" )
declare -A SECTIONS=(  [0]="1"        [1]="1"         [2]="1"    [3]="2"    )
declare -A WORKERS=(   [0]="22"       [1]="22"        [2]="30"   [3]="30"   )

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
SECTION="${SECTIONS[$SLURM_ARRAY_TASK_ID]}"
MAX_WORKERS="${WORKERS[$SLURM_ARRAY_TASK_ID]}"
QUALITY_FILTER="baseQ0mapQ0"

mkdir -p "slurm_output/original_snp_profile/${QUALITY_FILTER}"

echo "==============================================="
echo "SLURM_ARRAY_JOB_ID:  $SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Dataset:             $DATASET"
echo "Section:             $SECTION"
echo "Max workers:         $MAX_WORKERS"
echo "Hostname:            $(hostname)"
echo "PWD:                 $(pwd)"
echo "Start time:          $(date)"
echo "==============================================="

echo "[step 1] source activate snv_caller"
source activate snv_caller
ACT=$?
echo "[step 1] exit=${ACT}"

if [ "${ACT}" != "0" ]; then
    echo "[ERR] source activate snv_caller failed (status ${ACT})"
    exit 1
fi

echo "which python:  $(which python 2>/dev/null || echo '<not found>')"
echo "python version: $(python --version 2>&1)"
echo "CONDA_DEFAULT_ENV: ${CONDA_DEFAULT_ENV:-<unset>}"

echo "[step 2] launching python"
python -u scripts/tools/generate_original_snp_profile.py \
    --dataset     "${DATASET}"     \
    --section-id  "${SECTION}"     \
    --max-workers "${MAX_WORKERS}"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[OK]  ${DATASET} sec${SECTION} completed at $(date)"
else
    echo "[ERR] ${DATASET} sec${SECTION} failed (exit ${STATUS}) at $(date)"
fi

echo "End time: $(date)"
exit $STATUS


