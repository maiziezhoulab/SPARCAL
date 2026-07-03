#!/bin/bash
#SBATCH --job-name=beagle_in_snp_profile
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --time=24:00:00

#SBATCH --output=slurm_output/beagle_in_snp_profile/baseQ0mapQ0/beagle_in_snp_profile-%j.out
#SBATCH --error=slurm_output/beagle_in_snp_profile/baseQ0mapQ0/beagle_in_snp_profile-%j.err

#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int

DATASET="DCIS"
SECTION="1"
MAX_WORKERS=22
QUALITY_FILTER="baseQ0mapQ0"

VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/output_VCFs/beagle/${QUALITY_FILTER}/all_filtered_in.vcf.gz"

mkdir -p "slurm_output/beagle_in_snp_profile/${QUALITY_FILTER}"

echo "==============================================="
echo "SLURM_JOB_ID:  $SLURM_JOB_ID"
echo "Dataset:       $DATASET"
echo "Section:       $SECTION"
echo "Max workers:   $MAX_WORKERS"
echo "VCF:           $VCF"
echo "Hostname:      $(hostname)"
echo "PWD:           $(pwd)"
echo "Start time:    $(date)"
echo "==============================================="

echo "[step 1] initializing conda"
source /data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/etc/profile.d/conda.sh
conda activate snv_caller
ACT=$?
echo "[step 1] exit=${ACT}"

if [ "${ACT}" != "0" ]; then
    echo "[ERR] conda activate snv_caller failed (status ${ACT})"
    exit 1
fi

echo "which python:       $(which python 2>/dev/null || echo '<not found>')"
echo "python version:     $(python --version 2>&1)"
echo "CONDA_DEFAULT_ENV:  ${CONDA_DEFAULT_ENV:-<unset>}"

echo "[step 2] launching vcf_visualizer.py"
python -u scripts/tools/vcf_visualizer.py \
    --vcf         "${VCF}"         \
    --dataset     "${DATASET}"     \
    --section-id  "${SECTION}"     \
    --max-workers "${MAX_WORKERS}"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[OK]  ${DATASET} sec${SECTION} beagle_in completed at $(date)"
else
    echo "[ERR] ${DATASET} sec${SECTION} beagle_in failed (exit ${STATUS}) at $(date)"
fi

echo "End time: $(date)"
exit $STATUS
