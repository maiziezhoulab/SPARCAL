#!/bin/bash
#SBATCH --job-name=spacetracer_dcis1
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer/slurm/slurm_output/dcis1-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer/slurm/slurm_output/dcis1-%j.err
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

ST_DIR="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer"
CONFIG="${ST_DIR}/configs/dcis1_config.yaml"
ENV_NAME="SpaceTracer_dcis"

echo "======================================================"
echo "SpaceTracer — DCIS Section 1"
echo "SLURM_JOBID : $SLURM_JOBID"
echo "Config      : ${CONFIG}"
echo "Start       : $(date)"
echo "======================================================"

source activate "${ENV_NAME}"

# Verify resources are present before starting
RESOURCE_DIR="${ST_DIR}/resources/hg38"
if [ ! -d "${RESOURCE_DIR}" ] || [ -z "$(ls -A "${RESOURCE_DIR}" 2>/dev/null)" ]; then
    echo "ERROR: hg38 resources not found at ${RESOURCE_DIR}"
    echo "Run download_resources.sh first."
    exit 1
fi

echo "Running SpaceTracer..."
spacetracer run --config "${CONFIG}"

echo ""
echo "======================================================"
echo "DCIS1 complete: $(date)"
echo "Results: ${ST_DIR}/results/dcis1/"
echo "======================================================"
