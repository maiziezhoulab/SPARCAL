#!/bin/bash
#SBATCH --job-name=vcf_vis_pb
#SBATCH --array=1-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --time=6:00:00

#SBATCH --output=slurm_output/vcf_vis_pb/%x-%A_%a.out
#SBATCH --error=slurm_output/vcf_vis_pb/%x-%A_%a.err

#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int

# ── Select dataset / section (uncomment exactly one block) ────────────────
# Array task 1 → germline denovo
# Array task 2 → somatic  denovo
# Array task 3 → germline defined

# DCIS section 1
DATASET="DCIS"; SECTION="1"
DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis1"

# DCIS section 2
# DATASET="DCIS"; SECTION="2"
# DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis2"

# P4 section 1
DATASET="P4_TUMOR"; SECTION="1"
DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1"

# P4 section 2
# DATASET="P4_TUMOR"; SECTION="2"
# DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/2"

# P6 section 1
DATASET="P6_TUMOR"; SECTION="1"
DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor/1"

# P6 section 2
# DATASET="P6_TUMOR"; SECTION="2"
# DATA_ROOT="/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor/2"

# ─────────────────────────────────────────────────────────────────────────────

QUALITY_FILTER="baseQ0mapQ0"
MAX_WORKERS=22

SPF_DIR="${DATA_ROOT}/spatial_filter_purity/${QUALITY_FILTER}"

# Map array task ID to barcode directory and race filter
case $SLURM_ARRAY_TASK_ID in
  1) BARCODE_DIR="${SPF_DIR}/germline"; RACE="denovo"  ;;
  2) BARCODE_DIR="${SPF_DIR}/somatic";  RACE="denovo"  ;;
  3) BARCODE_DIR="${SPF_DIR}/germline"; RACE="defined" ;;
esac

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "slurm_output/vcf_vis_pb"

echo "==============================================="
echo "SLURM_JOB_ID:     $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK: $SLURM_ARRAY_TASK_ID"
echo "Dataset:          $DATASET"
echo "Section:          $SECTION"
echo "Barcode dir:      $BARCODE_DIR"
echo "Race:             $RACE"
echo "Max workers:      $MAX_WORKERS"
echo "Hostname:         $(hostname)"
echo "PWD:              $(pwd)"
echo "Start time:       $(date)"
echo "==============================================="

echo "[step 1] initializing conda"
source activate snv_caller

echo "which python:       $(which python 2>/dev/null || echo '<not found>')"
echo "python version:     $(python --version 2>&1)"
echo "CONDA_DEFAULT_ENV:  ${CONDA_DEFAULT_ENV:-<unset>}"

echo "[step 2] launching vcf_visualizer_per_barcode_presented.py"
python -u scripts/tools/vcf_visualizer_per_barcode_presented.py \
    --barcode-dir "${BARCODE_DIR}" \
    --race        "${RACE}"        \
    --dataset     "${DATASET}"     \
    --section-id  "${SECTION}"     \
    --max-workers "${MAX_WORKERS}"
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[OK]  ${DATASET} sec${SECTION} $(basename ${BARCODE_DIR}) ${RACE} completed at $(date)"
else
    echo "[ERR] ${DATASET} sec${SECTION} $(basename ${BARCODE_DIR}) ${RACE} failed (exit ${STATUS}) at $(date)"
fi

echo "End time: $(date)"
exit $STATUS
