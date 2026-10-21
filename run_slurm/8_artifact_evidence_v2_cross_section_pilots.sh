#!/bin/bash
#SBATCH --job-name=sparcal_artifact_v2
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%A_%a.err
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

# Fixed, non-destructive P6/DCIS pilot set.  Each task extracts a stable
# 500-site sample per provenance source (at most 2,000 unique sites) and
# writes only under that section's new artifact_evidence label.
set -euo pipefail

cd /data/maiziezhou_lab/leiy4/snv_calling
source activate snv_caller

case "${SLURM_ARRAY_TASK_ID:?SLURM array task ID is required}" in
    0)
        DATASET=P6_TUMOR
        SECTION_ID=1
        RUN_LABEL=v2_pilot_2026-07-16_p6
        ;;
    1)
        DATASET=DCIS
        SECTION_ID=dcis1
        RUN_LABEL=v2_pilot_2026-07-16_dcis1
        ;;
    2)
        DATASET=DCIS
        SECTION_ID=dcis2
        RUN_LABEL=v2_pilot_2026-07-16_dcis2
        ;;
    *)
        echo "Unsupported array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

python scripts/postanalyze/run_artifact_evidence_workflow.py \
  --dataset "${DATASET}" \
  --section-id "${SECTION_ID}" \
  --run-label "${RUN_LABEL}" \
  --max-sites-per-source 500
