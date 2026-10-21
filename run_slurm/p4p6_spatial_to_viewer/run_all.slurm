#!/bin/bash
# One submission builds four versioned P4/P6 spatial-filter -> viewer studies.
# Usage: sbatch run_slurm/p4p6_spatial_to_viewer/run_all.slurm RUN_LABEL VAF_MODE EVIDENCE_MODE [--overwrite]
# Example: sbatch run_slurm/p4p6_spatial_to_viewer/run_all.slurm ad_vaf_shadow_v1 shadow full
#SBATCH --job-name=p4p6_to_viewer
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
RUN_LABEL=${1:?RUN_LABEL is required, e.g. ad_vaf_shadow_v1}
VAF_MODE=${2:-shadow}
EVIDENCE_MODE=${3:-full}

case "${SLURM_ARRAY_TASK_ID}" in
    0) DATASET=P4_TUMOR; SECTION_ID=1 ;;
    1) DATASET=P4_TUMOR; SECTION_ID=2 ;;
    2) DATASET=P6_TUMOR; SECTION_ID=1 ;;
    3) DATASET=P6_TUMOR; SECTION_ID=2 ;;
    *) echo "ERROR: unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

case "$EVIDENCE_MODE:$VAF_MODE" in
    full:off|full:shadow|full:vote|spatial_only_no_calicost:off) ;;
    *) echo "ERROR: unsupported EVIDENCE_MODE:VAF_MODE combination: $EVIDENCE_MODE:$VAF_MODE" >&2; exit 2 ;;
esac

EXTRA_ARGS=()
if [[ "${4:-}" == "--overwrite" ]]; then
    EXTRA_ARGS+=(--overwrite)
fi

VIEWER_OUTPUT_ROOT=/data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/dist/studies/${RUN_LABEL}

cd "$PROJECT_DIR"
"$PYTHON" scripts/6_spatial_filter/run_spatial_to_viewer.py \
    --dataset "$DATASET" \
    --section-id "$SECTION_ID" \
    --quality-filter baseQ0mapQ0 \
    --run-label "$RUN_LABEL" \
    --evidence-mode "$EVIDENCE_MODE" \
    --mechanistic-vaf-mode "$VAF_MODE" \
    --matrix-class somatic \
    --viewer-output-root "$VIEWER_OUTPUT_ROOT" \
    "${EXTRA_ARGS[@]}"
