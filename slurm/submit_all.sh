#!/bin/bash
# =============================================================================
# SPARCAL Master Launcher — submit all dataset pipelines to SLURM
#
# Usage:
#   cd /data/maiziezhou_lab/leiy4/SPARCAL
#   sbatch slurm/submit_all.sh
#
# Or submit individually:
#   sbatch slurm/run_DLPFC.sh
#   sbatch slurm/run_P4_TUMOR.sh
#   sbatch slurm/run_P6_TUMOR.sh
#   sbatch slurm/run_DCIS.sh
# =============================================================================

set -euo pipefail

SLURM_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SLURM_DIR")"

echo "SPARCAL Project: ${PROJECT_DIR}"
echo "SLURM scripts:   ${SLURM_DIR}"
echo ""

# Create output directories for SLURM logs
mkdir -p ${PROJECT_DIR}/slurm_output/DLPFC
mkdir -p ${PROJECT_DIR}/slurm_output/P4_TUMOR
mkdir -p ${PROJECT_DIR}/slurm_output/P6_TUMOR
mkdir -p ${PROJECT_DIR}/slurm_output/DCIS

cd ${PROJECT_DIR}

echo "Submitting DLPFC (germline, 12 sections)..."
JOB_DLPFC=$(sbatch --parsable ${SLURM_DIR}/run_DLPFC.sh)
echo "  Job ID: ${JOB_DLPFC}"

echo "Submitting P4_TUMOR (somatic, 2 sections)..."
JOB_P4=$(sbatch --parsable ${SLURM_DIR}/run_P4_TUMOR.sh)
echo "  Job ID: ${JOB_P4}"

echo "Submitting P6_TUMOR (somatic, 2 sections)..."
JOB_P6=$(sbatch --parsable ${SLURM_DIR}/run_P6_TUMOR.sh)
echo "  Job ID: ${JOB_P6}"

echo "Submitting DCIS (somatic, 2 sections)..."
JOB_DCIS=$(sbatch --parsable ${SLURM_DIR}/run_DCIS.sh)
echo "  Job ID: ${JOB_DCIS}"

echo ""
echo "======================================================"
echo "All jobs submitted!"
echo "  DLPFC:    ${JOB_DLPFC}"
echo "  P4_TUMOR: ${JOB_P4}"
echo "  P6_TUMOR: ${JOB_P6}"
echo "  DCIS:     ${JOB_DCIS}"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel ${JOB_DLPFC} ${JOB_P4} ${JOB_P6} ${JOB_DCIS}"
echo "======================================================"