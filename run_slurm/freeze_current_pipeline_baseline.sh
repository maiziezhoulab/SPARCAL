#!/bin/bash
#SBATCH --job-name=sparcal_freeze
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.err
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00

# Manifest only: this never writes under the frozen roots.
set -euo pipefail
cd /data/maiziezhou_lab/leiy4/snv_calling

python scripts/postanalyze/freeze_baseline.py \
  --label current_pipeline_pre_artifact_v2_2026-07-15 \
  --root data/dlpfc \
  --root data/P4_tumor \
  --root data/P6_tumor \
  --root data/dcis1 \
  --root data/dcis2 \
  --root data/ovar_p5 \
  --out-dir data/_frozen_baselines/2026-07-15_current_pipeline \
  --exclude artifact_features \
  --exclude artifact_evidence
