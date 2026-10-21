#!/bin/bash
#SBATCH --job-name=sparcal_artifact_v2_dcis2
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.err
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Endpoint-first evaluation only: reports raw, UMI-normalized, and joint
# UMI-plus-evidence models against locked DCIS2 pathology labels.  It never
# filters a VCF or tunes evidence thresholds.
set -euo pipefail

cd /data/maiziezhou_lab/leiy4/snv_calling
source activate snv_caller

python scripts/postanalyze/evaluate_artifact_evidence_dcis2_pathology.py \
  --out-dir data/artifact_evidence_benchmark/v2_dcis2_pathology_2026-07-16
