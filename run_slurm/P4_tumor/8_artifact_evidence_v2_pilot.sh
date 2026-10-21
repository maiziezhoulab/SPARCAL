#!/bin/bash
#SBATCH --job-name=sparcal_artifact_v2_p4
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/slurm_output/%x_%j.err
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# Non-destructive balanced pilot: 500 sites from each of the current somatic,
# ambiguous, defined, and sequence-error candidate strata.  It never modifies
# the frozen baseline or existing spatial-filter output.
set -euo pipefail

cd /data/maiziezhou_lab/leiy4/snv_calling
source activate snv_caller

python scripts/postanalyze/run_artifact_evidence_workflow.py \
  --dataset P4_TUMOR \
  --section-id 1 \
  --run-label v2_pilot_2026-07-15_p4_batch \
  --max-sites-per-source 500
