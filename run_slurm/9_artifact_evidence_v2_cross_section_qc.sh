#!/bin/bash
#SBATCH --job-name=sparcal_artifact_v2_qc
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

# Submit with --dependency=afterany:<cross-section-array-job-id>.  The report
# records missing outputs as well as completed pilots and never edits inputs.
set -euo pipefail

cd /data/maiziezhou_lab/leiy4/snv_calling
source activate snv_caller

python scripts/postanalyze/summarize_artifact_evidence_pilots.py \
  --out-dir data/artifact_evidence_benchmark/v2_cross_section_2026-07-16 \
  --run P4=data/P4_tumor/1/artifact_evidence/v2_pilot_2026-07-15_p4_batch \
  --run P6=data/P6_tumor/1/artifact_evidence/v2_pilot_2026-07-16_p6 \
  --run DCIS1=data/dcis1/artifact_evidence/v2_pilot_2026-07-16_dcis1 \
  --run DCIS2=data/dcis2/artifact_evidence/v2_pilot_2026-07-16_dcis2
