#!/bin/bash
set -euo pipefail

HERE=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/spatial_filter_rebuild
SCORE_JOB=$(sbatch --parsable "$HERE/score_quality_balanced_array.slurm")
VIEWER_JOB=$(sbatch --parsable --dependency="afterok:${SCORE_JOB}" "$HERE/materialize_quality_balanced_array.slurm")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${VIEWER_JOB}" "$HERE/evaluate_quality_balanced.slurm")

echo "quality_balanced_score=${SCORE_JOB}"
echo "quality_balanced_viewer=${VIEWER_JOB}"
echo "quality_balanced_evaluation=${EVAL_JOB}"
