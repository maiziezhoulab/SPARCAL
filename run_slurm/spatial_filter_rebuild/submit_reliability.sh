#!/bin/bash
set -euo pipefail

HERE=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/spatial_filter_rebuild
SCORE_JOB=$(sbatch --parsable "$HERE/score_reliability_array.slurm")
VIEWER_JOB=$(sbatch --parsable --dependency="afterok:${SCORE_JOB}" "$HERE/materialize_reliability_array.slurm")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${VIEWER_JOB}" "$HERE/evaluate_reliability.slurm")

echo "reliability_score=${SCORE_JOB}"
echo "reliability_viewer=${VIEWER_JOB}"
echo "reliability_evaluation=${EVAL_JOB}"
