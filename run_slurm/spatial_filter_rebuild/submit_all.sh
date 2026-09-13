#!/bin/bash
set -euo pipefail

HERE=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/spatial_filter_rebuild
CACHE_JOB=$(sbatch --parsable "$HERE/build_features_array.slurm")
SCORE_JOB=$(sbatch --parsable --dependency="afterok:${CACHE_JOB}" "$HERE/score_cohort.slurm")
VIEWER_JOB=$(sbatch --parsable --dependency="afterok:${SCORE_JOB}" "$HERE/materialize_array.slurm")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${VIEWER_JOB}" "$HERE/evaluate.slurm")

echo "cache_features=${CACHE_JOB}"
echo "score=${SCORE_JOB}"
echo "viewer=${VIEWER_JOB}"
echo "evaluation=${EVAL_JOB}"

