#!/bin/bash
set -euo pipefail

HERE=/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/spatial_filter_rebuild
SCORE_JOB=$(sbatch --parsable "$HERE/score_quality_ablation_array.slurm")
VIEWER_JOB=$(sbatch --parsable --dependency="afterok:${SCORE_JOB}" "$HERE/materialize_quality_ablation_array.slurm")
EVAL_JOB=$(sbatch --parsable --dependency="afterok:${VIEWER_JOB}" "$HERE/evaluate_quality_ablation.slurm")

echo "quality_ablation_score=${SCORE_JOB}"
echo "quality_ablation_viewer=${VIEWER_JOB}"
echo "quality_ablation_evaluation=${EVAL_JOB}"
