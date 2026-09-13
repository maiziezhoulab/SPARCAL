#!/bin/bash
#SBATCH --job-name=sfv2_score
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%A_%a.err

set -euo pipefail
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
SECTIONS=(P4_rep1 P4_rep2 P6_rep1 P6_rep2)
SECTION=${SECTIONS[${SLURM_ARRAY_TASK_ID}]}
cd "$PROJECT"
"$PYTHON" -m scripts.spatial_filter_rebuild.score_reliability --section "$SECTION"
