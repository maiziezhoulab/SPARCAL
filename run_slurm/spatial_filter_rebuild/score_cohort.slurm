#!/bin/bash
#SBATCH --job-name=sfv1_score
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%j.err

set -euo pipefail
cd /data/maiziezhou_lab/leiy4/snv_calling
/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python \
  -m scripts.spatial_filter_rebuild.score_models

