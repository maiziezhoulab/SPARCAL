#!/bin/bash
#SBATCH --job-name=spatial_coupling
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/data/spatial_viewer_coupling_2026-09-16/logs/slurm/%x_%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/data/spatial_viewer_coupling_2026-09-16/logs/slurm/%x_%A_%a.err
set -euo pipefail
cd /data/maiziezhou_lab/leiy4/snv_calling
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
sections=(P6_rep1 P6_rep2 P4_rep1 DCIS1)
/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_clustering/bin/python -m scripts.spatial_viewer_coupling.run_coupled_filter_to_viewer --section "${sections[$SLURM_ARRAY_TASK_ID]}" --workers 4 --run-id registered_v1
