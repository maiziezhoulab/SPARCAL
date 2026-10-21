#!/bin/bash
#SBATCH --job-name=coupling_viewer
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/data/spatial_viewer_coupling_2026-09-16/logs/slurm/%x_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/data/spatial_viewer_coupling_2026-09-16/logs/slurm/%x_%j.err
set -euo pipefail
cd /data/maiziezhou_lab/leiy4/snv_calling
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_clustering/bin/python -m scripts.spatial_viewer_coupling.export_viewer
