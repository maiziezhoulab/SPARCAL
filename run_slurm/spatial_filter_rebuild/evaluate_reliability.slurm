#!/bin/bash
#SBATCH --job-name=sfv2_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/%x-%j.err

set -euo pipefail
cd /data/maiziezhou_lab/leiy4/snv_calling
/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python \
  -m scripts.spatial_filter_rebuild.evaluate_full \
  --models baseline molecular spatial calicost integrated consensus \
    molecular_reliability spatial_reliability calicost_reliability \
    integrated_reliability consensus_reliability \
  --output data/spatial_filter_rebuild_2026-09-12/multistate_v1_20260912/evaluation_v3_reliability
