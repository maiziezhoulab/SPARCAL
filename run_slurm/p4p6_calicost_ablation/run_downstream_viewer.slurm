#!/bin/bash
#SBATCH --job-name=p4p6_ablation_downstream
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p4p6_calicost_ablation/%x-%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p4p6_calicost_ablation/%x-%j.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python

cd "$PROJECT_DIR"
"$PYTHON" scripts/postanalyze/p4p6_calicost_ablation.py
"$PYTHON" scripts/preprocess/build_p4p6_calicost_ablation_viewer_profiles.py

test -s data/p4p6_calicost_ablation_20260907/REPORT.md
test -s /data/maiziezhou_lab/leiy4/SPARCAL_Benchmarking/viewer/study_manifests/P4P6_calicost_ablation_20260907.tsv
echo "[$(date --iso-8601=seconds)] downstream analysis and 8 viewer studies complete"
