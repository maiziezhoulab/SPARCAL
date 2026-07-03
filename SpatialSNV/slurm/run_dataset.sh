#!/bin/bash
#SBATCH --job-name=spatialsnv
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/slurm/slurm_output/%x-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV/slurm/slurm_output/%x-%j.err
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
# -----------------------------------------------------------------------------
# Generic SpatialSNV runner. Positional args (avoids sbatch --export comma issue):
#   $1 CONFIG         path to configs/*.env   (required)
#   $2 STAGES         e.g. prep,call,callback (optional, default all)
#   $3 SUBSET_REGION  single contig for smoke test (optional)
# Usage:
#   sbatch --job-name=ssnv_dcis1 slurm/run_dataset.sh configs/dcis1.env
#   sbatch --job-name=ssnv_smoke slurm/run_dataset.sh configs/dcis1.env prep,call,callback 22
# -----------------------------------------------------------------------------
set -euo pipefail
SSNV=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpatialSNV
CONFIG="${1:?usage: run_dataset.sh <config.env> [stages] [subset_region]}"
STAGES="${2:-prep,call,callback}"
export SUBSET_REGION="${3:-}"
source activate spatialsnv
echo "host=$(hostname) jobid=${SLURM_JOBID:-NA} config=$CONFIG stages=$STAGES subset=${SUBSET_REGION:-none}"
bash "$SSNV/scripts/run_spatialsnv.sh" "$CONFIG" "$STAGES"
