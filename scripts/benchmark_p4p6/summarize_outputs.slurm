#!/usr/bin/env bash
#SBATCH --job-name=p4p6_summary
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
"$SNV_CALLER_ENV/bin/python" "$SCRIPTROOT/summarize_outputs.py"
