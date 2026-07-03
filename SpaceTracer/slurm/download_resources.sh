#!/bin/bash
#SBATCH --job-name=spacetracer_download
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer/slurm/slurm_output/download-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer/slurm/slurm_output/download-%j.err
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

ST_DIR="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/SpaceTracer"

echo "SpaceTracer — Download hg38 Resources"
echo "Start: $(date)"

bash "${ST_DIR}/download_resources.sh"

echo "Done: $(date)"
