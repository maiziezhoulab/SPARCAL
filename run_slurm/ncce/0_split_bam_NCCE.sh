#!/usr/bin/env bash
# NCCE pre-calling step: split read-only SpaceRanger BAMs by CB tag.  Every
# output BAM and index is written under data/ncce/<sample>/split_BAM/.
#
# Submit: sbatch run_slurm/ncce/0_split_bam_NCCE.sh
# Resume one sample: sbatch --array=9 run_slurm/ncce/0_split_bam_NCCE.sh  # 11A
#
#SBATCH --job-name=split_bam_NCCE
#SBATCH --output=slurm_output/NCCE/split_bam_%A_%a.out
#SBATCH --error=slurm_output/NCCE/split_bam_%A_%a.err
#SBATCH --array=0-14%3
#SBATCH --time=18:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB

set -euo pipefail

SAMPLES=(6A 6B 6C 8A 8B 8C 10A 10B 10C 11A 11B 11C 14A 14B 14C)
SAMPLE=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
OUTS=/data/maiziezhou_lab/Weiman/ST_CNV/data/spaceranger_outs/NCCE_${SAMPLE}/outs
OUT_DIR=${PROJECT}/data/ncce/${SAMPLE}/split_BAM

cd "${PROJECT}"
# Compute nodes do not expose the login-node `activate` helper.  Initialize
# Conda explicitly so the pinned project environment is available everywhere.
source /panfs/accrepfs.vampire/accre/software/easybuild/software/2023/x86-64-v3/Core/miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller
mkdir -p slurm_output/NCCE "${OUT_DIR}"

echo "=== NCCE ${SAMPLE}: split BAM | ${SLURM_JOB_ID:-NA} | $(date) ==="
python scripts/0_split_bam/split_bam_by_cb.py \
    --bam "${OUTS}/possorted_genome_bam.bam" \
    --positions "${OUTS}/spatial/tissue_positions.csv" \
    --out-dir "${OUT_DIR}" \
    --in-tissue-only --index --threads 8

echo "=== NCCE ${SAMPLE}: split complete | $(date) ==="
