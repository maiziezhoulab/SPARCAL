#!/usr/bin/env bash
# Prepare local NCCE annotations, split BAMs, then launch the SPARCAL array.
# The full array is released only when every split-BAM task succeeds.

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
cd "${PROJECT}"

source /panfs/accrepfs.vampire/accre/software/easybuild/software/2023/x86-64-v3/Core/miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller
python scripts/preprocess/prepare_ncce_annotations.py --output-root data/ncce

mkdir -p slurm_output/NCCE/baseQ0mapQ0
SPLIT_JOB=$(sbatch --parsable run_slurm/ncce/0_split_bam_NCCE.sh)
PIPELINE_JOB=$(sbatch --parsable --dependency=afterok:${SPLIT_JOB} run_slurm/ncce/run_pipeline_NCCE.sh)

echo "NCCE split-BAM array: ${SPLIT_JOB}"
echo "NCCE SPARCAL array:   ${PIPELINE_JOB} (afterok:${SPLIT_JOB})"
echo "Monitor: squeue -u $USER -j ${SPLIT_JOB},${PIPELINE_JOB}"
