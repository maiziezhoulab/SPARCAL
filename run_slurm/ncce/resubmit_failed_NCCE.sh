#!/usr/bin/env bash
# Resume only the NCCE sections that did not finish on the first run.
# All inputs and outputs are project-local under data/ncce/; the original
# ST_CNV SpaceRanger and annotation directories remain read-only.

set -euo pipefail

cd /data/maiziezhou_lab/leiy4/snv_calling

# 6A, 6B, 6C, 8C, 11C, 14B: rerun Beagle onward after accepting empty contigs.
SPARSE_BEAGLE_JOB=$(sbatch --parsable --array=0,1,2,5,11,13%2 \
    run_slurm/ncce/run_pipeline_NCCE.sh 2)

# 10B: Beagle/genotype shifts completed; resume at the classifier fallback.
TEN_B_JOB=$(sbatch --parsable --array=7 \
    run_slurm/ncce/run_pipeline_NCCE.sh 5)

echo "NCCE sparse-Beagle resume: ${SPARSE_BEAGLE_JOB}"
echo "NCCE 10B classifier resume: ${TEN_B_JOB}"
