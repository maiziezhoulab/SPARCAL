#!/bin/bash
# Materialize the one selected Beagle representation after every planned
# DLPFC v2.3 run has completed. This script does not submit clustering jobs.

#SBATCH --job-name=bgl23_dlpfc_mat
#SBATCH --output=slurm_output/beagle_v2_3/dlpfc_materialize_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/dlpfc_materialize_%j.err
#SBATCH --time=06:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=96GB

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
PLAN_ROOT=${PLAN_ROOT:?PLAN_ROOT must be the completed immutable DLPFC plan}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must contain the completed 12/36-run grid}
OUTPUT_ROOT=${OUTPUT_ROOT:?OUTPUT_ROOT must be a new materialization root}
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python

check_hash() {
    local expected=$1
    local path=$2
    local observed
    read -r observed _ < <(sha256sum "$path")
    if [ "$observed" != "$expected" ]; then
        echo "ERROR: source hash mismatch: $path"
        exit 2
    fi
}

cd "$PROJECT"
mkdir -p "$PROJECT/slurm_output/beagle_v2_3"
if [ -e "$OUTPUT_ROOT" ]; then
    echo "ERROR: refusing to overwrite: $OUTPUT_ROOT"
    exit 2
fi
check_hash 68030a03a68a87cf0c07a62a0ccdcbe588f8a3fce0a1965182fad85a372bfbad scripts/postanalyze/materialize_beagle_v2_3_final_dlpfc.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash b994cd72f1c858af054810e9ee21cfb730727096862d26de8a570da392444cf4 scripts/postanalyze/beagle_posterior_ensemble.py
check_hash e9a611e27d44436a402bdd561095a2a903f92a9366cb10a3a30a6846254e4e11 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_DLPFC_FINAL.md

"$PYTHON" scripts/postanalyze/materialize_beagle_v2_3_final_dlpfc.py \
    --plan-root "$PLAN_ROOT" \
    --run-root "$RUN_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --protocol scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_DLPFC_FINAL.md

echo "Complete: $OUTPUT_ROOT"
