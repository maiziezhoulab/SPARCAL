#!/bin/bash
# Compare modelscale=4 with the clean ms2 comparator after both assessments pass.

#SBATCH --job-name=beagle_v23_ms4gate
#SBATCH --output=slurm_output/beagle_v2_3/ms4_gate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/ms4_gate_%j.err
#SBATCH --time=02:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB

set -euo pipefail

PROJECT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_selection_20260913_v1"
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
if [ -e "$OUTPUT" ]; then
    echo "ERROR: refusing to overwrite: $OUTPUT"
    exit 2
fi
check_hash e9080915d71a802c30a2231ca18deb45604bafe54ca49ca2cd84e82b9e460a3f scripts/postanalyze/assess_beagle_v2_3_modelscale_extension.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash b94bc7372bcb99f2d28c45c5433202ef13cd88140b94cefb219f65234342b009 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md

"$PYTHON" scripts/postanalyze/assess_beagle_v2_3_modelscale_extension.py --factorial-experiment-root "$PROJECT/data/experiments/beagle_v2_3_factorial_screen_20260913_v1" --factorial-assessment-root "$PROJECT/data/experiments/beagle_v2_3_factorial_screen_assessment_20260913_v1" --factorial-selection-root "$PROJECT/data/experiments/beagle_v2_3_factorial_screen_selection_20260913_v1" --extension-experiment-root "$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_20260913_v1" --extension-assessment-root "$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_assessment_20260913_v1" --output-dir "$OUTPUT"

echo "Complete: $OUTPUT"
