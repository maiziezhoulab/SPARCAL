#!/bin/bash
# Evaluate the frozen maxlr=50000 comparison only after both the maxlr sample
# tasks and the complete factorial evaluator pass.  The latter provides the
# ms2/niter0 P4/P6 comparator bundles consumed below; omitting that dependency
# can launch this job before the P6 comparator exists.

#SBATCH --job-name=beagle_v23_maxlrgate
#SBATCH --output=slurm_output/beagle_v2_3/maxlr_gate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/maxlr_gate_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96GB

set -euo pipefail

PROJECT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_selection_20260913_v1"
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

# Hashes are populated immediately before submission, after all tests pass.
check_hash 4bba091cf74e21cb35804dda1d0db0bda07bcc14608d64c6cda6dbb618aea491 scripts/postanalyze/assess_beagle_v2_3_maxlr_extension.py
check_hash 775de0e4fa35d18e35a12dc41d5083b74c2e05c92768e3375fce65d5ce656ac5 scripts/postanalyze/beagle_selection_integrity.py
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash fe842e79ee5e842617979bed519450bf16ba0bce05e211b83e563bd73c836a31 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_MAXLR.md

"$PYTHON" scripts/postanalyze/assess_beagle_v2_3_maxlr_extension.py \
    --factorial-experiment-root "$PROJECT/data/experiments/beagle_v2_3_factorial_screen_20260913_v1" \
    --maxlr-experiment-root "$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_20260913_v1" \
    --prescreen-root "$PROJECT/data/experiments/beagle_v2_3_maxlr50000_prescreen_assessment_20260913_v1" \
    --output-dir "$OUTPUT"

echo "Complete: $OUTPUT"
