#!/bin/bash
# Evaluate and gate the clean v2.3 factorial screen after all array tasks pass.

#SBATCH --job-name=beagle_v23_eval
#SBATCH --output=slurm_output/beagle_v2_3/evaluate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/evaluate_%j.err
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
EXPERIMENT="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_20260913_v1"
ASSESSMENT="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_assessment_20260913_v1"
SELECTION="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_selection_20260913_v1"
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
if [ -e "$ASSESSMENT" ] || [ -e "$SELECTION" ]; then
    echo "ERROR: refusing to overwrite assessment or selection root"
    exit 2
fi
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash 3bdcf961aad25140d7173450b8cda0938f0bc4bf07d6b3c8e47b6e64bc25dcfa scripts/postanalyze/assess_beagle_v2_3_screen.py
check_hash f6f26803336b01915da7670b39f0e2ba93d430c1f5e2c7d6210e9857eb323e8e scripts/2_beagle_filtering/beagle_evidence_router.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash a4e7de84f66403b981cbb30d72eec14bf4115ef41d44cb81572626f1ec623221 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md

"$PYTHON" scripts/postanalyze/beagle_configuration_benchmark.py --experiment-root "$EXPERIMENT" --output-dir "$ASSESSMENT" --anchor-score p_nonref --min-anchor-probability 0.90

"$PYTHON" scripts/postanalyze/assess_beagle_v2_3_screen.py --experiment-root "$EXPERIMENT" --assessment-root "$ASSESSMENT" --output-dir "$SELECTION"

echo "Complete: $ASSESSMENT"
echo "Complete: $SELECTION"
