#!/bin/bash
# Independently recompute all seed-screen metrics and certify every selector
# decision before any Beagle v2.3 replication tasks may be planned.

#SBATCH --job-name=beagle_v23_preaudit
#SBATCH --output=slurm_output/beagle_v2_3/preaudit_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/preaudit_%j.err
#SBATCH --time=08:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96GB

set -euo pipefail

PROJECT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling
FACTORIAL="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_20260913_v1"
FACTORIAL_SELECTION="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_selection_20260913_v1"
EXTENSION="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_20260913_v1"
EXTENSION_SELECTION="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_selection_20260913_v1"
MAXLR="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_20260913_v1"
MAXLR_SELECTION="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_selection_20260913_v1"
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_pre_replication_audit_20260913_v1"
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
for path in \
    "$FACTORIAL_SELECTION/RUN_COMPLETE.json" \
    "$EXTENSION_SELECTION/RUN_COMPLETE.json" \
    "$MAXLR_SELECTION/RUN_COMPLETE.json"; do
    [ -s "$path" ] || { echo "ERROR: missing prerequisite: $path"; exit 1; }
done

check_hash 15bf0120791b76eb1265369a88498ebfbc57861b953735cf8c53c02bfc4bf74f scripts/postanalyze/audit_beagle_v2_3_pre_replication.py
check_hash 775de0e4fa35d18e35a12dc41d5083b74c2e05c92768e3375fce65d5ce656ac5 scripts/postanalyze/beagle_selection_integrity.py
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash a4e7de84f66403b981cbb30d72eec14bf4115ef41d44cb81572626f1ec623221 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md
check_hash b94bc7372bcb99f2d28c45c5433202ef13cd88140b94cefb219f65234342b009 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md
check_hash fe842e79ee5e842617979bed519450bf16ba0bce05e211b83e563bd73c836a31 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_MAXLR.md

"$PYTHON" scripts/postanalyze/audit_beagle_v2_3_pre_replication.py \
    --factorial-experiment-root "$FACTORIAL" \
    --factorial-selection-root "$FACTORIAL_SELECTION" \
    --extension-experiment-root "$EXTENSION" \
    --extension-selection-root "$EXTENSION_SELECTION" \
    --maxlr-experiment-root "$MAXLR" \
    --maxlr-selection-root "$MAXLR_SELECTION" \
    --output-dir "$OUTPUT"

echo "Complete: $OUTPUT"
