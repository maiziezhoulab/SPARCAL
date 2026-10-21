#!/bin/bash
# Freeze the exact replication array only after the independent pre-replication
# audit confirms all three screen selectors from canonical run evidence.

#SBATCH --job-name=beagle_v23_replplan
#SBATCH --output=slurm_output/beagle_v2_3/repl_plan_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/repl_plan_%j.err
#SBATCH --time=00:30:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

set -euo pipefail

PROJECT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling
FACTORIAL_SELECTION="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_selection_20260913_v1"
EXTENSION_SELECTION="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_selection_20260913_v1"
MAXLR_SELECTION="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_selection_20260913_v1"
PREAUDIT="$PROJECT/data/experiments/beagle_v2_3_pre_replication_audit_20260913_v1"
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_replication_plan_20260913_v1"
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
    "$MAXLR_SELECTION/RUN_COMPLETE.json" \
    "$PREAUDIT/RUN_COMPLETE.json"; do
    [ -s "$path" ] || { echo "ERROR: missing prerequisite: $path"; exit 1; }
done

check_hash 06aa1273b972cc1199608255529be40efe5ecc432225dfa6d21031b46a417efd scripts/postanalyze/plan_beagle_v2_3_replication.py
check_hash 15bf0120791b76eb1265369a88498ebfbc57861b953735cf8c53c02bfc4bf74f scripts/postanalyze/audit_beagle_v2_3_pre_replication.py
check_hash 775de0e4fa35d18e35a12dc41d5083b74c2e05c92768e3375fce65d5ce656ac5 scripts/postanalyze/beagle_selection_integrity.py

"$PYTHON" scripts/postanalyze/plan_beagle_v2_3_replication.py \
    --factorial-selection-root "$FACTORIAL_SELECTION" \
    --extension-selection-root "$EXTENSION_SELECTION" \
    --maxlr-selection-root "$MAXLR_SELECTION" \
    --pre-replication-audit-root "$PREAUDIT" \
    --output-dir "$OUTPUT"

echo "Complete: $OUTPUT"
