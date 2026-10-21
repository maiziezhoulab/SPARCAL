#!/bin/bash
# Recompute, gate, and select the final Beagle v2.3 base configuration after
# the exact frozen replication array completes successfully.

#SBATCH --job-name=beagle_v23_replgate
#SBATCH --output=slurm_output/beagle_v2_3/repl_gate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/repl_gate_%j.err
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
PLAN="$PROJECT/data/experiments/beagle_v2_3_replication_plan_20260913_v1"
FACTORIAL="$PROJECT/data/experiments/beagle_v2_3_factorial_screen_20260913_v1"
EXTENSION="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_20260913_v1"
MAXLR="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_screen_20260913_v1"
PREAUDIT="$PROJECT/data/experiments/beagle_v2_3_pre_replication_audit_20260913_v1"
REPLICATION="$PROJECT/data/experiments/beagle_v2_3_replication_20260913_v1"
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_replication_assessment_20260913_v1"
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
    "$PLAN/RUN_COMPLETE.json" \
    "$PREAUDIT/RUN_COMPLETE.json" \
    "$FACTORIAL/ms2_niter0_seed20260915/P4_sec1/RUN_COMPLETE.json" \
    "$EXTENSION/ms4_niter0_seed20260915/P4_sec1/RUN_COMPLETE.json" \
    "$MAXLR/ms2_niter0_maxlr50000_seed20260915/P4_sec1/RUN_COMPLETE.json"; do
    [ -s "$path" ] || { echo "ERROR: missing prerequisite: $path"; exit 1; }
done

check_hash 7585ee5c938871dd096104be3e7e6c115542519c5ded0c5de87c477e5ca0ab60 scripts/postanalyze/assess_beagle_v2_3_replication.py
check_hash 06aa1273b972cc1199608255529be40efe5ecc432225dfa6d21031b46a417efd scripts/postanalyze/plan_beagle_v2_3_replication.py
check_hash 15bf0120791b76eb1265369a88498ebfbc57861b953735cf8c53c02bfc4bf74f scripts/postanalyze/audit_beagle_v2_3_pre_replication.py
check_hash 775de0e4fa35d18e35a12dc41d5083b74c2e05c92768e3375fce65d5ce656ac5 scripts/postanalyze/beagle_selection_integrity.py
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash a4e7de84f66403b981cbb30d72eec14bf4115ef41d44cb81572626f1ec623221 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md
check_hash b94bc7372bcb99f2d28c45c5433202ef13cd88140b94cefb219f65234342b009 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md
check_hash fe842e79ee5e842617979bed519450bf16ba0bce05e211b83e563bd73c836a31 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_MAXLR.md
check_hash 79ee2af1bf1248344d7f36af79fc2eeb4a168c9e8a8e0f7592e430739be14db7 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_REPLICATION.md
check_hash 9027480099c188500398e7d96c9afa523bda2b96823c4f16135d335ac06f06f9 run_slurm/run_beagle_v2_3_replication.sh

"$PYTHON" scripts/postanalyze/assess_beagle_v2_3_replication.py \
    --plan-root "$PLAN" \
    --factorial-experiment-root "$FACTORIAL" \
    --extension-experiment-root "$EXTENSION" \
    --maxlr-experiment-root "$MAXLR" \
    --replication-experiment-root "$REPLICATION" \
    --output-dir "$OUTPUT"

echo "Complete: $OUTPUT"
