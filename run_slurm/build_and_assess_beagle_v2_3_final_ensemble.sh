#!/bin/bash
# Build the prespecified posterior summaries for the selected base, recompute
# their P4/P6 metrics, and select the one final representation.  This does not
# run the mandatory downstream DLPFC STAGATE+mclust noninferiority benchmark.

#SBATCH --job-name=beagle_v23_ensgate
#SBATCH --output=slurm_output/beagle_v2_3/ensemble_gate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/ensemble_gate_%j.err
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
BASE="$PROJECT/data/experiments/beagle_v2_3_replication_assessment_20260913_v1"
ENSEMBLE="$PROJECT/data/experiments/beagle_v2_3_selected_posterior_ensemble_20260913_v1"
METRICS="$PROJECT/data/experiments/beagle_v2_3_selected_posterior_ensemble_assessment_20260913_v1"
OUTPUT="$PROJECT/data/experiments/beagle_v2_3_final_representation_selection_20260913_v1"
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
[ -s "$BASE/RUN_COMPLETE.json" ] || {
    echo "ERROR: final base assessment is incomplete: $BASE"
    exit 1
}
for root in "$ENSEMBLE" "$METRICS" "$OUTPUT"; do
    if [ -e "$root" ]; then
        echo "ERROR: refusing to overwrite: $root"
        exit 2
    fi
done

check_hash 2d1acffbda9a9458960138ab37090d7fd22376adb6612df044e3fb58a933ee59 scripts/postanalyze/build_selected_beagle_ensemble.py
check_hash b994cd72f1c858af054810e9ee21cfb730727096862d26de8a570da392444cf4 scripts/postanalyze/beagle_posterior_ensemble.py
check_hash d88a8198a89ab5e4a60df38725480516dcdd50963a630b10452adf9297163995 scripts/postanalyze/assess_beagle_posterior_ensemble.py
check_hash 775de0e4fa35d18e35a12dc41d5083b74c2e05c92768e3375fce65d5ce656ac5 scripts/postanalyze/beagle_selection_integrity.py
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash b28fa06b4ab386b094ac44be5aeb860295f465822df5b2954a11eb56c58e6710 scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py
check_hash 588c248b70120c67e5fa914db31283501ec67db53c998c2a293b30a59e2882fe scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_ENSEMBLE.md

"$PYTHON" scripts/postanalyze/build_selected_beagle_ensemble.py \
    --base-assessment-root "$BASE" \
    --output-root "$ENSEMBLE"

"$PYTHON" scripts/postanalyze/beagle_configuration_benchmark.py \
    --experiment-root "$ENSEMBLE" \
    --output-dir "$METRICS" \
    --anchor-score p_nonref \
    --min-anchor-probability 0.90

"$PYTHON" scripts/postanalyze/assess_beagle_posterior_ensemble.py \
    --base-assessment-root "$BASE" \
    --ensemble-root "$ENSEMBLE" \
    --ensemble-assessment-root "$METRICS" \
    --output-dir "$OUTPUT"

echo "Complete: $ENSEMBLE"
echo "Complete: $METRICS"
echo "Complete: $OUTPUT"
