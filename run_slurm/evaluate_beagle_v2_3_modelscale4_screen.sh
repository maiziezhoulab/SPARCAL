#!/bin/bash
# Evaluate the prespecified modelscale=4 arm after both samples pass.

#SBATCH --job-name=beagle_v23_ms4eval
#SBATCH --output=slurm_output/beagle_v2_3/ms4_evaluate_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/ms4_evaluate_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB

set -euo pipefail

PROJECT=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling
EXPERIMENT="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_20260913_v1"
ASSESSMENT="$PROJECT/data/experiments/beagle_v2_3_modelscale4_screen_assessment_20260913_v1"
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
if [ -e "$ASSESSMENT" ]; then
    echo "ERROR: refusing to overwrite: $ASSESSMENT"
    exit 2
fi
check_hash bdbda2ed252d17c53df6b7fc0f9cb036e07cefd76b011c9b8a850a53ec120d5c scripts/postanalyze/beagle_configuration_benchmark.py
check_hash f6f26803336b01915da7670b39f0e2ba93d430c1f5e2c7d6210e9857eb323e8e scripts/2_beagle_filtering/beagle_evidence_router.py

"$PYTHON" scripts/postanalyze/beagle_configuration_benchmark.py --experiment-root "$EXPERIMENT" --output-dir "$ASSESSMENT" --anchor-score p_nonref --min-anchor-probability 0.90
echo "Complete: $ASSESSMENT"
