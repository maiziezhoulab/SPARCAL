#!/bin/bash
# Clean P4/P6 2x2 factorial screen for the fail-closed Beagle v2.3 module.

#SBATCH --job-name=beagle_v23_screen
#SBATCH --output=slurm_output/beagle_v2_3/screen_%A_%a.out
#SBATCH --error=slurm_output/beagle_v2_3/screen_%A_%a.err
#SBATCH --array=0-7%4
#SBATCH --time=30:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96GB

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SAMPLE_INDEX=$((TASK % 2))
CONFIG_INDEX=$((TASK / 2))
SEED=20260915

case "$SAMPLE_INDEX" in
    0) DATASET=P4_TUMOR; SAMPLE=P4_sec1 ;;
    1) DATASET=P6_TUMOR; SAMPLE=P6_sec1 ;;
    *) echo "ERROR: invalid sample index: $SAMPLE_INDEX"; exit 2 ;;
esac
case "$CONFIG_INDEX" in
    0) MODEL_SCALE=2; ITERATIONS=0; CONFIG=ms2_niter0 ;;
    1) MODEL_SCALE=2; ITERATIONS=5; CONFIG=ms2_niter5 ;;
    2) MODEL_SCALE=0.8; ITERATIONS=0; CONFIG=ms0p8_niter0 ;;
    3) MODEL_SCALE=0.8; ITERATIONS=5; CONFIG=ms0p8_niter5 ;;
    *) echo "ERROR: invalid configuration index: $CONFIG_INDEX"; exit 2 ;;
esac

QUALITY=baseQ0mapQ0
RUN_LABEL=${RUN_LABEL:-beagle_v2_3_factorial_screen_20260913_v1}
SOURCE_ROOT="$PROJECT/data/experiments/beagle_panel_routing_gp090_20260911_v3/$SAMPLE"
RAW="$SOURCE_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY/merged_sorted_gt.vcf.gz"
LEDGER="$PROJECT/data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen"
ROOT="$PROJECT/data/experiments/$RUN_LABEL/${CONFIG}_seed${SEED}/$SAMPLE"
SNAPSHOT="$ROOT/code_snapshot/scripts/2_beagle_filtering"
INPUT="$ROOT/output_VCFs/mpileup_multi_bam/$QUALITY"

cd "$PROJECT"
[ -s "$RAW" ] || { echo "ERROR: missing raw VCF: $RAW"; exit 1; }
[ -s "$RAW.tbi" ] || { echo "ERROR: missing raw VCF index: $RAW.tbi"; exit 1; }
[ -s "$LEDGER/RUN_COMPLETE.json" ] || { echo "ERROR: missing bound reference ledger"; exit 1; }
if [ -e "$ROOT" ]; then
    echo "ERROR: refusing to overwrite: $ROOT"
    exit 2
fi

mkdir -p "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13" "$INPUT"
mkdir -p "$PROJECT/slurm_output/beagle_v2_3"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$ROOT/RUN_FAILED.txt"' ERR

cp scripts/2_beagle_filtering/run_beagle.py \
   scripts/2_beagle_filtering/beagle_evidence_router.py \
   "$SNAPSHOT/"
cp scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py \
   "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/"
cp run_slurm/run_beagle_v2_3_factorial_screen.sh \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md \
   "$ROOT/code_snapshot/"
find "$ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

ln -s "$RAW" "$INPUT/merged_sorted_gt.vcf.gz"
ln -s "$RAW.tbi" "$INPUT/merged_sorted_gt.vcf.gz.tbi"

PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
"$PYTHON" "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py" \
    --dataset "$DATASET" \
    --section-id 1 \
    --section-root "$ROOT" \
    --reference-ledger-root "$LEDGER" \
    --quality-filter "$QUALITY" \
    --threads 30 \
    --parallel-chromosomes 1 \
    --iterations "$ITERATIONS" \
    --seed "$SEED" \
    --modelscale "$MODEL_SCALE" \
    --maxlr 5000 \
    --anchor-score p_nonref \
    --min-anchor-probability 0.90

trap - ERR
echo "Complete: $ROOT"
