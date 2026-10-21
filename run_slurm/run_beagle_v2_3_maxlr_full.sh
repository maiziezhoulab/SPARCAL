#!/bin/bash
# Full P4/P6 maxlr=50000 comparison admitted by the frozen chr22 prescreen.

#SBATCH --job-name=beagle_v23_maxlr
#SBATCH --output=slurm_output/beagle_v2_3/maxlr_%A_%a.out
#SBATCH --error=slurm_output/beagle_v2_3/maxlr_%A_%a.err
#SBATCH --array=0-1%2
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
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SEED=20260915
MODEL_SCALE=2
ITERATIONS=0
MAX_LR=50000
CONFIG=ms2_niter0_maxlr50000

case "$TASK" in
    0) DATASET=P4_TUMOR; SAMPLE=P4_sec1 ;;
    1) DATASET=P6_TUMOR; SAMPLE=P6_sec1 ;;
    *) echo "ERROR: invalid sample index: $TASK"; exit 2 ;;
esac

QUALITY=baseQ0mapQ0
RUN_LABEL=${RUN_LABEL:-beagle_v2_3_maxlr50000_screen_20260913_v1}
SOURCE_ROOT="$PROJECT/data/experiments/beagle_panel_routing_gp090_20260911_v3/$SAMPLE"
RAW="$SOURCE_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY/merged_sorted_gt.vcf.gz"
LEDGER="$PROJECT/data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen"
PRESCREEN="$PROJECT/data/experiments/beagle_v2_3_maxlr50000_prescreen_assessment_20260913_v1"
ROOT="$PROJECT/data/experiments/$RUN_LABEL/${CONFIG}_seed${SEED}/$SAMPLE"
SNAPSHOT="$ROOT/code_snapshot/scripts/2_beagle_filtering"
INPUT="$ROOT/output_VCFs/mpileup_multi_bam/$QUALITY"

cd "$PROJECT"
[ -s "$RAW" ] || { echo "ERROR: missing raw VCF: $RAW"; exit 1; }
[ -s "$RAW.tbi" ] || { echo "ERROR: missing raw VCF index: $RAW.tbi"; exit 1; }
[ -s "$LEDGER/RUN_COMPLETE.json" ] || { echo "ERROR: missing bound reference ledger"; exit 1; }
[ -s "$PRESCREEN/RUN_COMPLETE.json" ] || { echo "ERROR: missing maxlr prescreen"; exit 1; }
(cd "$PRESCREEN" && sha256sum -c OUTPUT_SHA256SUMS.txt)
[ "$("$PYTHON" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("advance_to_full_genome", False)).lower())' "$PRESCREEN/RUN_COMPLETE.json")" = "true" ] || {
    echo "ERROR: maxlr prescreen did not advance"
    exit 2
}
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
cp run_slurm/run_beagle_v2_3_maxlr_full.sh \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_MAXLR.md \
   "$PRESCREEN/maxlr_prescreen_summary.tsv" \
   "$PRESCREEN/PROVENANCE.json" \
   "$PRESCREEN/CODE_AND_INPUT_SHA256SUMS.txt" \
   "$PRESCREEN/OUTPUT_SHA256SUMS.txt" \
   "$PRESCREEN/RUN_COMPLETE.json" \
   "$ROOT/code_snapshot/"
find "$ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

ln -s "$RAW" "$INPUT/merged_sorted_gt.vcf.gz"
ln -s "$RAW.tbi" "$INPUT/merged_sorted_gt.vcf.gz.tbi"

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
    --maxlr "$MAX_LR" \
    --anchor-score p_nonref \
    --min-anchor-probability 0.90

trap - ERR
echo "Complete: $ROOT"
