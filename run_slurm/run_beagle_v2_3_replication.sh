#!/bin/bash
# Replicate only configurations admitted by the frozen v2.3 selectors.
#
# Submit with the exact `array_max` recorded by the frozen plan, for example:
# sbatch --array=0-ARRAY_MAX%4 --export=ALL,TASKS_FILE=/absolute/replication_tasks.tsv \
#   run_slurm/run_beagle_v2_3_replication.sh

#SBATCH --job-name=beagle_v23_repl
#SBATCH --output=slurm_output/beagle_v2_3/repl_%A_%a.out
#SBATCH --error=slurm_output/beagle_v2_3/repl_%A_%a.err
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
TASKS_FILE=${TASKS_FILE:?TASKS_FILE must point to the frozen replication_tasks.tsv}
RUN_LABEL=${RUN_LABEL:-beagle_v2_3_replication_20260913_v1}

cd "$PROJECT"
[ -s "$TASKS_FILE" ] || { echo "ERROR: missing task file: $TASKS_FILE"; exit 1; }
PLAN_ROOT=$(dirname "$TASKS_FILE")
[ "$(readlink -f "$TASKS_FILE")" = "$(readlink -f "$PLAN_ROOT/replication_tasks.tsv")" ] || {
    echo "ERROR: task file must be the canonical replication_tasks.tsv"
    exit 2
}
[ -s "$PLAN_ROOT/OUTPUT_SHA256SUMS.txt" ] || { echo "ERROR: missing plan checksums"; exit 1; }
[ "$(tr -d '\r\n' < "$PLAN_ROOT/RUN_COMPLETE")" = "complete" ] || {
    echo "ERROR: replication plan is not complete"
    exit 2
}
(cd "$PLAN_ROOT" && sha256sum -c OUTPUT_SHA256SUMS.txt)
ROW_COUNT=$(( $(wc -l < "$TASKS_FILE") - 1 ))
EXPECTED_MAX=$(( ROW_COUNT - 1 ))
[ "$ROW_COUNT" -gt 0 ] || { echo "ERROR: empty replication task plan"; exit 2; }
[ "${SLURM_ARRAY_TASK_MIN:-}" = "0" ] || {
    echo "ERROR: replication array must start at 0"
    exit 2
}
[ "${SLURM_ARRAY_TASK_MAX:-}" = "$EXPECTED_MAX" ] || {
    echo "ERROR: array max ${SLURM_ARRAY_TASK_MAX:-missing} != frozen $EXPECTED_MAX"
    exit 2
}
EXPECTED_HEADER=$'task_id\tconfig\tmodelscale\tniterations\tmaxlr\tseed\tsample\tsample_root\tdataset'
OBSERVED_HEADER=$(sed -n '1p' "$TASKS_FILE")
[ "$OBSERVED_HEADER" = "$EXPECTED_HEADER" ] || {
    echo "ERROR: invalid task header"
    exit 2
}
ROW=$(sed -n "$((TASK + 2))p" "$TASKS_FILE")
[ -n "$ROW" ] || { echo "ERROR: missing task row: $TASK"; exit 2; }
IFS=$'\t' read -r ROW_ID CONFIG MODEL_SCALE ITERATIONS MAX_LR SEED SAMPLE SAMPLE_ROOT DATASET <<< "$ROW"
[ "$ROW_ID" = "$TASK" ] || { echo "ERROR: task id mismatch"; exit 2; }

case "$SAMPLE" in
    P4|P6) ;;
    *) echo "ERROR: invalid sample: $SAMPLE"; exit 2 ;;
esac
case "$SEED" in
    20260916|20260917) ;;
    *) echo "ERROR: invalid replication seed: $SEED"; exit 2 ;;
esac

QUALITY=baseQ0mapQ0
SOURCE_ROOT="$PROJECT/data/experiments/beagle_panel_routing_gp090_20260911_v3/$SAMPLE_ROOT"
RAW="$SOURCE_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY/merged_sorted_gt.vcf.gz"
LEDGER="$PROJECT/data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen"
ROOT="$PROJECT/data/experiments/$RUN_LABEL/${CONFIG}_seed${SEED}/$SAMPLE_ROOT"
SNAPSHOT="$ROOT/code_snapshot/scripts/2_beagle_filtering"
INPUT="$ROOT/output_VCFs/mpileup_multi_bam/$QUALITY"

[ -s "$RAW" ] || { echo "ERROR: missing raw VCF: $RAW"; exit 1; }
[ -s "$RAW.tbi" ] || { echo "ERROR: missing raw VCF index: $RAW.tbi"; exit 1; }
[ -s "$LEDGER/RUN_COMPLETE.json" ] || { echo "ERROR: missing bound reference ledger"; exit 1; }
if [ -e "$ROOT" ]; then
    echo "ERROR: refusing to overwrite: $ROOT"
    exit 2
fi

mkdir -p "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13" "$INPUT"
mkdir -p "$PROJECT/slurm_output/beagle_v2_3"
trap 'date --iso-8601=seconds > "$ROOT/RUN_FAILED.txt"' ERR

cp scripts/2_beagle_filtering/run_beagle.py scripts/2_beagle_filtering/beagle_evidence_router.py "$SNAPSHOT/"
cp scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/"
cp run_slurm/run_beagle_v2_3_replication.sh \
   scripts/postanalyze/beagle_selection_integrity.py \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL.md \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_EXTENDED_MODELSCALE.md \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_MAXLR.md \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_REPLICATION.md \
   "$TASKS_FILE" "$PLAN_ROOT/PROVENANCE.json" \
   "$PLAN_ROOT/OUTPUT_SHA256SUMS.txt" "$PLAN_ROOT/RUN_COMPLETE.json" \
   "$ROOT/code_snapshot/"
find "$ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

ln -s "$RAW" "$INPUT/merged_sorted_gt.vcf.gz"
ln -s "$RAW.tbi" "$INPUT/merged_sorted_gt.vcf.gz.tbi"

PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
"$PYTHON" "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py" --dataset "$DATASET" --section-id 1 --section-root "$ROOT" --reference-ledger-root "$LEDGER" --quality-filter "$QUALITY" --threads 30 --parallel-chromosomes 1 --iterations "$ITERATIONS" --seed "$SEED" --modelscale "$MODEL_SCALE" --maxlr "$MAX_LR" --anchor-score p_nonref --min-anchor-probability 0.90

trap - ERR
echo "Complete: $ROOT"
