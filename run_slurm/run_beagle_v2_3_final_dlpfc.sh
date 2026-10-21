#!/bin/bash
# Run the frozen final Beagle-v2.3 representation on the exact 12-section DLPFC grid.
#
# Required submission contract:
# sbatch --array=0-ARRAY_MAX%4 \
#   --export=ALL,TASKS_FILE=/absolute/plan/dlpfc_tasks.tsv,\
#RUN_ROOT=/absolute/fresh/run_root,FINAL_SELECTION_ROOT=/absolute/final_selection,\
#BASE_ASSESSMENT_ROOT=/absolute/base_assessment,\
#REFERENCE_LEDGER_ROOT=/absolute/reference_ledger \
#   run_slurm/run_beagle_v2_3_final_dlpfc.sh

#SBATCH --job-name=beagle_v23_dlpfc
#SBATCH --output=slurm_output/beagle_v2_3/final_dlpfc_%A_%a.out
#SBATCH --error=slurm_output/beagle_v2_3/final_dlpfc_%A_%a.err
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
TASKS_FILE=${TASKS_FILE:?TASKS_FILE must point to frozen dlpfc_tasks.tsv}
RUN_ROOT=${RUN_ROOT:?RUN_ROOT must be the frozen fresh DLPFC run root}
FINAL_SELECTION_ROOT=${FINAL_SELECTION_ROOT:?FINAL_SELECTION_ROOT is required}
BASE_ASSESSMENT_ROOT=${BASE_ASSESSMENT_ROOT:?BASE_ASSESSMENT_ROOT is required}
REFERENCE_LEDGER_ROOT=${REFERENCE_LEDGER_ROOT:?REFERENCE_LEDGER_ROOT is required}
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
QUALITY=baseQ0mapQ0

cd "$PROJECT"
[ -s "$TASKS_FILE" ] || { echo "ERROR: missing DLPFC task file: $TASKS_FILE"; exit 1; }
PLAN_ROOT=$(dirname "$TASKS_FILE")
[ "$(readlink -f "$TASKS_FILE")" = "$(readlink -f "$PLAN_ROOT/dlpfc_tasks.tsv")" ] || {
    echo "ERROR: task file must be the canonical dlpfc_tasks.tsv"
    exit 2
}
"$PYTHON" scripts/postanalyze/plan_beagle_v2_3_final_dlpfc.py \
    --verify-plan-root "$PLAN_ROOT" \
    --expected-run-root "$RUN_ROOT" \
    --expected-final-selection-root "$FINAL_SELECTION_ROOT" \
    --expected-base-assessment-root "$BASE_ASSESSMENT_ROOT" \
    --expected-reference-ledger-root "$REFERENCE_LEDGER_ROOT"

ROW_COUNT=$(( $(wc -l < "$TASKS_FILE") - 1 ))
EXPECTED_MAX=$(( ROW_COUNT - 1 ))
[ "$ROW_COUNT" -eq 12 ] || [ "$ROW_COUNT" -eq 36 ] || {
    echo "ERROR: final DLPFC plan must contain exactly 12 or 36 tasks"
    exit 2
}
[ "${SLURM_ARRAY_TASK_MIN:-}" = "0" ] || {
    echo "ERROR: final DLPFC array must start at 0"
    exit 2
}
[ "${SLURM_ARRAY_TASK_MAX:-}" = "$EXPECTED_MAX" ] || {
    echo "ERROR: array max ${SLURM_ARRAY_TASK_MAX:-missing} != frozen $EXPECTED_MAX"
    exit 2
}
EXPECTED_HEADER=$'task_id\tsection\tseed\tdataset\tmodelscale\tniterations\tmaxlr\tbase_config\trepresentation\tsection_root'
OBSERVED_HEADER=$(sed -n '1p' "$TASKS_FILE")
[ "$OBSERVED_HEADER" = "$EXPECTED_HEADER" ] || {
    echo "ERROR: invalid final DLPFC task header"
    exit 2
}
ROW=$(sed -n "$((TASK + 2))p" "$TASKS_FILE")
[ -n "$ROW" ] || { echo "ERROR: missing DLPFC task row: $TASK"; exit 2; }
IFS=$'\t' read -r ROW_ID SECTION SEED DATASET MODEL_SCALE ITERATIONS MAX_LR BASE_CONFIG REPRESENTATION SECTION_ROOT <<< "$ROW"
[ "$ROW_ID" = "$TASK" ] || { echo "ERROR: DLPFC task id mismatch"; exit 2; }

case "$SECTION" in
    151507|151508|151509|151510|151669|151670|151671|151672|151673|151674|151675|151676) ;;
    *) echo "ERROR: invalid DLPFC section: $SECTION"; exit 2 ;;
esac
[ "$DATASET" = "DLPFC" ] || { echo "ERROR: invalid dataset: $DATASET"; exit 2; }
case "$REPRESENTATION" in
    single_seed)
        [ "$SEED" = "20260915" ] || { echo "ERROR: single-seed plan has wrong seed"; exit 2; }
        ;;
    ensemble_mean|ensemble_median)
        case "$SEED" in
            20260915|20260916|20260917) ;;
            *) echo "ERROR: ensemble plan has invalid seed: $SEED"; exit 2 ;;
        esac
        ;;
    *) echo "ERROR: invalid final representation: $REPRESENTATION"; exit 2 ;;
esac

RUN_ROOT=$(readlink -m "$RUN_ROOT")
EXPECTED_SECTION_ROOT=$(readlink -m "$RUN_ROOT/${BASE_CONFIG}_seed${SEED}/$SECTION")
[ "$(readlink -m "$SECTION_ROOT")" = "$EXPECTED_SECTION_ROOT" ] || {
    echo "ERROR: task section_root is not the canonical path under RUN_ROOT"
    exit 2
}
ROOT="$EXPECTED_SECTION_ROOT"
RAW="$PROJECT/data/dlpfc/$SECTION/output_VCFs/mpileup_multi_bam/$QUALITY/merged_sorted_gt.vcf.gz"
INPUT="$ROOT/output_VCFs/mpileup_multi_bam/$QUALITY"
SNAPSHOT="$ROOT/code_snapshot"

[ -s "$RAW" ] || { echo "ERROR: missing canonical DLPFC raw VCF: $RAW"; exit 1; }
[ -s "$RAW.tbi" ] || { echo "ERROR: missing canonical DLPFC raw index: $RAW.tbi"; exit 1; }
[ -s "$REFERENCE_LEDGER_ROOT/RUN_COMPLETE.json" ] || {
    echo "ERROR: missing selected reference ledger"
    exit 1
}
if [ -e "$ROOT" ]; then
    echo "ERROR: refusing to overwrite final DLPFC task root: $ROOT"
    exit 2
fi

mkdir -p "$SNAPSHOT/scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13" \
         "$SNAPSHOT/final_selection" "$SNAPSHOT/base_assessment" "$INPUT" \
         "$PROJECT/slurm_output/beagle_v2_3"
trap 'date --iso-8601=seconds > "$ROOT/RUN_FAILED.txt"' ERR

cp scripts/2_beagle_filtering/run_beagle.py \
   scripts/2_beagle_filtering/beagle_evidence_router.py \
   "$SNAPSHOT/scripts/2_beagle_filtering/"
cp scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/BENCHMARK_PROTOCOL_DLPFC_FINAL.md \
   "$SNAPSHOT/scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/"
cp run_slurm/run_beagle_v2_3_final_dlpfc.sh \
   scripts/postanalyze/plan_beagle_v2_3_final_dlpfc.py \
   scripts/postanalyze/beagle_selection_integrity.py \
   "$TASKS_FILE" "$PLAN_ROOT/PROVENANCE.json" \
   "$PLAN_ROOT/OUTPUT_SHA256SUMS.txt" "$PLAN_ROOT/RUN_COMPLETE" \
   "$PLAN_ROOT/RUN_COMPLETE.json" "$SNAPSHOT/"
cp "$FINAL_SELECTION_ROOT/ensemble_gate_results.tsv" \
   "$FINAL_SELECTION_ROOT/FINAL_BEAGLE_REPRESENTATION.json" \
   "$FINAL_SELECTION_ROOT/PROVENANCE.json" \
   "$FINAL_SELECTION_ROOT/CODE_AND_INPUT_SHA256SUMS.txt" \
   "$FINAL_SELECTION_ROOT/OUTPUT_SHA256SUMS.txt" \
   "$FINAL_SELECTION_ROOT/RUN_COMPLETE" \
   "$FINAL_SELECTION_ROOT/RUN_COMPLETE.json" \
   "$SNAPSHOT/final_selection/"
cp "$BASE_ASSESSMENT_ROOT/replication_gate_results.tsv" \
   "$BASE_ASSESSMENT_ROOT/FINAL_BASE_CONFIGURATION.json" \
   "$BASE_ASSESSMENT_ROOT/PROVENANCE.json" \
   "$BASE_ASSESSMENT_ROOT/CODE_AND_INPUT_SHA256SUMS.txt" \
   "$BASE_ASSESSMENT_ROOT/OUTPUT_SHA256SUMS.txt" \
   "$BASE_ASSESSMENT_ROOT/RUN_COMPLETE" \
   "$BASE_ASSESSMENT_ROOT/RUN_COMPLETE.json" \
   "$SNAPSHOT/base_assessment/"
find "$SNAPSHOT" -type f -print0 | sort -z | xargs -0 sha256sum > "$ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

ln -s "$RAW" "$INPUT/merged_sorted_gt.vcf.gz"
ln -s "$RAW.tbi" "$INPUT/merged_sorted_gt.vcf.gz.tbi"

"$PYTHON" "$SNAPSHOT/scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py" \
    --dataset "$DATASET" \
    --section-id "$SECTION" \
    --section-root "$ROOT" \
    --reference-ledger-root "$REFERENCE_LEDGER_ROOT" \
    --quality-filter "$QUALITY" \
    --threads 30 \
    --memory 10g \
    --parallel-chromosomes 1 \
    --iterations "$ITERATIONS" \
    --seed "$SEED" \
    --modelscale "$MODEL_SCALE" \
    --maxlr "$MAX_LR" \
    --anchor-score p_nonref \
    --min-anchor-probability 0.90

trap - ERR
echo "Complete: $ROOT"
