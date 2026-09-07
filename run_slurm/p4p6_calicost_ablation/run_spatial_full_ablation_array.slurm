#!/bin/bash
#SBATCH --job-name=p4p6_spatial_ablation
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p4p6_calicost_ablation/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p4p6_calicost_ablation/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
CALICOST_ROOT=/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
QUALITY_FILTER=baseQ0mapQ0
FULL_TAG=baseQ0mapQ0_full_calicost
ABLATION_TAG=baseQ0mapQ0_ablation_no_calicost

case "${SLURM_ARRAY_TASK_ID}" in
    0) COHORT=P4; SECTION_ID=1; CLONE_REQUEST=2 ;;
    1) COHORT=P4; SECTION_ID=2; CLONE_REQUEST=2 ;;
    2) COHORT=P6; SECTION_ID=1; CLONE_REQUEST=3 ;;
    3) COHORT=P6; SECTION_ID=2; CLONE_REQUEST=3 ;;
    *) echo "ERROR: unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

dataset="${COHORT,,}_tumor"
section_root="$PROJECT_DIR/data/${COHORT}_tumor/${SECTION_ID}"
calicost_run="$CALICOST_ROOT/${COHORT}_sec${SECTION_ID}"
clone_dir="$calicost_run/calicost/clone${CLONE_REQUEST}_rectangle0_w1.0"
purity_file="$calicost_run/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
clone_labels="$clone_dir/clone_labels.tsv"
cnv_segments="$clone_dir/cnv_seglevel.tsv"
if [[ ! -s "$cnv_segments" ]]; then
    cnv_segments="$clone_dir/cnv_diploid_seglevel.tsv"
fi

for required in "$PYTHON" "$purity_file" "$clone_labels" "$cnv_segments"; do
    [[ -s "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

full_out="$section_root/spatial_filter_purity/$FULL_TAG"
ablation_out="$section_root/spatial_filter_purity/$ABLATION_TAG"
mkdir -p "$PROJECT_DIR/slurm_output/p4p6_calicost_ablation"

echo "[$(date --iso-8601=seconds)] ${COHORT} rep${SECTION_ID}: full CalicoST/purity spatial analysis"
"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py" \
    --dataset "$dataset" \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --tumor_purity_file "$purity_file" \
    --clone_labels "$clone_labels" \
    --cnv_segments "$cnv_segments" \
    --evidence_mode full \
    --output_dir "$full_out" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 2.0

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/visualize_spatial_filter.py" \
    --dataset "$dataset" \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --output_dir "$full_out"

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/generate_sparcal_matrices.py" \
    --dataset "${COHORT}_TUMOR" \
    --section_id "$SECTION_ID" \
    --quality_filter "$FULL_TAG" \
    --model SPARCAL_full_calicost

echo "[$(date --iso-8601=seconds)] ${COHORT} rep${SECTION_ID}: spatial-only no-CalicoST ablation"
"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py" \
    --dataset "$dataset" \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --evidence_mode spatial_only_no_calicost \
    --output_dir "$ablation_out" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 2.0

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/visualize_spatial_filter.py" \
    --dataset "$dataset" \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --output_dir "$ablation_out"

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/generate_sparcal_matrices.py" \
    --dataset "${COHORT}_TUMOR" \
    --section_id "$SECTION_ID" \
    --quality_filter "$ABLATION_TAG" \
    --model SPARCAL_ablation_no_calicost

test -s "$full_out/spatial_filter_summary.txt"
test -s "$ablation_out/spatial_filter_summary.txt"
test -s "$section_root/matrix/${COHORT}_TUMOR_${SECTION_ID}_SPARCAL_full_calicost_somatic_matrix.pkl"
test -s "$section_root/matrix/${COHORT}_TUMOR_${SECTION_ID}_SPARCAL_ablation_no_calicost_somatic_matrix.pkl"
echo "[$(date --iso-8601=seconds)] ${COHORT} rep${SECTION_ID}: complete"
