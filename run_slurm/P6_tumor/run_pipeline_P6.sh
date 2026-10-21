#!/bin/bash
# P6_TUMOR full SPARCAL pipeline — UMI dedup/split (step 0) through matrix (step 8).
#
# Usage (submit from project root):
#   sbatch run_slurm/P6_tumor/run_pipeline_P6.sh [start_step] [section_id] [end_step]
# Defaults: start_step=0, section_id=1, end_step=8.
#
# Steps: 0 UMI dedup/split · 1 mpileup · 2 beagle · 3 genotype-shift
#        4 seq-error · 5 NN classifier · 6 single-BAM filter · S6+ molecule recount
#        7 spatial filter (+viz) · 8 4-class SPARCAL matrices
#
#SBATCH --job-name=pipeline_P6
#SBATCH --output=slurm_output/P6_TUMOR/baseQ0mapQ0/pipeline_P6-%j.out
#SBATCH --error=slurm_output/P6_TUMOR/baseQ0mapQ0/pipeline_P6-%j.err
#SBATCH --time=50:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail

START_STEP=${1:-0}
SECTION_ID=${2:-1}
END_STEP=${3:-8}
BASEQ=0; MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
DATASET=P6_TUMOR
CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013"
CALICOST_RUN="${CALICOST_BASE}/P6"
CALICOST_CLONE_DIR="${CALICOST_RUN}/calicost_unsup/clone3_rectangle0_w1.0"
CALICOST_SAMPLE_ID="P6${SECTION_ID}"
TUMOR_PURITY_FILE="${CALICOST_RUN}/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
CLONE_LABELS="${CALICOST_CLONE_DIR}/clone_labels.tsv"
CNV_SEGMENTS="${CALICOST_CLONE_DIR}/cnv_seglevel.tsv"
CLONE_POSTERIOR="${CALICOST_CLONE_DIR}/posterior_clone_probability.npy"
SPOT_UMI_TSV="/data/maiziezhou_lab/leiy4/snv_calling/data/calicost_gate_2026-10-19/spot_umi/${CALICOST_SAMPLE_ID}.tsv"
# Optional alternate output root for every step (the model freeze writes to
# data/freeze_v1/...). Unset = the legacy tree data/P6_tumor/<section>.
SECTION_ROOT=${SECTION_ROOT:-}
SR=(); [ -n "$SECTION_ROOT" ] && SR=(--section-root "$SECTION_ROOT")
OUT_ROOT=${SECTION_ROOT:-data/P6_tumor/${SECTION_ID}}
BUNDLE="${OUT_ROOT}/output_VCFs/spotprofiles/${QUALITY_FILTER}"
# S6+ table lives inside the step-6 bundle it was counted from.
MOLECULE_TABLE="${BUNDLE}/molecule_recount.tsv.gz"

if ! [[ "$START_STEP" =~ ^[0-8]$ && "$END_STEP" =~ ^[0-8]$ ]]; then
    echo "ERROR: start_step and end_step must be integers from 0 through 8"; exit 2
fi
case "$SECTION_ID" in
    1|2) ;;
    *) echo "ERROR: section_id must be 1 or 2 (got: $SECTION_ID)"; exit 2 ;;
esac
if [ "$START_STEP" -gt "$END_STEP" ]; then
    echo "ERROR: start_step ($START_STEP) is greater than end_step ($END_STEP)"; exit 2
fi

mkdir -p slurm_output/P6_TUMOR/${QUALITY_FILTER}
source activate snv_caller 2>/dev/null || true
# Pin the env by absolute path: jobs must not depend on the submitting shell's conda setup.
export PATH=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin:$PATH

echo "=== P6 pipeline | section ${SECTION_ID} | steps ${START_STEP}..${END_STEP} | $(date) ==="

run_step() {                # run_step <step-number> <command...>; aborts on failure
    local step=$1; shift
    local n=${step%%[!0-9]*}
    if [ "${n}" -lt "${START_STEP}" ] || [ "${n}" -gt "${END_STEP}" ]; then
        echo "------ skip step ${step} (requested=${START_STEP}..${END_STEP}) ------"; return 0
    fi
    echo "------ step ${step} start: $(date) ------"
    "$@"; local rc=$?
    if [ ${rc} -ne 0 ]; then echo "ERROR: step ${step} failed (exit ${rc})"; exit ${rc}; fi
    echo "------ step ${step} done:  $(date) ------"
}

# 0. UMI deduplication and per-spot BAM split
run_step 0 bash run_slurm/P6_tumor/0_umidedup_split_P6.sh "${SECTION_ID}"

# 1. mpileup calling
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} \
    --base_quality ${BASEQ} --mapping_quality ${MAPQ} --threads 30 --filter_out_tissue "${SR[@]}"

# 2. beagle
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} --threads 30 "${SR[@]}"

# 3. genotype shifting
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

# 4. sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

# 5. NN classifier (run_supplimentary_models — NOT run_sparcal_net, which has the no_variance bug)
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --model-type neural_network --max-training-samples 90000 "${SR[@]}"

# 6. single-BAM SNV filter
run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset ${DATASET} --section-id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --max-workers 30 --classifier neural_network "${SR[@]}"

# S6+. UB-collapsed molecule recount of this run's step-6 bundle.
run_molecule_recount() {
    if [ -s "$MOLECULE_TABLE" ] && [ "$MOLECULE_TABLE" -nt "$BUNDLE/RUN_COMPLETE" ]; then
        echo "[reuse] S6+ molecule table newer than its bundle: $MOLECULE_TABLE"
        return 0
    fi
    local bam_dir="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep${SECTION_ID}_hg19/P6_Tumor_output/outs/split_BAM"
    python scripts/presence_fix/recount_spot_alleles.py \
        --spotprofile-dir "$BUNDLE" \
        --bam-dir "$bam_dir" --out "$MOLECULE_TABLE" --workers 30
}
run_step 6+ run_molecule_recount

# 7. spatial filter (clone + CNV integration via the corrected hg19 CalicoST run)
run_spatial_filter() {
    for f in "$TUMOR_PURITY_FILE" "$CLONE_LABELS" "$CNV_SEGMENTS" "$CLONE_POSTERIOR" "$SPOT_UMI_TSV" "$MOLECULE_TABLE"; do
        [ -s "$f" ] || { echo "ERROR: required CalicoST output missing/empty: $f"; return 1; }
    done
    python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
        --dataset p6_tumor --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
        --tumor_purity_file "$TUMOR_PURITY_FILE" \
        --clone_labels "$CLONE_LABELS" \
        --cnv_segments "$CNV_SEGMENTS" \
        --clone_posterior "$CLONE_POSTERIOR" \
        --spot_umi_tsv "$SPOT_UMI_TSV" \
        --calicost_sample_id "$CALICOST_SAMPLE_ID" \
        --calicost_gate on --cnv_feature clone_enrichment \
        --molecule_table "$MOLECULE_TABLE" --presence_rule G1 "${SR[@]}" \
        --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0
}
run_step 7 run_spatial_filter

# 7. visualization (best-effort; never blocks the matrix step)
if [ -z "$SECTION_ROOT" ] && [ "${START_STEP}" -le 7 ] && [ "${END_STEP}" -ge 7 ]; then
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset p6_tumor --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
        || echo "WARN: visualization failed (non-fatal)"
fi

# 8. 4-class SPARCAL matrices: 1000G / germline / somatic / merged
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

echo "=== P6 pipeline complete | section ${SECTION_ID} | $(date) ==="
