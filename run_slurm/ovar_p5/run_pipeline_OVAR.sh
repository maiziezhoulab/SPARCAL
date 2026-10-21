#!/bin/bash
# OVAR_P5 full SPARCAL pipeline — UMI dedup/split (step 0) through matrix (step 8).
#
# Usage (submit from project root):
#   sbatch run_slurm/ovar_p5/run_pipeline_OVAR.sh [start_step] [end_step]
# Defaults: start_step=0, end_step=8. Single section P5_sr13.
# Optional: export SECTION_ROOT=<dir> to write every step's output to a new root
# (the model freeze uses data/freeze_v1/ovar_p5/P5_sr13).
#
# Steps: 0 UMI dedup/split · 1 mpileup · 2 beagle · 3 genotype-shift
#        4 seq-error · 5 NN classifier · 6 single-BAM filter · S6+ molecule recount
#        7 spatial filter · 8 4-class SPARCAL matrices
#
#SBATCH --job-name=pipeline_OVAR
#SBATCH --output=slurm_output/OVAR_P5/pipeline_OVAR-%j.out
#SBATCH --error=slurm_output/OVAR_P5/pipeline_OVAR-%j.err
#SBATCH --time=50:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail

START_STEP=${1:-0}
END_STEP=${2:-8}
SECTION_ID=P5_sr13
BASEQ=0; MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
DATASET=OVAR_P5
# Colleague CalicoST run: single section (unsuffixed labels), 2 realized clones, no purity file.
CALICOST_CLONE_DIR="/data/maiziezhou_lab/Pankaj/calicost_p5/results/P5_calicost_cna/clone3_rectangle0_w1.0"
CLONE_LABELS="${CALICOST_CLONE_DIR}/clone_labels.tsv"
CNV_SEGMENTS="${CALICOST_CLONE_DIR}/cnv_seglevel.tsv"
CLONE_POSTERIOR="${CALICOST_CLONE_DIR}/posterior_clone_probability.npy"
SPOT_UMI_TSV="/data/maiziezhou_lab/leiy4/snv_calling/data/calicost_gate_2026-10-19/spot_umi/OVAR_P5.tsv"
SPLIT_DEDUP="data/ovar_p5/${SECTION_ID}/split_BAM_dedup"

SECTION_ROOT=${SECTION_ROOT:-}
SR=(); [ -n "$SECTION_ROOT" ] && SR=(--section-root "$SECTION_ROOT")
OUT_ROOT=${SECTION_ROOT:-data/ovar_p5/${SECTION_ID}}
BUNDLE="${OUT_ROOT}/output_VCFs/spotprofiles/${QUALITY_FILTER}"
MOLECULE_TABLE="${BUNDLE}/molecule_recount.tsv.gz"

if ! [[ "$START_STEP" =~ ^[0-8]$ && "$END_STEP" =~ ^[0-8]$ ]]; then
    echo "ERROR: start_step and end_step must be integers from 0 through 8"; exit 2
fi
if [ "$START_STEP" -gt "$END_STEP" ]; then
    echo "ERROR: start_step ($START_STEP) is greater than end_step ($END_STEP)"; exit 2
fi

mkdir -p slurm_output/OVAR_P5
source activate snv_caller 2>/dev/null || true
# Pin the env by absolute path: jobs must not depend on the submitting shell's conda setup.
export PATH=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin:$PATH

echo "=== OVAR_P5 pipeline | steps ${START_STEP}..${END_STEP} | root ${OUT_ROOT} | $(date) ==="

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

# 0. UMI deduplication and per-spot BAM split (into split_BAM_dedup/)
run_step 0 bash run_slurm/ovar_p5/0_umidedup_split_OVAR.sh

# 1. mpileup calling
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} \
    --base_quality ${BASEQ} --mapping_quality ${MAPQ} --threads 30 --filter_out_tissue "${SR[@]}"

# 2. beagle (frozen production interface, p_nonref anchors)
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} --threads 30 "${SR[@]}"

# 3. genotype shifting
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

# 4. sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

# 5. NN classifier (run_supplimentary_models — NOT run_sparcal_net)
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
    python scripts/presence_fix/recount_spot_alleles.py \
        --spotprofile-dir "$BUNDLE" \
        --bam-dir "$SPLIT_DEDUP" --out "$MOLECULE_TABLE" --workers 30
}
run_step 6+ run_molecule_recount

# 7. spatial filter: CNV gate on (single-section labels -> sample id NONE); no purity file exists.
run_spatial_filter() {
    for f in "$CLONE_LABELS" "$CNV_SEGMENTS" "$CLONE_POSTERIOR" "$SPOT_UMI_TSV" "$MOLECULE_TABLE"; do
        [ -s "$f" ] || { echo "ERROR: required input missing/empty: $f"; return 1; }
    done
    python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
        --dataset ovar_p5 --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
        --clone_labels "$CLONE_LABELS" \
        --cnv_segments "$CNV_SEGMENTS" \
        --clone_posterior "$CLONE_POSTERIOR" \
        --spot_umi_tsv "$SPOT_UMI_TSV" \
        --calicost_sample_id NONE \
        --calicost_gate on --cnv_feature clone_enrichment \
        --molecule_table "$MOLECULE_TABLE" --presence_rule G1 "${SR[@]}" \
        --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0 \
        --germline_threshold 0.5 --somatic_threshold 0.2
}
run_step 7 run_spatial_filter

# 8. 4-class SPARCAL matrices: 1000G / germline / somatic / merged
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} "${SR[@]}"

echo "=== OVAR_P5 pipeline complete | $(date) ==="
