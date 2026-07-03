#!/bin/bash
# P6_TUMOR full SPARCAL pipeline — steps 1-7 + 4-class matrix (step 8), one job.
#
# Usage (submit from project root: cd /data/maiziezhou_lab/leiy4/snv_calling):
#   sbatch run_slurm/P6_tumor/run_pipeline_P6.sh            # run all steps (1..8)
#   sbatch run_slurm/P6_tumor/run_pipeline_P6.sh 5          # resume from step 5
#   sbatch run_slurm/P6_tumor/run_pipeline_P6.sh 8          # only (re)build matrices
#
# Steps: 1 mpileup · 2 beagle · 3 genotype-shift · 4 seq-error · 5 NN classifier
#        6 single-BAM filter · 7 spatial filter (+viz) · 8 4-class SPARCAL matrices
#
#SBATCH --job-name=pipeline_P6
#SBATCH --output=slurm_output/P6_TUMOR/baseQ0mapQ0/pipeline_P6.out
#SBATCH --error=slurm_output/P6_TUMOR/baseQ0mapQ0/pipeline_P6.err
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

START_STEP=${1:-1}          # first step to run (1..8); earlier steps are skipped
BASEQ=0; MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=1                # P6 dedup ablation: replicate 1 only
DATASET=P6_TUMOR
CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"

mkdir -p slurm_output/P6_TUMOR/${QUALITY_FILTER}
source activate snv_caller

echo "=== P6 pipeline | section ${SECTION_ID} | start step ${START_STEP} | $(date) ==="

run_step() {                # run_step <step-number> <command...>; aborts on failure
    local step=$1; shift
    local n=${step%%[!0-9]*}
    if [ "${n}" -lt "${START_STEP}" ]; then
        echo "------ skip step ${step} (start=${START_STEP}) ------"; return 0
    fi
    echo "------ step ${step} start: $(date) ------"
    "$@"; local rc=$?
    if [ ${rc} -ne 0 ]; then echo "ERROR: step ${step} failed (exit ${rc})"; exit ${rc}; fi
    echo "------ step ${step} done:  $(date) ------"
}

# 1. mpileup calling
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} \
    --base_quality ${BASEQ} --mapping_quality ${MAPQ} --threads 30 --filter_out_tissue

# 2. beagle
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} --threads 30

# 3. genotype shifting
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

# 4. sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

# 5. NN classifier (run_supplimentary_models — NOT run_sparcal_net, which has the no_variance bug)
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --model-type neural_network --max-training-samples 90000

# 6. single-BAM SNV filter
run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset ${DATASET} --section-id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --max-workers 30 --classifier neural_network

# 7. spatial filter (clone + CNV integration via CalicoST P6_sec1)
run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset p6_tumor --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
    --tumor_purity_file "${CALICOST_BASE}/P6_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv" \
    --clone_labels     "${CALICOST_BASE}/P6_sec1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv" \
    --cnv_segments     "${CALICOST_BASE}/P6_sec1/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv" \
    --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0

# 7. visualization (best-effort; never blocks the matrix step)
if [ "${START_STEP}" -le 7 ]; then
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset p6_tumor --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
        || echo "WARN: visualization failed (non-fatal)"
fi

# 8. 4-class SPARCAL matrices: 1000G / germline / somatic / merged
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

echo "=== P6 pipeline complete | section ${SECTION_ID} | $(date) ==="
