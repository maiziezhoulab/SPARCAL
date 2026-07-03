#!/bin/bash
# DCIS full SPARCAL pipeline — steps 1-7 + 4-class matrix (step 8).
# Array job: one task per section (task 1 -> dcis1, task 2 -> dcis2).
#
# DCIS has a DUAL section-id convention:
#   steps 1-6 take a NUMERIC section id (1/2) — their output_dir template is
#              data/dcis{section_id} -> data/dcis1 / data/dcis2
#   steps 7-8 take the PREFIXED id (dcis1/dcis2) — the spatial filter's
#              output_base is data/ and it joins the section id directly.
# Both resolve to the SAME data/dcis1 | data/dcis2 tree.
#
# Usage (submit from project root: cd /data/maiziezhou_lab/leiy4/snv_calling):
#   sbatch run_slurm/DCIS/run_pipeline_DCIS.sh            # all steps, both sections
#   sbatch run_slurm/DCIS/run_pipeline_DCIS.sh 8          # only (re)build matrices
#   sbatch --array=1 run_slurm/DCIS/run_pipeline_DCIS.sh  # dcis1 only
#
#SBATCH --job-name=pipeline_DCIS
#SBATCH --output=slurm_output/DCIS/baseQ0mapQ0/pipeline_dcis%a.out
#SBATCH --error=slurm_output/DCIS/baseQ0mapQ0/pipeline_dcis%a.err
#SBATCH --array=1-2
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
N=${SLURM_ARRAY_TASK_ID}    # numeric section id (1 or 2) for steps 1-6
DCIS="dcis${N}"             # prefixed id (dcis1/dcis2) for steps 7-8
SECTION_UPPER="DCIS${N}"    # CalicoST dir
CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"
BEAGLE_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/${DCIS}/output_VCFs/beagle/${QUALITY_FILTER}/all_filtered_in.vcf.gz"

mkdir -p slurm_output/DCIS/${QUALITY_FILTER}
source activate snv_caller

echo "=== DCIS pipeline | section ${DCIS} (numeric ${N}) | start step ${START_STEP} | $(date) ==="

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

# 1. mpileup calling (numeric section id)
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset DCIS --section_id ${N} \
    --base_quality ${BASEQ} --mapping_quality ${MAPQ} --call_mode multi --threads 30 --filter_out_tissue

# 2. beagle
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset DCIS --section_id ${N} --quality-filter ${QUALITY_FILTER} --threads 30 --memory 200g

# 3. genotype shifting
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset DCIS --section_id ${N} --quality_filter ${QUALITY_FILTER}

# 4. sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset DCIS --section_id ${N} --quality_filter ${QUALITY_FILTER}

# 5. NN classifier (run_supplimentary_models — NOT run_sparcal_net)
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset DCIS --section_id ${N} --quality-filter ${QUALITY_FILTER} \
    --model-type neural_network --max-training-samples 90000

# 6. single-BAM SNV filter
run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset DCIS --section-id ${N} --quality-filter ${QUALITY_FILTER} \
    --max-workers 30 --classifier neural_network

# 7. spatial filter (prefixed id; clone + CNV integration via CalicoST DCIS{N})
run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset dcis --section_id ${DCIS} --quality_filter ${QUALITY_FILTER} \
    --tumor_purity_file "${CALICOST_BASE}/${SECTION_UPPER}/estimate_tumor_prop/loh_estimator_tumor_prop.tsv" \
    --clone_labels     "${CALICOST_BASE}/${SECTION_UPPER}/calicost/clone3_rectangle0_w1.0/clone_labels.tsv" \
    --cnv_segments     "${CALICOST_BASE}/${SECTION_UPPER}/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv" \
    --exclude_vcf ${BEAGLE_VCF} --kept_variants ${BEAGLE_VCF} \
    --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0 \
    --germline_threshold 0.5 --somatic_threshold 0.2

# 7. visualization (best-effort; never blocks the matrix step)
if [ "${START_STEP}" -le 7 ]; then
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset dcis --section_id ${DCIS} --quality_filter ${QUALITY_FILTER} \
        || echo "WARN: visualization failed (non-fatal)"
fi

# 8. 4-class SPARCAL matrices: 1000G / germline / somatic / merged  (prefixed id)
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset DCIS --section_id ${DCIS} --quality_filter ${QUALITY_FILTER}

echo "=== DCIS pipeline complete | section ${DCIS} | $(date) ==="
