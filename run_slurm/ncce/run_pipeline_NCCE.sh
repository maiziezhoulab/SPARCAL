#!/usr/bin/env bash
# NCCE full SPARCAL pipeline (steps 1-8) for one sample per array task.
#
# Source inputs stay read-only under ST_CNV/.  All generated files live in
# data/ncce/<sample>/, including annotation metadata, split BAMs, VCFs, spatial
# filter outputs, and the four final SPARCAL matrices.
#
# Before this launcher, run 0_split_bam_NCCE.sh (or use submit_NCCE.sh).
# Resume from one step: sbatch --array=9 run_slurm/ncce/run_pipeline_NCCE.sh 5
#
#SBATCH --job-name=pipeline_NCCE
#SBATCH --output=slurm_output/NCCE/baseQ0mapQ0/pipeline_%A_%a.out
#SBATCH --error=slurm_output/NCCE/baseQ0mapQ0/pipeline_%A_%a.err
#SBATCH --array=0-14%2
#SBATCH --time=60:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB

set -euo pipefail

START_STEP=${1:-1}
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SAMPLES=(6A 6B 6C 8A 8B 8C 10A 10B 10C 11A 11B 11C 14A 14B 14C)
SECTION_ID=${SAMPLES[${SLURM_ARRAY_TASK_ID}]}
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
DATASET=NCCE
SECTION_ROOT=${PROJECT}/data/ncce/${SECTION_ID}
TUMOR_LABELS=${SECTION_ROOT}/annotations/copykat_tumor_labels.tsv

cd "${PROJECT}"
# Compute nodes do not expose the login-node `activate` helper.  Initialize
# Conda explicitly so the pinned project environment is available everywhere.
source /panfs/accrepfs.vampire/accre/software/easybuild/software/2023/x86-64-v3/Core/miniconda3/23.9.0-0/etc/profile.d/conda.sh
conda activate /data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller
mkdir -p slurm_output/NCCE/${QUALITY_FILTER}

shopt -s nullglob
SPLIT_BAMS=("${SECTION_ROOT}/split_BAM"/*.bam)
if (( ${#SPLIT_BAMS[@]} == 0 )); then
    echo "ERROR: NCCE ${SECTION_ID} has no split BAMs. Run step 0 first."
    exit 1
fi
if [[ ! -s "${TUMOR_LABELS}" ]]; then
    echo "ERROR: missing copyKAT labels: ${TUMOR_LABELS}"
    exit 1
fi

echo "=== NCCE SPARCAL | sample ${SECTION_ID} | start step ${START_STEP} | $(date) ==="

run_step() {
    local step=$1
    shift
    if (( step < START_STEP )); then
        echo "------ skip step ${step} (start=${START_STEP}) ------"
        return 0
    fi
    echo "------ step ${step} start: $(date) ------"
    "$@"
    echo "------ step ${step} done:  $(date) ------"
}

run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} \
    --base_quality ${BASEQ} --mapping_quality ${MAPQ} --call_mode multi \
    --threads 30 --filter_out_tissue

run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} \
    --quality-filter ${QUALITY_FILTER} --threads 30 --memory 200g

run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

# Use the maintained classifier rather than run_sparcal_net.py (the latter has
# a label-encoder no-variance bug documented in CLAUDE.md).
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --model-type neural_network --max-training-samples 90000

run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset ${DATASET} --section-id ${SECTION_ID} --quality-filter ${QUALITY_FILTER} \
    --max-workers 30 --classifier neural_network

# copyKAT is a binary tumor annotation, not a CNV segmentation.  It is supplied
# as hard tumor probability; clone/CNV arguments are intentionally omitted.
run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset ncce --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER} \
    --tumor_purity_file ${TUMOR_LABELS} \
    --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0 \
    --germline_threshold 0.5 --somatic_threshold 0.2

run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset ${DATASET} --section_id ${SECTION_ID} --quality_filter ${QUALITY_FILTER}

echo "=== NCCE SPARCAL complete | sample ${SECTION_ID} | $(date) ==="
