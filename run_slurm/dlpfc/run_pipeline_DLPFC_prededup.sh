#!/bin/bash
# DLPFC PRE-DEDUP regeneration pipeline — one array task per section, steps 1-8.
#
# WHY: the original pre-dedup DLPFC SPARCAL matrices were overwritten in place by
# the 2026-07-07 UMI-dedup rerun. This script REBUILDS them from the ORIGINAL
# non-dedup per-cell BAMs into a SEPARATE tree so a fair pre-vs-post-dedup
# clustering comparison (10 runs each) can be made.
#
# ISOLATION: exports DLPFC_PREDEDUP=1, which flips ONLY the DLPFC paths in every
# pipeline script:
#   * BAMs  <- /data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/{section}/bam_bycell/*.bam
#             (the read-only ORIGINAL non-dedup source)
#   * output -> data/dlpfc_prededup/{section}/...   (post-dedup data/dlpfc is NEVER touched)
#
# Array index -> section:
#   0->151507  1->151508  2->151509  3->151510
#   4->151669  5->151670  6->151671  7->151672
#   8->151673  9->151674 10->151675 11->151676
#
# Usage (from project root):  bash run_slurm/dlpfc/run_pipeline_DLPFC_prededup.sh [START_STEP]
#   START_STEP (default 1) = first step to run; earlier steps skipped.
#   e.g. bash run_slurm/dlpfc/run_pipeline_DLPFC_prededup.sh 8   # only (re)build matrices
# NO step-0 dedup needed — this reads the non-dedup source directly.

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
START_STEP=${1:-1}
LOG_DIR="slurm_output/DLPFC_prededup/${QUALITY_FILTER}"

mkdir -p "${LOG_DIR}"

sbatch \
    --job-name=dlpfc_prededup \
    --output="${LOG_DIR}/pipeline_%a.out" \
    --error="${LOG_DIR}/pipeline_%a.err" \
    --array=0-11 \
    --time=50:00:00 \
    --account=maiziezhou_lab_phd_int \
    --partition=interactive \
    --qos=maiziezhou_lab_phd_int \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=30 \
    --mem=400GB \
    --mail-type=END,FAIL \
    --mail-user=yuqi.lei@vanderbilt.edu \
    << EOF
#!/bin/bash
export DLPFC_PREDEDUP=1          # <-- flips DLPFC paths to the pre-dedup tree
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SECTION_ID=\${SECTIONS[\$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=${QUALITY_FILTER}
START_STEP=${START_STEP}

echo "======================================================"
echo "DLPFC PRE-DEDUP pipeline  |  Section: \${SECTION_ID}  |  Task: \$SLURM_ARRAY_TASK_ID"
echo "DLPFC_PREDEDUP=\${DLPFC_PREDEDUP}  (output -> data/dlpfc_prededup)"
echo "Node: \$SLURMD_NODENAME  |  Start: \$(date)"
echo "======================================================"

source activate snv_caller

run_step() {
    local step=\$1; shift
    local n=\${step%%[!0-9]*}
    if [ "\${n}" -lt "\${START_STEP}" ]; then
        echo "------ skip Stage \${step} (start=\${START_STEP}) ------"
        return 0
    fi
    echo ""
    echo "------ Stage \${step} start: \$(date) ------"
    "\$@"
    local rc=\$?
    if [ \$rc -ne 0 ]; then
        echo "ERROR: Stage \${step} failed for section \${SECTION_ID} (exit \${rc})"
        exit \$rc
    fi
    echo "------ Stage \${step} done:  \$(date) ------"
}

# Stage 1: mpileup (reads ORIGINAL non-dedup bam_bycell)
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset DLPFC --section_id "\${SECTION_ID}" \
    --base_quality 0 --mapping_quality 0 --threads 30

# Stage 2: beagle
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset DLPFC --quality-filter "\${QUALITY_FILTER}" \
    --section_id "\${SECTION_ID}" --threads 30

# Stage 3: genotype shifting
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset DLPFC --section_id "\${SECTION_ID}" --quality_filter "\${QUALITY_FILTER}"

# Stage 4: sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset DLPFC --section_id "\${SECTION_ID}" --quality_filter "\${QUALITY_FILTER}"

# Stage 5: MLP classifier
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset DLPFC --section_id "\${SECTION_ID}" --quality-filter "\${QUALITY_FILTER}" \
    --model-type neural_network --max-training-samples 90000

# Stage 6: single-BAM SNV filter (reads ORIGINAL non-dedup bam_bycell)
run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset DLPFC --section-id "\${SECTION_ID}" --quality-filter "\${QUALITY_FILTER}" \
    --max-workers 30 --classifier neural_network

# Stage 7: spatial filter (normal tissue)
run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset dlpfc --section_id "\${SECTION_ID}" --quality_filter "\${QUALITY_FILTER}" \
    --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 1.5

# Stage 8: SPARCAL normal-tissue matrix -> data/dlpfc_prededup/{section}/matrix/
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset DLPFC --section_id "\${SECTION_ID}" --quality_filter "\${QUALITY_FILTER}" \
    --classes normal

echo ""
echo "======================================================"
echo "PRE-DEDUP done for section \${SECTION_ID}  |  End: \$(date)"
echo "  matrix -> data/dlpfc_prededup/\${SECTION_ID}/matrix/DLPFC_\${SECTION_ID}_SPARCAL_normal_matrix.pkl"
echo "======================================================"
EOF
