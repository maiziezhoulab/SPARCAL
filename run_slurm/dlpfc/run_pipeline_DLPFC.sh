#!/bin/bash
# DLPFC full pipeline — one array task per section, all 7 stages run sequentially within each task
#
# Array index → section:
#   0→151507  1→151508  2→151509  3→151510
#   4→151669  5→151670  6→151671  7→151672
#   8→151673  9→151674 10→151675 11→151676
#
# Usage: bash run_slurm/dlpfc/run_pipeline_DLPFC.sh [START_STEP]
#        START_STEP (default 1) = first step to run; earlier steps are skipped
#        e.g. bash run_slurm/dlpfc/run_pipeline_DLPFC.sh 8   # only (re)build matrices
# Run from project root: cd /data/maiziezhou_lab/leiy4/snv_calling
#
# PREREQUISITE: the DLPFC config now reads UMI-DEDUPED BAMs
# (data/dlpfc/{section}/bam_bycell_dedup/*.bam). Run step 0 FIRST, once:
#   sbatch run_slurm/dlpfc/0_umidedup_split_DLPFC.sh
# Step 1 (mpileup) will fail with "No BAM files found" until that completes.
# Matrix output: DLPFC_{section}_SPARCAL_normal_matrix.pkl (normal == all germline).

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
START_STEP=${1:-1}
LOG_DIR="slurm_output/DLPFC/${QUALITY_FILTER}"

# Optional: wait for the dedup array to finish first. Step 1 needs the deduped
# BAMs, so submit chained to job 0's id:
#   DEP=<umidedup_jobid> bash run_slurm/dlpfc/run_pipeline_DLPFC.sh
DEP_ARG=""
[ -n "${DEP:-}" ] && DEP_ARG="--dependency=afterok:${DEP}"

mkdir -p "${LOG_DIR}"

sbatch \
    ${DEP_ARG} \
    --job-name=dlpfc_pipeline \
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
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SECTION_ID=\${SECTIONS[\$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=${QUALITY_FILTER}
START_STEP=${START_STEP}

echo "======================================================"
echo "DLPFC pipeline  |  Section: \${SECTION_ID}  |  Task: \$SLURM_ARRAY_TASK_ID"
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

# Stage 1: mpileup
run_step 1 python scripts/1_calling/mpileup_pipeline.py \
    --dataset DLPFC \
    --section_id "\${SECTION_ID}" \
    --base_quality 0 \
    --mapping_quality 0 \
    --threads 30

# Stage 2: beagle
run_step 2 python scripts/2_beagle_filtering/run_beagle.py \
    --dataset DLPFC \
    --quality-filter "\${QUALITY_FILTER}" \
    --section_id "\${SECTION_ID}" \
    --threads 30

# Stage 3: genotype shifting → germline_defined + germline_denovo
run_step 3 python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py \
    --dataset DLPFC \
    --section_id "\${SECTION_ID}" \
    --quality_filter "\${QUALITY_FILTER}"

# Stage 4: sequence error model
run_step 4 python scripts/3_classifier_prep/run_sequence_error_model.py \
    --dataset DLPFC \
    --section_id "\${SECTION_ID}" \
    --quality_filter "\${QUALITY_FILTER}"

# Stage 5: MLP classifier
run_step 5 python scripts/4_classifier/run_supplimentary_models.py \
    --dataset DLPFC \
    --section_id "\${SECTION_ID}" \
    --quality-filter "\${QUALITY_FILTER}" \
    --model-type neural_network \
    --max-training-samples 90000

# Stage 6: single-BAM SNV filter
run_step 6 python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    --dataset DLPFC \
    --section-id "\${SECTION_ID}" \
    --quality-filter "\${QUALITY_FILTER}" \
    --max-workers 30 \
    --classifier neural_network

# Stage 7: spatial filter + matrix (normal tissue — no tumor purity / CNV)
run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset dlpfc \
    --section_id "\${SECTION_ID}" \
    --quality_filter "\${QUALITY_FILTER}" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 1.5

# Stage 8: SPARCAL normal-tissue matrix -> DLPFC_{section}_SPARCAL_normal_matrix.pkl
# (normal == all germline SNVs; DLPFC is normal brain tissue, no somatic class).
# Replaces the old run_generate_matrix bcftools_normal_6 output.
run_step 8 python scripts/6_spatial_filter/generate_sparcal_matrices.py \
    --dataset DLPFC \
    --section_id "\${SECTION_ID}" \
    --quality_filter "\${QUALITY_FILTER}" \
    --classes normal

echo ""
echo "======================================================"
echo "All stages complete for section \${SECTION_ID}  |  End: \$(date)"
echo "======================================================"
EOF
