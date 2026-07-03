#!/bin/bash
# Rerun stages 7 + 7b only for all 12 DLPFC sections.
# Stages 1–6 completed successfully in job 11288684; this picks up from Stage 7.

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
LOG_DIR="slurm_output/DLPFC/${QUALITY_FILTER}"

mkdir -p "${LOG_DIR}"

sbatch \
    --job-name=dlpfc_stage7 \
    --output="${LOG_DIR}/stage7_%a.out" \
    --error="${LOG_DIR}/stage7_%a.err" \
    --array=0-11 \
    --time=4:00:00 \
    --account=maiziezhou_lab_phd_int \
    --partition=interactive \
    --qos=maiziezhou_lab_phd_int \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=4 \
    --mem=64GB \
    --mail-type=END,FAIL \
    --mail-user=yuqi.lei@vanderbilt.edu \
    << 'EOF'
#!/bin/bash
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SECTION_ID=${SECTIONS[$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=baseQ0mapQ0

echo "======================================================"
echo "DLPFC Stage 7 rerun  |  Section: ${SECTION_ID}  |  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  |  Start: $(date)"
echo "======================================================"

source activate snv_caller

run_step() {
    local step=$1; shift
    echo ""
    echo "------ Stage ${step} start: $(date) ------"
    "$@"
    local rc=$?
    if [ $rc -ne 0 ]; then
        echo "ERROR: Stage ${step} failed for section ${SECTION_ID} (exit ${rc})"
        exit $rc
    fi
    echo "------ Stage ${step} done:  $(date) ------"
}

run_step 7 python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset dlpfc \
    --section_id "${SECTION_ID}" \
    --quality_filter "${QUALITY_FILTER}" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 1.5

run_step 7b python scripts/6_spatial_filter/run_generate_matrix.py \
    --dataset dlpfc \
    --section_id "${SECTION_ID}" \
    --quality-filter "${QUALITY_FILTER}" \
    --filter-subdir filtered_snvs \
    --output-name normal

echo ""
echo "======================================================"
echo "Stages 7+7b complete for section ${SECTION_ID}  |  End: $(date)"
echo "======================================================"
EOF
