#!/bin/bash
# Rerun Stage 7b for all 12 DLPFC sections using the enhanced Stage 7 output.
# Stage 7 (enhanced) completed for all 12 sections; per-barcode .txt files
# exist under spatial_filter_purity/baseQ0mapQ0/germline/.
# The 7 sections that previously had matrices used OLD filtered_snvs data
# (Jul-2025); this reruns all 12 for consistency with the enhanced pipeline.
# run_generate_matrix.py --input-dir bypasses the old spatial_analysis path.

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
LOG_DIR="slurm_output/DLPFC/${QUALITY_FILTER}"

mkdir -p "${LOG_DIR}"

sbatch \
    --job-name=dlpfc_stage7b \
    --output="${LOG_DIR}/stage7b_%a.out" \
    --error="${LOG_DIR}/stage7b_%a.err" \
    --array=0-11 \
    --time=2:00:00 \
    --account=maiziezhou_lab_phd_int \
    --partition=interactive \
    --qos=maiziezhou_lab_phd_int \
    --nodes=1 --ntasks=1 \
    --cpus-per-task=4 \
    --mem=32GB \
    --mail-type=END,FAIL \
    --mail-user=yuqi.lei@vanderbilt.edu \
    << 'EOF'
#!/bin/bash
SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SECTION_ID=${SECTIONS[$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=baseQ0mapQ0
DATA_BASE="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc"

echo "======================================================"
echo "DLPFC Stage 7b (all 12)  |  Section: ${SECTION_ID}  |  Task: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME  |  Start: $(date)"
echo "======================================================"

source activate snv_caller

INPUT_DIR="${DATA_BASE}/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}/germline"

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERROR: germline input dir not found: ${INPUT_DIR}"
    exit 1
fi

echo "Input dir: ${INPUT_DIR}"
echo "------ Stage 7b start: $(date) ------"

python scripts/6_spatial_filter/run_generate_matrix.py \
    --dataset dlpfc \
    --section_id "${SECTION_ID}" \
    --quality-filter "${QUALITY_FILTER}" \
    --input-dir "${INPUT_DIR}" \
    --output-name normal

rc=$?
if [ $rc -ne 0 ]; then
    echo "ERROR: Stage 7b failed for section ${SECTION_ID} (exit ${rc})"
    exit $rc
fi

echo "------ Stage 7b done:  $(date) ------"
echo ""
echo "======================================================"
echo "Stage 7b complete for section ${SECTION_ID}  |  End: $(date)"
echo "======================================================"
EOF
