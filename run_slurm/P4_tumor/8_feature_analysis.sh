#!/bin/bash
#SBATCH --job-name=feature_analysis
#SBATCH --output=slurm_output/feature_analysis/feature_analysis-%j.out
#SBATCH --error=slurm_output/feature_analysis/feature_analysis-%j.err
#SBATCH --time=1:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# ─────────────────────────────────────────────────────────────────────────────
# Feature analysis for SPARCAL spatial SNV filter
# Runs feature_analysis.py on all_variant_scores.txt produced by
# run_spatial_filter_enhanced.py for each dataset / section.
#
# Datasets configured below:
#   P4_TUMOR  section 1   (active — submitted now)
#   P6_TUMOR  section 1   (commented out — uncomment to run)
#   DCIS      section 1   (commented out — uncomment to run)
#   DCIS      section 2   (commented out — uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────

echo "SLURM_JOBID : $SLURM_JOBID"
echo "Start time  : $(date)"

source activate snv_caller

mkdir -p slurm_output/feature_analysis

GERMLINE_THRESHOLD=0.3
SOMATIC_THRESHOLD=0.2
QUALITY_FILTER="baseQ0mapQ0"

SCRIPT="scripts/6_spatial_filter/feature_analysis.py"

# Helper function — runs feature_analysis.py for one dataset/section and
# reports success/failure without stopping the whole job on a single failure.
run_feature_analysis() {
    local DATASET="$1"        # human-readable label used in titles and logs
    local INPUT_FILE="$2"     # path to all_variant_scores.txt
    local OUTPUT_DIR="$3"     # where plots and feature_summary.tsv go
    local TITLE="$4"          # plot title prefix

    echo ""
    echo "========================================================"
    echo "Dataset : ${DATASET}"
    echo "Input   : ${INPUT_FILE}"
    echo "Output  : ${OUTPUT_DIR}"
    echo "Start   : $(date)"
    echo "========================================================"

    if [ ! -f "${INPUT_FILE}" ]; then
        echo "ERROR: Input file not found: ${INPUT_FILE}"
        echo "  → Has run_spatial_filter_enhanced.py been run for this dataset/section?"
        return 1
    fi

    mkdir -p "${OUTPUT_DIR}"

    python "${SCRIPT}" \
        --input               "${INPUT_FILE}" \
        --output_dir          "${OUTPUT_DIR}" \
        --title               "${TITLE}" \
        --germline_threshold  "${GERMLINE_THRESHOLD}" \
        --somatic_threshold   "${SOMATIC_THRESHOLD}"

    if [ $? -eq 0 ]; then
        echo "SUCCESS: Feature analysis for ${DATASET} completed — $(date)"
    else
        echo "ERROR: Feature analysis failed for ${DATASET} — $(date)"
        return 1
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# P4 TUMOR — Section 1  (ACTIVE)
# ─────────────────────────────────────────────────────────────────────────────
P4_BASE="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor"

run_feature_analysis \
    "P4_TUMOR_sec1" \
    "${P4_BASE}/1/spatial_filter_purity/${QUALITY_FILTER}/all_variant_scores.txt" \
    "${P4_BASE}/1/spatial_filter_purity/${QUALITY_FILTER}/feature_analysis" \
    "P4 Tumor Section 1 (${QUALITY_FILTER})"

# ─────────────────────────────────────────────────────────────────────────────
# P6 TUMOR — Section 1  (commented out — uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────
# P6_BASE="/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor"
#
# run_feature_analysis \
#     "P6_TUMOR_sec1" \
#     "${P6_BASE}/1/spatial_filter_purity/${QUALITY_FILTER}/all_variant_scores.txt" \
#     "${P6_BASE}/1/spatial_filter_purity/${QUALITY_FILTER}/feature_analysis" \
#     "P6 Tumor Section 1 (${QUALITY_FILTER})"

# ─────────────────────────────────────────────────────────────────────────────
# DCIS — Section 1  (commented out — uncomment to run)
# Note: DCIS uses hg38 with chr-prefixed chromosomes; quality filter may differ.
# ─────────────────────────────────────────────────────────────────────────────
# DCIS_BASE="/data/maiziezhou_lab/leiy4/snv_calling/data/DCIS"
# DCIS_QUALITY_FILTER="baseQ0mapQ0"   # update if DCIS was run with different QF
#
# run_feature_analysis \
#     "DCIS_sec1" \
#     "${DCIS_BASE}/1/spatial_filter_purity/${DCIS_QUALITY_FILTER}/all_variant_scores.txt" \
#     "${DCIS_BASE}/1/spatial_filter_purity/${DCIS_QUALITY_FILTER}/feature_analysis" \
#     "DCIS Section 1 (${DCIS_QUALITY_FILTER})"

# ─────────────────────────────────────────────────────────────────────────────
# DCIS — Section 2  (commented out — uncomment to run)
# ─────────────────────────────────────────────────────────────────────────────
# run_feature_analysis \
#     "DCIS_sec2" \
#     "${DCIS_BASE}/2/spatial_filter_purity/${DCIS_QUALITY_FILTER}/all_variant_scores.txt" \
#     "${DCIS_BASE}/2/spatial_filter_purity/${DCIS_QUALITY_FILTER}/feature_analysis" \
#     "DCIS Section 2 (${DCIS_QUALITY_FILTER})"

echo ""
echo "All feature analysis jobs complete."
echo "End time: $(date)"