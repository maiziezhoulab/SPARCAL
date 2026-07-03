#!/bin/bash
#SBATCH --job-name=final_vcf_P4
#SBATCH --output=slurm_output/P4_TUMOR/baseQ0mapQ0/final_vcf_P4-%j.out
#SBATCH --error=slurm_output/P4_TUMOR/baseQ0mapQ0/final_vcf_P4-%j.err
#SBATCH --time=4:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# =============================================================================
# Step 8 — Generate final VCF outputs for P4_TUMOR (sections 1 and 2)
#
# Output per section:
#   Type 1: final_output/{qf}/final_vcf_per_spot/{barcode}.vcf.gz
#            Per-spot VCFs filtered to somatic variants; full FORMAT preserved.
#   Type 2: final_output/{qf}/final_snv_profile.vcf.gz
#            1000G-style multi-sample VCF; all variant stats in INFO; FORMAT=GT.
#
# Prerequisite: step 7 (run_spatial_snv_filter_enhanced.py) must be complete.
# The spatial filter base dir is detected automatically (enhanced vs basic).
# =============================================================================

set -euo pipefail

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

PROJECT_DIR="/data/maiziezhou_lab/leiy4/snv_calling"
SCRIPT="scripts/postprocess/run_generate_final_vcf.py"

echo "SLURM_JOBID : $SLURM_JOBID"
echo "Hostname    : $(hostname)"
echo "Start time  : $(date)"
echo ""

source activate snv_caller
mkdir -p slurm_output/P4_TUMOR/${QUALITY_FILTER}

# --------------------------------------------------------------------------- #
# SPATIAL_FILTER_DIR: base output directory of the enhanced spatial filter.
# For run_spatial_snv_filter_enhanced.py this is:
#   data/P4_tumor/{section}/spatial_filter_purity/{quality_filter}/
# The script auto-detects the structure and reads somatic/denovo/, somatic/defined/,
# and all_variant_scores.txt from this directory.
# --------------------------------------------------------------------------- #

for SECTION_ID in 1 2; do
    echo "======================================================="
    echo "Processing P4_TUMOR section ${SECTION_ID}"
    echo "Start time: $(date)"

    SPATIAL_FILTER_DIR="${PROJECT_DIR}/data/P4_tumor/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}"

    if [ ! -d "${SPATIAL_FILTER_DIR}" ]; then
        echo "ERROR: Spatial filter directory not found: ${SPATIAL_FILTER_DIR}"
        echo "       Please verify that step 7 completed successfully for section ${SECTION_ID}."
        continue
    fi

    OUTPUT_DIR="${PROJECT_DIR}/data/P4_tumor/${SECTION_ID}/final_output/${QUALITY_FILTER}"
    mkdir -p "${OUTPUT_DIR}"

    python ${SCRIPT} \
        --dataset P4_TUMOR \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER} \
        --spatial_filter_dir ${SPATIAL_FILTER_DIR} \
        --project_dir ${PROJECT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --classification somatic

    if [ $? -eq 0 ]; then
        echo "SUCCESS: P4_TUMOR section ${SECTION_ID} complete."
        echo "  Type 1: ${OUTPUT_DIR}/final_vcf_per_spot/"
        echo "  Type 2: ${OUTPUT_DIR}/final_snv_profile.vcf.gz"
    else
        echo "ERROR: Final VCF generation failed for P4_TUMOR section ${SECTION_ID}."
    fi

    echo "End time: $(date)"
    echo ""
done

echo "All P4_TUMOR sections processed."
echo "End time: $(date)"