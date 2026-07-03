#!/bin/bash
# =============================================================================
# gatk_spot_matrix_dlpfc.sh
# Build a spot×SNV matrix from per-barcode GATK VCFs, per DLPFC section, so it
# can be compared against the strelka2 and in-house pipeline matrices
# (same matrix builder).
#
# GATK already produces one VCF per barcode (no BAM scan needed). Two steps:
#   1. gatk_to_spot_snvs.py — parse each in-tissue per-barcode VCF (1000G-filtered) for
#      SNVs (single-base REF/ALT, GT!=0/0) -> one <barcode>.txt in
#      data/dlpfc/{section}/gatk/spot_snvs/
#   2. run_generate_matrix.py -> data/dlpfc/{section}/matrix/DLPFC_{section}_gatk_germline_6_matrix.pkl
#
# Submit (from project root):
#   cd /data/maiziezhou_lab/leiy4/snv_calling
#   sbatch --array=0-11 scripts/tools/gatk_spot_matrix_dlpfc.sh
# Single section:
#   sbatch --array=0 scripts/tools/gatk_spot_matrix_dlpfc.sh
#
# Filter is set via GATK_SUBDIR below: filtered_by_1000Genome/0 (default, 1000G only) or
# filtered_by_1000Genome_by_neighbor_1/0 (also neighbor>=1), or unfiltered/0 (raw, ~2.3 GB).
# =============================================================================

#SBATCH --job-name=gatk_spot_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=02:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=strelka2/slurm_output/gatk_spot_matrix_%a-%j.out
#SBATCH --error=strelka2/slurm_output/gatk_spot_matrix_%a-%j.err

set -u
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
cd "$PROJECT"
mkdir -p strelka2/slurm_output

SECTIONS=(151507 151508 151509 151510
          151669 151670 151671 151672
          151673 151674 151675 151676)
SECTION_ID=${SECTIONS[$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=baseQ0mapQ0
# Per-barcode VCFs under {section}/gatk/output_VCFs/.
#   filtered_by_1000Genome              = drop variants in the 1000 Genomes panel (always applied)
#   filtered_by_1000Genome_by_neighbor_1 = the above + keep only variants seen in >=1 of the 6
#                                          neighbor spots (much smaller; spatially coherent)
# Default: 1000G only (no neighbor constraint) — dimensionally comparable to strelka2 PASS.
GATK_SUBDIR="filtered_by_1000Genome/0"

if [ -z "${SECTION_ID:-}" ]; then
    echo "ERROR: no section for array index $SLURM_ARRAY_TASK_ID"; exit 1
fi

echo "======================================================"
echo "GATK spot×SNV matrix | section $SECTION_ID | task $SLURM_ARRAY_TASK_ID"
echo "node $SLURMD_NODENAME | subdir $GATK_SUBDIR | start $(date)"
echo "======================================================"

source activate snv_caller

SPOT_DIR="$PROJECT/data/dlpfc/$SECTION_ID/gatk/spot_snvs"

# --- Step 1: per-barcode GATK VCF -> per-spot SNV .txt ---------------------
echo "------ Step 1: GATK VCF -> per-spot SNVs ($(date)) ------"
python scripts/tools/gatk_to_spot_snvs.py \
    --section_id "$SECTION_ID" \
    --gatk-subdir "$GATK_SUBDIR"
rc=$?; [ $rc -ne 0 ] && { echo "ERROR: projection failed (exit $rc)"; exit $rc; }

# --- Step 2: build spot×SNV matrix ----------------------------------------
echo "------ Step 2: matrix generation ($(date)) ------"
python scripts/6_spatial_filter/run_generate_matrix.py \
    --dataset dlpfc \
    --section_id "$SECTION_ID" \
    --quality-filter "$QUALITY_FILTER" \
    --input-dir "$SPOT_DIR" \
    --caller gatk \
    --output-name germline
rc=$?; [ $rc -ne 0 ] && { echo "ERROR: matrix generation failed (exit $rc)"; exit $rc; }

echo "======================================================"
echo "DONE section $SECTION_ID | end $(date)"
echo "Matrix: data/dlpfc/$SECTION_ID/matrix/DLPFC_${SECTION_ID}_gatk_germline_6_matrix.pkl"
echo "======================================================"
