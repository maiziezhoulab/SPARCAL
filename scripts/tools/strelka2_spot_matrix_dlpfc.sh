#!/bin/bash
# =============================================================================
# strelka2_spot_matrix_dlpfc.sh
# Build a spot×SNV matrix from strelka2 germline calls, per DLPFC section, so it
# can be compared against the in-house pipeline's matrices (same matrix builder).
#
# Two steps per section:
#   1. strelka2_to_spot_snvs.py  — scan merged BAM (CB tag) at strelka2 PASS-SNV
#      positions -> one <barcode>.txt per in-tissue spot in
#      data/dlpfc/{section}/strelka2/spot_snvs/
#      Presence rule: >= MIN_ALT_READS reads carrying the strelka2 ALT base
#      (allele-aware), PASS SNVs only.
#   2. run_generate_matrix.py    — per-barcode .txt -> spot×SNV .pkl in
#      data/dlpfc/{section}/matrix/DLPFC_{section}_strelka2_germline_6_matrix.pkl
#
# Submit (from project root):
#   cd /data/maiziezhou_lab/leiy4/snv_calling
#   sbatch --array=0-11 scripts/tools/strelka2_spot_matrix_dlpfc.sh
# Single section (e.g. 151507):
#   sbatch --array=0 scripts/tools/strelka2_spot_matrix_dlpfc.sh
# =============================================================================

#SBATCH --job-name=strelka2_spot_matrix
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=64GB
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=strelka2/slurm_output/spot_matrix_%a-%j.out
#SBATCH --error=strelka2/slurm_output/spot_matrix_%a-%j.err

set -u
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
cd "$PROJECT"
mkdir -p strelka2/slurm_output

SECTIONS=(151507 151508 151509 151510
          151669 151670 151671 151672
          151673 151674 151675 151676)
SECTION_ID=${SECTIONS[$SLURM_ARRAY_TASK_ID]}
QUALITY_FILTER=baseQ0mapQ0
MIN_ALT_READS=1          # a spot "has" an SNV if >=1 ALT-supporting read; raise to match pipeline thresholds

if [ -z "${SECTION_ID:-}" ]; then
    echo "ERROR: no section for array index $SLURM_ARRAY_TASK_ID"; exit 1
fi

echo "======================================================"
echo "strelka2 spot×SNV matrix | section $SECTION_ID | task $SLURM_ARRAY_TASK_ID"
echo "node $SLURMD_NODENAME | start $(date)"
echo "======================================================"

source activate snv_caller   # has pysam + tqdm

SPOT_DIR="$PROJECT/data/dlpfc/$SECTION_ID/strelka2/spot_snvs"

# --- Step 1: project strelka2 SNVs onto spots -----------------------------
echo "------ Step 1: BAM-scan projection ($(date)) ------"
python scripts/tools/strelka2_to_spot_snvs.py \
    --section_id "$SECTION_ID" \
    --min-alt-reads "$MIN_ALT_READS" \
    --max-workers 22
rc=$?; [ $rc -ne 0 ] && { echo "ERROR: projection failed (exit $rc)"; exit $rc; }

# --- Step 2: build spot×SNV matrix ----------------------------------------
echo "------ Step 2: matrix generation ($(date)) ------"
python scripts/6_spatial_filter/run_generate_matrix.py \
    --dataset dlpfc \
    --section_id "$SECTION_ID" \
    --quality-filter "$QUALITY_FILTER" \
    --input-dir "$SPOT_DIR" \
    --caller strelka2 \
    --output-name germline
rc=$?; [ $rc -ne 0 ] && { echo "ERROR: matrix generation failed (exit $rc)"; exit $rc; }

echo "======================================================"
echo "DONE section $SECTION_ID | end $(date)"
echo "Matrix: data/dlpfc/$SECTION_ID/matrix/DLPFC_${SECTION_ID}_strelka2_germline_6_matrix.pkl"
echo "======================================================"
