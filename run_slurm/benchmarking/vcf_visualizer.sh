#!/bin/bash
#SBATCH --job-name=vcf_visualizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --time=24:00:00

#SBATCH --output=slurm_output/vcf_visualizer/%x-%j.out
#SBATCH --error=slurm_output/vcf_visualizer/%x-%j.err

#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int

DATASET="DCIS"
SECTION="1"
MAX_WORKERS=22
QUALITY_FILTER="baseQ0mapQ0"

# ── Select VCF (uncomment one) ────────────────────────────────────────────────

# Original mpileup merged VCF — multi-sample (one column per barcode):
# VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/output_VCFs/mpileup_multi_bam/${QUALITY_FILTER}/merged_sorted_gt.vcf.gz"

# Beagle filtered-in VCF — single-sample (DCIS merged BAM, BAM scan):
# VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/output_VCFs/beagle/${QUALITY_FILTER}/all_filtered_in.vcf.gz"

# Germline de-novo classified VCF — sites-only (BAM scan):
VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/spatial_filter_purity/${QUALITY_FILTER}/germline/denovo/germline_denovo.vcf.gz"

# Germline defined VCF — sites-only (BAM scan):
# VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/spatial_filter_purity/${QUALITY_FILTER}/germline/defined/germline_defined.vcf.gz"

# Somatic de-novo classified VCF — sites-only (BAM scan):
# VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis${SECTION}/spatial_filter_purity/${QUALITY_FILTER}/somatic/denovo/somatic_denovo.vcf.gz"
ls -lh "${VCF}"
# ls /lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS1/spaceranger_align_DCIS1_hg38/DCIS1_output/outs/spatial/tissue_positions.csv 
# ─────────────────────────────────────────────────────────────────────────────

# Output directory (leave empty to auto-place a vcf_visualizer/ folder next to the VCF)
OUTPUT_DIR=""

# Plot title (leave empty to auto-generate from dataset + VCF filename)
TITLE=""

# ─────────────────────────────────────────────────────────────────────────────

mkdir -p "slurm_output/vcf_visualizer"

echo "==============================================="
echo "SLURM_JOB_ID:  $SLURM_JOB_ID"
echo "Dataset:       $DATASET"
echo "Section:       $SECTION"
echo "Max workers:   $MAX_WORKERS"
echo "VCF:           $VCF"
echo "Output dir:    ${OUTPUT_DIR:-<auto: next to VCF>}"
echo "Hostname:      $(hostname)"
echo "PWD:           $(pwd)"
echo "Start time:    $(date)"
echo "==============================================="

# Source the user's conda directly to avoid picking up the system-level
# snv_caller environment loaded by "module load Anaconda3"
echo "[step 1] initializing conda"
source activate snv_caller

echo "which python:       $(which python 2>/dev/null || echo '<not found>')"
echo "python version:     $(python --version 2>&1)"
echo "CONDA_DEFAULT_ENV:  ${CONDA_DEFAULT_ENV:-<unset>}"

# Build optional args
EXTRA_ARGS=""
[ -n "${OUTPUT_DIR}" ] && EXTRA_ARGS="${EXTRA_ARGS} --output-dir ${OUTPUT_DIR}"
[ -n "${TITLE}" ]      && EXTRA_ARGS="${EXTRA_ARGS} --title \"${TITLE}\""

echo "[step 2] launching vcf_visualizer.py"
python -u scripts/tools/vcf_visualizer.py \
    --vcf         "${VCF}"         \
    --dataset     "${DATASET}"     \
    --section-id  "${SECTION}"     \
    --max-workers "${MAX_WORKERS}" \
    ${EXTRA_ARGS}
STATUS=$?

if [ $STATUS -eq 0 ]; then
    echo "[OK]  ${DATASET} sec${SECTION} $(basename ${VCF}) completed at $(date)"
else
    echo "[ERR] ${DATASET} sec${SECTION} $(basename ${VCF}) failed (exit ${STATUS}) at $(date)"
fi

echo "End time: $(date)"
exit $STATUS
