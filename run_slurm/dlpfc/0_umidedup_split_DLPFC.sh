#!/bin/bash
# =============================================================================
# DLPFC — UMI dedup + per-cell split  (pre-calling step 0, dedup ablation)
#
# DLPFC differs from P4/P6/DCIS: the source has NO possorted BAM, only read-only
# per-cell BAMs (bam_bycell/{barcode}.bam, ~4992/section, with CB+UB tags). So we
# MERGE them, UMI-dedup the merged BAM --per-cell (CB+UB, directional), then split
# back by CB. Output goes to the PROJECT dir (source is read-only):
#
#   {DATASET}/{section}/bam_bycell/*.bam                       (read-only source)
#     --(samtools merge)-->        data/dlpfc/{s}/dedup_tmp/merged.bam
#     --(umi_tools dedup --per-cell)--> data/dlpfc/{s}/dedup_tmp/merged.dedup.bam
#     --(samtools split -d CB)-->  data/dlpfc/{s}/bam_bycell_dedup/{CB}.bam (+ .bai)
#
# The DLPFC config in mpileup_pipeline.py + run_filter_bams_by_snv_pools.py already
# points its BAM glob at data/dlpfc/{s}/bam_bycell_dedup/ (via bam_base_path), so
# steps 1-7 inherit dedup with no further change once this finishes.
#
# SUBMIT (from repo root /data/maiziezhou_lab/leiy4/snv_calling):
#   sbatch run_slurm/dlpfc/0_umidedup_split_DLPFC.sh            # all 12 sections
#   sbatch --array=0 run_slurm/dlpfc/0_umidedup_split_DLPFC.sh # just 151507
# THEN the full pipeline: bash run_slurm/dlpfc/run_pipeline_DLPFC.sh
# =============================================================================
#SBATCH --job-name=umidedup_DLPFC
#SBATCH --output=slurm_output/DLPFC/umidedup_split_%a-%A.out
#SBATCH --error=slurm_output/DLPFC/umidedup_split_%a-%A.err
#SBATCH --array=0-11
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1 --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail

SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)
SECTION=${SECTIONS[$SLURM_ARRAY_TASK_ID]}

# --- tools (absolute paths; same as the P4/P6/DCIS dedup step) ---
UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools  # 1.23.1 (split -d TAG)
THREADS=16

SRC=/data/maiziezhou_lab/Datasets/ST_datasets/DLPFC_spatialLIBD/${SECTION}/bam_bycell
TMP=/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/${SECTION}/dedup_tmp
SPLITDIR=/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/${SECTION}/bam_bycell_dedup
MERGED=$TMP/merged.bam
DEDUP=$TMP/merged.dedup.bam

echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  Section: ${SECTION}  Start: $(date)"
mkdir -p slurm_output/DLPFC "$TMP" "$SPLITDIR"
for f in "$UMI" "$SAMTOOLS" "$SRC"; do [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done

# --- 1) merge per-cell BAMs -> one coordinate-sorted BAM (inputs are indexed => sorted) ---
echo "[1] merge per-cell BAMs  $(date)"
ls "$SRC"/*.bam > "$TMP/bam_list.txt"
echo "    merging $(wc -l < "$TMP/bam_list.txt") per-cell BAMs"
$SAMTOOLS merge -@ $THREADS -f -b "$TMP/bam_list.txt" "$MERGED" || { echo "ERROR: merge failed"; exit 1; }
$SAMTOOLS index -@ $THREADS "$MERGED"

# --- 2) UMI deduplication (collapse CB+UB+position -> one molecule) ---
echo "[2] umi_tools dedup --per-cell  $(date)"
$UMI dedup \
    -I "$MERGED" \
    --per-cell \
    --extract-umi-method=tag --cell-tag=CB --umi-tag=UB \
    --method=directional \
    --log="$TMP/umidedup.log" \
    -S "$DEDUP"
[ -s "$DEDUP" ] || { echo "ERROR: dedup BAM not produced"; exit 1; }
$SAMTOOLS index -@ $THREADS "$DEDUP"
echo "    reads: pre=$($SAMTOOLS view -c -@ $THREADS "$MERGED")  post-dedup=$($SAMTOOLS view -c -@ $THREADS "$DEDUP")"

# --- 3) split deduped BAM by CB -> bam_bycell_dedup/{CB}.bam  (-M > #cells; default 100 is a trap) ---
echo "[3] samtools split -d CB  $(date)"
$SAMTOOLS split -@ $THREADS -d CB -M 6000 -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" "$DEDUP"

# --- 4) index each per-cell deduped BAM (mpileup -r needs indexes) ---
echo "[4] index per-cell BAMs  $(date)"
ls "$SPLITDIR"/*.bam | grep -v '_nobarcode.bam' | xargs -P $THREADS -I{} "$SAMTOOLS" index {}

echo "[done] ${SECTION}: bam_bycell_dedup BAMs = $(ls "$SPLITDIR"/*.bam | grep -vc '_nobarcode.bam')   $(date)"
echo "       (optional) remove $TMP to reclaim merged/dedup intermediates once verified"
