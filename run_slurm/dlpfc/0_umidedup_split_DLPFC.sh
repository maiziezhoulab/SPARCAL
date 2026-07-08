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
# RESUMABLE: if data/dlpfc/{s}/dedup_tmp/merged.dedup.bam already exists and passes
# quickcheck, merge + dedup are SKIPPED and the job goes straight to the (OOM-safe)
# split — so a re-run after the 2026-07-02 split OOM only redoes the cheap split+index.
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

# --- 1+2) merge + UMI dedup  (RESUMABLE: skipped if a valid deduped BAM exists) ---
# The 2026-07-02 run left an intact merged.dedup.bam for every section (the ~4 h part
# succeeded; only the split below was OOM-Killed). quickcheck lets a re-run reuse it
# and jump straight to the split.
if $SAMTOOLS quickcheck "$DEDUP" 2>/dev/null; then
    echo "[1-2] valid $DEDUP already present -> skipping merge + dedup  $(date)"
else
    echo "[1] merge per-cell BAMs  $(date)"
    ls "$SRC"/*.bam > "$TMP/bam_list.txt"
    echo "    merging $(wc -l < "$TMP/bam_list.txt") per-cell BAMs"
    $SAMTOOLS merge -@ $THREADS -f -b "$TMP/bam_list.txt" "$MERGED" || { echo "ERROR: merge failed"; exit 1; }
    $SAMTOOLS index -@ $THREADS "$MERGED"

    echo "[2] umi_tools dedup --per-cell  $(date)"
    $UMI dedup \
        -I "$MERGED" \
        --per-cell \
        --extract-umi-method=tag --cell-tag=CB --umi-tag=UB \
        --method=directional \
        --log="$TMP/umidedup.log" \
        -S "$DEDUP" || { echo "ERROR: umi_tools dedup failed"; exit 1; }
    $SAMTOOLS quickcheck "$DEDUP" || { echo "ERROR: dedup BAM missing/truncated"; exit 1; }
    $SAMTOOLS index -@ $THREADS "$DEDUP"
    echo "    reads: pre=$($SAMTOOLS view -c -@ $THREADS "$MERGED")  post-dedup=$($SAMTOOLS view -c -@ $THREADS "$DEDUP")"
fi

# --- 3) split deduped BAM by CB -> bam_bycell_dedup/{CB}.bam  (-M > #cells; default 100 is a trap) ---
# CRITICAL: merged.dedup.bam carries a HUGE header (~1.1M lines: ~20k @RG + ~35k @PG,
# one set per input per-cell BAM accumulated through merge+dedup). samtools split keeps
# the parsed header in memory and attaches it to every one of the ~5000 open output
# streams, so the plain `-M 6000` split needed >128 GB and was OOM-Killed (both the
# 2026-07-02 `-@16` run and the 2026-07-05 `-@1` retry). Fix: reheader to a minimal
# @HD+@SQ header on the fly (measured 13.8 GB -> 0.86 GB peak RSS on a 4870-cell slice),
# piped straight into split so there is no 13 GB temp. The dropped @RG/@PG are unused
# downstream (bcftools mpileup ignores them); the per-cell BAMs end up with the same
# lean header the non-dedup source BAMs already have. `|| exit` guard is essential:
# without it a future kill masquerades as success.
echo "[3] reheader (slim @HD+@SQ) | samtools split -d CB  $(date)"
rm -f "$SPLITDIR"/*.bam "$SPLITDIR"/*.bai   # purge any truncated files from a prior killed run
SLIMHDR=$TMP/slim.header
$SAMTOOLS view -H "$DEDUP" | awk '/^@HD/ || /^@SQ/' > "$SLIMHDR"
printf '@PG\tID:slim_reheader\tPN:samtools_reheader\n' >> "$SLIMHDR"
[ "$(wc -l < "$SLIMHDR")" -gt 1 ] || { echo "ERROR: slim header build failed (no @SQ lines)"; exit 1; }
$SAMTOOLS reheader "$SLIMHDR" "$DEDUP" 2>/dev/null \
  | $SAMTOOLS split -@ 1 -d CB -M 6000 -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" - \
  || { echo "ERROR: reheader|split failed/killed"; exit 1; }

# --- 3b) validate: every per-cell BAM must be complete (quickcheck catches truncation) ---
echo "[3b] quickcheck split output  $(date)"
$SAMTOOLS quickcheck -v "$SPLITDIR"/*.bam \
    || { echo "ERROR: one or more split BAMs are truncated/corrupt"; exit 1; }

# --- 4) index each per-cell deduped BAM (mpileup -r needs indexes) ---
echo "[4] index per-cell BAMs  $(date)"
ls "$SPLITDIR"/*.bam | grep -v '_nobarcode.bam' | xargs -P $THREADS -I{} "$SAMTOOLS" index {} \
    || { echo "ERROR: indexing failed"; exit 1; }

echo "[done] ${SECTION}: bam_bycell_dedup BAMs = $(ls "$SPLITDIR"/*.bam | grep -vc '_nobarcode.bam')   $(date)"
echo "       (optional) remove $TMP to reclaim merged/dedup intermediates once verified"
