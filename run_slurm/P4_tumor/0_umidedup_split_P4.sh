#!/bin/bash
# =============================================================================
# P4 rep1 — UMI dedup + per-cell split (pre-calling step; mirrors P6's).
# Dedups molecules (CB+UB, directional) so "depth" = UMI count, VAF = molecular VAF,
# then regenerates split_BAM/{barcode}.bam FROM the deduped BAM → every downstream
# step that reads split_BAM/ inherits dedup. split_BAM was cleaned up, so it is
# regenerated here anyway.
#   SUBMIT (from repo root /data/maiziezhou_lab/leiy4/snv_calling):
#     sbatch run_slurm/P4_tumor/0_umidedup_split_P4.sh
#   THEN: sbatch run_slurm/P4_tumor/1_mpileup_pipeline.sh  then 2_..8_ (unchanged)
#   NOTE: like P6, step 1 wants barcode_file GSM4565823_barcodes.tsv.GZ but disk may
#         have the uncompressed .tsv — `gzip -k …/Meta_Data/GSM4565823_barcodes.tsv` first.
# =============================================================================
#SBATCH --job-name=umidedup_split_P4
#SBATCH --output=slurm_output/P4_tumor/umidedup_split_P4-%j.out
#SBATCH --error=slurm_output/P4_tumor/umidedup_split_P4-%j.err
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

SECTION_ID=${1:-1}
case "$SECTION_ID" in
    1|2) ;;
    *) echo "ERROR: section_id must be 1 or 2 (got: $SECTION_ID)"; exit 2 ;;
esac

echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  P4 section: ${SECTION_ID}  Start: $(date)"

UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools  # v1.23.1 (split -d)
THREADS=16

OUTS=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep${SECTION_ID}_hg19/P4_Tumor_output/outs
POSSORTED=$OUTS/possorted_genome_bam.bam
DEDUP=$OUTS/possorted_genome_bam.dedup.bam
SPLITDIR=$OUTS/split_BAM
MARKER=$OUTS/.sparcal_umi_dedup_complete
LOG=$OUTS/umidedup_rep${SECTION_ID}.log

mkdir -p slurm_output/P4_tumor
for f in "$POSSORTED" "$UMI" "$SAMTOOLS"; do
    [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
"$SAMTOOLS" quickcheck "$POSSORTED"

split_count=0
if [ -d "$SPLITDIR" ]; then
    split_count=$(find "$SPLITDIR" -maxdepth 1 -type f -name '*.bam' ! -name '_nobarcode.bam' -size +0c | wc -l)
fi
if [ -s "$MARKER" ] && [ -s "$DEDUP" ] && [ "$split_count" -gt 0 ]; then
    echo "[skip] completed UMI-dedup outputs already exist: $MARKER"
    exit 0
fi

[ -f "$POSSORTED.bai" ] || "$SAMTOOLS" index -@ "$THREADS" "$POSSORTED"

backup_tag="pre_rerun_$(date +%Y%m%d_%H%M%S).${SLURM_JOB_ID:-manual}"
for path in "$DEDUP" "$DEDUP.bai" "$SPLITDIR" "$LOG" "$MARKER"; do
    if [ -e "$path" ]; then
        mv "$path" "${path}.${backup_tag}"
        echo "[backup] $path -> ${path}.${backup_tag}"
    fi
done
mkdir -p "$SPLITDIR"

echo "[1] umi_tools dedup  $(date)"
"$UMI" dedup -I "$POSSORTED" --per-cell \
    --extract-umi-method=tag --cell-tag=CB --umi-tag=UB --method=directional \
    --log="$LOG" -S "$DEDUP"
[ -s "$DEDUP" ] || { echo "ERROR: dedup BAM not produced"; exit 1; }
"$SAMTOOLS" quickcheck "$DEDUP"
"$SAMTOOLS" index -@ "$THREADS" "$DEDUP"
pre_reads=$("$SAMTOOLS" view -c -@ "$THREADS" "$POSSORTED")
post_reads=$("$SAMTOOLS" view -c -@ "$THREADS" "$DEDUP")
echo "    reads: pre=$pre_reads  post-dedup=$post_reads"

echo "[2] samtools split -d CB -M 6000  $(date)"
"$SAMTOOLS" split -@ "$THREADS" -d CB -M 6000 \
    -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" "$DEDUP"

echo "[3] index per-spot BAMs  $(date)"
find "$SPLITDIR" -maxdepth 1 -type f -name '*.bam' ! -name '_nobarcode.bam' -print0 \
    | xargs -0 -r -P "$THREADS" -n 1 "$SAMTOOLS" index -@ 1
split_count=$(find "$SPLITDIR" -maxdepth 1 -type f -name '*.bam' ! -name '_nobarcode.bam' -size +0c | wc -l)
[ "$split_count" -gt 0 ] || { echo "ERROR: no per-spot BAMs were produced"; exit 1; }

{
    echo "status=complete"
    echo "sample=P4"
    echo "section_id=$SECTION_ID"
    echo "source_bam=$POSSORTED"
    echo "dedup_bam=$DEDUP"
    echo "pre_reads=$pre_reads"
    echo "post_reads=$post_reads"
    echo "split_bams=$split_count"
    echo "completed_at=$(date --iso-8601=seconds)"
    echo "slurm_job_id=${SLURM_JOB_ID:-NA}"
} > "${MARKER}.tmp"
mv "${MARKER}.tmp" "$MARKER"

echo "[done] split_BAM bams: $split_count  marker: $MARKER  $(date)"
