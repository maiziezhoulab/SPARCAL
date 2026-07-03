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
#SBATCH --job-name=umidedup_split_P4r1
#SBATCH --output=slurm_output/P4_tumor/umidedup_split_P4r1-%j.out
#SBATCH --error=slurm_output/P4_tumor/umidedup_split_P4r1-%j.err
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

set -o pipefail
echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  Start: $(date)"

UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools  # v1.23.1 (split -d)
THREADS=16

OUTS=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Visium/spaceranger_align_rep1_hg19/P4_Tumor_output/outs
POSSORTED=$OUTS/possorted_genome_bam.bam
DEDUP=$OUTS/possorted_genome_bam.dedup.bam
SPLITDIR=$OUTS/split_BAM

mkdir -p slurm_output/P4_tumor
for f in "$POSSORTED" "$UMI" "$SAMTOOLS"; do [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
[ -f "$POSSORTED.bai" ] || $SAMTOOLS index -@ $THREADS "$POSSORTED"

# preserve any existing (non-deduped) split_BAM rather than clobber it
if [ -d "$SPLITDIR" ] && [ -n "$(ls "$SPLITDIR"/*.bam 2>/dev/null)" ]; then
    mv "$SPLITDIR" "${SPLITDIR}.nodedup_bak.$(date +%s)"
fi
mkdir -p "$SPLITDIR"

echo "[1] umi_tools dedup  $(date)"
$UMI dedup -I "$POSSORTED" --per-cell \
    --extract-umi-method=tag --cell-tag=CB --umi-tag=UB --method=directional \
    --log="$OUTS/umidedup.log" -S "$DEDUP"
[ -s "$DEDUP" ] || { echo "ERROR: dedup BAM not produced"; exit 1; }
$SAMTOOLS index -@ $THREADS "$DEDUP"
echo "    reads: pre=$($SAMTOOLS view -c -@ $THREADS "$POSSORTED")  post=$($SAMTOOLS view -c -@ $THREADS "$DEDUP")"

echo "[2] samtools split -d CB -M 6000  $(date)"
$SAMTOOLS split -@ $THREADS -d CB -M 6000 -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" "$DEDUP"

echo "[3] index per-spot BAMs  $(date)"
ls "$SPLITDIR"/*.bam | grep -v '_nobarcode.bam' | xargs -P $THREADS -I{} "$SAMTOOLS" index {}
echo "[done] split_BAM bams: $(ls "$SPLITDIR"/*.bam | grep -vc _nobarcode)   $(date)"
