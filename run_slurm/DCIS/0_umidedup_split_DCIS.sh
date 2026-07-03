#!/bin/bash
# =============================================================================
# DCIS1 + DCIS2 — UMI dedup + per-cell split (pre-calling step; mirrors P6's).
# Array over the two sections. Dedups molecules (CB+UB, directional), then
# regenerates split_BAM/{barcode}.bam FROM the deduped BAM so every downstream step
# inherits dedup. DCIS already HAS a (non-deduped) split_BAM — it is moved aside, not
# clobbered. DCIS is hg38, ~25-26 GB BAMs → dedup is long (budget 48 h / 200 GB).
#   SUBMIT (from repo root /data/maiziezhou_lab/leiy4/snv_calling):
#     sbatch run_slurm/DCIS/0_umidedup_split_DCIS.sh
#   THEN the DCIS calling step + the rest (run_slurm/DCIS/, unchanged).
# =============================================================================
#SBATCH --job-name=umidedup_split_DCIS
#SBATCH --output=slurm_output/DCIS/umidedup_split_DCIS-%A_%a.out
#SBATCH --error=slurm_output/DCIS/umidedup_split_DCIS-%A_%a.err
#SBATCH --array=1-2
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail
SEC=${SLURM_ARRAY_TASK_ID:-1}
echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  DCIS${SEC}  Start: $(date)"

UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools  # v1.23.1 (split -d)
THREADS=16

OUTS=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/spatialSNV/10x-Visium/DCIS${SEC}/spaceranger_align_DCIS${SEC}_hg38/DCIS${SEC}_output/outs
POSSORTED=$OUTS/possorted_genome_bam.bam
DEDUP=$OUTS/possorted_genome_bam.dedup.bam
SPLITDIR=$OUTS/split_BAM

mkdir -p slurm_output/DCIS
for f in "$POSSORTED" "$UMI" "$SAMTOOLS"; do [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }; done
[ -f "$POSSORTED.bai" ] || $SAMTOOLS index -@ $THREADS "$POSSORTED"

# preserve the existing (non-deduped) split_BAM rather than clobber it
if [ -d "$SPLITDIR" ] && [ -n "$(ls "$SPLITDIR"/*.bam 2>/dev/null)" ]; then
    mv "$SPLITDIR" "${SPLITDIR}.nodedup_bak.$(date +%s)" && echo "  moved old split_BAM aside"
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
echo "[done] DCIS${SEC} split_BAM bams: $(ls "$SPLITDIR"/*.bam | grep -vc _nobarcode)   $(date)"
