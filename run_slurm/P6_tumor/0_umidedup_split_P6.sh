#!/bin/bash
# =============================================================================
# P6 rep1 — UMI dedup + per-cell split  (NEW pre-calling step for the dedup ablation)
#
# WHY: SPARCAL's calling consumes per-spot BAMs (split_BAM/{barcode}.bam). We add
# UMI deduplication so each retained alignment = one unique MOLECULE (UMI), not a
# read (PCR/optical duplicates collapsed). After this, "depth" = UMI count and VAF
# = molecular VAF. split_BAM/ for P6 rep1 was cleaned up, so it must be regenerated
# anyway -- we regenerate it FROM the deduped BAM, so every downstream step that
# reads split_BAM/ inherits dedup with no further code change.
#
# FLOW:  possorted_genome_bam.bam
#          --(umi_tools dedup, per-cell CB+UB, directional)-->  possorted.dedup.bam
#          --(samtools split -d CB)-->  split_BAM/{CB}.bam (+ .bai)   [feeds step 1]
#
# SUBMIT (from repo root /data/maiziezhou_lab/leiy4/snv_calling):
#   sbatch run_slurm/P6_tumor/0_umidedup_split_P6.sh
# THEN the existing calling step + the rest, unchanged:
#   sbatch run_slurm/P6_tumor/1_mpileup_pipeline.sh   # then 2..8
# =============================================================================
#SBATCH --job-name=umidedup_split_P6r1
#SBATCH --output=slurm_output/P6_tumor/umidedup_split_P6r1-%j.out
#SBATCH --error=slurm_output/P6_tumor/umidedup_split_P6r1-%j.err
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail
echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  Start: $(date)"

# --- tools (use absolute paths; no env switch needed) ---
UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools   # 1.23.1 (supports `split -d TAG`)
THREADS=16

# --- paths ---
OUTS=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep1_hg19/P6_Tumor_output/outs
POSSORTED=$OUTS/possorted_genome_bam.bam
DEDUP=$OUTS/possorted_genome_bam.dedup.bam
SPLITDIR=$OUTS/split_BAM

mkdir -p slurm_output/P6_tumor "$SPLITDIR"

for f in "$POSSORTED" "$UMI" "$SAMTOOLS"; do
    [ -e "$f" ] || { echo "ERROR: missing $f"; exit 1; }
done
[ -f "$POSSORTED.bai" ] || $SAMTOOLS index -@ $THREADS "$POSSORTED"

# --- 1) UMI deduplication (collapse CB+UB+position -> one molecule) ---
# single-end (10x Visium aligned cDNA read); directional network handles UMI seq errors.
echo "[1] umi_tools dedup  $(date)"
$UMI dedup \
    -I "$POSSORTED" \
    --per-cell \
    --extract-umi-method=tag --cell-tag=CB --umi-tag=UB \
    --method=directional \
    --log="$OUTS/umidedup.log" \
    -S "$DEDUP"
[ -s "$DEDUP" ] || { echo "ERROR: dedup BAM not produced"; exit 1; }
$SAMTOOLS index -@ $THREADS "$DEDUP"
echo "    reads: pre=$($SAMTOOLS view -c -@ $THREADS "$POSSORTED")  post-dedup=$($SAMTOOLS view -c -@ $THREADS "$DEDUP")"

# --- 2) split deduped BAM by CB tag -> split_BAM/{CB}.bam ---
# -M must exceed the number of distinct CB values (Visium ~5000 spots); the DEFAULT
# is 100 (!) which silently dumps all but 100 cells into _nobarcode.bam. ulimit -n is
# huge here (524288) so there is no file-descriptor concern.
echo "[2] samtools split -d CB  $(date)"
$SAMTOOLS split -@ $THREADS -d CB -M 6000 -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" "$DEDUP"

# --- 3) index each per-spot BAM (mpileup -r needs indexes) ---
echo "[3] index per-spot BAMs  $(date)"
ls "$SPLITDIR"/*.bam | grep -v '_nobarcode.bam' | xargs -P $THREADS -I{} "$SAMTOOLS" index {}

echo "[done] split_BAM bams: $(ls "$SPLITDIR"/*.bam | wc -l)   $(date)"
