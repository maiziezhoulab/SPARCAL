#!/bin/bash
# OVAR_P5 — UMI dedup + per-spot split (same procedure as run_slurm/P6_tumor/0_umidedup_split_P6.sh).
#
# possorted_genome_bam.bam (colleague, READ-ONLY)
#   --(umi_tools dedup, per-cell CB+UB, directional)-->  data/ovar_p5/P5_sr13/umidedup/possorted.dedup.bam
#   --(samtools split -d CB)-->  data/ovar_p5/P5_sr13/split_BAM_dedup/{CB}.bam (+ .bai)   [feeds steps 1 and 6]
# The July pre-dedup split_BAM/ is left untouched (rollback path in the step 1/6 configs).
#
#   sbatch run_slurm/ovar_p5/0_umidedup_split_OVAR.sh
#
#SBATCH --job-name=umidedup_split_OVAR
#SBATCH --output=slurm_output/OVAR_P5/umidedup_split_OVAR-%j.out
#SBATCH --error=slurm_output/OVAR_P5/umidedup_split_OVAR-%j.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

SECTION_ID=P5_sr13
echo "SLURM_JOBID: ${SLURM_JOBID:-NA}  OVAR_P5 section: ${SECTION_ID}  Start: $(date)"

UMI=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/SpaceTracer/bin/umi_tools
SAMTOOLS=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/spatialsnv/bin/samtools   # supports `split -d TAG`
THREADS=16

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
POSSORTED=/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs/${SECTION_ID}/outs/possorted_genome_bam.bam
WORK=$PROJECT/data/ovar_p5/${SECTION_ID}/umidedup
DEDUP=$WORK/possorted.dedup.bam
SPLITDIR=$PROJECT/data/ovar_p5/${SECTION_ID}/split_BAM_dedup
MARKER=$WORK/.sparcal_umi_dedup_complete
LOG=$WORK/umidedup.log

mkdir -p "$PROJECT/slurm_output/OVAR_P5" "$WORK"

for f in "$POSSORTED" "$POSSORTED.bai" "$UMI" "$SAMTOOLS"; do
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
if [ -e "$SPLITDIR" ] || [ -e "$DEDUP" ]; then
    echo "ERROR: partial outputs exist without a completion marker; inspect and remove them first:"
    echo "  $SPLITDIR  $DEDUP"
    exit 1
fi
mkdir -p "$SPLITDIR"

echo "[1] umi_tools dedup  $(date)"
"$UMI" dedup \
    -I "$POSSORTED" \
    --per-cell \
    --extract-umi-method=tag --cell-tag=CB --umi-tag=UB \
    --method=directional \
    --log="$LOG" \
    -S "$DEDUP"
[ -s "$DEDUP" ] || { echo "ERROR: dedup BAM not produced"; exit 1; }
"$SAMTOOLS" quickcheck "$DEDUP"
"$SAMTOOLS" index -@ "$THREADS" "$DEDUP"
pre_reads=$("$SAMTOOLS" view -c -@ "$THREADS" "$POSSORTED")
post_reads=$("$SAMTOOLS" view -c -@ "$THREADS" "$DEDUP")
echo "    reads: pre=$pre_reads  post-dedup=$post_reads"

# -M must exceed the number of distinct CB values (4,992 Visium spots); the default 100
# silently dumps all but 100 cells into _nobarcode.bam.
echo "[2] samtools split -d CB  $(date)"
"$SAMTOOLS" split -@ "$THREADS" -d CB -M 6000 \
    -f "$SPLITDIR/%!.bam" -u "$SPLITDIR/_nobarcode.bam" "$DEDUP"

echo "[3] index per-spot BAMs  $(date)"
find "$SPLITDIR" -maxdepth 1 -type f -name '*.bam' ! -name '_nobarcode.bam' -print0 \
    | xargs -0 -r -P "$THREADS" -n 1 "$SAMTOOLS" index -@ 1
split_count=$(find "$SPLITDIR" -maxdepth 1 -type f -name '*.bam' ! -name '_nobarcode.bam' -size +0c | wc -l)
[ "$split_count" -gt 0 ] || { echo "ERROR: no per-spot BAMs were produced"; exit 1; }

{
    echo "status=complete"
    echo "sample=OVAR_P5"
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

echo "[done] split_BAM_dedup bams: $split_count  marker: $MARKER  $(date)"
