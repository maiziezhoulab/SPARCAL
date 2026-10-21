#!/usr/bin/env bash
#SBATCH --job-name=p4p6_gatk_split
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
MODE=${1:-full}
if [[ $MODE == full ]]; then
    SECTION=${SECTIONS[${SLURM_ARRAY_TASK_ID:?array task required}]}
else
    SECTION=${2:-P4_rep2}
fi
BAM=$(section_bam "$SECTION")
OUTDIR=$(caller_dir "$MODE" "$SECTION" gatk)
SPLIT="$OUTDIR/${SECTION}.splitncigar.bam"
mkdir -p "$OUTDIR/tmp"
require_file "$BAM"
export PATH="$SPATIALSNV_ENV/bin:$PATH"
export CONDA_PREFIX="$SPATIALSNV_ENV"
"$GATK" --version

if [[ -s $SPLIT && -s ${SPLIT%.bam}.bai ]]; then
    "$SAMTOOLS" quickcheck -v "$SPLIT"
    echo "Existing SplitNCigarReads output found, leaving unchanged: $SPLIT"
    exit 0
fi
ARGS=(--java-options "-Xmx40g" SplitNCigarReads -R "$REF" -I "$BAM" -O "$SPLIT"
      --create-output-bam-index true --tmp-dir "$OUTDIR/tmp")
if [[ $MODE == smoke ]]; then ARGS+=(-L chr21); fi
"$GATK" "${ARGS[@]}"
require_file "$SPLIT"
require_file "${SPLIT%.bam}.bai"
echo "SplitNCigarReads complete: mode=$MODE section=$SECTION (no BQSR by design)"
