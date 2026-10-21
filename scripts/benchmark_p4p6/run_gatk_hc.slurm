#!/usr/bin/env bash
#SBATCH --job-name=p4p6_gatk_hc
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
MODE=${1:-full}
if [[ $MODE == full ]]; then
    TASK=${SLURM_ARRAY_TASK_ID:?array task required}
    SECTION=${SECTIONS[$((TASK / 24))]}
    CHROM=${CHROMS[$((TASK % 24))]}
else
    SECTION=${2:-P4_rep2}
    CHROM=${3:-chr21}
fi
OUTDIR=$(caller_dir "$MODE" "$SECTION" gatk)
SPLIT="$OUTDIR/${SECTION}.splitncigar.bam"
SCATTER="$OUTDIR/scatter"
VCF="$SCATTER/${SECTION}.${CHROM}.vcf.gz"
mkdir -p "$SCATTER" "$OUTDIR/tmp"
require_file "$SPLIT"
export PATH="$SPATIALSNV_ENV/bin:$PATH"
export CONDA_PREFIX="$SPATIALSNV_ENV"
if [[ -s $VCF && -s $VCF.tbi ]]; then
    echo "Existing HaplotypeCaller shard found, leaving unchanged: $VCF"
    exit 0
fi
"$GATK" --java-options "-Xmx26g" HaplotypeCaller \
    -R "$REF" -I "$SPLIT" -L "$CHROM" \
    --dont-use-soft-clipped-bases true -stand-call-conf 20 \
    --native-pair-hmm-threads "$SLURM_CPUS_PER_TASK" \
    --tmp-dir "$OUTDIR/tmp" -O "$VCF"
require_file "$VCF"
require_file "$VCF.tbi"
echo "HaplotypeCaller complete: mode=$MODE section=$SECTION chrom=$CHROM"
