#!/usr/bin/env bash
#SBATCH --job-name=p4p6_ssnv_cb
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
MODE=${1:-full}
if [[ $MODE == full ]]; then
    REP2=(P4_rep2 P6_rep2)
    SECTION=${REP2[${SLURM_ARRAY_TASK_ID:?array task required}]}
else
    SECTION=${2:-P4_rep2}
fi
ROOT=$(mode_root "$MODE")
OUTDIR=$(caller_dir "$MODE" "$SECTION" spatialsnv)
CALLBACK_DIR="$OUTDIR/callback"
mkdir -p "$OUTDIR" "$CALLBACK_DIR" "$OUTDIR/tmp_callback"

export PATH="$SPATIALSNV_ENV/bin:$PATH"
export CONDA_PREFIX="$SPATIALSNV_ENV"
SOURCE_SAMPLE=${SECTION/_rep/_sec}
SOURCE_ROOT=/data/maiziezhou_lab/leiy4/SPARCAL/data/benchmarks/spatialsnv/$SOURCE_SAMPLE
PREP_BAM="$SOURCE_ROOT/prepared/${SOURCE_SAMPLE}.rdfcall.bam"
SOURCE_VCF="$SOURCE_ROOT/calls/${SOURCE_SAMPLE}.vcf.gz"
require_file "$PREP_BAM"
require_file "$SOURCE_VCF"
require_file "$SOURCE_ROOT/prepared/${SOURCE_SAMPLE}.rdfcall.bai"
LOCAL_BAM="$OUTDIR/${SOURCE_SAMPLE}.rdfcall.bam"
if [[ ! -e $LOCAL_BAM ]]; then ln -s "$PREP_BAM" "$LOCAL_BAM"; fi
if [[ ! -e $LOCAL_BAM.bai ]]; then ln -s "$SOURCE_ROOT/prepared/${SOURCE_SAMPLE}.rdfcall.bai" "$LOCAL_BAM.bai"; fi
PREP_BAM=$LOCAL_BAM

if [[ $MODE == smoke ]]; then
    VERIFY_DIR="$ROOT/_spatialsnv_rep1_builder_verification"
    mkdir -p "$VERIFY_DIR"
    "$SNV_CALLER_ENV/bin/python" "$SCRIPTROOT/callback_to_pkl_adapter.py" \
        --matrix-dir "$PROJECT/SpatialSNV/results/p4/matrix" --sample p4 \
        --vcf "$PROJECT/SpatialSNV/results/p4/p4.vcf.gz" --outdir "$VERIFY_DIR"
    "$SNV_CALLER_ENV/bin/python" "$SCRIPTROOT/validate_spatialsnv_rep1.py" \
        --rerun "$VERIFY_DIR/p4_spatialsnv_presence_filtered_matrix.pkl" \
        --existing "$PROJECT/SpatialSNV/results/p4/matrix/p4_spatialsnv_presence_filtered_matrix.pkl" \
        --json "$VERIFY_DIR/verification.json"
    USE_VCF="$OUTDIR/${SOURCE_SAMPLE}.chr21.vcf.gz"
    "$BCFTOOLS" view -r chr21 -Oz -o "$USE_VCF" "$SOURCE_VCF"
    "$BCFTOOLS" index -f -t "$USE_VCF"
else
    USE_VCF=$SOURCE_VCF
fi

spatialsnvtools CallBack \
    -b "$PREP_BAM" -v "$USE_VCF" -s "$SECTION" \
    -o "$CALLBACK_DIR" --tmpdir "$OUTDIR/tmp_callback" \
    --only_autosome -c CB -u UB -@ "$SLURM_CPUS_PER_TASK"

require_file "$CALLBACK_DIR/${SECTION}_alt/matrix.mtx.gz"
require_file "$CALLBACK_DIR/${SECTION}_alt/features.tsv.gz"
require_file "$CALLBACK_DIR/${SECTION}_alt/barcodes.tsv.gz"
"$SNV_CALLER_ENV/bin/python" "$SCRIPTROOT/callback_to_pkl_adapter.py" \
    --matrix-dir "$CALLBACK_DIR" --sample "$SECTION" --vcf "$USE_VCF" --outdir "$OUTDIR"
require_file "$OUTDIR/${SECTION}_spatialsnv_presence_matrix.pkl"
require_file "$OUTDIR/${SECTION}_spatialsnv_presence_filtered_matrix.pkl"
echo "SpatialSNV CallBack complete: mode=$MODE section=$SECTION"
