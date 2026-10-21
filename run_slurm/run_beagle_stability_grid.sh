#!/bin/bash
# WES-free Beagle iteration/seed stability grid for P4/P6 section 1.
# Raw mpileup calls are linked read-only from the frozen 2026-09-11 experiment;
# each task writes a new immutable tree and then applies evidence router v1.

#SBATCH --job-name=beagle_stability
#SBATCH --output=slurm_output/beagle_stability/%A_%a.out
#SBATCH --error=slurm_output/beagle_stability/%A_%a.err
#SBATCH --array=0-11%4
#SBATCH --time=30:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
cd "$PROJECT_DIR"

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
SAMPLE_INDEX=$((TASK_ID % 2))
CONFIG_INDEX=$((TASK_ID / 2))

case "$SAMPLE_INDEX" in
    0) DATASET=P4_TUMOR; SAMPLE_LABEL=P4_sec1 ;;
    1) DATASET=P6_TUMOR; SAMPLE_LABEL=P6_sec1 ;;
    *) echo "ERROR: invalid sample index"; exit 2 ;;
esac

case "$CONFIG_INDEX" in
    0) ITERATIONS=0; SEED=20260912 ;;
    1) ITERATIONS=0; SEED=20260913 ;;
    2) ITERATIONS=0; SEED=20260914 ;;
    3) ITERATIONS=5; SEED=20260912 ;;
    4) ITERATIONS=5; SEED=20260913 ;;
    5) ITERATIONS=5; SEED=20260914 ;;
    *) echo "ERROR: invalid configuration index"; exit 2 ;;
esac

QUALITY_FILTER=baseQ0mapQ0
RUN_LABEL=${RUN_LABEL:-beagle_stability_niter_seed_20260912_v1}
SOURCE_ROOT="$PROJECT_DIR/data/experiments/beagle_panel_routing_gp090_20260911_v3/$SAMPLE_LABEL"
RAW_VCF="$SOURCE_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY_FILTER/merged_sorted_gt.vcf.gz"
SECTION_ROOT="$PROJECT_DIR/data/experiments/$RUN_LABEL/niter${ITERATIONS}_seed${SEED}/$SAMPLE_LABEL"
ROUTE_DIR="$SECTION_ROOT/beagle_evidence_v1_gp090"

[ -s "$RAW_VCF" ] || { echo "ERROR: missing raw VCF: $RAW_VCF"; exit 1; }
[ -s "$RAW_VCF.tbi" ] || { echo "ERROR: missing raw VCF index: $RAW_VCF.tbi"; exit 1; }
if [ -e "$SECTION_ROOT" ]; then
    echo "ERROR: refusing to overwrite existing experiment: $SECTION_ROOT"
    exit 2
fi

mkdir -p "$SECTION_ROOT/code_snapshot/scripts/2_beagle_filtering"
mkdir -p "$SECTION_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY_FILTER"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$SECTION_ROOT/RUN_FAILED.txt"' ERR

cp scripts/2_beagle_filtering/run_beagle.py \
   scripts/2_beagle_filtering/beagle_evidence_router.py \
   "$SECTION_ROOT/code_snapshot/scripts/2_beagle_filtering/"
cp run_slurm/run_beagle_stability_grid.sh "$SECTION_ROOT/code_snapshot/"
find "$SECTION_ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$SECTION_ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

ln -s "$RAW_VCF" "$SECTION_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY_FILTER/merged_sorted_gt.vcf.gz"
ln -s "$RAW_VCF.tbi" "$SECTION_ROOT/output_VCFs/mpileup_multi_bam/$QUALITY_FILTER/merged_sorted_gt.vcf.gz.tbi"

source activate snv_caller

{
    printf "run_label=%s\n" "$RUN_LABEL"
    printf "dataset=%s\nsection=1\nsample=%s\n" "$DATASET" "$SAMPLE_LABEL"
    printf "raw_vcf=%s\n" "$RAW_VCF"
    printf "raw_vcf_sha256=%s\n" "$(sha256sum "$RAW_VCF" | cut -d' ' -f1)"
    printf "wes_input=NONE\nimpute=false\nniterations=%s\nseed=%s\n" "$ITERATIONS" "$SEED"
    printf "router_schema=sparcal_beagle_evidence_v1\nmin_called_gp=0.90\n"
    printf "git_head=%s\nstarted=%s\n" "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)"
} > "$SECTION_ROOT/RUN_MANIFEST.txt"

python "$SECTION_ROOT/code_snapshot/scripts/2_beagle_filtering/run_beagle.py" \
    --dataset "$DATASET" --section_id 1 --quality-filter "$QUALITY_FILTER" \
    --threads 30 --parallel-chromosomes 1 --section-root "$SECTION_ROOT" \
    --min-anchor-gp 0.90 --iterations "$ITERATIONS" --seed "$SEED"

python "$SECTION_ROOT/code_snapshot/scripts/2_beagle_filtering/beagle_evidence_router.py" \
    --raw-vcf "$RAW_VCF" \
    --beagle-dir "$SECTION_ROOT/output_VCFs/beagle/$QUALITY_FILTER" \
    --output-dir "$ROUTE_DIR" --min-called-gp 0.90

printf "completed=%s\n" "$(date --iso-8601=seconds)" >> "$SECTION_ROOT/RUN_MANIFEST.txt"
touch "$SECTION_ROOT/RUN_COMPLETE"
trap - ERR
echo "Complete: $SECTION_ROOT"
