#!/bin/bash
# WES-free input-quality ablation for Beagle routing on P4/P6 section 1.
# Tests the project-standard BQ13/MQ20 and Monopogen-matched BQ20/MQ20 inputs.

#SBATCH --job-name=beagle_input_q
#SBATCH --output=slurm_output/beagle_input_quality/%A_%a.out
#SBATCH --error=slurm_output/beagle_input_quality/%A_%a.err
#SBATCH --array=0-3%2
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

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
TASK=${SLURM_ARRAY_TASK_ID:?array task id required}
SAMPLE_INDEX=$((TASK % 2))
QUALITY_INDEX=$((TASK / 2))
case "$SAMPLE_INDEX" in
    0) DATASET=P4_TUMOR; SAMPLE=P4_sec1 ;;
    1) DATASET=P6_TUMOR; SAMPLE=P6_sec1 ;;
esac
case "$QUALITY_INDEX" in
    0) BASEQ=13; MAPQ=20 ;;
    1) BASEQ=20; MAPQ=20 ;;
esac

RUN_LABEL=${RUN_LABEL:-beagle_input_quality_20260912_v1}
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
ROOT="$PROJECT/data/experiments/$RUN_LABEL/$QUALITY_FILTER/$SAMPLE"
cd "$PROJECT"
if [ -e "$ROOT" ]; then
    echo "ERROR: refusing to overwrite existing experiment: $ROOT"
    exit 2
fi
mkdir -p "$ROOT/code_snapshot/scripts/1_calling" "$ROOT/code_snapshot/scripts/2_beagle_filtering"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$ROOT/RUN_FAILED.txt"' ERR
cp scripts/1_calling/mpileup_pipeline.py "$ROOT/code_snapshot/scripts/1_calling/"
cp scripts/2_beagle_filtering/run_beagle.py \
   scripts/2_beagle_filtering/beagle_evidence_router.py \
   "$ROOT/code_snapshot/scripts/2_beagle_filtering/"
cp run_slurm/run_beagle_input_quality_ablation.sh "$ROOT/code_snapshot/"
find "$ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

source activate snv_caller
{
    printf "run_label=%s\ndataset=%s\nsample=%s\n" "$RUN_LABEL" "$DATASET" "$SAMPLE"
    printf "base_quality=%s\nmapping_quality=%s\n" "$BASEQ" "$MAPQ"
    printf "wes_input=NONE\nimpute=false\nniterations=0\nseed=20260912\n"
    printf "router_schema=sparcal_beagle_evidence_v1\nmin_called_gp=0.90\n"
    printf "git_head=%s\nstarted=%s\n" "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)"
} > "$ROOT/RUN_MANIFEST.txt"

python "$ROOT/code_snapshot/scripts/1_calling/mpileup_pipeline.py" \
    --dataset "$DATASET" --section_id 1 --base_quality "$BASEQ" \
    --mapping_quality "$MAPQ" --threads 30 --filter_out_tissue \
    --section-root "$ROOT"
python "$ROOT/code_snapshot/scripts/2_beagle_filtering/run_beagle.py" \
    --dataset "$DATASET" --section_id 1 --quality-filter "$QUALITY_FILTER" \
    --threads 30 --parallel-chromosomes 1 --section-root "$ROOT" \
    --min-anchor-gp 0.90 --iterations 0 --seed 20260912
python "$ROOT/code_snapshot/scripts/2_beagle_filtering/beagle_evidence_router.py" \
    --raw-vcf "$ROOT/output_VCFs/mpileup_multi_bam/$QUALITY_FILTER/merged_sorted_gt.vcf.gz" \
    --beagle-dir "$ROOT/output_VCFs/beagle/$QUALITY_FILTER" \
    --output-dir "$ROOT/beagle_evidence_v1_gp090" --min-called-gp 0.90

printf "completed=%s\n" "$(date --iso-8601=seconds)" >> "$ROOT/RUN_MANIFEST.txt"
touch "$ROOT/RUN_COMPLETE"
trap - ERR
