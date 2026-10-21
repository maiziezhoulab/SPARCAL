#!/bin/bash
# One-contig real-data smoke test for the fail-closed Beagle v2.3 module.

#SBATCH --job-name=beagle_v23_smoke
#SBATCH --output=slurm_output/beagle_v2_3/smoke_%j.out
#SBATCH --error=slurm_output/beagle_v2_3/smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=96GB

set -euo pipefail

PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
RUN_ROOT="$PROJECT/data/experiments/beagle_v2_3_smoke_20260913_v4/P4_chr22"
SOURCE_RAW="$PROJECT/data/experiments/beagle_panel_routing_gp090_20260911_v3/P4_sec1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz"
LEDGER="$PROJECT/data/experiments/beagle_reference_panel_checksums_20260913_v3_frozen"
SNAPSHOT="$RUN_ROOT/code_snapshot/scripts/2_beagle_filtering"
INPUT_DIR="$RUN_ROOT/output_VCFs/mpileup_multi_bam/baseQ0mapQ0"

cd "$PROJECT"
[ -s "$SOURCE_RAW" ] || { echo "ERROR: missing raw VCF: $SOURCE_RAW"; exit 1; }
[ -s "$SOURCE_RAW.tbi" ] || { echo "ERROR: missing raw VCF index: $SOURCE_RAW.tbi"; exit 1; }
[ -s "$LEDGER/RUN_COMPLETE.json" ] || { echo "ERROR: missing bound reference ledger"; exit 1; }
if [ -e "$RUN_ROOT" ]; then
    echo "ERROR: refusing to overwrite: $RUN_ROOT"
    exit 2
fi

mkdir -p "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13" "$INPUT_DIR"
mkdir -p "$PROJECT/slurm_output/beagle_v2_3"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$RUN_ROOT/RUN_FAILED.txt"' ERR

cp scripts/2_beagle_filtering/run_beagle.py \
   scripts/2_beagle_filtering/beagle_evidence_router.py \
   "$SNAPSHOT/"
cp scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/contracts.py \
   scripts/2_beagle_filtering/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py \
   "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/"
cp run_slurm/run_beagle_v2_3_smoke.sh "$RUN_ROOT/code_snapshot/"
find "$RUN_ROOT/code_snapshot" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$RUN_ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
"$PROJECT/apps/bcftools" view -r chr22 -Oz -o "$INPUT_DIR/merged_sorted_gt.vcf.gz" "$SOURCE_RAW"
"$PROJECT/apps/tabix" -p vcf "$INPUT_DIR/merged_sorted_gt.vcf.gz"

"$PYTHON" "$SNAPSHOT/beagle_v2_3_candidate_2026_09_13/run_beagle_v2_3.py" \
    --dataset P4_TUMOR \
    --section-id 1 \
    --section-root "$RUN_ROOT" \
    --reference-ledger-root "$LEDGER" \
    --quality-filter baseQ0mapQ0 \
    --threads 30 \
    --parallel-chromosomes 1 \
    --iterations 0 \
    --seed 20260915 \
    --modelscale 2 \
    --maxlr 5000 \
    --anchor-score p_nonref \
    --min-anchor-probability 0.90

trap - ERR
echo "Complete: $RUN_ROOT"
