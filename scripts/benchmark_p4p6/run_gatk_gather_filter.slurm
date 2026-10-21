#!/usr/bin/env bash
#SBATCH --job-name=p4p6_gatk_filter
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=08:00:00
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
    SECTION=${SECTIONS[${SLURM_ARRAY_TASK_ID:?array task required}]}
    USE_CHROMS=("${CHROMS[@]}")
else
    SECTION=${2:-P4_rep2}
    USE_CHROMS=(chr21)
fi
OUTDIR=$(caller_dir "$MODE" "$SECTION" gatk)
GATHERED="$OUTDIR/${SECTION}.gatk.gathered.vcf.gz"
FILTERED="$OUTDIR/${SECTION}.gatk.hard_filtered.vcf.gz"
PASS_VCF="$OUTDIR/${SECTION}_gatk_pass_snvs.vcf.gz"
mkdir -p "$OUTDIR/tmp"
export PATH="$SPATIALSNV_ENV/bin:$PATH"
export CONDA_PREFIX="$SPATIALSNV_ENV"

INPUT_ARGS=()
for CHROM in "${USE_CHROMS[@]}"; do
    SHARD="$OUTDIR/scatter/${SECTION}.${CHROM}.vcf.gz"
    require_file "$SHARD"
    INPUT_ARGS+=(-I "$SHARD")
done
"$GATK" --java-options "-Xmx26g" GatherVcfs "${INPUT_ARGS[@]}" -O "$GATHERED" \
    --TMP_DIR "$OUTDIR/tmp" --CREATE_INDEX true
"$GATK" --java-options "-Xmx4g" IndexFeatureFile -I "$GATHERED"
"$GATK" --java-options "-Xmx26g" VariantFiltration \
    -R "$REF" -V "$GATHERED" -O "$FILTERED" \
    --filter-name RNA_FS --filter-expression 'FS > 30.0' \
    --filter-name RNA_QD --filter-expression 'QD < 2.0' \
    --cluster-window-size 35 --cluster-size 3
"$BCFTOOLS" view -f PASS -m2 -M2 -v snps -Oz -o "$PASS_VCF" "$FILTERED"
"$BCFTOOLS" index -f -t "$PASS_VCF"
require_file "$PASS_VCF"
echo "PASS biallelic SNVs: $("$BCFTOOLS" view -H "$PASS_VCF" | wc -l)"
echo "GATK gather/filter complete: mode=$MODE section=$SECTION; no BQSR was performed"
