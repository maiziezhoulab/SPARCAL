#!/usr/bin/env bash
#SBATCH --job-name=p4p6_strelka_rna
# RNA-mode copy of run_strelka2.slurm (adds --rna; caller token strelka2_rna). The plain
# germline run in */strelka2/ is kept unchanged for DLPFC parity.
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
MODE=${1:-full}
if [[ $MODE == full ]]; then
    SECTION=${SECTIONS[${SLURM_ARRAY_TASK_ID:?array task required}]}
else
    SECTION=${2:-P4_rep2}
fi
BAM=$(section_bam "$SECTION")
OUTDIR=$(caller_dir "$MODE" "$SECTION" strelka2_rna)
RUNDIR="$OUTDIR/workflow"
PASS_VCF="$OUTDIR/${SECTION}_strelka2_rna_pass_snvs.vcf.gz"
mkdir -p "$OUTDIR"
require_file "$BAM"
require_file "$REF"

export PATH="$STRELKA_ENV/bin:$PATH"
export STRELKA_CONFIG="$STRELKA_ENV/bin/configureStrelkaGermlineWorkflow.py"
echo "Strelka metadata: $(basename "$STRELKA_ENV"), bioconda strelka 2.9.10"

if [[ -s $PASS_VCF ]]; then
    echo "Existing completed PASS VCF found, leaving unchanged: $PASS_VCF"
    exit 0
fi
if [[ -e $RUNDIR ]]; then
    echo "Refusing to overwrite partial/existing Strelka workflow: $RUNDIR" >&2
    exit 1
fi

CONFIG_ARGS=(--rna --bam "$BAM" --referenceFasta "$REF" --runDir "$RUNDIR")
if [[ $MODE == smoke ]]; then
    BED="$OUTDIR/chr21.callRegions.bed"
    BEDGZ="$BED.gz"
    awk 'BEGIN{OFS="\t"}$1=="chr21"{print $1,0,$2}' "$REF.fai" > "$BED"
    "$BGZIP" -f "$BED"
    "$TABIX" -f -p bed "$BEDGZ"
    CONFIG_ARGS+=(--callRegions "$BEDGZ")
fi

"$STRELKA_ENV/bin/python" "$STRELKA_CONFIG" "${CONFIG_ARGS[@]}"
"$STRELKA_ENV/bin/python" "$RUNDIR/runWorkflow.py" -m local -j "$SLURM_CPUS_PER_TASK"

SOURCE_VCF=""
for candidate in \
    "$RUNDIR/results/variants/variants.vcf.gz" \
    "$RUNDIR/results/variants/genome.S1.vcf.gz" \
    "$RUNDIR/results/variants/genome.vcf.gz"; do
    if [[ -s $candidate ]]; then SOURCE_VCF=$candidate; break; fi
done
[[ -n $SOURCE_VCF ]] || { echo "No Strelka result VCF found under $RUNDIR/results/variants" >&2; exit 1; }
"$BCFTOOLS" view -f PASS -m2 -M2 -v snps -Oz -o "$PASS_VCF" "$SOURCE_VCF"
"$BCFTOOLS" index -f -t "$PASS_VCF"
require_file "$PASS_VCF"
echo "PASS biallelic SNVs: $("$BCFTOOLS" view -H "$PASS_VCF" | wc -l)"
echo "Strelka2 RNA-mode complete: mode=$MODE section=$SECTION"
