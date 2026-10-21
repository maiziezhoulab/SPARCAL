#!/usr/bin/env bash
#SBATCH --job-name=p4p6_project
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/benchmark_p4p6/%x-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G

set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
MODE=${1:-full}
if [[ $MODE == full ]]; then
    SECTION=${SECTIONS[${SLURM_ARRAY_TASK_ID:?array task required}]}
else
    SECTION=${2:-P4_rep2}
fi
BAM=$(section_bam "$SECTION")
POSITIONS=$(section_positions "$SECTION")
PY="$SNV_CALLER_ENV/bin/python"
MATRICES=()
for CALLER in strelka2 gatk; do
    OUTDIR=$(caller_dir "$MODE" "$SECTION" "$CALLER")
    VCF="$OUTDIR/${SECTION}_${CALLER}_pass_snvs.vcf.gz"
    require_file "$VCF"
    "$PY" "$SCRIPTROOT/project_vcf_to_spots.py" \
        --vcf "$VCF" --bam "$BAM" --tissue-positions "$POSITIONS" \
        --outdir "$OUTDIR" --section "$SECTION" --caller "$CALLER" \
        --min-alt-reads 1 --max-workers "$SLURM_CPUS_PER_TASK"
    MATRIX="$OUTDIR/${SECTION}_${CALLER}_presence_matrix.pkl"
    require_file "$MATRIX"
    MATRICES+=(--matrix "$MATRIX")
done
"$PY" "$SCRIPTROOT/drop_1kgp.py" "${MATRICES[@]}"
for CALLER in strelka2 gatk; do
    require_file "$(caller_dir "$MODE" "$SECTION" "$CALLER")/${SECTION}_${CALLER}_no1kgp_presence_matrix.pkl"
done
echo "Projection + 1KGP removal complete: mode=$MODE section=$SECTION"
