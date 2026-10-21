#!/usr/bin/env bash
set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
mkdir -p "$OUTROOT/_smoke_chr21" "$LOGROOT"

SSNV=$(sbatch --parsable "$SCRIPTROOT/run_spatialsnv_callback.slurm" smoke P4_rep2)
STRELKA=$(sbatch --parsable "$SCRIPTROOT/run_strelka2.slurm" smoke P4_rep2)
GATK_SPLIT=$(sbatch --parsable "$SCRIPTROOT/run_gatk_split.slurm" smoke P4_rep2)
GATK_HC=$(sbatch --parsable --dependency=afterok:$GATK_SPLIT "$SCRIPTROOT/run_gatk_hc.slurm" smoke P4_rep2 chr21)
GATK_FILTER=$(sbatch --parsable --dependency=afterok:$GATK_HC "$SCRIPTROOT/run_gatk_gather_filter.slurm" smoke P4_rep2)
PROJECT_JOB=$(sbatch --parsable --dependency=afterok:$STRELKA:$GATK_FILTER "$SCRIPTROOT/run_projection.slurm" smoke P4_rep2)
VALIDATE=$(sbatch --parsable --dependency=afterok:$SSNV:$PROJECT_JOB "$SCRIPTROOT/validate_smoke.slurm")

printf 'stage\tjob_id\tdependency\nspatialsnv_callback\t%s\t-\nstrelka2\t%s\t-\ngatk_split\t%s\t-\ngatk_haplotypecaller_chr21\t%s\tafterok:%s\ngatk_gather_filter\t%s\tafterok:%s\nprojection_and_1kgp\t%s\tafterok:%s:%s\nvalidate\t%s\tafterok:%s:%s\n' \
    "$SSNV" "$STRELKA" "$GATK_SPLIT" "$GATK_HC" "$GATK_SPLIT" \
    "$GATK_FILTER" "$GATK_HC" "$PROJECT_JOB" "$STRELKA" "$GATK_FILTER" \
    "$VALIDATE" "$SSNV" "$PROJECT_JOB" > "$OUTROOT/job_ids_smoke.tsv"
cat "$OUTROOT/job_ids_smoke.tsv"
echo "Smoke terminal job: $VALIDATE"
