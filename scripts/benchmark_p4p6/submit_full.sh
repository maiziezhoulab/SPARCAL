#!/usr/bin/env bash
set -euo pipefail
source /data/maiziezhou_lab/leiy4/snv_calling/scripts/benchmark_p4p6/common.sh
SMOKE_JSON="$OUTROOT/_smoke_chr21/smoke_validation.json"
require_file "$SMOKE_JSON"
"$SNV_CALLER_ENV/bin/python" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["status"] == "PASS", p' "$SMOKE_JSON"
mkdir -p "$LOGROOT"

SSNV=$(sbatch --parsable --array=0-1 "$SCRIPTROOT/run_spatialsnv_callback.slurm" full)
STRELKA=$(sbatch --parsable --array=0-3 "$SCRIPTROOT/run_strelka2.slurm" full)
GATK_SPLIT=$(sbatch --parsable --array=0-3 "$SCRIPTROOT/run_gatk_split.slurm" full)
GATK_HC=$(sbatch --parsable --array=0-95%24 --dependency=afterok:$GATK_SPLIT "$SCRIPTROOT/run_gatk_hc.slurm" full)
GATK_FILTER=$(sbatch --parsable --array=0-3 --dependency=afterok:$GATK_HC "$SCRIPTROOT/run_gatk_gather_filter.slurm" full)
PROJECT_JOB=$(sbatch --parsable --array=0-3%2 --dependency=afterok:$STRELKA:$GATK_FILTER "$SCRIPTROOT/run_projection.slurm" full)
SUMMARY=$(sbatch --parsable --dependency=afterok:$SSNV:$PROJECT_JOB "$SCRIPTROOT/summarize_outputs.slurm")

printf 'stage\tjob_id\tdependency\nspatialsnv_callback_array_0-1\t%s\t-\nstrelka2_array_0-3\t%s\t-\ngatk_split_array_0-3\t%s\t-\ngatk_haplotypecaller_array_0-95%%24\t%s\tafterok:%s\ngatk_gather_filter_array_0-3\t%s\tafterok:%s\nprojection_and_1kgp_array_0-3%%2\t%s\tafterok:%s:%s\nsummary\t%s\tafterok:%s:%s\n' \
    "$SSNV" "$STRELKA" "$GATK_SPLIT" "$GATK_HC" "$GATK_SPLIT" \
    "$GATK_FILTER" "$GATK_HC" "$PROJECT_JOB" "$STRELKA" "$GATK_FILTER" \
    "$SUMMARY" "$SSNV" "$PROJECT_JOB" > "$OUTROOT/job_ids_full.tsv"
cat "$OUTROOT/job_ids_full.tsv"
echo "Full terminal job: $SUMMARY"
