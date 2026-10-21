#!/bin/bash
#SBATCH --job-name=calicost_hg19_joint
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013/slurm_output/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013/slurm_output/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
CALICOST_DIR=/data/maiziezhou_lab/leiy4/CalicoST
OUTPUT_ROOT=/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013
MANIFEST="$PROJECT_DIR/run_slurm/calicost_hg19_joint/samples.tsv"
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python
SNAKEMAKE=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/snakemake

manifest_fields=$(awk -F '\t' -v id="$SLURM_ARRAY_TASK_ID" \
    'NR > 1 && $1 == id {print $2 "\t" $4 "\t" $7}' "$MANIFEST")
if [[ -z "$manifest_fields" ]]; then
    echo "No manifest entry for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
IFS=$'\t' read -r patient sample_id_sec1 sample_id_sec2 <<< "$manifest_fields"
patient_dir="$OUTPUT_ROOT/$patient"
status_file="$patient_dir/run_status.tsv"
mkdir -p "$OUTPUT_ROOT"
# Two copies of this array may be queued (interactive + batch); the first to start claims the patient.
if ! mkdir "$OUTPUT_ROOT/.lock_$patient" 2>/dev/null; then
    echo "$patient already claimed by job $(cat "$OUTPUT_ROOT/.lock_$patient/job" 2>/dev/null); exiting"; exit 0
fi
echo "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)" > "$OUTPUT_ROOT/.lock_$patient/job"
mkdir -p "$patient_dir"
printf 'step\texit_code\tnote\n' > "$status_file"

record_status() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$status_file"
}

set +e
"$PYTHON" "$PROJECT_DIR/scripts/calicost/prepare_hg19_joint.py" \
    --manifest "$MANIFEST" --array-id "$SLURM_ARRAY_TASK_ID" --output-root "$OUTPUT_ROOT"
prepare_rc=$?
set -e
record_status prepare "$prepare_rc" "materialize joint hg19 configuration"
if [[ "$prepare_rc" -ne 0 ]]; then echo "ERROR: prepare failed" >&2; record_status job 1 "aborted at prepare"; exit 1; fi

set +e
"$PYTHON" "$PROJECT_DIR/scripts/calicost/validate_hg19_joint_inputs.py" \
    --sample-dir "$patient_dir"
validate_rc=$?
set -e
record_status validate "$validate_rc" "validate both sections and hg19 resources"
if [[ "$validate_rc" -ne 0 ]]; then echo "ERROR: validate failed" >&2; record_status job 1 "aborted at validate"; exit 1; fi

mkdir -p "$patient_dir/estimate_tumor_prop" \
    "$patient_dir/calicost" "$patient_dir/calicost_unsup"
record_status mkdir 0 "created purity and both CNA output directories"

# ACCRE's Lmod initialization references optional shell variables, so suspend
# nounset only while loading modules and restore it immediately afterwards.
set +u
module load StdEnv/2020
module load samtools/1.16
module load cellsnp-lite/1.2.2
set -u
export PYTHONPATH="$CALICOST_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

cd "$patient_dir"
echo "[$(date --iso-8601=seconds)] Starting $patient joint hg19 run"

set +e
"$SNAKEMAKE" --cores "$SLURM_CPUS_PER_TASK" \
    --configfile "$patient_dir/config_eagle2.yaml" \
    --snakefile "$CALICOST_DIR/src/calicost.smk" all
snakemake_rc=$?
set -e
record_status snakemake "$snakemake_rc" "joint two-section SNP extraction and phasing"
if [[ "$snakemake_rc" -ne 0 ]]; then echo "ERROR: snakemake failed" >&2; record_status job 1 "aborted at snakemake"; exit 1; fi

# Purity is advisory: estimate_tumor_proportion.py can fail when no
# high-tumor clone exists. Its failure must not prevent unsupervised CNA.
set +e
"$PYTHON" "$CALICOST_DIR/src/calicost/estimate_tumor_proportion.py" \
    -c "$patient_dir/configuration_purity"
purity_rc=$?
set -e
record_status purity "$purity_rc" "LOH estimator; failure is nonfatal"

purity_tsv="$patient_dir/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
set +e
"$PYTHON" - "$purity_tsv" "$sample_id_sec1" "$sample_id_sec2" <<'PY'
import csv
import math
import sys
from pathlib import Path

path = Path(sys.argv[1])
sample_ids = sys.argv[2:]
values = {sample_id: [] for sample_id in sample_ids}
if path.is_file():
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            barcode = row.get("BARCODES", "")
            for sample_id in sample_ids:
                if barcode.endswith("_" + sample_id):
                    try:
                        values[sample_id].append(float(row.get("Tumor", "nan")))
                    except (TypeError, ValueError):
                        values[sample_id].append(float("nan"))
                    break

eligible = True
for sample_id in sample_ids:
    section_values = values[sample_id]
    finite = sum(math.isfinite(value) for value in section_values)
    total = len(section_values)
    fraction = finite / total if total else 0.0
    print(
        f"purity finite fraction {sample_id}: {finite}/{total} = {fraction:.6f}",
        flush=True,
    )
    eligible &= total > 0 and fraction >= 0.5
raise SystemExit(0 if eligible else 1)
PY
finite_rc=$?
set -e
if [[ "$finite_rc" -eq 0 ]]; then
    finite_note="each section has at least 50% finite purity values"
else
    finite_note="one or both sections have less than 50% finite purity values, or TSV is missing"
fi
record_status purity_finite_check "$finite_rc" "$finite_note"

find_segment_file() {
    local arm_dir=$1
    local found
    found=$(find "$arm_dir" -mindepth 2 -maxdepth 2 -type f \
        -path "$arm_dir/clone*_rectangle0_w1.0/cnv_seglevel.tsv" \
        -size +0c -print -quit 2>/dev/null)
    if [[ -z "$found" ]]; then
        found=$(find "$arm_dir" -mindepth 2 -maxdepth 2 -type f \
            -path "$arm_dir/clone*_rectangle0_w1.0/cnv_diploid_seglevel.tsv" \
            -size +0c -print -quit 2>/dev/null)
    fi
    printf '%s' "$found"
}

# This arm is unconditional, even when purity failed or was too sparse.
set +e
"$PYTHON" "$CALICOST_DIR/src/calicost/calicost_main.py" \
    -c "$patient_dir/configuration_cna_unsup"
unsup_rc=$?
set -e
record_status cna_unsup "$unsup_rc" "unconditional unsupervised CNA arm"
unsup_segment=$(find_segment_file "$patient_dir/calicost_unsup")
if [[ -n "$unsup_segment" ]]; then
    record_status cna_unsup_output 0 "$unsup_segment"
    echo "[$(date --iso-8601=seconds)] Unsupervised CNV segments: $unsup_segment"
else
    record_status cna_unsup_output 1 "no nonempty canonical or diploid-fallback seglevel file"
    echo "ERROR: unsupervised arm produced no accepted seglevel file for $patient" >&2
fi

if [[ "$purity_rc" -eq 0 && "$finite_rc" -eq 0 ]]; then
    set +e
    "$PYTHON" "$CALICOST_DIR/src/calicost/calicost_main.py" \
        -c "$patient_dir/configuration_cna"
    purity_cna_rc=$?
    set -e
    record_status cna_purity "$purity_cna_rc" "purity-guided CNA arm"
    purity_segment=$(find_segment_file "$patient_dir/calicost")
    if [[ -n "$purity_segment" ]]; then
        record_status cna_purity_output 0 "$purity_segment"
        echo "[$(date --iso-8601=seconds)] Purity-guided CNV segments: $purity_segment"
    else
        record_status cna_purity_output 1 "no nonempty canonical or diploid-fallback seglevel file"
    fi
else
    record_status cna_purity NA "skipped: purity command failed or per-section finite fraction was below 0.5"
fi

final_rc=0
if [[ "$snakemake_rc" -ne 0 || "$unsup_rc" -ne 0 || -z "$unsup_segment" ]]; then
    final_rc=1
fi
record_status job "$final_rc" "nonzero only for snakemake failure, unsupervised CNA failure, or missing unsupervised segments"
if [[ "$final_rc" -ne 0 ]]; then
    echo "[$(date --iso-8601=seconds)] Joint hg19 run failed required checks for $patient" >&2
    exit "$final_rc"
fi
echo "[$(date --iso-8601=seconds)] Completed $patient joint hg19 run"
