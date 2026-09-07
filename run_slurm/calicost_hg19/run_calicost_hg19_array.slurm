#!/bin/bash
#SBATCH --job-name=calicost_hg19
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=25
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906/slurm_output/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906/slurm_output/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
CALICOST_DIR=/data/maiziezhou_lab/leiy4/CalicoST
OUTPUT_ROOT=/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906
MANIFEST="$PROJECT_DIR/run_slurm/calicost_hg19/samples.tsv"
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python
SNAKEMAKE=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/snakemake

sample=$(awk -F '\t' -v id="$SLURM_ARRAY_TASK_ID" 'NR > 1 && $1 == id {print $2}' "$MANIFEST")
if [[ -z "$sample" ]]; then
    echo "No manifest entry for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 2
fi
sample_dir="$OUTPUT_ROOT/$sample"

"$PYTHON" "$PROJECT_DIR/scripts/calicost/prepare_hg19_rerun.py" \
    --manifest "$MANIFEST" --array-id "$SLURM_ARRAY_TASK_ID" --output-root "$OUTPUT_ROOT"
"$PYTHON" "$PROJECT_DIR/scripts/calicost/validate_hg19_inputs.py" --sample-dir "$sample_dir"
# CalicoST's parse_input.py writes initial_phase.npz directly under each
# configured output_dir before it creates parsed_inputs/. Ensure both output
# roots exist for the purity and CNA passes.
mkdir -p "$sample_dir/estimate_tumor_prop" "$sample_dir/calicost"

# ACCRE's Lmod initialization references optional shell variables, so suspend
# nounset only while loading modules and restore it immediately afterwards.
set +u
module load StdEnv/2020
module load samtools/1.16
module load cellsnp-lite/1.2.2
set -u
export PYTHONPATH="$CALICOST_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1

cd "$sample_dir"
echo "[$(date --iso-8601=seconds)] Starting $sample on hg19"
"$SNAKEMAKE" --cores "$SLURM_CPUS_PER_TASK" \
    --configfile "$sample_dir/config_eagle2.yaml" \
    --snakefile "$CALICOST_DIR/src/calicost.smk" all
"$PYTHON" "$CALICOST_DIR/src/calicost/estimate_tumor_proportion.py" \
    -c "$sample_dir/configuration_purity"
"$PYTHON" "$CALICOST_DIR/src/calicost/calicost_main.py" \
    -c "$sample_dir/configuration_cna"

test -s "$sample_dir/snpinfo/unique_snp_ids.npy"
test -s "$sample_dir/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
cnv_file=
for clone_dir in clone3_rectangle0_w1.0 clone2_rectangle0_w1.0; do
    for cnv_name in cnv_seglevel.tsv cnv_diploid_seglevel.tsv; do
        candidate="$sample_dir/calicost/$clone_dir/$cnv_name"
        if [[ -s "$candidate" ]]; then
            cnv_file="$candidate"
            break 2
        fi
    done
done
if [[ -z "$cnv_file" ]]; then
    echo "ERROR: neither canonical nor diploid-fallback CNV segments were produced for $sample" >&2
    exit 1
fi
echo "[$(date --iso-8601=seconds)] Validated CNV segments: $cnv_file"
echo "[$(date --iso-8601=seconds)] Completed $sample on hg19"
