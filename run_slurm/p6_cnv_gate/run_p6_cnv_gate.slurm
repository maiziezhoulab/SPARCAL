#!/bin/bash
#SBATCH --job-name=p6_cnv_gate
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p6_cnv_gate/%x-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/p6_cnv_gate/%x-%A_%a.err

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
CALICOST_ROOT=/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013/P6
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
CALICOST_PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python
QUALITY_FILTER=baseQ0mapQ0
SPATIAL_TAG=baseQ0mapQ0_cnvgate_spatial_only
CNV_TAG=baseQ0mapQ0_cnvgate_cnv

case "${SLURM_ARRAY_TASK_ID}" in
    0) SECTION_ID=1; SAMPLE_ID=P61 ;;
    1) SECTION_ID=2; SAMPLE_ID=P62 ;;
    *) echo "ERROR: unexpected array index ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

section_root="$PROJECT_DIR/data/P6_tumor/$SECTION_ID"
spatial_out="$section_root/spatial_filter_purity/$SPATIAL_TAG"
cnv_out="$section_root/spatial_filter_purity/$CNV_TAG"
for output_dir in "$spatial_out" "$cnv_out"; do
    if [[ -e "$output_dir" ]]; then
        echo "ERROR: refusing existing output path: $output_dir" >&2
        exit 1
    fi
done

clone_dir="$CALICOST_ROOT/calicost_unsup/clone3_rectangle0_w1.0"
purity_file="$CALICOST_ROOT/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
clone_labels="$clone_dir/clone_labels.tsv"
cnv_segments="$clone_dir/cnv_seglevel.tsv"
posterior="$clone_dir/posterior_clone_probability.npy"
h5=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Visium/spaceranger_align_rep${SECTION_ID}_hg19/P6_Tumor_output/outs/filtered_feature_bc_matrix.h5
for required in "$PYTHON" "$CALICOST_PYTHON" "$purity_file" "$clone_labels" "$cnv_segments" "$posterior" "$h5"; do
    [[ -s "$required" ]] || { echo "ERROR: missing required input: $required" >&2; exit 1; }
done

scratch_root="${SLURM_TMPDIR:-/tmp}/sparcal_p6_cnvgate_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$scratch_root" "$PROJECT_DIR/slurm_output/p6_cnv_gate"
spot_umi="$scratch_root/${SAMPLE_ID}_spot_umi.tsv"
"$CALICOST_PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/spot_umi_from_h5.py" \
    --h5 "$h5" --out "$spot_umi"

echo "[$(date --iso-8601=seconds)] P6 rep${SECTION_ID}: spatial-only arm"
"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py" \
    --dataset p6_tumor \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --evidence_mode spatial_only_no_calicost \
    --output_dir "$spatial_out" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 2.0

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/generate_sparcal_matrices.py" \
    --dataset P6_TUMOR \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --spatial-filter-tag "$SPATIAL_TAG" \
    --model SPARCAL_cnvgate_spatial_only

echo "[$(date --iso-8601=seconds)] P6 rep${SECTION_ID}: gated CNV arm"
"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py" \
    --dataset p6_tumor \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --tumor_purity_file "$purity_file" \
    --clone_labels "$clone_labels" \
    --cnv_segments "$cnv_segments" \
    --clone_posterior "$posterior" \
    --spot_umi_tsv "$spot_umi" \
    --calicost_sample_id "$SAMPLE_ID" \
    --calicost_gate on \
    --cnv_feature clone_enrichment \
    --evidence_mode full \
    --output_dir "$cnv_out" \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 2.0

"$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/generate_sparcal_matrices.py" \
    --dataset P6_TUMOR \
    --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" \
    --spatial-filter-tag "$CNV_TAG" \
    --model SPARCAL_cnvgate_cnv

test -s "$spatial_out/RUN_COMPLETE"
test -s "$cnv_out/RUN_COMPLETE"
test -s "$section_root/matrix/P6_TUMOR_${SECTION_ID}_SPARCAL_cnvgate_spatial_only_somatic_matrix.pkl"
test -s "$section_root/matrix/P6_TUMOR_${SECTION_ID}_SPARCAL_cnvgate_cnv_somatic_matrix.pkl"
echo "[$(date --iso-8601=seconds)] P6 rep${SECTION_ID}: complete"
