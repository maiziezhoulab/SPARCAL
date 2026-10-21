#!/bin/bash
# WES-blind P4/P6 section-1 rerun for the Beagle panel-routing correction.
# Each array task creates a new section tree and refuses to reuse an existing one.
#
# Submit from the repository root:
#   sbatch run_slurm/run_beagle_panel_routing_experiment.sh
#
# Optional: export RUN_LABEL to a new unique label before submission.
# No WES path is accepted or used anywhere in this workflow.

#SBATCH --job-name=beagle_route
#SBATCH --output=slurm_output/beagle_panel_routing/%A_%a.out
#SBATCH --error=slurm_output/beagle_panel_routing/%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=50:00:00
#SBATCH --account=maiziezhou_lab_phd
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
cd "$PROJECT_DIR"

case "${SLURM_ARRAY_TASK_ID:-0}" in
    0)
        DATASET=P4_TUMOR
        DATASET_LOWER=p4_tumor
        SAMPLE_LABEL=P4_sec1
        CANONICAL_SUBDIR=P4_tumor
        CALICOST_CLONE=clone2_rectangle0_w1.0
        ;;
    1)
        DATASET=P6_TUMOR
        DATASET_LOWER=p6_tumor
        SAMPLE_LABEL=P6_sec1
        CANONICAL_SUBDIR=P6_tumor
        CALICOST_CLONE=clone3_rectangle0_w1.0
        ;;
    *)
        echo "ERROR: array index must be 0 (P4) or 1 (P6)"
        exit 2
        ;;
esac

SECTION_ID=1
QUALITY_FILTER=baseQ0mapQ0
RUN_LABEL=${RUN_LABEL:-beagle_pnonref090_full_20260913_v1}
EXPERIMENT_BASE="$PROJECT_DIR/data/experiments/$RUN_LABEL"
SECTION_ROOT="$EXPERIMENT_BASE/$SAMPLE_LABEL"
CANONICAL_ROOT="$PROJECT_DIR/data/$CANONICAL_SUBDIR/$SECTION_ID"
CALICOST_RUN="/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906/$SAMPLE_LABEL"
CALICOST_DIR="$CALICOST_RUN/calicost/$CALICOST_CLONE"
TUMOR_PURITY_FILE="$CALICOST_RUN/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
CLONE_LABELS="$CALICOST_DIR/clone_labels.tsv"
CNV_SEGMENTS="$CALICOST_DIR/cnv_seglevel.tsv"
CNV_SEGMENTS_FALLBACK="$CALICOST_DIR/cnv_diploid_seglevel.tsv"

if [ "$SECTION_ROOT" = "$CANONICAL_ROOT" ]; then
    echo "ERROR: experimental root resolves to canonical output root"
    exit 2
fi
if [ -e "$SECTION_ROOT" ]; then
    echo "ERROR: refusing to overwrite existing experimental tree: $SECTION_ROOT"
    exit 2
fi
mkdir -p "$SECTION_ROOT"
trap 'printf "failed_at=%s\n" "$(date --iso-8601=seconds)" > "$SECTION_ROOT/RUN_FAILED.txt"' ERR

# Freeze every Python source used by this job before any analysis. This avoids
# mixing code revisions when the shared working tree changes during a long run.
CODE_SNAPSHOT="$SECTION_ROOT/code_snapshot"
mkdir -p "$CODE_SNAPSHOT"
find scripts -type f \
    \( -name '*.py' -o -name '*.sh' -o -name '*.R' -o -name '*.yaml' -o -name '*.yml' \) \
    -print0 | xargs -0 cp --parents --target-directory="$CODE_SNAPSHOT"
cp "$PROJECT_DIR/run_slurm/run_beagle_panel_routing_experiment.sh" "$CODE_SNAPSHOT/"
find "$CODE_SNAPSHOT" -type f -print0 | sort -z | xargs -0 sha256sum \
    > "$SECTION_ROOT/CODE_SNAPSHOT_SHA256SUMS.txt"

for required in "$TUMOR_PURITY_FILE" "$CLONE_LABELS"; do
    [ -s "$required" ] || { echo "ERROR: required CalicoST file missing: $required"; exit 1; }
done
if [ ! -s "$CNV_SEGMENTS" ]; then
    [ -s "$CNV_SEGMENTS_FALLBACK" ] || {
        echo "ERROR: both CNV segment files are missing"; exit 1;
    }
    CNV_SEGMENTS="$CNV_SEGMENTS_FALLBACK"
fi

source activate snv_caller

printf "run_label=%s\ndataset=%s\nsection=%s\nsection_root=%s\nwes_input=NONE\nanchor_score=p_nonref\nmin_anchor_probability=0.90\nweak_label_score=called_gp\nweak_label_min_probability=0.90\ngit_head=%s\nstarted=%s\n" \
    "$RUN_LABEL" "$DATASET" "$SECTION_ID" "$SECTION_ROOT" "$(git rev-parse HEAD)" "$(date --iso-8601=seconds)" \
    > "$SECTION_ROOT/RUN_MANIFEST.txt"

python "$CODE_SNAPSHOT/scripts/1_calling/mpileup_pipeline.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --base_quality 0 --mapping_quality 0 --threads 30 --filter_out_tissue \
    --section-root "$SECTION_ROOT"

python "$CODE_SNAPSHOT/scripts/2_beagle_filtering/run_beagle.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --quality-filter "$QUALITY_FILTER" --threads 30 --parallel-chromosomes 1 \
    --anchor-score p_nonref --min-anchor-probability 0.90 \
    --section-root "$SECTION_ROOT"

python "$CODE_SNAPSHOT/scripts/2_beagle_filtering/run_beagle_genotype_shifting.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" --section-root "$SECTION_ROOT"

# This called-GP cutoff is only for zygosity-specific weak-label calibration.
# It does not re-filter the authoritative P(nonref) hard-anchor route.
python "$CODE_SNAPSHOT/scripts/3_classifier_prep/run_sequence_error_model.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" --min-beagle-gp 0.90 \
    --section-root "$SECTION_ROOT"

python "$CODE_SNAPSHOT/scripts/4_classifier/run_supplimentary_models.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --quality-filter "$QUALITY_FILTER" --model-type neural_network \
    --max-training-samples 90000 --min-beagle-gp 0.90 \
    --section-root "$SECTION_ROOT"

python "$CODE_SNAPSHOT/scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py" \
    --dataset "$DATASET" --section-id "$SECTION_ID" \
    --quality-filter "$QUALITY_FILTER" --max-workers 30 \
    --classifier neural_network --section-root "$SECTION_ROOT"

python "$CODE_SNAPSHOT/scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py" \
    --dataset "$DATASET_LOWER" --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" --section-root "$SECTION_ROOT" \
    --tumor_purity_file "$TUMOR_PURITY_FILE" --clone_labels "$CLONE_LABELS" \
    --cnv_segments "$CNV_SEGMENTS" --min_expression_germline 2 \
    --min_expression_somatic 1 --neighbor_distance 2.0

python "$CODE_SNAPSHOT/scripts/6_spatial_filter/generate_sparcal_matrices.py" \
    --dataset "$DATASET" --section_id "$SECTION_ID" \
    --quality_filter "$QUALITY_FILTER" --section-root "$SECTION_ROOT"

printf "completed=%s\n" "$(date --iso-8601=seconds)" >> "$SECTION_ROOT/RUN_MANIFEST.txt"
touch "$SECTION_ROOT/RUN_COMPLETE"
trap - ERR
echo "Complete: $SECTION_ROOT"
