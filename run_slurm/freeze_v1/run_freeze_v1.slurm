#!/bin/bash
# SPARCAL model freeze v1 (2026-10-20): rerun every tumor section through the frozen model
# into a NEW root, data/freeze_v1/<dataset>/<section>. Legacy outputs are never touched.
#
# Frozen model = frozen Beagle release (beagle_v2_3_default_20260916_v1, p_nonref >= 0.90)
#              + G1 presence (S6+ molecule recount) + CalicoST gate (CNV default-on, purity opt-in)
#              + clone-enrichment theta. See run_slurm/freeze_v1/README.md.
#
#   sbatch run_slurm/freeze_v1/run_freeze_v1.slurm              # all 7 elements
#   sbatch --array=2,3 run_slurm/freeze_v1/run_freeze_v1.slurm  # P6 only
#   START_STEP=7 sbatch --array=2 run_slurm/freeze_v1/run_freeze_v1.slurm   # resume an element
#
# Array: 0 P4 rep1 · 1 P4 rep2 · 2 P6 rep1 · 3 P6 rep2 · 4 DCIS1 · 5 DCIS2 · 6 OVAR_P5
#
#SBATCH --job-name=sparcal_freeze_v1
#SBATCH --array=0-6%3
#SBATCH --output=slurm_output/freeze_v1/%x-%A_%a.out
#SBATCH --error=slurm_output/freeze_v1/%x-%A_%a.err
#SBATCH --time=96:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

set -o pipefail
PROJECT=/data/maiziezhou_lab/leiy4/snv_calling
cd "$PROJECT"
FREEZE=data/freeze_v1
QF=baseQ0mapQ0

case "${SLURM_ARRAY_TASK_ID}" in
    0) KIND=P4;   SID=1;       ROOT=$FREEZE/P4_tumor/1 ;;
    1) KIND=P4;   SID=2;       ROOT=$FREEZE/P4_tumor/2 ;;
    2) KIND=P6;   SID=1;       ROOT=$FREEZE/P6_tumor/1 ;;
    3) KIND=P6;   SID=2;       ROOT=$FREEZE/P6_tumor/2 ;;
    4) KIND=DCIS; SID=1;       ROOT=$FREEZE/dcis1 ;;
    5) KIND=DCIS; SID=2;       ROOT=$FREEZE/dcis2 ;;
    6) KIND=OVAR; SID=P5_sr13; ROOT=$FREEZE/ovar_p5/P5_sr13 ;;
    *) echo "ERROR: unknown array index ${SLURM_ARRAY_TASK_ID}"; exit 2 ;;
esac
# Step 0 (UMI dedup/split) already exists for P4/P6/DCIS; only OVAR needs it.
DEFAULT_START=1; [ "$KIND" = OVAR ] && DEFAULT_START=0
START=${START_STEP:-$DEFAULT_START}

mkdir -p "$ROOT" slurm_output/freeze_v1
export SECTION_ROOT="$PROJECT/$ROOT"
echo "=== freeze_v1 | ${KIND} ${SID} | root ${SECTION_ROOT} | start ${START} | $(date) ==="
echo "code commit: $(git rev-parse HEAD 2>/dev/null) (+ working-tree changes; see README)"

case "$KIND" in
    P4)   bash run_slurm/P4_tumor/run_pipeline_P4.sh "$START" "$SID" 8 || exit $? ;;
    P6)   bash run_slurm/P6_tumor/run_pipeline_P6.sh "$START" "$SID" 8 || exit $? ;;
    DCIS) SLURM_ARRAY_TASK_ID=$SID bash run_slurm/DCIS/run_pipeline_DCIS.sh "$START" || exit $? ;;
    OVAR) bash run_slurm/ovar_p5/run_pipeline_OVAR.sh "$START" 8 || exit $? ;;
esac

# P6 only: CalicoST-free comparison arm (arm A) from the SAME frozen bundle. The production
# step 7 above (gate on) is arm B. Evaluate with scripts/postanalyze/p6_cnv_gate_eval.py.
if [ "$KIND" = P6 ]; then
    source activate snv_caller 2>/dev/null || true
    # Pin the env by absolute path: jobs must not depend on the submitting shell's conda setup.
    export PATH=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin:$PATH
    ARM_A_TAG=${QF}_spatial_only
    ARM_A_DIR="$SECTION_ROOT/spatial_filter_purity/$ARM_A_TAG"
    if [ -e "$ARM_A_DIR/RUN_COMPLETE" ]; then
        echo "[skip] arm A already complete: $ARM_A_DIR"
    else
        python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
            --dataset p6_tumor --section_id "$SID" --quality_filter "$QF" \
            --section-root "$SECTION_ROOT" --output_dir "$ARM_A_DIR" \
            --evidence_mode spatial_only_no_calicost \
            --molecule_table "$SECTION_ROOT/output_VCFs/spotprofiles/$QF/molecule_recount.tsv.gz" \
            --presence_rule G1 \
            --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0 || exit $?
        python scripts/6_spatial_filter/generate_sparcal_matrices.py \
            --dataset P6_TUMOR --section_id "$SID" --quality_filter "$QF" \
            --section-root "$SECTION_ROOT" --spatial-filter-tag "$ARM_A_TAG" \
            --model SPARCAL_spatial_only || exit $?
    fi
fi
echo "=== freeze_v1 complete | ${KIND} ${SID} | $(date) ==="
