#!/bin/bash
#SBATCH --job-name=step7_purityonly
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/step7_purityonly-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/step7_purityonly-%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --cpus-per-task=4
# Step 7 purity-only ablation (2026-10-12): CalicoST tumor purity is given, clone labels and CNV segments are
# not, so the a787c80 code uses its purity-only Stage-2 weights (delta, epsilon proxy, zeta). Run under both
# presence rules because the rule is not decided yet:
#   *_purityonly_record   record rule (any covering read = presence)
#   *_purityonly_altmol   >=1 ALT molecule per spot
set -euo pipefail
P=/data/maiziezhou_lab/leiy4/snv_calling
ENV=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller
export PATH=$ENV/bin:$P/apps:$PATH
PY=$ENV/bin/python
S7=$P/scripts/presence_fix/step7_a787c80_presence.py
CAL=/data/maiziezhou_lab/leiy4/CalicoST/hg19_rerun_20260906
case $SLURM_ARRAY_TASK_ID in
  0) C=P4; R=1; K=2 ;; 1) C=P4; R=2; K=2 ;; 2) C=P6; R=1; K=3 ;; 3) C=P6; R=2; K=3 ;;
esac
DS=${C,,}_tumor; ROOT=$P/data/${C}_tumor/$R/spatial_filter_purity
TAB=$P/data/presence_fix_2026-10-09/recount/${C}_rep${R}.tsv.gz
CR=$CAL/${C}_sec${R}; CD=$CR/calicost/clone${K}_rectangle0_w1.0
CNV=$CD/cnv_seglevel.tsv; [[ -s $CNV ]] || CNV=$CD/cnv_diploid_seglevel.tsv
FULL=(--tumor_purity_file $CR/estimate_tumor_prop/loh_estimator_tumor_prop.tsv --clone_labels $CD/clone_labels.tsv --cnv_segments $CNV --evidence_mode full)
COMMON=(--dataset $DS --section_id $R --quality_filter baseQ0mapQ0 --min_expression_germline 2 --min_expression_somatic 1 --neighbor_distance 2.0)
PUR=(--tumor_purity_file $CR/estimate_tumor_prop/loh_estimator_tumor_prop.tsv --evidence_mode full)
for d in baseQ0mapQ0_purityonly_record baseQ0mapQ0_purityonly_altmol; do
  [[ -e $ROOT/$d ]] && { echo "refusing to overwrite $ROOT/$d" >&2; exit 1; }
done
$PY $S7 "${COMMON[@]}" "${PUR[@]}" --presence_rule record --output_dir $ROOT/baseQ0mapQ0_purityonly_record
$PY $S7 "${COMMON[@]}" "${PUR[@]}" --presence_rule alt_mol --presence_table $TAB --output_dir $ROOT/baseQ0mapQ0_purityonly_altmol
