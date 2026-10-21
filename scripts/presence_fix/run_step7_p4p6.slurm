#!/bin/bash
#SBATCH --job-name=presence_step7
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/step7-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/step7-%A_%a.err
#SBATCH --array=0-3
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
# Step 7 (a787c80 code + --presence_rule) on P4/P6 rep1/rep2. CalicoST inputs and args mirror
# run_slurm/p4p6_calicost_ablation/run_spatial_full_ablation_array.slurm. Three arms per section:
#   *_presfix_record_full_calicost  record rule: must reproduce baseQ0mapQ0_full_calicost (2026-09-07) exactly
#   *_altmol_full_calicost          >=1 ALT molecule per spot, full CalicoST evidence
#   *_altmol_ablation_no_calicost   >=1 ALT molecule per spot, spatial-only ablation
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
for d in baseQ0mapQ0_presfix_record_full_calicost baseQ0mapQ0_altmol_full_calicost baseQ0mapQ0_altmol_ablation_no_calicost; do
  [[ -e $ROOT/$d ]] && { echo "refusing to overwrite $ROOT/$d" >&2; exit 1; }
done
$PY $S7 "${COMMON[@]}" "${FULL[@]}" --presence_rule record --output_dir $ROOT/baseQ0mapQ0_presfix_record_full_calicost
$PY $S7 "${COMMON[@]}" "${FULL[@]}" --presence_rule alt_mol --presence_table $TAB --output_dir $ROOT/baseQ0mapQ0_altmol_full_calicost
$PY $S7 "${COMMON[@]}" --evidence_mode spatial_only_no_calicost --presence_rule alt_mol --presence_table $TAB --output_dir $ROOT/baseQ0mapQ0_altmol_ablation_no_calicost
# reproduction check of the record arm against the 2026-09-07 output
for cls in somatic/denovo/somatic_denovo germline/denovo/germline_denovo ambiguous/denovo/ambiguous_denovo; do
  a=$(zcat $ROOT/baseQ0mapQ0_full_calicost/$cls.vcf.gz | grep -v '^#' | cut -f1,2,4,5 | sort | md5sum)
  b=$(zcat $ROOT/baseQ0mapQ0_presfix_record_full_calicost/$cls.vcf.gz | grep -v '^#' | cut -f1,2,4,5 | sort | md5sum)
  [[ $a == $b ]] && echo "REPRODUCED $cls" || echo "MISMATCH $cls"
done
