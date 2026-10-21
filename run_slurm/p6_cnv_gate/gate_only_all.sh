#!/bin/bash
set -euo pipefail

PROJECT_DIR=/data/maiziezhou_lab/leiy4/snv_calling
CALICOST_ROOT=/data/maiziezhou_lab/leiy4/CalicoST/hg19_joint_20261013
PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
CALICOST_PYTHON=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/calicost_env/bin/python
OUT_DIR="$PROJECT_DIR/data/calicost_gate_2026-10-19"

if [[ -e "$OUT_DIR" ]]; then
    echo "ERROR: refusing existing output path: $OUT_DIR" >&2
    exit 1
fi
mkdir -p "$OUT_DIR"
work_dir=$(mktemp -d /tmp/sparcal_calicost_gate.XXXXXX)

for sample_id in P41 P42 P61 P62; do
    cohort=${sample_id:0:2}
    rep=${sample_id:2:1}
    clone_count=2
    [[ "$cohort" == P6 ]] && clone_count=3
    cohort_root="$CALICOST_ROOT/$cohort"
    clone_dir="$cohort_root/calicost_unsup/clone${clone_count}_rectangle0_w1.0"
    h5=/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/${cohort}_Visium/spaceranger_align_rep${rep}_hg19/${cohort}_Tumor_output/outs/filtered_feature_bc_matrix.h5
    spot_umi="$work_dir/${sample_id}_spot_umi.tsv"
    "$CALICOST_PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/spot_umi_from_h5.py" \
        --h5 "$h5" --out "$spot_umi"
    "$PYTHON" "$PROJECT_DIR/scripts/6_spatial_filter/calicost_gate.py" \
        --clone_labels "$clone_dir/clone_labels.tsv" \
        --cnv_segments "$clone_dir/cnv_seglevel.tsv" \
        --posterior "$clone_dir/posterior_clone_probability.npy" \
        --purity "$cohort_root/estimate_tumor_prop/loh_estimator_tumor_prop.tsv" \
        --sample_id "$sample_id" \
        --spot_umi_tsv "$spot_umi" \
        --out "$OUT_DIR/${sample_id}.json"
done

echo "Gate JSONs written to $OUT_DIR"
