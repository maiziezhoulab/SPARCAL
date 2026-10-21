#!/bin/bash
#SBATCH --job-name=presence_recount
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/recount-%A_%a.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/slurm_output/presence_fix/recount-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
# Molecule-level recount of step-6 per-spot records (scripts/presence_fix/recount_spot_alleles.py).
# Tasks 0-11: DLPFC sections, restricted to the in_panel_germline columns and in-tissue rows of the 2026-09-19 Beagle
# experiment matrices. Tasks 12-15: P4/P6 rep1/rep2, all records, all spots (input to a step-7 rerun).
set -euo pipefail
P=/data/maiziezhou_lab/leiy4/snv_calling
O=$P/data/presence_fix_2026-10-09
PY=/data/maiziezhou_lab/download_yuqi/leiy4/anaconda3/envs/snv_caller/bin/python
OUTDIR=${OUTDIR:-$O/recount}
TASKS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676 P4_1 P4_2 P6_1 P6_2)
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
if [[ $T == P* ]]; then
    S=${T%_*}; R=${T#*_}
    $PY $P/scripts/presence_fix/recount_spot_alleles.py --spotprofile-dir $P/data/${S}_tumor/$R/output_VCFs/spotprofiles/baseQ0mapQ0 \
        --out $OUTDIR/${S}_rep${R}.tsv.gz --workers 16
else
    $PY $P/scripts/presence_fix/recount_spot_alleles.py --spotprofile-dir $P/data/dlpfc/$T/output_VCFs/spotprofiles/baseQ0mapQ0 \
        --bam-dir $P/data/dlpfc/$T/bam_bycell_dedup \
        --barcodes $O/inputs/bc_$T.txt --keys $O/inputs/keys_$T.txt --out $OUTDIR/DLPFC_$T.tsv.gz --workers 16
fi
