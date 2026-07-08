#!/bin/bash

#SBATCH --job-name=spatial_filter_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/baseQ0mapQ0/spatial_filter_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/baseQ0mapQ0/spatial_filter_OVAR_P5.err
#SBATCH --time=4:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"
SECTION_ID=P5_sr13

DATA_BASE="/data/maiziezhou_lab/leiy4/snv_calling/data/ovar_p5"

# CalicoST outputs for P5 (colleague's read-only run). NOTE: this CalicoST run did NOT
# produce a tumor-purity file (no estimate_tumor_prop/), so --tumor_purity_file is omitted
# (it is optional). Confirm the clone-model dir (clone3_rectangle0_w1.0) is the one you want.
CALICOST_DIR="/data/maiziezhou_lab/Pankaj/calicost_p5/results/P5_calicost_cna/clone3_rectangle0_w1.0"
CLONE_LABELS_FILE="${CALICOST_DIR}/clone_labels.tsv"
CNV_SEGMENTS_FILE="${CALICOST_DIR}/cnv_seglevel.tsv"

# NOTE: do NOT pass --exclude_vcf/--kept_variants here. all_filtered_in = the beagle-KEPT
# 1KGP-concordant "defined" germline variants; excluding it strips every defined variant
# from the per-spot sets (empties the 1000G matrix, leaves germline denovo-only). P4/P6/DCIS
# pass neither flag, so their defined variants survive into the per-spot germline files.
# (Fixed 2026-07-07 — was `--exclude_vcf ${BEAGLE_VCF} --kept_variants ${BEAGLE_VCF}`.)

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

mkdir -p slurm_output/OVAR_P5/${QUALITY_FILTER}

echo "Processing OVAR_P5 section: ${SECTION_ID} | ${QUALITY_FILTER}"

# --- Stage 7a: spatial SNV filter (germline / UPV / somatic partition) ---
python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
    --dataset ovar_p5 \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER} \
    --clone_labels ${CLONE_LABELS_FILE} \
    --cnv_segments ${CNV_SEGMENTS_FILE} \
    --min_expression_germline 2 \
    --min_expression_somatic 1 \
    --neighbor_distance 2.0 \
    --germline_threshold 0.5 \
    --somatic_threshold 0.2

if [ $? -ne 0 ]; then
    echo "ERROR: spatial SNV filtering failed for ${SECTION_ID}"
    exit 1
fi
echo "Spatial SNV filtering completed."

# --- Stage 7b: spot x SNV matrix (germline set -> 'normal' matrix, comparable to benchmarks) ---
python scripts/6_spatial_filter/run_generate_matrix.py \
    --dataset ovar_p5 \
    --section_id ${SECTION_ID} \
    --quality-filter ${QUALITY_FILTER} \
    --input-dir ${DATA_BASE}/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}/germline \
    --caller bcftools \
    --output-name normal

echo "End time: $(date)"
