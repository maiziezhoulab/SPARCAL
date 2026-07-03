#!/bin/bash

#SBATCH --job-name=split_bam_OVAR_P5
#SBATCH --output=slurm_output/OVAR_P5/split_bam_OVAR_P5.out
#SBATCH --error=slurm_output/OVAR_P5/split_bam_OVAR_P5.err
#SBATCH --time=12:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

SECTION_ID=P5_sr13
COLLEAGUE_OUTS=/data/maiziezhou_lab/Pankaj/calicost_p5/spaceranger_runs/${SECTION_ID}/outs
INPUT_BAM=${COLLEAGUE_OUTS}/possorted_genome_bam.bam
POSITIONS=${COLLEAGUE_OUTS}/spatial/tissue_positions_list.csv
OUT_DIR=data/ovar_p5/${SECTION_ID}/split_BAM

echo "Input BAM : ${INPUT_BAM}"
echo "Positions : ${POSITIONS}"
echo "Output dir: ${OUT_DIR}"

# The colleague BAM carries a single @RG line, so `samtools split` (RG-based) cannot
# make per-barcode BAMs. Split by the CB tag with pysam instead (in-tissue spots only).
python scripts/0_split_bam/split_bam_by_cb.py \
    --bam       ${INPUT_BAM} \
    --positions ${POSITIONS} \
    --out-dir   ${OUT_DIR} \
    --in-tissue-only \
    --index \
    --threads 8

echo "Created $(ls ${OUT_DIR}/*.bam 2>/dev/null | wc -l) per-barcode BAM files"
echo "End time: $(date)"
