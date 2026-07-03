#!/bin/bash

#SBATCH --job-name=upv_baf_gmm_P6
#SBATCH --output=slurm_output/P6_tumor/baseQ0mapQ0_section1/upv_baf_gmm_P6.out
#SBATCH --error=slurm_output/P6_tumor/baseQ0mapQ0_section1/upv_baf_gmm_P6.err
#SBATCH --time=01:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# Step 7c — UPV BAF-GMM sub-filter for P6_TUMOR (section 1).
# Splits UPV (germline_denovo) into germline-like vs somatic-candidate via a 2-D
# GMM on [BAF, PURITY_CORR]. BAF is recomputed from I16 (the merged_sorted_gt BAF
# FORMAT field is 0 for high-depth sites — parse_i16 int() bug, fixed in
# mpileup_pipeline.py for future runs). See On_going.md.

QUALITY_FILTER="baseQ0mapQ0"
SECTION_ID=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller
mkdir -p slurm_output/P6_tumor/${QUALITY_FILTER}_section${SECTION_ID}

python scripts/6_spatial_filter/upv_baf_gmm_subfilter.py \
    --dataset P6_TUMOR \
    --section_id ${SECTION_ID} \
    --quality_filter ${QUALITY_FILTER} \
    --min-dp 5 --n-components 3 --baf-cut 0.40 --pur-cut 0.10 --somatic-baf-max 0.35

echo "End time: $(date)"
