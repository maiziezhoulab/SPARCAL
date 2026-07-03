#!/bin/bash

#SBATCH --job-name=upv_baf_gmm_DCIS
#SBATCH --output=slurm_output/DCIS/baseQ0mapQ0/upv_baf_gmm_DCIS.out
#SBATCH --error=slurm_output/DCIS/baseQ0mapQ0/upv_baf_gmm_DCIS.err
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

# Step 7c — UPV BAF-GMM sub-filter (post-processing of step 7).
# Splits the UPV set (germline_denovo) into UPV-germline-like vs
# UPV-somatic-candidate via a 2-D GMM on [BAF, PURITY_CORR].
# Non-destructive: reads step-7 outputs, writes to .../germline/denovo/gmm_subfilter/.
#
# NOTE (2026-06-02): PURITY_CORR was found uninformative within UPV (clipped to >=0
# and flat, because UPV are ubiquitous by definition), so the split is effectively
# BAF-only. The planned upgrade is a per-clone BAF contrast (tumor vs normal spots)
# from spotprofiles/vcf_by_spot/ + CalicoST clone_labels — see On_going.md.

BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

source activate snv_caller

mkdir -p slurm_output/DCIS/${QUALITY_FILTER}

for SECTION_ID in dcis1 dcis2; do
    echo "==============================================="
    echo "UPV BAF-GMM sub-filter — DCIS section ${SECTION_ID} (${QUALITY_FILTER})"
    echo "Start time: $(date)"

    python scripts/6_spatial_filter/upv_baf_gmm_subfilter.py \
        --dataset DCIS \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER} \
        --min-dp 5 \
        --n-components 3 \
        --baf-cut 0.40 \
        --pur-cut 0.10 \
        --somatic-baf-max 0.35

    if [ $? -eq 0 ]; then
        echo "SUCCESS: ${SECTION_ID}"
    else
        echo "ERROR: UPV BAF-GMM failed for ${SECTION_ID}"
    fi
    echo "End time for ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""
done

echo "All DCIS sections processed"
echo "End time: $(date)"
