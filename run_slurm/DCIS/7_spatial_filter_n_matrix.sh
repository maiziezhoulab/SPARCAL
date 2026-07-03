#!/bin/bash

#SBATCH --job-name=spatial_filter_DCIS
#SBATCH --output=slurm_output/DCIS/baseQ0mapQ0/spatial_filter_DCIS.out
#SBATCH --error=slurm_output/DCIS/baseQ0mapQ0/spatial_filter_DCIS.err
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

CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"
DATA_BASE="/data/maiziezhou_lab/leiy4/snv_calling/data/dcis"

# Set VIZ_ONLY=1 to skip filtering and regenerate visualizations from existing outputs
VIZ_ONLY=${VIZ_ONLY:-0}

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "VIZ_ONLY: ${VIZ_ONLY}"

source activate snv_caller

mkdir -p slurm_output/DCIS/${QUALITY_FILTER}

for SECTION_ID in dcis1 dcis2; do
    echo "==============================================="
    echo "Processing DCIS section: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"

    if [ "${VIZ_ONLY}" -eq 0 ]; then
        # Capitalise section suffix for CalicoST paths: dcis1 -> DCIS1
        SECTION_UPPER=$(echo "${SECTION_ID}" | tr '[:lower:]' '[:upper:]')

        TUMOR_PURITY_FILE="${CALICOST_BASE}/${SECTION_UPPER}/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
        CLONE_LABELS_FILE="${CALICOST_BASE}/${SECTION_UPPER}/calicost/clone3_rectangle0_w1.0/clone_labels.tsv"
        CNV_SEGMENTS_FILE="${CALICOST_BASE}/${SECTION_UPPER}/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv"

        # Beagle output used for both exclude and kept_variants
        BEAGLE_VCF="${DATA_BASE}/${SECTION_ID}/output_VCFs/beagle/${QUALITY_FILTER}/all_filtered_in.vcf.gz"

        python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py \
            --dataset dcis \
            --section_id ${SECTION_ID} \
            --quality_filter ${QUALITY_FILTER} \
            --tumor_purity_file ${TUMOR_PURITY_FILE} \
            --clone_labels ${CLONE_LABELS_FILE} \
            --cnv_segments ${CNV_SEGMENTS_FILE} \
            --exclude_vcf ${BEAGLE_VCF} \
            --kept_variants ${BEAGLE_VCF} \
            --min_expression_germline 2 \
            --min_expression_somatic 1 \
            --neighbor_distance 2.0 \
            --germline_threshold 0.5 \
            --somatic_threshold 0.2

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run spatial SNV filtering for ${SECTION_ID}"
            continue
        fi
        echo "Spatial SNV filtering for ${SECTION_ID} completed successfully"
    else
        echo "VIZ_ONLY mode: skipping filter for ${SECTION_ID}"
    fi

    # Visualization (always runs, or re-runs when VIZ_ONLY=1)
    echo "Generating visualizations for ${SECTION_ID}..."
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset dcis \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER}

    if [ $? -eq 0 ]; then
        echo "Visualization for ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Visualization failed for ${SECTION_ID}"
    fi

    echo "End time for ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""

done

echo "All DCIS sections processed"
echo "End time: $(date)"