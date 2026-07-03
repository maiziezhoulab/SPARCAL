#!/bin/bash

#SBATCH --job-name=spatial_filter_n_matrix_P4
#SBATCH --output=slurm_output/P4_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P4.out
#SBATCH --error=slurm_output/P4_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P4.err
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

# Set base quality and mapping quality
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

# CalicoST output base paths (section-specific, filled in loop below)
CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"

# Set VIZ_ONLY=1 to skip filtering and regenerate visualizations from existing outputs
VIZ_ONLY=${VIZ_ONLY:-0}

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "VIZ_ONLY: ${VIZ_ONLY}"

# Load required modules
source activate snv_caller

# Make sure the output directory exists
mkdir -p slurm_output/P4_TUMOR/${QUALITY_FILTER}

# Process P4_TUMOR replicates 1 and 2
for SECTION_ID in 1; do   # dedup ablation: P4 rep1 (section 1) only
    echo "==============================================="
    echo "Processing P4_TUMOR replicate: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"

    CALICOST_BASE="/data/maiziezhou_lab/leiy4/CalicoST"

    # Section-specific paths
    if [ "${SECTION_ID}" == "1" ]; then
        TUMOR_PURITY_FILE="${CALICOST_BASE}/P4_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
        CLONE_LABELS_FILE="${CALICOST_BASE}/P4_sec1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv"
        CNV_SEGMENTS_FILE="${CALICOST_BASE}/P4_sec1/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv"
    elif [ "${SECTION_ID}" == "2" ]; then
        TUMOR_PURITY_FILE="${CALICOST_BASE}/P4_sec2/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
        CLONE_LABELS_FILE="${CALICOST_BASE}/P4_sec2/calicost/clone3_rectangle0_w1.0/clone_labels.tsv"
        CNV_SEGMENTS_FILE="${CALICOST_BASE}/P4_sec2/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv"
    fi
    # Pool VCFs
    EXCLUDE_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/${SECTION_ID}/output_VCFs/beagle/${QUALITY_FILTER}/all_filtered_in.vcf.gz"
    INCLUDE_VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
    INCLUDE_VCF_EXOME="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
    KEPT_VARIANTS_VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz"

    if [ "${VIZ_ONLY}" -eq 0 ]; then
        # Step 1: Run enhanced spatial SNV filtering
        echo "Running enhanced spatial SNV filtering (with clone+CNV integration)..."
        SPATIAL_ARGS=(
            --dataset p4_tumor
            --section_id "${SECTION_ID}"
            --quality_filter "${QUALITY_FILTER}"
            --tumor_purity_file "${TUMOR_PURITY_FILE}"
            --clone_labels "${CLONE_LABELS_FILE}"
            --cnv_segments "${CNV_SEGMENTS_FILE}"
            # --exclude_vcf "${EXCLUDE_VCF}"
            # --include_vcf "${INCLUDE_VCF_EXOME}"
            # --kept_variants "${EXCLUDE_VCF}"
            --min_expression_germline 2
            --min_expression_somatic 1
            --neighbor_distance 2.0
            # --germline_threshold 0.3
            # --somatic_threshold 0.2
        )
        python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py "${SPATIAL_ARGS[@]}"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run spatial SNV filtering for P4_TUMOR replicate ${SECTION_ID}"
            continue
        fi
        echo "Spatial SNV filtering for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "VIZ_ONLY mode: skipping filter for P4_TUMOR replicate ${SECTION_ID}"
    fi

    # Visualization
    echo "Generating visualizations for P4_TUMOR replicate ${SECTION_ID}..."
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset p4_tumor \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER}

    if [ $? -eq 0 ]; then
        echo "Visualization for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Visualization failed for P4_TUMOR replicate ${SECTION_ID}"
    fi

    # Step 2: Generate matrix
    echo "Generating matrix..."
    python scripts/6_spatial_filter/run_generate_matrix.py \
        --dataset p4_tumor \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --input-dir data/P4_tumor/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}/germline \
        --caller bcftools \
        --output-name normal

    if [ $? -eq 0 ]; then
        echo "Matrix generation for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to generate matrix for P4_TUMOR replicate ${SECTION_ID}"
    fi

    echo "End time for P4_TUMOR replicate ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""

    # Step 3: Plot variant score scatter
    echo "Plotting variant score scatter..."

    FILTER_OUTPUT_DIR="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}"

    python scripts/6_spatial_filter/plot_variant_scores.py \
        --input  ${FILTER_OUTPUT_DIR}/all_variant_scores.txt \
        --output_dir ${FILTER_OUTPUT_DIR}/plots \
        --title "P4 Tumor Section ${SECTION_ID} - ${QUALITY_FILTER}" \
        --germline_threshold 0.3 \
        --somatic_threshold 0.2 \
        --all

    if [ $? -eq 0 ]; then
        echo "Score scatter plots for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to generate score scatter plots for P4_TUMOR replicate ${SECTION_ID}"
    fi

done

echo "All P4_TUMOR replicates processed"
echo "End time: $(date)"