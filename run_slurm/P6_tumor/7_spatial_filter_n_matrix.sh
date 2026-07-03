#!/bin/bash

#SBATCH --job-name=spatial_filter_n_matrix_P6
#SBATCH --output=slurm_output/P6_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P6.out
#SBATCH --error=slurm_output/P6_TUMOR/baseQ0mapQ0/spatial_filter_n_matrix_P6.err
#SBATCH --time=12:00:00
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

# Set VIZ_ONLY=1 to skip filtering and regenerate visualizations from existing outputs
VIZ_ONLY=${VIZ_ONLY:-0}

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "VIZ_ONLY: ${VIZ_ONLY}"

source activate snv_caller

mkdir -p slurm_output/P6_TUMOR/${QUALITY_FILTER}

for SECTION_ID in 1; do   # section 2 N/A for dedup ablation (no P6_sec2 CalicoST/spotprofiles inputs)
    echo "==============================================="
    echo "Processing P6_TUMOR replicate: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"

    # Section-specific CalicoST paths
    if [ "${SECTION_ID}" == "1" ]; then
        TUMOR_PURITY_FILE="${CALICOST_BASE}/P6_sec1/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
        CLONE_LABELS_FILE="${CALICOST_BASE}/P6_sec1/calicost/clone3_rectangle0_w1.0/clone_labels.tsv"
        CNV_SEGMENTS_FILE="${CALICOST_BASE}/P6_sec1/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv"
    elif [ "${SECTION_ID}" == "2" ]; then
        TUMOR_PURITY_FILE="${CALICOST_BASE}/P6_sec2/estimate_tumor_prop/loh_estimator_tumor_prop.tsv"
        CLONE_LABELS_FILE="${CALICOST_BASE}/P6_sec2/calicost/clone3_rectangle0_w1.0/clone_labels.tsv"
        CNV_SEGMENTS_FILE="${CALICOST_BASE}/P6_sec2/calicost/clone3_rectangle0_w1.0/cnv_seglevel.tsv"
    fi

    # Pool VCFs
    INCLUDE_VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P6_Somatic_Mutect2/P6_somatic_snp_chr1_22.vcf.gz"

    if [ "${VIZ_ONLY}" -eq 0 ]; then
        # Step 1: Run enhanced spatial SNV filtering
        echo "Running enhanced spatial SNV filtering (with clone+CNV integration)..."
        SPATIAL_ARGS=(
            --dataset p6_tumor
            --section_id "${SECTION_ID}"
            --quality_filter "${QUALITY_FILTER}"
            --tumor_purity_file "${TUMOR_PURITY_FILE}"
            --clone_labels "${CLONE_LABELS_FILE}"
            --cnv_segments "${CNV_SEGMENTS_FILE}"
            # --include_vcf "${INCLUDE_VCF}"
            --min_expression_germline 2
            --min_expression_somatic 1
            --neighbor_distance 2.0
            # --germline_threshold 0.4
            # --somatic_threshold 0.4
        )
        python scripts/6_spatial_filter/run_spatial_snv_filter_enhanced.py "${SPATIAL_ARGS[@]}"

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to run spatial SNV filtering for P6_TUMOR replicate ${SECTION_ID}"
            continue
        fi
        echo "Spatial SNV filtering for P6_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "VIZ_ONLY mode: skipping filter for P6_TUMOR replicate ${SECTION_ID}"
    fi

    # Visualization
    echo "Generating visualizations for P6_TUMOR replicate ${SECTION_ID}..."
    python scripts/6_spatial_filter/visualize_spatial_filter.py \
        --dataset p6_tumor \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER}

    if [ $? -eq 0 ]; then
        echo "Visualization for P6_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Visualization failed for P6_TUMOR replicate ${SECTION_ID}"
    fi

    # Step 2: Generate matrix
    echo "Generating matrix..."
    python scripts/6_spatial_filter/run_generate_matrix.py \
        --dataset p6_tumor \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --input-dir data/P6_tumor/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}/germline \
        --caller bcftools \
        --output-name normal

    if [ $? -eq 0 ]; then
        echo "Matrix generation for P6_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to generate matrix for P6_TUMOR replicate ${SECTION_ID}"
    fi

    echo "End time for P6_TUMOR replicate ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""

    # Step 3: Plot variant score scatter
    echo "Plotting variant score scatter..."

    FILTER_OUTPUT_DIR="/data/maiziezhou_lab/leiy4/snv_calling/data/P6_tumor/${SECTION_ID}/spatial_filter_purity/${QUALITY_FILTER}"

    python scripts/6_spatial_filter/plot_variant_scores.py \
        --input  ${FILTER_OUTPUT_DIR}/all_variant_scores.txt \
        --output_dir ${FILTER_OUTPUT_DIR}/plots \
        --title "P6 Tumor Section ${SECTION_ID} - ${QUALITY_FILTER}" \
        --germline_threshold 0.4 \
        --somatic_threshold 0.4 \
        --all

    if [ $? -eq 0 ]; then
        echo "Score scatter plots for P6_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to generate score scatter plots for P6_TUMOR replicate ${SECTION_ID}"
    fi

done

echo "All P6_TUMOR replicates processed"
echo "End time: $(date)"