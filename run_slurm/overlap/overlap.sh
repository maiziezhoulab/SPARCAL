#!/bin/bash

# ========================================
# CALLER VCF GROUP (uncomment only ONE)
# ========================================
# SPARCAL result file
# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/all_filtered_variants.chr.vcf.gz"
# CALLER_LABEL="SPARCAL"

# SPARCAL Exome
# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/all_filtered_variants.chr.exome.vcf.gz"
# CALLER_LABEL="SPARCAL_exome"


# Previous BEAGLE result
# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.exome.vcf.gz"
# CALLER_LABEL="BEAGLE_IN_exome"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_out.exome.vcf.gz"
# CALLER_LABEL="BEAGLE_OUT_exome"

# Alternative: SPARCAL (different version)
# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/BAM_filtered/baseQ0mapQ0/all_detected_variants_summary.vcf.gz"
# CALLER_LABEL="SPARCAL_v2"

# Alternative: BEAGLE
# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz"
# CALLER_LABEL="BEAGLE"

# Alternative: MPILEUP
CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz"
CALLER_LABEL="MPILEUP"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/defined/somatic_defined.vcf.gz"
# CALLER_LABEL="SPARCAL_Somatic_Defined"
#------------------------------

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.exome.vcf.gz"
# CALLER_LABEL="SPARCAL_Germline_Exome_spatial"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.exome.vcf.gz"
# CALLER_LABEL="SPARCAL_Somatic_Exome_spatial"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/Classifier/baseQ0mapQ0/results/neural_network_high_confidence.vcf.gz"
# CALLER_LABEL="HOMO"

# -----------------------------

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/germline/germline_variants.vcf.gz"
# CALLER_LABEL="SPARCAL_Germline_All_spatial"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.vcf.gz"
# CALLER_LABEL="SPARCAL_Somatic_All_spatial"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/somatic_variants.vcf.gz"
# CALLER_LABEL="SPARCAL_Somatic_All_spatial"

# CALLER_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz"
# CALLER_LABEL="Mpileup"

# ========================================
# COMPARISON VCF FILES (add more as needed)
# ========================================
declare -A VCF_FILES=(
    # WES Comparisons
    # ["P4_tumor_wes_all"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_cSCC_WES/P4_cSCC_WES_gatk_snp_chr1_22.vcf.gz"
    # ["P4_tumor_wes_exome"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_cSCC_WES/P4_cSCC_WES_gatk_exome_snps_chr1_22.vcf.gz"
    # ["P4_normal_wes_all"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz"
    # ["P4_normal_wes_exome"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Normal_WES/P4_Normal_WES_gatk_exome_snps_chr1_22.vcf.gz"
    
    # Somatic Comparisons
    # ["P4_somatic_GATK_all"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_GATK/P4_Somatic_GATK_snp_chr1_22.vcf.gz"
    # ["P4_somatic_GATK_exome"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_GATK/P4_Somatic_GATK_exome_snp_chr1_22.vcf.gz"
    
    ["P4_somatic_Mutect2_all"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
    # ["P4_somatic_Mutect2_exome"]="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz"
)

# ========================================
# SETUP
# ========================================
# Create output directory
OUTPUT_DIR="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_${CALLER_LABEL}"
mkdir -p ${OUTPUT_DIR}

# Log file
LOG_FILE="${OUTPUT_DIR}/comparison_log.txt"
echo "Starting comprehensive ${CALLER_LABEL} comparison at $(date)" > ${LOG_FILE}
echo "Caller VCF: ${CALLER_VCF}" | tee -a ${LOG_FILE}
echo "" | tee -a ${LOG_FILE}

# Verify caller VCF exists
if [ ! -f "${CALLER_VCF}" ]; then
    echo "ERROR: Caller VCF file not found: ${CALLER_VCF}" | tee -a ${LOG_FILE}
    exit 1
fi

# ========================================
# RUN COMPARISONS
# ========================================
for label in "${!VCF_FILES[@]}"; do
    echo "================================================" | tee -a ${LOG_FILE}
    echo "Comparing ${CALLER_LABEL} with ${label}" | tee -a ${LOG_FILE}
    echo "================================================" | tee -a ${LOG_FILE}
    
    vcf_file="${VCF_FILES[$label]}"
    output_prefix="overlap_${CALLER_LABEL}_${label}"
    
    # Check if VCF file exists
    if [ ! -f "${vcf_file}" ]; then
        echo "WARNING: ${vcf_file} not found, skipping..." | tee -a ${LOG_FILE}
        continue
    fi
    
    # Run bcftools isec
    echo "Running bcftools isec..." | tee -a ${LOG_FILE}
    bcftools isec -n=2 -w1 -O z -p ${OUTPUT_DIR}/${output_prefix} \
        ${CALLER_VCF} \
        ${vcf_file} 2>&1 | tee -a ${LOG_FILE}
    
    # Run Python overlap analysis
    echo "Running Python overlap analysis..." | tee -a ${LOG_FILE}
    python overlap.py \
        --file1 ${CALLER_VCF} \
        --file2 ${vcf_file} \
        --overlap ${OUTPUT_DIR}/${output_prefix} \
        --label1 "${CALLER_LABEL}" \
        --label2 "${label}" 2>&1 | tee -a ${LOG_FILE}
    
    echo "Completed ${label} at $(date)" | tee -a ${LOG_FILE}
    echo "" | tee -a ${LOG_FILE}
done

# ========================================
# GENERATE SUMMARY STATISTICS
# ========================================
echo "================================================" | tee -a ${LOG_FILE}
echo "GENERATING SUMMARY STATISTICS" | tee -a ${LOG_FILE}
echo "================================================" | tee -a ${LOG_FILE}

for label in "${!VCF_FILES[@]}"; do
    output_prefix="overlap_${CALLER_LABEL}_${label}"
    result_dir="${OUTPUT_DIR}/${output_prefix}"
    
    if [ -d "${result_dir}" ]; then
        echo "Summary for ${label}:" | tee -a ${LOG_FILE}
        
        # Count variants in each file
        if [ -f "${result_dir}/0000.vcf.gz" ]; then
            count_0000=$(bcftools view -H ${result_dir}/0000.vcf.gz | wc -l)
            echo "  ${CALLER_LABEL} unique: ${count_0000}" | tee -a ${LOG_FILE}
        fi
        
        if [ -f "${result_dir}/0001.vcf.gz" ]; then
            count_0001=$(bcftools view -H ${result_dir}/0001.vcf.gz | wc -l)
            echo "  ${label} unique: ${count_0001}" | tee -a ${LOG_FILE}
        fi
        
        if [ -f "${result_dir}/0002.vcf.gz" ]; then
            count_0002=$(bcftools view -H ${result_dir}/0002.vcf.gz | wc -l)
            echo "  Overlapping variants: ${count_0002}" | tee -a ${LOG_FILE}
        fi
        
        echo "" | tee -a ${LOG_FILE}
    fi
done

echo "================================================" | tee -a ${LOG_FILE}
echo "All comparisons completed at $(date)" | tee -a ${LOG_FILE}
echo "Results saved in: ${OUTPUT_DIR}" | tee -a ${LOG_FILE}