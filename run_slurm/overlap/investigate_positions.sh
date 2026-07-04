#!/bin/bash

# ========================================
# VCF Position Offset Investigation
# ========================================
# This script helps investigate and fix position offsets between VCF files
# VCF format uses 1-based coordinates, but some tools may have bugs/inconsistencies

# Define your VCF files
MPILEUP_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/mpileup_multi_bam/baseQ0mapQ0/merged_sorted_gt.vcf.gz"
SPARCAL_VCF="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/all_filtered_variants.chr.vcf.gz"
BEAGLE_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/beagle/baseQ0mapQ0/all_filtered_in.vcf.gz"
MUTECT2_VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
TUMOR_VCF="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data/P4_cSCC_WES/P4_cSCC_WES_gatk_snp_chr1_22.vcf.gz"

# Output directory
OUTPUT_DIR="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/position_investigation"
mkdir -p ${OUTPUT_DIR}

# ========================================
# STEP 1: Check for position offsets
# ========================================
echo "========================================="
echo "STEP 1: Checking for position offsets"
echo "========================================="
echo ""

echo "Comparing MPILEUP vs Tumor..."
python check_vcf_positions.py \
    --vcf1 ${MPILEUP_VCF} --label1 MPILEUP \
    --vcf2 ${TUMOR_VCF} --label2 TUMOR \
    > ${OUTPUT_DIR}/check_mpileup_vs_tumor.txt

# echo ""
# echo "Comparing SPARCAL vs MUTECT2..."
# python check_vcf_positions.py \
#     --vcf1 ${SPARCAL_VCF} --label1 SPARCAL \
#     --vcf2 ${MUTECT2_VCF} --label2 MUTECT2 \
#     > ${OUTPUT_DIR}/check_sparcal_vs_mutect2.txt

# echo ""
# echo "Comparing BEAGLE vs MUTECT2..."
# python check_vcf_positions.py \
#     --vcf1 ${BEAGLE_VCF} --label1 BEAGLE \
#     --vcf2 ${MUTECT2_VCF} --label2 MUTECT2 \
#     > ${OUTPUT_DIR}/check_beagle_vs_mutect2.txt

echo ""
echo "Results saved in: ${OUTPUT_DIR}"
echo ""
echo "Review the output files to determine which VCF needs adjustment."
echo ""

# ========================================
# STEP 2: Manual adjustment (commented out)
# ========================================
# Uncomment and modify based on STEP 1 results

# Example: If MPILEUP needs +1 adjustment
# echo "========================================="
# echo "STEP 2: Adjusting MPILEUP positions (+1)"
# echo "========================================="
# python adjust_vcf_positions.py \
#     --input ${MPILEUP_VCF} \
#     --output ${OUTPUT_DIR}/mpileup_adjusted_plus1.vcf.gz \
#     --offset +1
#
# echo "Indexing adjusted VCF..."
# tabix -p vcf ${OUTPUT_DIR}/mpileup_adjusted_plus1.vcf.gz

# Example: If MPILEUP needs -1 adjustment
# echo "========================================="
# echo "STEP 2: Adjusting MPILEUP positions (-1)"
# echo "========================================="
# python adjust_vcf_positions.py \
#     --input ${MPILEUP_VCF} \
#     --output ${OUTPUT_DIR}/mpileup_adjusted_minus1.vcf.gz \
#     --offset -1
#
# echo "Indexing adjusted VCF..."
# tabix -p vcf ${OUTPUT_DIR}/mpileup_adjusted_minus1.vcf.gz

# ========================================
# STEP 3: Validation (commented out)
# ========================================
# After adjustment, validate that positions now align correctly

# echo "========================================="
# echo "STEP 3: Validating adjusted positions"
# echo "========================================="
# python check_vcf_positions.py \
#     --vcf1 ${OUTPUT_DIR}/mpileup_adjusted_plus1.vcf.gz --label1 MPILEUP_ADJUSTED \
#     --vcf2 ${MUTECT2_VCF} --label2 MUTECT2 \
#     > ${OUTPUT_DIR}/validation_after_adjustment.txt
#
# echo "Validation results saved in: ${OUTPUT_DIR}/validation_after_adjustment.txt"
# echo "Check that 'Same position' matches are now high (>90%)"

# ========================================
# STEP 4: Compare overlap statistics
# ========================================
# Compare before and after adjustment

# echo "========================================="
# echo "STEP 4: Comparing overlap before/after"
# echo "========================================="
#
# echo "Original MPILEUP overlap..."
# bcftools isec -n=2 -w1 -O z -p ${OUTPUT_DIR}/overlap_original \
#     ${MPILEUP_VCF} ${MUTECT2_VCF}
# OVERLAP_ORIGINAL=$(bcftools view -H ${OUTPUT_DIR}/overlap_original/0002.vcf.gz | wc -l)
# echo "  Overlapping variants (original): ${OVERLAP_ORIGINAL}"
#
# echo "Adjusted MPILEUP overlap..."
# bcftools isec -n=2 -w1 -O z -p ${OUTPUT_DIR}/overlap_adjusted \
#     ${OUTPUT_DIR}/mpileup_adjusted_plus1.vcf.gz ${MUTECT2_VCF}
# OVERLAP_ADJUSTED=$(bcftools view -H ${OUTPUT_DIR}/overlap_adjusted/0002.vcf.gz | wc -l)
# echo "  Overlapping variants (adjusted): ${OVERLAP_ADJUSTED}"
#
# if [ ${OVERLAP_ADJUSTED} -gt ${OVERLAP_ORIGINAL} ]; then
#     echo "✓ Adjustment IMPROVED overlap (+$(($OVERLAP_ADJUSTED - $OVERLAP_ORIGINAL)) variants)"
# else
#     echo "✗ WARNING: Adjustment did not improve overlap"
# fi

echo ""
echo "========================================="
echo "Position investigation complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Review the check_*.txt files in ${OUTPUT_DIR}"
echo "2. Based on results, uncomment STEP 2 in this script to adjust positions"
echo "3. Run validation (STEP 3) to confirm the adjustment worked"
echo "4. Compare overlap statistics (STEP 4) to quantify improvement"