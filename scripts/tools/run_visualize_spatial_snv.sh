#!/bin/bash

# Test SNV spatial visualization with 5 SNVs

DATASET="P4_TUMOR"
SECTION_ID="1"

SNV_LIST_SOMATIC_VCF="/data/maiziezhou_lab/leiy4/snv_calling/run_slurm/overlap/comprehensive_comparison_MPILEUP/overlap_MPILEUP_P4_somatic_Mutect2_all/0000.vcf.gz"

SNV_LIST_OUR_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_analysis/baseQ0mapQ0/filtered_snvs/all_filtered_variants_chr.vcf.gz"

SNV_LIST_SOMATIC_OUR_VCF="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/spatial_filter_purity/baseQ0mapQ0/somatic/denovo/somatic_denovo.vcf.gz"

SNV_VCF_DIR="/data/maiziezhou_lab/leiy4/snv_calling/data/P4_tumor/1/output_VCFs/BAM_filtered/baseQ0mapQ0/snv_vcf"

OUTPUT_DIR="./our_somatic_real"

MAX_SNVS=10

rm -rf $OUTPUT_DIR

python visualize_snv_spatial_distribution.py \
    --dataset $DATASET \
    --section_id $SECTION_ID \
    --snv_list_vcf $SNV_LIST_SOMATIC_OUR_VCF \
    --snv_vcf_dir $SNV_VCF_DIR \
    --output_dir $OUTPUT_DIR \
    --max_snvs $MAX_SNVS

echo ""
echo "Test complete. Check: $OUTPUT_DIR/SNV/"