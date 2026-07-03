#!/bin/bash

#SBATCH --job-name=P4_gatk_benchmark
#SBATCH --output=slurm_output/benchmark/gatk/P4_gatk_benchmark_%j.out
#SBATCH --error=slurm_output/benchmark/gatk/P4_gatk_benchmark_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_acc
#SBATCH --partition=batch_gpu
#SBATCH --gres=gpu:nvidia_titan_x:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# P4 GATK-specific configuration
DATASET="P4_TUMOR"
QUALITY_FILTER="baseQ0mapQ0"
MAX_WORKERS=30

# GATK VCF file paths for P4
GATK_BASE="/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data"

# P4 GATK variant file paths
P4_TUMOR_WES_ALL="${GATK_BASE}/P4_cSCC_WES/P4_cSCC_WES_gatk_snp_chr1_22.vcf.gz"
P4_TUMOR_WES_EXOME="${GATK_BASE}/P4_cSCC_WES/P4_cSCC_WES_gatk_exome_snps_chr1_22.vcf.gz"
P4_NORMAL_WES_ALL="${GATK_BASE}/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz"
P4_NORMAL_WES_EXOME="${GATK_BASE}/P4_Normal_WES/P4_Normal_WES_gatk_exome_snps_chr1_22.vcf.gz"
P4_SOMATIC_GATK_ALL="${GATK_BASE}/P4_Somatic_GATK/P4_Somatic_GATK_snp_chr1_22.vcf.gz"
P4_SOMATIC_GATK_EXOME="${GATK_BASE}/P4_Somatic_GATK/P4_Somatic_GATK_exome_snp_chr1_22.vcf.gz"
P4_SOMATIC_MUTECT2_ALL="${GATK_BASE}/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
P4_SOMATIC_MUTECT2_EXOME="${GATK_BASE}/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "Running GATK benchmark BAM filtering for P4 dataset"
echo "Dataset: $DATASET"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Create output directories
mkdir -p slurm_output/benchmark/gatk

# Function to validate VCF file existence
validate_vcf_file() {
    local vcf_file=$1
    local description=$2
    
    if [ -f "$vcf_file" ]; then
        echo "  ✓ Found $description: $vcf_file"
        return 0
    else
        echo "  ✗ Missing $description: $vcf_file"
        return 1
    fi
}

# Function to run GATK benchmark for specific variant type
run_gatk_benchmark() {
    local variant_type=$1
    local merged_vcf=$2
    local section_id=$3
    local description=$4
    
    echo ""
    echo "=================================================="
    echo "Processing GATK benchmark: $description"
    echo "Variant type: $variant_type"
    echo "Section ID: $section_id"
    echo "VCF file: $merged_vcf"
    echo "Start time: $(date)"
    echo "=================================================="
    
    # Validate VCF file exists
    if ! validate_vcf_file "$merged_vcf" "$description"; then
        echo "ERROR: VCF file not found. Skipping $variant_type..."
        return 1
    fi
    
    # Create a unique model name combining gatk with variant type
    local model_name="gatk_${variant_type}"
    
    # Run the benchmark filtering
    ~/.conda/envs/snv_caller_new/bin/python scripts/5_refilter_bam/run_benchmark_models_bam_filter.py \
        --dataset ${DATASET} \
        --section-id ${section_id} \
        --model ${model_name} \
        --quality-filter ${QUALITY_FILTER} \
        --max-workers ${MAX_WORKERS} \
        --merged-vcf "${merged_vcf}"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ $description completed successfully for section $section_id"
        echo "Results saved to: data/p4_tumor/${section_id}/output_VCFs/BAM_filtered_${model_name}/${QUALITY_FILTER}/"
    else
        echo "✗ $description failed for section $section_id (exit code: $exit_code)"
        return 1
    fi
    
    echo "End time: $(date)"
    echo "=================================================="
    
    return $exit_code
}

# Function to run all GATK variants for a specific P4 section
run_all_gatk_variants_for_section() {
    local section_id=$1
    
    echo ""
    echo "######################################################"
    echo "PROCESSING ALL GATK VARIANTS FOR P4 SECTION $section_id"
    echo "######################################################"
    
    local failed_variants=()
    
    # Run each GATK variant type
    echo "Running P4 Tumor WES (all SNPs)..."
    if ! run_gatk_benchmark "tumor_wes_all" "$P4_TUMOR_WES_ALL" "$section_id" "P4 Tumor WES All SNPs"; then
        failed_variants+=("tumor_wes_all")
    fi
    
    echo "Running P4 Tumor WES (exome SNPs)..."
    if ! run_gatk_benchmark "tumor_wes_exome" "$P4_TUMOR_WES_EXOME" "$section_id" "P4 Tumor WES Exome SNPs"; then
        failed_variants+=("tumor_wes_exome")
    fi
    
    echo "Running P4 Normal WES (all SNPs)..."
    if ! run_gatk_benchmark "normal_wes_all" "$P4_NORMAL_WES_ALL" "$section_id" "P4 Normal WES All SNPs"; then
        failed_variants+=("normal_wes_all")
    fi
    
    echo "Running P4 Normal WES (exome SNPs)..."
    if ! run_gatk_benchmark "normal_wes_exome" "$P4_NORMAL_WES_EXOME" "$section_id" "P4 Normal WES Exome SNPs"; then
        failed_variants+=("normal_wes_exome")
    fi
    
    echo "Running P4 Somatic GATK (all SNPs)..."
    if ! run_gatk_benchmark "somatic_gatk_all" "$P4_SOMATIC_GATK_ALL" "$section_id" "P4 Somatic GATK All SNPs"; then
        failed_variants+=("somatic_gatk_all")
    fi
    
    echo "Running P4 Somatic GATK (exome SNPs)..."
    if ! run_gatk_benchmark "somatic_gatk_exome" "$P4_SOMATIC_GATK_EXOME" "$section_id" "P4 Somatic GATK Exome SNPs"; then
        failed_variants+=("somatic_gatk_exome")
    fi
    
    echo "Running P4 Somatic Mutect2 (all SNPs)..."
    if ! run_gatk_benchmark "somatic_mutect2_all" "$P4_SOMATIC_MUTECT2_ALL" "$section_id" "P4 Somatic Mutect2 All SNPs"; then
        failed_variants+=("somatic_mutect2_all")
    fi
    
    echo "Running P4 Somatic Mutect2 (exome SNPs)..."
    if ! run_gatk_benchmark "somatic_mutect2_exome" "$P4_SOMATIC_MUTECT2_EXOME" "$section_id" "P4 Somatic Mutect2 Exome SNPs"; then
        failed_variants+=("somatic_mutect2_exome")
    fi
    
    # Report section summary
    echo ""
    echo "P4 SECTION $section_id SUMMARY:"
    echo "==============================="
    if [ ${#failed_variants[@]} -eq 0 ]; then
        echo "✓ All GATK variant types completed successfully"
    else
        echo "✗ Failed variant types: ${failed_variants[*]}"
    fi
    echo ""
}

# Main execution: Process P4 sections
echo "Processing P4_TUMOR dataset with all GATK variant types..."

# Validate all VCF files first
echo ""
echo "Validating all GATK VCF files..."
echo "================================"
validate_vcf_file "$P4_TUMOR_WES_ALL" "P4 Tumor WES All SNPs"
validate_vcf_file "$P4_TUMOR_WES_EXOME" "P4 Tumor WES Exome SNPs" 
validate_vcf_file "$P4_NORMAL_WES_ALL" "P4 Normal WES All SNPs"
validate_vcf_file "$P4_NORMAL_WES_EXOME" "P4 Normal WES Exome SNPs"
validate_vcf_file "$P4_SOMATIC_GATK_ALL" "P4 Somatic GATK All SNPs"
validate_vcf_file "$P4_SOMATIC_GATK_EXOME" "P4 Somatic GATK Exome SNPs"
validate_vcf_file "$P4_SOMATIC_MUTECT2_ALL" "P4 Somatic Mutect2 All SNPs"
validate_vcf_file "$P4_SOMATIC_MUTECT2_EXOME" "P4 Somatic Mutect2 Exome SNPs"

# Process P4 sections
for SECTION_ID in 1 2; do
    run_all_gatk_variants_for_section $SECTION_ID
done

echo ""
echo "########################################################"
echo "ALL P4 GATK BENCHMARKS COMPLETED"
echo "########################################################"
echo "Dataset: $DATASET"
echo "End time: $(date)"

# Generate comprehensive summary
echo ""
echo "P4 GATK BENCHMARK SUMMARY"
echo "========================="
echo "Processed variant types:"
echo "  - gatk_tumor_wes_all"
echo "  - gatk_tumor_wes_exome" 
echo "  - gatk_normal_wes_all"
echo "  - gatk_normal_wes_exome"
echo "  - gatk_somatic_gatk_all"
echo "  - gatk_somatic_gatk_exome"
echo "  - gatk_somatic_mutect2_all"
echo "  - gatk_somatic_mutect2_exome"
echo ""
echo "Output locations:"
echo "  Filtered BAMs: data/p4_tumor/*/output_VCFs/BAM_filtered_gatk_*/${QUALITY_FILTER}/"
echo "  SNV positions: data/p4_tumor/*/output_VCFs/snv_positions/"
echo "  Summary reports: data/p4_tumor/*/logs/BAM_filtered_gatk_*/${QUALITY_FILTER}/"
echo ""
echo "To check results, look for directories like:"
echo "  - BAM_filtered_gatk_tumor_wes_all/"
echo "  - BAM_filtered_gatk_somatic_mutect2_exome/"
echo "  - etc."

# Generate comparison across all variants
python3 << 'EOF'
import os
import glob
from pathlib import Path

def generate_p4_gatk_summary():
    """Generate a summary of P4 GATK benchmark results."""
    
    print(f"\n{'='*80}")
    print("P4 GATK BENCHMARK RESULTS SUMMARY")
    print(f"{'='*80}")
    
    base_path = "/data/maiziezhou_lab/leiy4/snv_calling"
    quality_filter = "baseQ0mapQ0"
    
    gatk_variants = [
        "gatk_tumor_wes_all",
        "gatk_tumor_wes_exome", 
        "gatk_normal_wes_all",
        "gatk_normal_wes_exome",
        "gatk_somatic_gatk_all",
        "gatk_somatic_gatk_exome",
        "gatk_somatic_mutect2_all",
        "gatk_somatic_mutect2_exome"
    ]
    
    sections = ["1", "2"]
    
    print(f"Sections: {sections}")
    print(f"GATK variant types: {len(gatk_variants)}")
    print("")
    
    for section in sections:
        print(f"Section {section}:")
        print("-" * 40)
        
        for variant in gatk_variants:
            bam_dir = os.path.join(base_path, f"data/p4_tumor/{section}/output_VCFs/BAM_filtered_{variant}", quality_filter)
            
            if os.path.exists(bam_dir):
                bam_files = glob.glob(os.path.join(bam_dir, "*_filtered.bam"))
                num_bams = len(bam_files)
                status = f"✓ Complete ({num_bams} BAMs)"
            else:
                status = "✗ Missing"
                
            print(f"  {variant:<25}: {status}")
        print("")
    
    print(f"{'='*80}")

try:
    generate_p4_gatk_summary()
except Exception as e:
    print(f"Error generating summary: {e}")

EOF

echo ""
echo "P4 GATK benchmark pipeline completed!"