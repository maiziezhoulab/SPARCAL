#!/bin/bash

#SBATCH --job-name=P4_all_benchmarks
#SBATCH --output=slurm_output/benchmark/P4_all_benchmarks_%j.out
#SBATCH --error=slurm_output/benchmark/P4_all_benchmarks_%j.err
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

# P4 comprehensive benchmark configuration
DATASET="P4_TUMOR"
QUALITY_FILTER="baseQ0mapQ0"
MAX_WORKERS=30

# VCF file paths
MONOPOGEN_BASE="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/CanLuo/ST_SNV/Monopogen"
GATK_BASE="/lio/lfs/maiziezhou_lab/maiziezhou_lab/Datasets/ST_datasets/STmut_Data"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "Running comprehensive P4 benchmark comparison"
echo "Dataset: $DATASET"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Create output directories
mkdir -p slurm_output/benchmark/{monopogen,gatk}

# Function to run Monopogen benchmark for P4
run_monopogen_p4() {
    local section_id=$1
    
    echo ""
    echo "=================================================="
    echo "Running Monopogen benchmark for P4 section $section_id"
    echo "=================================================="
    
    local germline_vcf="${MONOPOGEN_BASE}/P4_rep${section_id}/out/germline/merged.vcf.gz"
    
    if [ ! -f "$germline_vcf" ]; then
        echo "ERROR: Monopogen germline VCF not found: $germline_vcf"
        return 1
    fi
    
    ~/.conda/envs/snv_caller_new/bin/python scripts/5_refilter_bam/run_benchmark_models_bam_filter.py \
        --dataset ${DATASET} \
        --section-id ${section_id} \
        --model "monopogen" \
        --quality-filter ${QUALITY_FILTER} \
        --max-workers ${MAX_WORKERS} \
        --germline-vcf "${germline_vcf}"
    
    return $?
}

# Function to run GATK benchmark variants for P4
run_gatk_p4() {
    local section_id=$1
    
    echo ""
    echo "=================================================="
    echo "Running GATK benchmarks for P4 section $section_id"
    echo "=================================================="
    
    # Define GATK variant files
    local -A gatk_variants=(
        ["tumor_wes_all"]="${GATK_BASE}/P4_cSCC_WES/P4_cSCC_WES_gatk_snp_chr1_22.vcf.gz"
        ["tumor_wes_exome"]="${GATK_BASE}/P4_cSCC_WES/P4_cSCC_WES_gatk_exome_snps_chr1_22.vcf.gz"
        ["normal_wes_all"]="${GATK_BASE}/P4_Normal_WES/P4_Normal_WES_gatk_snp_chr1_22.vcf.gz"
        ["normal_wes_exome"]="${GATK_BASE}/P4_Normal_WES/P4_Normal_WES_gatk_exome_snps_chr1_22.vcf.gz"
        ["somatic_gatk_all"]="${GATK_BASE}/P4_Somatic_GATK/P4_Somatic_GATK_snp_chr1_22.vcf.gz"
        ["somatic_gatk_exome"]="${GATK_BASE}/P4_Somatic_GATK/P4_Somatic_GATK_exome_snp_chr1_22.vcf.gz"
        ["somatic_mutect2_all"]="${GATK_BASE}/P4_Somatic_Mutect2/P4_somatic_snp_chr1_22.vcf.gz"
        ["somatic_mutect2_exome"]="${GATK_BASE}/P4_Somatic_Mutect2/P4_somatic_exome_snps_chr1_22.vcf.gz"
    )
    
    local failed_variants=()
    
    # Process each GATK variant type
    for variant_type in "${!gatk_variants[@]}"; do
        local vcf_file="${gatk_variants[$variant_type]}"
        
        echo "Processing GATK variant: $variant_type"
        echo "VCF file: $vcf_file"
        
        if [ ! -f "$vcf_file" ]; then
            echo "ERROR: VCF file not found: $vcf_file"
            failed_variants+=("$variant_type")
            continue
        fi
        
        ~/.conda/envs/snv_caller_new/bin/python scripts/5_refilter_bam/run_benchmark_models_bam_filter.py \
            --dataset ${DATASET} \
            --section-id ${section_id} \
            --model "gatk_${variant_type}" \
            --quality-filter ${QUALITY_FILTER} \
            --max-workers ${MAX_WORKERS} \
            --merged-vcf "${vcf_file}"
        
        if [ $? -eq 0 ]; then
            echo "✓ GATK $variant_type completed successfully"
        else
            echo "✗ GATK $variant_type failed"
            failed_variants+=("$variant_type")
        fi
        
        # Brief pause between variants
        sleep 5
    done
    
    if [ ${#failed_variants[@]} -gt 0 ]; then
        echo "WARNING: Failed GATK variants for section $section_id: ${failed_variants[*]}"
        return 1
    fi
    
    return 0
}

# Function to run all benchmarks for a P4 section
run_all_benchmarks_for_section() {
    local section_id=$1
    
    echo ""
    echo "######################################################"
    echo "PROCESSING ALL BENCHMARKS FOR P4 SECTION $section_id"
    echo "######################################################"
    
    local failed_models=()
    
    # Run Monopogen
    echo "Starting Monopogen benchmark..."
    if ! run_monopogen_p4 $section_id; then
        failed_models+=("monopogen")
        echo "WARNING: Monopogen failed for section $section_id"
    fi
    
    # Brief pause between models
    sleep 10
    
    # Run GATK variants
    echo "Starting GATK benchmarks..."
    if ! run_gatk_p4 $section_id; then
        failed_models+=("gatk")
        echo "WARNING: GATK benchmarks failed for section $section_id"
    fi
    
    # Report section summary
    echo ""
    echo "P4 SECTION $section_id SUMMARY:"
    echo "==============================="
    if [ ${#failed_models[@]} -eq 0 ]; then
        echo "✓ All benchmark models completed successfully"
    else
        echo "✗ Failed models: ${failed_models[*]}"
    fi
    echo ""
}

# Main execution: Process P4 sections
echo "Processing P4_TUMOR dataset with all benchmark models..."

for SECTION_ID in 1 2; do
    run_all_benchmarks_for_section $