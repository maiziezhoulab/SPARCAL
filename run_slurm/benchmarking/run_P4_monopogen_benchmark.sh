#!/bin/bash

#SBATCH --job-name=P4_monopogen_benchmark
#SBATCH --output=slurm_output/benchmark/monopogen/P4_monopogen_benchmark_%j.out
#SBATCH --error=slurm_output/benchmark/monopogen/P4_monopogen_benchmark_%j.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab
#SBATCH --partition=batch
##SBATCH --gres=gpu:nvidia_titan_x:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=128GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# P4 MONOPOGEN-specific configuration
MODEL="monopogen"
DATASET="P4_TUMOR"
QUALITY_FILTER="baseQ0mapQ0"
MAX_WORKERS=30

# MONOPOGEN VCF file paths for P4
MONOPOGEN_BASE="/lfs/archer.accre.vu/maiziezhou_lab/maiziezhou_lab/CanLuo/ST_SNV/Monopogen"

# Function to get Monopogen VCF paths for P4 sections
get_monopogen_p4_paths() {
    local section_id=$1
    
    # Monopogen paths for P4
    GERMLINE_VCF="${MONOPOGEN_BASE}/P4_rep${section_id}/out/germline/merged.vcf.gz"
    SOMATIC_CSV="${MONOPOGEN_BASE}/P4_rep${section_id}/out/somatic/somatic.csv"
    SOMATIC_VCF=""  # Monopogen somatic is in CSV format, not VCF
}

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "Running Monopogen benchmark BAM filtering for P4 dataset"
echo "Dataset: $DATASET"
echo "Model: $MODEL"

# Load required modules
module load miniconda3
source activate anaconda3/envs/DLwithTorch2_new/

# Create output directories
mkdir -p slurm_output/benchmark/monopogen

# Function to run Monopogen benchmark for P4 specific section
run_monopogen_p4_benchmark() {
    local section_id=$1
    
    # Get VCF paths for this section
    get_monopogen_p4_paths $section_id
    
    echo "==============================================="
    echo "Processing Monopogen benchmark for P4"  
    echo "Section ID: ${section_id}"
    echo "Germline VCF: ${GERMLINE_VCF}"
    echo "Somatic CSV: ${SOMATIC_CSV}"
    echo "Start time: $(date)"
    
    # Validate that VCF files exist
    if [ ! -f "$GERMLINE_VCF" ]; then
        echo "ERROR: Germline VCF file not found: $GERMLINE_VCF"
        return 1
    fi
    
    if [ ! -f "$SOMATIC_CSV" ]; then
        echo "WARNING: Somatic CSV file not found: $SOMATIC_CSV"
        echo "Will proceed with germline variants only"
    fi
    
    # Run the benchmark filtering (Monopogen mainly focuses on germline)
    ~/.conda/envs/snv_caller_new/bin/python scripts/5_refilter_bam/run_benchmark_models_bam_filter.py \
        --dataset ${DATASET} \
        --section-id ${section_id} \
        --model ${MODEL} \
        --quality-filter ${QUALITY_FILTER} \
        --max-workers ${MAX_WORKERS} \
        --germline-vcf "${GERMLINE_VCF}"
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "Section ${section_id} completed successfully"
        echo "Results saved to: data/p4_tumor/${section_id}/output_VCFs/BAM_filtered_monopogen/${QUALITY_FILTER}/"
    else
        echo "ERROR: Failed to process section ${section_id}"
        return 1
    fi
    
    echo "End time for section ${section_id}: $(date)"
    echo "==============================================="
    echo ""
    
    return 0
}

# Process P4 sections
echo "Processing P4_TUMOR dataset for Monopogen benchmark..."

for SECTION_ID in 1 2; do
    run_monopogen_p4_benchmark $SECTION_ID
    if [ $? -ne 0 ]; then
        echo "WARNING: Failed processing section $SECTION_ID, continuing..."
    fi
done

echo "Monopogen benchmark processing completed for P4!"
echo "End time: $(date)"

# Generate summary of results
echo ""
echo "MONOPOGEN P4 BENCHMARK SUMMARY"
echo "=============================="
echo "Filtered BAM files location: data/p4_tumor/*/output_VCFs/BAM_filtered_monopogen/${QUALITY_FILTER}/"
echo "SNV position files location: data/p4_tumor/*/output_VCFs/snv_positions/"
echo "Summary reports location: data/p4_tumor/*/logs/BAM_filtered_monopogen/${QUALITY_FILTER}/"
echo ""
echo "To check results, look for files like:"
echo "  - *_filtered.bam (filtered BAM files)"  
echo "  - *_snv_positions.txt (detected SNV positions)"
echo "  - monopogen_filtering_summary.txt (processing summary)"