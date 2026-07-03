#!/bin/bash

#SBATCH --job-name=filter_bams_P6_TUMOR
#SBATCH --output=slurm_output/P6_TUMOR/baseQ0mapQ0/filter_single_bams_P6_TUMOR.out
#SBATCH --error=slurm_output/P6_TUMOR/baseQ0mapQ0/filter_single_bams_P6_TUMOR.err
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

# Set base quality and mapping quality
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# Make sure the output directory exists
mkdir -p slurm_output/P6_TUMOR/${QUALITY_FILTER}

# Process P6_TUMOR — only section 1 has data
for SECTION_ID in 1; do
    echo "==============================================="
    echo "Processing P6_TUMOR replicate: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Run the filter_bams_by_snv_pools.py script for each replicate
    # ~/.conda/envs/snv_caller/bin/python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
    python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py \
        --dataset P6_TUMOR \
        --section-id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --max-workers 30 \
        --classifier neural_network
    # python scripts/5_refilter_bam/run_filter_bams_by_snv_pools.py --dataset P6_TUMOR --section-id 1 --quality-filter baseQ0mapQ0 --classifier neural_network

    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "P6_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to process P6_TUMOR replicate ${SECTION_ID}"
    fi
    
    echo "End time for P6_TUMOR replicate ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""
done

echo "All P6_TUMOR replicates processed"
echo "End time: $(date)"