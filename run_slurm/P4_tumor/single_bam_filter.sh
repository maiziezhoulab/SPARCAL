#!/bin/bash

#SBATCH --job-name=filter_bams_P4
#SBATCH --output=slurm_output/P4_TUMOR/baseQ0mapQ0/filter_bams_P4.out
#SBATCH --error=slurm_output/P4_TUMOR/baseQ0mapQ0/filter_bams_P4.err
#SBATCH --time=24:00:00
#SBATCH --account=cgw_maizie2
#SBATCH --partition=cgw-maizie2
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
module load Anaconda3
source activate snv_caller_new

# Make sure the output directory exists
mkdir -p slurm_output/P4_TUMOR/${QUALITY_FILTER}

# Only run section 1
~/.conda/envs/snv_caller_new/bin/python scripts/postprocess/run_filter_bams_by_snv_pools.py \
    --dataset P4_TUMOR \
    --section-id 1 \
    --quality-filter ${QUALITY_FILTER} \
    --max-workers 30

# Run the filter_bams script for each section
# for SECTION_ID in 1 2; do
#     echo "==============================================="
#     echo "Processing P4_TUMOR section: ${SECTION_ID}"
#     echo "Quality filter: ${QUALITY_FILTER}"
#     echo "Start time: $(date)"
    
#     # Run the filter_bams_by_snv_pools.py script
#     python scripts/postprocess/run_filter_bams_by_snv_pools.py \
#         --dataset P4_TUMOR \
#         --section-id ${SECTION_ID} \
#         --quality-filter ${QUALITY_FILTER} \
#         --max-workers 30
    
#     # Check if the script ran successfully
#     if [ $? -eq 0 ]; then
#         echo "P4_TUMOR section ${SECTION_ID} completed successfully"
#     else
#         echo "ERROR: Failed to process P4_TUMOR section ${SECTION_ID}"
#     fi
    
#     echo "End time for P4_TUMOR section ${SECTION_ID}: $(date)"
#     echo "==============================================="
#     echo ""
# done

echo "All P4_TUMOR sections processed"
echo "End time: $(date)"