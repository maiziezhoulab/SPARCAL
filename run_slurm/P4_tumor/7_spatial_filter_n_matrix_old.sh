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
MIN_NEIGHBOURS=1
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# Make sure the output directory exists
mkdir -p slurm_output/P4_TUMOR/${QUALITY_FILTER}

# Process P4_TUMOR replicates 1 and 2
for SECTION_ID in 1 2; do
    echo "==============================================="
    echo "Processing P4_TUMOR replicate: ${SECTION_ID}"
    echo "Quality filter: ${QUALITY_FILTER}"
    echo "Start time: $(date)"
    
    # Step 1: Run spatial SNV filtering
    echo "Running spatial SNV filtering..."
    python scripts/6_spatial_filter/run_spatial_snv_filter.py \
        --dataset p4_tumor \
        --section_id ${SECTION_ID} \
        --quality_filter ${QUALITY_FILTER} \
        --min_neighbours ${MIN_NEIGHBOURS} \
        
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "Spatial SNV filtering for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to run spatial SNV filtering for P4_TUMOR replicate ${SECTION_ID}"
        continue
    fi
    
    # Step 2: Generate matrix
    echo "Generating matrix..."
    python scripts/6_spatial_filter/run_generate_matrix.py \
        --dataset p4_tumor \
        --section_id ${SECTION_ID} \
        --quality-filter ${QUALITY_FILTER} \
        --filter-subdir filtered_snvs \
        --output-name normal
    
    # Check if the script ran successfully
    if [ $? -eq 0 ]; then
        echo "Matrix generation for P4_TUMOR replicate ${SECTION_ID} completed successfully"
    else
        echo "ERROR: Failed to generate matrix for P4_TUMOR replicate ${SECTION_ID}"
    fi
    
    echo "End time for P4_TUMOR replicate ${SECTION_ID}: $(date)"
    echo "==============================================="
    echo ""
done

echo "All P4_TUMOR replicates processed"
echo "End time: $(date)"