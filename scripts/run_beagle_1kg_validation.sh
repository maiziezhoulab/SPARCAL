#!/bin/bash

#SBATCH --job-name=beagle_1kg_validation
#SBATCH --output=beagle_1kg_validation.out
#SBATCH --error=beagle_1kg_validation.err
#SBATCH --time=60:00:00
#SBATCH --account=cgw_maizie
#SBATCH --partition=cgw-maizie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=400GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Set output directory for validation results
OUTPUT_DIR="/data/maiziezhou_lab/yuqi/snv_calling/analysis/beagle_1kg_validation"
mkdir -p $OUTPUT_DIR

# Set path to validation script
VALIDATION_SCRIPT="/data/maiziezhou_lab/yuqi/snv_calling/scripts/tools/Beagle_1kG_validation.py"

echo "Running Beagle 1000 Genomes validation..."
echo "-----------------------------------------"

# DLPFC samples - multiple sections
echo "Processing DLPFC dataset..."
# for SECTION_ID in 151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676; do
for SECTION_ID in 151507; do
    echo "  Processing DLPFC section ${SECTION_ID}..."
    python $VALIDATION_SCRIPT \
        --dataset DLPFC \
        --section-id $SECTION_ID \
        --threads 28 \
        --output-dir $OUTPUT_DIR \
        --af-threshold 0.01
    
    # Check if script execution was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Validation failed for DLPFC section ${SECTION_ID}"
    else
        echo "  Completed DLPFC section ${SECTION_ID}"
    fi
done

# P4_TUMOR samples - sections 1 and 2
echo "Processing P4_TUMOR dataset..."
for SECTION_ID in 1 ; do
    echo "  Processing P4_TUMOR section ${SECTION_ID}..."
    python $VALIDATION_SCRIPT \
        --dataset P4_TUMOR \
        --section-id $SECTION_ID \
        --threads 28 \
        --output-dir $OUTPUT_DIR \
        --af-threshold 0.01
    
    # Check if script execution was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Validation failed for P4_TUMOR section ${SECTION_ID}"
    else
        echo "  Completed P4_TUMOR section ${SECTION_ID}"
    fi
done

# P6_TUMOR samples - sections 1 and 2
echo "Processing P6_TUMOR dataset..."
for SECTION_ID in 1 ; do
    echo "  Processing P6_TUMOR section ${SECTION_ID}..."
    python $VALIDATION_SCRIPT \
        --dataset P6_TUMOR \
        --section-id $SECTION_ID \
        --threads 28 \
        --output-dir $OUTPUT_DIR \
        --af-threshold 0.01
    
    # Check if script execution was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Validation failed for P6_TUMOR section ${SECTION_ID}"
    else
        echo "  Completed P6_TUMOR section ${SECTION_ID}"
    fi
done

# Generate a summary of all results
echo "Generating summary of all results..."
SUMMARY_FILE="${OUTPUT_DIR}/beagle_1kg_validation_summary.txt"
echo "Beagle Validation with 1000 Genomes Project - Summary" > $SUMMARY_FILE
echo "=====================================================" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE
echo "Date: $(date)" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

echo "Dataset,Section,Total Variants,Common in 1KG,Common Ratio (%)" >> $SUMMARY_FILE

# Extract results from individual summary files and compile them
for RESULT_FILE in ${OUTPUT_DIR}/*_summary.txt; do
    DATASET=$(basename $RESULT_FILE | cut -d '_' -f 1)
    SECTION=$(basename $RESULT_FILE | cut -d '_' -f 2)
    
    # Extract key metrics from summary file
    TOTAL=$(grep "Total Beagle variants" $RESULT_FILE | cut -d ' ' -f 4 | tr -d ',')
    COMMON=$(grep "Common variants in 1000G" $RESULT_FILE | cut -d ' ' -f 5 | tr -d ',')
    RATIO=$(grep "Common variant ratio" $RESULT_FILE | cut -d ' ' -f 4)
    
    # Append to summary
    echo "${DATASET},${SECTION},${TOTAL},${COMMON},${RATIO}" >> $SUMMARY_FILE
done

echo "Summary file created: $SUMMARY_FILE"

echo "All validation runs completed"
echo "End time: $(date)"
