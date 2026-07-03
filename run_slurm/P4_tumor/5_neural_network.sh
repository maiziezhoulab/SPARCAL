#!/bin/bash

#SBATCH --job-name=nn_classifier_P4_TUMOR
#SBATCH --output=slurm_output/P4_TUMOR/nn_classifier_P4_TUMOR_%a.out
#SBATCH --error=slurm_output/P4_TUMOR/nn_classifier_P4_TUMOR_%a.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=256GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --array=1   # dedup ablation: P4 rep1 (section 1) only

echo "SLURM_JOBID: $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Start time: $(date)"

# Load required modules
source activate snv_caller

# Set quality parameters
BASEQ=0
MAPQ=0
QUALITY_FILTER="baseQ${BASEQ}mapQ${MAPQ}"

# Get section ID from array task ID
SECTION_ID=${SLURM_ARRAY_TASK_ID}

echo "==============================================="
echo "Processing P4_TUMOR section: ${SECTION_ID}"
echo "Quality filter: ${QUALITY_FILTER}"
echo "Start time: $(date)"

# Run the neural network classifier
# NOTE: use run_supplimentary_models.py (as P6/DCIS do). run_sparcal_net.py crashes here with
# "ValueError: y contains previously unseen labels: 'no_variance'" (trains on het/hom only, then
# hard-looks-up 'no_variance' in apply_to_vcf) -> empty classifier output -> 0 denovo variants.
python /data/maiziezhou_lab/leiy4/snv_calling/scripts/4_classifier/run_supplimentary_models.py \
    --dataset P4_TUMOR \
    --section_id ${SECTION_ID} \
    --quality-filter ${QUALITY_FILTER} \
    --model-type neural_network \
    --max-training-samples 90000
# python /data/maiziezhou_lab/leiy4/snv_calling/scripts/4_classifier/run_sparcal_net.py --dataset P4_TUMOR --section_id ${SECTION_ID} --quality-filter ${QUALITY_FILTER}

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "Section ${SECTION_ID} completed successfully"
else
    echo "ERROR: Failed to process section ${SECTION_ID}"
fi

echo "End time for section ${SECTION_ID}: $(date)"
echo "==============================================="
