#!/bin/bash

#SBATCH --job-name=mpileup_pipeline_10X_BC_FFPE
#SBATCH --output=slurm_output/10X_BC_FFPE/baseQ5mapQ5/mpileup_pipeline_10X_BC_FFPE.out
#SBATCH --error=slurm_output/10X_BC_FFPE/baseQ5mapQ5/mpileup_pipeline_10X_BC_FFPE.err
#SBATCH --time=24:00:00
#SBATCH --account=cgw_maizie3
#SBATCH --partition=cgw-maizie3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=5
MAPQ=5

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Run the pipeline
python scripts/calling/mpileup_pipeline.py --dataset 10X_BC_FFPE --base_quality $BASEQ --mapping_quality $MAPQ

echo "End time: $(date)"

# Other available experiments (for reference):
# For DLPFC sections:
#   python scripts/calling/mpileup_pipeline.py --dataset DLPFC --section_id 151507
#
# For P4 Tumor sections:
#   python scripts/calling/mpileup_pipeline.py --dataset P4_TUMOR --section_id 1
#   python scripts/calling/mpileup_pipeline.py --dataset P4_TUMOR --section_id 2
#
# For P6 Tumor sections:
#   python scripts/calling/mpileup_pipeline.py --dataset P6_TUMOR --section_id 1
#   python scripts/calling/mpileup_pipeline.py --dataset P6_TUMOR --section_id 2