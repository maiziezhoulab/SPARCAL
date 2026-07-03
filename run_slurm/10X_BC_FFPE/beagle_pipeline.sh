#!/bin/bash

#SBATCH --job-name=beagle_pipeline_10X_BC_6.5mm
#SBATCH --output=slurm_output/10X_BC_6.5mm/baseQ${BASEQ}mapQ${MAPQ}/beagle_pipeline_10X_BC_6.5mm.out
#SBATCH --error=slurm_output/10X_BC_6.5mm/baseQ${BASEQ}mapQ${MAPQ}/beagle_pipeline_10X_BC_6.5mm.err
#SBATCH --time=24:00:00
#SBATCH --account=cgw_maizie3
#SBATCH --partition=cgw-maizie3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=400GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# Set base quality and mapping quality
BASEQ=0
MAPQ=0

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Run the pipeline
python scripts/filtering/run_beagle.py --dataset 10X_BC_6.5MM --quality-filter baseQ${BASEQ}mapQ${MAPQ}

echo "End time: $(date)"

# Other available experiments (for reference):
# For DLPFC sections:
#   python scripts/filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ0mapQ0
#
# For P4 Tumor sections:
#   python scripts/filtering/run_beagle.py --dataset P4_TUMOR --quality-filter baseQ0mapQ0
#
# For P6 Tumor sections:
#   python scripts/filtering/run_beagle.py --dataset P6_TUMOR --quality-filter baseQ0mapQ0