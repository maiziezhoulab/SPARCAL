#!/bin/bash

#SBATCH --job-name=generate_bam_DLPFC
#SBATCH --output=slurm_output/generate_bam_DLPFC.out
#SBATCH --error=slurm_output/generate_bam_DLPFC.err
#SBATCH --time=40:00:00
#SBATCH --account=cgw_maizie2
#SBATCH --partition=cgw-maizie2
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

which python
# Run the BAM generation script for DLPFC section
~/.conda/envs/snv_caller_new/bin/python scripts/postprocess/run_generate_bam.py --dataset DLPFC --section-id 151507 --quality-filter baseQ13mapQ20

echo "End time: $(date)"