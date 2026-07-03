#!/bin/bash

#SBATCH --job-name=mpileup_pipeline_DLPFC
#SBATCH --output=slurm_output/mpileup_single_DLPFC_snv_filtered.out
#SBATCH --error=slurm_output/mpileup_single_DLPFC_snv_filtered.err
#SBATCH --time=60:00:00
#SBATCH --account=cgw_maizie
#SBATCH --partition=cgw-maizie
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Run the pipeline for DLPFC section 151507-151510, base quality 0, mapping quality 0
python scripts/calling/mpileup_pipeline.py --dataset DLPFC_SVM_FILTERED --section_id 151507 --base_quality 13 --mapping_quality 20 --call_mode single --threads 30

echo "End time: $(date)"