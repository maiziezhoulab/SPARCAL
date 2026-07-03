#!/bin/bash

#SBATCH --job-name=de_novo_filter_P4_tumor
#SBATCH --output=slurm_output/P4_tumor/baseQ0mapQ0_section1/de_novo_filter.out
#SBATCH --error=slurm_output/P4_tumor/baseQ0mapQ0_section1/de_novo_filter.err

#SBATCH --time=24:00:00
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

# Run the pipeline for 4 exprs: sec1/2, baseQ0mapQ0/baseQ13mapQ20
# ~/.conda/envs/snv_caller_new/bin/python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ0mapQ0
~/.conda/envs/snv_caller_new/bin/python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ13mapQ20
~/.conda/envs/snv_caller_new/bin/python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ0mapQ0
~/.conda/envs/snv_caller_new/bin/python scripts/3_classifier_prep/consecutive_denovo_finder.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ13mapQ20




echo "End time: $(date)"