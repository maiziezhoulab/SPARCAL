#!/bin/bash

#SBATCH --job-name=beagle_genotype_shifting_P4_tumor
#SBATCH --output=slurm_output/P4_tumor/baseQ0mapQ0_section1/seq_err_model_P4.out
#SBATCH --error=slurm_output/P4_tumor/baseQ0mapQ0_section1/seq_err_model_P4.err

#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=400GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# dedup ablation: P4 rep1 (section 1), baseQ0mapQ0 only
python scripts/3_classifier_prep/run_sequence_error_model.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ0mapQ0
# python scripts/3_classifier_prep/run_sequence_error_model.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ13mapQ20
# python scripts/3_classifier_prep/run_sequence_error_model.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ0mapQ0
# python scripts/3_classifier_prep/run_sequence_error_model.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ13mapQ20



echo "End time: $(date)"