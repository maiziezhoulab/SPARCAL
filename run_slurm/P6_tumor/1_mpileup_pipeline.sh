#!/bin/bash
# P6 Tumor Dataset Script (mpileup_pipeline_P6_TUMOR.sh)
#SBATCH --job-name=mpileup_pipeline_P6_TUMOR
#SBATCH --output=slurm_output/P6_tumor/b0m0/mpileup_pipeline_P6_TUMOR.out
#SBATCH --error=slurm_output/P6_tumor/b0m0/mpileup_pipeline_P6_TUMOR.err
#SBATCH --output=slurm_output/P6_tumor/b13m20/mpileup_pipeline_P6_TUMOR.out1
#SBATCH --error=slurm_output/P6_tumor/b13m20/mpileup_pipeline_P6_TUMOR.err1
#SBATCH --time=40:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# BASEQ=0
# MAPQ=0
BASEQ=0
MAPQ=0
section_id=1


echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

echo "Processing P6 Tumor section ${section_id}"
python scripts/1_calling/mpileup_pipeline.py --dataset P6_TUMOR --section_id ${section_id} --base_quality $BASEQ --mapping_quality $MAPQ --filter_out_tissue

echo "End time: $(date)"
