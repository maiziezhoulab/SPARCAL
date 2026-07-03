#!/bin/bash
# P4 Tumor Dataset Script (mpileup_pipeline_P4_TUMOR.sh)
#SBATCH --job-name=mpileup_pipeline_P4_TUMOR
#SBATCH --output=slurm_output/P4_tumor/b0m0/mpileup_pipeline_P4_TUMOR.out1
#SBATCH --error=slurm_output/P4_tumor/b0m0/mpileup_pipeline_P4_TUMOR.err1
##SBATCH --output=slurm_output/P4_tumor/b13m20/mpileup_pipeline_P4_TUMOR.out2
##SBATCH --error=slurm_output/P4_tumor/b13m20/mpileup_pipeline_P4_TUMOR.err2
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

BASEQ=13
MAPQ=20
BASEQ=0
MAPQ=0
section_id=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

echo "Processing P4 Tumor section ${section_id}"
python scripts/1_calling/mpileup_pipeline.py --dataset P4_TUMOR --section_id ${section_id} --base_quality $BASEQ --mapping_quality $MAPQ --filter_out_tissue

echo "End time: $(date)"

