#!/bin/bash
# P6 Tumor Dataset Script (mpileup_pipeline_P6_TUMOR.sh)
#SBATCH --job-name=mpileup_pipeline_P6_TUMOR
#SBATCH --output=slurm_output/P6_tumor/b0m0/mpileup_pipeline_P6_TUMOR.out
#SBATCH --error=slurm_output/P6_tumor/b0m0/mpileup_pipeline_P6_TUMOR.err
##SBATCH --output=slurm_output/P6_tumor/b13m20/mpileup_pipeline_P6_TUMOR.out
##SBATCH --error=slurm_output/P6_tumor/b13m20/mpileup_pipeline_P6_TUMOR.err
#SBATCH --time=40:00:00
#SBATCH --account=cgw_maizie3
#SBATCH --partition=cgw-maizie3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=300GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

BASEQ=0
MAPQ=0
BASEQ=13
MAPQ=20
section_id=2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

echo "Processing P6 Tumor section ${section_id}"
python scripts/calling/mpileup_pipeline.py --dataset P6_TUMOR --section_id ${section_id} --base_quality $BASEQ --mapping_quality $MAPQ

echo "End time: $(date)"
