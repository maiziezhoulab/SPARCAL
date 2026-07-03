#!/bin/bash

#SBATCH --job-name=beagle_pipeline_10X_BC_6.5mm
#SBATCH --output=slurm_output/P4_tumor/baseQ0mapQ0/beagle_pipeline_P4_tumor_mono.out
#SBATCH --error=slurm_output/P4_tumor/baseQ0mapQ0/beagle_pipeline_P4_tumor_mono.err
#SBATCH --time=24:00:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=100GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu

# Set base quality and mapping quality
BASEQ=0
MAPQ=0
# BASEQ=13
# MAPQ=20
SECTION_ID=2
# SECTION_ID=1

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
# module load Anaconda3
source activate snv_caller

# Run the pipeline
# python scripts/2_beagle_filtering/run_beagle.py --dataset P4_TUMOR --quality-filter baseQ${BASEQ}mapQ${MAPQ} --section_id ${SECTION_ID}

# python scripts/2_beagle_filtering/run_beagle.py --dataset P4_TUMOR --section_id 1 --caller GATK

# python scripts/2_beagle_filtering/run_beagle.py --dataset P4_TUMOR --section_id 1 --caller Mutect2

python scripts/2_beagle_filtering/run_beagle.py --dataset P4_TUMOR --quality-filter baseQ0mapQ0 --section_id 1

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