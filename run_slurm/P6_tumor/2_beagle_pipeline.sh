#!/bin/bash

#SBATCH --job-name=beagle_pipeline_P6_tumor
#SBATCH --output=slurm_output/P6_tumor/baseQ0mapQ0_section1_beagle_pipeline_P6_tumor.out
#SBATCH --error=slurm_output/P6_tumor/baseQ0mapQ0_section1_beagle_pipeline_P6_tumor.err

#SBATCH --output=slurm_output/P6_tumor/baseQ0mapQ0_section2_beagle_pipeline_P6_tumor.out
#SBATCH --error=slurm_output/P6_tumor/baseQ0mapQ0_section1_beagle_pipeline_P6_tumor.err

#SBATCH --output=slurm_output/P6_tumor/baseQ13mapQ20_section1_beagle_pipeline_P6_tumor.out
#SBATCH --error=slurm_output/P6_tumor/baseQ13mapQ20_section1_beagle_pipeline_P6_tumor.err

#SBATCH --output=slurm_output/P6_tumor/baseQ13mapQ20_section2_beagle_pipeline_P6_tumor.out
#SBATCH --error=slurm_output/P6_tumor/baseQ13mapQ20_section2_beagle_pipeline_P6_tumor.err

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

# Set base quality and mapping quality
BASEQ=0
MAPQ=0
# BASEQ=13
# MAPQ=20
SECTION_ID=2

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules

source activate snv_caller

# Run the pipeline for 4 exprs: sec1/2, baseQ0mapQ0/baseQ13mapQ20
python scripts/2_beagle_filtering/run_beagle.py --dataset P6_TUMOR --quality-filter baseQ0mapQ0 --section_id 1
# python scripts/2_beagle_filtering/run_beagle.py --dataset P6_TUMOR --quality-filter baseQ0mapQ0 --section_id 2
# python scripts/2_beagle_filtering/run_beagle.py --dataset P6_TUMOR --quality-filter baseQ13mapQ20 --section_id 1
# python scripts/2_beagle_filtering/run_beagle.py --dataset P6_TUMOR --quality-filter baseQ13mapQ20 --section_id 2

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