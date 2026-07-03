#!/bin/bash

#SBATCH --job-name=beagle_genotype_shifting_P4_tumor
#SBATCH --output=slurm_output/P4_tumor/baseQ0mapQ0_section1/beagle_shifting_P4_tumor.out
#SBATCH --error=slurm_output/P4_tumor/baseQ0mapQ0_section1/beagle_shifting_P4_tumor.err

#SBATCH --output=slurm_output/P4_tumor/baseQ0mapQ0_section2/beagle_shifting_P4_tumor.out
#SBATCH --error=slurm_output/P4_tumor/baseQ0mapQ0_section2/beagle_shifting_P4_tumor.err

#SBATCH --output=slurm_output/P4_tumor/baseQ13mapQ20_section1/beagle_shifting_P4_tumor.out
#SBATCH --error=slurm_output/P4_tumor/baseQ13mapQ20_section1/beagle_shifting_P4_tumor.err

#SBATCH --output=slurm_output/P4_tumor/baseQ13mapQ20_section2/beagle_shifting_P4_tumor.out
#SBATCH --error=slurm_output/P4_tumor/baseQ13mapQ20_section2/beagle_shifting_P4_tumor.err

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
python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ0mapQ0
# python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ0mapQ0
# python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 1 --quality_filter baseQ13mapQ20
# python scripts/2_beagle_filtering/run_beagle_genotype_shifting.py --dataset P4_TUMOR --section_id 2 --quality_filter baseQ13mapQ20

echo "End time: $(date)"