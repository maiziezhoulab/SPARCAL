#!/bin/bash

#SBATCH --job-name=beagle_pipeline_10X_BC_6.5mm
#SBATCH --output=slurm_output/DLPFC/baseQ13mapQ20/beagle_pipeline_DLPFC.out
#SBATCH --error=slurm_output/DLPFC/baseQ13mapQ20/beagle_pipeline_DLPFC.err
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
BASEQ=13
MAPQ=20

echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"

# Load required modules
module load Anaconda3
source activate snv_caller_new

# Run the pipeline for sections 151507-151510 and 151669-151676
for SECTION_ID in {151507..151510}; do
    python scripts/2_beagle_filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ${BASEQ}mapQ${MAPQ} --section_id ${SECTION_ID}
done

# for SECTION_ID in {151669..151672}; do
#     python scripts/2_beagle_filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ${BASEQ}mapQ${MAPQ} --section_id ${SECTION_ID}
# done

# for SECTION_ID in {151673..151676}; do
#     python scripts/2_beagle_filtering/run_beagle.py --dataset DLPFC --quality-filter baseQ${BASEQ}mapQ${MAPQ} --section_id ${SECTION_ID}
# done
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