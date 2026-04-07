#!/bin/bash
#SBATCH --job-name=sparcal_DCIS
#SBATCH --output=slurm_output/DCIS/sparcal_DCIS-%j.out
#SBATCH --error=slurm_output/DCIS/sparcal_DCIS-%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=200GB
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int

# =============================================================================
# SPARCAL — DCIS (Somatic / Breast Cancer)
# mpileup → beagle → bam_filter → somatic_spatial → matrix
# =============================================================================

set -euo pipefail

echo "======================================================"
echo "SPARCAL: DCIS (Somatic)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start: $(date)"
echo "======================================================"

module load Anaconda3
source activate snv_caller_new

DATASET="DCIS"
S="/data/maiziezhou_lab/leiy4/SPARCAL/scripts"

SECTIONS=(1 2)

run_step() {
    local name="$1"; shift
    echo ""; echo "--- [$name] $(date) ---"; echo "CMD: $@"
    "$@"
    [ $? -eq 0 ] && echo "[$name] DONE" || { echo "[$name] FAILED"; exit 1; }
}

for SID in "${SECTIONS[@]}"; do
    echo ""; echo "=== Section ${SID} ==="

    run_step "mpileup"    python ${S}/1_calling/mpileup_pipeline.py \
        --dataset ${DATASET} --section_id ${SID} --call_mode multi --threads 30

    run_step "beagle"     python ${S}/2_genotyping/run_beagle.py \
        --dataset ${DATASET} --section_id ${SID} --threads 24

    run_step "bam_filter" python ${S}/3_germline_filter/3_refilter_bam_by_snv_pool.py \
        --dataset ${DATASET} --section-id ${SID} --max-workers 30

    # TODO: uncomment when somatic spatial filter is ready
    # run_step "spatial"    python ${S}/4_somatic_filter/1_somatic_spatial_filter.py \
    #     --dataset ${DATASET} --section_id ${SID}

    run_step "matrix"     python ${S}/5_generate_matrix/generate_matrix.py \
        --dataset ${DATASET} --section_id ${SID} --filter-subdir filtered_snvs

    echo "Section ${SID} COMPLETE at $(date)"
done

echo ""; echo "All DCIS sections complete! $(date)"