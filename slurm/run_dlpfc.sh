#!/bin/bash
#SBATCH --job-name=sparcal_DLPFC
#SBATCH --output=slurm_output/DLPFC/sparcal_DLPFC-%j.out
#SBATCH --error=slurm_output/DLPFC/sparcal_DLPFC-%j.err
#SBATCH --time=72:00:00
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
# SPARCAL — DLPFC (Germline)
# mpileup → beagle → geno_shift → seq_error → sparcal_net → bam_filter
#   → germline_spatial → matrix
# =============================================================================

set -euo pipefail

echo "======================================================"
echo "SPARCAL: DLPFC (Germline)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start: $(date)"
echo "======================================================"

module load Anaconda3
source activate snv_caller_new

DATASET="DLPFC"
S="/data/maiziezhou_lab/leiy4/SPARCAL/scripts"

SECTIONS=(151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676)

run_step() {
    local name="$1"; shift
    echo ""; echo "--- [$name] $(date) ---"; echo "CMD: $@"
    "$@"
    [ $? -eq 0 ] && echo "[$name] DONE" || { echo "[$name] FAILED"; exit 1; }
}

for SID in "${SECTIONS[@]}"; do
    echo ""; echo "=== Section ${SID} ==="

    run_step "mpileup"     python ${S}/1_calling/mpileup_pipeline.py \
        --dataset ${DATASET} --section_id ${SID} --call_mode multi --threads 30

    run_step "beagle"      python ${S}/2_genotyping/run_beagle.py \
        --dataset ${DATASET} --section_id ${SID} --threads 24

    run_step "geno_shift"  python ${S}/2_genotyping/run_beagle_genotype_shifting.py \
        --dataset ${DATASET} --section_id ${SID}

    run_step "seq_error"   python ${S}/3_germline_filter/1_sequencing_error_model.py \
        --dataset ${DATASET} --section_id ${SID} --hom_baf_threshold 0.99

    run_step "sparcal_net" python ${S}/3_germline_filter/2_sparcal_net.py \
        --dataset ${DATASET} --section_id ${SID} --model-type neural_network

    run_step "bam_filter"  python ${S}/3_germline_filter/3_refilter_bam_by_snv_pool.py \
        --dataset ${DATASET} --section-id ${SID} --classifier neural_network --max-workers 30

    run_step "spatial"     python ${S}/3_germline_filter/4_germline_spatial_filter.py \
        --dataset ${DATASET} --section_id ${SID} --min_neighbours 2

    run_step "matrix"      python ${S}/5_generate_matrix/generate_matrix.py \
        --dataset ${DATASET} --section_id ${SID} --filter-subdir filtered_snvs

    echo "Section ${SID} COMPLETE at $(date)"
done

echo ""; echo "All DLPFC sections complete! $(date)"