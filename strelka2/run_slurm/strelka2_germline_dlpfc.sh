#!/bin/bash
# =============================================================================
# strelka2_germline_dlpfc.sh
# SLURM array job — Strelka2 germline SNV calling for all DLPFC sections.
#
# Sections: 151507 151508 151509 151510 151669 151670 151671 151672
#           151673 151674 151675 151676  (12 total, array index 0-11)
#
# Submit:
#   sbatch --array=0-11 strelka2_germline_dlpfc.sh
# Or test a single section (e.g. 151507, index 0):
#   sbatch --array=0 strelka2_germline_dlpfc.sh
# =============================================================================

#SBATCH --job-name=strelka2_germline_dlpfc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yuqi.lei@vanderbilt.edu
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=slurm_output/strelka2_germline_dlpfc_%a-%j.out
#SBATCH --error=slurm_output/strelka2_germline_dlpfc_%a-%j.err

# ── Section ID lookup ─────────────────────────────────────────────────────────

SECTIONS=(151507 151508 151509 151510
          151669 151670 151671 151672
          151673 151674 151675 151676)

SECTION_ID=${SECTIONS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$SECTION_ID" ]; then
    echo "ERROR: No section ID found for array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "========================================"
echo "SLURM_JOBID       : $SLURM_JOBID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Section ID        : $SECTION_ID"
echo "Start time        : $(date)"
echo "========================================"

# ── Environment ───────────────────────────────────────────────────────────────

# module load Anaconda3
# Use the bioconda strelka 2.9.10 rebuild (the prebuilt CentOS6 binary is broken
# on RHEL9 compute nodes — see DEBUGGING.md Bug 5; env built by install_strelka_conda.sh).
source activate strelka

mkdir -p slurm_output

# ── Run ───────────────────────────────────────────────────────────────────────

cd /data/maiziezhou_lab/leiy4/snv_calling/strelka2

# Point the runner at the conda strelka configure script. If left unset, the
# runner falls back to `configureStrelkaGermlineWorkflow.py` on PATH (provided by
# the activated `strelka` env), so this export is belt-and-suspenders.
export STRELKA_CONFIG=$(command -v configureStrelkaGermlineWorkflow.py)

CALL_REGIONS=/data/maiziezhou_lab/leiy4/snv_calling/strelka2/main_chroms_grch38.bed.gz

python scripts/run_strelka2_germline.py \
    --section_id ${SECTION_ID} \
    --threads 8 \
    --call_regions ${CALL_REGIONS}

EXIT_CODE=$?

echo "========================================"
echo "End time : $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE