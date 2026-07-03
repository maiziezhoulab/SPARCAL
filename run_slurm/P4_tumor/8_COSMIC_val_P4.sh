#!/bin/bash
#SBATCH --job-name=cosmic_P4
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_P4_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_P4_%j.err

mkdir -p /data/maiziezhou_lab/leiy4/COSMIC/logs

source activate snv_caller

python /data/maiziezhou_lab/leiy4/snv_calling/scripts/postprocess/COSMIC_validation.py \
    --dataset P4_tumor \
    --sections 1 2 \
    --genome GRCh37 \
    --outdir /data/maiziezhou_lab/leiy4/COSMIC/validation/P4_tumor
