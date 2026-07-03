#!/bin/bash
# ── P4 tumor ──────────────────────────────────────────────────────────────────
cat > /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/P4_tumor/8_COSMIC_val_P4.sh << 'EOF'
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
EOF

# ── P6 tumor ──────────────────────────────────────────────────────────────────
cat > /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/P6_tumor/8_COSMIC_val_P6.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=cosmic_P6
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_P6_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_P6_%j.err

mkdir -p /data/maiziezhou_lab/leiy4/COSMIC/logs

source activate snv_caller

python /data/maiziezhou_lab/leiy4/snv_calling/scripts/postprocess/COSMIC_validation.py \
    --dataset P6_tumor \
    --sections 1 2 \
    --genome GRCh37 \
    --outdir /data/maiziezhou_lab/leiy4/COSMIC/validation/P6_tumor
EOF

# ── DCIS ──────────────────────────────────────────────────────────────────────
cat > /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/rDCIS/8_COSMIC_val_DCIS.sh << 'EOF'
#!/bin/bash
#SBATCH --job-name=cosmic_DCIS
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_DCIS_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/COSMIC/logs/cosmic_DCIS_%j.err

mkdir -p /data/maiziezhou_lab/leiy4/COSMIC/logs

source activate snv_caller

python /data/maiziezhou_lab/leiy4/snv_calling/scripts/postprocess/COSMIC_validation.py \
    --dataset DCIS \
    --sections dcis1 dcis2 \
    --genome GRCh38 \
    --outdir /data/maiziezhou_lab/leiy4/COSMIC/validation/DCIS
EOF

echo "All sbatch scripts written. Submit with:"
echo "  sbatch /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/run_cosmic_P4.sh"
echo "  sbatch /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/run_cosmic_P6.sh"
echo "  sbatch /data/maiziezhou_lab/leiy4/snv_calling/run_slurm/run_cosmic_DCIS.sh"