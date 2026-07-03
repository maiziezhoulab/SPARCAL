#!/bin/bash
# =============================================================================
# test_strelka2_ref_on_compute.sh
# Run directly on a compute node to verify:
#   1. genome.fa can be copied to /tmp
#   2. GetSequenceErrorCounts accepts --ref /tmp/... on the compute node
#   3. File size after copy matches source
# =============================================================================

#SBATCH --job-name=test_strelka2_ref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_strelka2_ref-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_strelka2_ref-%j.err

GENOME_SRC=/panfs/accrepfs.vampire/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa
GENOME_TMP=/tmp/strelka2_ref/genome.fa
GETSEQ=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts

echo "========================================"
echo "Host      : $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "========================================"

# ── 1. Copy genome.fa to /tmp ─────────────────────────────────────────────────

echo ""
echo "=== Copying genome.fa to /tmp/strelka2_ref/ ==="
mkdir -p /tmp/strelka2_ref

SRC_SIZE=$(stat -c%s "$GENOME_SRC")
echo "Source size: $SRC_SIZE bytes"

echo "Starting copy at $(date)"
cp "$GENOME_SRC" "$GENOME_TMP"
cp "${GENOME_SRC}.fai" "${GENOME_TMP}.fai"
cp "$(dirname $GENOME_SRC)/genome.dict" "/tmp/strelka2_ref/genome.dict"
echo "Copy finished at $(date)"

DST_SIZE=$(stat -c%s "$GENOME_TMP")
echo "Destination size: $DST_SIZE bytes"

if [ "$SRC_SIZE" -eq "$DST_SIZE" ]; then
    echo "File size matches: OK"
else
    echo "FILE SIZE MISMATCH: source=$SRC_SIZE dst=$DST_SIZE"
fi

echo ""
echo "=== /tmp/strelka2_ref/ contents ==="
ls -lh /tmp/strelka2_ref/

# ── 2. Test GetSequenceErrorCounts --ref /tmp/strelka2_ref/genome.fa ──────────

echo ""
echo "=== Testing GetSequenceErrorCounts with --ref on /tmp path ==="
echo "Running: $GETSEQ --ref $GENOME_TMP --max-indel-size 49"
echo "(expects: 'Must specify at least one input alignment file' — NOT 'can't resolve reference path')"
echo ""
$GETSEQ --ref "$GENOME_TMP" --max-indel-size 49 2>&1 | grep -E "ERROR|resolve|reference|alignment|version"

# ── 3. Also test with the panfs canonical path (no /tmp copy) ─────────────────

echo ""
echo "=== Testing GetSequenceErrorCounts with --ref on panfs canonical path ==="
echo "Running: $GETSEQ --ref $GENOME_SRC --max-indel-size 49"
$GETSEQ --ref "$GENOME_SRC" --max-indel-size 49 2>&1 | grep -E "ERROR|resolve|reference|alignment|version"

# ── 4. Verify file is still present at end ────────────────────────────────────

echo ""
echo "=== File still present after test? ==="
ls -lh /tmp/strelka2_ref/genome.fa 2>/dev/null || echo "MISSING: /tmp/strelka2_ref/genome.fa"

rm -rf /tmp/strelka2_ref/

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
