#!/bin/bash
#SBATCH --job-name=test_cn1328
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --nodelist=cn1328
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_cn1328-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_cn1328-%j.err

source activate strelka_py2

BIN="/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"
REF_PANFS="/panfs/accrepfs.vampire/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
REF_SYM="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BAM_PAN="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/151507_merged.bam"

echo "Host: $(hostname)"
echo "Date: $(date)"

# Check reference file accessibility
echo ""
echo "=== Reference file check ==="
ls -lh "$REF_PANFS" 2>&1 | head -2
python -c "import os; print('realpath:', os.path.realpath('$REF_SYM'))"
python -c "import ctypes; libc=ctypes.CDLL(None); buf=ctypes.create_string_buffer(4096); r=libc.realpath(b'$REF_PANFS', buf); print('C realpath exit:', r, '| result:', buf.value if r else 'NULL/FAIL')"

echo ""
echo "=== Test 1: --ref /panfs canonical (no --align-file) ==="
"$BIN" --ref "$REF_PANFS" --max-indel-size 49 2>&1 | tail -3
echo "exit: ${PIPESTATUS[0]}"

echo ""
echo "=== Test 2: --ref symlink (no --align-file) ==="
"$BIN" --ref "$REF_SYM" --max-indel-size 49 2>&1 | tail -3
echo "exit: ${PIPESTATUS[0]}"

echo ""
echo "=== Test 3: Copy ref to /tmp, use /tmp ==="
mkdir -p /tmp/strelka2_ref_test
cp "$REF_PANFS" /tmp/strelka2_ref_test/genome.fa
cp "${REF_PANFS}.fai" /tmp/strelka2_ref_test/genome.fa.fai
echo "Copied. Size: $(stat -c%s /tmp/strelka2_ref_test/genome.fa)"
"$BIN" --ref /tmp/strelka2_ref_test/genome.fa --max-indel-size 49 2>&1 | tail -3
echo "exit: ${PIPESTATUS[0]}"

echo ""
echo "=== Test 4: /tmp ref + canonical panfs --align-file ==="
"$BIN" --ref /tmp/strelka2_ref_test/genome.fa --max-indel-size 49 \
    --align-file "$BAM_PAN" 2>&1 | tail -3
echo "exit: ${PIPESTATUS[0]}"

echo ""
echo "=== Test 5: /tmp ref + canonical panfs --align-file + --chrom-depth-file ==="
DEPTH="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/chromDepth.tsv"
if [ -f "$DEPTH" ]; then
    "$BIN" --ref /tmp/strelka2_ref_test/genome.fa --max-indel-size 49 \
        --region 11:53637337-55623904 \
        --align-file "$BAM_PAN" \
        --chrom-depth-file "$DEPTH" 2>&1 | tail -3
    echo "exit: ${PIPESTATUS[0]}"
else
    echo "chromDepth.tsv not found, skipping"
fi

echo ""
echo "Done: $(date)"
