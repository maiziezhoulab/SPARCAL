#!/bin/bash
# =============================================================================
# test_tmp_access.sh
# One-off test: verify /tmp filesystem type and whether Python 2 / Python 3
# can write to it and resolve paths via os.path.realpath() — mirroring what
# boost::filesystem::canonical() does internally in the strelka2 C++ binary.
# =============================================================================

#SBATCH --job-name=test_tmp_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --time=00:10:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_tmp_access-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_tmp_access-%j.err

echo "========================================"
echo "Host      : $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "========================================"

# ── 1. /tmp filesystem info ───────────────────────────────────────────────────

echo ""
echo "=== /tmp filesystem ==="
df -hT /tmp
echo ""
echo "=== /tmp mount entry ==="
mount | grep " /tmp "
echo ""
echo "=== /tmp permissions ==="
ls -la /tmp | head -5

# ── 2. Write a test file to /tmp ──────────────────────────────────────────────

TESTFILE=/tmp/strelka2_pathtest_$$
echo "hello from $$" > "$TESTFILE"
if [ -f "$TESTFILE" ]; then
    echo ""
    echo "=== Bash: created $TESTFILE OK ==="
else
    echo ""
    echo "=== Bash: FAILED to create $TESTFILE ==="
fi

# ── 3. Python 3 (snv_caller env) ─────────────────────────────────────────────

echo ""
echo "=== Python 3 (snv_caller) ==="
source activate snv_caller
python3 - <<'PYEOF'
import os, sys, tempfile

print("Python version:", sys.version)
print("Python executable:", sys.executable)

testfile = "/tmp/strelka2_py3_pathtest"
with open(testfile, "w") as f:
    f.write("py3 test\n")

print("Write to /tmp: OK")
print("os.path.exists:", os.path.exists(testfile))
print("os.path.realpath:", os.path.realpath(testfile))
print("os.path.abspath:", os.path.abspath(testfile))

# Mimic what boost::filesystem::canonical() does: resolve to absolute real path
import ctypes, ctypes.util
libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.realpath.restype = ctypes.c_char_p
result = libc.realpath(testfile.encode(), None)
print("libc realpath():", result)

os.unlink(testfile)
print("Cleanup: OK")
PYEOF

conda deactivate

# ── 4. Python 2 (strelka_py2 env) ────────────────────────────────────────────

echo ""
echo "=== Python 2 (strelka_py2) ==="
source activate strelka_py2
python2 - <<'PYEOF'
import os, sys, ctypes, ctypes.util

print("Python version:", sys.version)
print("Python executable:", sys.executable)

testfile = "/tmp/strelka2_py2_pathtest"
with open(testfile, "w") as f:
    f.write("py2 test\n")

print("Write to /tmp: OK")
print("os.path.exists:", os.path.exists(testfile))
print("os.path.realpath:", os.path.realpath(testfile))
print("os.path.abspath:", os.path.abspath(testfile))

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.realpath.restype = ctypes.c_char_p
result = libc.realpath(testfile, None)
print("libc realpath():", result)

os.unlink(testfile)
print("Cleanup: OK")
PYEOF

conda deactivate

# ── 5. Test strelka2 C++ binary directly on a /tmp FASTA ─────────────────────
# Use the strelka2 noise extractor (simpler binary) to probe reference resolution.
# We create a tiny dummy FASTA in /tmp and see if strelka2 can open it.

echo ""
echo "=== Strelka2 C++ binary: /tmp path resolution ==="
DUMMY_FA=/tmp/strelka2_dummy_$$.fa
printf ">chr1\nACGT\n" > "$DUMMY_FA"
samtools=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/apps/samtools
$samtools faidx "$DUMMY_FA" 2>&1 && echo "samtools faidx on /tmp FASTA: OK" || echo "samtools faidx on /tmp FASTA: FAILED"

STRELKA2_BIN=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts
if [ -x "$STRELKA2_BIN" ]; then
    echo "Testing GetSequenceErrorCounts --help (no reference needed):"
    $STRELKA2_BIN --help 2>&1 | head -3
else
    echo "GetSequenceErrorCounts binary not found at $STRELKA2_BIN"
fi

rm -f "$DUMMY_FA" "$DUMMY_FA.fai" "$TESTFILE"

# ── 6. Check /tmp size ────────────────────────────────────────────────────────

echo ""
echo "=== /tmp available space ==="
df -h /tmp

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
