#!/bin/bash
# =============================================================================
# test_exact_cmd_strace.sh
#
# Definitive test: run the EXACT pyflow GetSequenceErrorCounts command
# directly in bash (bypassing pyflow), then via bash -c (simulating pyflow
# isShellCmd wrapping), and with strace to identify the failing syscall.
#
# The actual pyflow command (from job 11391299 log) is:
#   /data/maiziezhou_lab/.../GetSequenceErrorCounts
#     --ref /panfs/accrepfs.vampire/.../genome.fa
#     --align-file /panfs/accrepfs.vampire/.../151507_merged.bam
#     --chrom-depth-file /panfs/accrepfs.vampire/.../chromDepth.tsv
#     ...all canonical paths...
# =============================================================================

#SBATCH --job-name=test_exact_strace
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=00:20:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_exact_strace-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_exact_strace-%j.err

echo "========================================"
echo "Host      : $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "========================================"

source activate strelka_py2

echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "CWD: $(pwd)"
echo "Physical CWD from /proc: $(readlink -f /proc/$$/cwd)"

# ── Paths (exactly as pyflow used them) ───────────────────────────────────────

BIN_SYM="/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"
BIN_REAL="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"

REF_PANFS="/panfs/accrepfs.vampire/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BAM_PANFS="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/151507_merged.bam"
DEPTH_PANFS="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/chromDepth.tsv"
ERRDIR="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/errorEstimation.tmpdir"

echo ""
echo "=== File existence checks ==="
for f in "$REF_PANFS" "$BAM_PANFS" "$DEPTH_PANFS" "$ERRDIR"; do
    if [ -e "$f" ]; then echo "OK: $f"; else echo "MISSING: $f"; fi
done

# run_test: captures last 5 lines of output (error message is at the end)
run_test() {
    local label="$1"; shift
    echo ""
    echo "=== TEST: $label ==="
    out=$("$@" 2>&1)
    rc=$?
    echo "$out" | tail -5
    echo "exit: $rc"
}

# ── SECTION 1: Run EXACT pyflow command directly in bash ──────────────────────
# This is the command that pyflow ran (and failed). If it passes here,
# the issue is in pyflow's execution context, not the command itself.

echo ""
echo "############################################################"
echo "# SECTION 1: Exact pyflow command, direct bash execution   #"
echo "############################################################"

run_test "Exact pyflow cmd (symlink binary, all canonical panfs paths)" \
    "$BIN_SYM" \
    --region 13:63097569-65069367 \
    --ref "$REF_PANFS" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErrCounts.test1.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmptySiteCounts.test1.tsv" \
    --align-file "$BAM_PANFS" \
    --chrom-depth-file "$DEPTH_PANFS"

# ── SECTION 2: Same command via bash -c (simulating pyflow isShellCmd) ────────

echo ""
echo "############################################################"
echo "# SECTION 2: Same command via bash --noprofile -o pipefail #"
echo "# (exactly how pyflowTaskWrapper.py runs isShellCmd tasks) #"
echo "############################################################"

FULL_CMD="$BIN_SYM --region 13:63097569-65069367 --ref $REF_PANFS --max-indel-size 49 --counts-file $ERRDIR/seqErrCounts.test2.bin --nonempty-site-count-file $ERRDIR/nonEmptySiteCounts.test2.tsv --align-file $BAM_PANFS --chrom-depth-file $DEPTH_PANFS"

echo ""
echo "=== TEST: via bash --noprofile -o pipefail -c CMD ==="
out=$(bash --noprofile -o pipefail -c "$FULL_CMD" 2>&1)
rc=$?
echo "$out" | tail -5
echo "exit: $rc"

# ── SECTION 3: Canonical binary path ─────────────────────────────────────────

echo ""
echo "############################################################"
echo "# SECTION 3: Canonical binary path                         #"
echo "############################################################"

run_test "Canonical binary path, all canonical paths" \
    "$BIN_REAL" \
    --region 13:63097569-65069367 \
    --ref "$REF_PANFS" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErrCounts.test3.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmptySiteCounts.test3.tsv" \
    --align-file "$BAM_PANFS" \
    --chrom-depth-file "$DEPTH_PANFS"

# ── SECTION 4: strace to find which realpath() call fails ─────────────────────

echo ""
echo "############################################################"
echo "# SECTION 4: strace — identify failing syscall             #"
# strace filters for realpath, lstat, readlink, stat calls and errors
echo "############################################################"

STRACE_OUT="/tmp/strace_getseqerr_$SLURM_JOBID.txt"
echo "strace output → $STRACE_OUT"
echo ""

strace -e trace=lstat,stat,statx,readlink,openat,open -o "$STRACE_OUT" \
    "$BIN_SYM" \
    --region 13:63097569-65069367 \
    --ref "$REF_PANFS" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErrCounts.test4.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmptySiteCounts.test4.tsv" \
    --align-file "$BAM_PANFS" \
    --chrom-depth-file "$DEPTH_PANFS" \
    2>&1 | tail -5

echo "exit: $?"

echo ""
echo "=== strace: lines with ENOENT or ENOTDIR or failed calls ==="
grep -E "ENOENT|ENOTDIR|EACCES|ELOOP|ENOLINK|= -1" "$STRACE_OUT" | head -30

echo ""
echo "=== strace: last 30 lines (where failure occurs) ==="
tail -30 "$STRACE_OUT"

# Copy strace output to persistent location
cp "$STRACE_OUT" "/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/strace_${SLURM_JOBID}.txt"

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
