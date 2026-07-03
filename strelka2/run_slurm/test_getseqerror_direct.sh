#!/bin/bash
# =============================================================================
# test_getseqerror_direct.sh
# Isolate why GetSequenceErrorCounts fails inside pyflow but not in direct bash.
#
# Approach:
#   A) Check /tmp reference file status; copy if missing.
#   B) Show LAST 3 lines of output (where the actual error lives), not first 5.
#   C) Run key tests WITH and WITHOUT LD_LIBRARY_PATH to separate conda effect
#      from /tmp-file-missing effect.
# =============================================================================

#SBATCH --job-name=test_getseqerr
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_getseqerror-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_getseqerror-%j.err

echo "========================================"
echo "Host      : $(hostname)"
echo "SLURM_JOBID: $SLURM_JOBID"
echo "Start time: $(date)"
echo "========================================"

# Activate the same conda env the real strelka2 job uses
source activate strelka_py2

echo "--- Environment after source activate strelka_py2 ---"
echo "CONDA_DEFAULT_ENV: $CONDA_DEFAULT_ENV"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PATH (first 3 entries): $(echo $PATH | tr ':' '\n' | head -3 | tr '\n' ':')"
echo "------------------------------------------------------"

# ── A) Reference file status ──────────────────────────────────────────────────

REF_SRC="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
REF_TMP="/tmp/strelka2_ref/genome.fa"

echo ""
echo "=== /tmp reference status ==="
if [ -f "$REF_TMP" ]; then
    echo "EXISTS: $(ls -lh $REF_TMP)"
    echo "Size bytes: $(stat -c%s $REF_TMP)"
    SRC_SIZE=$(stat -c%s "$REF_SRC" 2>/dev/null || echo "unknown")
    echo "Source size: $SRC_SIZE"
else
    echo "MISSING: $REF_TMP — copying now"
    mkdir -p /tmp/strelka2_ref
    cp "$REF_SRC" "$REF_TMP"
    cp "${REF_SRC}.fai" "${REF_TMP}.fai"
    echo "Copy done. Size: $(stat -c%s $REF_TMP)"
fi

# ── Paths ─────────────────────────────────────────────────────────────────────

BIN_SYM="/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"
BAM_SYM="/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/151507_merged.bam"
DEPTH_SYM="/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/chromDepth.tsv"
ERRDIR_SYM="/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/errorEstimation.tmpdir"

mkdir -p "$ERRDIR_SYM"

# run_test: show the LAST 3 lines of combined stdout+stderr — that's where the
# actual COMMAND-LINE ERROR lives, after the help text preamble.
run_test() {
    local label="$1"; shift
    echo ""
    echo "=== TEST: $label ==="
    echo "CMD: $*"
    local out
    out=$("$@" 2>&1)
    local rc=$?
    echo "$out" | tail -3
    echo "exit: $rc"
}

# ── SECTION 1: With current LD_LIBRARY_PATH (conda env active) ───────────────

echo ""
echo "###############################################"
echo "# SECTION 1: conda env active (as-is)         #"
echo "# LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "###############################################"

run_test "Minimal: --ref /tmp only" \
    "$BIN_SYM" --ref "$REF_TMP" --max-indel-size 49

run_test "Full pyflow args (symlink paths)" \
    "$BIN_SYM" \
    --region 11:53637337-55623904 \
    --ref "$REF_TMP" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR_SYM/sequenceErrorCounts.Sample000.test.bin" \
    --nonempty-site-count-file "$ERRDIR_SYM/nonEmptySiteCounts.Sample000.test.tsv" \
    --align-file "$BAM_SYM" \
    --chrom-depth-file "$DEPTH_SYM"

# ── SECTION 2: With LD_LIBRARY_PATH unset ────────────────────────────────────

echo ""
echo "###############################################"
echo "# SECTION 2: LD_LIBRARY_PATH unset            #"
echo "###############################################"

OLD_LLP="$LD_LIBRARY_PATH"
unset LD_LIBRARY_PATH

run_test "Minimal: --ref /tmp only (no LD_LIBRARY_PATH)" \
    "$BIN_SYM" --ref "$REF_TMP" --max-indel-size 49

run_test "Full pyflow args, no LD_LIBRARY_PATH" \
    "$BIN_SYM" \
    --region 11:53637337-55623904 \
    --ref "$REF_TMP" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR_SYM/sequenceErrorCounts.Sample000.test.bin" \
    --nonempty-site-count-file "$ERRDIR_SYM/nonEmptySiteCounts.Sample000.test.tsv" \
    --align-file "$BAM_SYM" \
    --chrom-depth-file "$DEPTH_SYM"

export LD_LIBRARY_PATH="$OLD_LLP"

# ── SECTION 3: Strip only apps/ from LD_LIBRARY_PATH ─────────────────────────

APPS_DIR="/data/maiziezhou_lab/leiy4/snv_calling/apps"
STRIPPED=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "^${APPS_DIR}$" | tr '\n' ':' | sed 's/:$//')

echo ""
echo "###############################################"
echo "# SECTION 3: apps/ stripped from LD_LIBRARY_PATH"
echo "# Stripped LD_LIBRARY_PATH=$STRIPPED"
echo "###############################################"

LD_LIBRARY_PATH="$STRIPPED" run_test "Minimal: --ref /tmp, apps/ stripped" \
    "$BIN_SYM" --ref "$REF_TMP" --max-indel-size 49

LD_LIBRARY_PATH="$STRIPPED" run_test "Full pyflow args, apps/ stripped" \
    "$BIN_SYM" \
    --region 11:53637337-55623904 \
    --ref "$REF_TMP" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR_SYM/sequenceErrorCounts.Sample000.test.bin" \
    --nonempty-site-count-file "$ERRDIR_SYM/nonEmptySiteCounts.Sample000.test.tsv" \
    --align-file "$BAM_SYM" \
    --chrom-depth-file "$DEPTH_SYM"

echo ""
echo "========================================"
echo "End time: $(date)"
echo "========================================"
