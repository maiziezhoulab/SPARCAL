#!/bin/bash
# =============================================================================
# test_realpath_and_tmp_ref.sh
#
# Two investigations:
#   A) Does the C realpath() syscall work for panfs canonical paths from a
#      compute node? (isolates whether the kernel itself is the problem)
#   B) Full GetSequenceErrorCounts command with /tmp reference + canonical
#      panfs BAM — the one combination not yet tested.
# =============================================================================

#SBATCH --job-name=test_realpath_tmpref
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:30:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_realpath_tmpref-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_realpath_tmpref-%j.err

echo "Host: $(hostname)  Job: $SLURM_JOBID  Start: $(date)"
source activate strelka_py2

BIN="/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"

REF_PANFS="/panfs/accrepfs.vampire/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
REF_PANFS_SYM="/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
REF_TMP="/tmp/strelka2_ref_$SLURM_JOBID/genome.fa"

BAM_CANONICAL="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/151507_merged.bam"
DEPTH="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/chromDepth.tsv"
ERRDIR="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/errorEstimation.tmpdir"

# ── A) Test C realpath() directly ────────────────────────────────────────────

echo ""
echo "############################################################"
echo "# A) C realpath() test (system glibc, not boost)           #"
echo "############################################################"
/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/test_realpath

# ── B) Copy genome.fa to /tmp (job-specific dir) ─────────────────────────────

echo ""
echo "############################################################"
echo "# B) Copy genome.fa to local /tmp                          #"
echo "############################################################"

REF_DIR=$(dirname "$REF_TMP")
mkdir -p "$REF_DIR"

SRC_SIZE=$(stat -c%s "$REF_PANFS")
echo "genome.fa source size: $SRC_SIZE"
echo "Copying genome.fa at $(date)..."
cp "$REF_PANFS"     "$REF_TMP"
cp "${REF_PANFS}.fai" "${REF_TMP}.fai"
cp "$(dirname $REF_PANFS)/genome.dict" "${REF_DIR}/genome.dict"
echo "Copy done at $(date)"

DST_SIZE=$(stat -c%s "$REF_TMP" 2>/dev/null || echo 0)
echo "genome.fa dest size: $DST_SIZE"
[ "$SRC_SIZE" -eq "$DST_SIZE" ] && echo "SIZE MATCH: OK" || echo "SIZE MISMATCH: source=$SRC_SIZE dst=$DST_SIZE"

run_test() {
    local label="$1"; shift
    echo ""
    echo "--- $label ---"
    out=$("$@" 2>&1); rc=$?
    echo "$out" | tail -4
    echo "exit: $rc"
    rm -f "$ERRDIR/seqErr.test.bin" "$ERRDIR/nonEmpty.test.tsv"
}

# ── C) Full command with /tmp ref + canonical BAM ────────────────────────────

echo ""
echo "############################################################"
echo "# C) Full command: /tmp ref + canonical panfs BAM          #"
echo "############################################################"

run_test "C1: /tmp ref + canonical BAM" \
    "$BIN" \
    --region 13:63097569-65069367 \
    --ref "$REF_TMP" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErr.test.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmpty.test.tsv" \
    --align-file "$BAM_CANONICAL" \
    --chrom-depth-file "$DEPTH"

# ── D) Full command with canonical panfs ref + canonical panfs BAM ────────────
# (repeat of test 8 from arg_isolation — confirm reproducibility)

echo ""
echo "############################################################"
echo "# D) Canonical panfs ref + canonical panfs BAM (control)   #"
echo "############################################################"

run_test "D1: canonical panfs ref + canonical BAM" \
    "$BIN" \
    --region 13:63097569-65069367 \
    --ref "$REF_PANFS" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErr.test.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmpty.test.tsv" \
    --align-file "$BAM_CANONICAL" \
    --chrom-depth-file "$DEPTH"

# ── E) Full command with symlink panfs ref + canonical panfs BAM ──────────────

echo ""
echo "############################################################"
echo "# E) Symlink panfs ref (/data/...) + canonical panfs BAM   #"
echo "############################################################"

run_test "E1: symlink ref + canonical BAM" \
    "$BIN" \
    --region 13:63097569-65069367 \
    --ref "$REF_PANFS_SYM" \
    --max-indel-size 49 \
    --counts-file "$ERRDIR/seqErr.test.bin" \
    --nonempty-site-count-file "$ERRDIR/nonEmpty.test.tsv" \
    --align-file "$BAM_CANONICAL" \
    --chrom-depth-file "$DEPTH"

# Cleanup
rm -rf "$REF_DIR"

echo ""
echo "Host: $(hostname)  End: $(date)"
