#!/bin/bash
# =============================================================================
# test_arg_isolation.sh
#
# Systematically add one argument at a time to find which one triggers
# "can't resolve reference path" in GetSequenceErrorCounts.
#
# Known baseline: --ref /panfs/... --max-indel-size 49   =>  PASSES
# All canonical panfs paths, full args                   =>  FAILS
#
# Strategy: test every combination that adds one argument to the baseline.
# =============================================================================

#SBATCH --job-name=test_arg_isolation
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --time=00:15:00
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --output=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_arg_isolation-%j.out
#SBATCH --error=/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/test_arg_isolation-%j.err

echo "Host: $(hostname)  Job: $SLURM_JOBID  Start: $(date)"
source activate strelka_py2

BIN="/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64/libexec/GetSequenceErrorCounts"
REF="/panfs/accrepfs.vampire/data/maiziezhou_lab/Softwares/GRCh38-3.0.0/fasta/genome.fa"
BAM="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/151507_merged.bam"
DEPTH="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/chromDepth.tsv"
ERRDIR="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/snv_calling/data/dlpfc/151507/strelka2/workspace/errorEstimation.tmpdir"

# Output paths (do NOT exist before the binary runs)
COUNTS="$ERRDIR/seqErrCounts.isolation.bin"
NONEMPTY="$ERRDIR/nonEmptySiteCounts.isolation.tsv"
# Local output paths (local /tmp)
COUNTS_LOCAL="/tmp/seqErrCounts.isolation.$SLURM_JOBID.bin"
NONEMPTY_LOCAL="/tmp/nonEmptySiteCounts.isolation.$SLURM_JOBID.tsv"

# Remove any leftover output files
rm -f "$COUNTS" "$NONEMPTY" "$COUNTS_LOCAL" "$NONEMPTY_LOCAL"

run_test() {
    local label="$1"; shift
    echo ""
    echo "--- $label ---"
    out=$("$@" 2>&1); rc=$?
    echo "$out" | tail -3
    echo "exit: $rc"
    # Clean up output files between tests
    rm -f "$COUNTS" "$NONEMPTY" "$COUNTS_LOCAL" "$NONEMPTY_LOCAL"
}

# ── 0. BASELINE: known to pass (from test 11390056) ──────────────────────────
run_test "0: baseline --ref only" \
    "$BIN" --ref "$REF" --max-indel-size 49

# ── 1. Add only --region ──────────────────────────────────────────────────────
run_test "1: +--region" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --region 13:63097569-65069367

# ── 2. Add only --align-file (canonical panfs BAM) ───────────────────────────
run_test "2: +--align-file (canonical panfs)" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --align-file "$BAM"

# ── 3. Add only --chrom-depth-file ───────────────────────────────────────────
run_test "3: +--chrom-depth-file" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --chrom-depth-file "$DEPTH"

# ── 4. Add only --counts-file (panfs path, file does NOT exist) ──────────────
run_test "4: +--counts-file (panfs, nonexistent)" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --counts-file "$COUNTS"

# ── 5. Add only --nonempty-site-count-file (panfs, nonexistent) ──────────────
run_test "5: +--nonempty-site-count-file (panfs, nonexistent)" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --nonempty-site-count-file "$NONEMPTY"

# ── 6. Add only --counts-file with LOCAL path (does not exist) ───────────────
run_test "6: +--counts-file (local /tmp, nonexistent)" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --counts-file "$COUNTS_LOCAL"

# ── 7. Add --align-file + --chrom-depth-file (no output files) ───────────────
run_test "7: +--align-file +--chrom-depth-file (no output files)" \
    "$BIN" --ref "$REF" --max-indel-size 49 \
    --align-file "$BAM" \
    --chrom-depth-file "$DEPTH"

# ── 8. Full command but output files pre-created with touch ──────────────────
touch "$COUNTS" "$NONEMPTY"
run_test "8: full cmd, output files pre-created (touch)" \
    "$BIN" \
    --region 13:63097569-65069367 \
    --ref "$REF" \
    --max-indel-size 49 \
    --counts-file "$COUNTS" \
    --nonempty-site-count-file "$NONEMPTY" \
    --align-file "$BAM" \
    --chrom-depth-file "$DEPTH"

# ── 9. Full command, output files in local /tmp ───────────────────────────────
run_test "9: full cmd, output files in local /tmp" \
    "$BIN" \
    --region 13:63097569-65069367 \
    --ref "$REF" \
    --max-indel-size 49 \
    --counts-file "$COUNTS_LOCAL" \
    --nonempty-site-count-file "$NONEMPTY_LOCAL" \
    --align-file "$BAM" \
    --chrom-depth-file "$DEPTH"

echo ""
echo "Host: $(hostname)  End: $(date)"
