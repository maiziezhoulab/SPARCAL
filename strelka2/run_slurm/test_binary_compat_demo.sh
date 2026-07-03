#!/bin/bash
#SBATCH --job-name=strelka_compat
#SBATCH --account=maiziezhou_lab_phd_int
#SBATCH --partition=interactive
#SBATCH --qos=maiziezhou_lab_phd_int
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/compat_demo_%j.out
#SBATCH --error=/data/maiziezhou_lab/leiy4/snv_calling/strelka2/slurm_output/compat_demo_%j.err
#
# ----------------------------------------------------------------------------
# DECISIVE strelka2 binary-vs-compute-node compatibility test.
#
# The whole debugging history (DEBUGGING.md Bugs 1-5) only ever ran the failing
# binary (GetSequenceErrorCounts) against the REAL data, whose paths all live on
# panfs. Every "isolation" test still had at least one panfs path (the BAM, the
# ref, or the output dir). So we could never separate two hypotheses:
#
#     (A) the binary itself is broken on RHEL9 compute nodes
#         (boost::filesystem::canonical ABI/seccomp break), OR
#     (B) something specific to the long /panfs/accrepfs.vampire/... paths.
#
# This test removes panfs ENTIRELY. It copies the 5 KB bundled demo dataset to a
# short, local, symlink-free path on the compute node's /tmp and runs the exact
# failing binary with all required arguments. The demo workflow normally passes
# --disableSequenceErrorEstimation, which SKIPS this binary -- that is why the
# demo never exposed the bug. Here we invoke GetSequenceErrorCounts directly.
#
# READ THE RESULT:
#   * If the /tmp (local, no-symlink) run FAILS with "can't resolve reference
#     path" -> hypothesis (A) confirmed: the binary is incompatible with the
#     compute-node environment, independent of any path/filesystem. The only
#     fixes are: rebuild strelka against modern boost, run it in a container,
#     or run on the login node. No amount of path-munging will help.
#   * If the /tmp run SUCCEEDS (prints "Must specify..." is NOT possible here
#     since all args are given; success = it starts counting / exits 0 or with
#     a data-level message) but the panfs run FAILS -> hypothesis (B): the
#     problem is the panfs paths, and we keep chasing path resolution.
# ----------------------------------------------------------------------------

set -u

STRELKA=/data/maiziezhou_lab/leiy4/snv_calling/strelka-2.9.2.centos6_x86_64
BIN=$STRELKA/libexec/GetSequenceErrorCounts
DEMO=$STRELKA/share/demo/strelka/data
REGION="demo20:1-5000"

echo "############################################################"
echo "# NODE / ENVIRONMENT"
echo "############################################################"
echo "Host          : $(hostname)"
echo "Date          : $(date)"
echo "Kernel        : $(uname -r)"
echo "OS            : $(cat /etc/redhat-release 2>/dev/null || echo unknown)"
echo "glibc         : $(ldd --version | head -1)"
echo "Binary target : $(file $BIN | sed 's/.*for /for /')"
echo "Binary ldd    :"; ldd $BIN 2>&1 | sed 's/^/                /'
echo

# ----------------------------------------------------------------------------
# Stage demo data to a SHORT, LOCAL, SYMLINK-FREE path.
# ----------------------------------------------------------------------------
LOCAL=/tmp/strelka_compat.$$
mkdir -p "$LOCAL"
cp "$DEMO"/demo20.fa "$DEMO"/demo20.fa.fai \
   "$DEMO"/NA12891_demo20.bam "$DEMO"/NA12891_demo20.bam.bai "$LOCAL"/

echo "Local staging dir (realpath): $(realpath $LOCAL)"
echo "  -> any symlinks in this path?  $(realpath $LOCAL | grep -q '^/tmp/' && echo 'NO (pure local /tmp)' || echo 'CHECK')"
echo "Demo fa size local : $(stat -c %s $LOCAL/demo20.fa) bytes (expect 5092)"
echo

run_test () {
    local label="$1"; shift
    local ref="$1";   shift
    local bam="$1";   shift
    local workdir="$1"; shift
    echo "============================================================"
    echo "TEST: $label"
    echo "  --ref        $ref"
    echo "  --align-file $bam"
    echo "  CWD          $workdir"
    echo "------------------------------------------------------------"
    ( cd "$workdir" && \
      "$BIN" \
        --ref "$ref" \
        --align-file "$bam" \
        --region "$REGION" \
        --counts-file "$workdir/counts.bin" \
        --nonempty-site-count-file "$workdir/nonempty.txt" \
        2>&1 )
    local rc=$?
    echo "------------------------------------------------------------"
    echo "EXIT CODE: $rc"
    if [ $rc -eq 0 ]; then
        echo ">>> SUCCESS on this path form."
    else
        echo ">>> FAILED (rc=$rc) on this path form."
    fi
    echo
}

echo "############################################################"
echo "# TEST 1 — pure local /tmp (NO panfs, NO symlinks) [DECISIVE]"
echo "############################################################"
run_test "local /tmp demo, CWD=/tmp" \
    "$LOCAL/demo20.fa" "$LOCAL/NA12891_demo20.bam" "$LOCAL"

echo "############################################################"
echo "# TEST 2 — canonical /panfs demo path (control)"
echo "############################################################"
run_test "panfs-canonical demo, CWD=/tmp" \
    "$(realpath $DEMO/demo20.fa)" "$(realpath $DEMO/NA12891_demo20.bam)" "$LOCAL"

echo "############################################################"
echo "# TEST 3 — /data symlink demo path (control)"
echo "############################################################"
run_test "data-symlink demo, CWD=/tmp" \
    "$DEMO/demo20.fa" "$DEMO/NA12891_demo20.bam" "$LOCAL"

echo "############################################################"
echo "# Cleanup"
echo "############################################################"
rm -rf "$LOCAL"
echo "Done."
