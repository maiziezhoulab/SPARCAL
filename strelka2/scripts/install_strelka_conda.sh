#!/bin/bash
# ============================================================================
# install_strelka_conda.sh
# ----------------------------------------------------------------------------
# Rebuild strelka2 "from scratch" via bioconda, replacing the broken prebuilt
# CentOS6 binary (strelka-2.9.2.centos6_x86_64) whose statically-linked old
# boost::filesystem::canonical() fails inside GetSequenceErrorCounts on the
# RHEL9 compute nodes. See strelka2/DEBUGGING.md Bug 5.
#
# This installs strelka 2.9.10 build hdfd78af_2 (a modern conda-forge rebuild)
# into a fresh conda env, then RE-APPLIES the SMTP-timeout patch (DEBUGGING.md
# Bug 1) which a fresh install would otherwise lack — without it, runWorkflow.py
# hangs forever on compute nodes because smtplib.SMTP('localhost') has no timeout.
#
# Run on the LOGIN node (no SLURM, not affected by the QOS CPU limit):
#   bash strelka2/scripts/install_strelka_conda.sh
# ============================================================================

set -euo pipefail

ENV_NAME=strelka

echo "==> Creating conda env '$ENV_NAME' with bioconda strelka 2.9.10 ..."
conda create -n "$ENV_NAME" -c bioconda -c conda-forge \
    "strelka=2.9.10=hdfd78af_2" python=2.7 -y

# Resolve the env prefix without needing `conda activate` in a non-interactive shell.
ENV_PREFIX=$(conda env list | awk -v n="$ENV_NAME" '$1==n {print $NF}')
if [ -z "${ENV_PREFIX:-}" ] || [ ! -d "$ENV_PREFIX" ]; then
    echo "ERROR: could not locate env prefix for '$ENV_NAME'." >&2
    exit 1
fi
echo "==> Env prefix: $ENV_PREFIX"

echo
echo "==> Locating strelka components ..."
CONFIG_SCRIPT=$(find "$ENV_PREFIX" -name configureStrelkaGermlineWorkflow.py 2>/dev/null | head -1)
PYFLOW_PY=$(find "$ENV_PREFIX" -path "*pyflow/pyflow.py" 2>/dev/null | head -1)
MAKERUN_PY=$(find "$ENV_PREFIX" -name makeRunScript.py 2>/dev/null | head -1)

echo "    configure : ${CONFIG_SCRIPT:-NOT FOUND}"
echo "    pyflow.py : ${PYFLOW_PY:-NOT FOUND}"
echo "    makeRun.py: ${MAKERUN_PY:-NOT FOUND}"

if [ -z "${CONFIG_SCRIPT:-}" ]; then
    echo "ERROR: configureStrelkaGermlineWorkflow.py not found in env." >&2
    exit 1
fi

# ── Re-apply DEBUGGING.md Bug 1 fix: add a timeout to smtplib.SMTP('localhost') ──
# Idempotent: only rewrites the no-timeout form. On compute nodes port 25 accepts
# the TCP connect but never sends a banner, so a timeout-less SMTP() blocks forever.
patch_smtp () {
    local f="$1"
    [ -z "$f" ] && return 0
    [ -f "$f" ] || return 0
    if grep -q "smtplib.SMTP('localhost')" "$f" 2>/dev/null \
       || grep -q 'smtplib.SMTP("localhost")' "$f" 2>/dev/null; then
        sed -i \
          -e "s/smtplib\.SMTP('localhost')/smtplib.SMTP('localhost', timeout=5)/g" \
          -e 's/smtplib\.SMTP("localhost")/smtplib.SMTP("localhost", timeout=5)/g' \
          "$f"
        echo "    patched SMTP timeout -> $f"
    else
        echo "    no bare smtplib.SMTP('localhost') in $f (ok / already patched)"
    fi
}

echo
echo "==> Applying SMTP-timeout patch (DEBUGGING.md Bug 1) ..."
patch_smtp "$PYFLOW_PY"
patch_smtp "$MAKERUN_PY"

echo
echo "============================================================"
echo "DONE. New strelka configure script:"
echo "    $CONFIG_SCRIPT"
echo
echo "Use it by setting in the SLURM script:"
echo "    source activate $ENV_NAME"
echo "    export STRELKA_CONFIG=$CONFIG_SCRIPT"
echo "============================================================"
