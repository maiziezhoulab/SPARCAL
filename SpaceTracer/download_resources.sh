#!/usr/bin/env bash
# =============================================================================
# download_resources.sh — Download SpaceTracer hg38 reference resources
#
# Source: Zenodo record 19896967  (DOI: 10.64898/2026.02.04.703493)
# File  : resources.tar  (~7 GB)
# Contains: hg38_resources.tar.zst, mm10_resources.tar.zst
#
# Run once from an interactive node or via the accompanying SLURM script.
# Requires: wget (or curl), zstd, tar
#
# Usage:
#   bash download_resources.sh
#   # or via SLURM:
#   sbatch slurm/download_resources.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOURCES_DIR="${SCRIPT_DIR}/resources"
HG38_DIR="${RESOURCES_DIR}/hg38"
DOWNLOAD_URL="https://zenodo.org/api/records/19896967/files/resources.tar?download=1"
TARBALL="${RESOURCES_DIR}/resources.tar"

echo "============================================================"
echo "SpaceTracer — Download hg38 Resources"
echo "Target dir : ${RESOURCES_DIR}"
echo "============================================================"

mkdir -p "${RESOURCES_DIR}"

# ---------------------------------------------------------------------------
# 1. Download the archive
# ---------------------------------------------------------------------------
if [ -f "${TARBALL}" ]; then
    echo "[1/3] resources.tar already present — skipping download."
else
    echo "[1/3] Downloading resources.tar (~7 GB)..."
    wget --progress=dot:giga \
         --continue \
         -O "${TARBALL}" \
         "${DOWNLOAD_URL}"
    echo "Download complete."
fi

# ---------------------------------------------------------------------------
# 2. Extract outer tar to get hg38_resources.tar.zst
# ---------------------------------------------------------------------------
echo "[2/3] Extracting outer archive..."
tar -xf "${TARBALL}" -C "${RESOURCES_DIR}"

HG38_ZST="${RESOURCES_DIR}/hg38_resources.tar.zst"
if [ ! -f "${HG38_ZST}" ]; then
    echo "ERROR: hg38_resources.tar.zst not found after extraction."
    echo "Contents of ${RESOURCES_DIR}:"
    ls "${RESOURCES_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# 3. Decompress hg38_resources.tar.zst
# ---------------------------------------------------------------------------
echo "[3/3] Decompressing hg38_resources.tar.zst into ${HG38_DIR}..."
mkdir -p "${HG38_DIR}"

# zstd may not be on PATH — try loading module first
if ! command -v zstd &>/dev/null; then
    module load zstd 2>/dev/null || true
fi

if command -v zstd &>/dev/null; then
    zstd -d "${HG38_ZST}" -c | tar -xf - -C "${HG38_DIR}"
else
    # Fallback: some tar versions understand .zst natively
    tar --use-compress-program=unzstd -xf "${HG38_ZST}" -C "${HG38_DIR}" || {
        echo "ERROR: zstd not available. Install with: conda install -c conda-forge zstd"
        exit 1
    }
fi

echo ""
echo "============================================================"
echo "hg38 resources extracted to: ${HG38_DIR}"
echo "Contents:"
ls "${HG38_DIR}"
echo ""
echo "You can now run the DCIS SpaceTracer jobs:"
echo "  sbatch slurm/run_dcis1.sh"
echo "  sbatch slurm/run_dcis2.sh"
echo "============================================================"
