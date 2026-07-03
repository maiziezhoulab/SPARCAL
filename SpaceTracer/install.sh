#!/usr/bin/env bash
# =============================================================================
# install.sh — Set up SpaceTracer for the SNV-calling project
#
# Run this once from an interactive SLURM session or login node.
# Requires internet access (or module-loaded git/conda).
#
# Usage:
#   bash install.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the existing clone under leiy4/ to avoid re-downloading; fall back to local repo/
EXISTING_REPO="/panfs/accrepfs.vampire/data/maiziezhou_lab/leiy4/SpaceTracer"
REPO_DIR="${SCRIPT_DIR}/repo"
ENV_NAME="SpaceTracer_dcis"

echo "============================================================"
echo "SpaceTracer Install Script"
echo "Script dir : ${SCRIPT_DIR}"
echo "Conda env  : ${ENV_NAME}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1. Ensure the repo is at the latest v2.x release
# ---------------------------------------------------------------------------
if [ -d "${EXISTING_REPO}/.git" ]; then
    echo "[1/4] Existing repo found at ${EXISTING_REPO} — pulling latest v2.x..."
    git -C "${EXISTING_REPO}" pull
    REPO_DIR="${EXISTING_REPO}"
elif [ -d "${REPO_DIR}/.git" ]; then
    echo "[1/4] Local repo found — pulling latest..."
    git -C "${REPO_DIR}" pull
else
    echo "[1/4] Cloning SpaceTracer..."
    git clone https://github.com/douymLab/SpaceTracer.git "${REPO_DIR}"
fi
echo "    Using repo: ${REPO_DIR}"

# ---------------------------------------------------------------------------
# 2. Create conda environment
# ---------------------------------------------------------------------------
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[2/4] Conda env '${ENV_NAME}' already exists — skipping creation."
    echo "      To recreate: conda env remove -n ${ENV_NAME}"
else
    echo "[2/4] Creating conda environment '${ENV_NAME}' (this may take 20-40 min)..."
    # Use the repo's official environment.yml; our copy mirrors it exactly.
    conda env create -n "${ENV_NAME}" -f "${SCRIPT_DIR}/environment.yml"
fi

# ---------------------------------------------------------------------------
# 3. Install SpaceTracer package into the env
# ---------------------------------------------------------------------------
echo "[3/4] Installing SpaceTracer package..."
conda run -n "${ENV_NAME}" pip install -e "${REPO_DIR}"

# ---------------------------------------------------------------------------
# 4. Verify
# ---------------------------------------------------------------------------
echo "[4/4] Verifying installation..."
conda run -n "${ENV_NAME}" spacetracer --help | head -5

echo ""
echo "============================================================"
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "  1. Download hg38 resources (one-time, ~7 GB):"
echo "       bash ${SCRIPT_DIR}/download_resources.sh"
echo "  2. Submit DCIS jobs:"
echo "       sbatch ${SCRIPT_DIR}/slurm/run_dcis1.sh"
echo "       sbatch ${SCRIPT_DIR}/slurm/run_dcis2.sh"
echo "============================================================"
