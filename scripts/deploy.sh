#!/bin/bash
# ---------------------------------------------------------------------------
# Deploy and run on the vast.ai instance
# ---------------------------------------------------------------------------
# Usage: bash scripts/deploy.sh [NUM_PHOTONS]
#
# Reads SSH connection from .env or uses defaults.
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/.."
ENV_FILE="$PROJECT_DIR/.env"

# Load .env
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

SSH_HOST="${VAST_SSH_HOST:-199.68.217.31}"
SSH_PORT="${VAST_SSH_PORT:-47153}"
SSH_USER="${VAST_SSH_USER:-root}"
NUM_PHOTONS=${1:-1000000000}

SSH_CMD="ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
SCP_CMD="scp -P $SSH_PORT"

echo "================================================"
echo "  fNIRS MC - Deploying to vast.ai"
echo "  Server: $SSH_USER@$SSH_HOST:$SSH_PORT"
echo "  Photons: $NUM_PHOTONS"
echo "================================================"

# Upload source code
echo ""
echo "[1/4] Uploading source code..."
$SCP_CMD -r \
    "$PROJECT_DIR/CMakeLists.txt" \
    "$PROJECT_DIR/include" \
    "$PROJECT_DIR/src" \
    "$PROJECT_DIR/python" \
    "$SSH_USER@$SSH_HOST:~/"

# Build on the instance
echo ""
echo "[2/4] Building on GPU server..."
$SSH_CMD << 'REMOTE_BUILD'
set -ex
apt-get update -qq && apt-get install -y -qq cmake python3 python3-pip > /dev/null 2>&1 || true
pip3 install numpy matplotlib tqdm > /dev/null 2>&1 || true

cd ~
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1
make -j$(nproc) 2>&1
REMOTE_BUILD

# Run simulation
echo ""
echo "[3/4] Running simulation ($NUM_PHOTONS photons)..."
$SSH_CMD << REMOTE_RUN
set -ex
cd ~
mkdir -p data figures
echo "Starting simulation..."
time ./build/mc_fnirs --photons $NUM_PHOTONS --output data

echo ""
echo "Running analysis..."
cd python
python3 analyze.py --data-dir ../data
python3 visualize.py --data-dir ../data --output-dir ../figures
REMOTE_RUN

# Download results
echo ""
echo "[4/4] Downloading results..."
mkdir -p "$PROJECT_DIR/results"
$SCP_CMD -r "$SSH_USER@$SSH_HOST:~/data/*" "$PROJECT_DIR/results/"
$SCP_CMD -r "$SSH_USER@$SSH_HOST:~/figures/*" "$PROJECT_DIR/results/"

echo ""
echo "================================================"
echo "  Done! Results saved to results/"
echo "================================================"
