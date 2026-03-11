#!/bin/bash
# ---------------------------------------------------------------------------
# Deploy and run on a vast.ai instance
# ---------------------------------------------------------------------------
#
# Usage: bash scripts/run_on_vast.sh <INSTANCE_ID> [NUM_PHOTONS]
#
# Prerequisites:
#   - vast.ai CLI installed and authenticated
#   - Instance already created and running
# ---------------------------------------------------------------------------

set -euo pipefail

INSTANCE_ID=${1:?"Usage: $0 <INSTANCE_ID> [NUM_PHOTONS]"}
NUM_PHOTONS=${2:-1000000000}

echo "Getting instance SSH info..."
SSH_INFO=$(vastai ssh-url $INSTANCE_ID)
SSH_HOST=$(echo $SSH_INFO | cut -d@ -f2 | cut -d: -f1)
SSH_PORT=$(echo $SSH_INFO | cut -d: -f3)
SSH_USER=$(echo $SSH_INFO | cut -d@ -f1 | cut -d' ' -f2)

SSH_CMD="ssh -p $SSH_PORT $SSH_USER@$SSH_HOST"
SCP_CMD="scp -P $SSH_PORT"

echo "SSH: $SSH_CMD"

# Upload source code
echo ""
echo "Uploading source code..."
$SCP_CMD -r CMakeLists.txt include/ src/ python/ scripts/ $SSH_USER@$SSH_HOST:~/

# Build and run on the instance
echo ""
echo "Building and running simulation..."
$SSH_CMD << REMOTE_EOF
set -ex

# Install build deps if needed
which cmake || (apt-get update && apt-get install -y cmake python3 python3-pip)
pip3 install numpy matplotlib 2>/dev/null || true

# Build
cd ~
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j\$(nproc)

# Run simulation
cd ~
mkdir -p data figures
echo ""
echo "Starting simulation with $NUM_PHOTONS photons..."
time ./build/mc_fnirs --photons $NUM_PHOTONS --output data

# Run analysis
echo ""
echo "Running analysis..."
cd python
python3 analyze.py --data-dir ../data
python3 visualize.py --data-dir ../data --output-dir ../figures
REMOTE_EOF

# Download results
echo ""
echo "Downloading results..."
mkdir -p results
$SCP_CMD -r $SSH_USER@$SSH_HOST:~/data/* results/
$SCP_CMD -r $SSH_USER@$SSH_HOST:~/figures/* results/

echo ""
echo "================================================"
echo "  Done! Results saved to results/"
echo "================================================"
