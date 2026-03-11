#!/bin/bash
# ---------------------------------------------------------------------------
# vast.ai setup and launch script for fNIRS Monte Carlo simulation
# ---------------------------------------------------------------------------
#
# Usage:
#   1. Install vast.ai CLI: pip install vastai
#   2. Set your API key:    vastai set api-key YOUR_KEY
#   3. Run this script:     bash scripts/vast_setup.sh [NUM_PHOTONS]
#
# This script:
#   - Searches for a suitable GPU instance
#   - Creates the instance
#   - Uploads the code
#   - Builds and runs the simulation
#   - Downloads the results
# ---------------------------------------------------------------------------

set -euo pipefail

# Load API key from .env if present
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/../.env"
if [ -f "$ENV_FILE" ]; then
    source "$ENV_FILE"
fi

if [ -n "${VAST_API_KEY:-}" ] && [ "$VAST_API_KEY" != "your_api_key_here" ]; then
    echo "Setting vast.ai API key from .env..."
    vastai set api-key "$VAST_API_KEY"
fi

NUM_PHOTONS=${1:-1000000000}  # Default: 1 billion photons
MIN_GPU_RAM=8                 # GB
DISK_SPACE=20                 # GB

echo "================================================"
echo "  fNIRS MC Simulation - vast.ai Deployment"
echo "  Photons: $NUM_PHOTONS"
echo "================================================"

# Search for cheap GPU instances with CUDA 12+
echo ""
echo "Searching for GPU instances..."
vastai search offers \
    "reliability > 0.95 cuda_vers >= 12.0 gpu_ram >= ${MIN_GPU_RAM} disk_space >= ${DISK_SPACE}" \
    --order "dph_total" \
    --limit 10

echo ""
echo "To create an instance, run:"
echo "  vastai create instance <OFFER_ID> --image nvidia/cuda:12.4.1-devel-ubuntu22.04 --disk $DISK_SPACE"
echo ""
echo "Once the instance is running, get the SSH command:"
echo "  vastai show instances"
echo ""
echo "Then use scripts/run_on_vast.sh <INSTANCE_ID> to deploy and run."
