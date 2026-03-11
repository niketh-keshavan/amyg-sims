#!/bin/bash
# ---------------------------------------------------------------------------
# Build and run using Docker (works on any machine with nvidia-docker)
# ---------------------------------------------------------------------------
#
# Usage: bash scripts/run_docker.sh [NUM_PHOTONS]
# ---------------------------------------------------------------------------

set -euo pipefail

NUM_PHOTONS=${1:-100000000}

echo "Building Docker image..."
docker build -t fnirs-mc .

echo ""
echo "Running simulation with $NUM_PHOTONS photons..."
docker run --gpus all \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/figures:/app/figures" \
    fnirs-mc \
    --photons $NUM_PHOTONS --output /app/data

echo ""
echo "Running analysis..."
docker run --gpus all \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/figures:/app/figures" \
    --entrypoint python3 \
    fnirs-mc \
    /app/python/analyze.py --data-dir /app/data

docker run --gpus all \
    -v "$(pwd)/data:/app/data" \
    -v "$(pwd)/figures:/app/figures" \
    --entrypoint python3 \
    fnirs-mc \
    /app/python/visualize.py --data-dir /app/data --output-dir /app/figures

echo ""
echo "Done! Results in data/ and figures/"
