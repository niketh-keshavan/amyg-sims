#!/bin/bash
# Build script for MMC fNIRS simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Building MMC fNIRS Simulation"
echo "========================================"

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo ""
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "========================================"
echo "Build complete: ./build/mmc_fnirs"
echo "========================================"
