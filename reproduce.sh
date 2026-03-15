#!/bin/bash
# Reproduce full simulation and analysis pipeline
# Usage: ./reproduce.sh [num_photons]

set -e  # Exit on error

NUM_PHOTONS=${1:-100000000}  # Default: 100M photons
OUTPUT_DIR="results"
FIGURE_DIR="figures"

echo "=========================================="
echo "fNIRS Monte Carlo Reproduction Script"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Photons per wavelength: $NUM_PHOTONS"
echo "  Output directory: $OUTPUT_DIR"
echo "  Figure directory: $FIGURE_DIR"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$FIGURE_DIR"

# Record environment
echo "Recording environment..."
date > "$OUTPUT_DIR/reproduction.log"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" >> "$OUTPUT_DIR/reproduction.log"
echo "CUDA version: $(nvcc --version 2>/dev/null | head -1 || echo 'N/A')" >> "$OUTPUT_DIR/reproduction.log"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')" >> "$OUTPUT_DIR/reproduction.log"
echo "" >> "$OUTPUT_DIR/reproduction.log"

# Build
echo ""
echo "Step 1: Building simulation..."
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
cd ..

# Run simulation
echo ""
echo "Step 2: Running Monte Carlo simulation..."
echo "  This may take 10-60 minutes depending on GPU and photon count"
echo ""
./build/mc_fnirs --photons "$NUM_PHOTONS" --output "$OUTPUT_DIR" --wavelengths 730,850 2>&1 | tee -a "$OUTPUT_DIR/reproduction.log"

# Analysis
echo ""
echo "Step 3: Running analysis..."
cd python
pip install -q -r requirements.txt

# Main analysis
python analyze.py --data-dir "../$OUTPUT_DIR" 2>&1 | tee -a "../$OUTPUT_DIR/reproduction.log"

# Validation
python validate_diffusion.py --data-dir "../$OUTPUT_DIR" --output-dir "../$FIGURE_DIR" 2>&1 | tee -a "../$OUTPUT_DIR/reproduction.log"

# Sensitivity analysis
python sensitivity_analysis.py --data-dir "../$OUTPUT_DIR" --output-dir "../$FIGURE_DIR" 2>&1 | tee -a "../$OUTPUT_DIR/reproduction.log"

# Visualization
python visualize.py --data-dir "../$OUTPUT_DIR" --output-dir "../$FIGURE_DIR" 2>&1 | tee -a "../$OUTPUT_DIR/reproduction.log"

# 3D Visualization
python visualize_3d.py --data-dir "../$OUTPUT_DIR" --output-dir "../$FIGURE_DIR" 2>&1 | tee -a "../$OUTPUT_DIR/reproduction.log"

cd ..

echo ""
echo "=========================================="
echo "Reproduction complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "Figures saved to: $FIGURE_DIR/"
echo "Log file: $OUTPUT_DIR/reproduction.log"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR/" | tail -n +2
ls -lh "$FIGURE_DIR/" | tail -n +2
