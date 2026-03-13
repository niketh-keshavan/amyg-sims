#!/bin/bash
# Run analysis and visualization for simulation data
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default data/figure directories (override with env vars)
DATA_DIR="${DATA_DIR:-data/730-850}"
FIG_DIR="${FIG_DIR:-figures/730-850}"

echo "=== Running analysis ==="
echo "  Data dir: $DATA_DIR"
echo "  Figure dir: $FIG_DIR"
python3 -u python/analyze.py --data-dir "$DATA_DIR"

echo ""
echo "=== Generating figures ==="
python3 -u python/visualize.py --data-dir "$DATA_DIR" --output-dir "$FIG_DIR"

echo ""
echo "=== Done ==="
echo ""
echo "# Example: run simulation with custom wavelengths:"
echo "#   ./build/mc_fnirs --photons 10000000000 --wavelengths 690,730,780,830,850 --output data/5wl"
echo "#   DATA_DIR=data/5wl FIG_DIR=figures/5wl ./run_analysis.sh"
