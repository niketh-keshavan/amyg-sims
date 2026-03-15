# Mesh-based Monte Carlo (MMC) fNIRS Simulation

This subfolder contains a mesh-based Monte Carlo photon transport simulation using a tetrahedral head mesh derived from the MNI152 (ICBM 2009c) atlas.

## Overview

Unlike the voxel-based Monte Carlo in the parent directory, this implementation:

1. **Uses a tetrahedral mesh** generated from MNI152 anatomical data
2. **Implements ray-tetrahedron traversal** for photon propagation
3. **Models realistic 5-layer anatomy**: scalp, skull, CSF, gray matter, white matter + amygdala ROI
4. **Supports 22-channel fNIRS array** targeting the right amygdala

## Architecture

```
mmc/
├── include/          # Header files
│   ├── types.h              # Core data structures
│   ├── mesh_loader.h        # MNI152 mesh I/O
│   ├── mmc_kernel.h         # CUDA kernel declarations
│   ├── optical_props.h      # Tissue optical properties
│   ├── detector_array.h     # 22-channel fNIRS array
│   └── simulation_config.h  # Configuration management
├── src/              # Implementation
│   ├── main.cpp             # Entry point
│   ├── mesh_loader.cpp      # Mesh loading
│   ├── optical_props.cpp    # Optical properties
│   ├── detector_array.cpp   # Detector geometry
│   ├── simulation_config.cpp # CLI/config parsing
│   └── mmc_kernel.cu        # CUDA photon transport
├── python/           # Analysis tools
│   ├── analyze_mmc.py       # Sensitivity analysis
│   └── visualize_mmc.py     # Result visualization
└── data/             # Mesh storage (generated)
    └── mni152_head.mmcmesh  # Tetrahedral mesh
```

## Building

### Prerequisites

- CUDA Toolkit >= 11.0
- CMake >= 3.18
- GPU with compute capability >= 7.0

### Build Instructions

```bash
cd mmc
mkdir build && cd build
cmake ..
make -j$(nproc)
```

Or from the project root:

```bash
mkdir -p mmc/build && cd mmc/build
cmake ..
make
```

## Usage

### 1. Generate the MNI152 Mesh

First, generate the tetrahedral mesh from MNI152 atlas:

```bash
cd python
pip install nilearn nibabel scikit-image scipy numpy tetgen
python generate_mni152_mesh.py --output ../mmc/data/mni152_head.mmcmesh --max-vol 15
```

This downloads the ICBM152 2009c atlas and creates a ~500K element tetrahedral mesh.

### 2. Run the Simulation

```bash
cd mmc/build
./mmc_fnirs --mesh ../data/mni152_head.mmcmesh --photons 100000000 --output ../results_mmc
```

Options:
- `--mesh <path>`: Path to .mmcmesh file
- `--photons <N>`: Photons per wavelength
- `--photons-760 <N>`: Photons at 760 nm only
- `--photons-850 <N>`: Photons at 850 nm only
- `--output <dir>`: Output directory
- `--max-bounces <N>`: Maximum scattering events
- `--gpu <id>`: GPU device ID

### 3. Analyze Results

```bash
cd python
python analyze_mmc.py --data-dir ../results_mmc --output mmc_analysis.json
```

### 4. Visualize

```bash
python visualize_mmc.py --data-dir ../results_mmc --output-dir ../figures_mmc
```

## Output Files

| File | Description |
|------|-------------|
| `tpsf_760mm.bin` | Time-resolved photon counts at 760 nm (22 detectors × 512 bins) |
| `tpsf_850mm.bin` | Time-resolved photon counts at 850 nm |
| `paths_760mm.bin` | Photon path records for sensitivity analysis |
| `paths_850mm.bin` | Photon path records |

## Photon Transport Algorithm

The simulation uses a **ray-tetrahedron traversal** algorithm:

1. **Launch**: Photons start at source position with isotropic direction
2. **Find tet**: Determine which tetrahedron contains the photon
3. **Step**: Sample exponential step length based on local μ_t
4. **Intersect**: Find intersection with tetrahedron faces
5. **Transfer**: If step crosses face, move to neighboring tet
6. **Scatter**: Apply Henyey-Greenstein scattering
7. **Detect**: Check if photon exits through detector aperture

### Key Features

- **Plücker coordinates** for robust ray-tetrahedron intersection
- **Russian roulette** for variance reduction
- **Neighbor connectivity** for O(1) tet traversal
- **Path recording** for sensitivity analysis

## Optical Properties

| Tissue | μa (760nm) | μs (760nm) | g | n |
|--------|-----------|-----------|---|---|
| Scalp | 0.019 | 1.2 | 0.9 | 1.4 |
| Skull | 0.012 | 1.0 | 0.9 | 1.55 |
| CSF | 0.004 | 0.01 | 0.0 | 1.35 |
| Gray | 0.018 | 0.9 | 0.89 | 1.36 |
| White | 0.017 | 1.2 | 0.84 | 1.36 |
| Amygdala | 0.020 | 0.95 | 0.89 | 1.36 |

Sources: Okada & Delpy (2003), Jacques (2013), Strangman et al. (2014)

## Detector Array

The 22-channel array is arranged on the right temporal scalp:

- **4 short-separation** (8 mm): scalp regression
- **6 primary** (15-45 mm): toward amygdala
- **8 off-axis ±30°** (15-45 mm): angular sampling
- **4 off-axis ±60°** (15-35 mm): wide-angle sampling

Target: Right amygdala at MNI coordinates (+24, -2, -20) mm

## Differences from Voxel-Based MC

| Aspect | Voxel MC | Mesh MC |
|--------|----------|---------|
| Geometry | Ellipsoidal layers | MNI152 tetrahedral |
| Propagation | Grid stepping | Ray-tetra traversal |
| Boundary | Implicit | Explicit (faces) |
| Memory | 3D volume | Mesh elements |
| Accuracy | Layered approximation | Anatomically accurate |

## References

- Fang Q, Boas DA. (2009). Monte Carlo simulation of photon migration in 3D turbid media accelerated by graphics processing units. *Optics Express*.
- Fang Q. (2010). Mesh-based Monte Carlo method using fast ray-tracing in Plücker coordinates. *Biomedical Optics Express*.
- Okada E, Delpy DT. (2003). Near-infrared light propagation in an adult head model. *Applied Optics*.
