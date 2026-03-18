# fNIRS Monte Carlo Simulation — Amygdala Oxygenation

GPU-accelerated Monte Carlo photon transport simulation for evaluating the feasibility of measuring amygdala hemodynamic activity using functional near-infrared spectroscopy (fNIRS).

**Now with realistic MNI152 mesh-based Monte Carlo (MMC) simulation!**

---

## Overview

This project simulates NIR photon propagation through realistic adult head models to assess whether amygdala-level signals are detectable with current fNIRS technology. It supports two simulation approaches:

### 1. Voxel-Based Monte Carlo (Original)
Ellipsoidal five-layer head model with 0.5mm voxel resolution. Fast, efficient, but uses simplified geometry.

### 2. Mesh-Based Monte Carlo (MMC) — **NEW!**
Tetrahedral mesh generated from the MNI152 (ICBM2009c) brain atlas with:
- Realistic scalp surface (20% volume, ~5-7mm thick)
- Accurate amygdala placement at MNI coordinates (±24, -2, -20) mm
- Source and detectors placed on actual scalp surface
- 123,812 tetrahedral elements (max_vol=5.0)

---

## Head Models

### MNI152 Mesh Model (Recommended)

| Tissue | Tets | Volume | Fraction |
|--------|------|--------|----------|
| Air | 1,315 | 2.2 cm³ | 0.1% |
| Scalp | 41,833 | 582.4 cm³ | 20.1% |
| Skull | 21,238 | 598.1 cm³ | 20.6% |
| Gray matter | 37,261 | 1075.5 cm³ | 37.1% |
| White matter | 22,023 | 634.4 cm³ | 21.9% |
| Amygdala | 142 | 4.0 cm³ | 0.1% |

**Amygdala position**: (24.3, -1.8, -20.5) mm — within 0.5mm of MNI atlas coordinates!

### Voxel Ellipsoidal Model (Legacy)

| Layer | Semi-axes (ML x AP x SI) mm | Thickness |
|-------|----------------------------|-----------|
| Scalp | 78 x 95 x 85 | 4 mm |
| Skull | 74 x 91 x 81 | 7 mm |
| CSF | 67 x 84 x 74 | 1.5 mm |
| Gray matter | 65.5 x 82.5 x 72.5 | 3.5 mm |
| White matter | 62 x 79 x 69 | — |
| Amygdala | 5 x 9 x 6 (at depth ~50 mm) | — |

---

## Key Features

### Measurement Modalities
1. **Continuous-wave (CW)** — standard intensity-based fNIRS
2. **Time-domain (TD)** — time-gated photon detection using TPSFs (10 gates: 0-10 ns)
3. **Frequency-domain (FD)** — phase and amplitude from FFT of TPSF
4. **Chirp-correlated** — matched-filter processing gain with frequency-swept modulation

### Detector Array
23-channel high-density array on the right temporal scalp:
- 2 short-separation channels (8 mm SDS) for superficial regression
- 11 primary-direction channels (15–40 mm SDS) 
- 10 off-axis channels at ±30° and ±60°

All detectors placed on actual scalp surface with iterative SDS correction (±2mm tolerance).

### Source Placement
Projects from amygdala centroid to nearest external scalp boundary face, then placed 0.5mm inward with direction toward amygdala.

---

## Building

### Requirements
- CUDA toolkit (>=11.0, tested with 12.x and 13.x)
- CMake (>=3.18)
- GPU with compute capability >=7.0 (tested on RTX 4090)

### Build Both Simulators

```bash
mkdir build && cd build
cmake .. -DBUILD_MMC=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
make -j4
```

Binaries:
- `build/mc_fnirs` — Voxel-based Monte Carlo
- `build/mmc/mmc_fnirs` — Mesh-based Monte Carlo

---

## Running

### Mesh-Based Monte Carlo (Recommended)

```bash
# Generate mesh (first time only, ~30 min)
python python/generate_mni152_mesh.py --output mni152_head.mmcmesh --max-vol 5.0

# Run 100M photon simulation
./build/mmc/mmc_fnirs \
  --mesh mni152_head.mmcmesh \
  --photons 100000000 \
  --wavelengths 730,850 \
  --output data_mmc_100M

# Verify mesh and results
python python/diagnose_mesh.py --mesh mni152_head.mmcmesh --results data_mmc_100M

# Full analysis
python python/analyze.py --data-dir data_mmc_100M
```

### Voxel-Based Monte Carlo (Legacy)

```bash
# Run simulation
./build/mc_fnirs --photons 100000000 --output ../results

# Analysis
cd python
pip install -r requirements.txt
python analyze.py --data-dir ../results

# Visualization (generates 12 figures)
python visualize.py --data-dir ../results --output-dir ../figures
```

---

## Mesh Visualization

Generate interactive 3D HTML viewer:

```bash
# Surface-only viewer (fast, ~46MB)
python python/view_mmc_surface.py --mesh mni152_head.mmcmesh --output surface_viewer.html

# Full tetrahedral viewer (slower, larger)
python python/view_mmc_mesh.py --mesh mni152_head.mmcmesh --max-tets 50000 --output mesh_viewer.html
```

Open the HTML file in any browser — no server needed!

---

## Recent Fixes (March 2026)

### Critical Bug Fixes
1. **Source placement**: Now projects to actual mesh scalp surface (was 26mm inside head)
2. **Detector placement**: Iterative SDS correction ensures targets within ±2mm
3. **Scalp volume**: Fixed from 73% to ~20% with thin-shell algorithm (7mm thickness)
4. **Amygdala position**: Fixed dual-placement bug, now at correct MNI coordinates

### Results
- **Before**: Amygdala z = -10.7 mm (wrong), photons barely reaching
- **After**: Amygdala z = -20.5 mm (correct), late-gate PL = 0.1-1.5 mm

---

## Output Files

### MMC Mesh Output
| File | Description |
|------|-------------|
| `results_{730,850}nm.json` | Per-detector CW + time-gated results |
| `tpsf_{730,850}nm.bin` | TPSF histograms (512 bins x 10 ps) |
| `paths_pos_{730,850}nm.bin` | Recorded photon trajectories |
| `mesh_meta.json` | Mesh info and source position |

### Voxel Output
| File | Description |
|------|-------------|
| `fluence_{760,850}nm.bin` | 3D absorbed energy distribution |
| `volume.bin` | Voxelized tissue label volume |
| `volume_meta.json` | Grid dimensions and tissue labels |

---

## Analysis Pipeline

1. **CW sensitivity** — detected weight, amygdala partial pathlength
2. **Time-domain** — TPSF statistics, time-gated sensitivity per gate
3. **Frequency-domain** — phase shift, amplitude attenuation
4. **Chirp correlation** — matched filter processing gain
5. **SNR comparison** — minimum detectable hemoglobin concentration
6. **Laser safety** — ANSI Z136.1 MPE compliance verification

---

## Laser Safety

100 mW pulsed laser at 760/850 nm. ANSI Z136.1 MPE analysis confirms compliance for both skin and ocular exposure limits.

---

## Cloud GPU (Vast.ai)

```bash
# SSH to instance
ssh -p <port> root@<ip> -L 8080:localhost:8080

# Clone and build
git clone https://github.com/niketh-keshavan/amyg-sims.git
cd amyg-sims
mkdir build && cd build
cmake .. -DBUILD_MMC=ON -DCMAKE_CUDA_ARCHITECTURES=89
make -j4

# Run simulation
./build/mmc/mmc_fnirs --mesh mni152_head.mmcmesh --photons 10000000000 --output data_mmc_10B
```

---

## Docker

```bash
bash scripts/run_docker.sh
```

---

## References

- Okada E, Delpy DT (2003). Near-infrared light propagation in an adult head model. *Applied Optics*.
- Strangman GE et al. (2014). Scalp and skull influence on near infrared photon propagation. *NeuroImage*.
- Jacques SL (2013). Optical properties of biological tissues. *Physics in Medicine and Biology*.
- Amunts K et al. (2005). Cytoarchitectonic mapping of the human amygdala. *Brain Research Reviews*.
- ANSI Z136.1-2014. Safe use of lasers.
- Fang Q, Boas DA (2009). Tetrahedral mesh generation from medical imaging data. *SPIE*.

---

## License

MIT License — see CITATION.cff for attribution.
