# fNIRS Monte Carlo Simulation — Amygdala Oxygenation

GPU-accelerated Monte Carlo photon transport simulation for evaluating the feasibility of measuring amygdala hemodynamic activity using functional near-infrared spectroscopy (fNIRS).

## Overview

This project simulates NIR photon propagation through a realistic five-layer ellipsoidal adult head model to assess whether amygdala-level signals are detectable with current fNIRS technology. It compares four measurement modalities:

1. **Continuous-wave (CW)** — standard intensity-based fNIRS
2. **Time-domain (TD)** — time-gated photon detection using TPSFs
3. **Frequency-domain (FD)** — phase and amplitude from FFT of TPSF
4. **Chirp-correlated** — matched-filter processing gain with frequency-swept modulation

## Head Model

Ellipsoidal five-layer model with realistic adult proportions:

| Layer | Semi-axes (ML x AP x SI) mm | Thickness |
|-------|----------------------------|-----------|
| Scalp | 78 x 95 x 85 | 4 mm |
| Skull | 74 x 91 x 81 | 7 mm |
| CSF | 67 x 84 x 74 | 1.5 mm |
| Gray matter | 65.5 x 82.5 x 72.5 | 3.5 mm |
| White matter | 62 x 79 x 69 | — |
| Amygdala | 5 x 9 x 6 (at depth ~50 mm from temporal scalp) | — |

Optical properties at 760 nm and 850 nm from Okada & Delpy (2003), Jacques (2013), Strangman et al. (2014).

## Detector Array

22-channel high-density array on the right temporal scalp:
- 4 short-separation channels (8 mm) for superficial regression
- 6 primary-direction channels (15–45 mm SDS)
- 8 off-axis channels at +/-30 deg and +/-60 deg

## Laser Safety

100 mW pulsed laser at 760/850 nm. ANSI Z136.1 MPE analysis is included in the analysis pipeline (`analyze.py` section 6) confirming compliance for both skin and ocular exposure limits.

## Building

Requirements: CUDA toolkit (>=11.0), CMake (>=3.18), GPU with compute capability >=7.0.

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Running

```bash
# Run simulation (default: 100M photons per wavelength)
./mc_fnirs --photons 100000000 --output ../results

# Analysis
cd python
pip install -r requirements.txt
python analyze.py --data-dir ../results

# Visualization (generates 12 figures)
python visualize.py --data-dir ../results --output-dir ../figures
```

### Docker

```bash
bash scripts/run_docker.sh
```

### Cloud GPU (Vast.ai)

```bash
bash scripts/run_on_vast.sh
```

## Output Files

| File | Description |
|------|-------------|
| `results_{760,850}nm.json` | Per-detector CW + time-gated results |
| `tpsf_{760,850}nm.bin` | TPSF histograms (512 bins x 10 ps) |
| `fluence_{760,850}nm.bin` | 3D absorbed energy distribution |
| `paths_pos_{760,850}nm.bin` | Recorded photon trajectories |
| `volume.bin` | Voxelized tissue label volume |
| `volume_meta.json` | Grid dimensions and tissue labels |

## Analysis Pipeline

1. **CW sensitivity** — detected weight, amygdala partial pathlength, sensitivity ratio
2. **Time-domain** — TPSF statistics, time-gated sensitivity per gate
3. **Frequency-domain** — phase shift, amplitude attenuation, differential phase
4. **Chirp correlation** — matched filter processing gain (5–500 MHz sweep)
5. **SNR comparison** — minimum detectable hemoglobin concentration (MBLL inversion)
6. **ANSI Z136.1 MPE** — laser safety compliance verification

## References

- Okada E, Delpy DT (2003). Near-infrared light propagation in an adult head model. *Applied Optics*.
- Strangman GE et al. (2014). Scalp and skull influence on near infrared photon propagation. *NeuroImage*.
- Jacques SL (2013). Optical properties of biological tissues. *Physics in Medicine and Biology*.
- Amunts K et al. (2005). Cytoarchitectonic mapping of the human amygdala. *Brain Research Reviews*.
- ANSI Z136.1-2014. Safe use of lasers.
