# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

Compare methods head-to-head: CW alone → TD gating → SSR → TD+SSR → DOT

## Current State

10B photon simulation complete but with broken detector geometry (short-SDS detectors placed at 27+ mm instead of 8-15mm). Naive single-channel MBLL gives min detectable ΔHbO = 549 μM (need 2-5 μM). Need ~200x improvement via proper depth-discrimination methods.

## Step 1: Fix Detector Placement + Add Short-SDS [COMPLETED - NOT TESTED]

**Bug**: `mmc_main.cu:359-391` tangent-stepping overshoots on curved scalp. 8mm target → 27mm actual. Iterative correction diverges for short distances.

**Fix**: Binary search on `scale` parameter for short SDS (≤15mm). Update detector array:

| SDS (mm) | Angle | Purpose | Count |
|----------|-------|---------|-------|
| 8 | 0°, 180° | SSR reference | 2 |
| 12 | ±60° | SSR reference | 2 |
| 27-46 | 0°, ±30°, ±60° | Deep sensing (keep existing) | ~20 |
| 45, 50 | ±45° | Extended sweet spot | 4 |

Target: ~28 detectors, 4 verified short-SDS for SSR.

**Files**: `mmc/src/mmc_main.cu:290-400`

## Step 2: Re-run 10B Simulation [PENDING]

Rebuild and run with fixed geometry:
```bash
cd build && cmake .. -DBUILD_MMC=ON -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
./mmc/mmc_fnirs --mesh ../mni152_head.mmcmesh --photons 10000000000 --wavelengths 730,850 --output ../data_mmc_10B_v2
```

**Verify**: results JSON has detectors at actual SDS 8-12mm.

Requires RTX 4090 (~30 min, <$1). **STOP AND PROMPT before running.**

## Step 3: Implement SSR + TPSF Moment Analysis [COMPLETED - NOT TESTED]

### 3a: Short-Separation Regression
- Use short-SDS (8-12mm) as scalp-only reference
- Regress out scalp component from each long-SDS detector
- Apply within each time gate (TD+SSR combined)
- Expected: 10-100x scalp contamination reduction
- **File**: `python/analyze.py:708-768` (existing stub, enable + fix)

### 3b: TPSF Moment Analysis
- Compute mean TOF ⟨t⟩ and variance σ²_t from 512-bin TPSF
- These moments are more sensitive to deep absorption changes than simple gating
- Δ⟨t⟩ and Δσ²_t as contrast metrics for amygdala
- Expected: 5-20x improvement over simple gating
- **File**: `python/analyze.py` (new section)

### 3c: Combined TD + SSR
- Apply SSR within each time gate
- Expected: 50-500x total improvement over CW-only

## Step 4: Multiple Sources + DOT Reconstruction [COMPLETED - NOT TESTED]

### 4a: Add Source Position CLI Support
- Support `--source-index N` to select from predefined source positions
- Source positions (MNI space, on temporal scalp surface):
  - Source 0: Current (T8 area) — (68.87, -26.49, -30.67)
  - Source 1: ~15mm anterior
  - Source 2: ~15mm posterior
  - Source 3: ~15mm superior
- **File**: `mmc/src/mmc_main.cu:137-270`

### 4b: Run 10B per Source (4 sources × 2 wavelengths = 8 runs)
Each ~30 min on RTX 4090. Total ~4 hr, ~$5.

**STOP AND PROMPT before running.**

### 4c: DOT Reconstruction
- Compute Jacobian J (n_measurements × n_voxels) from per-source MC sensitivity profiles
- Tikhonov-regularized inversion: Δx = (J^T J + λI)^{-1} J^T Δy
- Spatial constraints localize deep signals
- **File**: `python/dot_reconstruction.py` (new)

## Key Files

| File | Change |
|------|--------|
| `mmc/src/mmc_main.cu:290-400` | Fix detector placement, add short-SDS, add source selection |
| `python/analyze.py` | Enable SSR, add TPSF moments, add TD+SSR |
| `python/dot_reconstruction.py` | New: Jacobian DOT reconstruction |
| `python/visualize.py` | Update figures for new analysis methods |

## Verification

1. Smoke test (1M photons): short-SDS detectors at actual SDS ±2mm of target
2. 10B results: detector SDS values match targets
3. Analysis: min detectable ΔHbO < 20 μM with TD+SSR
4. DOT: reconstructed amygdala activation map shows localized signal
