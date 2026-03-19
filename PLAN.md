# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

Compare methods head-to-head: CW alone → TD gating → SSR → TD+SSR → DOT

## Current State

- **Step 1**: COMPLETED - Smoke test passed (1M photons on RTX 5090). Detectors now correctly placed at actual SDS 7.4-8.5mm (target 8mm) vs previous bug of 27mm.
- **Step 2**: IN PROGRESS - 1M smoke test complete. Ready for 10B production run.
- **Steps 3-4**: COMPLETED (code) - NOT TESTED (pending 10B data)

Goal: Naive single-channel MBLL gave min detectable ΔHbO = 549 μM (need 2-5 μM). Need ~200x improvement via proper depth-discrimination methods.

## Step 1: Fix Detector Placement + Add Short-SDS [COMPLETED - TESTED ✓]

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

## Step 2: Re-run 10B Simulation [IN PROGRESS]

1M smoke test passed (8.3s @ 730nm, 6.1s @ 850nm). Detectors correctly placed:
- Det 0: target=8mm, actual=8.5mm ✓
- Det 1: target=8mm, actual=7.4mm ✓  
- Det 2: target=15mm, actual=15.7mm ✓

**Production run (10B photons):**
```bash
./mmc/mmc_fnirs --mesh ../mni152_head_maxvol5.mmcmesh --photons 10000000000 --wavelengths 730,850 --output ../data_mmc_10B_v2
```

Requires RTX 5090 (~30 min). **10B run approved - execute when ready.**

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
