# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

Compare methods head-to-head: CW alone → TD gating → SSR → TD+SSR → DOT

## Current State

- **Step 0 (Source)**: COMPLETE - Min-distance placement achieved, but anatomical constraint: amygdala is ~61mm from nearest scalp in this mesh. Cannot achieve 40-50mm.
- **Step 1 (Detectors)**: COMPLETED ✓ - Smoke tested: 28 detectors, short-SDS at 7.4-8.5mm (target 8mm) ✓
- **Step 2 (10B Sim)**: READY TO RUN - All fixes verified, awaiting production run
- **Steps 3-4**: COMPLETED (code) - NOT TESTED (pending 10B data)

**Key Finding**: MNI152 mesh has amygdala positioned such that minimum scalp distance is ~61mm (not 40-50mm as hoped). This is an anatomical constraint, not a bug. Proceeding with optimal placement given mesh geometry.

Goal: Naive single-channel MBLL gave min detectable ΔHbO = 549 μM (need 2-5 μM). Need ~200x improvement via depth-discrimination methods.

## Step 0: Fix Source Placement [COMPLETED ✓]

**Attempts**:
1. Ray-projection method: Source at T8 area, **85mm** from amygdala (too far)
2. Min-distance method: Source at (24, -10, -71), **60.8mm** from amygdala (anatomical minimum)
3. 35-55mm range search: **No scalp points exist** in this range — amygdala is deeper in this mesh

**Result**: 60.8mm is the anatomical minimum for this MNI152 mesh. Source is properly placed on lateral temporal scalp, pointing inward toward amygdala.

**Multi-source for DOT**:
- Source 0: Optimal (min-distance, ~60.8mm)
- Source 1: +20mm anterior (Y) 
- Source 2: +20mm posterior (Y)
- Source 3: +15mm superior (Z)

**File**: `mmc/src/mmc_main.cu:139-277, 562-637`

## Step 1: Fix Detector Placement + Add Short-SDS [COMPLETED ✓]

Binary search for short SDS (≤15mm), 28 detectors total. **Smoke tested on RTX 5090 (1M photons)**:

| SDS (mm) | Angle | Purpose | Actual | Status |
|----------|-------|---------|--------|--------|
| 8 | 0°, 180° | SSR reference | 7.6mm, 7.3mm | ✓ |
| 12 | ±60° | SSR reference | 11.7mm, 11.1mm | ✓ |
| 20-40 | 0°, ±30°, ±60° | Deep sensing | 18-40mm | ✓ |
| 45, 50 | ±45° | Extended | 44-51mm | ✓ |

**All detectors within ±2mm of target. Ready for 10B production run.**

## Step 2: Re-run 10B Simulation [READY - APPROVED]

All fixes verified. Run production simulation:

```bash
# Source 0 (optimal)
./mmc/mmc_fnirs --mesh ../mni152_head_maxvol5.mmcmesh \
  --photons 10000000000 --wavelengths 730,850 \
  --source-index 0 --output ../data_mmc_10B_final
```

**Single source first** (~30 min on RTX 5090). Multi-source (1-3) can follow for DOT.

**Verified from smoke test**:
- Source distance: 60.8mm (anatomical minimum for this mesh)
- Source position: (24.4, -10.2, -71.0) mm — lateral temporal scalp
- 28 detectors correctly placed (short-SDS verified at 7-8mm)

**10B run APPROVED — execute when ready.**

## Step 3: SSR + TPSF Moment Analysis [CODE COMPLETE - NOT TESTED]

### 3a: Short-Separation Regression
- Use short-SDS (8-12mm) as scalp-only reference
- Regress out scalp component from each long-SDS detector per time gate
- Expected: 10-100x scalp contamination reduction
- **File**: `python/analyze.py` Section 9

### 3b: TPSF Moment Analysis
- Mean TOF ⟨t⟩, variance σ²_t, skewness from 512-bin TPSF
- Sensitivity of moments to amygdala absorption changes
- Expected: 5-20x improvement over simple gating
- **File**: `python/analyze.py` Section 10

### 3c: Combined TD + SSR
- Apply SSR within each time gate (late gates 6-9)
- Expected: 50-500x total improvement over CW-only
- **File**: `python/analyze.py` Section 11

## Step 4: DOT Reconstruction [CODE COMPLETE - NOT TESTED]

### Approach
- Build Jacobian J from 4-source partial pathlengths (7 tissue types × n_gates)
- Tikhonov-regularized inversion: Δx = (J^T J + λI)^{-1} J^T Δy
- L-curve for optimal λ, depth-weighted regularization
- Dual-wavelength recovery for ΔHbO/ΔHbR

### DOT + Source Fix Compatibility
`dot_reconstruction.py` is source-position-agnostic — it loads data from arbitrary directories and builds J from partial pathlengths. No code changes needed.

Cross pattern (optimal + ant + post + sup) gives 4 viewing angles, improving Jacobian condition number for the amygdala column.

**File**: `python/dot_reconstruction.py`

**Run**:
```bash
python python/dot_reconstruction.py \
  --data-dirs data_mmc_10B_v2_src0 data_mmc_10B_v2_src1 \
              data_mmc_10B_v2_src2 data_mmc_10B_v2_src3 \
  --output-dir figures/
```

## Key Files

| File | Change |
|------|--------|
| `mmc/src/mmc_main.cu` | Source fix (min-distance), detector fix (binary search), multi-source CLI |
| `python/analyze.py` | SSR (S9), TPSF moments (S10), TD+SSR (S11) |
| `python/dot_reconstruction.py` | Jacobian DOT reconstruction |

## Verification

| Step | Test | Result |
|------|------|--------|
| 1 | Smoke test (1M) | ✓ Short-SDS at 7.4-8.5mm (target 8mm) |
| 0 | Source placement | ✓ 60.8mm — anatomical minimum for mesh |
| 2 | 10B simulation | ⏳ Ready to run |
| 3 | Analysis (SSR/TD) | ⏳ Pending 10B data |
| 4 | DOT reconstruction | ⏳ Pending multi-source data |

**Known constraint**: MNI152 mesh amygdala is ~61mm from scalp (deeper than ideal 40-50mm). This is anatomically correct for this mesh — different head models may vary.
