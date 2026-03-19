# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

Compare methods head-to-head: CW alone → TD gating → SSR → TD+SSR → DOT

## Current State

- **Step 0**: IN PROGRESS - Source placement fix (min-distance instead of ray-projection)
- **Step 1**: COMPLETED - Detector placement fix (short-SDS binary search, 28 detectors)
- **Steps 2-3**: COMPLETED (code) - NOT TESTED (pending 10B data with fixed source)
- **Step 4**: COMPLETED (code) - NOT TESTED (pending multi-source data)

Goal: Naive single-channel MBLL gave min detectable ΔHbO = 549 μM (need 2-5 μM). Need ~200x improvement via source optimization + depth-discrimination methods.

## Step 0: Fix Source Placement [IN PROGRESS]

**Bug**: `find_source_on_scalp()` uses ray-projection (perpendicular distance to ray from origin through amygdala). This places source at T8 (y=-26mm), **52mm from amygdala**. The amygdala at (24, -2, -20) is only ~40-45mm from the scalp directly above it.

**Fix**: Replace ray-projection with minimum Euclidean distance from scalp face centroids to amygdala centroid. Source moves to FT8 area (y≈0), reducing distance by ~10mm.

**Multi-source update**: Offsets from the NEW optimal position for DOT cross pattern:
- Source 0: Optimal (min-distance to amygdala, FT8 area)
- Source 1: +20mm anterior (Y) — for DOT spatial diversity
- Source 2: +20mm posterior (Y) — brackets amygdala A-P
- Source 3: +15mm superior (Z) — vertical diversity

**File**: `mmc/src/mmc_main.cu:139-277, 562-637`

## Step 1: Fix Detector Placement + Add Short-SDS [COMPLETED ✓]

Binary search for short SDS (≤15mm), 28 detectors total:

| SDS (mm) | Angle | Purpose | Count |
|----------|-------|---------|-------|
| 8 | 0°, 180° | SSR reference | 2 |
| 12 | ±60° | SSR reference | 2 |
| 20-40 | 0°, ±30°, ±60° | Deep sensing | ~20 |
| 45, 50 | ±45° | Extended sweet spot | 4 |

Smoke-tested on RTX 5090: Det 0 actual=8.5mm, Det 1 actual=7.4mm ✓

## Step 2: Re-run 10B Simulation [PENDING]

After source fix, rebuild and run for each source position:
```bash
for i in 0 1 2 3; do
  ./mmc/mmc_fnirs --mesh ../mni152_head_maxvol5.mmcmesh \
    --photons 10000000000 --wavelengths 730,850 \
    --source-index $i --output ../data_mmc_10B_v2_src$i
done
```

4 sources × 2 wavelengths = 8 runs. ~30 min each on RTX 4090/5090. Total ~4 hr, ~$5.

**STOP AND PROMPT before running.**

**Verify**: `mesh_meta.json` shows source at y≈0 (not y=-26), distance to amygdala ~40-45mm.

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

1. Smoke test: source at y≈0, distance to amygdala ~40-45mm (not 52mm)
2. 10B results: improved amygdala PL (expect 2-5x better than previous)
3. Analysis: min detectable ΔHbO with TD+SSR < 20 μM
4. DOT: 4-source reconstruction recovers localized amygdala signal
