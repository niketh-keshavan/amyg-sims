# Publication-Ready Analysis: TD-fNIRS Amygdala Sensitivity

## Role: Architect Plan

This plan upgrades `python/analyze.py` from a shot-noise-only feasibility check to a publication-quality analysis with realistic noise, depth specificity, and SSR comparison.

## Current Problem

The analysis reports min detectable HbO = 0.003 μM and SNR > 1000. These are ~100-1000x too optimistic because:
- Noise model is Poisson-only (`1/√N`), ignoring physiological noise, IRF jitter, source fluctuations
- No depth specificity — amygdala is 0.077% of late-gate signal, scalp is 3.1%
- Detector selection metric favors short-SDS (high photon count) over depth-specific channels
- No SSR regression comparison

---

## Step 1: Realistic Noise Model in `analyze.py` [DELEGATE TO SONNET]

**Difficulty: MEDIUM — delegate to another agent**

### Physics

Total noise has three independent contributions summed in quadrature:

```
σ_total = √(σ_shot² + σ_physio² + σ_detector²)
```

Where:
- `σ_shot = 1/√N` (already implemented)
- `σ_physio = k_physio` (relative noise floor from physiological fluctuations)
- `σ_detector = √(dark_counts) / N` (dark count contribution, already partially implemented)

### Physiological Noise Parameters

Add these constants to the system parameters section (~line 46):

```python
# Physiological noise floor (relative intensity fluctuation)
# Cardiac (~1 Hz): 0.1-0.5% of DC signal
# Respiratory (~0.2 Hz): 0.05-0.2%
# Mayer waves (~0.1 Hz): 0.1-0.5%
# Combined RMS: ~0.2-1.0% (Scholkmann et al. 2014, Tian & Liu 2014)
# TD-fNIRS has LOWER physio noise in late gates because late photons
# are a smaller fraction of total signal — use gate-dependent model
PHYSIO_NOISE_CW = 0.005      # 0.5% for CW (conservative estimate)
PHYSIO_NOISE_LATE_GATE = 0.002  # 0.2% for late gates (>3ns) — reduced by gating

# Source intensity fluctuation (laser power stability)
SOURCE_STABILITY = 0.001     # 0.1% RMS (typical pulsed laser)

# IRF timing jitter
IRF_JITTER_PS = 50.0         # 50 ps RMS jitter in timing electronics
```

### Where to Change Noise Calculation

Every place that computes `noise = 1/√N` or `sigma = 1/√N` must become:

```python
def compute_noise(N_photons, gate_idx, N_total_cw):
    """Realistic noise model combining shot + physiological + source noise.

    Args:
        N_photons: detected photon count in this gate
        gate_idx: time gate index (0-9), used for gate-dependent physio noise
        N_total_cw: total CW photon count for this detector (for physio noise scaling)
    """
    # Shot noise
    sigma_shot = 1.0 / np.sqrt(N_photons) if N_photons > 0 else float('inf')

    # Physiological noise — gate-dependent
    # Late gates (>3ns, idx 6+) have reduced physio noise
    if gate_idx >= 6:
        k_physio = PHYSIO_NOISE_LATE_GATE
    elif gate_idx >= 3:
        k_physio = (PHYSIO_NOISE_CW + PHYSIO_NOISE_LATE_GATE) / 2  # interpolate
    else:
        k_physio = PHYSIO_NOISE_CW

    # Physio noise is relative to CW signal, affects gate measurement
    # as a fraction of the gate's signal that leaks from physio fluctuations
    sigma_physio = k_physio

    # Source intensity noise
    sigma_source = SOURCE_STABILITY

    # Quadrature sum
    return np.sqrt(sigma_shot**2 + sigma_physio**2 + sigma_source**2)
```

### Files & Locations to Modify

| Location | Current Code | Change |
|----------|-------------|--------|
| `analyze.py:225-226` | `N_det = gw * scale + dark; snr = delta_od * sqrt(N_det)` | Use `compute_noise()` instead of `1/√N` |
| `analyze.py:268` | `noise = 1.0 / np.sqrt(N_total)` | `noise = compute_noise(N_total, g_idx, cw_weight * scale)` |
| `analyze.py:373-374` | `'sigma0': 1.0/np.sqrt(N0)` | `'sigma0': compute_noise(N0, g_idx, ...)` |
| `analyze.py:557-560` | `n0 = 1.0 / np.sqrt(N0)` | Same pattern |

### Expected Impact

- Min detectable HbO should increase from 0.003 → ~0.3-3 μM
- Block-design SNR should decrease from >1000 → ~1-20
- Late gates should still show advantage over CW (physio noise is lower in late gates)

---

## Step 2: Depth Specificity Analysis [ARCHITECT — I will design, DELEGATE implementation]

**Difficulty: MEDIUM — delegate implementation to another agent**

### Physics

The key question isn't "can we detect a signal change?" but "can we attribute it to the amygdala vs scalp?"

Add a new analysis section (Section 8) to `analyze.py`:

```
8. DEPTH SPECIFICITY ANALYSIS
```

### Contamination Ratio

For each detector/gate, compute:

```python
# Typical task-evoked scalp hemodynamics during emotional paradigm
# Sympathetic activation → peripheral vasoconstriction → ΔHbO_scalp ~ 1-3 μM
DELTA_HBO_SCALP_UM = 2.0  # μM (conservative)
DELTA_HBO_AMYGDALA_UM = 3.0  # μM (expected amygdala response)

# Signal from scalp contamination
scalp_PL = gate['partial_pathlength_mm']['scalp']
amyg_PL = gate['partial_pathlength_mm']['amygdala']

# ΔOD from scalp
dOD_scalp = epsilon_hbo * DELTA_HBO_SCALP_UM * 1e-3 * scalp_PL
# ΔOD from amygdala
dOD_amyg = epsilon_hbo * DELTA_HBO_AMYGDALA_UM * 1e-3 * amyg_PL

contamination_ratio = dOD_scalp / dOD_amyg  # want this < 1 for specificity
depth_specificity = dOD_amyg / (dOD_amyg + dOD_scalp)  # fraction of signal from amygdala
```

### Output Format

```
  Det  SDS  Gate    AmygPL   ScalpPL  ContamRatio  Specificity  AmygCNR
  ---  ---  ----    ------   -------  -----------  -----------  -------
    0    8  5ns+    0.983    38.87       26.4x       3.6%        0.04
   22   35  5ns+    0.795    39.90       33.5x       2.9%        0.03
```

### Key Insight to Report

The contamination ratio will be ~25-40x for all detectors/gates. This means:
- **TD-gating helps** (CW contamination ratio is ~1000x, so late-gate is 25-40x better)
- **But specificity is still poor** — amygdala contributes <4% of signal even in the best case
- This is the honest conclusion to present in the paper

---

## Step 3: Fix Detector Selection Metric [DELEGATE TO SONNET]

**Difficulty: LOW — delegate to another agent**

### Current Problem

`analyze.py:379` uses `metric = Σ(L_amyg × √N)` which favors high-N detectors (short SDS).

### Fix

Replace with depth-weighted metric:

```python
# New metric: weight by depth specificity, not raw photon count
# amyg_PL / scalp_PL captures how "deep-selective" the measurement is
metric = sum(
    (gd['L0'] / max(scalp_pl_0, 0.001)) * np.sqrt(gd['N0']) +
    (gd['L1'] / max(scalp_pl_1, 0.001)) * np.sqrt(gd['N1'])
    for gd in gate_data
)
```

This requires passing scalp PL through the gate_data structure. Add `'scalp_pl_0'` and `'scalp_pl_1'` fields to the gate_data dict at line 370.

---

## Step 4: SSR Regression Comparison [DELEGATE TO SONNET]

**Difficulty: MEDIUM — delegate to another agent**

### Concept

Short-separation regression uses 8mm SDS channels to estimate superficial signal, then subtracts from long-SDS channels:

```python
# For each long-SDS detector:
# signal_corrected = signal_long - beta * signal_short
# where beta = cov(long, short) / var(short)
#
# In MC simulation terms:
# Residual amygdala sensitivity after SSR:
# L_amyg_corrected = L_amyg_long - beta * L_amyg_short
# where beta ≈ scalp_PL_long / scalp_PL_short (regression coefficient)
```

### Implementation

Add Section 9 to `analyze.py`:

```
9. SHORT-SEPARATION REGRESSION vs TD-GATING
```

For CW mode (no gating):
```python
for each long-SDS detector:
    beta = scalp_PL_long / scalp_PL_short  # regression coefficient
    L_amyg_after_ssr = L_amyg_long - beta * L_amyg_short
    # If L_amyg_after_ssr > 0, SSR helps isolate amygdala
    # Compare this with TD-gated L_amyg (no SSR needed)
```

For TD + SSR combined:
```python
# Apply SSR within each time gate
# This is the strongest approach — TD-gating + SSR together
```

### Expected Result

- CW + SSR will likely show near-zero amygdala sensitivity (SSR removes everything including the tiny amygdala component)
- TD late-gate + SSR may preserve some amygdala signal
- TD late-gate alone (no SSR) has ~0.077% specificity
- This comparison demonstrates the value of TD-gating for deep targets

---

## Step 5: Run Diffusion Validation [DELEGATE TO SONNET]

**Difficulty: LOW — delegate to another agent**

`python/validate_diffusion.py` already exists but hasn't been run against MMC results.

Task: Run it on available data and report the agreement. If MMC data isn't available yet, run it on the voxel data to validate the shared physics engine.

```bash
python3 python/validate_diffusion.py --data-dir <results_dir>
```

---

## Step 6: Skull Thickness Sensitivity [REQUIRES NEW SIM RUNS — DEFERRED]

**Difficulty: HIGH — requires re-running simulations with modified geometry**

This requires modifying the mesh generation (`python/generate_mni152_mesh.py`) to produce meshes with different temporal bone thickness (1.5, 2.0, 2.5, 3.0, 3.5, 4.0 mm), then running 1B+ photons on each.

**Decision**: Defer until after Steps 1-4 are implemented and the 10B MMC results are in. The analysis improvements alone may change whether this is necessary.

---

## Step 7: Confidence Intervals [REQUIRES NEW SIM RUNS — DEFERRED]

**Difficulty: MEDIUM — requires multiple sim runs with different seeds**

Run 5-10 simulations with different random seeds, report mean ± SD for all metrics. Deferred until after Steps 1-4.

---

## Implementation Order

| Step | Priority | Difficulty | Who | Status |
|------|----------|------------|-----|--------|
| 1. Realistic noise model | CRITICAL | Medium | Sonnet agent | [Completed] |
| 2. Depth specificity analysis | CRITICAL | Medium | Sonnet agent | [Completed] |
| 3. Fix detector selection metric | HIGH | Low | Sonnet agent | [Completed] |
| 4. SSR regression comparison | HIGH | Medium | Sonnet agent | [Completed] |
| 5. Run diffusion validation | MEDIUM | Low | Sonnet agent | PENDING |
| 6. Skull thickness sweep | LOW | High | Deferred (needs GPU) | DEFERRED |
| 7. Confidence intervals | LOW | Medium | Deferred (needs GPU) | DEFERRED |
| 8. Fix multi-channel gate filter | HIGH | Low | Sonnet agent | [Completed] |

## Results After Steps 1-4 (voxel 10B proxy)

### Key Numbers (Before → After realistic noise)

| Metric | Shot-noise only | With realistic noise | Expected range |
|--------|----------------|---------------------|----------------|
| Multi-ch min HbO (120s) | 0.003 μM | **9.28 μM** | 0.3-10 μM |
| Block-design SNR (5 μM) | 2840 | **0.9** | 1-20 |
| Single-det best HbO | 0.000 μM | **0.119 μM** | 0.1-3 μM |
| Depth specificity (best) | N/A | **4.45%** (det 1, gate 9) | — |
| Contamination ratio (best) | N/A | **21.5x** (det 1, gate 9) | — |
| TD vs CW improvement | N/A | **430-55000x** | — |
| SSR signal preserved | N/A | **85-99%** | — |

### Critical Finding: Multi-channel is WORSE than single-detector

Multi-channel min HbO = 9.28 μM vs single-detector best = 0.119 μM. The multi-channel MBLL includes early gates (1.5-3ns) where amygdala PL is negligible (0.0003mm) but physiological noise is 0.5%. These measurements add noise without information, degrading the weighted least-squares solution.

**Fix needed (Step 8)**: Filter gates in multi-channel to only include those with amygdala PL > 0.01 mm. This should bring multi-channel closer to single-detector performance.

### Honest Conclusion

With realistic noise, amygdala HbO is **marginally detectable** (single-detector, best gate, ~0.12 μM threshold) but **HbR is NOT detectable** (17.4 μM threshold vs 0.5-1.5 μM expected). The MBLL matrix is ill-conditioned at these pathlengths.

TD-gating provides a genuine 430-55000x improvement in depth specificity over CW, which IS the publishable result.

## Files to Modify

| File | Changes |
|------|---------|
| `python/analyze.py` | Steps 1-4 [done], Step 8: gate filter in multi-channel |
| `python/validate_diffusion.py` | Step 5: run and verify |

## Verification

After Steps 1-4, re-run on voxel 10B data as proxy:
```bash
python3 python/analyze.py --data-dir results/voxel_based/results_10b_pulled/results_10b/
```

Expected changes in output:
- Min detectable HbO: 0.003 → 0.3-3 μM (realistic range)
- Block-design SNR: >1000 → 1-20 (realistic range)
- New Section 8 showing contamination ratio ~25-40x
- New Section 9 showing TD vs SSR comparison
- Best detectors should shift from SDS 8mm to SDS 25-35mm

---

## Change Log

| Date | Change |
|------|--------|
| 2026-03-17 | Initial plan: 7 steps to publication-quality analysis |
| 2026-03-17 | Step 8 completed: Added 0.01 mm minimum amygdala PL threshold to mbll_multi_channel() gate selection |
