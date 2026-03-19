# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

## Current State — 1B Results (4 sources)

| Method | Min ΔHbO (μM) | Target: 2-5 μM | Notes |
|--------|--------------|-----------------|-------|
| CW MBLL (multi-ch) | 73.6 | ❌ 15× worse | Scalp dominates |
| SSR (CW) | worse than raw | ❌ | Adds noise |
| **TD gating (5ns+)** | **1.242** | **✓ 2-4× better** | Most trustworthy |
| TD + SSR | 24.985 | ❌ | SSR hurts late gates |
| TPSF mean-time | 0.055 | ✓ (shot-noise only) | Needs realistic noise model |
| TPSF variance | 0.016 | ✓ (shot-noise only) | Needs realistic noise model |
| DOT (4-source) | ∞ | ❌ | Cannot reconstruct |
| **Block design (20 trials)** | **SNR=6.36** | **✓ YES** | Det 29, 5 μM ΔHbO |

**Headline**: TD gating at 5ns+ achieves 1.242 μM (feasible). Block design with 20-trial averaging achieves SNR=6.36. TPSF moments show 0.055 μM but need realistic noise validation.

## Completed Steps

### Step 0: Source Placement [COMPLETED]
- Min-distance algorithm: source at (24.4, -10.2, -71.0), 60.8mm from amygdala
- This is the anatomical minimum for MNI152 mesh
- 4 source positions for DOT: optimal, +20mm ant, +20mm post, +15mm sup
- **Bug**: src3 (+15mm sup) snapped to same face as src0 — only 3 unique positions

### Step 1: Detector Placement [COMPLETED]
- 28 detectors with binary search for short SDS
- 4 short-SDS (8-12mm) verified for SSR reference
- All within ±2mm of target

### Step 2: 1B Simulation [COMPLETED]
- 4 sources × 2 wavelengths, RTX 5090, 6.0 Mph/s
- Data in `results_1B_full/src{0,1,2,3}/`

### Step 3: Analysis Code [COMPLETED]
- SSR (S9), TPSF moments (S10), TD+SSR (S11) in `python/analyze.py`
- SSR hurts late gates — regression adds noise exceeding amygdala signal
- TPSF moments give best shot-noise numbers but unrealistic (no IRF jitter, no physio noise)

### Step 4: DOT [COMPLETED — NEGATIVE]
- 4-source Jacobian: condition number 3-4M
- Amygdala in near-null SV (σ=0.6, 6th/7), <1% self-sensitivity
- MC validation: zero detection at any amplitude
- **DOT cannot reconstruct amygdala with 7-tissue model**

## Next Steps

### Step 5: 10B Production Run [PENDING]

10B photons needed for reliable amygdala statistics (1B shows amygdala PL ≈ 0).

```bash
./mmc/mmc_fnirs --mesh ../mni152_head_maxvol5.mmcmesh \
  --photons 10000000000 --wavelengths 730,850 \
  --source-index 0 --output ../data_mmc_10B_final
```

~30 min on RTX 5090. **STOP AND PROMPT before running.**

### Step 6: Fix TPSF Moment Noise Model [COMPLETED]

Added realistic noise model to `python/analyze.py` Section 10:

1. **IRF jitter**: Convolve TPSF with 80 ps FWHM Gaussian IRF before moment calculation
2. **Physiological noise**: 5 ps RMS added in quadrature to mean-time uncertainty
3. **Scalp contamination**: Mean-time shifts from scalp (50 ps) >> amygdala signal, adds multiplicative noise factor

**Noise model for mean-time**:
```
sigma_mean_t = sqrt(sigma_shot^2 + sigma_irf^2 + sigma_physio^2) * sqrt(scalp_factor)
```

**Expected outcome**: Realistic TPSF moment sensitivity moves from 0.055 μM (shot-noise only) to ~1-10 μM range.

### Step 7: Final Analysis + Figures [PENDING]

Run full pipeline on 10B data with fixed noise model:
```bash
python python/analyze.py --data-dir data_mmc_10B_final
python python/sensitivity_analysis.py --data-dir data_mmc_10B_final --output-dir figures/
python python/visualize.py --data-dir data_mmc_10B_final --output-dir figures/
```

## Key Files

| File | Role |
|------|------|
| `mmc/src/mmc_main.cu` | Source/detector placement, multi-source CLI |
| `python/analyze.py` | Full analysis: CW, TD, SSR, TPSF moments, TD+SSR |
| `python/dot_reconstruction.py` | DOT reconstruction (negative result) |
| `python/sensitivity_analysis.py` | Min detectable ΔHbO, skull sensitivity |
| `python/visualize.py` | Publication figures |
| `results_1B_full/src{0-3}/` | 1B multi-source data |

## Methods That Work vs Don't

| Works | Doesn't Work | Why |
|-------|-------------|-----|
| TD gating (5ns+) | CW MBLL | Scalp dominates CW by 70,000× |
| Block design averaging | SSR | Regression adds noise > amygdala signal |
| TPSF moments (with caveats) | DOT | Amygdala in null space of Jacobian |
| 850nm > 730nm | Multi-channel MBLL | System matrix ill-conditioned |
