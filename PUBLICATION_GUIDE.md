# Publication Guide: fNIRS Monte Carlo Simulation

This document provides actionable steps to prepare the amygdala fNIRS simulation for peer-reviewed publication.

## Pre-Submission Checklist

### 1. Validation & Verification (CRITICAL)

- [ ] **Analytical validation**: Add `python/validate_diffusion.py`
  - Compare against diffusion approximation for semi-infinite medium
  - Use same optical properties as simulation
  - Target: <5% error in reflectance for SDS > 10mm
  
- [ ] **Convergence study**: Generate figure showing RSE vs photon count
  - Run 1M, 10M, 100M, 500M photons
  - Plot standard error of amygdala partial pathlength
  - Document minimum photons needed for target precision

- [ ] **Cross-validation with MCX** (optional but strong):
  - Export same geometry to MCX format
  - Compare TPSF shapes and CW reflectance

### 2. Sensitivity Analysis (HIGH PRIORITY)

- [ ] **Skull thickness sweep**: `python/sensitivity_skull.py`
  - Vary temporal bone: 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 mm
  - Show impact on min detectable HbO
  
- [ ] **Optical property uncertainty**: 
  - ±20% on μ_a, μ_s', g
  - Use Latin hypercube sampling for efficient coverage
  - Report 95% confidence intervals on main results

- [ ] **Amygdala position variability**:
  - Shift ±5mm in x, y, z directions
  - Document sensitivity to individual anatomy

### 3. Enhanced Analysis

- [ ] **Complete FD modality implementation**:
  - FFT of TPSF to get phase and amplitude
  - Add phase noise model
  - Compare FD vs TD sensitivity

- [ ] **Chirp modulation processing gain**:
  - Implement matched filter for 5-500 MHz sweep
  - Calculate compression gain
  - Compare SNR with TD gates

- [ ] **Short-separation regression comparison**:
  - Current code has 8mm SSR detectors
  - Add SSR regression noise model
  - Quantify TD advantage over SSR+CW

### 4. Statistical Improvements

- [ ] Add confidence intervals to all figures:
  ```python
  # Example for sensitivity plots
  se = std / np.sqrt(n_photons)  # standard error
  ci_95 = 1.96 * se
  ax.errorbar(x, y, yerr=ci_95, capsize=3)
  ```

- [ ] Report statistical significance:
  - Use paired t-tests for CW vs TD comparisons
  - Report effect sizes (Cohen's d)

- [ ] Multiple simulation runs:
  - Run 10 simulations with different seeds
  - Report mean ± SD for all metrics

### 5. Figure Enhancements

Priority improvements to existing figures:

| Figure | Improvement |
|--------|-------------|
| 01_tissue_slices | Add scale bar; label source position |
| 03_tpsf_curves | Add theoretical diffusion prediction |
| 04_td_sensitivity_heatmap | Add confidence intervals |
| 07_min_detectable_td | Add skull thickness comparison panel |
| 09_block_design | Add statistical power calculation |
| 10_photon_paths | Add 3D interactive version (HTML) |

New figures to add:
- **Figure 13**: Convergence study (RSE vs photons)
- **Figure 14**: Parameter sensitivity (tornado plot)
- **Figure 15**: Comparison with SSR regression

### 6. Documentation

- [ ] **METHODS.md**: Draft methods section for paper
- [ ] **PARAMETERS.md**: Justify all parameter choices with citations
- [ ] **VALIDATION.md**: Document all validation tests and results
- [ ] **README update**: Add "Citing this work" section with BibTeX

### 7. Reproducibility

- [ ] Pin exact versions in `requirements.txt`:
  ```
  numpy==1.24.3
  matplotlib==3.7.2
  ...
  ```

- [ ] Add `environment.yml` for conda users
- [ ] Create `CITATION.cff` file
- [ ] Add `reproduce.sh` script that runs full pipeline
- [ ] Include git commit hash in all output files

### 8. Supplementary Materials

Prepare for journal submission:

- **Supplementary Video**: Animation of photon propagation
- **Supplementary Data**: All TPSF and sensitivity values (CSV)
- **Supplementary Code**: Link to GitHub with release tag
- **Supplementary Figures**: Convergence, validation, sensitivity

## Suggested Paper Structure

### Introduction
- Motivation: Amygdala inaccessible to fNIRS due to depth
- Current state: CW fNIRS limited to cortex
- Innovation: TD-gated detection for deep sensitivity
- Hypothesis: TD-fNIRS can detect amygdala hemodynamics

### Methods
1. **Monte Carlo Simulation**: Photon transport, geometry, optical properties
2. **Head Model**: Ellipsoidal 5-layer, non-uniform skull
3. **Detector Array**: 22-channel high-density layout
4. **Signal Processing**: MBLL, time-gating, multi-channel combination
5. **Performance Metrics**: Min detectable ΔHbO/ΔHbR, SNR

### Results
1. **CW Baseline**: Amygdala sensitivity is minimal
2. **TD Enhancement**: Late gates preferentially sample deep tissue
3. **Optimal Configuration**: Best SDS, gate, and integration time
4. **Detectability**: Block-design paradigm feasibility

### Discussion
- Interpretation: Physical basis for TD advantage
- Limitations: Idealized geometry, single subject
- Clinical Implications: Feasibility for emotion research
- Future Work: Realistic head models, validation studies

## Reviewer-Expected Analyses

Be prepared to address:

1. **"Did you validate against experimental data?"**
   - Have a response ready (cite comparable published validations)
   - Consider adding comparison with existing fNIRS datasets

2. **"How sensitive are results to skull thickness?"**
   - Run the sensitivity analysis beforehand
   - Include as supplementary figure

3. **"Why 730/850 nm? What about other wavelengths?"**
   - Add wavelength optimization analysis
   - Document trade-offs (water absorption, PMT QE)

4. **"What about inter-subject variability?"**
   - Run simulations with varied head size (±10%)
   - Report robustness of conclusions

5. **"How does this compare with SSR regression?"**
   - Add SSR comparison analysis
   - Quantify TD advantage

## Quick Wins (Do These First)

1. **Add convergence plot** (~2 hours)
2. **Add confidence intervals to figures** (~3 hours)
3. **Document parameters** (~4 hours)
4. **Run skull thickness sweep** (~8 hours GPU time)
5. **Add CITATION.cff** (~30 minutes)

## Time Estimate

| Task | Time | Priority |
|------|------|----------|
| Validation script | 1 day | Critical |
| Sensitivity analysis | 2 days | High |
| Figure improvements | 2 days | High |
| Documentation | 1 day | Medium |
| Supplementary materials | 2 days | Medium |
| **Total** | **~1 week** | |

## Journal Recommendations

Suitable venues based on scope:

1. **NeuroImage** (high impact, methods focus)
2. **Biomedical Optics Express** (OSA, optics focus)
3. **Journal of Biomedical Optics** (SPIE, clinical translation)
4. **Neurophotonics** (SPIE, neuro-focused)

## Questions to Address

Before submission, ensure you can answer:

- [ ] What is the minimum detectable HbO change with 95% confidence?
- [ ] How many photons are needed for stable results?
- [ ] Which parameter has the largest impact on results?
- [ ] How do results compare to state-of-the-art?
- [ ] What are the practical limitations for real-world implementation?

---

*Last updated: 2026-03-14*
