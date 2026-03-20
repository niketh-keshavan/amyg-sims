# MMC Amygdala fNIRS — Depth-Discrimination Study

## Role: Actuator

Do not change architecture without updating this plan.
Designate all medium/low effort tasks to other agents via prompting me in chat.

## Paper Framing

"Evaluation of depth-discrimination strategies for deep-brain fNIRS via mesh-based Monte Carlo"

## Current State — 10B Results (single source, optimal placement)

| Method | Min ΔHbO (μM) | Target: <1 μM | Notes |
|--------|--------------|----------------|-------|
| **TD gating (5ns+), MBLL** | **0.231** | **✓ 4× better** | Det 29 (SDS=29mm), best single-det |
| TD gating, SDS=19mm | 0.520 | ✓ | Second best |
| TD gating, SDS=8mm | 0.987 | ✓ (marginal) | Short SDS, high photon count |
| TD gating, SDS=23mm | 0.831 | ✓ | |
| TD gating, SDS=36mm | 0.320 | ✓ | |
| CW MBLL (multi-ch) | 71.8 | ❌ | Scalp dominates |
| SSR (CW) | 300 | ❌ | Adds noise |
| TD + SSR | 300 | ❌ | SSR hurts late gates |
| TPSF mean-time | 33,967 | ❌ | Realistic noise kills it (IRF+physio+scalp) |
| TPSF variance | 8,454 | ❌ | Best at SDS=49mm, 850nm — still 8000× too high |
| DOT (4-source) | ∞ | ❌ | Cannot reconstruct |
| **Block design (20 trials)** | **visible** | **✓** | Clear task-evoked signal |

**Headline**: TD gating at 5ns+ achieves **0.231 μM** — 5.4× better than 1B (1.242 μM). Five detectors cross the 1 μM threshold. Block design shows clear task-evoked amygdala signal. TPSF moments, SSR, and DOT are all non-viable. **TD gating is the only viable method.**

### Key Physics (10B)
- Amygdala PL in 5ns+ gate: ~0.13-0.15mm (730nm), ~0.25-0.29mm (850nm)
- Scalp PL: ~800-900mm → contamination ratio 3700-13000× (even with TD gating)
- TD gating provides 100-4300× improvement over CW contamination
- 850nm gives ~2× higher amygdala PL than 730nm
- Noise floor is **physiological** (0.2% late gates), not shot noise
- Best detector: Det 29 (SDS=29mm, on-axis) — sweet spot of photon count vs depth

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

### Step 5: 10B Production Run [COMPLETED]
- 10B photons, single source (src0), 730/850nm
- Data in `results_10B_final/` (results_730nm.json, results_850nm.json, mesh_meta.json)
- All 28 detectors show non-zero amygdala PL in 5ns+ gate
- Best single-detector min ΔHbO: **0.231 μM** (Det 29, SDS=29mm)

### Step 6: Fix TPSF Moment Noise Model [COMPLETED]
- Added realistic noise model to `python/analyze.py` Section 10
- IRF jitter (100ps FWHM Gaussian), physiological noise, scalp contamination
- TPSF bins now available, moment analysis run on 10B data
- **Result**: TPSF moments completely non-viable — best is 8,454 μM (variance, Det 26, 850nm)
- Shot-noise-only estimate (0.055 μM) was off by 150,000× — realistic noise dominates

### Step 7: Final Analysis + Figures [COMPLETED]
- Full analysis pipeline run on 10B data: `python/analyze.py --data-dir results_10B_final`
- 15 publication figures generated in `results_10B_final/10B/`
- Sensitivity analysis and visualization complete

## 10B Detailed Results

### Single-Detector MBLL (5ns+ gate, 120s integration)

| SDS (mm) | Det | Min ΔHbO (μM) | Min ΔHbR (μM) | Status |
|----------|-----|--------------|---------------|--------|
| 8 | 0 | 0.987 | 129.3 | DETECTABLE |
| 19 | 4 | 0.520 | 124.9 | DETECTABLE |
| 23 | 5 | 0.831 | 118.9 | DETECTABLE |
| 25 | 6 | 9.240 | 113.3 | need >9.2 μM |
| **29** | **7** | **0.231** | **129.7** | **DETECTABLE** |
| 33 | 8 | 7.011 | — | need >7.0 μM |
| 34 | 9 | 1.720 | — | marginal |
| 36 | 10 | 0.320 | — | DETECTABLE |
| 39 | 11 | 3.330 | — | need >3.3 μM |

### Integration Time Scaling (min detectable ΔHbO in μM)

| SDS | 1s | 5s | 15s | 30s | 1m | 2m |
|-----|------|------|------|------|------|------|
| 29 | 0.23 | 0.10 | 0.06 | 0.04 | 0.03 | 0.02 |
| 19 | 0.52 | 0.23 | 0.13 | 0.09 | 0.07 | 0.05 |
| 8 | 0.99 | 0.44 | 0.26 | 0.18 | 0.13 | 0.09 |

### Depth Specificity (5ns+ gate)
- CW contamination ratio: 400,000–19,000,000× (scalp/amygdala)
- TD 5ns+ contamination ratio: 3,700–13,000×
- TD gain over CW: 25–4,300×

## Next Steps

### Step 8: TPSF Moment Analysis [COMPLETED — NEGATIVE]
- TPSF bins loaded (28 dets × 512 bins), convolved with 100ps FWHM IRF
- Best mean-time: 33,967 μM (Det 26, SDS=49mm, 850nm)
- Best variance: 8,454 μM (Det 26, SDS=49mm, 850nm)
- **TPSF moments cannot detect amygdala** — realistic noise (IRF jitter + physiological + scalp contamination) overwhelms the tiny temporal shifts from amygdala absorption

### Step 9: Publication Figures [COMPLETED]
- New script: `python/pub_figures.py` — reads binary MMCB mesh directly
- **Fig A** (`pub_A_anatomy_setup.png`): Coronal + sagittal mesh slices (element centroids colored by tissue type), source star, amygdala annotated, key detectors in sagittal view
- **Fig B** (`pub_B_main_results.png`): Min detectable ΔHbO vs SDS (CW vs TD-gated 5ns+) + amygdala partial pathlength vs time gate
- **Fig C** (`pub_C_tpsf.png`): TPSF curves for all primary detectors, SDS=29mm highlighted, 5ns+ gate marked
- **Fig D** (`pub_D_method_comparison.png`): Bar chart comparing all 6 methods (log scale), TD-gated 0.231 μM vs CW 71.8 μM – 300× better
- All figures saved to `figures/pub/` at 300 dpi

### Step 10: Paper Draft [IN PROGRESS]

#### Code Verification Summary [COMPLETE]

All physics components verified correct:
- Henyey-Greenstein, Beer-Lambert, Fresnel/TIR, Snell, Russian roulette: ✅
- Face-pair boundary traversal, atomicAdd fallback, scattering rescale: ✅
- IRF convolution, TPSF noise model (shot+IRF+physio+scalp in quadrature): ✅
- Non-blocking issue: point-in-tet tolerance ±1e-6 harmless at maxvol 5 mm³

#### Target Venue
**Neurophotonics** (primary, IF 3.8); **Biomedical Optics Express** (backup, IF 3.2); **Journal of Biomedical Optics** (tertiary, IF 2.9).
Paper written journal-agnostic; formatting applied at submission time.

#### Title
*"Time-domain fNIRS enables sub-micromolar amygdala detection: a mesh-based Monte Carlo simulation study"*

#### Paper Structure
1. **Introduction** — clinical motivation (amygdala in fear/PTSD, fMRI-only), fNIRS advantages, depth problem, prior art (CW SSR DOT TPSF), our contribution (systematic MC comparison)
2. **Methods** — head model (MNI152, 7-tissue, 347k tets), MMC solver (CUDA, HG, Fresnel, TPSF binning), optical properties table, optode config (FT8, 28 detectors), 6 analysis strategies, noise model, hardware (1W laser, S14160, ANSI safety)
3. **Results** — TD-gated 0.231 μM (main positive), TPSF failure (0.5ps shift vs 50ps scalp), method comparison table, depth specificity (TD gain 25–4300× over CW), block design SNR=40×
4. **Discussion** — why TD works (TOF separates depth), why CW/SSR/TPSF/DOT fail, hardware path (MONSTIR II, BONuS), limitations (single source/head model/no motion), future work
5. **Conclusion** — TD only viable method; 0.231 μM = 4× margin; single-trial feasible

#### Writing Steps

**Step 10A** — Finalize figures [COMPLETE]
- All 4 figures verified: font ≥8pt, axes labeled, threshold lines visible
- Added 600 dpi TIFF export alongside 300 dpi PNG for all 4 figures in `python/pub_figures.py`

**Step 10B** — Supplementary material [IN PROGRESS]
- Written in `paper/main.tex` (Supplementary sections S1–S4):
  - S1: Mesh generation procedure (iso2mesh, TetGen quality constraints)
  - S2: Full optical properties table with $n$ values (Jacques 2013)
  - S3: Noise model derivation — all 4 terms (shot, IRF, physiological, scalp contamination)
  - S4: 15 supplementary figure placeholders from `results_10B_final/10B/`
- `paper/refs.bib` created with 18 references

**Step 10C** — Methods section [IN PROGRESS]
- Written in `paper/main.tex` Section 2 (Methods):
  - MMC solver: RTE (Eq. 1), HG phase function (Eq. 2–3), Beer-Lambert weight update (Eq. 4–5)
  - Boundary physics: Fresnel (Eq. 6), Snell's law, TIR, Russian roulette
  - CUDA parallelization: BVH, atomicAdd, 6 Mph/s on RTX 5090
  - Optode config: FT8 source at MNI (24.4, -10.2, -71.0), 28 detectors 8–49 mm
  - MBLL matrix derivation (Eq. 7–9), photon count scaling (Eq. 9)
  - TPSF binning: 512 × 10 ps, moment definitions (Eq. 10–12)
  - TPSF moment sensitivity dt_dua (Eq. 13–14)
  - Full noise model (Eq. 15) — all 4 terms
  - SSR and DOT methods

**Step 10D** — Results section [COMPLETED]
- Min ΔHbO table (9 SDS values), integration-time table (3 detectors × 6 durations)
- Method comparison table delegated to Fig D (bar chart)
- All 6 method paragraphs written with numbers from 10B data
- Sensitivity analysis paragraph added (skull thickness ±3mm, optical props ±20%)
- ✅ FIGURE GENERATED: `figures/contamination_ratio_vs_gate.png` showing scalp/amygdala PL ratio vs time gate
  for best 4 detectors (SDS = 8,19,29,36mm), both wavelengths, log-scale y-axis.
  Replaced the old `11_cw_vs_td_sensitivity.png` reference in the paper.

**Step 10E** — Discussion + Conclusion [COMPLETED]
- Discussion written: 6 subsections (TD mechanism, failure modes, prior work, hardware, translation, limitations)
- Conclusion written: ~200 words, covers all 6 methods, hardware specs, next steps
- Abstract fixed: SNSPD claim replaced with SiPM (matches actual simulation)
- Integration time text clarified: shot-noise-dominated regime, not shot-noise-limited
- All \cite{} commands included in Discussion + Conclusion

**Step 10F** — Remaining fixes [COMPLETED]
- ✅ Add ALL \cite{} commands to Introduction, Methods, Results — added citations throughout
- ✅ Add missing refs to refs.bib — added Fonov 2011 (MNI), Selb 2014 (physiological noise), ANSI Z136.1
- ✅ Generate Figure 1 (anatomy/setup schematic) — already COMPLETED
- ✅ Generate contamination_ratio_vs_gate.png figure — COMPLETED, replaces old Fig 6
- ✅ Add end-matter: Disclosures, Code/Data Availability, Acknowledgments — all added
- ✅ Fix critical issues: detector IDs verified correct, RTX 5090 core count fixed (21,760), sensitivity methods clarified
- ✅ Fix minor issues: typo fixed, MC "exactly solving" wording improved, SSR description clarified
- ✅ Add block-design protocol to Methods, convergence analysis text — both added
- Organize supplementary material [IN PROGRESS]

**Step 10G** — Journal-specific formatting (at submission time)
- Neurophotonics: SPIE template, structured abstract (Purpose/Approach/Results/Conclusions, ≤200 words)
- BOE: Optica template, unstructured abstract
- Cover letter, author info, ORCIDs
- GitHub repo + Zenodo DOI deposit for code + data

## Key Files

| File | Role |
|------|------|
| `mmc/src/mmc_main.cu` | Source/detector placement, multi-source CLI |
| `python/analyze.py` | Full analysis: CW, TD, SSR, TPSF moments, TD+SSR |
| `python/dot_reconstruction.py` | DOT reconstruction (negative result) |
| `python/sensitivity_analysis.py` | Min detectable ΔHbO, skull sensitivity |
| `python/visualize.py` | Publication figures |
| `results_1B_full/src{0-3}/` | 1B multi-source data |
| `results_10B_final/` | **10B production data** |
| `results_10B_final/10B/` | **Publication figures (15 PNGs)** |

## Methods That Work vs Don't

| Works | Doesn't Work | Why |
|-------|-------------|-----|
| TD gating (5ns+) — 0.231 μM | CW MBLL — 71.8 μM | Scalp dominates CW by 4000–13000× |
| Block design averaging | SSR — 300 μM | Regression adds noise > amygdala signal |
| 850nm > 730nm (2× better PL) | DOT | Amygdala in null space of Jacobian |
| Single-det MBLL at 29mm SDS | Multi-channel MBLL | System matrix ill-conditioned |
| | TPSF moments — 8,454 μM | IRF+physio noise >> amygdala temporal shift |
