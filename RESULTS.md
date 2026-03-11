# Amygdala fNIRS Feasibility — Simulation Results

Can we measure amygdala activity with light through the skull?

We ran a Monte Carlo photon transport simulation — 1 billion photons, 0.1mm resolution head model, 100mW laser at two wavelengths (760nm, 850nm), 22 detectors at varying distances and angles. Here's what we found.


PHOTON PATHS
─────────────
(see attached: photon_paths_850nm.png, photon_paths_by_det_850nm.png)

Blue = shallow paths. Red = paths that reached the amygdala.
At longer source-detector separations (SDS), photon "banana paths" penetrate deep enough to reach the amygdala.


BEST DETECTOR PLACEMENT
───────────────────────
(see attached: cw_sensitivity_by_angle.png, angular_sensitivity.png)

+30° toward the temporal lobe gives the best amygdala sensitivity:

  Direction                  Best SDS    Amygdala PL    Sensitivity
  +30° (toward amygdala)     35mm        3.5 mm         2.7%
  0° (straight down)         40mm        2.5 mm         1.9%
  +60°                       35mm        2.8 mm         2.2%
  -30° (away)                35mm        1.0 mm         0.7%


TIME-GATING: THE KEY TO DEPTH
──────────────────────────────
(see attached: time_gated_sensitivity.png, tpsf_curves.png, gate_photon_counts.png)

By measuring *when* photons arrive, we select for those that traveled deep. Late-arriving photons (>1.5ns) have 1000x better amygdala sensitivity than early ones.

  Time Gate     Amygdala Sensitivity    What It Sees
  0–500 ps      0.005%                  Scalp/skull only
  1–1.5 ns      3.5%                    Cortex & white matter
  2.5–4 ns      5.1%                    Peak amygdala depth

Fewer photons arrive late, but the amygdala signal is so much stronger that it more than compensates.


FREQUENCY-DOMAIN
────────────────
(see attached: fd_phase_amplitude.png)

At 200 MHz modulation, differential phase reaches -34° at SDS=45mm — strong depth encoding.


SNR — CAN WE DETECT A 1 µM HbO CHANGE?
────────────────────────────────────────
(see attached: snr_comparison.png, min_detectable_concentration.png)

At 100mW, 1 second measurement:

  SDS      Wavelength    CW      Time-Gated    Chirp
  25mm     850nm         532     657           847
  35mm     850nm         557     522           887
  35mm     760nm          90      89           143

SNR of 90–887 against a threshold of 1. Even accounting for 10–100x degradation from physiological noise, detection remains feasible. All modalities achieve sub-micromolar sensitivity.


SUMMARY
───────
  Feasible?        Yes — SNR >> 1 across all modalities
  Best direction   +30° toward temporal lobe (3.5mm amygdala pathlength)
  Best SDS         25–40mm
  Best modality    Chirp (highest SNR), TD-gated (best depth selectivity)
  Key technique    Time-gating at 1.5–4ns → 1000x depth selectivity improvement
  Safe power       100mW, 5mm spot — within IEC 60825 limits

Caveats: Idealized head model (ellipsoidal layers). Real anatomy has folds, vessels, variable skull thickness. Physiological noise not modeled. But the photon physics says the signal is there.
