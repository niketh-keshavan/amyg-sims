#!/usr/bin/env python3
"""
fNIRS Monte Carlo Analysis -- TD-Gated Focus (optimized config)
----------------------------------------------------------------
730/850nm | High-power ANSI-safe beam | Si-PMT detectors | Multi-channel MBLL

Computes:
  1. TD-gated sensitivity by gate and SDS
  2. Optimal gate selection per detector
  3. Single-detector dual-wavelength MBLL
  4. Multi-channel MBLL (best 3 detectors combined)
  5. Block-design paradigm detectability
  6. Integration time sweep
  7. ANSI Z136.1 safety check
  8. Depth specificity (contamination ratio)
  9. Short-separation regression (SSR) with sensitivity analysis
  10. TPSF moment analysis (mean ToF, variance, skewness)
  11. Combined TD-gating + SSR (per-gate regression)

Usage:
    python analyze.py --data-dir ../results
"""

import json
import argparse
import numpy as np
import sys
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Physical constants and extinction coefficients (Cope 1991 / Matcher 1995)
# ---------------------------------------------------------------------------
# Units: mM^-1 cm^-1 (divided by 10 -> mM^-1 mm^-1)
EPSILON_HBR = np.array([1.3080,  0.6918]) / 10.0  # [1/(mM*mm)] at 730, 850
EPSILON_HBO = np.array([0.1348,  1.0507]) / 10.0   # [1/(mM*mm)] at 730, 850

H_PLANCK = 6.626e-34
C_LIGHT  = 3e8
WAVELENGTHS_NM = np.array([730, 850])
WAVELENGTHS_M  = WAVELENGTHS_NM * 1e-9
WL_KEYS = ["730nm", "850nm"]

GATE_LABELS = [
    "0-500ps", "0.5-1ns", "1-1.5ns", "1.5-2ns", "2-2.5ns",
    "2.5-3ns", "3-3.5ns", "3.5-4ns", "4-5ns", "5ns+"
]

# ---------------------------------------------------------------------------
# System parameters -- optimized TD-fNIRS
# ---------------------------------------------------------------------------
LASER_POWER_W = 1.0       # 1 W average power
ANSI_MARGIN = 0.95        # keep irradiance at 95% of the strictest MPE
MEAS_TIME_S = 120.0       # 2 minutes integration

# Hamamatsu S14160-3050HS SiPM specifications
# https://www.hamamatsu.com/us/en/product/optical-sensors/mppc/mppc_m-Series/S14160-3050HS.html
DETECTOR_AREA_MM2 = 9.0   # 3mm x 3mm active area
DETECTOR_RADIUS_MM = 1.69 # Equivalent circular radius: sqrt(9/pi)

# SiPM PDE (Photon Detection Efficiency) - wavelength dependent
# Typical values for S14160-3050HS at operating voltage
DET_QE = {
    "730nm": 0.35,   # ~35% at 730nm (peak PDE region)
    "850nm": 0.25,   # ~25% at 850nm (NIR falloff)
}

DARK_COUNT_RATE = 1000    # counts/s per detector (typical at 25°C)

# ---------------------------------------------------------------------------
# Depth specificity / SSR constants
# ---------------------------------------------------------------------------
DELTA_HBO_SCALP_UM = 2.0     # typical task-evoked scalp ΔHbO (μM)
DELTA_HBO_AMYGDALA_UM = 3.0  # expected amygdala response (μM)

# ---------------------------------------------------------------------------
# Realistic noise model (Scholkmann et al. 2014, Tian & Liu 2014)
# ---------------------------------------------------------------------------
# Physiological noise floor (relative intensity fluctuation per measurement)
# Cardiac (~1 Hz), respiratory (~0.2 Hz), Mayer waves (~0.1 Hz)
# TD-fNIRS late gates have lower physio noise — gated out superficial pulsation
PHYSIO_NOISE_CW = 0.005         # 0.5% for CW / early gates
PHYSIO_NOISE_LATE_GATE = 0.002  # 0.2% for late gates (>3ns)
SOURCE_STABILITY = 0.001        # 0.1% RMS laser power fluctuation


def compute_noise(N_photons, gate_idx):
    """Realistic noise: shot + physiological + source stability in quadrature."""
    sigma_shot = 1.0 / np.sqrt(N_photons) if N_photons > 0 else float('inf')

    if gate_idx >= 6:
        k_physio = PHYSIO_NOISE_LATE_GATE
    elif gate_idx >= 3:
        frac = (gate_idx - 3) / 3.0
        k_physio = PHYSIO_NOISE_CW * (1 - frac) + PHYSIO_NOISE_LATE_GATE * frac
    else:
        k_physio = PHYSIO_NOISE_CW

    return np.sqrt(sigma_shot**2 + k_physio**2 + SOURCE_STABILITY**2)


def ansi_safe_beam_diameter_mm(power_w, wavelengths_nm, ansi_margin=0.95):
    """Compute minimum beam diameter to keep irradiance <= margin * min(MPE)."""
    mpe_vals = []
    for wl in wavelengths_nm:
        C_A = 10 ** (0.002 * (wl - 700))
        mpe_vals.append(0.2 * C_A)  # W/cm^2
    strictest_mpe = min(mpe_vals)
    max_irradiance = ansi_margin * strictest_mpe
    beam_area_cm2 = power_w / max_irradiance
    beam_radius_cm = np.sqrt(beam_area_cm2 / np.pi)
    return beam_radius_cm * 2.0 * 10.0


BEAM_DIAMETER_MM = ansi_safe_beam_diameter_mm(LASER_POWER_W, WAVELENGTHS_NM, ANSI_MARGIN)


def load_results(data_dir):
    results = {}
    for wl in tqdm(WL_KEYS, desc="Loading results", unit="wl"):
        fpath = data_dir / f"results_{wl}.json"
        if fpath.exists():
            with open(fpath) as f:
                results[wl] = json.load(f)
    return results


def gaussian_irf(fwhm_ps, bin_width_ps):
    """Generate a Gaussian IRF kernel normalized to unit area."""
    sigma = fwhm_ps / (2.0 * np.sqrt(2.0 * np.log(2.0)))  # sigma from FWHM
    # Use ±4 sigma range for the kernel
    half_width = int(np.ceil(4 * sigma / bin_width_ps))
    x = np.arange(-half_width, half_width + 1) * bin_width_ps
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / np.sum(kernel)  # normalize to unit area


def convolve_tpsf_with_irf(tpsf, fwhm_ps=100.0, bin_width_ps=10.0):
    """
    Convolve TPSF with a Gaussian IRF.
    
    Parameters:
    -----------
    tpsf : ndarray, shape (n_dets, n_bins)
        Raw TPSF histogram
    fwhm_ps : float
        IRF full-width at half-maximum in picoseconds (default: 100 ps)
    bin_width_ps : float
        TPSF bin width in picoseconds (default: 10 ps)
    
    Returns:
    --------
    ndarray
        Convolved TPSF with same shape as input
    """
    if tpsf is None or tpsf.shape[1] == 0:
        return tpsf
    
    kernel = gaussian_irf(fwhm_ps, bin_width_ps)
    
    # Convolve each detector's TPSF
    convolved = np.zeros_like(tpsf)
    for i in range(tpsf.shape[0]):
        # mode='same' keeps the output size consistent
        convolved[i, :] = np.convolve(tpsf[i, :], kernel, mode='same')
    
    return convolved


def load_tpsf(data_dir, wavelength_key, apply_irf=True, irf_fwhm_ps=100.0):
    fpath = data_dir / f"tpsf_{wavelength_key}.bin"
    if not fpath.exists():
        return None
    data = np.fromfile(fpath, dtype=np.float64)
    n_bins = 512
    n_dets = len(data) // n_bins
    tpsf = data.reshape(n_dets, n_bins)
    
    if apply_irf:
        # TPSF bin width is 10 ps (TPSF_BIN_PS from constants)
        tpsf = convolve_tpsf_with_irf(tpsf, fwhm_ps=irf_fwhm_ps, bin_width_ps=10.0)
    
    return tpsf


def photons_per_second(power_W, wavelength_m):
    E_photon = H_PLANCK * C_LIGHT / wavelength_m
    return power_W / E_photon


# ===================================================================
# 1. TD-GATED SENSITIVITY ANALYSIS
# ===================================================================
def td_sensitivity(results):
    print("\n" + "=" * 80)
    print("1. TD-GATED AMYGDALA SENSITIVITY (best gate per detector)")
    print("=" * 80)

    for wl_key in WL_KEYS:
        r = results[wl_key]
        print(f"\n  Wavelength: {r['wavelength_nm']} nm")
        print(f"  {'Det':>4s}  {'SDS':>5s}  {'Ang':>5s}  {'BestGate':>8s}  "
              f"{'Photons':>12s}  {'AmygPL':>8s}  {'Sens%':>8s}  {'TotalPL':>8s}")
        print("  " + "-" * 70)

        for det in tqdm(r["detectors"], desc=f"Processing {wl_key}", unit="det", leave=False):
            gates = det.get("time_gates", [])
            best_gate = -1
            best_sens = 0
            for g_idx, gate in enumerate(gates):
                ppl = gate.get("partial_pathlength_mm", {})
                amyg = ppl.get("amygdala", 0)
                total = sum(ppl.values())
                sens = amyg / total if total > 0 else 0
                if sens > best_sens:
                    best_sens = sens
                    best_gate = g_idx

            if best_gate >= 0 and best_gate < len(gates):
                gate = gates[best_gate]
                ppl = gate.get("partial_pathlength_mm", {})
                amyg = ppl.get("amygdala", 0)
                total = sum(ppl.values())
                angle = det.get("angle_deg", 0)
                print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {angle:+5.0f}  "
                      f"{GATE_LABELS[best_gate]:>8s}  "
                      f"{gate['detected_photons']:12d}  {amyg:8.6f}  "
                      f"{best_sens*100:8.4f}  {total:8.1f}")


# ===================================================================
# 2. GATE PHOTON BUDGET
# ===================================================================
def gate_budget(results):
    print("\n" + "=" * 80)
    print(f"2. GATE PHOTON BUDGET ({LASER_POWER_W*1e3:.0f}mW, {MEAS_TIME_S:.0f}s, Si-PMT QE)")
    print("=" * 80)

    for wl_idx, wl_key in enumerate(WL_KEYS):
        r = results[wl_key]
        qe = DET_QE[wl_key]
        N_per_sec = photons_per_second(LASER_POWER_W, WAVELENGTHS_M[wl_idx])
        num_sim = r["num_photons"]
        scale = N_per_sec / num_sim * MEAS_TIME_S * qe

        print(f"\n  {wl_key} (QE={qe*100:.0f}%, primary direction, gates with amygdala signal):")
        print(f"  {'SDS':>5s}  {'Gate':>8s}  {'DetCounts':>12s}  {'AmygPL':>10s}  "
              f"{'dOD/uM':>10s}  {'SNR_1uM':>10s}")
        print("  " + "-" * 65)

        for det in r["detectors"]:
            if abs(det.get("angle_deg", 0)) > 1:
                continue
            gates = det.get("time_gates", [])
            for g_idx, gate in enumerate(gates):
                amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                if amyg <= 0:
                    continue
                gw = gate.get("weight", 0)
                N_det = gw * scale + DARK_COUNT_RATE * MEAS_TIME_S
                delta_hbo = 0.001
                delta_hbr = -0.0003
                delta_od = abs((EPSILON_HBO[wl_idx] * delta_hbo +
                                EPSILON_HBR[wl_idx] * delta_hbr) * amyg)
                noise = compute_noise(N_det, g_idx)
                snr = delta_od / noise if noise > 0 else 0

                print(f"  {det['sds_mm']:5.0f}  {GATE_LABELS[g_idx]:>8s}  "
                      f"{N_det:12.2e}  {amyg:10.6f}  {delta_od:10.2e}  {snr:10.4f}")


# ===================================================================
# 3. SINGLE-DETECTOR DUAL-WAVELENGTH MBLL
# ===================================================================
def mbll_single(results):
    print("\n" + "=" * 80)
    print("3. SINGLE-DETECTOR MBLL (best gate per SDS)")
    print(f"   {LASER_POWER_W*1e3:.0f} mW | {MEAS_TIME_S:.0f}s | Si-PMT | {DARK_COUNT_RATE} cps dark")
    print("=" * 80)

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]

    det_data = {}
    for wl_idx, wl_key in enumerate(WL_KEYS):
        r = results[wl_key]
        num_sim = r["num_photons"]
        qe = DET_QE[wl_key]
        scale = N_per_sec[wl_idx] / num_sim

        for det in r["detectors"]:
            sds = det["sds_mm"]
            angle = det.get("angle_deg", 0)
            if abs(angle) > 1:
                continue
            gates = det.get("time_gates", [])
            for g_idx, gate in enumerate(gates):
                key = (sds, g_idx)
                if key not in det_data:
                    det_data[key] = {}
                gw = gate.get("weight", 0)
                amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                N_gate = gw * scale * MEAS_TIME_S * qe
                N_total = N_gate + DARK_COUNT_RATE * MEAS_TIME_S
                noise = compute_noise(N_total, g_idx)
                det_data[key][wl_key] = {
                    'amyg_pl': amyg, 'N_gate': N_gate, 'noise': noise,
                    'det_id': det['id']
                }

    sds_set = sorted(set(s for s, g in det_data.keys()))

    print(f"\n  {'SDS':>5s}  {'Gate':>8s}  {'N730':>10s}  {'N850':>10s}  "
          f"{'L_amyg730':>10s}  {'L_amyg850':>10s}  {'minHbO':>8s}  {'minHbR':>8s}  "
          f"{'Status':>12s}")
    print("  " + "-" * 95)

    best_configs = []

    for sds in sds_set:
        best_min_hbo = float('inf')
        best_result = None

        for g_idx in range(len(GATE_LABELS)):
            key = (sds, g_idx)
            if key not in det_data:
                continue
            d = det_data[key]
            if WL_KEYS[0] not in d or WL_KEYS[1] not in d:
                continue

            d0, d1 = d[WL_KEYS[0]], d[WL_KEYS[1]]
            L0, L1 = d0['amyg_pl'], d1['amyg_pl']
            if L0 <= 0 or L1 <= 0:
                continue

            E = np.array([
                [EPSILON_HBO[0]*L0, EPSILON_HBR[0]*L0],
                [EPSILON_HBO[1]*L1, EPSILON_HBR[1]*L1]
            ])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            E_inv = np.linalg.inv(E)
            dc = np.abs(E_inv @ np.array([d0['noise'], d1['noise']])) * 1e3

            if dc[0] < best_min_hbo:
                best_min_hbo = dc[0]
                best_result = {
                    'gate': g_idx, 'N0': d0['N_gate'], 'N1': d1['N_gate'],
                    'L0': L0, 'L1': L1,
                    'min_hbo': dc[0], 'min_hbr': dc[1],
                    'det_id': d0['det_id']
                }

        if best_result:
            r = best_result
            status = "DETECTABLE" if r['min_hbo'] < 1.0 else f"need>{r['min_hbo']:.1f}uM"
            print(f"  {sds:5.0f}  {GATE_LABELS[r['gate']]:>8s}  "
                  f"{r['N0']:10.2e}  {r['N1']:10.2e}  "
                  f"{r['L0']:10.6f}  {r['L1']:10.6f}  "
                  f"{r['min_hbo']:8.3f}  {r['min_hbr']:8.3f}  {status:>12s}")
            best_configs.append((sds, best_result))

    return best_configs


# ===================================================================
# 4. MULTI-CHANNEL MBLL (BEST 3 DETECTORS)
# ===================================================================
def mbll_multi_channel(results):
    print("\n" + "=" * 80)
    print("4. MULTI-CHANNEL MBLL (best 3 detectors, all usable gates)")
    print(f"   {LASER_POWER_W*1e3:.0f} mW | {MEAS_TIME_S:.0f}s | Si-PMT | {DARK_COUNT_RATE} cps dark")
    print("=" * 80)

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]

    # Gather per-detector, per-gate, per-wavelength data
    all_det_info = []  # list of (det_id, sds, angle, gate_measurements)
    n_dets = len(results[WL_KEYS[0]]["detectors"])

    for det_idx in tqdm(range(n_dets), desc="Processing detectors", unit="det", leave=False):
        det0 = results[WL_KEYS[0]]["detectors"][det_idx]
        det1 = results[WL_KEYS[1]]["detectors"][det_idx]
        sds = det0["sds_mm"]
        angle = det0.get("angle_deg", 0)

        gate_data = []
        n_gates = min(len(det0.get("time_gates", [])),
                      len(det1.get("time_gates", [])))
        for g_idx in range(n_gates):
            g0 = det0["time_gates"][g_idx]
            g1 = det1["time_gates"][g_idx]
            L0 = g0.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L1 = g1.get("partial_pathlength_mm", {}).get("amygdala", 0)
            # Minimum amygdala pathlength threshold: 0.001 mm (based on 10B photon results)
            if L0 <= 0.001 or L1 <= 0.001:
                continue

            gw0 = g0.get("weight", 0)
            gw1 = g1.get("weight", 0)
            qe0, qe1 = DET_QE[WL_KEYS[0]], DET_QE[WL_KEYS[1]]
            scale0 = N_per_sec[0] / results[WL_KEYS[0]]["num_photons"]
            scale1 = N_per_sec[1] / results[WL_KEYS[1]]["num_photons"]
            N0 = gw0 * scale0 * MEAS_TIME_S * qe0 + DARK_COUNT_RATE * MEAS_TIME_S
            N1 = gw1 * scale1 * MEAS_TIME_S * qe1 + DARK_COUNT_RATE * MEAS_TIME_S

            scalp_pl_0 = g0.get("partial_pathlength_mm", {}).get("scalp", 1.0)
            scalp_pl_1 = g1.get("partial_pathlength_mm", {}).get("scalp", 1.0)
            gate_data.append({
                'gate': g_idx, 'L0': L0, 'L1': L1,
                'N0': N0, 'N1': N1,
                'sigma0': compute_noise(N0, g_idx),
                'sigma1': compute_noise(N1, g_idx),
                'scalp_pl_0': scalp_pl_0, 'scalp_pl_1': scalp_pl_1,
            })

        if gate_data:
            metric = sum(
                (gd['L0'] / max(gd['scalp_pl_0'], 0.001)) * np.sqrt(gd['N0']) +
                (gd['L1'] / max(gd['scalp_pl_1'], 0.001)) * np.sqrt(gd['N1'])
                for gd in gate_data
            )
            all_det_info.append({
                'det_id': det_idx, 'sds': sds, 'angle': angle,
                'gates': gate_data, 'metric': metric
            })

    # Sort by metric, take top 3
    all_det_info.sort(key=lambda x: x['metric'], reverse=True)
    top3 = all_det_info[:3]

    print(f"\n  Selected detectors (by amygdala sensitivity metric):")
    for rank, info in enumerate(top3):
        n_valid = len(info['gates'])
        print(f"    #{rank+1}: Det {info['det_id']} SDS={info['sds']:.0f}mm "
              f"angle={info['angle']:+.0f}deg  ({n_valid} valid gates, metric={info['metric']:.2f})")

    # Build overdetermined weighted least-squares system
    # Each measurement: dOD_{d,g,w} = (e_HbO * dHbO + e_HbR * dHbR) * L_amyg_{d,g,w}
    # Noise: sigma = 1/sqrt(N)
    rows_A = []
    sigma_vec = []

    for info in top3:
        for gd in info['gates']:
            # Wavelength 0 (730nm) measurement
            rows_A.append([EPSILON_HBO[0] * gd['L0'], EPSILON_HBR[0] * gd['L0']])
            sigma_vec.append(gd['sigma0'])
            # Wavelength 1 (850nm) measurement
            rows_A.append([EPSILON_HBO[1] * gd['L1'], EPSILON_HBR[1] * gd['L1']])
            sigma_vec.append(gd['sigma1'])

    A = np.array(rows_A)
    sigma = np.array(sigma_vec)
    n_measurements = len(sigma)

    print(f"\n  System: {n_measurements} measurements x 2 unknowns (HbO, HbR)")

    # Weighted least squares: W = diag(1/sigma^2)
    W = np.diag(1.0 / sigma**2)
    AW = A.T @ W
    AWA = AW @ A
    if abs(np.linalg.det(AWA)) < 1e-30:
        print("  ERROR: singular system")
        return None

    cov = np.linalg.inv(AWA)
    min_hbo = np.sqrt(cov[0, 0]) * 1e3  # convert to uM
    min_hbr = np.sqrt(cov[1, 1]) * 1e3

    print(f"\n  MULTI-CHANNEL RESULT ({MEAS_TIME_S:.0f}s integration):")
    print(f"    Min detectable HbO: {min_hbo:.4f} uM")
    print(f"    Min detectable HbR: {min_hbr:.4f} uM")
    print(f"    Correlation(HbO,HbR): {cov[0,1]/np.sqrt(cov[0,0]*cov[1,1]):.3f}")

    # Compare with single-detector best
    print(f"\n  vs. Single-best-detector comparison:")
    if all_det_info:
        best_single = all_det_info[0]
        # Single-detector MBLL at best gate
        best_gate = max(best_single['gates'],
                        key=lambda g: min(g['L0']*np.sqrt(g['N0']),
                                          g['L1']*np.sqrt(g['N1'])))
        E = np.array([
            [EPSILON_HBO[0]*best_gate['L0'], EPSILON_HBR[0]*best_gate['L0']],
            [EPSILON_HBO[1]*best_gate['L1'], EPSILON_HBR[1]*best_gate['L1']]
        ])
        if abs(np.linalg.det(E)) > 1e-20:
            E_inv = np.linalg.inv(E)
            dc_single = np.abs(E_inv @ np.array([best_gate['sigma0'],
                                                  best_gate['sigma1']])) * 1e3
            print(f"    Single det {best_single['det_id']} best gate: "
                  f"HbO={dc_single[0]:.4f} uM, HbR={dc_single[1]:.4f} uM")
            print(f"    Multi-channel improvement: "
                  f"HbO {dc_single[0]/min_hbo:.1f}x, HbR {dc_single[1]/min_hbr:.1f}x")

    # Scale to different integration times
    print(f"\n  Min detectable vs integration time:")
    print(f"  {'Time':>8s}  {'HbO [uM]':>10s}  {'HbR [uM]':>10s}  {'HbO status':>12s}  {'HbR status':>12s}")
    print("  " + "-" * 55)
    for t in tqdm([1, 5, 10, 15, 30, 60, 120, 300], desc="Integration times", unit="time", leave=False):
        hbo_t = min_hbo * np.sqrt(MEAS_TIME_S / t)
        hbr_t = min_hbr * np.sqrt(MEAS_TIME_S / t)
        label = f"{t}s" if t < 60 else f"{t//60}m"
        hbo_s = "OK" if hbo_t < 1.0 else "marginal" if hbo_t < 2.0 else "no"
        hbr_s = "OK" if hbr_t < 1.0 else "marginal" if hbr_t < 2.0 else "no"
        print(f"  {label:>8s}  {hbo_t:10.4f}  {hbr_t:10.4f}  {hbo_s:>12s}  {hbr_s:>12s}")

    return {'min_hbo': min_hbo, 'min_hbr': min_hbr, 'cov': cov, 'top3': top3}


# ===================================================================
# 5. BLOCK DESIGN FEASIBILITY
# ===================================================================
def block_design(multi_result, best_configs):
    print("\n" + "=" * 80)
    print("5. BLOCK-DESIGN PARADIGM (15s stim, 15s rest, 20 trials)")
    print("   Expected amygdala: 2-5 uM HbO, -0.5 to -1.5 uM HbR")
    print("=" * 80)

    n_trials = 20
    trial_dur = 15.0
    trial_gain = np.sqrt(n_trials)

    if multi_result:
        mc_hbo = multi_result['min_hbo']
        mc_hbr = multi_result['min_hbr']
        trial_min_hbo = mc_hbo * np.sqrt(MEAS_TIME_S / trial_dur)
        trial_min_hbr = mc_hbr * np.sqrt(MEAS_TIME_S / trial_dur)
        avg_min_hbo = trial_min_hbo / trial_gain
        avg_min_hbr = trial_min_hbr / trial_gain

        print(f"\n  MULTI-CHANNEL (3 best detectors):")
        print(f"    Single-trial noise floor: HbO={trial_min_hbo:.3f} uM, HbR={trial_min_hbr:.3f} uM")
        print(f"    After {n_trials}-trial avg:    HbO={avg_min_hbo:.4f} uM, HbR={avg_min_hbr:.4f} uM")

        print(f"\n    SNR for expected amygdala responses:")
        for hbo_mag, hbr_mag in [(2.0, -0.5), (3.0, -1.0), (5.0, -1.5)]:
            snr_hbo = hbo_mag / avg_min_hbo if avg_min_hbo > 0 else 0
            snr_hbr = abs(hbr_mag) / avg_min_hbr if avg_min_hbr > 0 else 0
            print(f"      HbO={hbo_mag:+.0f}uM -> SNR={snr_hbo:.1f}  |  "
                  f"HbR={hbr_mag:+.1f}uM -> SNR={snr_hbr:.1f}")

    print(f"\n  SINGLE-DETECTOR (per SDS, trial-averaged):")
    print(f"  {'SDS':>5s}  {'Gate':>8s}  {'eff_min':>8s}  "
          f"{'1uM':>7s}  {'2uM':>7s}  {'3uM':>7s}  {'5uM':>7s}  {'verdict':>10s}")
    print("  " + "-" * 68)

    for sds, cfg in tqdm(best_configs, desc="Computing SNR", unit="SDS", leave=False):
        trial_min = cfg['min_hbo'] * np.sqrt(MEAS_TIME_S / trial_dur)
        avg_min = trial_min / trial_gain

        snrs = [h / avg_min if avg_min > 0 else 0 for h in [1, 2, 3, 5]]
        verdict = "YES" if snrs[1] >= 1.0 else ("marginal" if snrs[2] >= 1.0 else "no")

        print(f"  {sds:5.0f}  {GATE_LABELS[cfg['gate']]:>8s}  {avg_min:8.3f}  "
              f"{''.join(f'{s:7.2f}' for s in snrs)}  {verdict:>10s}")


# ===================================================================
# 6. INTEGRATION TIME SWEEP
# ===================================================================
def time_sweep(results):
    print("\n" + "=" * 80)
    print("6. MIN DETECTABLE HbO [uM] vs INTEGRATION TIME (single detector)")
    print("=" * 80)

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]
    times = [1, 5, 10, 15, 30, 60, 120]

    print(f"\n  {'SDS':>5s}", end="")
    for t in times:
        label = f"{t}s" if t < 60 else f"{t//60}m"
        print(f"  {label:>6s}", end="")
    print()
    print("  " + "-" * (7 + 8 * len(times)))

    r0, r1 = results[WL_KEYS[0]], results[WL_KEYS[1]]

    for det0, det1 in zip(r0["detectors"], r1["detectors"]):
        if abs(det0.get("angle_deg", 0)) > 1:
            continue
        sds = det0["sds_mm"]

        best_min = float('inf')

        for g_idx in range(min(len(det0.get("time_gates", [])),
                               len(det1.get("time_gates", [])))):
            g0 = det0["time_gates"][g_idx]
            g1 = det1["time_gates"][g_idx]
            L0 = g0.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L1 = g1.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if L0 <= 0 or L1 <= 0:
                continue

            gw0, gw1 = g0.get("weight", 0), g1.get("weight", 0)
            scale0 = N_per_sec[0] / r0["num_photons"]
            scale1 = N_per_sec[1] / r1["num_photons"]
            N0 = gw0 * scale0 * 1.0 * DET_QE[WL_KEYS[0]] + DARK_COUNT_RATE
            N1 = gw1 * scale1 * 1.0 * DET_QE[WL_KEYS[1]] + DARK_COUNT_RATE
            n0 = compute_noise(N0, g_idx)
            n1 = compute_noise(N1, g_idx)

            E = np.array([
                [EPSILON_HBO[0]*L0, EPSILON_HBR[0]*L0],
                [EPSILON_HBO[1]*L1, EPSILON_HBR[1]*L1]
            ])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            E_inv = np.linalg.inv(E)
            dc = np.abs(E_inv @ np.array([n0, n1])) * 1e3

            if dc[0] < best_min:
                best_min = dc[0]

        if best_min > 1e5:
            continue

        line = f"  {sds:5.0f}"
        for t in times:
            min_hbo = best_min / np.sqrt(t)
            line += f"  {min_hbo:6.2f}"
        print(line)

    print(f"\n  Values = min detectable dHbO [uM] (dual-wavelength MBLL)")
    print(f"  Target: <1 uM single-trial, <2 uM block-averaged")


# ===================================================================
# 7. SAFETY CHECK
# ===================================================================
def safety_check():
    print("\n" + "=" * 80)
    print("7. ANSI Z136.1 LASER SAFETY")
    print("=" * 80)

    beam_radius_cm = (BEAM_DIAMETER_MM / 2.0) / 10.0
    beam_area = np.pi * beam_radius_cm ** 2
    irradiance = LASER_POWER_W / beam_area

    print(f"  Beam: {BEAM_DIAMETER_MM:.0f}mm diameter, area={beam_area:.3f} cm^2")
    print(f"  Power: {LASER_POWER_W*1e3:.0f} mW -> Irradiance: {irradiance:.4f} W/cm^2")
    print(f"  ANSI margin target: {ANSI_MARGIN*100:.0f}% of strictest wavelength MPE")

    for wl in WAVELENGTHS_NM:
        C_A = 10 ** (0.002 * (wl - 700))
        mpe = 0.2 * C_A
        ratio = irradiance / mpe
        print(f"  {wl} nm: MPE={mpe:.4f} W/cm^2, ratio={ratio:.4f} "
              f"({'SAFE' if ratio < 1 else 'EXCEEDS MPE'})")


# ===================================================================
# 8. DEPTH SPECIFICITY (contamination ratio)
# ===================================================================
def depth_specificity(results):
    print("\n" + "=" * 80)
    print("8. DEPTH SPECIFICITY (amygdala vs scalp contamination)")
    print(f"   Assumed: scalp dHbO={DELTA_HBO_SCALP_UM} uM, amygdala dHbO={DELTA_HBO_AMYGDALA_UM} uM")
    print("=" * 80)

    r = results[WL_KEYS[0]]
    wl_idx = 0

    print(f"\n  {'Det':>4s}  {'SDS':>5s}  {'Gate':>8s}  {'AmygPL':>8s}  {'ScalpPL':>8s}  "
          f"{'ContamR':>8s}  {'Specif%':>8s}  {'CW_ContR':>8s}  {'TD_gain':>8s}")
    print("  " + "-" * 82)

    for det in r["detectors"]:
        cw_ppl = det.get("partial_pathlength_mm", {})
        cw_amyg = cw_ppl.get("amygdala", 0)
        cw_scalp = cw_ppl.get("scalp", 1)

        if cw_amyg <= 0:
            continue

        cw_dOD_scalp = EPSILON_HBO[wl_idx] * DELTA_HBO_SCALP_UM * 1e-3 * cw_scalp
        cw_dOD_amyg = EPSILON_HBO[wl_idx] * DELTA_HBO_AMYGDALA_UM * 1e-3 * cw_amyg
        cw_contam = cw_dOD_scalp / cw_dOD_amyg if cw_dOD_amyg > 0 else float('inf')

        gates = det.get("time_gates", [])
        best_gate = None
        best_spec = 0
        for g_idx, gate in enumerate(gates):
            ppl = gate.get("partial_pathlength_mm", {})
            amyg = ppl.get("amygdala", 0)
            scalp = ppl.get("scalp", 1)
            if amyg <= 0:
                continue
            dOD_s = EPSILON_HBO[wl_idx] * DELTA_HBO_SCALP_UM * 1e-3 * scalp
            dOD_a = EPSILON_HBO[wl_idx] * DELTA_HBO_AMYGDALA_UM * 1e-3 * amyg
            spec = dOD_a / (dOD_a + dOD_s) if (dOD_a + dOD_s) > 0 else 0
            if spec > best_spec:
                best_spec = spec
                best_gate = (g_idx, amyg, scalp, dOD_s / dOD_a if dOD_a > 0 else float('inf'))

        if best_gate:
            g_idx, amyg, scalp, contam = best_gate
            td_gain = cw_contam / contam if contam > 0 else float('inf')
            print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {GATE_LABELS[g_idx]:>8s}  "
                  f"{amyg:8.4f}  {scalp:8.1f}  {contam:7.1f}x  "
                  f"{best_spec*100:7.2f}%  {cw_contam:7.0f}x  {td_gain:7.1f}x")

    print(f"\n  ContamR = scalp_signal / amygdala_signal (lower = better)")
    print(f"  Specif% = amygdala / (amygdala + scalp) signal fraction")
    print(f"  TD_gain = CW_contam / TD_contam (TD gating improvement factor)")


# ===================================================================
# 9. SHORT-SEPARATION REGRESSION (SSR) WITH SENSITIVITY ANALYSIS
# ===================================================================
SSR_SDS_THRESHOLD_MM = 15.0  # detectors with SDS < this are SSR references

def ssr_analysis(results):
    print("\n" + "=" * 80)
    print("9. SHORT-SEPARATION REGRESSION (SSR) -- DEPTH DISCRIMINATION")
    print(f"   SSR reference: SDS < {SSR_SDS_THRESHOLD_MM:.0f} mm")
    print("=" * 80)

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]

    # Find short-SDS reference detectors in both wavelengths
    short_dets_by_wl = {}
    long_dets_by_wl = {}
    for wl_idx, wl_key in enumerate(WL_KEYS):
        r = results[wl_key]
        short_dets_by_wl[wl_key] = [d for d in r["detectors"]
                                     if d["sds_mm"] < SSR_SDS_THRESHOLD_MM]
        long_dets_by_wl[wl_key] = [d for d in r["detectors"]
                                    if d["sds_mm"] >= SSR_SDS_THRESHOLD_MM]

    n_short = len(short_dets_by_wl[WL_KEYS[0]])
    if n_short == 0:
        print(f"\n  WARNING: No short-SDS detectors found (SDS < {SSR_SDS_THRESHOLD_MM} mm).")
        print(f"  SSR analysis requires short-separation reference channels.")
        print(f"  Add detectors at SDS = 5-10 mm to enable SSR.")
        return None

    # --- CW-level SSR reference characterization ---
    print(f"\n  SSR reference channels: {n_short} detectors")
    for wl_key in WL_KEYS:
        sdets = short_dets_by_wl[wl_key]
        scalps = [d.get("partial_pathlength_mm", {}).get("scalp", 0) for d in sdets]
        amygs = [d.get("partial_pathlength_mm", {}).get("amygdala", 0) for d in sdets]
        sds_vals = [d["sds_mm"] for d in sdets]
        print(f"    {wl_key}: SDS={np.mean(sds_vals):.1f}mm, "
              f"scalp_PL={np.mean(scalps):.2f}mm, amyg_PL={np.mean(amygs):.6f}mm")

    # --- CW SSR regression ---
    print(f"\n  9a. CW SSR REGRESSION (broadband, no gating):")
    print(f"  {'Det':>4s}  {'SDS':>5s}  {'L_amyg':>8s}  {'L_scalp':>8s}  "
          f"{'beta':>6s}  {'L_corr':>10s}  {'Pres%':>7s}")
    print("  " + "-" * 58)

    r0 = results[WL_KEYS[0]]
    ssr_ref_scalp_cw = np.mean([d.get("partial_pathlength_mm", {}).get("scalp", 0)
                                 for d in short_dets_by_wl[WL_KEYS[0]]])
    ssr_ref_amyg_cw = np.mean([d.get("partial_pathlength_mm", {}).get("amygdala", 0)
                                for d in short_dets_by_wl[WL_KEYS[0]]])

    for det in long_dets_by_wl[WL_KEYS[0]]:
        ppl = det.get("partial_pathlength_mm", {})
        amyg = ppl.get("amygdala", 0)
        scalp = ppl.get("scalp", 0)
        if amyg <= 0:
            continue
        beta = scalp / ssr_ref_scalp_cw if ssr_ref_scalp_cw > 0 else 0
        amyg_corr = amyg - beta * ssr_ref_amyg_cw
        preserved = amyg_corr / amyg * 100 if amyg > 0 else 0
        print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {amyg:8.6f}  {scalp:8.2f}  "
              f"{beta:6.2f}  {amyg_corr:10.6f}  {preserved:6.1f}%")

    # --- Per-gate SSR: compute min detectable ΔHbO with SSR correction ---
    print(f"\n  9b. SSR-CORRECTED SENSITIVITY (per gate, dual-wavelength MBLL):")
    print(f"  {'SDS':>5s}  {'Gate':>8s}  {'minHbO_raw':>11s}  {'minHbO_SSR':>11s}  "
          f"{'Improv':>7s}  {'Status':>10s}")
    print("  " + "-" * 62)

    ssr_results = []
    best_ssr_hbo = float('inf')
    best_ssr_config = None

    # Build SSR reference signals per gate
    for wl_idx, wl_key in enumerate(WL_KEYS):
        r = results[wl_key]
        num_sim = r["num_photons"]
        qe = DET_QE[wl_key]
        scale = N_per_sec[wl_idx] / num_sim

        sdets = short_dets_by_wl[wl_key]
        if not sdets:
            continue

        # Average SSR reference per gate
        n_gates_ref = len(sdets[0].get("time_gates", []))
        ssr_ref_scalp_gate = np.zeros(n_gates_ref)
        ssr_ref_amyg_gate = np.zeros(n_gates_ref)
        for sd in sdets:
            for g_idx, gate in enumerate(sd.get("time_gates", [])):
                ppl = gate.get("partial_pathlength_mm", {})
                ssr_ref_scalp_gate[g_idx] += ppl.get("scalp", 0)
                ssr_ref_amyg_gate[g_idx] += ppl.get("amygdala", 0)
        ssr_ref_scalp_gate /= len(sdets)
        ssr_ref_amyg_gate /= len(sdets)

        if wl_idx == 0:
            ssr_ref_scalp_730 = ssr_ref_scalp_gate
            ssr_ref_amyg_730 = ssr_ref_amyg_gate
        else:
            ssr_ref_scalp_850 = ssr_ref_scalp_gate
            ssr_ref_amyg_850 = ssr_ref_amyg_gate

    # For each long-SDS detector, compute raw and SSR-corrected min ΔHbO
    n_long = len(long_dets_by_wl[WL_KEYS[0]])
    for d_idx in range(n_long):
        det0 = long_dets_by_wl[WL_KEYS[0]][d_idx]
        det1 = long_dets_by_wl[WL_KEYS[1]][d_idx] if d_idx < len(long_dets_by_wl[WL_KEYS[1]]) else None
        if det1 is None:
            continue

        sds = det0["sds_mm"]
        gates0 = det0.get("time_gates", [])
        gates1 = det1.get("time_gates", [])
        n_gates = min(len(gates0), len(gates1))

        for g_idx in range(n_gates):
            g0, g1 = gates0[g_idx], gates1[g_idx]
            L0 = g0.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L1 = g1.get("partial_pathlength_mm", {}).get("amygdala", 0)
            scalp0 = g0.get("partial_pathlength_mm", {}).get("scalp", 0)
            scalp1 = g1.get("partial_pathlength_mm", {}).get("scalp", 0)
            if L0 <= 0 or L1 <= 0:
                continue

            # Photon counts
            gw0, gw1 = g0.get("weight", 0), g1.get("weight", 0)
            scale0 = N_per_sec[0] / results[WL_KEYS[0]]["num_photons"]
            scale1 = N_per_sec[1] / results[WL_KEYS[1]]["num_photons"]
            N0 = gw0 * scale0 * MEAS_TIME_S * DET_QE[WL_KEYS[0]] + DARK_COUNT_RATE * MEAS_TIME_S
            N1 = gw1 * scale1 * MEAS_TIME_S * DET_QE[WL_KEYS[1]] + DARK_COUNT_RATE * MEAS_TIME_S
            noise0 = compute_noise(N0, g_idx)
            noise1 = compute_noise(N1, g_idx)

            # Raw MBLL (no SSR)
            E_raw = np.array([
                [EPSILON_HBO[0] * L0, EPSILON_HBR[0] * L0],
                [EPSILON_HBO[1] * L1, EPSILON_HBR[1] * L1]
            ])
            if abs(np.linalg.det(E_raw)) < 1e-20:
                continue
            E_inv_raw = np.linalg.inv(E_raw)
            dc_raw = np.abs(E_inv_raw @ np.array([noise0, noise1])) * 1e3
            min_hbo_raw = dc_raw[0]

            # SSR correction: subtract beta * reference
            ref_scalp_0 = ssr_ref_scalp_730[g_idx] if g_idx < len(ssr_ref_scalp_730) else 0
            ref_scalp_1 = ssr_ref_scalp_850[g_idx] if g_idx < len(ssr_ref_scalp_850) else 0
            ref_amyg_0 = ssr_ref_amyg_730[g_idx] if g_idx < len(ssr_ref_amyg_730) else 0
            ref_amyg_1 = ssr_ref_amyg_850[g_idx] if g_idx < len(ssr_ref_amyg_850) else 0

            beta0 = scalp0 / ref_scalp_0 if ref_scalp_0 > 0 else 0
            beta1 = scalp1 / ref_scalp_1 if ref_scalp_1 > 0 else 0

            L0_corr = max(L0 - beta0 * ref_amyg_0, 0)
            L1_corr = max(L1 - beta1 * ref_amyg_1, 0)

            if L0_corr <= 0 or L1_corr <= 0:
                continue

            # SSR adds noise from reference channel (in quadrature)
            noise0_ssr = np.sqrt(noise0**2 + (beta0 * noise0)**2)
            noise1_ssr = np.sqrt(noise1**2 + (beta1 * noise1)**2)

            E_ssr = np.array([
                [EPSILON_HBO[0] * L0_corr, EPSILON_HBR[0] * L0_corr],
                [EPSILON_HBO[1] * L1_corr, EPSILON_HBR[1] * L1_corr]
            ])
            if abs(np.linalg.det(E_ssr)) < 1e-20:
                continue
            E_inv_ssr = np.linalg.inv(E_ssr)
            dc_ssr = np.abs(E_inv_ssr @ np.array([noise0_ssr, noise1_ssr])) * 1e3
            min_hbo_ssr = dc_ssr[0]

            improvement = min_hbo_raw / min_hbo_ssr if min_hbo_ssr > 0 else 0
            status = "BETTER" if improvement > 1.0 else "WORSE"
            if min_hbo_ssr < 5.0:
                status = "GOOD" if min_hbo_ssr < 2.0 else "OK"

            print(f"  {sds:5.0f}  {GATE_LABELS[g_idx]:>8s}  "
                  f"{min_hbo_raw:11.3f}  {min_hbo_ssr:11.3f}  "
                  f"{improvement:6.2f}x  {status:>10s}")

            ssr_results.append({
                'sds': sds, 'gate': g_idx, 'det_id': det0['id'],
                'min_hbo_raw': min_hbo_raw, 'min_hbo_ssr': min_hbo_ssr,
                'improvement': improvement
            })

            if min_hbo_ssr < best_ssr_hbo:
                best_ssr_hbo = min_hbo_ssr
                best_ssr_config = ssr_results[-1]

    if best_ssr_config:
        print(f"\n  BEST SSR RESULT:")
        print(f"    SDS={best_ssr_config['sds']:.0f}mm, "
              f"gate={GATE_LABELS[best_ssr_config['gate']]}")
        print(f"    Min detectable HbO: {best_ssr_config['min_hbo_ssr']:.3f} uM "
              f"(was {best_ssr_config['min_hbo_raw']:.3f} uM without SSR)")
        print(f"    SSR improvement: {best_ssr_config['improvement']:.2f}x")
    else:
        print(f"\n  No valid SSR results (need both short and long SDS detectors)")

    return ssr_results


# ===================================================================
# 10. TPSF MOMENT ANALYSIS (depth discrimination via temporal moments)
# ===================================================================
TPSF_BIN_PS = 10.0   # picoseconds per bin
TPSF_N_BINS = 512
C_TISSUE_MM_PS = 0.2148  # speed of light in tissue ~0.215 mm/ps (n~1.4)

# Realistic noise model for TPSF moments (Step 6)
IRF_FWHM_PS = 80.0           # 80 ps IRF FWHM (typical for fast SPAD/SiPM systems)
PHYSIO_NOISE_MEANTIME_PS = 5.0  # 5 ps RMS from cardiac/respiratory pulsation
SCALP_MOMENT_SHIFT_PS = 50.0    # Mean-time shift from scalp hemodynamics (50 ps >> amygdala signal)

def tpsf_moment_analysis(results, tpsf_730, tpsf_850):
    print("\n" + "=" * 80)
    print("10. TPSF MOMENT ANALYSIS (depth discrimination via temporal statistics)")
    print(f"    Bins: {TPSF_N_BINS}, bin width: {TPSF_BIN_PS} ps, "
          f"range: 0-{TPSF_N_BINS * TPSF_BIN_PS / 1000:.1f} ns")
    print("=" * 80)

    if tpsf_730 is None and tpsf_850 is None:
        print("  WARNING: No TPSF data available. Skipping moment analysis.")
        return None

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]
    t_bins = (np.arange(TPSF_N_BINS) + 0.5) * TPSF_BIN_PS  # bin centers in ps

    moment_results = []

    for wl_idx, (wl_key, tpsf_raw) in enumerate(zip(WL_KEYS, [tpsf_730, tpsf_850])):
        if tpsf_raw is None:
            print(f"\n  {wl_key}: No TPSF data, skipping.")
            continue

        # Step 6: Convolve with IRF for realistic TPSF
        tpsf = convolve_tpsf_with_irf(tpsf_raw, fwhm_ps=IRF_FWHM_PS, bin_width_ps=TPSF_BIN_PS)

        r = results[wl_key]
        n_dets = min(tpsf.shape[0], len(r["detectors"]))
        num_sim = r["num_photons"]
        qe = DET_QE[wl_key]
        scale = N_per_sec[wl_idx] / num_sim * MEAS_TIME_S * qe

        print(f"\n  {wl_key} ({n_dets} detectors):")
        print(f"  {'Det':>4s}  {'SDS':>5s}  {'<t> ps':>8s}  {'var ps^2':>10s}  "
              f"{'skew':>8s}  {'N_phot':>10s}  {'AmygPL':>8s}  "
              f"{'d<t>/dua':>9s}  {'minHbO_t':>10s}")
        print("  " + "-" * 85)

        for d_idx in range(n_dets):
            det = r["detectors"][d_idx]
            sds = det["sds_mm"]
            w = tpsf[d_idx, :]  # photon weights per bin
            total_w = np.sum(w)

            if total_w < 1e-30:
                continue

            # Moment 1: mean time-of-flight
            mean_t = np.sum(t_bins * w) / total_w

            # Moment 2: variance
            var_t = np.sum(t_bins**2 * w) / total_w - mean_t**2

            # Moment 3: skewness
            if var_t > 0:
                std_t = np.sqrt(var_t)
                skew_t = (np.sum((t_bins - mean_t)**3 * w) / total_w) / std_t**3
            else:
                skew_t = 0.0

            # Detected photon count (scaled to real system)
            N_det = total_w * scale

            # Amygdala pathlength from CW data
            amyg_pl = det.get("partial_pathlength_mm", {}).get("amygdala", 0)

            # Sensitivity of <t> to amygdala absorption change
            # d<t>/d(mu_a) ~ -<t * L_amyg> for photons reaching this detector
            # Approximate: late-arriving photons traverse more amygdala
            # Use partial pathlength from time-gated data to weight
            gates = det.get("time_gates", [])
            dt_dua = 0.0
            if amyg_pl > 0 and len(gates) > 0:
                # Estimate from gate-resolved amygdala pathlengths
                # d<t>/d(mu_a) = -sum_g(t_g * L_amyg_g * w_g) / sum_g(w_g)
                gate_edges_ps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 5120]
                numerator = 0.0
                denominator = 0.0
                for g_idx, gate in enumerate(gates):
                    g_amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                    gw = gate.get("weight", 0)
                    if g_idx < len(gate_edges_ps) - 1:
                        t_gate = (gate_edges_ps[g_idx] + gate_edges_ps[g_idx + 1]) / 2.0
                    else:
                        t_gate = gate_edges_ps[-1]
                    numerator += t_gate * g_amyg * gw
                    denominator += gw
                if denominator > 0:
                    dt_dua = -numerator / denominator  # ps * mm

            # Min detectable ΔHbO from moment sensitivity
            # Step 6: Realistic noise model includes:
            # 1. Shot noise: sigma_shot = sqrt(var_t / N_det)
            # 2. IRF jitter: sigma_irf ~ IRF_FWHM / sqrt(N_det) (temporal precision limit)
            # 3. Physiological noise: sigma_physio = PHYSIO_NOISE_MEANTIME_PS
            min_hbo_moment = float('inf')
            if N_det > 0 and abs(dt_dua) > 0:
                sigma_shot = np.sqrt(var_t / N_det) if var_t > 0 else 0
                # IRF jitter: precision improves with sqrt(N) but IRF width sets floor
                sigma_irf = IRF_FWHM_PS / (2.355 * np.sqrt(N_det))  # 2.355 = sigma to FWHM
                sigma_physio = PHYSIO_NOISE_MEANTIME_PS
                # Combined noise in quadrature
                sigma_mean_t = np.sqrt(sigma_shot**2 + sigma_irf**2 + sigma_physio**2)
                
                # Step 6: Scalp contamination - mean-time shifts from scalp hemodynamics
                # Scalp signal is ~70,000x stronger than amygdala in CW
                # For moments: scalp shift ~50 ps >> amygdala-induced shift
                # Effective SNR degradation from scalp contamination
                scalp_contamination_factor = 1.0 + (SCALP_MOMENT_SHIFT_PS / abs(dt_dua))**2
                sigma_mean_t *= np.sqrt(scalp_contamination_factor)
                
                # dt_dua has units ps*mm; converting ΔHbO in mM to Δμ_a:
                epsilon_hbo = EPSILON_HBO[wl_idx]
                if epsilon_hbo > 0 and not np.isinf(sigma_mean_t):
                    min_hbo_moment = sigma_mean_t / (abs(dt_dua) * epsilon_hbo) * 1e3  # uM

            print(f"  {d_idx:4d}  {sds:5.0f}  {mean_t:8.1f}  {var_t:10.1f}  "
                  f"{skew_t:8.3f}  {N_det:10.2e}  {amyg_pl:8.6f}  "
                  f"{dt_dua:9.2f}  {min_hbo_moment:10.3f}")

            moment_results.append({
                'det_id': d_idx, 'sds': sds, 'wl': wl_key,
                'mean_t': mean_t, 'var_t': var_t, 'skew_t': skew_t,
                'N_det': N_det, 'amyg_pl': amyg_pl,
                'dt_dua': dt_dua, 'min_hbo_moment': min_hbo_moment
            })

    # Variance-based contrast metric
    print(f"\n  VARIANCE-BASED DEPTH CONTRAST:")
    print(f"  {'Det':>4s}  {'SDS':>5s}  {'WL':>5s}  {'var_t':>10s}  "
          f"{'dVar/dua':>10s}  {'sigma_var':>10s}  {'minHbO_var':>10s}")
    print("  " + "-" * 70)

    best_moment_hbo = float('inf')
    best_moment_config = None

    for mr in moment_results:
        var_t = mr['var_t']
        N_det = mr['N_det']
        dt_dua = mr['dt_dua']
        wl_idx = 0 if mr['wl'] == WL_KEYS[0] else 1

        if N_det <= 0 or var_t <= 0:
            continue

        # Sensitivity of variance to absorption change
        # dVar/d(mu_a) ~ -2 * <t> * d<t>/d(mu_a) (first-order approximation)
        mean_t = mr['mean_t']
        dvar_dua = -2.0 * mean_t * dt_dua  # ps^2 * mm

        # Step 6: Realistic noise on variance
        # Shot noise + IRF broadening + physiological variance fluctuations
        sigma_var_shot = var_t * np.sqrt(2.0 / N_det)
        # IRF adds variance ~ (IRF width)^2, reduces precision
        sigma_irf_var = (IRF_FWHM_PS / 2.355)**2 / np.sqrt(N_det)
        # Physiological variance noise (cardiac/respiration modulates variance)
        sigma_physio_var = var_t * 0.05  # 5% variance fluctuation from physiology
        sigma_var = np.sqrt(sigma_var_shot**2 + sigma_irf_var**2 + sigma_physio_var**2)
        
        # Step 6: Scalp contamination for variance
        # Scalp hemodynamics dominate variance shifts
        scalp_var_factor = 1.0 + (SCALP_MOMENT_SHIFT_PS * mean_t / var_t)**2
        sigma_var *= np.sqrt(scalp_var_factor)

        # Min detectable ΔHbO from variance
        epsilon_hbo = EPSILON_HBO[wl_idx]
        if abs(dvar_dua) > 0 and epsilon_hbo > 0:
            min_hbo_var = sigma_var / (abs(dvar_dua) * epsilon_hbo) * 1e3  # uM
        else:
            min_hbo_var = float('inf')

        print(f"  {mr['det_id']:4d}  {mr['sds']:5.0f}  {mr['wl']:>5s}  "
              f"{var_t:10.1f}  {dvar_dua:10.2f}  {sigma_var:10.2e}  {min_hbo_var:10.3f}")

        if min_hbo_var < best_moment_hbo:
            best_moment_hbo = min_hbo_var
            best_moment_config = {
                'det_id': mr['det_id'], 'sds': mr['sds'], 'wl': mr['wl'],
                'min_hbo_var': min_hbo_var, 'min_hbo_mean': mr['min_hbo_moment']
            }

    if best_moment_config:
        print(f"\n  BEST MOMENT RESULT:")
        print(f"    Det {best_moment_config['det_id']}, "
              f"SDS={best_moment_config['sds']:.0f}mm, {best_moment_config['wl']}")
        print(f"    Min detectable HbO (mean-time): {best_moment_config['min_hbo_mean']:.3f} uM")
        print(f"    Min detectable HbO (variance):  {best_moment_config['min_hbo_var']:.3f} uM")
    else:
        print(f"\n  No valid moment results (insufficient TPSF signal)")

    return moment_results


# ===================================================================
# 11. COMBINED TD-GATED + SSR (per-gate regression)
# ===================================================================
def combined_td_ssr(results):
    print("\n" + "=" * 80)
    print("11. COMBINED TD-GATING + SSR (per-gate short-channel regression)")
    print(f"    SSR reference: SDS < {SSR_SDS_THRESHOLD_MM:.0f} mm | "
          f"Focus: late gates 6-9")
    print("=" * 80)

    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]

    # Identify short and long detectors
    short_dets = {}
    long_dets = {}
    for wl_key in WL_KEYS:
        r = results[wl_key]
        short_dets[wl_key] = [d for d in r["detectors"]
                               if d["sds_mm"] < SSR_SDS_THRESHOLD_MM]
        long_dets[wl_key] = [d for d in r["detectors"]
                              if d["sds_mm"] >= SSR_SDS_THRESHOLD_MM]

    if not short_dets[WL_KEYS[0]]:
        print(f"\n  WARNING: No short-SDS detectors. Cannot perform combined analysis.")
        print(f"  Add detectors at SDS < {SSR_SDS_THRESHOLD_MM:.0f} mm.")
        return None

    # Build per-gate SSR reference (averaged over short detectors)
    ssr_ref = {}  # ssr_ref[wl_key][g_idx] = {'scalp': ..., 'amyg': ...}
    for wl_key in WL_KEYS:
        sdets = short_dets[wl_key]
        n_gates_ref = len(sdets[0].get("time_gates", [])) if sdets else 0
        ssr_ref[wl_key] = []
        for g_idx in range(n_gates_ref):
            scalp_sum, amyg_sum = 0.0, 0.0
            for sd in sdets:
                gates = sd.get("time_gates", [])
                if g_idx < len(gates):
                    ppl = gates[g_idx].get("partial_pathlength_mm", {})
                    scalp_sum += ppl.get("scalp", 0)
                    amyg_sum += ppl.get("amygdala", 0)
            n = len(sdets)
            ssr_ref[wl_key].append({
                'scalp': scalp_sum / n, 'amyg': amyg_sum / n
            })

    # Focus on late gates (6-9) where deep sensitivity is highest
    focus_gates = list(range(6, 10))
    available_gates = [g for g in focus_gates
                       if g < len(ssr_ref[WL_KEYS[0]])]

    if not available_gates:
        available_gates = list(range(len(ssr_ref[WL_KEYS[0]])))

    print(f"\n  Analyzing gates: {[GATE_LABELS[g] for g in available_gates]}")

    print(f"\n  {'SDS':>5s}  {'Gate':>8s}  "
          f"{'raw_HbO':>9s}  {'SSR_HbO':>9s}  {'Improv':>7s}  "
          f"{'ScalpRej':>9s}  {'Status':>10s}")
    print("  " + "-" * 68)

    combined_results = []
    best_combined_hbo = float('inf')
    best_combined_config = None

    n_long = len(long_dets[WL_KEYS[0]])
    for d_idx in range(n_long):
        if d_idx >= len(long_dets[WL_KEYS[1]]):
            break
        det0 = long_dets[WL_KEYS[0]][d_idx]
        det1 = long_dets[WL_KEYS[1]][d_idx]
        sds = det0["sds_mm"]

        for g_idx in available_gates:
            gates0 = det0.get("time_gates", [])
            gates1 = det1.get("time_gates", [])
            if g_idx >= len(gates0) or g_idx >= len(gates1):
                continue

            g0, g1 = gates0[g_idx], gates1[g_idx]
            L0 = g0.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L1 = g1.get("partial_pathlength_mm", {}).get("amygdala", 0)
            scalp0 = g0.get("partial_pathlength_mm", {}).get("scalp", 0)
            scalp1 = g1.get("partial_pathlength_mm", {}).get("scalp", 0)
            if L0 <= 0 or L1 <= 0:
                continue

            # Photon counts
            gw0, gw1 = g0.get("weight", 0), g1.get("weight", 0)
            scale0 = N_per_sec[0] / results[WL_KEYS[0]]["num_photons"]
            scale1 = N_per_sec[1] / results[WL_KEYS[1]]["num_photons"]
            N0 = gw0 * scale0 * MEAS_TIME_S * DET_QE[WL_KEYS[0]] + DARK_COUNT_RATE * MEAS_TIME_S
            N1 = gw1 * scale1 * MEAS_TIME_S * DET_QE[WL_KEYS[1]] + DARK_COUNT_RATE * MEAS_TIME_S
            noise0 = compute_noise(N0, g_idx)
            noise1 = compute_noise(N1, g_idx)

            # Raw MBLL (TD-gated, no SSR)
            E_raw = np.array([
                [EPSILON_HBO[0] * L0, EPSILON_HBR[0] * L0],
                [EPSILON_HBO[1] * L1, EPSILON_HBR[1] * L1]
            ])
            if abs(np.linalg.det(E_raw)) < 1e-20:
                continue
            E_inv_raw = np.linalg.inv(E_raw)
            dc_raw = np.abs(E_inv_raw @ np.array([noise0, noise1])) * 1e3
            min_hbo_raw = dc_raw[0]

            # SSR correction within this gate
            ref0 = ssr_ref[WL_KEYS[0]][g_idx] if g_idx < len(ssr_ref[WL_KEYS[0]]) else {'scalp': 0, 'amyg': 0}
            ref1 = ssr_ref[WL_KEYS[1]][g_idx] if g_idx < len(ssr_ref[WL_KEYS[1]]) else {'scalp': 0, 'amyg': 0}

            beta0 = scalp0 / ref0['scalp'] if ref0['scalp'] > 0 else 0
            beta1 = scalp1 / ref1['scalp'] if ref1['scalp'] > 0 else 0

            L0_corr = max(L0 - beta0 * ref0['amyg'], 0)
            L1_corr = max(L1 - beta1 * ref1['amyg'], 0)

            if L0_corr <= 0 or L1_corr <= 0:
                continue

            # Scalp rejection ratio
            scalp_rej_0 = 1.0 - (scalp0 - beta0 * ref0['scalp']) / scalp0 if scalp0 > 0 else 0
            scalp_rej = scalp_rej_0 * 100

            # SSR noise (reference channel adds noise in quadrature)
            noise0_ssr = np.sqrt(noise0**2 + (beta0 * noise0)**2)
            noise1_ssr = np.sqrt(noise1**2 + (beta1 * noise1)**2)

            E_ssr = np.array([
                [EPSILON_HBO[0] * L0_corr, EPSILON_HBR[0] * L0_corr],
                [EPSILON_HBO[1] * L1_corr, EPSILON_HBR[1] * L1_corr]
            ])
            if abs(np.linalg.det(E_ssr)) < 1e-20:
                continue
            E_inv_ssr = np.linalg.inv(E_ssr)
            dc_ssr = np.abs(E_inv_ssr @ np.array([noise0_ssr, noise1_ssr])) * 1e3
            min_hbo_ssr = dc_ssr[0]

            improvement = min_hbo_raw / min_hbo_ssr if min_hbo_ssr > 0 else 0
            status = "IMPROVED" if improvement > 1.0 else "no gain"
            if min_hbo_ssr < 2.0:
                status = "EXCELLENT" if min_hbo_ssr < 1.0 else "GOOD"

            print(f"  {sds:5.0f}  {GATE_LABELS[g_idx]:>8s}  "
                  f"{min_hbo_raw:9.3f}  {min_hbo_ssr:9.3f}  "
                  f"{improvement:6.2f}x  {scalp_rej:8.1f}%  {status:>10s}")

            combined_results.append({
                'sds': sds, 'gate': g_idx, 'det_id': det0['id'],
                'min_hbo_raw': min_hbo_raw, 'min_hbo_ssr': min_hbo_ssr,
                'improvement': improvement, 'scalp_rejection': scalp_rej
            })

            if min_hbo_ssr < best_combined_hbo:
                best_combined_hbo = min_hbo_ssr
                best_combined_config = combined_results[-1]

    # Summary
    print(f"\n  COMBINED TD+SSR SUMMARY:")
    if best_combined_config:
        bc = best_combined_config
        print(f"    Best config: SDS={bc['sds']:.0f}mm, gate={GATE_LABELS[bc['gate']]}")
        print(f"    Min detectable HbO (TD only):     {bc['min_hbo_raw']:.3f} uM")
        print(f"    Min detectable HbO (TD + SSR):    {bc['min_hbo_ssr']:.3f} uM")
        print(f"    Combined improvement factor:      {bc['improvement']:.2f}x")
        print(f"    Scalp signal rejection:           {bc['scalp_rejection']:.1f}%")
    else:
        print(f"    No valid combined results.")

    # Compare all methods
    print(f"\n  METHOD COMPARISON (best result from each):")
    print(f"  {'Method':<25s}  {'minHbO [uM]':>12s}  {'Notes':>30s}")
    print("  " + "-" * 70)

    if combined_results:
        raw_results = [cr['min_hbo_raw'] for cr in combined_results]
        ssr_results = [cr['min_hbo_ssr'] for cr in combined_results]
        best_raw = min(raw_results) if raw_results else float('inf')
        best_ssr = min(ssr_results) if ssr_results else float('inf')

        print(f"  {'TD-gating only':<25s}  {best_raw:12.3f}  {'late gates, no regression':>30s}")
        print(f"  {'TD + SSR combined':<25s}  {best_ssr:12.3f}  {'per-gate regression':>30s}")
        if best_raw > 0:
            print(f"  {'Improvement':<25s}  {best_raw/best_ssr:12.2f}x")

    return combined_results


def main():
    parser = argparse.ArgumentParser(
        description="fNIRS MC -- TD-Gated Analysis (optimized config)")
    parser.add_argument("--data-dir", type=str, default="../results")
    parser.add_argument("--no-irf", action="store_true", 
                        help="Disable IRF convolution (use delta-function response)")
    parser.add_argument("--irf-fwhm", type=float, default=100.0, 
                        help="IRF FWHM in picoseconds (default: 100 ps)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 80)
    print("  fNIRS MC Analysis -- 730/850nm, Si-PMT, Multi-Channel")
    print(f"  Laser: {LASER_POWER_W*1e3:.0f} mW ({BEAM_DIAMETER_MM:.0f}mm beam) | "
          f"Det: Si-PMT ({DET_QE[WL_KEYS[0]]*100:.0f}%/{DET_QE[WL_KEYS[1]]*100:.0f}%) | "
          f"Dark: {DARK_COUNT_RATE} cps")
    print(f"  Noise: shot + physio ({PHYSIO_NOISE_CW*100:.1f}%CW/{PHYSIO_NOISE_LATE_GATE*100:.1f}%late) + source ({SOURCE_STABILITY*100:.1f}%) in quadrature")
    if not args.no_irf:
        print(f"  IRF: Gaussian {args.irf_fwhm:.0f}ps FWHM (realistic temporal blurring)")
    else:
        print(f"  IRF: Delta-function (ideal response)")
    print("=" * 80)

    results = load_results(data_dir)
    missing = [wl for wl in WL_KEYS if wl not in results]
    if missing:
        print(f"ERROR: missing required result files for {', '.join(missing)} in {data_dir}")
        print("Expected files like: results_730nm.json, results_850nm.json")
        print("Run the simulator with both wavelengths and copy fresh data before analysis.")
        sys.exit(1)

    apply_irf = not args.no_irf
    tpsf_0 = load_tpsf(data_dir, WL_KEYS[0], apply_irf=apply_irf, irf_fwhm_ps=args.irf_fwhm)
    tpsf_1 = load_tpsf(data_dir, WL_KEYS[1], apply_irf=apply_irf, irf_fwhm_ps=args.irf_fwhm)

    for key, tpsf in [(WL_KEYS[0], tpsf_0), (WL_KEYS[1], tpsf_1)]:
        if tpsf is not None:
            print(f"  TPSF {key}: {tpsf.shape}")

    td_sensitivity(results)
    gate_budget(results)
    best_configs = mbll_single(results)
    multi_result = mbll_multi_channel(results)
    block_design(multi_result, best_configs)
    time_sweep(results)
    safety_check()
    depth_specificity(results)
    ssr_results = ssr_analysis(results)
    moment_results = tpsf_moment_analysis(results, tpsf_0, tpsf_1)
    combined_results = combined_td_ssr(results)

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if multi_result:
        print(f"  Multi-channel (3 best detectors, all gates):")
        print(f"    Min detectable (120s): HbO={multi_result['min_hbo']:.4f} uM, "
              f"HbR={multi_result['min_hbr']:.4f} uM")
        if multi_result['min_hbo'] < 1.0 and multi_result['min_hbr'] < 1.0:
            print(f"  >> BOTH HbO AND HbR DETECTABLE <<")
        elif multi_result['min_hbo'] < 1.0:
            print(f"  >> HbO DETECTABLE, HbR marginal/needs block averaging <<")
    if best_configs:
        best = min(best_configs, key=lambda x: x[1]['min_hbo'])
        sds, cfg = best
        print(f"  Single-detector best: SDS={sds:.0f}mm, gate={GATE_LABELS[cfg['gate']]}")
        print(f"    Min detectable (120s): HbO={cfg['min_hbo']:.3f} uM, "
              f"HbR={cfg['min_hbr']:.3f} uM")

    # SSR results summary
    if ssr_results:
        best_ssr = min(ssr_results, key=lambda x: x['min_hbo_ssr'])
        print(f"  SSR-corrected best: SDS={best_ssr['sds']:.0f}mm, "
              f"gate={GATE_LABELS[best_ssr['gate']]}")
        print(f"    Min detectable (120s): HbO={best_ssr['min_hbo_ssr']:.3f} uM "
              f"({best_ssr['improvement']:.1f}x vs raw)")

    # Moment analysis summary
    if moment_results:
        valid_moments = [m for m in moment_results if m['min_hbo_moment'] < 1e6]
        if valid_moments:
            best_m = min(valid_moments, key=lambda x: x['min_hbo_moment'])
            print(f"  TPSF moment best: Det {best_m['det_id']}, "
                  f"SDS={best_m['sds']:.0f}mm, {best_m['wl']}")
            print(f"    Min detectable (mean-time): HbO={best_m['min_hbo_moment']:.3f} uM")

    # Combined TD+SSR summary
    if combined_results:
        best_comb = min(combined_results, key=lambda x: x['min_hbo_ssr'])
        print(f"  TD+SSR combined best: SDS={best_comb['sds']:.0f}mm, "
              f"gate={GATE_LABELS[best_comb['gate']]}")
        print(f"    Min detectable (120s): HbO={best_comb['min_hbo_ssr']:.3f} uM "
              f"({best_comb['improvement']:.1f}x improvement)")


if __name__ == "__main__":
    main()
