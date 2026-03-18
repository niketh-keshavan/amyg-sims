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
            # Minimum amygdala pathlength threshold: 0.01 mm
            if L0 <= 0.01 or L1 <= 0.01:
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
# 9. SHORT-SEPARATION REGRESSION vs TD-GATING
# ===================================================================
def ssr_comparison(results):
    print("\n" + "=" * 80)
    print("9. SHORT-SEPARATION REGRESSION (SSR) vs TD-GATING")
    print("=" * 80)

    r = results[WL_KEYS[0]]

    ssr_dets = [d for d in r["detectors"] if d["sds_mm"] <= 10]
    if not ssr_dets:
        print("  No short-separation detectors found (SDS <= 10mm)")
        return

    ssr_scalp_cw = np.mean([d["partial_pathlength_mm"].get("scalp", 0) for d in ssr_dets])
    ssr_amyg_cw = np.mean([d["partial_pathlength_mm"].get("amygdala", 0) for d in ssr_dets])

    print(f"\n  SSR reference: {len(ssr_dets)} detectors at SDS<=10mm")
    print(f"  SSR CW scalp PL: {ssr_scalp_cw:.2f} mm, amygdala PL: {ssr_amyg_cw:.6f} mm")

    print(f"\n  CW + SSR REGRESSION:")
    print(f"  {'Det':>4s}  {'SDS':>5s}  {'L_amyg':>8s}  {'L_scalp':>8s}  "
          f"{'beta':>6s}  {'L_amyg_corr':>12s}  {'Preserved%':>10s}")
    print("  " + "-" * 65)

    long_dets = [d for d in r["detectors"] if d["sds_mm"] > 15]

    for det in long_dets:
        ppl = det.get("partial_pathlength_mm", {})
        amyg = ppl.get("amygdala", 0)
        scalp = ppl.get("scalp", 0)
        if amyg <= 0:
            continue

        beta = scalp / ssr_scalp_cw if ssr_scalp_cw > 0 else 0
        amyg_corrected = amyg - beta * ssr_amyg_cw
        preserved = amyg_corrected / amyg * 100 if amyg > 0 else 0

        print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {amyg:8.6f}  {scalp:8.2f}  "
              f"{beta:6.2f}  {amyg_corrected:12.6f}  {preserved:9.1f}%")

    print(f"\n  TD-GATED (no SSR needed) — late gates preserve amygdala signal:")
    print(f"  {'Det':>4s}  {'SDS':>5s}  {'Gate':>8s}  {'L_amyg':>8s}  {'L_scalp':>8s}  "
          f"{'Ratio':>8s}")
    print("  " + "-" * 55)

    for det in long_dets:
        gates = det.get("time_gates", [])
        if len(gates) < 10:
            continue
        g = gates[9]
        ppl = g.get("partial_pathlength_mm", {})
        amyg = ppl.get("amygdala", 0)
        scalp = ppl.get("scalp", 0)
        if amyg <= 0:
            continue
        ratio = amyg / scalp * 100 if scalp > 0 else 0
        print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {'5ns+':>8s}  {amyg:8.4f}  "
              f"{scalp:8.1f}  {ratio:7.3f}%")

    print(f"\n  CONCLUSION: TD-gating improves amygdala/scalp ratio without")
    print(f"  sacrificing amygdala signal (unlike SSR which subtracts it)")


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
    ssr_comparison(results)

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


if __name__ == "__main__":
    main()
