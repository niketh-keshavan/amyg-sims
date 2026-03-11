#!/usr/bin/env python3
"""
fNIRS Monte Carlo Analysis Pipeline - CW / TD / FD / Chirp
------------------------------------------------------------
Loads simulation outputs and computes:
  1. CW sensitivity analysis (baseline)
  2. Time-domain: TPSF analysis, time-gated sensitivity & MBLL
  3. Frequency-domain: amplitude & phase from FFT of TPSF
  4. Chirp correlation: matched filter processing gain (5-500 MHz)
  5. SNR comparison across all modalities with 1W laser

Usage:
    python analyze.py --data-dir ../results
"""

import json
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical constants and extinction coefficients
# ---------------------------------------------------------------------------
#                     760 nm         850 nm
EPSILON_HBR = np.array([1.1058,  0.6918]) / 10.0  # [1/(mM*mm)]
EPSILON_HBO = np.array([0.1496,  1.0507]) / 10.0   # [1/(mM*mm)]

H_PLANCK = 6.626e-34     # J*s
C_LIGHT  = 3e8           # m/s
WAVELENGTHS_M = np.array([760e-9, 850e-9])

GATE_LABELS = [
    "0-500ps", "500-1000ps", "1.0-1.5ns",
    "1.5-2.5ns", "2.5-4.0ns", "4.0ns+"
]


def load_results(data_dir):
    """Load simulation results for both wavelengths."""
    results = {}
    for wl in ["760nm", "850nm"]:
        fpath = data_dir / f"results_{wl}.json"
        if fpath.exists():
            with open(fpath) as f:
                results[wl] = json.load(f)
    return results


def load_tpsf(data_dir, wavelength_key):
    """Load TPSF binary data: double[n_dets * TPSF_BINS]."""
    fpath = data_dir / f"tpsf_{wavelength_key}.bin"
    if not fpath.exists():
        return None
    data = np.fromfile(fpath, dtype=np.float64)
    n_bins = 512
    n_dets = len(data) // n_bins
    return data.reshape(n_dets, n_bins)


def photons_per_second(power_W, wavelength_m):
    """Compute photons per second for given laser power and wavelength."""
    E_photon = H_PLANCK * C_LIGHT / wavelength_m
    return power_W / E_photon


# ===================================================================
# 1. CW SENSITIVITY ANALYSIS
# ===================================================================
def cw_analysis(results):
    print("\n" + "=" * 74)
    print("1. CW SENSITIVITY ANALYSIS")
    print("=" * 74)

    for wl_key in ["760nm", "850nm"]:
        r = results[wl_key]
        print(f"\n  Wavelength: {r['wavelength_nm']} nm")
        print(f"  {'Det':>4s}  {'SDS':>5s}  {'Ang':>5s}  {'Detected':>12s}  "
              f"{'MeanPL':>8s}  {'AmygPL':>8s}  {'Sens':>10s}")
        print("  " + "-" * 62)

        for det in r["detectors"]:
            ppl = det["partial_pathlength_mm"]
            amyg_pl = ppl["amygdala"]
            mean_pl = det["mean_pathlength_mm"]
            sens = amyg_pl / mean_pl if mean_pl > 0 else 0
            angle = det.get("angle_deg", 0)

            print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {angle:+5.0f}  "
                  f"{det['detected_photons']:12d}  {mean_pl:8.1f}  "
                  f"{amyg_pl:8.4f}  {sens:10.6f}")


# ===================================================================
# 2. TIME-DOMAIN ANALYSIS
# ===================================================================
def td_analysis(results, tpsf_760, tpsf_850):
    print("\n" + "=" * 74)
    print("2. TIME-DOMAIN ANALYSIS")
    print("=" * 74)

    bin_ps = 10.0
    time_axis = np.arange(512) * bin_ps

    # --- TPSF summary ---
    if tpsf_760 is not None:
        r760 = results["760nm"]
        n_dets = tpsf_760.shape[0]

        print(f"\n  TPSF Summary (760 nm, primary direction):")
        print(f"  {'Det':>4s}  {'SDS':>5s}  {'MeanTOF':>10s}  "
              f"{'StdTOF':>10s}  {'PeakTOF':>10s}  {'FWHM':>8s}")
        print("  " + "-" * 58)

        for i, det in enumerate(r760["detectors"]):
            if i >= n_dets:
                break
            if abs(det.get("angle_deg", 0)) > 1:
                continue
            tpsf = tpsf_760[i]
            if tpsf.sum() <= 0:
                continue
            tpsf_n = tpsf / tpsf.sum()
            mean_tof = np.sum(time_axis * tpsf_n)
            var_tof = np.sum((time_axis - mean_tof)**2 * tpsf_n)
            std_tof = np.sqrt(var_tof)
            peak_ps = time_axis[np.argmax(tpsf)]
            # FWHM
            half_max = tpsf.max() / 2
            above = np.where(tpsf >= half_max)[0]
            fwhm = (above[-1] - above[0]) * bin_ps if len(above) > 1 else 0

            print(f"  {i:4d}  {det['sds_mm']:5.0f}  {mean_tof:10.1f}  "
                  f"{std_tof:10.1f}  {peak_ps:10.1f}  {fwhm:8.1f}  [ps]")

    # --- Time-gated amygdala sensitivity ---
    print(f"\n  Time-Gated Amygdala Sensitivity (760 nm, primary direction):")
    print(f"  Sensitivity = amygdala_PL / total_PL for each gate\n")

    r760 = results["760nm"]
    header = f"  {'Det':>4s}  {'SDS':>5s}"
    for gl in GATE_LABELS:
        header += f"  {gl:>10s}"
    print(header)
    print("  " + "-" * (14 + 12 * len(GATE_LABELS)))

    for det in r760["detectors"]:
        if abs(det.get("angle_deg", 0)) > 1:
            continue
        gates = det.get("time_gates", [])
        if not gates:
            continue
        line = f"  {det['id']:4d}  {det['sds_mm']:5.0f}"
        for g in gates:
            ppl = g.get("partial_pathlength_mm", {})
            amyg = ppl.get("amygdala", 0)
            total = sum(ppl.values())
            sens = amyg / total if total > 0 else 0
            line += f"  {sens:10.6f}"
        print(line)

    # --- Time-gated amygdala partial pathlength ---
    print(f"\n  Time-Gated Amygdala Partial Pathlength [mm] (760 nm):\n")
    header = f"  {'Det':>4s}  {'SDS':>5s}"
    for gl in GATE_LABELS:
        header += f"  {gl:>10s}"
    print(header)
    print("  " + "-" * (14 + 12 * len(GATE_LABELS)))

    for det in r760["detectors"]:
        if abs(det.get("angle_deg", 0)) > 1:
            continue
        gates = det.get("time_gates", [])
        if not gates:
            continue
        line = f"  {det['id']:4d}  {det['sds_mm']:5.0f}"
        for g in gates:
            amyg = g.get("partial_pathlength_mm", {}).get("amygdala", 0)
            line += f"  {amyg:10.4f}"
        print(line)

    # --- Photon count per gate ---
    print(f"\n  Detected Photons per Gate (760 nm, primary direction):\n")
    header = f"  {'Det':>4s}  {'SDS':>5s}"
    for gl in GATE_LABELS:
        header += f"  {gl:>10s}"
    print(header)
    print("  " + "-" * (14 + 12 * len(GATE_LABELS)))

    for det in r760["detectors"]:
        if abs(det.get("angle_deg", 0)) > 1:
            continue
        gates = det.get("time_gates", [])
        if not gates:
            continue
        line = f"  {det['id']:4d}  {det['sds_mm']:5.0f}"
        for g in gates:
            cnt = g.get("detected_photons", 0)
            line += f"  {cnt:10d}"
        print(line)


# ===================================================================
# 3. FREQUENCY-DOMAIN ANALYSIS
# ===================================================================
def fd_analysis(tpsf_760, tpsf_850, results):
    print("\n" + "=" * 74)
    print("3. FREQUENCY-DOMAIN ANALYSIS (from TPSF via FFT)")
    print("=" * 74)

    if tpsf_760 is None:
        print("  No TPSF data available - skipping.")
        return

    bin_ps = 10.0
    dt_s = bin_ps * 1e-12
    n_bins = 512

    # Zero-pad for fine frequency resolution
    n_fft = 32768
    freqs_hz = np.fft.rfftfreq(n_fft, d=dt_s)
    freqs_mhz = freqs_hz / 1e6

    target_freqs = [10, 50, 100, 200, 300, 500]  # MHz
    r760 = results["760nm"]

    print(f"\n  FFT zero-padded to {n_fft} points")
    print(f"  Frequency resolution: {freqs_mhz[1]:.2f} MHz")
    print(f"  Max frequency: {freqs_mhz[-1]:.0f} MHz\n")

    # --- Phase shift ---
    print(f"  Phase shift [deg] (760 nm, primary direction):")
    header = f"  {'Det':>4s}  {'SDS':>5s}"
    for tf in target_freqs:
        header += f"  {tf:>6d}MHz"
    print(header)
    print("  " + "-" * (14 + 10 * len(target_freqs)))

    for i, det in enumerate(r760["detectors"]):
        if abs(det.get("angle_deg", 0)) > 1:
            continue
        if i >= tpsf_760.shape[0]:
            break
        tpsf = tpsf_760[i]
        if tpsf.sum() == 0:
            continue
        H = np.fft.rfft(tpsf, n=n_fft)
        phase_deg = np.angle(H) * 180 / np.pi

        line = f"  {det['id']:4d}  {det['sds_mm']:5.0f}"
        for tf in target_freqs:
            idx = np.argmin(np.abs(freqs_mhz - tf))
            line += f"  {phase_deg[idx]:9.2f}"
        print(line)

    # --- Amplitude attenuation ---
    print(f"\n  Amplitude |H(f)|/|H(0)| [dB] (760 nm, primary direction):")
    header = f"  {'Det':>4s}  {'SDS':>5s}"
    for tf in target_freqs:
        header += f"  {tf:>6d}MHz"
    print(header)
    print("  " + "-" * (14 + 10 * len(target_freqs)))

    for i, det in enumerate(r760["detectors"]):
        if abs(det.get("angle_deg", 0)) > 1:
            continue
        if i >= tpsf_760.shape[0]:
            break
        tpsf = tpsf_760[i]
        if tpsf.sum() == 0:
            continue
        H = np.fft.rfft(tpsf, n=n_fft)
        H_db = 20 * np.log10(np.abs(H) / (np.abs(H[0]) + 1e-30) + 1e-30)

        line = f"  {det['id']:4d}  {det['sds_mm']:5.0f}"
        for tf in target_freqs:
            idx = np.argmin(np.abs(freqs_mhz - tf))
            line += f"  {H_db[idx]:9.2f}"
        print(line)

    # --- Phase difference between SDS (depth encoding) ---
    print(f"\n  Differential phase: phase(SDS) - phase(SDS=8mm) at 200 MHz:")
    print(f"  Depth-encoded phase increases with SDS (deeper photon paths).\n")
    # Find the SDS=8mm reference (first primary-direction detector)
    ref_idx = None
    for i, det in enumerate(r760["detectors"]):
        if abs(det.get("angle_deg", 0)) < 1 and det["sds_mm"] < 10:
            ref_idx = i
            break

    if ref_idx is not None and tpsf_760[ref_idx].sum() > 0:
        H_ref = np.fft.rfft(tpsf_760[ref_idx], n=n_fft)
        idx_200 = np.argmin(np.abs(freqs_mhz - 200))
        phase_ref = np.angle(H_ref[idx_200])

        print(f"  {'Det':>4s}  {'SDS':>5s}  {'dPhase_200MHz':>14s}")
        print("  " + "-" * 30)
        for i, det in enumerate(r760["detectors"]):
            if abs(det.get("angle_deg", 0)) > 1:
                continue
            if i >= tpsf_760.shape[0]:
                break
            tpsf = tpsf_760[i]
            if tpsf.sum() == 0:
                continue
            H = np.fft.rfft(tpsf, n=n_fft)
            dphase = (np.angle(H[idx_200]) - phase_ref) * 180 / np.pi
            print(f"  {det['id']:4d}  {det['sds_mm']:5.0f}  {dphase:14.2f} deg")


# ===================================================================
# 4. CHIRP CORRELATION ANALYSIS
# ===================================================================
def chirp_analysis(tpsf_760, tpsf_850, results):
    print("\n" + "=" * 74)
    print("4. CHIRP CORRELATION ANALYSIS (5-500 MHz sweep)")
    print("=" * 74)

    if tpsf_760 is None:
        print("  No TPSF data available - skipping.")
        return

    bin_ps = 10.0
    dt_s = bin_ps * 1e-12
    n_bins = 512
    T_chirp = n_bins * dt_s   # 5.12 ns

    f_start = 5e6     # 5 MHz
    f_stop  = 500e6   # 500 MHz
    BW = f_stop - f_start  # 495 MHz

    # Time-bandwidth product = processing gain
    tbp = T_chirp * BW
    processing_gain_db = 10 * np.log10(tbp) if tbp > 0 else 0

    print(f"\n  Chirp parameters:")
    print(f"    Frequency sweep:     {f_start/1e6:.0f} - {f_stop/1e6:.0f} MHz")
    print(f"    Chirp duration:      {T_chirp*1e9:.2f} ns")
    print(f"    Bandwidth:           {BW/1e6:.0f} MHz")
    print(f"    Time-bandwidth product (TBP): {tbp:.2f}")
    print(f"    Processing gain:     {processing_gain_db:.1f} dB")
    print(f"    SNR improvement:     {np.sqrt(tbp):.2f}x\n")

    # Generate chirp
    t = np.arange(n_bins) * dt_s
    chirp_rate = (f_stop - f_start) / T_chirp
    chirp = np.cos(2 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2))

    # For 1W laser at 500 MHz rep rate (matched to chirp bandwidth)
    laser_power = 1.0
    wl_m = 760e-9
    N_per_sec = photons_per_second(laser_power, wl_m)
    r760 = results["760nm"]
    num_sim = r760["num_photons"]
    scale = N_per_sec / num_sim

    print(f"  1W laser @ 760 nm: {N_per_sec:.3e} photons/s")
    print(f"  Simulation scale factor: {scale:.3e}\n")

    print(f"  {'Det':>4s}  {'SDS':>5s}  {'Ang':>5s}  {'CW_SNR':>10s}  "
          f"{'Chirp_SNR':>12s}  {'Gain_dB':>8s}  {'Peak_corr':>10s}")
    print("  " + "-" * 64)

    for i, det in enumerate(r760["detectors"]):
        if i >= tpsf_760.shape[0]:
            break
        tpsf = tpsf_760[i]
        w = det["total_weight"]
        if w <= 0:
            continue

        # CW SNR (1 second measurement, shot noise limited)
        N_cw = w * scale
        snr_cw = np.sqrt(N_cw) if N_cw > 0 else 0

        # Chirp-correlated response
        response = np.convolve(tpsf, chirp, mode='full')[:n_bins]
        correlation = np.correlate(response, chirp, mode='full')
        peak_corr = np.max(np.abs(correlation))

        # Chirp SNR = CW SNR * sqrt(TBP)
        snr_chirp = snr_cw * np.sqrt(tbp)
        gain_db = 10 * np.log10(snr_chirp / snr_cw) if snr_cw > 0 else 0

        angle = det.get("angle_deg", 0)
        print(f"  {i:4d}  {det['sds_mm']:5.0f}  {angle:+5.0f}  "
              f"{snr_cw:10.1f}  {snr_chirp:12.1f}  {gain_db:8.1f}  "
              f"{peak_corr:10.4e}")


# ===================================================================
# 5. COMPREHENSIVE SNR COMPARISON
# ===================================================================
def snr_comparison(results, tpsf_760, tpsf_850):
    print("\n" + "=" * 74)
    print("5. SNR COMPARISON: CW vs TD-GATED vs CHIRP-CORRELATED")
    print("    Can we detect a 1 uM HbO change in the amygdala?")
    print("=" * 74)

    laser_power = 1.0   # W
    meas_time = 1.0     # s

    # Typical hemodynamic response magnitude in amygdala
    delta_hbo = 0.001    # 1 uM in mM
    delta_hbr = -0.0003  # -0.3 uM in mM

    # Chirp parameters
    tbp = 5.12e-9 * 495e6  # T_chirp * BW

    print(f"\n  Laser: {laser_power} W | Measurement: {meas_time} s")
    print(f"  Target: d[HbO] = {delta_hbo*1e3:.1f} uM, d[HbR] = {delta_hbr*1e3:.1f} uM")
    print(f"  Chirp TBP = {tbp:.2f} (gain = {np.sqrt(tbp):.2f}x)\n")

    wl_keys = ["760nm", "850nm"]
    wl_m = [760e-9, 850e-9]
    N_per_sec = [photons_per_second(laser_power, w) for w in wl_m]

    # Collect data for primary-direction detectors at both wavelengths
    det_data = {}  # keyed by SDS
    for wl_idx, wl_key in enumerate(wl_keys):
        r = results[wl_key]
        num_sim = r["num_photons"]
        scale = N_per_sec[wl_idx] / num_sim

        for det in r["detectors"]:
            sds = det["sds_mm"]
            angle = det.get("angle_deg", 0)
            if abs(angle) > 1:
                continue  # primary direction only

            if sds not in det_data:
                det_data[sds] = {}

            w = det["total_weight"]
            amyg_pl = det["partial_pathlength_mm"]["amygdala"]
            N_det = w * scale * meas_time

            # Signal: delta_OD = (eps_HbO * dHbO + eps_HbR * dHbR) * L_amyg
            delta_od = (EPSILON_HBO[wl_idx] * delta_hbo +
                        EPSILON_HBR[wl_idx] * delta_hbr) * amyg_pl

            # CW noise floor
            noise_cw = 1.0 / np.sqrt(N_det) if N_det > 0 else float('inf')
            snr_cw = abs(delta_od) / noise_cw if noise_cw < float('inf') else 0

            # Best time gate
            gates = det.get("time_gates", [])
            best_gate_snr = 0
            best_gate_idx = -1
            best_gate_details = {}
            for g_idx, gate in enumerate(gates):
                gw = gate.get("weight", 0)
                g_amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                N_gate = gw * scale * meas_time
                delta_od_g = (EPSILON_HBO[wl_idx] * delta_hbo +
                              EPSILON_HBR[wl_idx] * delta_hbr) * g_amyg
                noise_g = 1.0 / np.sqrt(N_gate) if N_gate > 0 else float('inf')
                snr_g = abs(delta_od_g) / noise_g if noise_g < float('inf') else 0
                if snr_g > best_gate_snr:
                    best_gate_snr = snr_g
                    best_gate_idx = g_idx
                    best_gate_details = {
                        'N_gate': N_gate, 'amyg_pl': g_amyg,
                        'delta_od': delta_od_g
                    }

            # Chirp SNR = CW SNR * sqrt(TBP)
            snr_chirp = snr_cw * np.sqrt(tbp)

            det_data[sds][wl_key] = {
                'N_det': N_det, 'amyg_pl': amyg_pl,
                'delta_od': delta_od, 'noise_cw': noise_cw,
                'snr_cw': snr_cw,
                'snr_td': best_gate_snr, 'best_gate': best_gate_idx,
                'gate_details': best_gate_details,
                'snr_chirp': snr_chirp,
            }

    # Print per-wavelength comparison
    sds_list = sorted(det_data.keys())
    print(f"  {'SDS':>5s}  {'WL':>5s}  {'N_det':>10s}  {'L_amyg':>7s}  "
          f"{'dOD':>10s}  {'SNR_CW':>8s}  {'SNR_TD':>8s}  "
          f"{'Gate':>5s}  {'SNR_chirp':>10s}")
    print("  " + "-" * 82)

    for sds in sds_list:
        for wl_key in wl_keys:
            if wl_key not in det_data[sds]:
                continue
            d = det_data[sds][wl_key]
            gate_lbl = str(d['best_gate']) if d['best_gate'] >= 0 else "N/A"
            print(f"  {sds:5.0f}  {wl_key:>5s}  {d['N_det']:10.2e}  "
                  f"{d['amyg_pl']:7.4f}  {d['delta_od']:10.2e}  "
                  f"{d['snr_cw']:8.4f}  {d['snr_td']:8.4f}  "
                  f"{gate_lbl:>5s}  {d['snr_chirp']:10.4f}")

    # MBLL inversion: minimum detectable concentration
    print(f"\n  --- Minimum Detectable Concentration (MBLL inversion) ---")
    print(f"  Using dual-wavelength (760+850nm), 1W, 1s measurement\n")

    for sds in sds_list:
        if "760nm" not in det_data[sds] or "850nm" not in det_data[sds]:
            continue

        d760 = det_data[sds]["760nm"]
        d850 = det_data[sds]["850nm"]

        L760 = d760['amyg_pl']
        L850 = d850['amyg_pl']
        if L760 <= 0 or L850 <= 0:
            continue

        E = np.array([
            [EPSILON_HBO[0] * L760, EPSILON_HBR[0] * L760],
            [EPSILON_HBO[1] * L850, EPSILON_HBR[1] * L850]
        ])
        det_E = np.linalg.det(E)
        if abs(det_E) < 1e-20:
            continue

        E_inv = np.linalg.inv(E)

        # CW minimum detectable
        dOD_cw = np.array([d760['noise_cw'], d850['noise_cw']])
        dc_cw = np.abs(E_inv @ dOD_cw)

        # TD gated minimum
        gd760 = d760['gate_details']
        gd850 = d850['gate_details']
        if gd760 and gd850 and gd760.get('N_gate', 0) > 0 and gd850.get('N_gate', 0) > 0:
            Lg760 = gd760['amyg_pl']
            Lg850 = gd850['amyg_pl']
            if Lg760 > 0 and Lg850 > 0:
                E_g = np.array([
                    [EPSILON_HBO[0] * Lg760, EPSILON_HBR[0] * Lg760],
                    [EPSILON_HBO[1] * Lg850, EPSILON_HBR[1] * Lg850]
                ])
                det_Eg = np.linalg.det(E_g)
                if abs(det_Eg) > 1e-20:
                    E_g_inv = np.linalg.inv(E_g)
                    dOD_td = np.array([
                        1.0 / np.sqrt(gd760['N_gate']),
                        1.0 / np.sqrt(gd850['N_gate'])
                    ])
                    dc_td = np.abs(E_g_inv @ dOD_td)
                else:
                    dc_td = np.array([float('inf'), float('inf')])
            else:
                dc_td = np.array([float('inf'), float('inf')])
        else:
            dc_td = np.array([float('inf'), float('inf')])

        # Chirp minimum (processing gain reduces noise)
        dOD_chirp = dOD_cw / np.sqrt(tbp)
        dc_chirp = np.abs(E_inv @ dOD_chirp)

        can_cw = dc_cw[0] * 1e3 < 1.0
        can_td = dc_td[0] * 1e3 < 1.0
        can_chirp = dc_chirp[0] * 1e3 < 1.0

        print(f"  SDS = {sds:.0f} mm:")
        print(f"    CW:    min d[HbO] = {dc_cw[0]*1e3:8.3f} uM  "
              f"d[HbR] = {dc_cw[1]*1e3:8.3f} uM  "
              f"{'** DETECTABLE **' if can_cw else '(below threshold)'}")
        print(f"    TD:    min d[HbO] = {dc_td[0]*1e3:8.3f} uM  "
              f"d[HbR] = {dc_td[1]*1e3:8.3f} uM  "
              f"{'** DETECTABLE **' if can_td else '(below threshold)'}")
        print(f"    Chirp: min d[HbO] = {dc_chirp[0]*1e3:8.3f} uM  "
              f"d[HbR] = {dc_chirp[1]*1e3:8.3f} uM  "
              f"{'** DETECTABLE **' if can_chirp else '(below threshold)'}")

    # Final verdict
    print(f"\n  {'=' * 60}")
    print(f"  VERDICT: Can we detect 1 uM HbO change in the amygdala?")
    print(f"  {'=' * 60}")

    # Check SDS=35-40mm range (best amygdala sensitivity)
    for sds in [35, 40]:
        if sds not in det_data:
            continue
        if "760nm" not in det_data[sds]:
            continue
        d760 = det_data[sds]["760nm"]
        print(f"\n  At SDS = {sds} mm (1W laser, 1s measurement):")
        print(f"    CW fNIRS:      SNR = {d760['snr_cw']:.4f} "
              f"{'-> YES' if d760['snr_cw'] > 1 else '-> NO'}")
        print(f"    TD-gated:      SNR = {d760['snr_td']:.4f} "
              f"{'-> YES' if d760['snr_td'] > 1 else '-> NO'}")
        print(f"    Chirp-corr:    SNR = {d760['snr_chirp']:.4f} "
              f"{'-> YES' if d760['snr_chirp'] > 1 else '-> NO'}")


def main():
    parser = argparse.ArgumentParser(
        description="fNIRS MC Analysis - CW/TD/FD/Chirp Comparison")
    parser.add_argument("--data-dir", type=str, default="../results")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("Loading simulation data...")
    results = load_results(data_dir)
    tpsf_760 = load_tpsf(data_dir, "760nm")
    tpsf_850 = load_tpsf(data_dir, "850nm")

    if tpsf_760 is not None:
        print(f"  TPSF 760nm: {tpsf_760.shape} (sum={tpsf_760.sum():.3e})")
    else:
        print("  TPSF 760nm: not found")
    if tpsf_850 is not None:
        print(f"  TPSF 850nm: {tpsf_850.shape} (sum={tpsf_850.sum():.3e})")
    else:
        print("  TPSF 850nm: not found")

    cw_analysis(results)
    td_analysis(results, tpsf_760, tpsf_850)
    fd_analysis(tpsf_760, tpsf_850, results)
    chirp_analysis(tpsf_760, tpsf_850, results)
    snr_comparison(results, tpsf_760, tpsf_850)


if __name__ == "__main__":
    main()
