#!/usr/bin/env python3
"""
fNIRS Monte Carlo Analysis — TD-Gated Focus (2-min integration)
----------------------------------------------------------------
Optimized for time-domain gated measurement of amygdala hemodynamics.

Computes:
  1. TD-gated sensitivity by gate and SDS
  2. Optimal gate selection per detector
  3. Dual-wavelength MBLL inversion at 120s integration
  4. Expected block-design paradigm detectability
  5. Integration time sweep
  6. ANSI Z136.1 safety check

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
EPSILON_HBR = np.array([1.1058,  0.6918]) / 10.0  # [1/(mM*mm)]
EPSILON_HBO = np.array([0.1496,  1.0507]) / 10.0   # [1/(mM*mm)]

H_PLANCK = 6.626e-34
C_LIGHT  = 3e8
WAVELENGTHS_M = np.array([760e-9, 850e-9])

GATE_LABELS = [
    "0-500ps", "0.5-1ns", "1-1.5ns", "1.5-2ns", "2-2.5ns",
    "2.5-3ns", "3-3.5ns", "3.5-4ns", "4-5ns", "5ns+"
]

# ---------------------------------------------------------------------------
# System parameters for realistic TD-fNIRS
# ---------------------------------------------------------------------------
LASER_POWER_W = 0.1       # 100 mW average
MEAS_TIME_S = 120.0       # 2 minutes integration
DET_EFFICIENCY = 0.10     # 10% detector quantum efficiency (InGaAs SPAD)
DARK_COUNT_RATE = 1000    # counts/s per detector


def load_results(data_dir):
    results = {}
    for wl in ["760nm", "850nm"]:
        fpath = data_dir / f"results_{wl}.json"
        if fpath.exists():
            with open(fpath) as f:
                results[wl] = json.load(f)
    return results


def load_tpsf(data_dir, wavelength_key):
    fpath = data_dir / f"tpsf_{wavelength_key}.bin"
    if not fpath.exists():
        return None
    data = np.fromfile(fpath, dtype=np.float64)
    n_bins = 512
    n_dets = len(data) // n_bins
    return data.reshape(n_dets, n_bins)


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

    for wl_key in ["760nm", "850nm"]:
        r = results[wl_key]
        print(f"\n  Wavelength: {r['wavelength_nm']} nm")
        print(f"  {'Det':>4s}  {'SDS':>5s}  {'Ang':>5s}  {'BestGate':>8s}  "
              f"{'Photons':>12s}  {'AmygPL':>8s}  {'Sens%':>8s}  {'TotalPL':>8s}")
        print("  " + "-" * 70)

        for det in r["detectors"]:
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
    print("2. GATE PHOTON BUDGET (scaled to 100mW, 120s, 10% QE)")
    print("=" * 80)

    for wl_key in ["760nm", "850nm"]:
        r = results[wl_key]
        wl_idx = 0 if "760" in wl_key else 1
        N_per_sec = photons_per_second(LASER_POWER_W, WAVELENGTHS_M[wl_idx])
        num_sim = r["num_photons"]
        scale = N_per_sec / num_sim * MEAS_TIME_S * DET_EFFICIENCY

        print(f"\n  {wl_key} (primary direction, gates with amygdala signal):")
        print(f"  {'SDS':>5s}  {'Gate':>8s}  {'DetCounts':>12s}  {'AmygPL':>10s}  "
              f"{'dOD/µM':>10s}  {'SNR_1µM':>10s}")
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
                snr = delta_od * np.sqrt(N_det) if N_det > 0 else 0

                print(f"  {det['sds_mm']:5.0f}  {GATE_LABELS[g_idx]:>8s}  "
                      f"{N_det:12.2e}  {amyg:10.6f}  {delta_od:10.2e}  {snr:10.4f}")


# ===================================================================
# 3. DUAL-WAVELENGTH MBLL @ 120s
# ===================================================================
def mbll_analysis(results):
    print("\n" + "=" * 80)
    print("3. DUAL-WAVELENGTH MBLL INVERSION")
    print(f"   {LASER_POWER_W*1e3:.0f} mW | {MEAS_TIME_S:.0f}s | "
          f"{DET_EFFICIENCY*100:.0f}% QE | {DARK_COUNT_RATE} cps dark")
    print("=" * 80)

    wl_keys = ["760nm", "850nm"]
    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]

    det_data = {}
    for wl_idx, wl_key in enumerate(wl_keys):
        r = results[wl_key]
        num_sim = r["num_photons"]
        scale = N_per_sec[wl_idx] / num_sim

        for det in r["detectors"]:
            sds = det["sds_mm"]
            if abs(det.get("angle_deg", 0)) > 1:
                continue
            gates = det.get("time_gates", [])
            for g_idx, gate in enumerate(gates):
                key = (sds, g_idx)
                if key not in det_data:
                    det_data[key] = {}
                gw = gate.get("weight", 0)
                amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                N_gate = gw * scale * MEAS_TIME_S * DET_EFFICIENCY
                N_total = N_gate + DARK_COUNT_RATE * MEAS_TIME_S
                noise = 1.0 / np.sqrt(N_total) if N_total > 0 else float('inf')
                det_data[key][wl_key] = {
                    'amyg_pl': amyg, 'N_gate': N_gate, 'noise': noise
                }

    sds_set = sorted(set(s for s, g in det_data.keys()))

    print(f"\n  {'SDS':>5s}  {'Gate':>8s}  {'N760':>10s}  {'N850':>10s}  "
          f"{'L_amyg760':>10s}  {'L_amyg850':>10s}  {'minHbO':>8s}  {'minHbR':>8s}  "
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
            if "760nm" not in d or "850nm" not in d:
                continue

            d760, d850 = d["760nm"], d["850nm"]
            L760, L850 = d760['amyg_pl'], d850['amyg_pl']
            if L760 <= 0 or L850 <= 0:
                continue

            E = np.array([
                [EPSILON_HBO[0]*L760, EPSILON_HBR[0]*L760],
                [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]
            ])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            E_inv = np.linalg.inv(E)
            dc = np.abs(E_inv @ np.array([d760['noise'], d850['noise']])) * 1e3

            if dc[0] < best_min_hbo:
                best_min_hbo = dc[0]
                best_result = {
                    'gate': g_idx, 'N760': d760['N_gate'], 'N850': d850['N_gate'],
                    'L760': L760, 'L850': L850,
                    'min_hbo': dc[0], 'min_hbr': dc[1]
                }

        if best_result:
            r = best_result
            status = "DETECTABLE" if r['min_hbo'] < 1.0 else f"need>{r['min_hbo']:.1f}uM"
            print(f"  {sds:5.0f}  {GATE_LABELS[r['gate']]:>8s}  "
                  f"{r['N760']:10.2e}  {r['N850']:10.2e}  "
                  f"{r['L760']:10.6f}  {r['L850']:10.6f}  "
                  f"{r['min_hbo']:8.3f}  {r['min_hbr']:8.3f}  {status:>12s}")
            best_configs.append((sds, best_result))

    return best_configs


# ===================================================================
# 4. BLOCK DESIGN FEASIBILITY
# ===================================================================
def block_design(best_configs):
    print("\n" + "=" * 80)
    print("4. BLOCK-DESIGN PARADIGM (15s stim, 15s rest, 20 trials)")
    print("   Expected amygdala: 2-5 µM HbO, -0.5 to -1.5 µM HbR")
    print("=" * 80)

    n_trials = 20
    trial_dur = 15.0
    trial_gain = np.sqrt(n_trials)

    print(f"\n  SNR per HbO magnitude (trial-averaged, {n_trials} trials):\n")
    print(f"  {'SDS':>5s}  {'Gate':>8s}  {'eff_min':>8s}  "
          f"{'1µM':>7s}  {'2µM':>7s}  {'3µM':>7s}  {'5µM':>7s}  {'verdict':>10s}")
    print("  " + "-" * 68)

    for sds, cfg in best_configs:
        trial_min = cfg['min_hbo'] * np.sqrt(MEAS_TIME_S / trial_dur)
        avg_min = trial_min / trial_gain

        snrs = [h / avg_min if avg_min > 0 else 0 for h in [1, 2, 3, 5]]
        verdict = "YES" if snrs[1] >= 1.0 else ("marginal" if snrs[2] >= 1.0 else "no")

        print(f"  {sds:5.0f}  {GATE_LABELS[cfg['gate']]:>8s}  {avg_min:8.3f}  "
              f"{''.join(f'{s:7.2f}' for s in snrs)}  {verdict:>10s}")


# ===================================================================
# 5. INTEGRATION TIME SWEEP
# ===================================================================
def time_sweep(results):
    print("\n" + "=" * 80)
    print("5. MIN DETECTABLE HbO [µM] vs INTEGRATION TIME")
    print("=" * 80)

    wl_keys = ["760nm", "850nm"]
    N_per_sec = [photons_per_second(LASER_POWER_W, w) for w in WAVELENGTHS_M]
    times = [1, 5, 10, 15, 30, 60, 120]

    print(f"\n  {'SDS':>5s}", end="")
    for t in times:
        label = f"{t}s" if t < 60 else f"{t//60}m"
        print(f"  {label:>6s}", end="")
    print()
    print("  " + "-" * (7 + 8 * len(times)))

    r760, r850 = results["760nm"], results["850nm"]

    for det760, det850 in zip(r760["detectors"], r850["detectors"]):
        if abs(det760.get("angle_deg", 0)) > 1:
            continue
        sds = det760["sds_mm"]

        # Find best gate (at 1s baseline)
        best_min = float('inf')
        best_noise_760_1s = None
        best_noise_850_1s = None
        best_L760 = 0
        best_L850 = 0

        for g_idx in range(min(len(det760.get("time_gates", [])),
                               len(det850.get("time_gates", [])))):
            g760 = det760["time_gates"][g_idx]
            g850 = det850["time_gates"][g_idx]
            L760 = g760.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if L760 <= 0 or L850 <= 0:
                continue

            # Noise at 1s
            gw760 = g760.get("weight", 0)
            gw850 = g850.get("weight", 0)
            scale760 = N_per_sec[0] / r760["num_photons"]
            scale850 = N_per_sec[1] / r850["num_photons"]
            N760 = gw760 * scale760 * 1.0 * DET_EFFICIENCY + DARK_COUNT_RATE
            N850 = gw850 * scale850 * 1.0 * DET_EFFICIENCY + DARK_COUNT_RATE
            n760 = 1.0 / np.sqrt(N760) if N760 > 0 else float('inf')
            n850 = 1.0 / np.sqrt(N850) if N850 > 0 else float('inf')

            E = np.array([
                [EPSILON_HBO[0]*L760, EPSILON_HBR[0]*L760],
                [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]
            ])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            E_inv = np.linalg.inv(E)
            dc = np.abs(E_inv @ np.array([n760, n850])) * 1e3

            if dc[0] < best_min:
                best_min = dc[0]
                best_noise_760_1s = n760
                best_noise_850_1s = n850
                best_L760 = L760
                best_L850 = L850

        if best_noise_760_1s is None:
            continue

        line = f"  {sds:5.0f}"
        for t in times:
            min_hbo = best_min / np.sqrt(t)
            line += f"  {min_hbo:6.2f}"
        print(line)

    print(f"\n  Values = min detectable ΔHbO [µM] (dual-λ MBLL)")
    print(f"  Target: <1 µM single-trial, <2 µM block-averaged")


# ===================================================================
# 6. SAFETY CHECK
# ===================================================================
def safety_check():
    print("\n" + "=" * 80)
    print("6. ANSI Z136.1 LASER SAFETY")
    print("=" * 80)

    beam_area = np.pi * (0.35) ** 2  # 7mm diameter
    irradiance = LASER_POWER_W / beam_area

    for wl in [760, 850]:
        C_A = 10 ** (0.002 * (wl - 700))
        mpe = 0.2 * C_A
        ratio = irradiance / mpe
        print(f"  {wl} nm: MPE={mpe:.3f} W/cm^2, actual={irradiance:.3f}, "
              f"ratio={ratio:.3f} ({'SAFE' if ratio < 1 else 'EXCEEDS'})")


def main():
    parser = argparse.ArgumentParser(
        description="fNIRS MC — TD-Gated Analysis (2-min integration)")
    parser.add_argument("--data-dir", type=str, default="../results")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 80)
    print("  fNIRS MC Analysis — TD-Gated, 2-min Integration")
    print(f"  Laser: {LASER_POWER_W*1e3:.0f} mW | Det: {DET_EFFICIENCY*100:.0f}% QE | "
          f"Dark: {DARK_COUNT_RATE} cps")
    print("=" * 80)

    results = load_results(data_dir)
    tpsf_760 = load_tpsf(data_dir, "760nm")
    tpsf_850 = load_tpsf(data_dir, "850nm")

    for key, tpsf in [("760nm", tpsf_760), ("850nm", tpsf_850)]:
        if tpsf is not None:
            print(f"  TPSF {key}: {tpsf.shape}")

    td_sensitivity(results)
    gate_budget(results)
    best_configs = mbll_analysis(results)
    block_design(best_configs)
    time_sweep(results)
    safety_check()

    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    if best_configs:
        best = min(best_configs, key=lambda x: x[1]['min_hbo'])
        sds, cfg = best
        print(f"  Best: SDS={sds:.0f}mm, gate={GATE_LABELS[cfg['gate']]}")
        print(f"  Min detectable (120s): HbO={cfg['min_hbo']:.3f} µM, "
              f"HbR={cfg['min_hbr']:.3f} µM")
        if cfg['min_hbo'] < 1.0:
            print(f"  >> 1 µM HbO DETECTABLE with 2-min TD-gated fNIRS <<")
        elif cfg['min_hbo'] < 2.0:
            print(f"  >> Detectable with block averaging (20 trials) <<")
        else:
            print(f"  >> Marginal — need longer integration or more power <<")


if __name__ == "__main__":
    main()
