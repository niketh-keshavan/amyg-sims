#!/usr/bin/env python3
"""
Wavelength-pair optimizer for TD-fNIRS amygdala detection
---------------------------------------------------------
Tests all C(n,2) pairs from a multi-wavelength simulation and ranks them
by minimum detectable HbO/HbR concentration using dual-wavelength MBLL.

Also evaluates the best 3-wavelength and all-wavelength overdetermined systems
for comparison.

Usage:
    python optimize_wavelengths.py --data-dir ../data/all-wl
"""

import json
import argparse
import numpy as np
from pathlib import Path
from itertools import combinations

# ---------------------------------------------------------------------------
# Extinction coefficients: HbO2 and HHb  (Cope 1991 / Matcher 1995 / Prahl)
# Units: mM^-1 cm^-1  (we divide by 10 -> mM^-1 mm^-1 for pathlength in mm)
# ---------------------------------------------------------------------------
# Sources: Scott Prahl compilation, verified against Cope & Matcher tables
EXTINCTION = {
    #  nm:  (HbO2,   HHb)   in mM^-1 cm^-1
    690: (0.0618,  1.6456),
    730: (0.1348,  1.3080),
    770: (0.2682,  0.8784),
    780: (0.2874,  0.7998),
    810: (0.4170,  0.6918),   # near isosbestic
    830: (0.6406,  0.6314),   # isosbestic ~800-805
    850: (1.0507,  0.6918),
    940: (1.3702,  0.8163),
}

H_PLANCK = 6.626e-34
C_LIGHT  = 3e8

# ---------------------------------------------------------------------------
# System parameters (must match analyze.py)
# ---------------------------------------------------------------------------
LASER_POWER_W   = 1.0        # W average per wavelength
MEAS_TIME_S     = 120.0      # seconds
DARK_COUNT_RATE  = 1000       # cps per detector

# Si-PMT QE model: peak ~40% at 500nm, falls off in NIR
# Interpolated from typical Hamamatsu S-series MPPC specs
SIPMT_QE = {
    690: 0.35,
    730: 0.30,
    770: 0.25,
    780: 0.24,
    810: 0.20,
    830: 0.18,
    850: 0.15,
    940: 0.07,
}

GATE_LABELS = [
    "0-500ps", "0.5-1ns", "1-1.5ns", "1.5-2ns", "2-2.5ns",
    "2.5-3ns", "3-3.5ns", "3.5-4ns", "4-5ns", "5ns+"
]


def eps(wl_nm):
    """Return (eps_HbO, eps_HbR) in mM^-1 mm^-1 for a given wavelength."""
    e = EXTINCTION[wl_nm]
    return e[0] / 10.0, e[1] / 10.0


def photons_per_second(power_W, wavelength_nm):
    wl_m = wavelength_nm * 1e-9
    return power_W / (H_PLANCK * C_LIGHT / wl_m)


def load_all_results(data_dir, wavelengths):
    """Load results JSON for each wavelength. Returns dict {wl_nm: data}."""
    results = {}
    for wl in wavelengths:
        fpath = data_dir / f"results_{wl}nm.json"
        if fpath.exists():
            with open(fpath) as f:
                results[wl] = json.load(f)
    return results


def eval_pair(results, wl_a, wl_b):
    """
    Evaluate a wavelength pair using multi-detector, multi-gate weighted
    least squares MBLL. Returns best single-detector and multi-channel results.

    Returns dict with:
      - best_single: {sds, gate, min_hbo, min_hbr, N_a, N_b, L_a, L_b}
      - multi_channel: {min_hbo, min_hbr, n_measurements, dets_used}
    """
    ra, rb = results[wl_a], results[wl_b]
    e_hbo_a, e_hbr_a = eps(wl_a)
    e_hbo_b, e_hbr_b = eps(wl_b)

    qe_a, qe_b = SIPMT_QE[wl_a], SIPMT_QE[wl_b]
    nps_a = photons_per_second(LASER_POWER_W, wl_a)
    nps_b = photons_per_second(LASER_POWER_W, wl_b)
    num_sim_a, num_sim_b = ra["num_photons"], rb["num_photons"]
    scale_a = nps_a / num_sim_a * MEAS_TIME_S * qe_a
    scale_b = nps_b / num_sim_b * MEAS_TIME_S * qe_b
    dark = DARK_COUNT_RATE * MEAS_TIME_S

    n_dets = min(len(ra["detectors"]), len(rb["detectors"]))

    # --- Single-detector scan (primary direction only) ---
    best_single = None
    best_single_hbo = float('inf')

    # --- Multi-channel accumulator ---
    det_metrics = []  # (metric, det_idx, gate_data_list)

    for di in range(n_dets):
        da, db = ra["detectors"][di], rb["detectors"][di]
        if abs(da.get("angle_deg", 0)) > 1:
            continue
        sds = da["sds_mm"]

        gates_a = da.get("time_gates", [])
        gates_b = db.get("time_gates", [])
        n_gates = min(len(gates_a), len(gates_b))

        det_gate_data = []

        for gi in range(n_gates):
            ga, gb = gates_a[gi], gates_b[gi]
            La = ga.get("partial_pathlength_mm", {}).get("amygdala", 0)
            Lb = gb.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if La <= 0 or Lb <= 0:
                continue

            Na = ga.get("weight", 0) * scale_a + dark
            Nb = gb.get("weight", 0) * scale_b + dark
            if Na <= 0 or Nb <= 0:
                continue

            sigma_a = 1.0 / np.sqrt(Na)
            sigma_b = 1.0 / np.sqrt(Nb)

            # 2x2 MBLL
            E = np.array([
                [e_hbo_a * La, e_hbr_a * La],
                [e_hbo_b * Lb, e_hbr_b * Lb]
            ])
            det_E = np.linalg.det(E)
            if abs(det_E) < 1e-20:
                continue
            E_inv = np.linalg.inv(E)
            dc = np.abs(E_inv @ np.array([sigma_a, sigma_b])) * 1e3  # uM

            # Track best single detector
            if dc[0] < best_single_hbo:
                best_single_hbo = dc[0]
                best_single = {
                    'sds': sds, 'gate': gi, 'det_id': di,
                    'min_hbo': dc[0], 'min_hbr': dc[1],
                    'Na': Na, 'Nb': Nb, 'La': La, 'Lb': Lb,
                }

            det_gate_data.append({
                'gate': gi, 'La': La, 'Lb': Lb,
                'Na': Na, 'Nb': Nb,
                'sigma_a': sigma_a, 'sigma_b': sigma_b,
            })

        if det_gate_data:
            metric = sum(
                gd['La'] * np.sqrt(gd['Na']) + gd['Lb'] * np.sqrt(gd['Nb'])
                for gd in det_gate_data
            )
            det_metrics.append((metric, di, sds, det_gate_data))

    # --- Multi-channel: top 3 detectors, WLS ---
    det_metrics.sort(key=lambda x: x[0], reverse=True)
    top3 = det_metrics[:3]

    multi = None
    if top3:
        rows_A = []
        sigma_vec = []
        for _, di, sds, gate_data in top3:
            for gd in gate_data:
                rows_A.append([e_hbo_a * gd['La'], e_hbr_a * gd['La']])
                sigma_vec.append(gd['sigma_a'])
                rows_A.append([e_hbo_b * gd['Lb'], e_hbr_b * gd['Lb']])
                sigma_vec.append(gd['sigma_b'])

        A = np.array(rows_A)
        sigma = np.array(sigma_vec)
        W = np.diag(1.0 / sigma**2)
        AWA = A.T @ W @ A

        if abs(np.linalg.det(AWA)) > 1e-30:
            cov = np.linalg.inv(AWA)
            multi = {
                'min_hbo': np.sqrt(cov[0, 0]) * 1e3,
                'min_hbr': np.sqrt(cov[1, 1]) * 1e3,
                'n_measurements': len(sigma),
                'dets_used': [(di, sds) for _, di, sds, _ in top3],
                'corr': cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
            }

    return {'single': best_single, 'multi': multi}


def eval_multi_wavelength(results, wavelengths):
    """
    Evaluate an N-wavelength overdetermined system (N >= 2).
    Uses all wavelengths simultaneously in WLS across top 3 detectors.
    """
    n_wl = len(wavelengths)
    eps_hbo = np.array([eps(wl)[0] for wl in wavelengths])
    eps_hbr = np.array([eps(wl)[1] for wl in wavelengths])
    qe = np.array([SIPMT_QE[wl] for wl in wavelengths])
    nps = np.array([photons_per_second(LASER_POWER_W, wl) for wl in wavelengths])
    num_sim = np.array([results[wl]["num_photons"] for wl in wavelengths])
    scale = nps / num_sim * MEAS_TIME_S * qe
    dark = DARK_COUNT_RATE * MEAS_TIME_S

    n_dets = min(len(results[wl]["detectors"]) for wl in wavelengths)

    det_metrics = []

    for di in range(n_dets):
        dets = [results[wl]["detectors"][di] for wl in wavelengths]
        if abs(dets[0].get("angle_deg", 0)) > 1:
            continue
        sds = dets[0]["sds_mm"]

        n_gates = min(len(d.get("time_gates", [])) for d in dets)
        gate_data = []

        for gi in range(n_gates):
            gates = [d["time_gates"][gi] for d in dets]
            Ls = np.array([
                g.get("partial_pathlength_mm", {}).get("amygdala", 0)
                for g in gates
            ])
            if np.any(Ls <= 0):
                continue

            Ns = np.array([
                gates[wi].get("weight", 0) * scale[wi] + dark
                for wi in range(n_wl)
            ])
            if np.any(Ns <= 0):
                continue

            sigmas = 1.0 / np.sqrt(Ns)
            gate_data.append({'gate': gi, 'Ls': Ls, 'Ns': Ns, 'sigmas': sigmas})

        if gate_data:
            metric = sum(
                np.sum(gd['Ls'] * np.sqrt(gd['Ns'])) for gd in gate_data
            )
            det_metrics.append((metric, di, sds, gate_data))

    det_metrics.sort(key=lambda x: x[0], reverse=True)
    top3 = det_metrics[:3]

    if not top3:
        return None

    rows_A = []
    sigma_vec = []
    for _, di, sds, gate_data in top3:
        for gd in gate_data:
            for wi in range(n_wl):
                rows_A.append([eps_hbo[wi] * gd['Ls'][wi],
                               eps_hbr[wi] * gd['Ls'][wi]])
                sigma_vec.append(gd['sigmas'][wi])

    A = np.array(rows_A)
    sigma = np.array(sigma_vec)
    W = np.diag(1.0 / sigma**2)
    AWA = A.T @ W @ A

    if abs(np.linalg.det(AWA)) < 1e-30:
        return None

    cov = np.linalg.inv(AWA)
    return {
        'min_hbo': np.sqrt(cov[0, 0]) * 1e3,
        'min_hbr': np.sqrt(cov[1, 1]) * 1e3,
        'n_measurements': len(sigma),
        'dets_used': [(di, sds) for _, di, sds, _ in top3],
        'corr': cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]),
        'wavelengths': wavelengths,
    }


def condition_number(wl_a, wl_b):
    """Condition number of the 2x2 extinction matrix (geometry-independent)."""
    e_hbo_a, e_hbr_a = eps(wl_a)
    e_hbo_b, e_hbr_b = eps(wl_b)
    E = np.array([[e_hbo_a, e_hbr_a], [e_hbo_b, e_hbr_b]])
    return np.linalg.cond(E)


def main():
    parser = argparse.ArgumentParser(
        description="Wavelength-pair optimizer for TD-fNIRS amygdala detection")
    parser.add_argument("--data-dir", type=str, default="../data/all-wl",
                        help="Directory with results_XXXnm.json files")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Discover available wavelengths
    available = sorted([
        int(f.stem.replace("results_", "").replace("nm", ""))
        for f in data_dir.glob("results_*nm.json")
    ])

    if len(available) < 2:
        print(f"ERROR: need at least 2 wavelengths, found {available} in {data_dir}")
        return

    print("=" * 90)
    print("  WAVELENGTH-PAIR OPTIMIZER — TD-fNIRS Amygdala Detection")
    print(f"  Data: {data_dir}")
    print(f"  Available wavelengths: {available} nm")
    print(f"  System: {LASER_POWER_W*1e3:.0f}mW/wl | {MEAS_TIME_S:.0f}s | Si-PMT | {DARK_COUNT_RATE} cps dark")
    print("=" * 90)

    results = load_all_results(data_dir, available)
    loaded = sorted(results.keys())
    print(f"  Loaded: {loaded} ({len(loaded)} wavelengths)")

    for wl in loaded:
        r = results[wl]
        print(f"    {wl}nm: {r['num_photons']/1e9:.1f}B photons, "
              f"{len(r['detectors'])} detectors")

    # =======================================================================
    # PART 1: Extinction coefficient analysis
    # =======================================================================
    print("\n" + "=" * 90)
    print("1. EXTINCTION COEFFICIENTS & SPECTRAL CONTRAST")
    print("=" * 90)

    print(f"\n  {'WL':>5s}  {'e_HbO':>10s}  {'e_HbR':>10s}  {'ratio':>8s}  {'QE':>6s}")
    print("  " + "-" * 45)
    for wl in loaded:
        e_o, e_r = eps(wl)
        print(f"  {wl:5d}  {e_o:10.6f}  {e_r:10.6f}  {e_o/e_r:8.3f}  {SIPMT_QE[wl]*100:5.0f}%")

    # =======================================================================
    # PART 2: All pairs — condition number (geometry-free ranking)
    # =======================================================================
    print("\n" + "=" * 90)
    print("2. CONDITION NUMBER RANKING (lower = better spectral separation)")
    print("=" * 90)

    pairs = list(combinations(loaded, 2))
    cond_list = [(wl_a, wl_b, condition_number(wl_a, wl_b)) for wl_a, wl_b in pairs]
    cond_list.sort(key=lambda x: x[2])

    print(f"\n  {'Pair':>12s}  {'CondNum':>10s}  {'Separation':>12s}")
    print("  " + "-" * 38)
    for wl_a, wl_b, cn in cond_list:
        sep = abs(wl_b - wl_a)
        print(f"  {wl_a:4d}/{wl_b:<4d}  {cn:10.2f}  {sep:10d} nm")

    # =======================================================================
    # PART 3: All pairs — full simulation-based evaluation
    # =======================================================================
    print("\n" + "=" * 90)
    print("3. SIMULATION-BASED PAIR EVALUATION (multi-channel, 3 best detectors)")
    print("=" * 90)

    pair_results = []
    for wl_a, wl_b in pairs:
        res = eval_pair(results, wl_a, wl_b)
        pair_results.append((wl_a, wl_b, res))

    # Sort by multi-channel min_hbo
    pair_results.sort(
        key=lambda x: x[2]['multi']['min_hbo'] if x[2]['multi'] else float('inf')
    )

    print(f"\n  MULTI-CHANNEL (WLS, top 3 detectors, all valid gates):")
    print(f"  {'Rank':>4s}  {'Pair':>10s}  {'HbO[uM]':>10s}  {'HbR[uM]':>10s}  "
          f"{'Corr':>6s}  {'Nmsmt':>6s}  {'CondNum':>8s}")
    print("  " + "-" * 62)

    for rank, (wl_a, wl_b, res) in enumerate(pair_results):
        m = res['multi']
        if m is None:
            print(f"  {rank+1:4d}  {wl_a:4d}/{wl_b:<4d}  {'---':>10s}  {'---':>10s}")
            continue
        cn = condition_number(wl_a, wl_b)
        print(f"  {rank+1:4d}  {wl_a:4d}/{wl_b:<4d}  {m['min_hbo']:10.4f}  {m['min_hbr']:10.4f}  "
              f"{m['corr']:+6.3f}  {m['n_measurements']:6d}  {cn:8.2f}")

    print(f"\n  SINGLE-DETECTOR BEST (per pair):")
    # Re-sort by single-detector min_hbo
    pair_by_single = sorted(
        pair_results,
        key=lambda x: x[2]['single']['min_hbo'] if x[2]['single'] else float('inf')
    )
    print(f"  {'Rank':>4s}  {'Pair':>10s}  {'HbO[uM]':>10s}  {'HbR[uM]':>10s}  "
          f"{'SDS':>5s}  {'Gate':>8s}  {'CondNum':>8s}")
    print("  " + "-" * 62)

    for rank, (wl_a, wl_b, res) in enumerate(pair_by_single):
        s = res['single']
        if s is None:
            continue
        cn = condition_number(wl_a, wl_b)
        print(f"  {rank+1:4d}  {wl_a:4d}/{wl_b:<4d}  {s['min_hbo']:10.4f}  {s['min_hbr']:10.4f}  "
              f"{s['sds']:5.0f}  {GATE_LABELS[s['gate']]:>8s}  {cn:8.2f}")

    # =======================================================================
    # PART 4: Winner deep-dive
    # =======================================================================
    if pair_results and pair_results[0][2]['multi']:
        wl_a, wl_b, res = pair_results[0]
        m = res['multi']
        s = res['single']

        print("\n" + "=" * 90)
        print(f"4. WINNER DEEP-DIVE: {wl_a}/{wl_b} nm")
        print("=" * 90)

        print(f"\n  Multi-channel ({MEAS_TIME_S:.0f}s):")
        print(f"    Min detectable HbO: {m['min_hbo']:.4f} uM")
        print(f"    Min detectable HbR: {m['min_hbr']:.4f} uM")
        print(f"    HbO-HbR correlation: {m['corr']:+.3f}")
        print(f"    Measurements: {m['n_measurements']}")
        print(f"    Detectors: {m['dets_used']}")

        if s:
            print(f"\n  Best single detector:")
            print(f"    SDS={s['sds']:.0f}mm, gate={GATE_LABELS[s['gate']]}")
            print(f"    Min detectable HbO: {s['min_hbo']:.4f} uM")
            print(f"    Min detectable HbR: {s['min_hbr']:.4f} uM")
            print(f"    Amygdala pathlength: {wl_a}nm={s['La']:.6f}mm, "
                  f"{wl_b}nm={s['Lb']:.6f}mm")

        # Integration time sweep
        print(f"\n  Integration time sweep (multi-channel):")
        print(f"  {'Time':>8s}  {'HbO[uM]':>10s}  {'HbR[uM]':>10s}  {'Status':>12s}")
        print("  " + "-" * 45)
        for t in [1, 5, 10, 15, 30, 60, 120, 300]:
            hbo_t = m['min_hbo'] * np.sqrt(MEAS_TIME_S / t)
            hbr_t = m['min_hbr'] * np.sqrt(MEAS_TIME_S / t)
            label = f"{t}s" if t < 60 else f"{t//60}m"
            status = "OK" if hbo_t < 1.0 else "marginal" if hbo_t < 2.0 else "no"
            print(f"  {label:>8s}  {hbo_t:10.4f}  {hbr_t:10.4f}  {status:>12s}")

        # Block design
        print(f"\n  Block design (15s stim, 20 trials):")
        trial_noise_hbo = m['min_hbo'] * np.sqrt(MEAS_TIME_S / 15.0)
        avg_noise_hbo = trial_noise_hbo / np.sqrt(20)
        trial_noise_hbr = m['min_hbr'] * np.sqrt(MEAS_TIME_S / 15.0)
        avg_noise_hbr = trial_noise_hbr / np.sqrt(20)
        print(f"    Single-trial noise: HbO={trial_noise_hbo:.4f}, HbR={trial_noise_hbr:.4f} uM")
        print(f"    20-trial average:   HbO={avg_noise_hbo:.4f}, HbR={avg_noise_hbr:.4f} uM")
        for hbo_mag in [2.0, 3.0, 5.0]:
            snr = hbo_mag / avg_noise_hbo
            print(f"    HbO={hbo_mag:.0f}uM -> SNR={snr:.1f}")

    # =======================================================================
    # PART 5: Multi-wavelength comparison (3-best, all)
    # =======================================================================
    print("\n" + "=" * 90)
    print("5. MULTI-WAVELENGTH SYSTEMS (overdetermined)")
    print("=" * 90)

    # Best 3 wavelengths: try all combos
    if len(loaded) >= 3:
        triple_results = []
        for combo in combinations(loaded, 3):
            r = eval_multi_wavelength(results, list(combo))
            if r:
                triple_results.append(r)

        triple_results.sort(key=lambda x: x['min_hbo'])

        print(f"\n  BEST 3-WAVELENGTH COMBINATIONS (top 5):")
        print(f"  {'Rank':>4s}  {'Wavelengths':>20s}  {'HbO[uM]':>10s}  {'HbR[uM]':>10s}  "
              f"{'Corr':>6s}  {'Nmsmt':>6s}")
        print("  " + "-" * 62)
        for rank, r in enumerate(triple_results[:5]):
            wls = "/".join(str(w) for w in r['wavelengths'])
            print(f"  {rank+1:4d}  {wls:>20s}  {r['min_hbo']:10.4f}  {r['min_hbr']:10.4f}  "
                  f"{r['corr']:+6.3f}  {r['n_measurements']:6d}")

    # All wavelengths
    if len(loaded) >= 3:
        r_all = eval_multi_wavelength(results, loaded)
        if r_all:
            wls = "/".join(str(w) for w in loaded)
            print(f"\n  ALL {len(loaded)} WAVELENGTHS ({wls}):")
            print(f"    Min detectable HbO: {r_all['min_hbo']:.4f} uM")
            print(f"    Min detectable HbR: {r_all['min_hbr']:.4f} uM")
            print(f"    Correlation: {r_all['corr']:+.3f}")
            print(f"    Measurements: {r_all['n_measurements']}")

    # =======================================================================
    # PART 6: Summary / recommendation
    # =======================================================================
    print("\n" + "=" * 90)
    print("6. SUMMARY & RECOMMENDATION")
    print("=" * 90)

    if pair_results and pair_results[0][2]['multi']:
        wl_a, wl_b, res = pair_results[0]
        m = res['multi']
        print(f"\n  Best 2-wavelength pair: {wl_a}/{wl_b} nm")
        print(f"    HbO={m['min_hbo']:.4f} uM, HbR={m['min_hbr']:.4f} uM (120s, multi-ch)")
        cn = condition_number(wl_a, wl_b)
        print(f"    Condition number: {cn:.2f}")

    if len(loaded) >= 3 and triple_results:
        best3 = triple_results[0]
        wls = "/".join(str(w) for w in best3['wavelengths'])
        print(f"\n  Best 3-wavelength set: {wls} nm")
        print(f"    HbO={best3['min_hbo']:.4f} uM, HbR={best3['min_hbr']:.4f} uM")
        imp_hbo = m['min_hbo'] / best3['min_hbo'] if best3['min_hbo'] > 0 else 0
        imp_hbr = m['min_hbr'] / best3['min_hbr'] if best3['min_hbr'] > 0 else 0
        print(f"    Improvement over best pair: {imp_hbo:.2f}x HbO, {imp_hbr:.2f}x HbR")

    if len(loaded) >= 3 and r_all:
        print(f"\n  All {len(loaded)} wavelengths:")
        print(f"    HbO={r_all['min_hbo']:.4f} uM, HbR={r_all['min_hbr']:.4f} uM")
        if pair_results and pair_results[0][2]['multi']:
            imp_hbo = m['min_hbo'] / r_all['min_hbo'] if r_all['min_hbo'] > 0 else 0
            imp_hbr = m['min_hbr'] / r_all['min_hbr'] if r_all['min_hbr'] > 0 else 0
            print(f"    Improvement over best pair: {imp_hbo:.2f}x HbO, {imp_hbr:.2f}x HbR")


if __name__ == "__main__":
    main()
