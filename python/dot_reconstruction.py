#!/usr/bin/env python3
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path

EPSILON_HBR = np.array([1.3080, 0.6918]) / 10.0
EPSILON_HBO = np.array([0.1348, 1.0507]) / 10.0

H_PLANCK = 6.626e-34
C_LIGHT = 3e8
WAVELENGTHS_NM = np.array([730, 850])
WAVELENGTHS_M = WAVELENGTHS_NM * 1e-9
WL_KEYS = ["730nm", "850nm"]

TISSUE_LABELS = ["air", "scalp", "skull", "csf", "gray_matter", "white_matter", "amygdala"]
N_TISSUES = len(TISSUE_LABELS)

GATE_LABELS = [
    "0-500ps", "0.5-1ns", "1-1.5ns", "1.5-2ns", "2-2.5ns",
    "2.5-3ns", "3-3.5ns", "3.5-4ns", "4-5ns", "5ns+"
]

LASER_POWER_W = 1.0
MEAS_TIME_S = 120.0
DET_QE = {"730nm": 0.35, "850nm": 0.25}
DARK_COUNT_RATE = 1000
PHYSIO_NOISE_CW = 0.005
PHYSIO_NOISE_LATE_GATE = 0.002
SOURCE_STABILITY = 0.001


def photons_per_second(power_W, wavelength_m):
    return power_W / (H_PLANCK * C_LIGHT / wavelength_m)


def compute_noise(N_photons, gate_idx):
    sigma_shot = 1.0 / np.sqrt(N_photons) if N_photons > 0 else float('inf')
    if gate_idx >= 6:
        k_physio = PHYSIO_NOISE_LATE_GATE
    elif gate_idx >= 3:
        frac = (gate_idx - 3) / 3.0
        k_physio = PHYSIO_NOISE_CW * (1 - frac) + PHYSIO_NOISE_LATE_GATE * frac
    else:
        k_physio = PHYSIO_NOISE_CW
    return np.sqrt(sigma_shot**2 + k_physio**2 + SOURCE_STABILITY**2)


def load_results(data_dir):
    results = {}
    for wl in WL_KEYS:
        fpath = data_dir / f"results_{wl}.json"
        if not fpath.exists():
            continue
        with open(fpath) as f:
            results[wl] = json.load(f)
    return results


def load_multi_source_data(data_dirs):
    all_results = []
    for d in data_dirs:
        data_dir = Path(d)
        if not data_dir.exists():
            print(f"WARNING: {data_dir} does not exist, skipping")
            continue
        r = load_results(data_dir)
        if not r:
            print(f"WARNING: no results found in {data_dir}, skipping")
            continue
        all_results.append({'dir': data_dir, 'results': r})
    return all_results


def extract_pathlengths(gate):
    ppl = gate.get("partial_pathlength_mm", {})
    return np.array([ppl.get(t, 0.0) for t in TISSUE_LABELS])


def build_jacobian(all_source_data, wl_key, min_gate=5, max_gate=10):
    rows = []
    meas_info = []

    wl_idx = WL_KEYS.index(wl_key)
    qe = DET_QE[wl_key]
    N_per_sec = photons_per_second(LASER_POWER_W, WAVELENGTHS_M[wl_idx])

    for src_idx, src in enumerate(all_source_data):
        r = src['results'].get(wl_key)
        if r is None:
            continue
        num_sim = r["num_photons"]
        scale = N_per_sec / num_sim * MEAS_TIME_S * qe

        for det in r["detectors"]:
            gates = det.get("time_gates", [])
            for g_idx in range(min_gate, min(max_gate, len(gates))):
                gate = gates[g_idx]
                ppl = extract_pathlengths(gate)
                if ppl[TISSUE_LABELS.index("amygdala")] <= 0:
                    continue

                gw = gate.get("weight", 0)
                N_det = gw * scale + DARK_COUNT_RATE * MEAS_TIME_S
                noise = compute_noise(N_det, g_idx)

                rows.append(ppl)
                meas_info.append({
                    'src': src_idx, 'det_id': det['id'],
                    'sds': det['sds_mm'], 'gate': g_idx,
                    'N_det': N_det, 'noise': noise, 'weight': gw,
                })

    if not rows:
        return None, None
    return np.array(rows), meas_info


def whiten_system(J, noise_vec):
    W = 1.0 / noise_vec
    return J * W[:, np.newaxis], W


def column_scale(J):
    norms = np.linalg.norm(J, axis=0)
    norms[norms == 0] = 1.0
    return J / norms[np.newaxis, :], norms


def reconstruct_tikhonov(J, delta_y, lambda_reg, noise_vec=None):
    if noise_vec is not None:
        Jw, W = whiten_system(J, noise_vec)
        yw = delta_y * W
    else:
        Jw, yw = J, delta_y

    Jn, col_norms = column_scale(Jw)
    A = Jn.T @ Jn + lambda_reg * np.eye(Jn.shape[1])
    x_scaled = np.linalg.solve(A, Jn.T @ yw)
    return x_scaled / col_norms


def reconstruct_depth_weighted(J, delta_y, lambda_reg, noise_vec=None,
                                depth_weights=None):
    if depth_weights is None:
        depth_weights = np.array([0.01, 0.1, 0.3, 0.5, 1.0, 1.0, 2.0])

    if noise_vec is not None:
        Jw, W = whiten_system(J, noise_vec)
        yw = delta_y * W
    else:
        Jw, yw = J, delta_y

    D_inv = np.diag(1.0 / depth_weights)
    JD = Jw @ D_inv
    JDn, col_norms = column_scale(JD)
    A = JDn.T @ JDn + lambda_reg * np.eye(JDn.shape[1])
    x_scaled = np.linalg.solve(A, JDn.T @ yw)
    return D_inv @ (x_scaled / col_norms)


def estimator_covariance(J, lambda_reg, noise_vec):
    Jw, W = whiten_system(J, noise_vec)
    Jn, col_norms = column_scale(Jw)

    A = Jn.T @ Jn + lambda_reg * np.eye(Jn.shape[1])
    A_inv = np.linalg.inv(A)
    G = A_inv @ Jn.T
    cov_scaled = G @ G.T
    D = np.diag(1.0 / col_norms)
    return D @ cov_scaled @ D


def resolution_matrix(J, lambda_reg, noise_vec):
    Jw, W = whiten_system(J, noise_vec)
    Jn, col_norms = column_scale(Jw)

    A = Jn.T @ Jn + lambda_reg * np.eye(Jn.shape[1])
    A_inv = np.linalg.inv(A)
    R_scaled = A_inv @ (Jn.T @ Jn)
    D = np.diag(col_norms)
    D_inv = np.diag(1.0 / col_norms)
    return D_inv @ R_scaled @ D


def find_optimal_lambda_lcurve(J, delta_y, noise_vec=None, lambdas=None):
    if lambdas is None:
        lambdas = np.logspace(-6, 2, 80)

    residuals = np.zeros(len(lambdas))
    solutions = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        x = reconstruct_tikhonov(J, delta_y, lam, noise_vec=noise_vec)
        residuals[i] = np.linalg.norm(J @ x - delta_y)
        solutions[i] = np.linalg.norm(x)

    log_r = np.log10(residuals + 1e-30)
    log_s = np.log10(solutions + 1e-30)

    curvature = np.zeros(len(lambdas))
    for i in range(1, len(lambdas) - 1):
        dr = log_r[i+1] - log_r[i-1]
        ds = log_s[i+1] - log_s[i-1]
        d2r = log_r[i+1] - 2*log_r[i] + log_r[i-1]
        d2s = log_s[i+1] - 2*log_s[i] + log_s[i-1]
        num = dr * d2s - ds * d2r
        den = (dr**2 + ds**2)**1.5
        curvature[i] = num / den if den > 1e-30 else 0

    best_idx = np.argmax(curvature[1:-1]) + 1
    return lambdas[best_idx], lambdas, residuals, solutions, curvature


def compute_noise_vector(meas_info):
    return np.array([m['noise'] for m in meas_info])


def simulate_amygdala_activation(J, meas_info, delta_hbo_um=2.0, delta_hbr_um=-0.5,
                                  wl_idx=0):
    delta_mua = np.zeros(N_TISSUES)
    amyg_idx = TISSUE_LABELS.index("amygdala")
    delta_mua[amyg_idx] = (EPSILON_HBO[wl_idx] * delta_hbo_um * 1e-3 +
                            EPSILON_HBR[wl_idx] * delta_hbr_um * 1e-3)
    delta_y_clean = J @ delta_mua
    noise_vec = compute_noise_vector(meas_info)
    delta_y_noisy = delta_y_clean + noise_vec * np.random.randn(len(noise_vec))
    return delta_mua, delta_y_clean, delta_y_noisy, noise_vec


def dual_wavelength_recovery(J_730, J_850, meas_info_730, meas_info_850,
                              delta_hbo_um=2.0, delta_hbr_um=-0.5, lambda_reg=0.01):
    _, _, dy_noisy_730, noise_730 = simulate_amygdala_activation(
        J_730, meas_info_730, delta_hbo_um, delta_hbr_um, wl_idx=0)
    _, _, dy_noisy_850, noise_850 = simulate_amygdala_activation(
        J_850, meas_info_850, delta_hbo_um, delta_hbr_um, wl_idx=1)

    rec_mua_730 = reconstruct_tikhonov(J_730, dy_noisy_730, lambda_reg, noise_730)
    rec_mua_850 = reconstruct_tikhonov(J_850, dy_noisy_850, lambda_reg, noise_850)

    E_inv = np.linalg.inv(np.array([
        [EPSILON_HBO[0], EPSILON_HBR[0]],
        [EPSILON_HBO[1], EPSILON_HBR[1]]
    ]))

    rec_hbo = np.zeros(N_TISSUES)
    rec_hbr = np.zeros(N_TISSUES)
    for i in range(N_TISSUES):
        conc = E_inv @ np.array([rec_mua_730[i], rec_mua_850[i]])
        rec_hbo[i] = conc[0] * 1e3
        rec_hbr[i] = conc[1] * 1e3

    return {
        'rec_hbo': rec_hbo, 'rec_hbr': rec_hbr,
        'rec_mua_730': rec_mua_730, 'rec_mua_850': rec_mua_850,
        'true_hbo_um': delta_hbo_um, 'true_hbr_um': delta_hbr_um,
        'dy_noisy_730': dy_noisy_730, 'dy_noisy_850': dy_noisy_850,
        'noise_730': noise_730, 'noise_850': noise_850,
    }


def analytical_sensitivity(J_730, J_850, info_730, info_850, lambda_reg):
    noise_730 = compute_noise_vector(info_730)
    noise_850 = compute_noise_vector(info_850)

    cov_730 = estimator_covariance(J_730, lambda_reg, noise_730)
    cov_850 = estimator_covariance(J_850, lambda_reg, noise_850)

    R_730 = resolution_matrix(J_730, lambda_reg, noise_730)
    R_850 = resolution_matrix(J_850, lambda_reg, noise_850)

    E_inv = np.linalg.inv(np.array([
        [EPSILON_HBO[0], EPSILON_HBR[0]],
        [EPSILON_HBO[1], EPSILON_HBR[1]]
    ]))

    std_mua_730 = np.sqrt(np.diag(cov_730))
    std_mua_850 = np.sqrt(np.diag(cov_850))

    std_hbo = np.zeros(N_TISSUES)
    std_hbr = np.zeros(N_TISSUES)
    for i in range(N_TISSUES):
        cov_conc = E_inv @ np.array([[cov_730[i, i], 0], [0, cov_850[i, i]]]) @ E_inv.T
        std_hbo[i] = np.sqrt(cov_conc[0, 0]) * 1e3
        std_hbr[i] = np.sqrt(cov_conc[1, 1]) * 1e3

    return {
        'std_mua_730': std_mua_730, 'std_mua_850': std_mua_850,
        'std_hbo': std_hbo, 'std_hbr': std_hbr,
        'R_730': R_730, 'R_850': R_850,
        'cov_730': cov_730, 'cov_850': cov_850,
    }


def snr_analysis_analytical(J_730, J_850, info_730, info_850, lambda_reg):
    amyg_idx = TISSUE_LABELS.index("amygdala")
    sens = analytical_sensitivity(J_730, J_850, info_730, info_850, lambda_reg)

    print("\n" + "=" * 80)
    print("DOT ANALYTICAL SENSITIVITY")
    print(f"  lambda={lambda_reg:.2e}")
    print("=" * 80)

    print(f"\n  Noise floor per tissue (1-sigma, {MEAS_TIME_S:.0f}s integration):")
    print(f"  {'Tissue':<15s}  {'std(mua_730)':>12s}  {'std(mua_850)':>12s}  "
          f"{'std(HbO) uM':>12s}  {'std(HbR) uM':>12s}")
    print("  " + "-" * 65)
    for i, t in enumerate(TISSUE_LABELS):
        print(f"  {t:<15s}  {sens['std_mua_730'][i]:12.2e}  {sens['std_mua_850'][i]:12.2e}  "
              f"{sens['std_hbo'][i]:12.4f}  {sens['std_hbr'][i]:12.4f}")

    amyg_std_hbo = sens['std_hbo'][amyg_idx]
    amyg_std_hbr = sens['std_hbr'][amyg_idx]
    min_hbo_2sigma = 2.0 * amyg_std_hbo
    min_hbr_2sigma = 2.0 * amyg_std_hbr

    print(f"\n  Amygdala min detectable (2-sigma):")
    print(f"    dHbO: {min_hbo_2sigma:.4f} uM")
    print(f"    dHbR: {min_hbr_2sigma:.4f} uM")

    print(f"\n  Resolution matrix diagonal (self-sensitivity):")
    print(f"  {'Tissue':<15s}  {'R_730':>8s}  {'R_850':>8s}")
    print("  " + "-" * 34)
    for i, t in enumerate(TISSUE_LABELS):
        print(f"  {t:<15s}  {sens['R_730'][i,i]:8.4f}  {sens['R_850'][i,i]:8.4f}")

    amyg_R730 = sens['R_730'][amyg_idx, amyg_idx]
    amyg_R850 = sens['R_850'][amyg_idx, amyg_idx]
    print(f"\n  Amygdala resolution: R_730={amyg_R730:.4f}, R_850={amyg_R850:.4f}")
    if amyg_R730 < 0.01:
        print("  WARNING: Very low amygdala resolution -- reconstruction will be dominated")
        print("  by cross-talk from other tissues. Multiple source positions may help.")

    return sens, min_hbo_2sigma


def snr_mc_validation(J_730, J_850, info_730, info_850, lambda_reg, n_trials=50):
    amyg_idx = TISSUE_LABELS.index("amygdala")
    test_hbo_values = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0]

    print("\n" + "=" * 80)
    print("DOT MC VALIDATION (noisy reconstruction)")
    print(f"  {n_trials} noise realizations, lambda={lambda_reg:.2e}")
    print("=" * 80)

    print(f"\n  {'dHbO_true':>10s}  {'mean_amyg':>10s}  {'std_amyg':>10s}  "
          f"{'bias':>8s}  {'SNR':>8s}  {'detected':>10s}")
    print("  " + "-" * 65)

    min_detectable = None

    for hbo in test_hbo_values:
        rec_amyg_vals = []
        for _ in range(n_trials):
            result = dual_wavelength_recovery(
                J_730, J_850, info_730, info_850,
                delta_hbo_um=hbo, delta_hbr_um=-hbo * 0.25,
                lambda_reg=lambda_reg)
            rec_amyg_vals.append(result['rec_hbo'][amyg_idx])

        mean_rec = np.mean(rec_amyg_vals)
        std_rec = np.std(rec_amyg_vals)
        bias = mean_rec - hbo
        snr = abs(mean_rec) / std_rec if std_rec > 0 else 0
        detected = "YES" if snr >= 2.0 else "no"

        if snr >= 2.0 and min_detectable is None:
            min_detectable = hbo

        print(f"  {hbo:10.1f}  {mean_rec:10.3f}  {std_rec:10.3f}  "
              f"{bias:8.3f}  {snr:8.2f}  {detected:>10s}")

    if min_detectable is None:
        min_detectable = float('inf')

    print(f"\n  Min detectable dHbO (SNR>=2): {min_detectable:.1f} uM")
    return min_detectable


def lambda_sweep(J_730, J_850, info_730, info_850):
    amyg_idx = TISSUE_LABELS.index("amygdala")
    lambdas = np.logspace(-4, 2, 30)

    print("\n" + "=" * 80)
    print("REGULARIZATION SWEEP")
    print("=" * 80)

    print(f"\n  {'lambda':>10s}  {'std_HbO_amyg':>14s}  {'R_amyg_730':>12s}  "
          f"{'R_amyg_850':>12s}  {'bias_factor':>12s}")
    print("  " + "-" * 65)

    best_lambda = None
    best_std = float('inf')

    for lam in lambdas:
        sens = analytical_sensitivity(J_730, J_850, info_730, info_850, lam)
        std_hbo = sens['std_hbo'][amyg_idx]
        r730 = sens['R_730'][amyg_idx, amyg_idx]
        r850 = sens['R_850'][amyg_idx, amyg_idx]
        bias_factor = 1.0 - 0.5 * (r730 + r850)

        effective = std_hbo / max(0.5 * (r730 + r850), 1e-10)

        if effective < best_std and r730 > 0.001:
            best_std = effective
            best_lambda = lam

        print(f"  {lam:10.2e}  {std_hbo:14.4f}  {r730:12.6f}  "
              f"{r850:12.6f}  {bias_factor:12.4f}")

    if best_lambda is not None:
        print(f"\n  Best lambda (min effective noise): {best_lambda:.2e}")
    return best_lambda


def comparison_table(dot_min_hbo):
    print("\n" + "=" * 80)
    print("METHOD COMPARISON")
    print("=" * 80)

    cw_mbll = 50.0
    td_gated = 5.0
    ssr = 20.0
    td_ssr = 3.0

    print("  (CW/TD/SSR values are approximate -- run analyze.py for precise results)")

    rows = [
        ("CW MBLL", cw_mbll, 1.0),
        ("TD gating (best gate)", td_gated, cw_mbll / td_gated),
        ("SSR (CW)", ssr, cw_mbll / ssr),
        ("TD + SSR", td_ssr, cw_mbll / td_ssr),
        ("DOT (this script)", dot_min_hbo,
         cw_mbll / dot_min_hbo if dot_min_hbo > 0 and dot_min_hbo < float('inf') else 0),
    ]

    print(f"\n  {'Method':<25s}  {'Min dHbO (uM)':>14s}  {'vs CW':>10s}")
    print("  " + "-" * 52)
    for name, val, imp in rows:
        val_str = f"{val:14.2f}" if val < float('inf') else "           inf"
        imp_str = f"{imp:9.1f}x" if imp > 0 else "       N/A"
        print(f"  {name:<25s}  {val_str}  {imp_str}")


def plot_jacobian_sensitivity(J_730, J_850, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, J, wl in zip(axes, [J_730, J_850], ["730nm", "850nm"]):
        col_norms = np.linalg.norm(J, axis=0)
        col_norms_safe = np.where(col_norms > 0, col_norms, 1e-10)
        colors_val = np.log10(col_norms_safe)
        colors_val = (colors_val - colors_val.min()) / (colors_val.max() - colors_val.min() + 1e-10)
        colors = plt.cm.RdYlBu_r(colors_val)
        ax.bar(range(N_TISSUES), col_norms_safe, color=colors)
        ax.set_xticks(range(N_TISSUES))
        ax.set_xticklabels(TISSUE_LABELS, rotation=45, ha='right')
        ax.set_ylabel("Sensitivity (||J column||)")
        ax.set_title(f"Jacobian Sensitivity - {wl}")
        ax.set_yscale('log')

    plt.tight_layout()
    fpath = output_dir / "dot_jacobian_sensitivity.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


def plot_resolution_matrix(R_730, R_850, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, R, wl in zip(axes, [R_730, R_850], ["730nm", "850nm"]):
        im = ax.imshow(np.abs(R), cmap='hot', aspect='equal')
        ax.set_xticks(range(N_TISSUES))
        ax.set_xticklabels(TISSUE_LABELS, rotation=45, ha='right')
        ax.set_yticks(range(N_TISSUES))
        ax.set_yticklabels(TISSUE_LABELS)
        ax.set_title(f"Resolution Matrix |R| - {wl}")
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fpath = output_dir / "dot_resolution_matrix.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


def plot_reconstruction(result, output_dir):
    amyg_idx = TISSUE_LABELS.index("amygdala")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, label, true_val in [
        (axes[0], 'rec_hbo', 'HbO', result['true_hbo_um']),
        (axes[1], 'rec_hbr', 'HbR', result['true_hbr_um']),
    ]:
        vals = result[key]
        true_arr = np.zeros(N_TISSUES)
        true_arr[amyg_idx] = true_val

        x = np.arange(N_TISSUES)
        w = 0.35
        ax.bar(x - w/2, true_arr, w, label='True', alpha=0.7, color='steelblue')
        ax.bar(x + w/2, vals, w, label='Reconstructed', alpha=0.7, color='coral')
        ax.set_xticks(x)
        ax.set_xticklabels(TISSUE_LABELS, rotation=45, ha='right')
        ax.set_ylabel(f"d{label} (uM)")
        ax.set_title(f"DOT Reconstruction: {label}")
        ax.legend()
        ax.axhline(y=0, color='gray', linewidth=0.5)

    plt.tight_layout()
    fpath = output_dir / "dot_reconstruction.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


def plot_lcurve(lambdas, residuals, solutions, curvature, best_lambda, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].loglog(residuals, solutions, 'b.-')
    best_idx = np.argmin(np.abs(lambdas - best_lambda))
    axes[0].plot(residuals[best_idx], solutions[best_idx], 'ro', markersize=10,
                 label=f'opt lambda={best_lambda:.2e}')
    axes[0].set_xlabel('||Jx - y||')
    axes[0].set_ylabel('||x||')
    axes[0].set_title('L-Curve')
    axes[0].legend()

    axes[1].semilogx(lambdas[1:-1], curvature[1:-1], 'g.-')
    axes[1].axvline(best_lambda, color='r', linestyle='--', label=f'opt lambda={best_lambda:.2e}')
    axes[1].set_xlabel('lambda')
    axes[1].set_ylabel('Curvature')
    axes[1].set_title('L-Curve Curvature')
    axes[1].legend()

    plt.tight_layout()
    fpath = output_dir / "dot_lcurve.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


def plot_comparison_bar(dot_min_hbo, output_dir):
    methods = ["CW MBLL", "TD gating", "SSR (CW)", "TD+SSR", "DOT"]
    values = [50.0, 5.0, 20.0, 3.0, min(dot_min_hbo, 100.0)]
    colors = ['#d62728', '#ff7f0e', '#bcbd22', '#2ca02c', '#1f77b4']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Min Detectable dHbO (uM)')
    ax.set_title('Method Comparison: Amygdala Sensitivity')
    ax.set_yscale('log')
    ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.7, label='Target (2 uM)')
    ax.legend()

    for bar, val in zip(bars, values):
        label = f'{val:.1f}' if val < 100 else 'inf'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                label, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    fpath = output_dir / "dot_method_comparison.png"
    plt.savefig(fpath, dpi=150)
    plt.close()
    print(f"  Saved: {fpath}")


def main():
    parser = argparse.ArgumentParser(description="DOT Reconstruction from MC data")
    parser.add_argument("--data-dirs", nargs='+', required=True)
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--lambda-reg", type=float, default=None,
                        help="Regularization param (auto via sweep if not set)")
    parser.add_argument("--min-gate", type=int, default=5)
    parser.add_argument("--max-gate", type=int, default=10)
    parser.add_argument("--snr-trials", type=int, default=50)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("  DOT RECONSTRUCTION FROM MONTE CARLO DATA")
    print(f"  Sources: {len(args.data_dirs)} directories")
    print(f"  Gates: {args.min_gate} to {args.max_gate-1}")
    print("=" * 80)

    all_source_data = load_multi_source_data(args.data_dirs)
    if not all_source_data:
        print("ERROR: No valid source data found")
        sys.exit(1)

    print(f"\n  Loaded {len(all_source_data)} source position(s)")
    for i, src in enumerate(all_source_data):
        wls = list(src['results'].keys())
        print(f"    Source {i}: {src['dir']} ({', '.join(wls)})")

    print("\n  Building Jacobian matrices...")
    J_730, info_730 = build_jacobian(all_source_data, "730nm",
                                      min_gate=args.min_gate, max_gate=args.max_gate)
    J_850, info_850 = build_jacobian(all_source_data, "850nm",
                                      min_gate=args.min_gate, max_gate=args.max_gate)

    if J_730 is None or J_850 is None:
        print("ERROR: Could not build Jacobian (no measurements with amygdala signal)")
        print("  Try lowering --min-gate or checking data directories")
        sys.exit(1)

    print(f"  J_730: {J_730.shape} ({J_730.shape[0]} meas x {J_730.shape[1]} tissues)")
    print(f"  J_850: {J_850.shape} ({J_850.shape[0]} meas x {J_850.shape[1]} tissues)")

    print(f"\n  Jacobian column norms (sensitivity per tissue):")
    print(f"  {'Tissue':<15s}  {'730nm':>10s}  {'850nm':>10s}  {'ratio_to_amyg':>14s}")
    print("  " + "-" * 52)
    amyg_idx = TISSUE_LABELS.index("amygdala")
    n730_amyg = np.linalg.norm(J_730[:, amyg_idx])
    n850_amyg = np.linalg.norm(J_850[:, amyg_idx])
    for i, t in enumerate(TISSUE_LABELS):
        n730 = np.linalg.norm(J_730[:, i])
        n850 = np.linalg.norm(J_850[:, i])
        ratio = n730 / n730_amyg if n730_amyg > 0 else 0
        print(f"  {t:<15s}  {n730:10.4f}  {n850:10.4f}  {ratio:14.0f}x")

    cond_730 = np.linalg.cond(J_730)
    cond_850 = np.linalg.cond(J_850)
    print(f"\n  Condition number: J_730={cond_730:.1e}, J_850={cond_850:.1e}")

    # SVD analysis
    print(f"\n  SVD analysis (J_730):")
    U, S, Vt = np.linalg.svd(J_730, full_matrices=False)
    print(f"  Singular values: {', '.join(f'{s:.2e}' for s in S)}")
    print(f"  Amygdala participation in singular vectors (|V_amyg|):")
    for k in range(min(len(S), N_TISSUES)):
        print(f"    SV {k}: sigma={S[k]:.2e}, |V_amyg|={abs(Vt[k, amyg_idx]):.6f}")

    # Lambda sweep
    best_lambda = lambda_sweep(J_730, J_850, info_730, info_850)
    lambda_reg = args.lambda_reg if args.lambda_reg is not None else best_lambda
    if lambda_reg is None:
        lambda_reg = 0.01

    # Analytical sensitivity
    sens, min_hbo_analytical = snr_analysis_analytical(
        J_730, J_850, info_730, info_850, lambda_reg)

    # L-curve
    print("\n  Computing L-curve...")
    _, dy_clean, _, noise_lc = simulate_amygdala_activation(
        J_730, info_730, delta_hbo_um=2.0, wl_idx=0)
    lc_lambda, lc_lambdas, lc_res, lc_sol, lc_curv = find_optimal_lambda_lcurve(
        J_730, dy_clean, noise_vec=noise_lc)
    print(f"  L-curve optimal lambda: {lc_lambda:.2e}")

    # MC validation
    np.random.seed(42)
    dot_min_hbo = snr_mc_validation(J_730, J_850, info_730, info_850,
                                     lambda_reg, n_trials=args.snr_trials)

    # Single reconstruction for visualization
    np.random.seed(42)
    result = dual_wavelength_recovery(
        J_730, J_850, info_730, info_850,
        delta_hbo_um=2.0, delta_hbr_um=-0.5, lambda_reg=lambda_reg)

    print(f"\n  Sample reconstruction (dHbO=2uM, dHbR=-0.5uM, lambda={lambda_reg:.2e}):")
    print(f"  {'Tissue':<15s}  {'True HbO':>10s}  {'Rec HbO':>10s}  "
          f"{'True HbR':>10s}  {'Rec HbR':>10s}")
    print("  " + "-" * 60)
    for i, t in enumerate(TISSUE_LABELS):
        true_hbo = 2.0 if i == amyg_idx else 0.0
        true_hbr = -0.5 if i == amyg_idx else 0.0
        print(f"  {t:<15s}  {true_hbo:10.3f}  {result['rec_hbo'][i]:10.3f}  "
              f"{true_hbr:10.3f}  {result['rec_hbr'][i]:10.3f}")

    # Comparison
    comparison_table(dot_min_hbo)

    # Figures
    print("\n  Generating figures...")
    plot_jacobian_sensitivity(J_730, J_850, output_dir)
    plot_resolution_matrix(sens['R_730'], sens['R_850'], output_dir)
    plot_reconstruction(result, output_dir)
    plot_lcurve(lc_lambdas, lc_res, lc_sol, lc_curv, lc_lambda, output_dir)
    plot_comparison_bar(dot_min_hbo, output_dir)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Sources: {len(all_source_data)}")
    print(f"  Measurements: {J_730.shape[0]} (730nm), {J_850.shape[0]} (850nm)")
    print(f"  Regularization: lambda={lambda_reg:.2e}")
    print(f"  Amygdala resolution: R_730={sens['R_730'][amyg_idx,amyg_idx]:.6f}, "
          f"R_850={sens['R_850'][amyg_idx,amyg_idx]:.6f}")
    print(f"  Analytical min dHbO (2-sigma): {min_hbo_analytical:.4f} uM")
    print(f"  MC validation min dHbO (SNR>=2): {dot_min_hbo}")
    if min_hbo_analytical > 10:
        print(f"\n  NOTE: Amygdala sensitivity is very low with {len(all_source_data)} source(s).")
        print(f"  The 7-tissue DOT problem is ill-conditioned when amygdala pathlengths")
        print(f"  are ~{n730_amyg/np.sqrt(J_730.shape[0]):.4f} mm vs scalp ~{np.linalg.norm(J_730[:,1])/np.sqrt(J_730.shape[0]):.0f} mm.")
        print(f"  Consider: more source positions, constrained reconstruction,")
        print(f"  or spatial priors from anatomical data.")

    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
