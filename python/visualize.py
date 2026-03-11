#!/usr/bin/env python3
"""
Visualization for fNIRS Monte Carlo TD/FD/Chirp simulation results.

Generates:
  1. Tissue volume cross-sections
  2. Fluence maps (log scale)
  3. CW sensitivity by angle direction
  4. TPSF curves for different SDS
  5. Time-gated sensitivity heatmap
  6. Frequency-domain phase & amplitude
  7. SNR comparison (CW vs TD vs Chirp)
  8. Minimum detectable concentration

Usage:
    python visualize.py --data-dir ../results --output-dir ../figures
"""

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LogNorm
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TISSUE_COLORS = [
    [0.0, 0.0, 0.0, 0.0],   # 0: air
    [0.9, 0.7, 0.6, 1.0],   # 1: scalp
    [0.95, 0.95, 0.85, 1.0], # 2: skull
    [0.6, 0.8, 1.0, 1.0],   # 3: CSF
    [0.8, 0.5, 0.5, 1.0],   # 4: gray matter
    [0.95, 0.95, 0.95, 1.0], # 5: white matter
    [1.0, 0.2, 0.2, 1.0],   # 6: amygdala
]
TISSUE_CMAP = ListedColormap(TISSUE_COLORS)
TISSUE_NAMES = ["Air", "Scalp", "Skull", "CSF", "Gray Matter",
                "White Matter", "Amygdala"]

GATE_LABELS = ["0–500 ps", "0.5–1 ns", "1–1.5 ns",
               "1.5–2.5 ns", "2.5–4 ns", "4+ ns"]

EPSILON_HBR = np.array([1.1058, 0.6918]) / 10.0   # [1/(mM·mm)]
EPSILON_HBO = np.array([0.1496, 1.0507]) / 10.0
H_PLANCK = 6.626e-34
C_LIGHT = 3e8

ANGLE_COLORS = {0: "#2196F3", 30: "#4CAF50", -30: "#FF9800", 60: "#9C27B0", -60: "#F44336"}
ANGLE_MARKERS = {0: "o", 30: "s", -30: "D", 60: "^", -60: "v"}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_data(data_dir):
    meta_path = data_dir / "volume_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    nx, ny, nz = meta["nx"], meta["ny"], meta["nz"]

    vol_path = data_dir / "volume.bin"
    if vol_path.exists():
        vol = np.fromfile(vol_path, dtype=np.uint8).reshape((nz, ny, nx))
    else:
        vol = None  # Large grid — volume.bin skipped

    fluence = {}
    for wl in ["760nm", "850nm"]:
        fp = data_dir / f"fluence_{wl}.bin"
        if fp.exists():
            fluence[wl] = np.fromfile(fp, dtype=np.float32).reshape((nz, ny, nx))

    results = {}
    for wl in ["760nm", "850nm"]:
        fp = data_dir / f"results_{wl}.json"
        if fp.exists():
            with open(fp) as f:
                results[wl] = json.load(f)

    tpsf = {}
    for wl in ["760nm", "850nm"]:
        fp = data_dir / f"tpsf_{wl}.bin"
        if fp.exists():
            raw = np.fromfile(fp, dtype=np.float64)
            tpsf[wl] = raw.reshape(len(raw) // 512, 512)

    # Photon paths
    paths = {}
    MAX_PATH_STEPS = 2048
    for wl in ["760nm", "850nm"]:
        meta_fp = data_dir / f"paths_meta_{wl}.bin"
        pos_fp = data_dir / f"paths_pos_{wl}.bin"
        if meta_fp.exists() and pos_fp.exists():
            raw_meta = np.fromfile(meta_fp, dtype=np.int32)
            num_paths = len(raw_meta) // 2
            det_ids = raw_meta[0::2]
            path_lens = raw_meta[1::2]
            raw_pos = np.fromfile(pos_fp, dtype=np.float32)
            positions = raw_pos.reshape(num_paths, MAX_PATH_STEPS, 3)
            # Filter out invalidated paths (path_len == 0)
            valid = path_lens > 0
            paths[wl] = {
                "det_ids": det_ids[valid],
                "path_lens": path_lens[valid],
                "positions": positions[valid],
            }
            print(f"  Loaded {valid.sum()}/{num_paths} valid paths for {wl}")

    return vol, fluence, results, tpsf, meta, paths


def _get_dets_by_angle(results, wl_key):
    """Group detectors by angle. Returns dict angle -> sorted list of dets."""
    groups = {}
    for det in results[wl_key]["detectors"]:
        ang = int(round(det.get("angle_deg", 0)))
        groups.setdefault(ang, []).append(det)
    for ang in groups:
        groups[ang].sort(key=lambda d: d["sds_mm"])
    return groups


# ---------------------------------------------------------------------------
# 1. Tissue slices
# ---------------------------------------------------------------------------
def plot_tissue_slices(vol, meta, output_dir):
    dx = meta["dx"]
    nz, ny, nx = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    z_amyg = nz // 2 - int(15.0 / dx)
    im = axes[0].imshow(vol[z_amyg], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                        extent=[0, nx*dx, ny*dx, 0], interpolation="nearest")
    axes[0].set_title(f"Axial  z = {z_amyg * dx:.0f} mm (amygdala level)")
    axes[0].set_xlabel("X [mm]"); axes[0].set_ylabel("Y [mm]")

    y_amyg = ny // 2 + int(5.0 / dx)
    axes[1].imshow(vol[:, y_amyg, :], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                   extent=[0, nx*dx, nz*dx, 0], interpolation="nearest")
    axes[1].set_title(f"Coronal  y = {y_amyg * dx:.0f} mm")
    axes[1].set_xlabel("X [mm]"); axes[1].set_ylabel("Z [mm]")

    x_amyg = nx // 2 + int(24.0 / dx)
    axes[2].imshow(vol[:, :, x_amyg], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                   extent=[0, ny*dx, nz*dx, 0], interpolation="nearest")
    axes[2].set_title(f"Sagittal  x = {x_amyg * dx:.0f} mm (R amygdala)")
    axes[2].set_xlabel("Y [mm]"); axes[2].set_ylabel("Z [mm]")

    cbar = fig.colorbar(im, ax=axes, ticks=range(7), shrink=0.8)
    cbar.ax.set_yticklabels(TISSUE_NAMES)
    plt.tight_layout()
    plt.savefig(output_dir / "tissue_slices.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved tissue_slices.png")


# ---------------------------------------------------------------------------
# 2. Fluence maps
# ---------------------------------------------------------------------------
def plot_fluence(vol, fluence, meta, output_dir):
    dx = meta["dx"]
    nz, ny, nx = vol.shape

    for wl_key, flu in fluence.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_amyg = nx // 2 + int(24.0 / dx)
        z_amyg = nz // 2 - int(15.0 / dx)

        slices = [
            (vol[:, :, x_amyg], flu[:, :, x_amyg],
             f"Sagittal x={x_amyg*dx:.0f} mm", [0, ny*dx, nz*dx, 0], "Y [mm]", "Z [mm]"),
            (vol[z_amyg], flu[z_amyg],
             "Axial (amygdala level)", [0, nx*dx, ny*dx, 0], "X [mm]", "Y [mm]"),
        ]
        for ax, (ts, fs, title, ext, xl, yl) in zip(axes, slices):
            ax.imshow(ts, cmap=TISSUE_CMAP, vmin=0, vmax=6,
                      extent=ext, interpolation="nearest", alpha=0.3)
            fm = np.ma.masked_where(fs <= 0, fs)
            if fm.count() > 0:
                im = ax.imshow(fm, cmap="hot", norm=LogNorm(vmin=fm.min(), vmax=fm.max()),
                               extent=ext, interpolation="bilinear", alpha=0.7)
                fig.colorbar(im, ax=ax, label="Fluence [a.u.]", shrink=0.8)
            ax.set_title(f"Fluence {wl_key} — {title}")
            ax.set_xlabel(xl); ax.set_ylabel(yl)

        plt.tight_layout()
        plt.savefig(output_dir / f"fluence_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved fluence_{wl_key}.png")


# ---------------------------------------------------------------------------
# 3. CW sensitivity by angle direction
# ---------------------------------------------------------------------------
def plot_cw_sensitivity(results, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for wl_key, ls in [("760nm", "-"), ("850nm", "--")]:
        groups = _get_dets_by_angle(results, wl_key)
        for ang, dets in sorted(groups.items()):
            if len(dets) < 2:
                continue
            sds = [d["sds_mm"] for d in dets]
            weights = [d["total_weight"] for d in dets]
            amyg_pl = [d["partial_pathlength_mm"]["amygdala"] for d in dets]
            sens = [a / d["mean_pathlength_mm"] if d["mean_pathlength_mm"] > 0 else 0
                    for a, d in zip(amyg_pl, dets)]

            color = ANGLE_COLORS.get(ang, "gray")
            marker = ANGLE_MARKERS.get(ang, "x")
            label = f"{wl_key} {ang:+d}°" if wl_key == "760nm" else None

            axes[0].semilogy(sds, weights, ls=ls, marker=marker, color=color,
                             label=label, markersize=5, linewidth=1.2)
            axes[1].plot(sds, amyg_pl, ls=ls, marker=marker, color=color,
                         label=label, markersize=5, linewidth=1.2)
            axes[2].plot(sds, [s * 100 for s in sens], ls=ls, marker=marker,
                         color=color, label=label, markersize=5, linewidth=1.2)

    axes[0].set_ylabel("Detected Weight"); axes[0].set_title("Signal Intensity")
    axes[1].set_ylabel("Amygdala Partial PL [mm]"); axes[1].set_title("Amygdala Pathlength")
    axes[2].set_ylabel("Sensitivity [%]"); axes[2].set_title("Amygdala Sensitivity")

    for ax in axes:
        ax.set_xlabel("Source-Detector Separation [mm]")
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, ncol=2, loc="upper right")
    fig.suptitle("CW Analysis — HD Array (solid = 760 nm, dashed = 850 nm)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "cw_sensitivity_by_angle.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved cw_sensitivity_by_angle.png")


# ---------------------------------------------------------------------------
# 4. TPSF curves
# ---------------------------------------------------------------------------
def plot_tpsf(results, tpsf, output_dir):
    if "760nm" not in tpsf:
        return
    bin_ps = 10.0
    time_ns = np.arange(512) * bin_ps / 1000.0  # ns

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax, wl_key in zip(axes, ["760nm", "850nm"]):
        if wl_key not in tpsf:
            continue
        r = results[wl_key]
        data = tpsf[wl_key]
        cmap = plt.cm.viridis

        primary_dets = [(i, d) for i, d in enumerate(r["detectors"])
                        if abs(d.get("angle_deg", 0)) < 1 and i < data.shape[0]]

        sds_vals = [d["sds_mm"] for _, d in primary_dets]
        sds_min, sds_max = min(sds_vals), max(sds_vals)

        for idx, det in primary_dets:
            t = data[idx]
            if t.sum() <= 0:
                continue
            normed = t / t.max()
            frac = (det["sds_mm"] - sds_min) / (sds_max - sds_min + 1e-9)
            ax.plot(time_ns, normed, color=cmap(frac), linewidth=1.5,
                    label=f"SDS={det['sds_mm']:.0f} mm")

        ax.set_xlabel("Time of Flight [ns]")
        ax.set_ylabel("Normalized TPSF")
        ax.set_title(f"TPSF — {wl_key} (primary direction)")
        ax.set_xlim(0, 3.0)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "tpsf_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved tpsf_curves.png")


# ---------------------------------------------------------------------------
# 5. Time-gated sensitivity heatmap
# ---------------------------------------------------------------------------
def plot_time_gated_sensitivity(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, wl_key in zip(axes, ["760nm", "850nm"]):
        r = results[wl_key]
        dets = [d for d in r["detectors"] if abs(d.get("angle_deg", 0)) < 1]
        dets.sort(key=lambda d: d["sds_mm"])

        n_gates = len(GATE_LABELS)
        matrix = np.zeros((len(dets), n_gates))
        sds_labels = []

        for i, det in enumerate(dets):
            sds_labels.append(f"{det['sds_mm']:.0f}")
            gates = det.get("time_gates", [])
            for g_idx, gate in enumerate(gates):
                if g_idx >= n_gates:
                    break
                ppl = gate.get("partial_pathlength_mm", {})
                amyg = ppl.get("amygdala", 0)
                total = sum(ppl.values())
                matrix[i, g_idx] = (amyg / total * 100) if total > 0 else 0

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")
        ax.set_xticks(range(n_gates))
        ax.set_xticklabels(GATE_LABELS, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(sds_labels)))
        ax.set_yticklabels([f"SDS {s} mm" for s in sds_labels], fontsize=9)
        ax.set_title(f"Amygdala Sensitivity [%] — {wl_key}")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        fig.colorbar(im, ax=ax, label="Sensitivity [%]", shrink=0.8)

    fig.suptitle("Time-Gated Amygdala Sensitivity (primary direction)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "time_gated_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved time_gated_sensitivity.png")


# ---------------------------------------------------------------------------
# 6. Photon counts per gate
# ---------------------------------------------------------------------------
def plot_gate_photon_counts(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    for ax, wl_key in zip(axes, ["760nm", "850nm"]):
        r = results[wl_key]
        dets = [d for d in r["detectors"] if abs(d.get("angle_deg", 0)) < 1]
        dets.sort(key=lambda d: d["sds_mm"])

        sds_vals = [d["sds_mm"] for d in dets]
        colors = plt.cm.tab10(np.linspace(0, 1, len(GATE_LABELS)))

        x = np.arange(len(dets))
        width = 0.13
        for g_idx in range(len(GATE_LABELS)):
            counts = []
            for det in dets:
                gates = det.get("time_gates", [])
                cnt = gates[g_idx]["detected_photons"] if g_idx < len(gates) else 0
                counts.append(cnt)
            ax.bar(x + g_idx * width, counts, width, label=GATE_LABELS[g_idx],
                   color=colors[g_idx])

        ax.set_yscale("log")
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_vals])
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Detected Photons")
        ax.set_title(f"Photons per Time Gate — {wl_key}")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "gate_photon_counts.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved gate_photon_counts.png")


# ---------------------------------------------------------------------------
# 7. Frequency-domain phase & amplitude
# ---------------------------------------------------------------------------
def plot_fd_analysis(results, tpsf, output_dir):
    if "760nm" not in tpsf:
        return

    bin_ps = 10.0
    dt_s = bin_ps * 1e-12
    n_fft = 32768
    freqs_hz = np.fft.rfftfreq(n_fft, d=dt_s)
    freqs_mhz = freqs_hz / 1e6

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    cmap = plt.cm.viridis

    for wl_key, ls in [("760nm", "-"), ("850nm", "--")]:
        if wl_key not in tpsf:
            continue
        r = results[wl_key]
        data = tpsf[wl_key]

        primary = [(i, d) for i, d in enumerate(r["detectors"])
                   if abs(d.get("angle_deg", 0)) < 1 and i < data.shape[0]]
        sds_vals = [d["sds_mm"] for _, d in primary]
        sds_min, sds_max = min(sds_vals), max(sds_vals)

        # Reference for differential phase
        ref_H = None
        for idx, det in primary:
            if det["sds_mm"] < 10:
                ref_H = np.fft.rfft(data[idx], n=n_fft)
                break

        for idx, det in primary:
            t = data[idx]
            if t.sum() <= 0:
                continue
            H = np.fft.rfft(t, n=n_fft)
            phase_deg = np.angle(H) * 180 / np.pi
            amp_db = 20 * np.log10(np.abs(H) / (np.abs(H[0]) + 1e-30) + 1e-30)

            frac = (det["sds_mm"] - sds_min) / (sds_max - sds_min + 1e-9)
            color = cmap(frac)
            label = f"SDS={det['sds_mm']:.0f}" if wl_key == "760nm" else None

            mask = (freqs_mhz > 1) & (freqs_mhz <= 600)
            axes[0].plot(freqs_mhz[mask], phase_deg[mask], ls=ls, color=color,
                         linewidth=1.2, label=label)
            axes[1].plot(freqs_mhz[mask], amp_db[mask], ls=ls, color=color,
                         linewidth=1.2, label=label)

            if ref_H is not None:
                dphase = (np.angle(H) - np.angle(ref_H)) * 180 / np.pi
                axes[2].plot(freqs_mhz[mask], dphase[mask], ls=ls, color=color,
                             linewidth=1.2, label=label)

    axes[0].set_ylabel("Phase [deg]"); axes[0].set_title("Phase Shift")
    axes[1].set_ylabel("|H(f)/H(0)| [dB]"); axes[1].set_title("Amplitude Attenuation")
    axes[2].set_ylabel("Δφ vs SDS=8 mm [deg]"); axes[2].set_title("Differential Phase (depth encoding)")

    for ax in axes:
        ax.set_xlabel("Frequency [MHz]")
        ax.set_xlim(0, 600)
        ax.grid(True, alpha=0.3)

    axes[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Frequency-Domain Analysis (solid = 760 nm, dashed = 850 nm)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "fd_phase_amplitude.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved fd_phase_amplitude.png")


# ---------------------------------------------------------------------------
# 8. SNR comparison bar chart
# ---------------------------------------------------------------------------
def _compute_snr_data(results, tpsf):
    """Compute SNR for CW, TD-gated, and chirp for primary-direction detectors."""
    # Read laser power from results JSON (default 0.1 W = 100 mW)
    laser_power = results.get("760nm", results.get("850nm", {})).get("laser_power_W", 0.1)
    meas_time = 1.0
    delta_hbo = 0.001    # 1 µM in mM
    delta_hbr = -0.0003
    tbp = 5.12e-9 * 495e6

    wl_keys = ["760nm", "850nm"]
    wl_m = [760e-9, 850e-9]
    N_per_sec = [laser_power / (H_PLANCK * C_LIGHT / w) for w in wl_m]

    det_data = {}
    for wl_idx, wl_key in enumerate(wl_keys):
        r = results[wl_key]
        num_sim = r["num_photons"]
        scale = N_per_sec[wl_idx] / num_sim

        for det in r["detectors"]:
            if abs(det.get("angle_deg", 0)) > 1:
                continue
            sds = det["sds_mm"]
            if sds not in det_data:
                det_data[sds] = {}

            w = det["total_weight"]
            amyg_pl = det["partial_pathlength_mm"]["amygdala"]
            N_det = w * scale * meas_time

            delta_od = (EPSILON_HBO[wl_idx] * delta_hbo +
                        EPSILON_HBR[wl_idx] * delta_hbr) * amyg_pl
            noise_cw = 1.0 / np.sqrt(N_det) if N_det > 0 else float('inf')
            snr_cw = abs(delta_od) / noise_cw if noise_cw < float('inf') else 0

            best_snr_td, best_gate = 0, -1
            gates = det.get("time_gates", [])
            for g_idx, gate in enumerate(gates):
                gw = gate.get("weight", 0)
                g_amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                N_gate = gw * scale * meas_time
                dod_g = (EPSILON_HBO[wl_idx] * delta_hbo +
                         EPSILON_HBR[wl_idx] * delta_hbr) * g_amyg
                noise_g = 1.0 / np.sqrt(N_gate) if N_gate > 0 else float('inf')
                snr_g = abs(dod_g) / noise_g if noise_g < float('inf') else 0
                if snr_g > best_snr_td:
                    best_snr_td = snr_g
                    best_gate = g_idx

            snr_chirp = snr_cw * np.sqrt(tbp)

            det_data[sds][wl_key] = dict(
                snr_cw=snr_cw, snr_td=best_snr_td, snr_chirp=snr_chirp,
                best_gate=best_gate, amyg_pl=amyg_pl, N_det=N_det,
                noise_cw=noise_cw,
            )

    return det_data, tbp, laser_power


def plot_snr_comparison(results, tpsf, output_dir):
    det_data, tbp, laser_power = _compute_snr_data(results, tpsf)
    sds_list = sorted(det_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, wl_key, wl_label in zip(axes, ["760nm", "850nm"], ["760 nm", "850 nm"]):
        x = np.arange(len(sds_list))
        w = 0.25
        cw_vals = [det_data[s].get(wl_key, {}).get("snr_cw", 0) for s in sds_list]
        td_vals = [det_data[s].get(wl_key, {}).get("snr_td", 0) for s in sds_list]
        ch_vals = [det_data[s].get(wl_key, {}).get("snr_chirp", 0) for s in sds_list]

        bars1 = ax.bar(x - w, cw_vals, w, label="CW", color="#42A5F5", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x, td_vals, w, label="TD-gated", color="#66BB6A", edgecolor="black", linewidth=0.5)
        bars3 = ax.bar(x + w, ch_vals, w, label="Chirp-corr", color="#FFA726", edgecolor="black", linewidth=0.5)

        ax.axhline(1, color="red", linestyle="--", linewidth=1, label="SNR = 1")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_list])
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel(f"SNR (1 µM ΔHbO, {laser_power*1e3:.0f} mW, 1 s)")
        ax.set_title(f"SNR Comparison — {wl_label}")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Modality Comparison: CW vs Time-Gated vs Chirp (primary direction)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "snr_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved snr_comparison.png")


# ---------------------------------------------------------------------------
# 9. Minimum detectable concentration
# ---------------------------------------------------------------------------
def plot_min_detectable(results, tpsf, output_dir):
    det_data, tbp, laser_power = _compute_snr_data(results, tpsf)
    sds_list = sorted(det_data.keys())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    titles = ["Min Detectable Δ[HbO] [µM]", "Min Detectable Δ[HbR] [µM]"]

    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):
        cw_vals, td_vals, ch_vals = [], [], []

        for sds in sds_list:
            d760 = det_data[sds].get("760nm")
            d850 = det_data[sds].get("850nm")
            if not d760 or not d850:
                cw_vals.append(np.nan); td_vals.append(np.nan); ch_vals.append(np.nan)
                continue

            L760, L850 = d760["amyg_pl"], d850["amyg_pl"]
            if L760 <= 0 or L850 <= 0:
                cw_vals.append(np.nan); td_vals.append(np.nan); ch_vals.append(np.nan)
                continue

            E = np.array([[EPSILON_HBO[0]*L760, EPSILON_HBR[0]*L760],
                          [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]])
            if abs(np.linalg.det(E)) < 1e-20:
                cw_vals.append(np.nan); td_vals.append(np.nan); ch_vals.append(np.nan)
                continue
            E_inv = np.linalg.inv(E)

            dOD_cw = np.array([d760["noise_cw"], d850["noise_cw"]])
            dc_cw = np.abs(E_inv @ dOD_cw) * 1e3  # µM

            # Chirp
            dc_chirp = np.abs(E_inv @ (dOD_cw / np.sqrt(tbp))) * 1e3

            # TD: approximate using same noise model
            # We use the gate-level improvement factor
            td_factor = max(d760["snr_td"] / (d760["snr_cw"] + 1e-30),
                            d850["snr_td"] / (d850["snr_cw"] + 1e-30), 1.0)
            dc_td = dc_cw / td_factor

            cw_vals.append(dc_cw[ax_idx])
            td_vals.append(dc_td[ax_idx])
            ch_vals.append(dc_chirp[ax_idx])

        x = np.arange(len(sds_list))
        w = 0.25
        ax.bar(x - w, cw_vals, w, label="CW", color="#42A5F5", edgecolor="black", linewidth=0.5)
        ax.bar(x, td_vals, w, label="TD-gated", color="#66BB6A", edgecolor="black", linewidth=0.5)
        ax.bar(x + w, ch_vals, w, label="Chirp-corr", color="#FFA726", edgecolor="black", linewidth=0.5)

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="1 µM target")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_list])
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Concentration [µM]")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Minimum Detectable Hemoglobin Change (MBLL, dual-λ, {laser_power*1e3:.0f} mW, 1 s)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "min_detectable_concentration.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved min_detectable_concentration.png")


# ---------------------------------------------------------------------------
# 10. Amygdala pathlength by angle (polar-ish plot)
# ---------------------------------------------------------------------------
def plot_angular_sensitivity(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, wl_key in zip(axes, ["760nm", "850nm"]):
        groups = _get_dets_by_angle(results, wl_key)

        for ang, dets in sorted(groups.items()):
            sds = [d["sds_mm"] for d in dets]
            amyg_pl = [d["partial_pathlength_mm"]["amygdala"] for d in dets]
            color = ANGLE_COLORS.get(ang, "gray")
            marker = ANGLE_MARKERS.get(ang, "x")
            ax.plot(sds, amyg_pl, marker=marker, color=color, linewidth=1.5,
                    markersize=7, label=f"{ang:+d}°")

        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Amygdala Partial Pathlength [mm]")
        ax.set_title(f"Amygdala PL by Direction — {wl_key}")
        ax.legend(fontsize=9, title="Direction")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Angular Dependence of Amygdala Sensitivity", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "angular_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved angular_sensitivity.png")


# ---------------------------------------------------------------------------
# 11. Photon paths — sagittal "banana" view
# ---------------------------------------------------------------------------
def _ellipsoid_boundary(cx, cy, a, b, n=100):
    """Return (x, y) for ellipse outline."""
    t = np.linspace(0, 2 * np.pi, n)
    return cx + a * np.cos(t), cy + b * np.sin(t)


def _tissue_at_point(x, y, z, meta):
    """Approximate tissue type from ellipsoidal head model (mm coords from center)."""
    cx = meta["nx"] * meta["dx"] / 2
    cy = meta["ny"] * meta["dx"] / 2
    cz = meta["nz"] * meta["dx"] / 2
    rx, ry, rz = (x - cx) / 48, (y - cy) / 48, (z - cz) / 48
    r = np.sqrt(rx**2 + ry**2 + rz**2)
    if r > 1.0:
        return 0  # air
    # Check amygdala (right)
    ax, ay, az = (x - cx - 24) / 5, (y - cy - 5) / 9, (z - cz + 15) / 6
    if ax**2 + ay**2 + az**2 <= 1.0:
        return 6
    # Check amygdala (left)
    ax2 = (x - cx + 24) / 5
    if ax2**2 + ay**2 + az**2 <= 1.0:
        return 6
    r44 = np.sqrt(((x-cx)/44)**2 + ((y-cy)/44)**2 + ((z-cz)/44)**2)
    if r44 > 1.0:
        return 1  # scalp
    r38 = np.sqrt(((x-cx)/38)**2 + ((y-cy)/38)**2 + ((z-cz)/38)**2)
    if r38 > 1.0:
        return 2  # skull
    r365 = np.sqrt(((x-cx)/36.5)**2 + ((y-cy)/36.5)**2 + ((z-cz)/36.5)**2)
    if r365 > 1.0:
        return 3  # CSF
    r33 = np.sqrt(((x-cx)/33)**2 + ((y-cy)/33)**2 + ((z-cz)/33)**2)
    if r33 > 1.0:
        return 4  # gray matter
    return 5  # white matter


def plot_photon_paths(paths, results, meta, output_dir):
    """Plot photon paths in sagittal view through the right amygdala."""
    dx = meta["dx"]
    cx = meta["nx"] * dx / 2  # center in mm
    cy = meta["ny"] * dx / 2
    cz = meta["nz"] * dx / 2

    for wl_key in paths:
        pdata = paths[wl_key]
        det_ids = pdata["det_ids"]
        path_lens = pdata["path_lens"]
        positions = pdata["positions"]  # (N, MAX_STEPS, 3)
        n_paths = len(det_ids)

        if n_paths == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # --- Left: Sagittal view (X=const through R amygdala) ---
        ax = axes[0]
        # Draw tissue boundaries (sagittal = Y-Z plane at x=cx+24)
        for radius, color, label in [
            (48, "#D2B48C", "Scalp"), (44, "#DEB887", "Skull"),
            (38, "#87CEEB", "CSF"), (36.5, "#CD5C5C", "Gray"),
            (33, "#F5F5DC", "White")
        ]:
            ey, ez = _ellipsoid_boundary(cy, cz, radius, radius)
            ax.plot(ey, ez, color=color, linewidth=1.0, alpha=0.5)

        # Right amygdala ellipse in sagittal view
        ay, az = _ellipsoid_boundary(cy + 5, cz - 15, 9, 6)
        ax.fill(ay, az, color=TISSUE_COLORS[6], alpha=0.3, label="Amygdala")
        ax.plot(ay, az, color="red", linewidth=1.5)

        # Plot paths (subsample if too many)
        max_show = min(200, n_paths)
        indices = np.random.default_rng(42).choice(n_paths, max_show, replace=False) if n_paths > max_show else np.arange(n_paths)

        amyg_count = 0
        for idx in indices:
            nsteps = path_lens[idx]
            pts = positions[idx, :nsteps, :]  # (nsteps, 3) = x, y, z in mm

            # Color by whether path passed near amygdala
            # Check if any point is inside R amygdala ellipsoid
            ax_r = (pts[:, 0] - cx - 24) / 5
            ay_r = (pts[:, 1] - cy - 5) / 9
            az_r = (pts[:, 2] - cz + 15) / 6
            in_amyg = np.any(ax_r**2 + ay_r**2 + az_r**2 <= 1.0)

            if in_amyg:
                ax.plot(pts[:, 1], pts[:, 2], color="red", alpha=0.4, linewidth=0.8)
                amyg_count += 1
            else:
                ax.plot(pts[:, 1], pts[:, 2], color="blue", alpha=0.1, linewidth=0.3)

        ax.set_xlim(cy - 50, cy + 50)
        ax.set_ylim(cz + 50, cz - 50)  # invert Z so surface is on top
        ax.set_xlabel("Y [mm] (anterior-posterior)")
        ax.set_ylabel("Z [mm] (inferior-superior)")
        ax.set_title(f"Sagittal Photon Paths — {wl_key}\n"
                      f"({max_show} paths shown, {amyg_count} reach amygdala)")
        ax.set_aspect("equal")
        ax.legend(fontsize=9, loc="lower right")

        # --- Right: Coronal view (Y=const) ---
        ax2 = axes[1]
        for radius, color, label in [
            (48, "#D2B48C", "Scalp"), (44, "#DEB887", "Skull"),
            (38, "#87CEEB", "CSF"), (36.5, "#CD5C5C", "Gray"),
            (33, "#F5F5DC", "White")
        ]:
            ex, ez = _ellipsoid_boundary(cx, cz, radius, radius)
            ax2.plot(ex, ez, color=color, linewidth=1.0, alpha=0.5)

        # Both amygdalae in coronal view
        for amyg_cx_off in [-24, 24]:
            aex, aez = _ellipsoid_boundary(cx + amyg_cx_off, cz - 15, 5, 6)
            ax2.fill(aex, aez, color=TISSUE_COLORS[6], alpha=0.3)
            ax2.plot(aex, aez, color="red", linewidth=1.5)

        for idx in indices:
            nsteps = path_lens[idx]
            pts = positions[idx, :nsteps, :]
            ax_r = (pts[:, 0] - cx - 24) / 5
            ay_r = (pts[:, 1] - cy - 5) / 9
            az_r = (pts[:, 2] - cz + 15) / 6
            in_amyg = np.any(ax_r**2 + ay_r**2 + az_r**2 <= 1.0)
            if in_amyg:
                ax2.plot(pts[:, 0], pts[:, 2], color="red", alpha=0.4, linewidth=0.8)
            else:
                ax2.plot(pts[:, 0], pts[:, 2], color="blue", alpha=0.1, linewidth=0.3)

        ax2.set_xlim(cx - 50, cx + 50)
        ax2.set_ylim(cz + 50, cz - 50)
        ax2.set_xlabel("X [mm] (left-right)")
        ax2.set_ylabel("Z [mm] (inferior-superior)")
        ax2.set_title(f"Coronal Photon Paths — {wl_key}")
        ax2.set_aspect("equal")

        fig.suptitle(f"Photon Trajectories — {wl_key} (red = passes through amygdala)",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / f"photon_paths_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved photon_paths_{wl_key}.png")


# ---------------------------------------------------------------------------
# 12. Photon paths grouped by detector
# ---------------------------------------------------------------------------
def plot_photon_paths_by_detector(paths, results, meta, output_dir):
    """Plot photon paths for selected detectors showing banana shapes at different SDS."""
    dx = meta["dx"]
    cx = meta["nx"] * dx / 2
    cy = meta["ny"] * dx / 2
    cz = meta["nz"] * dx / 2

    for wl_key in paths:
        pdata = paths[wl_key]
        det_ids = pdata["det_ids"]
        path_lens = pdata["path_lens"]
        positions = pdata["positions"]

        if len(det_ids) == 0:
            continue

        # Get unique detector IDs that have paths
        unique_dets = np.unique(det_ids)
        # Pick up to 6 detectors with most paths, spread across SDS
        det_counts = {d: np.sum(det_ids == d) for d in unique_dets}
        # Sort by ID (roughly by SDS since placed in order)
        sorted_dets = sorted(unique_dets)
        # Select evenly spaced subset
        if len(sorted_dets) > 6:
            step = len(sorted_dets) / 6
            selected = [sorted_dets[int(i * step)] for i in range(6)]
        else:
            selected = sorted_dets

        n_sel = len(selected)
        if n_sel == 0:
            continue

        cols = min(3, n_sel)
        rows = (n_sel + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows),
                                 squeeze=False)

        # Get detector info from results
        det_info = {}
        if wl_key in results:
            for d in results[wl_key].get("detectors", []):
                det_info[d.get("id", -1)] = d

        for ax_i, det_id in enumerate(selected):
            ax = axes[ax_i // cols][ax_i % cols]

            # Draw tissue boundaries (sagittal)
            for radius, color in [
                (48, "#D2B48C"), (44, "#DEB887"), (38, "#87CEEB"),
                (36.5, "#CD5C5C"), (33, "#F5F5DC")
            ]:
                ey, ez = _ellipsoid_boundary(cy, cz, radius, radius)
                ax.plot(ey, ez, color=color, linewidth=0.8, alpha=0.4)

            aey, aez = _ellipsoid_boundary(cy + 5, cz - 15, 9, 6)
            ax.fill(aey, aez, color=TISSUE_COLORS[6], alpha=0.3)
            ax.plot(aey, aez, color="red", linewidth=1.0)

            mask = det_ids == det_id
            det_paths = positions[mask]
            det_plens = path_lens[mask]
            n = len(det_paths)

            max_show = min(50, n)
            rng = np.random.default_rng(det_id)
            idxs = rng.choice(n, max_show, replace=False) if n > max_show else np.arange(n)

            for idx in idxs:
                nsteps = det_plens[idx]
                pts = det_paths[idx, :nsteps, :]

                # Color by depth reached
                max_depth = (cz - pts[:, 2].min())  # how far below surface center
                depth_frac = np.clip(max_depth / 40, 0, 1)
                color = plt.cm.plasma(depth_frac)
                ax.plot(pts[:, 1], pts[:, 2], color=color, alpha=0.5, linewidth=0.6)

            info = det_info.get(det_id, {})
            sds = info.get("sds_mm", "?")
            angle = info.get("angle_deg", "?")
            ax.set_title(f"Det {det_id}: SDS={sds} mm, {angle}°\n({n} paths total)")
            ax.set_xlim(cy - 50, cy + 50)
            ax.set_ylim(cz + 50, cz - 50)
            ax.set_xlabel("Y [mm]")
            ax.set_ylabel("Z [mm]")
            ax.set_aspect("equal")

        # Hide empty subplots
        for ax_i in range(n_sel, rows * cols):
            axes[ax_i // cols][ax_i % cols].set_visible(False)

        fig.suptitle(f"Photon Paths by Detector — {wl_key}\n(color = penetration depth)",
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig(output_dir / f"photon_paths_by_det_{wl_key}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved photon_paths_by_det_{wl_key}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="fNIRS MC TD/FD Visualization")
    parser.add_argument("--data-dir", type=str, default="../results")
    parser.add_argument("--output-dir", type=str, default="../figures")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    vol, fluence, results, tpsf, meta, paths = load_data(data_dir)
    if vol is not None:
        print(f"  Volume: {vol.shape}, Fluence: {list(fluence.keys())}")
    else:
        print(f"  Volume: skipped (large grid), Fluence: {list(fluence.keys())}")
    print(f"  TPSF: {', '.join(f'{k}: {v.shape}' for k, v in tpsf.items())}")
    path_summary = ", ".join(f"{k}: {v['det_ids'].shape[0]}" for k, v in paths.items())
    print(f"  Paths: {path_summary}")

    print("\nGenerating figures...")
    if vol is not None:
        plot_tissue_slices(vol, meta, output_dir)
        plot_fluence(vol, fluence, meta, output_dir)
    else:
        print("  Skipping tissue_slices.png and fluence (no volume.bin)")
    plot_cw_sensitivity(results, output_dir)
    plot_angular_sensitivity(results, output_dir)
    plot_tpsf(results, tpsf, output_dir)
    plot_time_gated_sensitivity(results, output_dir)
    plot_gate_photon_counts(results, output_dir)
    plot_fd_analysis(results, tpsf, output_dir)
    plot_snr_comparison(results, tpsf, output_dir)
    plot_min_detectable(results, tpsf, output_dir)
    if paths:
        plot_photon_paths(paths, results, meta, output_dir)
        plot_photon_paths_by_detector(paths, results, meta, output_dir)
    else:
        print("  Skipping photon path plots (no path data)")

    n_figs = 10 + (2 if paths else 0) - (2 if vol is None else 0)
    print(f"\nAll {n_figs} figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
