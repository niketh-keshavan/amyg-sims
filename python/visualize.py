#!/usr/bin/env python3
"""
Visualization for 730/850nm TD-Gated fNIRS Monte Carlo results.

Generates:
  1. Tissue volume cross-sections
  2. Fluence maps
  3. TPSF curves
  4. Time-gated amygdala sensitivity heatmap
  5. Gate photon counts
  6. TD-gated SNR vs SDS (120s integration)
  7. Min detectable HbO/HbR (dual-lambda MBLL, 120s)
  8. Integration time vs sensitivity curve
  9. Block-design expected signal
 10. Photon paths
 11. CW vs TD sensitivity comparison
 12. Dual-wavelength MBLL sensitivity map

Usage:
    python visualize.py --data-dir ../data/730-850 --output-dir ../figures/730-850
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
    [0.0, 0.0, 0.0, 0.0],    # 0: air
    [0.9, 0.7, 0.6, 1.0],    # 1: scalp
    [0.95, 0.95, 0.85, 1.0], # 2: skull
    [0.6, 0.8, 1.0, 1.0],    # 3: CSF
    [0.8, 0.5, 0.5, 1.0],    # 4: gray matter
    [0.95, 0.95, 0.95, 1.0], # 5: white matter
    [1.0, 0.2, 0.2, 1.0],    # 6: amygdala
]
TISSUE_CMAP = ListedColormap(TISSUE_COLORS)
TISSUE_NAMES = ["Air", "Scalp", "Skull", "CSF", "Gray Matter",
                "White Matter", "Amygdala"]

GATE_LABELS = [
    "0-500ps", "0.5-1ns", "1-1.5ns", "1.5-2ns", "2-2.5ns",
    "2.5-3ns", "3-3.5ns", "3.5-4ns", "4-5ns", "5ns+"
]

EPSILON_HBR = np.array([1.3080, 0.6918]) / 10.0  # mM^-1 mm^-1
EPSILON_HBO = np.array([0.1348, 1.0507]) / 10.0
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
WAVELENGTHS_M = np.array([730e-9, 850e-9])
WL_KEYS = ["730nm", "850nm"]

# System parameters
LASER_POWER = 1.0  # 1W
MEAS_TIME = 120.0
DET_QE = {"730nm": 0.30, "850nm": 0.15}
DARK_RATE = 1000

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
    vol = np.fromfile(vol_path, dtype=np.uint8).reshape((nz, ny, nx)) if vol_path.exists() else None

    fluence = {}
    for wl in WL_KEYS:
        fp = data_dir / f"fluence_{wl}.bin"
        if fp.exists():
            fluence[wl] = np.fromfile(fp, dtype=np.float32).reshape((nz, ny, nx))

    results = {}
    for wl in WL_KEYS:
        fp = data_dir / f"results_{wl}.json"
        if fp.exists():
            with open(fp) as f:
                results[wl] = json.load(f)

    tpsf = {}
    for wl in WL_KEYS:
        fp = data_dir / f"tpsf_{wl}.bin"
        if fp.exists():
            raw = np.fromfile(fp, dtype=np.float64)
            tpsf[wl] = raw.reshape(len(raw) // 512, 512)

    paths = {}
    MAX_PATH_STEPS = 2048
    for wl in WL_KEYS:
        meta_fp = data_dir / f"paths_meta_{wl}.bin"
        pos_fp = data_dir / f"paths_pos_{wl}.bin"
        if meta_fp.exists() and pos_fp.exists():
            raw_meta = np.fromfile(meta_fp, dtype=np.int32)
            num_paths = len(raw_meta) // 2
            det_ids = raw_meta[0::2]
            path_lens = raw_meta[1::2]
            raw_pos = np.fromfile(pos_fp, dtype=np.float32)
            positions = raw_pos.reshape(num_paths, MAX_PATH_STEPS, 3)
            valid = path_lens > 0
            paths[wl] = {
                "det_ids": det_ids[valid],
                "path_lens": path_lens[valid],
                "positions": positions[valid],
            }

    return vol, fluence, results, tpsf, meta, paths


def _primary_dets(results, wl_key):
    return [d for d in results[wl_key]["detectors"] if abs(d.get("angle_deg", 0)) < 1]


def _photons_per_sec(wl_idx):
    return LASER_POWER / (H_PLANCK * C_LIGHT / WAVELENGTHS_M[wl_idx])


# ---------------------------------------------------------------------------
# 1. Tissue slices
# ---------------------------------------------------------------------------
def plot_tissue_slices(vol, meta, output_dir):
    dx = meta["dx"]
    nz, ny, nx = vol.shape
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    z_amyg = nz // 2 - int(18.0 / dx)
    im = axes[0].imshow(vol[z_amyg], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                        extent=[0, nx*dx, ny*dx, 0], interpolation="nearest")
    axes[0].set_title(f"Axial z={z_amyg*dx:.0f}mm (amygdala level)")
    axes[0].set_xlabel("X [mm]"); axes[0].set_ylabel("Y [mm]")

    y_amyg = ny // 2 - int(2.0 / dx)
    axes[1].imshow(vol[:, y_amyg, :], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                   extent=[0, nx*dx, nz*dx, 0], interpolation="nearest")
    axes[1].set_title(f"Coronal y={y_amyg*dx:.0f}mm")
    axes[1].set_xlabel("X [mm]"); axes[1].set_ylabel("Z [mm]")

    x_amyg = nx // 2 + int(24.0 / dx)
    axes[2].imshow(vol[:, :, x_amyg], cmap=TISSUE_CMAP, vmin=0, vmax=6,
                   extent=[0, ny*dx, nz*dx, 0], interpolation="nearest")
    axes[2].set_title(f"Sagittal x={x_amyg*dx:.0f}mm (R amygdala)")
    axes[2].set_xlabel("Y [mm]"); axes[2].set_ylabel("Z [mm]")

    cbar = fig.colorbar(im, ax=axes, ticks=range(7), shrink=0.8)
    cbar.ax.set_yticklabels(TISSUE_NAMES)
    plt.tight_layout()
    plt.savefig(output_dir / "01_tissue_slices.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 2. Fluence maps
# ---------------------------------------------------------------------------
def plot_fluence(vol, fluence, meta, output_dir):
    dx = meta["dx"]
    nz, ny, nx = vol.shape

    for wl_key, flu in fluence.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x_amyg = nx // 2 + int(24.0 / dx)
        z_amyg = nz // 2 - int(18.0 / dx)

        slices = [
            (vol[:, :, x_amyg], flu[:, :, x_amyg],
             f"Sagittal x={x_amyg*dx:.0f}mm", [0, ny*dx, nz*dx, 0]),
            (vol[z_amyg], flu[z_amyg],
             "Axial (amygdala level)", [0, nx*dx, ny*dx, 0]),
        ]
        for ax, (ts, fs, title, ext) in zip(axes, slices):
            ax.imshow(ts, cmap=TISSUE_CMAP, vmin=0, vmax=6,
                      extent=ext, interpolation="nearest", alpha=0.3)
            fm = np.ma.masked_where(fs <= 0, fs)
            if fm.count() > 0:
                im = ax.imshow(fm, cmap="hot", norm=LogNorm(vmin=fm.min(), vmax=fm.max()),
                               extent=ext, interpolation="bilinear", alpha=0.7)
                fig.colorbar(im, ax=ax, label="Fluence [a.u.]", shrink=0.8)
            ax.set_title(f"Fluence {wl_key} - {title}")

        plt.tight_layout()
        plt.savefig(output_dir / f"02_fluence_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# 3. TPSF curves with gate overlay
# ---------------------------------------------------------------------------
def plot_tpsf(results, tpsf, output_dir):
    if WL_KEYS[0] not in tpsf:
        return
    bin_ps = 10.0
    time_ns = np.arange(512) * bin_ps / 1000.0
    gate_edges_ns = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, wl_key in zip(axes, WL_KEYS):
        if wl_key not in tpsf:
            continue
        r = results[wl_key]
        data = tpsf[wl_key]
        cmap = plt.cm.viridis

        primary = [(i, d) for i, d in enumerate(r["detectors"])
                    if abs(d.get("angle_deg", 0)) < 1 and i < data.shape[0]]
        sds_vals = [d["sds_mm"] for _, d in primary]
        sds_min, sds_max = min(sds_vals), max(sds_vals)

        for idx, det in primary:
            t = data[idx]
            if t.sum() <= 0:
                continue
            normed = t / t.max()
            frac = (det["sds_mm"] - sds_min) / (sds_max - sds_min + 1e-9)
            ax.plot(time_ns, normed, color=cmap(frac), linewidth=1.5,
                    label=f"SDS={det['sds_mm']:.0f}mm")

        for edge in gate_edges_ns:
            ax.axvline(edge, color='gray', linewidth=0.5, alpha=0.3)
        ax.axvspan(2.0, 5.0, alpha=0.08, color='red', label='Amygdala-sensitive gates')

        ax.set_xlabel("Time of Flight [ns]")
        ax.set_ylabel("Normalized TPSF")
        ax.set_title(f"TPSF - {wl_key}")
        ax.set_xlim(0, 5.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "03_tpsf_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 4. Time-gated sensitivity heatmap
# ---------------------------------------------------------------------------
def plot_sensitivity_heatmap(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    for ax, wl_key in zip(axes, WL_KEYS):
        if wl_key not in results:
            continue
        r = results[wl_key]
        dets = sorted(_primary_dets(results, wl_key), key=lambda d: d["sds_mm"])

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
        ax.set_xticklabels(GATE_LABELS, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(sds_labels)))
        ax.set_yticklabels([f"SDS {s}mm" for s in sds_labels], fontsize=9)
        ax.set_title(f"Amygdala Sensitivity [%] - {wl_key}")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                color = "white" if val > matrix.max() * 0.6 else "black"
                ax.text(j, i, f"{val:.3f}" if val > 0 else "0",
                        ha="center", va="center", fontsize=7, color=color)

        fig.colorbar(im, ax=ax, label="Sensitivity [%]", shrink=0.8)

    fig.suptitle("TD-Gated Amygdala Sensitivity (730/850nm, primary direction)", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "04_td_sensitivity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Gate photon counts
# ---------------------------------------------------------------------------
def plot_gate_counts(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, wl_key in zip(axes, WL_KEYS):
        dets = sorted(_primary_dets(results, wl_key), key=lambda d: d["sds_mm"])
        sds_vals = [d["sds_mm"] for d in dets]
        colors = plt.cm.tab10(np.linspace(0, 1, len(GATE_LABELS)))

        x = np.arange(len(dets))
        width = 0.08
        for g_idx in range(len(GATE_LABELS)):
            counts = []
            for det in dets:
                gates = det.get("time_gates", [])
                cnt = gates[g_idx]["detected_photons"] if g_idx < len(gates) else 0
                counts.append(max(cnt, 0.5))
            ax.bar(x + g_idx * width, counts, width, label=GATE_LABELS[g_idx],
                   color=colors[g_idx])

        ax.set_yscale("log")
        ax.set_xticks(x + width * 4.5)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_vals], fontsize=8)
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Detected Photons (simulation)")
        ax.set_title(f"Photons per Gate - {wl_key}")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_dir / "05_gate_photon_counts.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 6. TD-gated SNR @ 120s
# ---------------------------------------------------------------------------
def plot_td_snr(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, wl_key, wl_idx in zip(axes, WL_KEYS, [0, 1]):
        r = results[wl_key]
        qe = DET_QE[wl_key]
        N_ps = _photons_per_sec(wl_idx)
        num_sim = r["num_photons"]
        scale = N_ps / num_sim

        dets = sorted(_primary_dets(results, wl_key), key=lambda d: d["sds_mm"])
        sds_vals = [d["sds_mm"] for d in dets]

        snr_vals = []
        gate_used = []
        for det in dets:
            best_snr = 0
            best_g = -1
            for g_idx, gate in enumerate(det.get("time_gates", [])):
                gw = gate.get("weight", 0)
                amyg = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                N = gw * scale * MEAS_TIME * qe + DARK_RATE * MEAS_TIME
                delta_od = abs((EPSILON_HBO[wl_idx]*0.001 + EPSILON_HBR[wl_idx]*(-0.0003)) * amyg)
                snr = delta_od * np.sqrt(N) if N > 0 else 0
                if snr > best_snr:
                    best_snr = snr
                    best_g = g_idx
            snr_vals.append(best_snr)
            gate_used.append(best_g)

        colors = plt.cm.Set2(np.array(gate_used) / max(max(gate_used), 1))
        bars = ax.bar(range(len(dets)), snr_vals, color=colors, edgecolor="black", linewidth=0.5)

        for i, (bar, g) in enumerate(zip(bars, gate_used)):
            if snr_vals[i] > 0 and g >= 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        GATE_LABELS[g], ha='center', va='bottom', fontsize=6, rotation=45)

        ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="SNR=1")
        ax.set_xticks(range(len(dets)))
        ax.set_xticklabels([f"{s:.0f}" for s in sds_vals], fontsize=8)
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel(f"SNR (1uM HbO, {MEAS_TIME:.0f}s)")
        ax.set_title(f"TD-Gated Best-Gate SNR - {wl_key}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"TD-Gated Best-Gate SNR ({LASER_POWER*1e3:.0f}mW, {MEAS_TIME:.0f}s, Si-PMT)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "06_td_snr_120s.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 7. Min detectable HbO/HbR (dual-wavelength)
# ---------------------------------------------------------------------------
def plot_min_detectable(results, output_dir):
    N_ps = [_photons_per_sec(i) for i in range(2)]
    dets730 = sorted(_primary_dets(results, "730nm"), key=lambda d: d["sds_mm"])
    dets850 = sorted(_primary_dets(results, "850nm"), key=lambda d: d["sds_mm"])

    sds_vals = []
    min_hbo_vals = []
    min_hbr_vals = []
    best_gates = []

    for d730, d850 in zip(dets730, dets850):
        sds = d730["sds_mm"]
        best_hbo = float('inf')
        best_hbr = float('inf')
        best_g = -1

        n_gates = min(len(d730.get("time_gates", [])), len(d850.get("time_gates", [])))
        for g_idx in range(n_gates):
            g730, g850 = d730["time_gates"][g_idx], d850["time_gates"][g_idx]
            L730 = g730.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if L730 <= 0 or L850 <= 0:
                continue

            gw730 = g730.get("weight", 0)
            gw850 = g850.get("weight", 0)
            scale730 = N_ps[0] / results["730nm"]["num_photons"]
            scale850 = N_ps[1] / results["850nm"]["num_photons"]
            N730 = gw730 * scale730 * MEAS_TIME * DET_QE["730nm"] + DARK_RATE * MEAS_TIME
            N850 = gw850 * scale850 * MEAS_TIME * DET_QE["850nm"] + DARK_RATE * MEAS_TIME
            n730 = 1/np.sqrt(N730) if N730 > 0 else float('inf')
            n850 = 1/np.sqrt(N850) if N850 > 0 else float('inf')

            E = np.array([[EPSILON_HBO[0]*L730, EPSILON_HBR[0]*L730],
                          [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            dc = np.abs(np.linalg.inv(E) @ np.array([n730, n850])) * 1e3
            if dc[0] < best_hbo:
                best_hbo, best_hbr = dc[0], dc[1]
                best_g = g_idx

        sds_vals.append(sds)
        min_hbo_vals.append(best_hbo if best_hbo < 1e6 else np.nan)
        min_hbr_vals.append(best_hbr if best_hbr < 1e6 else np.nan)
        best_gates.append(best_g)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(sds_vals))

    for ax, vals, title, target in zip(axes,
            [min_hbo_vals, min_hbr_vals],
            ["Min Detectable dHbO [uM]", "Min Detectable dHbR [uM]"],
            [1.0, 1.0]):
        colors = ['#66BB6A' if v < target else '#FFA726' if v < target*5 else '#EF5350'
                   for v in vals]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(target, color="red", linestyle="--", linewidth=1.5, label=f"{target} uM target")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_vals], fontsize=8)
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Concentration [uM]")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        # Annotate best gate on each bar
        for i, (bar, g) in enumerate(zip(bars, best_gates)):
            if g >= 0 and not np.isnan(vals[i]):
                ax.text(bar.get_x() + bar.get_width()/2, vals[i]*1.1,
                        GATE_LABELS[g], ha='center', va='bottom', fontsize=6, rotation=45)

    fig.suptitle(f"Dual-wavelength MBLL Min Detectable (TD-gated, {MEAS_TIME:.0f}s, Si-PMT)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "07_min_detectable_td.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 8. Integration time curve
# ---------------------------------------------------------------------------
def plot_integration_curve(results, output_dir):
    N_ps = [_photons_per_sec(i) for i in range(2)]
    dets730 = sorted(_primary_dets(results, "730nm"), key=lambda d: d["sds_mm"])
    dets850 = sorted(_primary_dets(results, "850nm"), key=lambda d: d["sds_mm"])

    times = np.logspace(0, 3, 100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cmap = plt.cm.viridis

    sds_all = [d["sds_mm"] for d in dets730]
    sds_min, sds_max = min(sds_all), max(sds_all)

    for d730, d850 in zip(dets730, dets850):
        sds = d730["sds_mm"]

        best_min_1s = float('inf')
        n_gates = min(len(d730.get("time_gates", [])), len(d850.get("time_gates", [])))
        for g_idx in range(n_gates):
            g730, g850 = d730["time_gates"][g_idx], d850["time_gates"][g_idx]
            L730 = g730.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if L730 <= 0 or L850 <= 0:
                continue
            gw730 = g730.get("weight", 0)
            gw850 = g850.get("weight", 0)
            scale730 = N_ps[0] / results["730nm"]["num_photons"]
            scale850 = N_ps[1] / results["850nm"]["num_photons"]
            N730 = gw730 * scale730 * 1.0 * DET_QE["730nm"] + DARK_RATE
            N850 = gw850 * scale850 * 1.0 * DET_QE["850nm"] + DARK_RATE
            n730 = 1/np.sqrt(N730) if N730 > 0 else float('inf')
            n850 = 1/np.sqrt(N850) if N850 > 0 else float('inf')
            E = np.array([[EPSILON_HBO[0]*L730, EPSILON_HBR[0]*L730],
                          [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            dc = np.abs(np.linalg.inv(E) @ np.array([n730, n850])) * 1e3
            if dc[0] < best_min_1s:
                best_min_1s = dc[0]

        if best_min_1s > 1e5:
            continue

        curve = best_min_1s / np.sqrt(times)
        frac = (sds - sds_min) / (sds_max - sds_min + 1e-9)
        ax.plot(times, curve, color=cmap(frac), linewidth=1.5,
                label=f"SDS={sds:.0f}mm")

    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="1 uM target")
    ax.axhline(2.0, color="orange", linestyle=":", linewidth=1, label="2 uM (block avg)")
    ax.axvline(120, color="gray", linestyle="--", alpha=0.5, label="2 min")
    ax.axvline(15, color="gray", linestyle=":", alpha=0.5, label="15s (1 trial)")

    ax.set_xlabel("Integration Time [s]")
    ax.set_ylabel("Min Detectable dHbO [uM]")
    ax.set_title("Integration Time vs Sensitivity (dual-wavelength TD-gated MBLL)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.01, 100)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "08_integration_time_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 9. Block design expected signal
# ---------------------------------------------------------------------------
def plot_block_design(results, output_dir):
    N_ps = [_photons_per_sec(i) for i in range(2)]
    dets730 = sorted(_primary_dets(results, "730nm"), key=lambda d: d["sds_mm"])
    dets850 = sorted(_primary_dets(results, "850nm"), key=lambda d: d["sds_mm"])

    # Find best SDS
    best_sds_min = float('inf')
    best_sds = 0
    for d730, d850 in zip(dets730, dets850):
        n_gates = min(len(d730.get("time_gates", [])), len(d850.get("time_gates", [])))
        for g_idx in range(n_gates):
            g730, g850 = d730["time_gates"][g_idx], d850["time_gates"][g_idx]
            L730 = g730.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
            if L730 <= 0 or L850 <= 0:
                continue
            gw730 = g730.get("weight", 0)
            gw850 = g850.get("weight", 0)
            scale730 = N_ps[0] / results["730nm"]["num_photons"]
            scale850 = N_ps[1] / results["850nm"]["num_photons"]
            N730 = gw730 * scale730 * MEAS_TIME * DET_QE["730nm"] + DARK_RATE * MEAS_TIME
            N850 = gw850 * scale850 * MEAS_TIME * DET_QE["850nm"] + DARK_RATE * MEAS_TIME
            n730 = 1/np.sqrt(N730) if N730 > 0 else float('inf')
            n850 = 1/np.sqrt(N850) if N850 > 0 else float('inf')
            E = np.array([[EPSILON_HBO[0]*L730, EPSILON_HBR[0]*L730],
                          [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]])
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            dc = np.abs(np.linalg.inv(E) @ np.array([n730, n850])) * 1e3
            if dc[0] < best_sds_min:
                best_sds_min = dc[0]
                best_sds = d730["sds_mm"]

    # Simulate block design time course
    dt = 0.5
    n_trials = 20
    trial_on = 15.0
    trial_off = 15.0
    trial_dur = trial_on + trial_off
    total_time = n_trials * trial_dur

    t = np.arange(0, total_time, dt)
    hbo_true = np.zeros_like(t)
    hbr_true = np.zeros_like(t)

    for trial in range(n_trials):
        onset = trial * trial_dur
        for i, ti in enumerate(t):
            rel = ti - onset
            if 0 < rel < trial_on + 10:
                if rel < trial_on:
                    hrf = 3.0 * (rel / 5.0) * np.exp(-(rel - 5.0) / 3.0)
                else:
                    decay_t = rel - trial_on
                    peak_val = 3.0 * (trial_on / 5.0) * np.exp(-(trial_on - 5.0) / 3.0)
                    hrf = peak_val * np.exp(-decay_t / 4.0)
                hbo_true[i] += hrf
                hbr_true[i] -= hrf * 0.3

    noise_std_hbo = best_sds_min / np.sqrt(dt) if best_sds_min < 1e5 else 10
    noise_std_hbr = noise_std_hbo * 3

    np.random.seed(42)
    hbo_meas = hbo_true + np.random.normal(0, noise_std_hbo, len(t))
    hbr_meas = hbr_true + np.random.normal(0, noise_std_hbr, len(t))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(t, hbo_true, 'r-', linewidth=2, label='True HbO', alpha=0.8)
    axes[0].plot(t, hbo_meas, 'r-', linewidth=0.3, alpha=0.3, label='Measured')
    win = int(15.0 / dt)
    if len(hbo_meas) > win:
        hbo_smooth = np.convolve(hbo_meas, np.ones(win)/win, mode='same')
        axes[0].plot(t, hbo_smooth, 'darkred', linewidth=1.5, label='15s moving avg')
    axes[0].set_ylabel("dHbO [uM]")
    axes[0].set_title(f"Block Design: {n_trials} trials x ({trial_on:.0f}s on / {trial_off:.0f}s off) "
                      f"- SDS={best_sds:.0f}mm TD-gated")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, hbr_true, 'b-', linewidth=2, label='True HbR', alpha=0.8)
    axes[1].plot(t, hbr_meas, 'b-', linewidth=0.3, alpha=0.3, label='Measured')
    if len(hbr_meas) > win:
        hbr_smooth = np.convolve(hbr_meas, np.ones(win)/win, mode='same')
        axes[1].plot(t, hbr_smooth, 'darkblue', linewidth=1.5, label='15s moving avg')
    axes[1].set_ylabel("dHbR [uM]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    for trial in range(n_trials):
        onset = trial * trial_dur
        for ax in axes:
            ax.axvspan(onset, onset + trial_on, alpha=0.05, color='yellow')

    plt.tight_layout()
    plt.savefig(output_dir / "09_block_design_signal.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 10. Photon paths
# ---------------------------------------------------------------------------
def _ellipsoid_boundary(cx, cy, a, b, n=100):
    t = np.linspace(0, 2 * np.pi, n)
    return cx + a * np.cos(t), cy + b * np.sin(t)


def plot_photon_paths(paths, results, meta, output_dir):
    dx = meta["dx"]
    cx = meta["nx"] * dx / 2
    cy = meta["ny"] * dx / 2
    cz = meta["nz"] * dx / 2

    for wl_key in paths:
        pdata = paths[wl_key]
        det_ids = pdata["det_ids"]
        path_lens = pdata["path_lens"]
        positions = pdata["positions"]
        n_paths = len(det_ids)
        if n_paths == 0:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        for ax_idx, (ax, view) in enumerate(zip(axes, ["sagittal", "coronal"])):
            for (sa_y, sa_z), color in [
                ((95, 85), "#D2B48C"), ((91, 81), "#DEB887"), ((84, 74), "#87CEEB"),
                ((82.5, 72.5), "#CD5C5C"), ((79, 69), "#F5F5DC")
            ]:
                if view == "sagittal":
                    ey, ez = _ellipsoid_boundary(cy, cz, sa_y, sa_z)
                else:
                    ey, ez = _ellipsoid_boundary(cx, cz, sa_y * 78 / 95, sa_z)
                ax.plot(ey, ez, color=color, linewidth=1.0, alpha=0.5)

            if view == "sagittal":
                aey, aez = _ellipsoid_boundary(cy - 2, cz - 18, 9, 6)
                ax.fill(aey, aez, color=TISSUE_COLORS[6], alpha=0.3, label="Amygdala")
                ax.plot(aey, aez, color="red", linewidth=1.5)
            else:
                for off in [-24, 24]:
                    aex, aez = _ellipsoid_boundary(cx + off, cz - 18, 5, 6)
                    ax.fill(aex, aez, color=TISSUE_COLORS[6], alpha=0.3)
                    ax.plot(aex, aez, color="red", linewidth=1.5)

            max_show = min(200, n_paths)
            rng = np.random.default_rng(42)
            indices = rng.choice(n_paths, max_show, replace=False) if n_paths > max_show else np.arange(n_paths)

            amyg_count = 0
            for idx in indices:
                nsteps = path_lens[idx]
                pts = positions[idx, :nsteps, :]
                ax_r = (pts[:, 0] - cx - 24) / 5
                ay_r = (pts[:, 1] - cy + 2) / 9
                az_r = (pts[:, 2] - cz + 18) / 6
                in_amyg = np.any(ax_r**2 + ay_r**2 + az_r**2 <= 1.0)

                if view == "sagittal":
                    px, py = pts[:, 1], pts[:, 2]
                else:
                    px, py = pts[:, 0], pts[:, 2]

                if in_amyg:
                    ax.plot(px, py, color="red", alpha=0.4, linewidth=0.8)
                    amyg_count += 1
                else:
                    ax.plot(px, py, color="blue", alpha=0.1, linewidth=0.3)

            lim_c = cy if view == "sagittal" else cx
            ax.set_xlim(lim_c - 100, lim_c + 100)
            ax.set_ylim(cz + 90, cz - 90)
            ax.set_xlabel("Y [mm]" if view == "sagittal" else "X [mm]")
            ax.set_ylabel("Z [mm]")
            title = "Sagittal" if view == "sagittal" else "Coronal"
            ax.set_title(f"{title} - {wl_key} ({max_show} paths, {amyg_count} reach amygdala)")
            ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(output_dir / f"10_photon_paths_{wl_key}.png", dpi=150, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# 11. CW vs TD sensitivity comparison
# ---------------------------------------------------------------------------
def plot_cw_vs_td(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, wl_key in zip(axes, WL_KEYS):
        dets = sorted(_primary_dets(results, wl_key), key=lambda d: d["sds_mm"])
        sds_vals = [d["sds_mm"] for d in dets]

        cw_amyg = []
        td_best_amyg = []
        td_best_gate = []

        for det in dets:
            cw_amyg.append(det["partial_pathlength_mm"].get("amygdala", 0))
            gates = det.get("time_gates", [])
            best_a = 0
            best_g = -1
            for g_idx, gate in enumerate(gates):
                a = gate.get("partial_pathlength_mm", {}).get("amygdala", 0)
                if a > best_a:
                    best_a = a
                    best_g = g_idx
            td_best_amyg.append(best_a)
            td_best_gate.append(best_g)

        x = np.arange(len(sds_vals))
        w = 0.35
        ax.bar(x - w/2, cw_amyg, w, label="CW (all gates)", color="#4FC3F7", edgecolor="black", linewidth=0.5)
        bars = ax.bar(x + w/2, td_best_amyg, w, label="TD best gate", color="#FF7043", edgecolor="black", linewidth=0.5)

        for i, (bar, g) in enumerate(zip(bars, td_best_gate)):
            if g >= 0 and td_best_amyg[i] > 0:
                ax.text(bar.get_x() + bar.get_width()/2, td_best_amyg[i],
                        GATE_LABELS[g], ha='center', va='bottom', fontsize=6, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:.0f}" for s in sds_vals], fontsize=8)
        ax.set_xlabel("SDS [mm]")
        ax.set_ylabel("Amygdala Partial Pathlength [mm]")
        ax.set_title(f"CW vs TD-Gated Amygdala Sensitivity - {wl_key}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CW vs TD-gated: amygdala partial pathlength gain", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "11_cw_vs_td_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 12. Dual-wavelength MBLL sensitivity map (all detectors, all gates)
# ---------------------------------------------------------------------------
def plot_mbll_heatmap(results, output_dir):
    N_ps = [_photons_per_sec(i) for i in range(2)]

    # Use all detectors (not just primary)
    all_dets_730 = results["730nm"]["detectors"]
    all_dets_850 = results["850nm"]["detectors"]

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    for ax, metric_idx, metric_name in zip(axes, [0, 1], ["dHbO", "dHbR"]):
        labels_y = []
        labels_x = GATE_LABELS
        matrix = []

        for d730, d850 in zip(all_dets_730, all_dets_850):
            sds = d730["sds_mm"]
            angle = d730.get("angle_deg", 0)
            labels_y.append(f"SDS{sds:.0f} {angle:+.0f}deg")

            row = []
            n_gates = min(len(d730.get("time_gates", [])), len(d850.get("time_gates", [])))
            for g_idx in range(n_gates):
                g730, g850 = d730["time_gates"][g_idx], d850["time_gates"][g_idx]
                L730 = g730.get("partial_pathlength_mm", {}).get("amygdala", 0)
                L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
                if L730 <= 0 or L850 <= 0:
                    row.append(np.nan)
                    continue

                gw730 = g730.get("weight", 0)
                gw850 = g850.get("weight", 0)
                scale730 = N_ps[0] / results["730nm"]["num_photons"]
                scale850 = N_ps[1] / results["850nm"]["num_photons"]
                N730 = gw730 * scale730 * MEAS_TIME * DET_QE["730nm"] + DARK_RATE * MEAS_TIME
                N850 = gw850 * scale850 * MEAS_TIME * DET_QE["850nm"] + DARK_RATE * MEAS_TIME
                n730 = 1/np.sqrt(N730) if N730 > 0 else float('inf')
                n850 = 1/np.sqrt(N850) if N850 > 0 else float('inf')

                E = np.array([[EPSILON_HBO[0]*L730, EPSILON_HBR[0]*L730],
                              [EPSILON_HBO[1]*L850, EPSILON_HBR[1]*L850]])
                if abs(np.linalg.det(E)) < 1e-20:
                    row.append(np.nan)
                    continue
                dc = np.abs(np.linalg.inv(E) @ np.array([n730, n850])) * 1e3
                row.append(dc[metric_idx])

            while len(row) < len(GATE_LABELS):
                row.append(np.nan)
            matrix.append(row)

        matrix = np.array(matrix)
        masked = np.ma.masked_invalid(matrix)

        im = ax.imshow(masked, aspect="auto", cmap="RdYlGn_r",
                        norm=LogNorm(vmin=0.01, vmax=100),
                        interpolation="nearest")
        ax.set_xticks(range(len(labels_x)))
        ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(labels_y)))
        ax.set_yticklabels(labels_y, fontsize=7)
        ax.set_title(f"Min Detectable {metric_name} [uM] ({MEAS_TIME:.0f}s)")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isnan(val):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=5, color="gray")
                else:
                    color = "white" if val > 5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

        fig.colorbar(im, ax=ax, label=f"Min {metric_name} [uM]", shrink=0.8)

    fig.suptitle(f"MBLL Sensitivity Map: All Detectors x Gates ({MEAS_TIME:.0f}s, 1W, Si-PMT)",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / "12_mbll_sensitivity_map.png", dpi=200, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="fNIRS MC 730/850nm Visualization")
    parser.add_argument("--data-dir", type=str, default="../data/730-850")
    parser.add_argument("--output-dir", type=str, default="../figures/730-850")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading 730/850nm data...")
    vol, fluence, results, tpsf, meta, paths = load_data(data_dir)
    print(f"  Volume: {'loaded' if vol is not None else 'skipped (large grid)'}")
    print(f"  Fluence: {', '.join(fluence.keys()) if fluence else 'none'}")
    print(f"  TPSF: {', '.join(f'{k}: {v.shape}' for k, v in tpsf.items())}")
    print(f"  Paths: {', '.join(k + ': ' + str(len(v['det_ids'])) for k, v in paths.items())}")

    print(f"\nGenerating figures in {output_dir}/...")

    if vol is not None:
        plot_tissue_slices(vol, meta, output_dir)
        print("  01_tissue_slices.png")
        plot_fluence(vol, fluence, meta, output_dir)
        print("  02_fluence_*.png")
    else:
        print("  (skipping tissue/fluence plots - volume too large)")

    plot_tpsf(results, tpsf, output_dir)
    print("  03_tpsf_curves.png")

    plot_sensitivity_heatmap(results, output_dir)
    print("  04_td_sensitivity_heatmap.png")

    plot_gate_counts(results, output_dir)
    print("  05_gate_photon_counts.png")

    plot_td_snr(results, output_dir)
    print("  06_td_snr_120s.png")

    plot_min_detectable(results, output_dir)
    print("  07_min_detectable_td.png")

    plot_integration_curve(results, output_dir)
    print("  08_integration_time_curve.png")

    plot_block_design(results, output_dir)
    print("  09_block_design_signal.png")

    if paths:
        plot_photon_paths(paths, results, meta, output_dir)
        print("  10_photon_paths_*.png")

    plot_cw_vs_td(results, output_dir)
    print("  11_cw_vs_td_sensitivity.png")

    plot_mbll_heatmap(results, output_dir)
    print("  12_mbll_sensitivity_map.png")

    print("\nDone! All figures saved to", output_dir)


if __name__ == "__main__":
    main()
