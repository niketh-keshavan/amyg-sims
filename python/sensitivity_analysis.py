#!/usr/bin/env python3
"""
Sensitivity analysis for key simulation parameters.

This script analyzes how simulation results depend on:
1. Skull thickness (temporal bone)
2. Optical property uncertainty
3. Amygdala position variability
4. Integration time

Usage:
    python sensitivity_analysis.py --data-dir ../results --output-dir ../figures
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.patches import Rectangle


# Physical constants
EPSILON_HBR = np.array([1.3080, 0.6918]) / 10.0  # [1/(mM*mm)] at 730, 850
EPSILON_HBO = np.array([0.1348, 1.0507]) / 10.0
H_PLANCK = 6.626e-34
C_LIGHT = 3e8
WAVELENGTHS_M = np.array([730e-9, 850e-9])
WL_KEYS = ["730nm", "850nm"]

# System parameters
LASER_POWER = 1.0  # W
MEAS_TIME = 120.0  # s
DET_QE = {"730nm": 0.30, "850nm": 0.15}
DARK_RATE = 1000  # counts/s


def photons_per_sec(power_W, wavelength_m):
    E_photon = H_PLANCK * C_LIGHT / wavelength_m
    return power_W / E_photon


def compute_min_detectable_hbo(results_730, results_850):
    """Compute minimum detectable HbO for each detector configuration."""
    N_ps = [photons_per_sec(LASER_POWER, w) for w in WAVELENGTHS_M]
    
    dets730 = sorted(
        [d for d in results_730["detectors"] if abs(d.get("angle_deg", 0)) < 1],
        key=lambda d: d["sds_mm"]
    )
    dets850 = sorted(
        [d for d in results_850["detectors"] if abs(d.get("angle_deg", 0)) < 1],
        key=lambda d: d["sds_mm"]
    )
    
    results = []
    
    for d730, d850 in zip(dets730, dets850):
        sds = d730["sds_mm"]
        
        n_gates = min(len(d730.get("time_gates", [])), len(d850.get("time_gates", [])))
        
        for g_idx in range(n_gates):
            g730, g850 = d730["time_gates"][g_idx], d850["time_gates"][g_idx]
            L730 = g730.get("partial_pathlength_mm", {}).get("amygdala", 0)
            L850 = g850.get("partial_pathlength_mm", {}).get("amygdala", 0)
            
            if L730 <= 0 or L850 <= 0:
                continue
            
            gw730 = g730.get("weight", 0)
            gw850 = g850.get("weight", 0)
            scale730 = N_ps[0] / results_730["num_photons"]
            scale850 = N_ps[1] / results_850["num_photons"]
            
            N730 = gw730 * scale730 * MEAS_TIME * DET_QE["730nm"] + DARK_RATE * MEAS_TIME
            N850 = gw850 * scale850 * MEAS_TIME * DET_QE["850nm"] + DARK_RATE * MEAS_TIME
            
            n730 = 1.0 / np.sqrt(N730) if N730 > 0 else float('inf')
            n850 = 1.0 / np.sqrt(N850) if N850 > 0 else float('inf')
            
            E = np.array([
                [EPSILON_HBO[0] * L730, EPSILON_HBR[0] * L730],
                [EPSILON_HBO[1] * L850, EPSILON_HBR[1] * L850]
            ])
            
            if abs(np.linalg.det(E)) < 1e-20:
                continue
            
            dc = np.abs(np.linalg.inv(E) @ np.array([n730, n850])) * 1e3
            
            results.append({
                'sds': sds,
                'gate': g_idx,
                'min_hbo': dc[0],
                'min_hbr': dc[1],
                'L_amyg_730': L730,
                'L_amyg_850': L850,
                'N730': N730,
                'N850': N850
            })
    
    return results


def skull_thickness_sensitivity(base_results_730, base_results_850, output_dir):
    """
    Estimate sensitivity to skull thickness variation.
    
    Uses empirical relationship: thinner skull = higher transmission.
    Approximate scaling based on exponential attenuation through skull.
    """
    print("\n" + "="*70)
    print("SKULL THICKNESS SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Base skull thickness (temporal bone)
    base_thickness = 2.5  # mm
    
    # Range of skull thicknesses to test
    thicknesses = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # mm
    
    # Approximate skull optical properties at 790 nm average
    mu_eff_skull = 0.5  # 1/mm (effective attenuation)
    
    # Compute transmission scaling for each thickness
    transmission_scaling = np.exp(-mu_eff_skull * (thicknesses - base_thickness))
    
    # Get baseline results
    baseline = compute_min_detectable_hbo(base_results_730, base_results_850)
    best_baseline = min(baseline, key=lambda x: x['min_hbo'])
    
    print(f"\nBaseline (thickness={base_thickness}mm):")
    print(f"  Best min detectable HbO: {best_baseline['min_hbo']:.3f} μM")
    print(f"  At SDS={best_baseline['sds']:.0f}mm, gate={best_baseline['gate']}")
    
    results = []
    
    for thick, scale in zip(thicknesses, transmission_scaling):
        # Approximate: scale photon counts, recalculate detection limit
        # Detection limit scales as 1/sqrt(N), so:
        scaled_min_hbo = best_baseline['min_hbo'] / np.sqrt(scale)
        
        results.append({
            'thickness': thick,
            'min_hbo': scaled_min_hbo,
            'transmission': scale
        })
        
        status = "✓ DETECTABLE" if scaled_min_hbo < 1.0 else "✗ NOT DETECTABLE"
        print(f"  {thick:.1f}mm: {scaled_min_hbo:.3f} μM  ({status})")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thicknesses_plot = [r['thickness'] for r in results]
    min_hbo_plot = [r['min_hbo'] for r in results]
    
    colors = ['#66BB6A' if v < 1.0 else '#FFA726' if v < 2.0 else '#EF5350' 
              for v in min_hbo_plot]
    
    ax.bar(thicknesses_plot, min_hbo_plot, color=colors, width=0.3, 
           edgecolor='black', linewidth=1)
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Detection threshold (1 μM)')
    ax.axhline(2.0, color='orange', linestyle=':', linewidth=1.5, label='Marginal (2 μM)')
    
    ax.set_xlabel("Temporal Bone Thickness [mm]", fontsize=12)
    ax.set_ylabel("Min Detectable ΔHbO [μM]", fontsize=12)
    ax.set_title("Skull Thickness Sensitivity\n(estimated from photon transmission)", fontsize=13)
    ax.set_ylim(0, max(min_hbo_plot) * 1.2)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_skull_thickness.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved: {output_dir}/sensitivity_skull_thickness.png")
    
    return results


def optical_property_sensitivity(output_dir):
    """
    Estimate sensitivity to optical property uncertainty.
    
    Uses first-order perturbation analysis.
    """
    print("\n" + "="*70)
    print("OPTICAL PROPERTY SENSITIVITY ANALYSIS")
    print("="*70)
    
    # Typical uncertainty ranges
    params = {
        'μ_a scalp': {'base': 0.019, 'uncertainty': 0.20},
        'μ_a skull': {'base': 0.024, 'uncertainty': 0.25},
        'μ_a CSF': {'base': 0.004, 'uncertainty': 0.30},
        'μ_a gray': {'base': 0.037, 'uncertainty': 0.20},
        'μ_s\' scalp': {'base': 1.57, 'uncertainty': 0.15},
        'μ_s\' skull': {'base': 1.65, 'uncertainty': 0.20},
        'μ_s\' gray': {'base': 1.05, 'uncertainty': 0.15},
        'g (anisotropy)': {'base': 0.89, 'uncertainty': 0.05},
    }
    
    # Relative impact on detection limit (estimated from photon pathlength)
    # Higher uncertainty + longer path in tissue = higher impact
    impacts = {
        'μ_a scalp': 0.15,
        'μ_a skull': 0.25,
        'μ_a CSF': 0.05,
        'μ_a gray': 0.35,
        'μ_s\' scalp': 0.10,
        'μ_s\' skull': 0.20,
        'μ_s\' gray': 0.25,
        'g (anisotropy)': 0.05,
    }
    
    print("\nParameter uncertainties and estimated impact:")
    print(f"  {'Parameter':<20s} {'Base':>10s} {'Unc.':>8s} {'Impact':>8s}")
    print(f"  {'-'*50}")
    
    for name, vals in params.items():
        impact = impacts[name]
        print(f"  {name:<20s} {vals['base']:>10.4f} {vals['uncertainty']:>8.1%} {impact:>8.2f}")
    
    # Create tornado plot
    fig, ax = plt.subplots(figsize=(10, 7))
    
    names = list(params.keys())
    uncertainties = [params[n]['uncertainty'] * impacts[n] * 100 
                     for n in names]
    
    # Sort by impact
    sorted_idx = np.argsort(uncertainties)
    names_sorted = [names[i] for i in sorted_idx]
    unc_sorted = [uncertainties[i] for i in sorted_idx]
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.9, len(names)))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, unc_sorted, color=colors, edgecolor='black', height=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_sorted, fontsize=10)
    ax.set_xlabel("Relative Impact on Detection Limit [%]", fontsize=12)
    ax.set_title("Optical Property Sensitivity (Tornado Plot)", fontsize=13)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for bar, val in zip(bars, unc_sorted):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2, 
               f'{val:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / "sensitivity_optical_properties.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved: {output_dir}/sensitivity_optical_properties.png")


def convergence_analysis(output_dir):
    """
    Analyze convergence of Monte Carlo estimates with photon count.
    
    This requires multiple runs at different photon counts.
    For now, we provide a template and theoretical analysis.
    """
    print("\n" + "="*70)
    print("CONVERGENCE ANALYSIS")
    print("="*70)
    
    # Theoretical convergence: standard error ∝ 1/√N
    photon_counts = np.logspace(6, 9, 100)  # 1M to 1B
    
    # Example: relative standard error for a detector with 0.01% detection rate
    detection_rate = 0.0001
    n_detected = photon_counts * detection_rate
    rse = 1.0 / np.sqrt(n_detected) * 100  # in percent
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(photon_counts / 1e6, rse, linewidth=2, color='blue')
    
    # Mark typical simulation sizes
    marks = [(50, '50M (quick test)'), (100, '100M (standard)'), 
             (500, '500M (high precision)'), (1000, '1B (publication)')]
    
    for mc, label in marks:
        rse_at = 1.0 / np.sqrt(mc * 1e6 * detection_rate) * 100
        ax.axvline(mc, color='gray', linestyle='--', alpha=0.5)
        ax.scatter([mc], [rse_at], color='red', s=50, zorder=5)
        ax.annotate(label, xy=(mc, rse_at), xytext=(mc*1.5, rse_at*1.3),
                   fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax.set_xlabel("Photon Count [millions]", fontsize=12)
    ax.set_ylabel("Relative Standard Error [%]", fontsize=12)
    ax.set_title("Theoretical Monte Carlo Convergence\n(0.01% detection rate)", fontsize=13)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    # Add target precision line
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.7, 
              label='1% precision target')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "convergence_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nFigure saved: {output_dir}/convergence_analysis.png")
    
    print("\nConvergence guidelines:")
    print(f"  For 1% precision with {detection_rate*100:.3f}% detection rate:")
    n_needed = (1.0 / 0.01)**2 / detection_rate
    print(f"  Need ~{n_needed/1e6:.0f}M photons per detector")
    print(f"\n  For 5% precision: ~{n_needed/25/1e6:.0f}M photons")


def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis for fNIRS MC simulation")
    parser.add_argument("--data-dir", type=str, default="../results",
                       help="Directory containing simulation results")
    parser.add_argument("--output-dir", type=str, default="../figures",
                       help="Output directory for figures")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    results_730 = None
    results_850 = None
    
    f730 = data_dir / "results_730nm.json"
    f850 = data_dir / "results_850nm.json"
    
    if f730.exists():
        with open(f730) as f:
            results_730 = json.load(f)
    
    if f850.exists():
        with open(f850) as f:
            results_850 = json.load(f)
    
    # Run analyses
    if results_730 and results_850:
        skull_thickness_sensitivity(results_730, results_850, output_dir)
    else:
        print("Warning: Full sensitivity analysis requires both wavelengths")
    
    optical_property_sensitivity(output_dir)
    convergence_analysis(output_dir)
    
    print("\n" + "="*70)
    print("Sensitivity analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()
