#!/usr/bin/env python3
"""
Validation against semi-infinite medium diffusion approximation.

This script validates the Monte Carlo simulation against the analytical
solution for a semi-infinite turbid medium, following the method of
Contini et al. (1997) Applied Optics.

Usage:
    python validate_diffusion.py --data-dir ../results
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def diffusion_reflectance_steady_state(rho, mu_a, mu_s_prime, n=1.4):
    """
    Steady-state diffuse reflectance for semi-infinite medium.
    
    Using the extrapolated boundary method with single point-source
    approximation (Contini et al., 1997).
    
    Parameters:
    -----------
    rho : float or array
        Source-detector separation [mm]
    mu_a : float
        Absorption coefficient [1/mm]
    mu_s_prime : float
        Reduced scattering coefficient [1/mm]
    n : float
        Refractive index
        
    Returns:
    --------
    Reflectance : float or array
        Diffuse reflectance [1/mm^2]
    """
    D = 1.0 / (3.0 * (mu_a + mu_s_prime))  # Diffusion coefficient
    mu_eff = np.sqrt(mu_a / D)  # Effective attenuation coefficient
    
    # Extrapolation length (approximate for mismatched boundaries)
    # Using simple approximation: z_b ≈ 2D * (1 + R_eff) / (1 - R_eff)
    # where R_eff is effective reflection coefficient
    R_eff = 0.493  # Approximate for n=1.4
    z_b = 2.0 * D * (1.0 + R_eff) / (1.0 - R_eff)
    
    # Source position (1 mean free path into medium)
    z_s = 1.0 / (mu_a + mu_s_prime)
    
    # Image source position
    z_i = z_s + 4.0 * z_b
    
    # Distances
    r1 = np.sqrt(rho**2 + z_s**2)
    r2 = np.sqrt(rho**2 + z_i**2)
    
    # Reflectance
    coeff = 1.0 / (4.0 * np.pi)
    term1 = z_s * (mu_eff * r1 + 1.0) * np.exp(-mu_eff * r1) / r1**3
    term2 = z_i * (mu_eff * r2 + 1.0) * np.exp(-mu_eff * r2) / r2**3
    
    R = coeff * (term1 + term2)
    
    return R


def validate_against_diffusion(results_730, results_850, output_dir):
    """
    Compare MC results against diffusion approximation.
    
    Uses scalp optical properties as approximation for semi-infinite medium.
    """
    # Optical properties at 730 nm (scalp)
    mu_a_730 = 0.0193  # 1/mm (from optical_properties.cu)
    mu_s_730 = 14.29   # 1/mm
    g_730 = 0.89
    mu_s_prime_730 = mu_s_730 * (1 - g_730)
    
    # Optical properties at 850 nm
    mu_a_850 = 0.0179
    mu_s_850 = 12.50
    g_850 = 0.89
    mu_s_prime_850 = mu_s_850 * (1 - g_850)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, results, wl_name, mu_a, mu_s_prime in zip(
        axes, 
        [results_730, results_850], 
        ["730nm", "850nm"],
        [mu_a_730, mu_a_850],
        [mu_s_prime_730, mu_s_prime_850]
    ):
        # Extract MC data
        detectors = results["detectors"]
        
        # Get primary direction detectors
        primary_dets = [d for d in detectors if abs(d.get("angle_deg", 0)) < 1]
        
        sds_mm = []
        mc_reflectance = []
        mc_error = []
        
        for det in primary_dets:
            sds = det["sds_mm"]
            weight = det["total_weight"]
            n_photons = results["num_photons"]
            
            # Reflectance = detected weight / (4π * SDS²) [approximate normalization]
            # More accurate: weight per unit detector area
            det_area = np.pi * (4.0**2)  # 4mm radius detector
            R_mc = weight / (n_photons * det_area)
            
            # Standard error (Poisson statistics)
            n_detected = det["detected_photons"]
            if n_detected > 0:
                se = R_mc / np.sqrt(n_detected)
            else:
                se = np.nan
            
            sds_mm.append(sds)
            mc_reflectance.append(R_mc)
            mc_error.append(se)
        
        sds_mm = np.array(sds_mm)
        mc_reflectance = np.array(mc_reflectance)
        mc_error = np.array(mc_error)
        
        # Compute diffusion approximation
        sds_fine = np.linspace(min(sds_mm), max(sds_mm), 100)
        R_diffusion = diffusion_reflectance_steady_state(sds_fine, mu_a, mu_s_prime)
        
        # Plot
        ax.errorbar(sds_mm, mc_reflectance, yerr=mc_error, 
                   fmt='o', markersize=6, capsize=3, 
                   label='Monte Carlo', color='blue', alpha=0.7)
        ax.plot(sds_fine, R_diffusion, '-', 
               label='Diffusion Approximation', color='red', linewidth=2)
        
        ax.set_xlabel("Source-Detector Separation [mm]")
        ax.set_ylabel("Reflectance [mm⁻²]")
        ax.set_title(f"Validation: {wl_name}")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Compute relative error at each SDS
        R_diffusion_at_mc = diffusion_reflectance_steady_state(sds_mm, mu_a, mu_s_prime)
        relative_error = np.abs(mc_reflectance - R_diffusion_at_mc) / R_diffusion_at_mc * 100
        
        print(f"\n{wl_name} Validation Results:")
        print(f"  {'SDS [mm]':>10s} {'MC [1e-4]':>12s} {'Diff [1e-4]':>12s} {'Rel. Error %':>12s}")
        print(f"  {'-'*50}")
        for sds, mc, diff, err in zip(sds_mm, mc_reflectance*1e4, R_diffusion_at_mc*1e4, relative_error):
            print(f"  {sds:10.1f} {mc:12.4f} {diff:12.4f} {err:12.1f}")
        
        # Summary statistics
        valid_errors = relative_error[~np.isnan(relative_error)]
        if len(valid_errors) > 0:
            print(f"\n  Mean relative error: {np.mean(valid_errors):.1f}%")
            print(f"  Max relative error: {np.max(valid_errors):.1f}%")
            
            # Diffusion is typically valid for SDS > 3 * l* (transport mean free path)
            l_star = 1.0 / (mu_a + mu_s_prime)
            valid_sds = sds_mm > 3 * l_star
            if np.any(valid_sds):
                valid_err = relative_error[valid_sds]
                print(f"  Mean error for SDS > {3*l_star:.1f}mm: {np.mean(valid_err):.1f}%")
    
    plt.tight_layout()
    plt.savefig(output_dir / "validation_diffusion.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nValidation figure saved to {output_dir}/validation_diffusion.png")


def main():
    parser = argparse.ArgumentParser(
        description="Validate MC against diffusion approximation")
    parser.add_argument("--data-dir", type=str, default="../results",
                       help="Directory containing simulation results")
    parser.add_argument("--output-dir", type=str, default="../figures",
                       help="Output directory for validation figure")
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
    else:
        print(f"Warning: {f730} not found")
    
    if f850.exists():
        with open(f850) as f:
            results_850 = json.load(f)
    else:
        print(f"Warning: {f850} not found")
    
    if results_730 is None and results_850 is None:
        print("Error: No result files found")
        return
    
    validate_against_diffusion(results_730, results_850, output_dir)


if __name__ == "__main__":
    main()
