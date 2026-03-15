#!/usr/bin/env python3
"""
Analyze MMC simulation results for amygdala sensitivity.

This script processes the TPSF and path record outputs from the MMC simulation
to compute sensitivity metrics, SNR estimates, and compare modalities (CW, TD, FD).
"""

import argparse
import json
import struct
import numpy as np
from pathlib import Path


def load_tpsf(filepath: str) -> tuple:
    """Load TPSF histogram from binary file.
    
    Returns:
        (num_detectors, num_bins, tpsf_array)
    """
    with open(filepath, 'rb') as f:
        num_dets = struct.unpack('i', f.read(4))[0]
        num_bins = struct.unpack('i', f.read(4))[0]
        data = np.fromfile(f, dtype=np.float32)
    
    tpsf = data.reshape((num_dets, num_bins))
    return num_dets, num_bins, tpsf


def load_path_records(filepath: str) -> dict:
    """Load photon path records from binary file.
    
    Record format (from types.h):
        float3 exit_pos (12 bytes)
        float3 exit_dir (12 bytes)
        float total_pathlen
        float amyg_pathlen
        float time_of_flight
        float weight
        int32 bounces
        int32 hit_amyg
    """
    record_dtype = np.dtype([
        ('exit_x', np.float32),
        ('exit_y', np.float32),
        ('exit_z', np.float32),
        ('dir_x', np.float32),
        ('dir_y', np.float32),
        ('dir_z', np.float32),
        ('total_pathlen', np.float32),
        ('amyg_pathlen', np.float32),
        ('tof', np.float32),
        ('weight', np.float32),
        ('bounces', np.int32),
        ('hit_amyg', np.int32),
    ])
    
    with open(filepath, 'rb') as f:
        count = struct.unpack('Q', f.read(8))[0]
        data = np.fromfile(f, dtype=record_dtype)
    
    return {'count': count, 'records': data}


def analyze_sensitivity(records_760: dict, records_850: dict) -> dict:
    """Compute sensitivity metrics from path records."""
    
    results = {}
    
    for wl, records in [('760nm', records_760), ('850nm', records_850)]:
        if records is None or records['count'] == 0:
            continue
            
        rec = records['records']
        total_detected = len(rec)
        amyg_hits = np.sum(rec['hit_amyg'])
        
        # Detection probability
        det_prob = amyg_hits / total_detected if total_detected > 0 else 0
        
        # Path length statistics
        if amyg_hits > 0:
            amyg_paths = rec[rec['hit_amyg'] == 1]['amyg_pathlen']
            mean_amyg_path = np.mean(amyg_paths)
            std_amyg_path = np.std(amyg_paths)
        else:
            mean_amyg_path = 0
            std_amyg_path = 0
        
        # Time of flight for amygdala photons
        if amyg_hits > 0:
            amyg_tof = rec[rec['hit_amyg'] == 1]['tof']
            mean_tof = np.mean(amyg_tof)
        else:
            mean_tof = 0
        
        results[wl] = {
            'total_detected': int(total_detected),
            'amygdala_hits': int(amyg_hits),
            'detection_probability': float(det_prob),
            'mean_amygdala_pathlen_mm': float(mean_amyg_path),
            'std_amygdala_pathlen_mm': float(std_amyg_path),
            'mean_time_of_flight_ps': float(mean_tof),
        }
    
    return results


def analyze_tpsf(tpsf: np.ndarray, dt_ps: float = 10.0) -> dict:
    """Analyze TPSF histogram."""
    
    num_dets, num_bins = tpsf.shape
    time_axis = np.arange(num_bins) * dt_ps
    
    results = []
    
    for d in range(num_dets):
        tpsf_d = tpsf[d, :]
        total_counts = np.sum(tpsf_d)
        
        if total_counts > 0:
            # Mean arrival time
            mean_time = np.sum(time_axis * tpsf_d) / total_counts
            
            # Variance
            var_time = np.sum(((time_axis - mean_time) ** 2) * tpsf_d) / total_counts
            
            # Late photons (time-gated sensitivity)
            late_threshold = 1000  # ps
            late_counts = np.sum(tpsf_d[time_axis > late_threshold])
            late_fraction = late_counts / total_counts
        else:
            mean_time = 0
            var_time = 0
            late_fraction = 0
        
        results.append({
            'detector': d,
            'total_counts': float(total_counts),
            'mean_time_ps': float(mean_time),
            'std_time_ps': float(np.sqrt(var_time)),
            'late_fraction': float(late_fraction),
        })
    
    return {
        'detectors': results,
        'time_axis_ps': time_axis.tolist(),
    }


def compute_mdll(sensitivity: dict, snr: float = 100.0) -> dict:
    """Compute minimum detectable hemoglobin concentration change.
    
    Uses the Modified Beer-Lambert Law (MBLL) to estimate minimum
    detectable ΔHbO and ΔHbR given sensitivity and SNR.
    
    Args:
        sensitivity: Dict with mean_amygdala_pathlen_mm for each wavelength
        snr: Signal-to-noise ratio of the measurement
        
    Returns:
        Dict with MDLL in μM for HbO and HbR
    """
    # Extinction coefficients (mM^-1 cm^-1)
    # From Prahl: http://omlc.org/spectra/hemoglobin/summary.html
    e_hbo_760 = 0.536
    e_hbo_850 = 1.058
    e_hbr_760 = 1.356
    e_hbr_850 = 0.691
    
    results = {}
    
    for wl, data in sensitivity.items():
        pathlen_cm = data['mean_amygdala_pathlen_mm'] / 10.0  # Convert to cm
        
        if pathlen_cm > 0:
            # Minimum detectable absorption change
            min_dod = 1.0 / snr  # Approximate from shot noise
            min_dmu = min_dod / pathlen_cm  # cm^-1
            
            results[wl] = {
                'min_delta_od': float(min_dod),
                'pathlen_cm': float(pathlen_cm),
                'min_delta_mua_cm': float(min_dmu),
            }
    
    # Two-wavelength solution if both available
    if '760nm' in sensitivity and '850nm' in sensitivity:
        L1 = sensitivity['760nm']['mean_amygdala_pathlen_mm'] / 10.0
        L2 = sensitivity['850nm']['mean_amygdala_pathlen_mm'] / 10.0
        
        if L1 > 0 and L2 > 0:
            # Matrix form: [ΔOD1]   [e1_hbo  e1_hbr] [L1  0 ] [ΔHbO]
            #              [ΔOD2] = [e2_hbo  e2_hbr] [0   L2] [ΔHbR]
            
            E = np.array([[e_hbo_760, e_hbr_760],
                          [e_hbo_850, e_hbr_850]])
            L = np.diag([L1, L2])
            
            # Sensitivity matrix
            S = E @ L
            
            # Minimum detectable concentration (assuming equal SNR at both wavelengths)
            min_dod = 1.0 / snr
            dOD = np.array([min_dod, min_dod])
            
            try:
                dC = np.linalg.inv(S) @ dOD
                results['concentration'] = {
                    'min_delta_hbo_uM': float(abs(dC[0]) * 1000),  # Convert mM to uM
                    'min_delta_hbr_uM': float(abs(dC[1]) * 1000),
                    'snr_assumed': snr,
                }
            except np.linalg.LinAlgError:
                pass
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze MMC fNIRS simulation results'
    )
    parser.add_argument('--data-dir', default='results_mmc',
                       help='Directory containing simulation outputs')
    parser.add_argument('--output', default='mmc_analysis.json',
                       help='Output JSON file for results')
    parser.add_argument('--snr', type=float, default=100.0,
                       help='Assumed SNR for MDLL calculation')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    print(f"Analyzing MMC results from: {data_dir}")
    
    # Load data
    results = {}
    
    # Load TPSF
    tpsf_760_path = data_dir / 'tpsf_760mm.bin'
    tpsf_850_path = data_dir / 'tpsf_850mm.bin'
    
    if tpsf_760_path.exists():
        print("Loading 760 nm TPSF...")
        num_dets, num_bins, tpsf_760 = load_tpsf(str(tpsf_760_path))
        results['tpsf_760nm'] = analyze_tpsf(tpsf_760)
        print(f"  Detectors: {num_dets}, Bins: {num_bins}")
    
    if tpsf_850_path.exists():
        print("Loading 850 nm TPSF...")
        num_dets, num_bins, tpsf_850 = load_tpsf(str(tpsf_850_path))
        results['tpsf_850nm'] = analyze_tpsf(tpsf_850)
        print(f"  Detectors: {num_dets}, Bins: {num_bins}")
    
    # Load path records
    paths_760_path = data_dir / 'paths_760mm.bin'
    paths_850_path = data_dir / 'paths_850mm.bin'
    
    records_760 = None
    records_850 = None
    
    if paths_760_path.exists():
        print("Loading 760 nm path records...")
        records_760 = load_path_records(str(paths_760_path))
        print(f"  Records: {records_760['count']}")
    
    if paths_850_path.exists():
        print("Loading 850 nm path records...")
        records_850 = load_path_records(str(paths_850_path))
        print(f"  Records: {records_850['count']}")
    
    # Analyze sensitivity
    print("\nAnalyzing amygdala sensitivity...")
    sensitivity = analyze_sensitivity(records_760, records_850)
    results['sensitivity'] = sensitivity
    
    for wl, data in sensitivity.items():
        print(f"\n  {wl}:")
        print(f"    Detection probability: {data['detection_probability']:.4f}")
        print(f"    Mean amygdala path: {data['mean_amygdala_pathlen_mm']:.2f} mm")
        print(f"    Mean TOF: {data['mean_time_of_flight_ps']:.1f} ps")
    
    # Compute MDLL
    print(f"\nComputing minimum detectable concentrations (SNR={args.snr})...")
    mdll = compute_mdll(sensitivity, args.snr)
    results['mdll'] = mdll
    
    if 'concentration' in mdll:
        c = mdll['concentration']
        print(f"  Min ΔHbO: {c['min_delta_hbo_uM']:.3f} μM")
        print(f"  Min ΔHbR: {c['min_delta_hbr_uM']:.3f} μM")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
