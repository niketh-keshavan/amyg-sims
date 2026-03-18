#!/usr/bin/env python3
"""Analysis script for MMC mesh-based simulation results"""

import json
import sys
import numpy as np
from pathlib import Path

def analyze_mmc_results(input_dir, output_dir=None):
    """Analyze MMC simulation results"""
    input_path = Path(input_dir)
    
    # Load results
    try:
        with open(input_path / 'results_730nm.json') as f:
            r730 = json.load(f)
        with open(input_path / 'results_850nm.json') as f:
            r850 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Looking for: {input_path}/results_730nm.json and results_850nm.json")
        return
    
    print("="*70)
    print("MMC MESH-BASED fNIRS SIMULATION RESULTS")
    print("="*70)
    print(f"\nSimulation Parameters:")
    print(f"  Photons: {r730['num_photons']:,} per wavelength")
    print(f"  Model: {r730.get('geometry_model', 'MMC MNI152')}")
    print(f"  Wavelengths: 730nm, 850nm")
    print(f"  Time gates: {r730.get('time_gate_edges_ps', 'N/A')}")
    
    print("\n" + "="*70)
    print("DETECTOR SUMMARY - Continuous Wave (CW)")
    print("="*70)
    print(f"{'Det':<5} {'SDS':<6} {'Angle':<7} {'Detected':<12} {'Weight':<12} {'MeanPL':<8} {'AmygPL':<10}")
    print("-"*70)
    
    for det in r730['detectors']:
        det_id = det['id']
        sds = det['sds_mm']
        angle = det.get('angle_deg', 0)
        detected = det['detected_photons']
        weight = det['total_weight']
        mean_pl = det.get('mean_pathlength_mm', 0)
        amyg_pl = det.get('partial_pathlength_mm', {}).get('amygdala', 0)
        print(f"{det_id:<5} {sds:<6.0f} {angle:<7.0f} {detected:<12,} {weight:<12.4e} {mean_pl:<8.2f} {amyg_pl:<10.6f}")
    
    print("\n" + "="*70)
    print("TIME-GATED AMYGDALA ANALYSIS (730nm)")
    print("="*70)
    
    # Find best detectors for amygdala
    best_detectors = []
    for det in r730['detectors']:
        det_id = det['id']
        sds = det['sds_mm']
        angle = det.get('angle_deg', 0)
        
        # Skip angled detectors for primary analysis
        if abs(angle) > 5:
            continue
            
        # Find best gate
        best_gate_amyg = 0
        best_gate_idx = -1
        for g, gate in enumerate(det.get('time_gates', [])):
            amyg_pl = gate.get('partial_pathlength_mm', {}).get('amygdala', 0)
            if amyg_pl > best_gate_amyg:
                best_gate_amyg = amyg_pl
                best_gate_idx = g
        
        if best_gate_amyg > 0.0001:
            best_detectors.append({
                'id': det_id,
                'sds': sds,
                'angle': angle,
                'best_gate': best_gate_idx,
                'amyg_pl': best_gate_amyg
            })
    
    # Sort by amygdala pathlength
    best_detectors.sort(key=lambda x: x['amyg_pl'], reverse=True)
    
    print(f"\n{'Rank':<5} {'Det':<5} {'SDS':<6} {'Gate':<6} {'AmygPL(mm)':<12} {'Status':<15}")
    print("-"*50)
    for i, d in enumerate(best_detectors[:10], 1):
        status = "GOOD" if d['amyg_pl'] > 0.01 else "LOW" if d['amyg_pl'] > 0.001 else "VERY LOW"
        print(f"{i:<5} {d['id']:<5} {d['sds']:<6.0f} {d['best_gate']:<6} {d['amyg_pl']:<12.6f} {status:<15}")
    
    if not best_detectors:
        print("\n⚠️  WARNING: No significant amygdala signal detected!")
        print("   Possible causes:")
        print("   - Mesh resolution too coarse (try smaller max_vol)")
        print("   - Source not positioned near amygdala")
        print("   - Amygdala tissue label not properly assigned in mesh")
    
    print("\n" + "="*70)
    print("KEY METRICS")
    print("="*70)
    
    total_detected_730 = sum(d['detected_photons'] for d in r730['detectors'])
    total_detected_850 = sum(d['detected_photons'] for d in r850['detectors'])
    
    print(f"\nTotal detected photons:")
    print(f"  730nm: {total_detected_730:,}")
    print(f"  850nm: {total_detected_850:,}")
    
    if best_detectors:
        best = best_detectors[0]
        print(f"\nBest detector: #{best['id']} at SDS={best['sds']:.0f}mm")
        print(f"  Amygdala pathlength: {best['amyg_pl']:.6f} mm")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze MMC simulation results')
    parser.add_argument('--input', '-i', default='data_mmc_maxvol5', help='Input directory')
    parser.add_argument('--output', '-o', help='Output directory for figures (optional)')
    args = parser.parse_args()
    
    analyze_mmc_results(args.input, args.output)
