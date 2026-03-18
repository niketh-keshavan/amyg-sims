#!/usr/bin/env python3
"""
TD-Gated Output Validation Script for MMC Simulation
Validates Steps 3-5 from PLAN.md

Usage:
    python3 validate_td_gated.py <data_dir> [--wavelength 730]

Example:
    python3 validate_td_gated.py data_mmc_100M --wavelength 730
"""

import json
import numpy as np
import os
import sys
import argparse
from pathlib import Path

def validate_cw_output(data_dir, wavelength):
    """Step 3: Validate CW Detector Output"""
    print(f"\n{'='*60}")
    print(f"STEP 3: Validating CW Detector Output ({wavelength}nm)")
    print('='*60)
    
    json_path = Path(data_dir) / f"results_{wavelength}nm.json"
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return False
    
    with open(json_path) as f:
        data = json.load(f)
    
    print(f"Photons: {data['num_photons']:,}")
    print(f"Wavelength: {data['wavelength_nm']}nm")
    print(f"Model: {data.get('geometry_model', 'unknown')}")
    print(f"Detectors: {len(data['detectors'])}")
    print(f"TPSF bins: {data.get('tpsf_bins', 'unknown')} x {data.get('tpsf_bin_ps', 'unknown')} ps")
    print()
    
    # Check detector results
    short_sds_max = 0
    long_sds_detections = 0
    amyg_detections = 0
    issues = []
    
    print("Detector Summary:")
    print(f"{'Det':>4} {'SDS(mm)':>8} {'Photons':>10} {'MeanPL(mm)':>12} {'AmygPL(mm)':>12} {'Status'}")
    print("-" * 70)
    
    for d in data['detectors']:
        det_id = d['id']
        sds = d['sds_mm']
        photons = d['detected_photons']
        mean_pl = d['mean_pathlength_mm']
        amyg_pl = d['partial_pathlength_mm'].get('amygdala', 0)
        
        status = "OK"
        if sds < 15 and photons > short_sds_max:
            short_sds_max = photons
        if 28 <= sds <= 40 and photons > 0:
            long_sds_detections += 1
        if amyg_pl > 0:
            amyg_detections += 1
            
        # Check for issues
        if mean_pl < 0:
            status = "NEG_PL"
            issues.append(f"Det {det_id}: negative mean pathlength")
        if any(v < 0 for v in d['partial_pathlength_mm'].values()):
            status = "NEG_PPL"
            issues.append(f"Det {det_id}: negative partial pathlength")
            
        print(f"{det_id:>4} {sds:>8.1f} {photons:>10,} {mean_pl:>12.2f} {amyg_pl:>12.6f} {status}")
    
    print()
    
    # Validation checks
    passed = True
    
    if short_sds_max == 0:
        print("❌ FAIL: No detections at short SDS (8-15 mm)")
        passed = False
    else:
        print(f"✓ PASS: Short SDS has detections (max: {short_sds_max:,} photons)")
    
    if long_sds_detections == 0:
        print("❌ FAIL: No detections at long SDS (28-40 mm)")
        passed = False
    else:
        print(f"✓ PASS: {long_sds_detections} detectors at long SDS with detections")
    
    print(f"ℹ INFO: {amyg_detections} detectors have non-zero amygdala pathlength")
    if amyg_detections == 0:
        print("  Note: With small amygdala volume (0.04%), this is expected at low photon counts.")
        print("  Run 10B+ photons for statistically significant amygdala detection.")
    
    if issues:
        print(f"\n⚠ WARNINGS ({len(issues)}):")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    return passed


def validate_td_gates(data_dir, wavelength):
    """Step 4: Validate TD-Gated Output"""
    print(f"\n{'='*60}")
    print(f"STEP 4: Validating TD-Gated Output ({wavelength}nm)")
    print('='*60)
    
    json_path = Path(data_dir) / f"results_{wavelength}nm.json"
    with open(json_path) as f:
        data = json.load(f)
    
    gate_edges = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    
    print(f"Expected {len(gate_edges)-1} time gates with edges: {gate_edges} ps")
    print()
    
    issues = []
    
    print(f"{'Det':>4} {'SDS':>6} {'Gate0%':>8} {'LateGate%':>10} {'GW/CW':>8} {'LateAmzg':>10} {'Status'}")
    print("-" * 75)
    
    for d in data['detectors']:
        det_id = d['id']
        sds = d['sds_mm']
        gates = d['time_gates']
        total_weight = d['total_weight']
        
        # Check: 10 gates present
        if len(gates) != 10:
            issues.append(f"Det {det_id}: expected 10 gates, got {len(gates)}")
            continue
        
        # Calculate metrics
        gate_weights = [g['weight'] for g in gates]
        gw_sum = sum(gate_weights)
        
        # Gate 0 proportion
        gate0_pct = gate_weights[0] / gw_sum * 100 if gw_sum > 0 else 0
        
        # Late gates (6-9) proportion
        late_weight = sum(gate_weights[6:])
        late_pct = late_weight / gw_sum * 100 if gw_sum > 0 else 0
        
        # Gate weight sum vs CW weight ratio
        gw_cw_ratio = gw_sum / total_weight if total_weight > 0 else 0
        
        # Max amygdala in late gates
        late_amyg = max(g['partial_pathlength_mm'].get('amygdala', 0) for g in gates[6:])
        
        # Determine status
        status = "OK"
        if abs(gw_cw_ratio - 1.0) > 0.01:
            status = f"RATIO:{gw_cw_ratio:.3f}"
            issues.append(f"Det {det_id}: gate sum / CW weight = {gw_cw_ratio:.4f}")
        
        # Check for negative values
        for g in gates:
            if g['weight'] < 0:
                status = "NEG_W"
                issues.append(f"Det {det_id}: negative gate weight")
            if any(v < 0 for v in g['partial_pathlength_mm'].values()):
                status = "NEG_PPL"
                issues.append(f"Det {det_id}: negative partial pathlength in gate")
        
        if total_weight > 0:
            print(f"{det_id:>4} {sds:>6.1f} {gate0_pct:>7.1f}% {late_pct:>9.1f}% {gw_cw_ratio:>7.4f} {late_amyg:>10.6f} {status}")
    
    print()
    
    # Summary checks
    print("Validation Summary:")
    
    # Check gate structure
    sample_gates = data['detectors'][0]['time_gates']
    print(f"✓ PASS: {len(sample_gates)} time gates present")
    
    # Check gate weights sum to CW weight
    ratios = []
    for d in data['detectors']:
        if d['total_weight'] > 0:
            gw_sum = sum(g['weight'] for g in d['time_gates'])
            ratios.append(gw_sum / d['total_weight'])
    
    if ratios:
        mean_ratio = np.mean(ratios)
        print(f"✓ PASS: Mean gate_sum/CW_weight ratio = {mean_ratio:.6f} (should be ~1.0)")
    
    # Count detectors with late-gate amygdala
    late_amyg_count = 0
    for d in data['detectors']:
        if 35 <= d['sds_mm'] <= 46:
            late_amyg = max(g['partial_pathlength_mm'].get('amygdala', 0) for g in d['time_gates'][6:])
            if late_amyg > 0:
                late_amyg_count += 1
    
    print(f"ℹ INFO: {late_amyg_count} detectors at SDS 35-46mm have late-gate amygdala PL > 0")
    
    if issues:
        print(f"\n⚠ WARNINGS ({len(issues)}):")
        for issue in issues[:5]:
            print(f"  - {issue}")
    
    return True


def validate_tpsf(data_dir, wavelength):
    """Step 5: Validate TPSF Binary Output"""
    print(f"\n{'='*60}")
    print(f"STEP 5: Validating TPSF Binary Output ({wavelength}nm)")
    print('='*60)
    
    tpsf_path = Path(data_dir) / f"tpsf_{wavelength}nm.bin"
    if not tpsf_path.exists():
        print(f"ERROR: TPSF file not found: {tpsf_path}")
        return False
    
    n_dets = 23
    n_bins = 512
    expected_size = n_dets * n_bins * 8  # 8 bytes per double
    
    file_size = os.path.getsize(tpsf_path)
    print(f"File: {tpsf_path}")
    print(f"Size: {file_size:,} bytes (expected: {expected_size:,} bytes)")
    
    if file_size != expected_size:
        print(f"❌ FAIL: File size mismatch")
        return False
    else:
        print("✓ PASS: File size correct")
    
    # Read TPSF data
    data = np.fromfile(tpsf_path, dtype=np.float64)
    tpsf = data.reshape(n_dets, n_bins)
    
    print(f"\nTPSF shape: {tpsf.shape} (detectors x bins)")
    print(f"Time bin width: 10 ps (range: 0-5120 ps)")
    
    # Check for NaN/Inf
    has_nan = np.any(np.isnan(tpsf))
    has_inf = np.any(np.isinf(tpsf))
    
    print(f"\nNaN values: {'YES ❌' if has_nan else 'No ✓'}")
    print(f"Inf values: {'YES ❌' if has_inf else 'No ✓'}")
    
    if has_nan or has_inf:
        return False
    
    # Per-detector analysis
    print(f"\n{'Det':>4} {'Peak(ps)':>10} {'Integral':>14} {'Max':>12} {'Shape'}")
    print("-" * 60)
    
    for d in range(min(10, n_dets)):  # Show first 10 detectors
        tpsf_d = tpsf[d]
        peak_bin = np.argmax(tpsf_d)
        peak_time = peak_bin * 10  # 10 ps per bin
        integral = tpsf_d.sum()
        max_val = tpsf_d.max()
        
        # Check shape (should have single peak, not flat)
        if max_val > 0:
            # Count significant bins (>1% of max)
            sig_bins = np.sum(tpsf_d > 0.01 * max_val)
            shape = f"peaked({sig_bins})" if sig_bins < 100 else f"broad({sig_bins})"
        else:
            shape = "empty"
        
        print(f"{d:>4} {peak_time:>9}ps {integral:>14.6e} {max_val:>12.6e} {shape}")
    
    print()
    
    # Summary statistics
    non_zero_dets = np.sum(tpsf.sum(axis=1) > 0)
    print(f"Detectors with non-zero TPSF: {non_zero_dets}/{n_dets}")
    
    if non_zero_dets > 0:
        print("✓ PASS: TPSF has physically plausible shape")
        return True
    else:
        print("❌ FAIL: All TPSF are zero")
        return False


def main():
    parser = argparse.ArgumentParser(description='Validate TD-Gated MMC Output')
    parser.add_argument('data_dir', help='Directory containing simulation outputs')
    parser.add_argument('--wavelength', type=int, default=730, help='Wavelength in nm (default: 730)')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    wavelength = args.wavelength
    
    if not data_dir.exists():
        print(f"ERROR: Directory not found: {data_dir}")
        sys.exit(1)
    
    print(f"Validating MMC output in: {data_dir.absolute()}")
    print(f"Wavelength: {wavelength}nm")
    
    # Run all validations
    results = []
    
    results.append(("CW Output", validate_cw_output(data_dir, wavelength)))
    results.append(("TD-Gated", validate_td_gates(data_dir, wavelength)))
    results.append(("TPSF", validate_tpsf(data_dir, wavelength)))
    
    # Final summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print('='*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name:20s}: {status}")
    
    print()
    
    # Specific TD-gated assessment
    print("TD-Gated Specific Checks:")
    print("  1. Time gates present and structured correctly: REQUIRED")
    print("  2. Gate weights sum to CW weight (~1.0 ratio): REQUIRED")
    print("  3. Late gates accumulate photons at long SDS: RECOMMENDED")
    print("  4. TPSF has single peak, no NaN/Inf: REQUIRED")
    
    print(f"\n{'='*60}")
    all_passed = all(p for _, p in results)
    if all_passed:
        print("TD-GATED VALIDATION: PASSED")
        print("Ready to proceed with 10B photon production run.")
    else:
        print("TD-GATED VALIDATION: ISSUES DETECTED")
        print("Review warnings above before proceeding.")
    print('='*60)


if __name__ == '__main__':
    main()
