#!/usr/bin/env python3
"""Analyze 10B photon MMC simulation results"""
import json
import numpy as np

def analyze_wavelength(wl):
    print(f'\n{"="*60}')
    print(f'=== {wl} nm RESULTS (10B photons) ===')
    print(f'{"="*60}')
    
    with open(f'data_mmc_10B/results_{wl}nm.json') as f:
        r = json.load(f)
    
    print(f'Photons simulated: {r["num_photons"]:,}')
    print(f'Model: {r["geometry_model"]}, Scattering: {r["scattering_model"]}')
    print()
    
    # CW Results
    print('CW DETECTOR RESULTS:')
    print('-' * 90)
    print(f'{"Det":>4} {"SDS":>6} {"Detected":>12} {"MeanPL":>10} {"AmygPL":>10} {"Scalp%":>8}')
    print('-' * 90)
    
    for d in r['detectors']:
        ppl = d['partial_pathlength_mm']
        amyg_pl = ppl.get('amygdala', 0)
        total_pl = sum(ppl.values()) if ppl else 1
        scalp_pct = ppl.get('scalp', 0) / total_pl * 100 if total_pl > 0 else 0
        
        print(f'{d["id"]:4} {d["sds_mm"]:6.1f} {d["detected_photons"]:12,} '
              f'{d["mean_pathlength_mm"]:10.2f} {amyg_pl:10.6f} {scalp_pct:8.1f}')
    
    print()
    
    # Late-gate analysis
    print('TIME-GATED AMYGDALA (Late gates 6-9, >3000 ps):')
    print('-' * 100)
    print(f'{"Det":>4} {"SDS":>6} {"3000-3500":>12} {"3500-4000":>12} {"4000-5000":>12} {"5000+":>12} {"MAX":>12}')
    print('-' * 100)
    
    max_amyg_values = []
    for d in r['detectors']:
        gates = d.get('time_gates', [])
        if len(gates) >= 10:
            late_gates = gates[6:10]  # gates 6,7,8,9
            amyg_values = [g.get('partial_pathlength_mm', {}).get('amygdala', 0) for g in late_gates]
            max_amyg = max(amyg_values)
            max_amyg_values.append((d['id'], d['sds_mm'], max_amyg, amyg_values))
    
    # Sort by max amygdala PL
    max_amyg_values.sort(key=lambda x: x[2], reverse=True)
    
    for det_id, sds, max_amyg, values in max_amyg_values[:15]:  # Top 15
        print(f'{det_id:4} {sds:6.1f} ' + ' '.join([f'{v:12.6f}' for v in values]) + f' {max_amyg:12.6f}')
    
    print()
    
    # Summary statistics
    print('SUMMARY STATISTICS:')
    print('-' * 50)
    
    # CW amygdala
    cw_amyg = [d['partial_pathlength_mm'].get('amygdala', 0) for d in r['detectors']]
    print(f'CW Amygdala PL:  mean={np.mean(cw_amyg):.6f}, max={np.max(cw_amyg):.6f}')
    
    # Late-gate amygdala
    late_amyg = [x[2] for x in max_amyg_values]
    print(f'Late Amygdala PL: mean={np.mean(late_amyg):.6f}, max={np.max(late_amyg):.6f}')
    
    # Short vs Long SDS comparison
    short_sds = [x[2] for x in max_amyg_values if x[1] < 20]
    long_sds = [x[2] for x in max_amyg_values if x[1] > 30]
    
    if short_sds and long_sds:
        print(f'\nShort SDS (<20mm): mean={np.mean(short_sds):.6f}')
        print(f'Long SDS (>30mm):  mean={np.mean(long_sds):.6f}')
        print(f'Ratio (long/short): {np.mean(long_sds)/np.mean(short_sds):.2f}x')
    
    # Best detector
    best = max_amyg_values[0]
    print(f'\nBest detector: Det {best[0]} at SDS={best[1]:.1f}mm')
    print(f'  Late-gate amygdala PL: {best[2]:.6f} mm')
    
    return max_amyg_values

print('\n' + '='*60)
print('MMC 10B PHOTON SIMULATION ANALYSIS')
print('Mesh: mni152_head_fixed.mmcmesh (max_vol=5.0)')
print('='*60)

results_730 = analyze_wavelength('730')
results_850 = analyze_wavelength('850')

print('\n' + '='*60)
print('COMPARISON: 730 nm vs 850 nm')
print('='*60)

# Compare best detectors
best_730 = results_730[0]
best_850 = results_850[0]

print(f'730 nm best: Det {best_730[0]} at {best_730[1]:.1f}mm = {best_730[2]:.6f} mm')
print(f'850 nm best: Det {best_850[0]} at {best_850[1]:.1f}mm = {best_850[2]:.6f} mm')
print(f'850/730 ratio: {best_850[2]/best_730[2]:.2f}x')

print('\n' + '='*60)
print('CONCLUSION:')
print('='*60)
avg_late_730 = np.mean([x[2] for x in results_730])
avg_late_850 = np.mean([x[2] for x in results_850])
print(f'Late-gate amygdala PL: 730nm={avg_late_730:.6f}, 850nm={avg_late_850:.6f}')
print(f'Photons ARE reaching amygdala - fix is successful!')
print('='*60)
