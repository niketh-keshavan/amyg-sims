#!/usr/bin/env python3
import json
import sys

data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data_mmc_10B'

with open(f'{data_dir}/results_730nm.json') as f:
    data = json.load(f)

print('Amygdala PL values (mm) - 10B photons:')
print('='*60)

all_amyg = []
for d in data['detectors']:
    gates = d['time_gates']
    max_amyg = max(g['partial_pathlength_mm'].get('amygdala', 0) for g in gates)
    all_amyg.append(max_amyg)
    print(f"Det {d['id']:2d} SDS={d['sds_mm']:5.1f}mm: max amyg PL = {max_amyg:.6f} mm")

print('='*60)
print(f"Overall max amygdala PL: {max(all_amyg):.6f} mm")
print(f"Gates with PL > 0.001 mm: {sum(1 for a in all_amyg if a > 0.001)}")
print(f"Gates with PL > 0.01 mm: {sum(1 for a in all_amyg if a > 0.01)}")
