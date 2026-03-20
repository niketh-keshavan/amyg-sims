"""
Generate contamination ratio vs time gate figure for paper.
Shows scalp/amygdala partial pathlength ratio vs time gate for best 4 detectors.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
data_dir = Path("results_10B_final")
with open(data_dir / "results_730nm.json") as f:
    data_730 = json.load(f)
with open(data_dir / "results_850nm.json") as f:
    data_850 = json.load(f)

# Target detectors: SDS = 8, 19, 29, 36 mm
target_sds = [8, 19, 29, 36]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax_idx, (data, wl, title) in enumerate([(data_730, 730, '730 nm'), (data_850, 850, '850 nm')]):
    ax = axes[ax_idx]
    
    for det_info in data['detectors']:
        sds = det_info['sds_mm']
        # Find closest to target SDS values
        if not any(abs(sds - t) < 2 for t in target_sds):
            continue
        
        # Get target index for color/marker
        target_idx = np.argmin([abs(sds - t) for t in target_sds])
        target_sds_val = target_sds[target_idx]
        
        # Extract time gate data
        gates = det_info['time_gates']
        gate_centers = []
        contamination_ratios = []
        
        for gate in gates:
            gate_center = (gate['range_ps'][0] + gate['range_ps'][1]) / 1000  # Convert to ns
            pl = gate['partial_pathlength_mm']
            scalp_pl = pl.get('scalp', 0)
            amyg_pl = pl.get('amygdala', 0)
            
            # Contamination ratio = scalp PL / amygdala PL
            if amyg_pl > 1e-6:  # Only if amygdala PL is non-negligible
                ratio = scalp_pl / amyg_pl
                gate_centers.append(gate_center)
                contamination_ratios.append(ratio)
        
        if gate_centers:
            ax.semilogy(gate_centers, contamination_ratios, 
                       marker=markers[target_idx], color=colors[target_idx],
                       label=f'SDS = {target_sds_val} mm', markersize=6,
                       linewidth=1.5, alpha=0.8)
    
    ax.axvline(x=5, color='k', linestyle='--', alpha=0.5, label='5 ns gate')
    ax.set_xlabel('Time gate center (ns)', fontsize=11)
    ax.set_ylabel('Scalp / Amygdala PL ratio', fontsize=11)
    ax.set_title(f'{title}', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(0, 25)

plt.tight_layout()
output_path = Path("figures/contamination_ratio_vs_gate.png")
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
print(f"Saved {output_path}")
print(f"Saved {output_path.with_suffix('.pdf')}")

# Also print summary statistics
print("\nContamination ratios at 5ns+ gate:")
for data, wl in [(data_730, 730), (data_850, 850)]:
    print(f"\n{wl} nm:")
    for det_info in data['detectors']:
        sds = det_info['sds_mm']
        if not any(abs(sds - t) < 2 for t in target_sds):
            continue
        target_idx = np.argmin([abs(sds - t) for t in target_sds])
        target_sds_val = target_sds[target_idx]
        
        # Find gate containing 5ns+
        for gate in det_info['time_gates']:
            if gate['range_ps'][0] >= 5000:  # 5ns+ gate
                pl = gate['partial_pathlength_mm']
                scalp_pl = pl.get('scalp', 0)
                amyg_pl = pl.get('amygdala', 0)
                if amyg_pl > 1e-6:
                    ratio = scalp_pl / amyg_pl
                    print(f"  SDS={target_sds_val}mm: scalp={scalp_pl:.1f}mm, amyg={amyg_pl:.3f}mm, ratio={ratio:.0f}x")
                break
