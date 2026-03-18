# MMC Simulation Validation & 10-100B Photon Production Run

## Role: Actuator

You are implementing and validating the MMC (Mesh-based Monte Carlo) fNIRS simulation. Do not change architecture without updating this plan.

## Current State

- MNI152 mesh generated: 74,845 nodes, 347,255 tets, **146 amygdala tets (0.04%)**
- Mesh file: `mni152_head.mmcmesh` (13.4 MB)
- TD-gated analysis fully implemented in kernel (identical to voxel MC)
- Binary: `build/mmc/mmc_fnirs`

## Known Limitation: Small Amygdala Volume

**Issue**: The mesh contains only 146 amygdala tets (0.04% of total volume), which is anatomically small but statistically challenging.

**Decision**: Proceed with **Option 2** - use extremely high photon counts (10-100 billion) to achieve statistically meaningful amygdala pathlength measurements.

**Rationale**:
- 100M photons → trace amygdala PL (~0.0001-0.0004 mm)
- 10B photons → ~100x more stats → ~0.01-0.04 mm amygdala PL
- 100B photons → ~1000x more stats → ~0.1-0.4 mm amygdala PL
- Expected CW amygdala sensitivity: 0.001-0.1% (tiny target, deep brain)
- Late-gate sensitivity improvement: 5-50x over CW

## Step 1: Build [COMPLETED]

```bash
cd ~/amyg-sims
mkdir -p build && cd build
cmake .. -DBUILD_MMC=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Verify: binary at `build/mmc/mmc_fnirs`, no build errors.

## Step 2: Smoke Test — 100M Photons, Dual Wavelength [COMPLETED]

```bash
cd ~/amyg-sims
./build/mmc/mmc_fnirs --mesh mni152_head.mmcmesh --photons 100000000 --wavelengths 730,850 --output data_mmc_100M
```

**Results** (100M photons, ~0.37 M photons/sec):
- Throughput: ~0.37 M photons/sec
- Detectors 10, 19, 20 show trace amygdala PL: 0.0001-0.0004 mm
- All detectors at SDS 27-46 mm detecting photons
- Mean pathlength increases with SDS (245-415 mm) ✓

**Conclusion**: Physics working correctly; amygdala detection requires higher photon counts due to small target volume.

## Step 3: Validate CW Detector Output [PENDING - Requires 10B+ Run]

For 10B photon run, check `data_mmc_10B/results_730nm.json`:

- Short SDS detectors (8-15 mm) have highest photon counts
- Long SDS detectors (33-40 mm) have lower counts but still > 0
- `mean_pathlength_mm` increases with SDS (expected: 20-200 mm)
- **Adjusted**: `partial_pathlength_mm.amygdala` > 0.001 mm for detectors at SDS 35-46 mm
- All tissue partial pathlengths are non-negative
- `scalp` pathlength dominates at short SDS; `gray_matter`/`white_matter` grow at longer SDS

## Step 4: Validate TD-Gated Output [PENDING - Requires 10B+ Run]

From same JSON, check `time_gates` array per detector:

- 10 gates present (indices 0-9), edges: [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000] ps
- Gate 0 (0-500 ps) has most photons for short SDS
- Later gates (3-9) have more photons proportionally at longer SDS
- Sum of `gate.weight` across all gates ≈ `total_weight` (ratio ~1.0)
- **Adjusted**: `gate.partial_pathlength_mm.amygdala` > 0.005 mm in late gates (>= gate 6, 3000+ ps) for SDS 35-46 mm
- No gate has negative weight or pathlength values

Quick check script:
```bash
python3 -c "
import json
with open('data_mmc_10B/results_730nm.json') as f:
    r = json.load(f)
for d in r['detectors']:
    gates = d['time_gates']
    gw_sum = sum(g['weight'] for g in gates)
    amyg_cw = d['partial_pathlength_mm'].get('amygdala', 0)
    late_amyg = max(g['partial_pathlength_mm'].get('amygdala', 0) for g in gates[6:])
    if d['total_weight'] > 0:
        print(f\"Det {d['id']:2d}  SDS={d['sds_mm']:5.1f}  photons={d['detected_photons']:8d}  CW_amyg={amyg_cw:.6f}  late_amyg={late_amyg:.6f}  gw_sum/cw={gw_sum/d['total_weight']:.4f}\")
    else:
        print(f\"Det {d['id']:2d}  SDS={d['sds_mm']:5.1f}  NO DETECTIONS\")
"
```

## Step 5: Validate TPSF Binary Output [PENDING]

Check `data_mmc_10B/tpsf_730nm.bin`:

- File size = n_dets x 512 x 8 bytes
- No NaN or Inf values
- TPSF integral per detector ≈ total detected weight

```bash
python3 -c "
import numpy as np, os
n_dets = 23
data = np.fromfile('data_mmc_10B/tpsf_730nm.bin', dtype=np.float64)
expected = n_dets * 512
print(f'TPSF size: {len(data)} doubles (expected {expected})')
tpsf = data.reshape(n_dets, 512)
for d in range(min(5, n_dets)):
    peak_bin = np.argmax(tpsf[d])
    print(f'  Det {d}: peak at {peak_bin*10} ps, integral={tpsf[d].sum():.6e}, has_nan={np.any(np.isnan(tpsf[d]))}')
"
```

## Step 6: Run Analysis Pipeline [PENDING]

```bash
python3 python/analyze.py --data-dir data_mmc_10B
```

Verify: no crashes, MBLL sensitivity estimates produced, amygdala delta-OD non-zero for late gates at long SDS.

## Step 7: Physics Sanity Checks [PENDING]

**Adjusted for 0.04% amygdala volume**:
- Mean pathlength at SDS 30 mm ≈ 100-200 mm
- Amygdala partial pathlength << total pathlength (expected: ~0.001-0.1 mm at 10B photons)
- CW amygdala sensitivity (amyg_pl / total_pl) ~ 0.0001-0.01% at optimal SDS
- Late-gate amygdala sensitivity should be 5-50x higher than CW

## Go/No-Go Decision [PENDING]

**GO if**:
1. Detectors at SDS 35-46 mm detect photons with amygdala pathlength > 0.001 mm at 10B photons
2. Late time gates show higher amygdala sensitivity than CW (5-50x improvement)
3. TPSF has physically plausible shape (single peak, no NaN)
4. analyze.py runs without errors

**NO-GO if**:
- Zero amygdala pathlength everywhere at 10B photons -> mesh labeling issue
- Zero detections at all SDS -> source/detector placement or boundary physics bug
- TPSF is all zeros -> TOF computation bug
- Gate weights don't sum to CW weight -> accumulation bug
- analyze.py crashes -> JSON format mismatch

## Step 8: Production Run — 10-100B Photons [PENDING, requires GO from Step 7]

### 10 Billion Photon Run:
```bash
./build/mmc/mmc_fnirs \
  --mesh mni152_head.mmcmesh \
  --photons 10000000000 \
  --wavelengths 730,850 \
  --output data_mmc_10B
```

**Estimated runtime**: ~7.5 hours at 0.37 M photons/sec

### 100 Billion Photon Run (Final Production):
```bash
./build/mmc/mmc_fnirs \
  --mesh mni152_head.mmcmesh \
  --photons 100000000000 \
  --wavelengths 730,850 \
  --output data_mmc_100B
```

**Estimated runtime**: ~75 hours (~3 days) at 0.37 M photons/sec

### Post-Processing:
```bash
# For 10B run
python3 python/analyze.py --data-dir data_mmc_10B
python3 python/sensitivity_analysis.py data_mmc_10B/

# For 100B run
python3 python/analyze.py --data-dir data_mmc_100B
python3 python/sensitivity_analysis.py data_mmc_100B/
```

## High-Photon-Count Considerations

1. **Numerical Precision**: Using double-precision accumulators for weights and pathlengths to handle 10B+ photons
2. **Memory**: Output files scale with photon count for path recording (optional, can be disabled)
3. **Checkpointing**: Consider running in batches if system has time limits
4. **Verification**: Run 1B photons first as intermediate checkpoint (~45 min)

## Key Files

| File | Role |
|------|------|
| `mmc/src/mmc_kernel.cu:140-194` | TD gate assignment + accumulation |
| `mmc/src/mmc_main.cu:30-113` | JSON output (must match analyze.py field names) |
| `mmc/src/mmc_main.cu:357-607` | main() — source/detector setup, launch |
| `mmc/include/mmc_kernel.cuh` | Constants: TPSF_BINS=512, NUM_TIME_GATES=10, NUM_TISSUE_TYPES=7 |
| `python/analyze.py` | Full TD-gated analysis (works with both voxel and MMC JSON) |
| `python/sensitivity_analysis.py` | Sensitivity metrics |
| `mni152_head.mmcmesh` | Input mesh (must be uploaded to remote) |

---

## Change Log

| Date | Change |
|------|--------|
| Initial | 1B photon target, standard amygdala volume assumptions |
| 2026-03-17 | Updated to 10-100B photons after discovering 0.04% amygdala volume (146 tets) |
