# MMC Source/Detector Fix — Build, Test & Validate

## Role: Actuator

You are completing the fix for the MMC source/detector placement bug. Do not change architecture without updating this plan.

## Background

**Bug**: Source was placed 26mm inside the head (at skull/brain boundary) due to an ellipsoidal head model that didn't match the actual MNI152 mesh surface. All detectors landed at SDS 27-46mm instead of the targeted 8-40mm. This invalidated the entire 10B photon run.

**Fix applied** (commit `141d04f`):
- Source: now projects to actual mesh scalp surface (nearest external boundary face)
- Detectors: iterative SDS correction ensures targets within 1mm

**Mesh issues still present** (not blocking, monitor):
- Scalp volume 5350 cm³ (69% of mesh) — ~7x too large
- Amygdala center at z=-10.7 vs expected z=-20

---

## Step 1: Build on GPU instance [COMPLETED]

```bash
cd ~/amyg-sims
git pull origin master
mkdir -p build && cd build
cmake .. -DBUILD_MMC=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

**Verify**: no compile errors, binary at `build/mmc/mmc_fnirs`.

*Please have another agent handle this.*

---

## Step 2: Verify source/detector positions [COMPLETED]

Run a minimal test (1M photons) just to check placement output:

```bash
cd ~/amyg-sims
./build/mmc/mmc_fnirs --mesh mni152_head.mmcmesh --photons 1000000 --wavelengths 730 --output data_mmc_test_1M
```

**Results** (commit `001a73c`):
```
--- Source placement ---
  Right amygdala centroid: (24.3, -9.3, -10.8) mm [308 vertices]
  Found 55307 external scalp boundary faces
  Scalp surface point: (97.8, -38.3, -42.7) mm
  Surface normal: (0.970, 0.175, -0.168)
  Source position: (97.3, -38.4, -42.6) mm        ← x=97.3mm (was 68.9mm) ✓
  Source direction: (-0.970, -0.175, 0.168)
  Distance to amygdala: 85.2 mm                   ← was 52.2mm

--- Detector placement ---
    Det  0: target=    8mm actual=  8.5mm angle=  +0  ← within ±2mm ✓
    Det  1: target=    8mm actual=  7.4mm angle=+180
    Det  2: target=   15mm actual= 15.7mm angle=  +0
    ...
    Det 22: target=   35mm actual= 34.5mm angle= -60
  Built 23 detectors on scalp surface
```

**✓ Source placement FIXED**: x=97.3mm (actual scalp surface at ~98mm)
**✓ Detector placement FIXED**: all 23 detectors within ±2mm of target SDS

---

## Step 2b: CUDA Environment Issue [BLOCKING]

**Problem**: Kernel launch fails with "unknown error" on vast.ai RTX 4090 instance.

**Symptoms**:
- `cudaGetDeviceProperties()` returns garbage: SMs=0, memory=133GB (actual: 128 SMs, 24GB)
- Simple CUDA programs work, but our MMC kernel fails
- Error occurs at first kernel launch (`mmc_kernel<<<...>>>`)

**Environment**:
- Driver: 590.48.01
- CUDA Runtime: 13.1.80  
- GPU: RTX 4090 (compute capability 8.9)

**Workarounds applied**:
1. Added fallback to hardcoded RTX 4090 properties when `cudaGetDeviceProperties` fails
   - File: `mmc/src/mmc_kernel.cu:735-745`
2. Fixed `mmc/CMakeLists.txt` to use parent `CMAKE_CUDA_ARCHITECTURES` (was hardcoded to 75;80;86)
3. Added `CUDA::cudart` to link libraries

**Root cause hypothesis**: CUDA 13.1 + driver 590.48.01 compatibility issue, or container/VM GPU passthrough problem specific to this vast.ai instance.

**Next steps**:
- Try running on different GPU instance (RunPod, Lambda Labs, or different vast.ai template)
- Or downgrade to CUDA 12.x based build

*Blocked until CUDA environment resolved.*

---

## Step 3: Smoke test — 100M photons [MEDIUM — delegate]

```bash
./build/mmc/mmc_fnirs \
  --mesh mni152_head.mmcmesh \
  --photons 100000000 \
  --wavelengths 730,850 \
  --output data_mmc_fixed_100M
```

**Expected runtime**: ~3 min per wavelength at 0.7 Mph/s.

**Verify with**:
```bash
python3 python/diagnose_mesh.py --mesh mni152_head.mmcmesh --results data_mmc_fixed_100M
```

### Key checks:
1. Detector SDS table: all targets within ±2mm
2. Short-SDS (8mm) detectors detect photons
3. Late-gate tissue PL breakdown: white matter should be significant (not 0.7%)
4. Amygdala PL at short SDS vs long SDS — are short-SDS late gates higher?

Also run quick amygdala check:
```bash
python3 -c "
import json
with open('data_mmc_fixed_100M/results_730nm.json') as f:
    r = json.load(f)
for d in r['detectors']:
    gates = d.get('time_gates', [])
    late_amyg = max((g.get('partial_pathlength_mm',{}).get('amygdala',0) for g in gates[6:]), default=0)
    cw_amyg = d.get('partial_pathlength_mm',{}).get('amygdala',0)
    print(f'Det {d[\"id\"]:2d}  SDS={d[\"sds_mm\"]:5.1f}  CW_amyg={cw_amyg:.6f}  late_amyg={late_amyg:.6f}')
"
```

*Please have another agent handle this.*

---

## Step 4: Assess results and decide on 10B re-run [HARD — Architect]

**Decision tree based on 100M smoke test:**

### If short-SDS late-gate amygdala PL ≈ 0.01-0.1 mm:
- Fix worked, anatomy is partially responsible for lower values vs voxel
- **Action**: Re-run 10B with fixed source/detectors
- Run full analysis: `python3 python/analyze.py --data-dir data_mmc_fixed_10B`
- Expected cost: ~$2-3 on vast.ai 4090

### If short-SDS late-gate amygdala PL ≈ 0.1-1.0 mm (matching voxel):
- Fix fully resolved the discrepancy
- **Action**: Re-run 10B, proceed directly to publication analysis

### If short-SDS late-gate amygdala PL still < 0.001 mm:
- The mesh itself has issues (scalp volume, amygdala position)
- **Action**: Investigate mesh tissue labeling before re-running
- May need to regenerate mesh with corrected tissue segmentation
- Check `python/generate_mni152_mesh.py` — the `build_tissue_labels` function
- Known issues: scalp=5350cm³ (should be ~700cm³), amygdala z-offset

---

## Step 5: If mesh needs fixing [HARD — Architect decision needed]

**Scalp volume issue**: 69% of mesh is labeled scalp. The `build_tissue_labels` function in `generate_mni152_mesh.py:118-199` likely assigns everything outside the skull as scalp, including the large volume between the brain surface and the mesh outer boundary.

**Potential fix**: Limit scalp to a thin shell (5-8mm) around the skull exterior. Everything outside would be air.

**Amygdala z-offset**: Center at z=-10.7 vs expected z=-20. Either the MNI coordinate transform is wrong or the sphere placement in `add_amygdala_to_mesh` doesn't properly account for the voxel-to-MNI affine.

**These require re-meshing** which takes ~30 min and needs Python + TetGen.

---

## File Reference

| File | Purpose |
|------|---------|
| `mmc/src/mmc_main.cu:139-277` | Fixed source placement (uses actual mesh surface) |
| `mmc/src/mmc_main.cu:353-420` | Fixed detector placement (iterative SDS correction) |
| `python/diagnose_mesh.py` | Mesh diagnostic tool (tissue volumes, SDS check) |
| `python/generate_mni152_mesh.py:118-199` | Tissue labeling (potential scalp volume fix) |
| `python/generate_mni152_mesh.py:418-482` | Amygdala sphere placement |
| `python/analyze.py` | Full analysis pipeline (run after 10B re-run) |
| `mni152_head.mmcmesh` | Input mesh (in mesh_archive/ locally) |

---

## Change Log

| Date | Change |
|------|--------|
| 2026-03-18 | Diagnosed source 26mm inside head, all detectors at wrong SDS |
| 2026-03-18 | Fixed source placement: project to actual mesh scalp surface |
| 2026-03-18 | Fixed detector placement: iterative SDS correction (1mm tolerance) |
| 2026-03-18 | Created diagnose_mesh.py — found scalp=5350cm³, amygdala z-offset |
| 2026-03-18 | **Step 2 complete**: Source at x=97.3mm (correct), detectors within ±2mm |
| 2026-03-18 | **BLOCKING ISSUE**: CUDA 13.1 + driver 590.48.01 compatibility on vast.ai |
| 2026-03-18 | Added workaround for `cudaGetDeviceProperties` returning garbage values |
| 2026-03-18 | Fixed `mmc/CMakeLists.txt`: use parent CUDA_ARCHITECTURES, add cudart link |
