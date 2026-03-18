# Debugging face_pair Kernel Hang

## Step 1: Validate face_pair array on host

```bash
python3 debug_face_pair.py
```

Expected output should show:
- All boundary faces have face_pair = -1
- All interior faces have face_pair = 0-3
- No asymmetry errors

## Step 2: Add kernel debug checks

Edit `mmc/src/mmc_kernel.cu` and add assertions:

```cpp
// After computing new_entry_face
int new_entry_face = face_pair[current_tet * 4 + exit_face];

// DEBUG: Check for invalid values
if (new_entry_face < 0 || new_entry_face > 3) {
    printf("DEBUG: Invalid new_entry_face=%d at tet=%d, exit_face=%d, neighbor=%d\n",
           new_entry_face, current_tet, exit_face, neighbor);
    // Fallback to old method
    new_entry_face = -1;
    for (int f = 0; f < 4; f++) {
        if (neighbors[neighbor * 4 + f] == current_tet) {
            new_entry_face = f;
            break;
        }
    }
}
```

## Step 3: Check for infinite loop conditions

The hang is likely in this loop in `mmc_kernel.cu`:

```cpp
for (int step = 0; step < 500000; step++) {
    // ...
    while (s_remain > 1e-8f) {
        // ...
        if (neighbor < 0) {
            // External boundary
        } else {
            // Cross into neighbor
            int new_entry_face = face_pair[current_tet * 4 + exit_face];
            // If new_entry_face is wrong, photon might bounce between tets forever
        }
    }
}
```

Add a step counter watchdog:

```cpp
__device__ int d_debug_hang_count = 0;

// In kernel, add check:
if (step > 490000) {
    atomicAdd(&d_debug_hang_count, 1);
    printf("DEBUG: Photon approaching step limit at tet=%d\n", current_tet);
    break;
}
```

## Step 4: Quick test with photon path recording

```bash
# Run with very few photons and path recording
./build/mmc/mmc_fnirs \
  --mesh mni152_head.mmcmesh \
  --photons 1000 \
  --wavelengths 730 \
  --output debug_test
```

If it hangs with just 1000 photons, the issue is deterministic (bad mesh/kernel).

If it works, the issue is probabilistic (race condition or rare edge case).

## Step 5: Binary search for problematic photon

Add photon ID logging:

```cpp
// In kernel, at start of photon loop:
int photon_id = blockIdx.x * blockDim.x + threadIdx.x;
if (photon_id < 10) {
    printf("DEBUG: Photon %d starting\n", photon_id);
}
```

## Most Likely Causes

1. **face_pair[-1] access**: If neighbor is -1 (boundary), we still access `face_pair[current_tet * 4 + exit_face]` which might be -1, then use it as entry_face in next iteration.

2. **face_pair not uploaded**: Check that face_pair is actually copied to GPU in `upload_mesh_to_gpu()`.

3. **Memory corruption**: The face_pair array might be corrupted during upload.

## Quick Fix

Temporarily revert to old method with timeout:

```cpp
// Old method with safety limit
int new_entry_face = -1;
int search_count = 0;
for (int f = 0; f < 4 && search_count < 10; f++, search_count++) {
    if (neighbors[neighbor * 4 + f] == current_tet) {
        new_entry_face = f;
        break;
    }
}
if (new_entry_face < 0) {
    // Photon escaped somehow, terminate
    weight = 0.0f;
    break;
}
```
