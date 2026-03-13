#pragma once
#include "mmc_types.cuh"
#include <curand_kernel.h>

// ---------------------------------------------------------------------------
// MMC kernel declarations
// ---------------------------------------------------------------------------

// Main photon transport kernel — one logical photon per thread, strided loop
__global__ void mmc_photon_kernel(
    MMCMeshDevice       mesh,
    MMCConfig           config,
    const Detector*     detectors,
    int                 num_detectors,
    double*             det_weight,         // [num_det]
    double*             det_pathlength,     // [num_det * NUM_TISSUE_TYPES]
    unsigned long long* det_count,          // [num_det]
    float*              tpsf,               // [num_det * TPSF_BINS]
    double*             gated_weight,       // [num_det * NUM_TIME_GATES]
    double*             gated_pathlength,   // [num_det * NUM_TIME_GATES * NUM_TISSUE_TYPES]
    curandState*        rng_states);

// RNG initializer
__global__ void mmc_init_rng(curandState* states, unsigned long long seed, int n);

// Device utility: find element containing point (uses spatial grid)
// Returns -1 if not found.
__device__ int mmc_find_element(float3 pos, const MMCMeshDevice& mesh);
