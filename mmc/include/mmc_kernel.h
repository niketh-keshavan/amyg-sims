/**
 * @file mmc_kernel.h
 * @brief CUDA kernel declarations for mesh-based Monte Carlo simulation.
 */

#pragma once

#include "types.h"
#include "mesh_loader.h"

namespace mmc {

// Maximum threads per block for various kernels
constexpr int MMC_THREADS_PER_BLOCK = 256;

/**
 * @brief Ray-tetrahedron intersection result.
 */
struct RayTetIntersect {
    float t;             // Distance to intersection
    float u, v, w;       // Barycentric coords
    int exit_face;       // Which face was hit (0-3)
    bool hit;            // True if intersection found
};

/**
 * @brief Propagate photon through mesh until detection, absorption, or escape.
 * 
 * Core Monte Carlo photon transport using ray-tetrahedron traversal.
 * 
 * @param photon Photon state (modified in-place)
 * @param mesh GPU mesh
 * @param props Optical properties array (per tissue)
 * @param rng_state CURAND state
 * @param detector Detector configuration
 */
__device__ void propagate_photon_mesh(
    Photon& photon,
    const GPUMesh& mesh,
    const OpticalProps* props,
    curandState& rng_state,
    const Detector& detector
);

/**
 * @brief Ray-tetrahedron intersection.
 * 
 * Möller-Trumbore algorithm adapted for tetrahedra.
 * 
 * @param orig Ray origin
 * @param dir Ray direction (normalized)
 * @param tet Tetrahedron vertices
 * @return Intersection result
 */
__device__ RayTetIntersect intersect_ray_tet(
    const float3& orig,
    const float3& dir,
    const float3* tet
);

/**
 * @brief Russian roulette for photon termination.
 * 
 * @param photon Photon to possibly terminate
 * @param rng_state CURAND state
 * @return true if photon survived
 */
__device__ bool russian_roulette(Photon& photon, curandState& rng_state);

/**
 * @brief Apply Henyey-Greenstein scattering.
 * 
 * @param dir Current direction (modified in-place)
 * @param g Anisotropy factor
 * @param rng_state CURAND state
 */
__device__ void hg_scatter(float3& dir, float g, curandState& rng_state);

/**
 * @brief Check if photon hits detector aperture.
 * 
 * @param photon Photon state
 * @param detector Detector configuration
 * @return true if detected
 */
__device__ bool check_detection(const Photon& photon, const Detector& detector);

/**
 * @brief Launch photons from a source position.
 * 
 * @param mesh GPU mesh
 * @param props Optical properties
 * @param detector Detector array
 * @param num_detectors Number of detectors
 * @param records Output path records
 * @param record_count Atomic counter for records
 * @param tpsf TPSF histogram (num_detectors × TPSF_BINS)
 * @param stats Output statistics
 * @param num_photons Total photons to launch
 * @param source_pos Source position
 * @param source_radius Source aperture
 * @param wavelength_nm Current wavelength
 */
__global__ void mmc_kernel(
    GPUMesh mesh,
    OpticalProps* props,
    Detector* detectors,
    int num_detectors,
    PathRecord* records,
    uint64_t* record_count,
    float* tpsf,           // [detector_idx][bin]
    SimulationStats* stats,
    uint64_t num_photons,
    float3 source_pos,
    float source_radius,
    int wavelength_nm
);

/**
 * @brief Initialize CURAND states.
 */
__global__ void init_rng_kernel(
    curandState* states,
    uint64_t seed,
    int num_states
);

} // namespace mmc
