/**
 * @file types.h
 * @brief Core type definitions for MNI152-based Mesh Monte Carlo (MMC) simulation.
 * 
 * This header defines the fundamental data structures used throughout the MMC simulation,
 * including tissue types, photon states, detector configurations, and mesh elements.
 */

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

// Constants
#define MAX_DETECTOR_RECORDS 10000000  // Max photon path records per detector
#define TPSF_BINS 512                  // Time bins for TPSF histogram
#define TPSF_DT_PS 10.0f               // Time bin width in picoseconds
#define MAX_TET_NEIGHBORS 4            // Tetrahedrons have 4 faces
#define EPS 1e-6f                      // Small value for numerical precision
#define photon_meminium  10000         // Minimum photons for progress reporting

/**
 * @brief Tissue type enumeration matching MNI152 segmentation.
 * 
 * These values must match the labels assigned by generate_mni152_mesh.py
 */
enum TissueType : int32_t {
    TISSUE_AIR      = 0,   // External/background
    TISSUE_SCALP    = 1,   // Skin/scalp layer
    TISSUE_SKULL    = 2,   // Skull bone
    TISSUE_CSF      = 3,   // Cerebrospinal fluid
    TISSUE_GRAY     = 4,   // Gray matter
    TISSUE_WHITE    = 5,   // White matter
    TISSUE_AMYGDALA = 6,   // Amygdala ROI
    NUM_TISSUES     = 7
};

/**
 * @brief Photon propagation state.
 * 
 * Tracks position, direction, weight, and current tetrahedron during Monte Carlo
 * photon transport through the mesh.
 */
struct Photon {
    float3 pos;          // Current position (mm)
    float3 dir;          // Direction vector (normalized)
    float weight;        // Current photon weight (for Russian roulette)
    float pathlen;       // Total path length traveled (mm)
    float time;          // Time of flight (ps)
    int32_t tet;         // Current tetrahedron index (-1 = outside)
    int32_t tissue;      // Current tissue type
    int32_t status;      // 0=alive, 1=detected, 2=escaped, 3=absorbed
    int32_t bounces;     // Number of scattering events
    
    // Amygdala-specific tracking
    float amyg_pathlen;  // Path length through amygdala
    int32_t hit_amyg;    // Flag if photon passed through amygdala
};

/**
 * @brief Optical properties for a tissue type at a specific wavelength.
 */
struct OpticalProps {
    float mu_a;          // Absorption coefficient (1/mm)
    float mu_s;          // Scattering coefficient (1/mm)
    float g;             // Anisotropy factor
    float n;             // Refractive index
    float mu_s_prime;    // Reduced scattering = mu_s * (1 - g)
    
    __host__ __device__ float get_albedo() const {
        float mu_t = mu_a + mu_s;
        return (mu_t > EPS) ? mu_s / mu_t : 0.0f;
    }
    
    __host__ __device__ float get_mean_free_path() const {
        float mu_t = mu_a + mu_s;
        return (mu_t > EPS) ? 1.0f / mu_t : 1e6f;
    }
};

/**
 * @brief Tetrahedral mesh element.
 * 
 * Stores node indices, tissue label, and neighbor information for each tetrahedron.
 */
struct Tetrahedron {
    int32_t v[4];        // Node indices (0-based)
    int32_t tissue;      // Tissue type label
    int32_t neighbor[4]; // Adjacent tet across face i (-1 = boundary)
};

/**
 * @brief Detector configuration for temporal measurements.
 * 
 * Defines a source-detector pair with optional time-gating capability.
 */
struct Detector {
    float3 pos;          // Detector position (mm)
    float radius;        // Detection aperture radius (mm)
    float3 source_pos;   // Associated source position (mm)
    float source_radius; // Source aperture radius (mm)
    float sds;           // Source-detector separation (mm)
    
    // Time gating
    float gate_start;    // Gate start time (ps)
    float gate_end;      // Gate end time (ps)
    int use_gate;        // Enable time-gating
};

/**
 * @brief Photon path record for detected photons.
 * 
 * Stores trajectory information for sensitivity analysis and pathlength calculations.
 */
struct PathRecord {
    float3 exit_pos;     // Exit position at detector
    float3 exit_dir;     // Exit direction
    float total_pathlen; // Total path length
    float amyg_pathlen;  // Path length through amygdala
    float time_of_flight;// Time of flight (ps)
    float weight;        // Weight at detection
    int32_t bounces;     // Number of scattering events
    int32_t hit_amyg;    // Passed through amygdala
};

/**
 * @brief Simulation statistics output.
 */
struct SimulationStats {
    uint64_t launched;       // Total photons launched
    uint64_t detected;       // Photons detected
    uint64_t escaped;        // Photons escaped to air
    uint64_t absorbed;       // Photons absorbed (weight too low)
    uint64_t max_bounces;    // Photons stopped at bounce limit
    float detection_prob;    // Detected / launched
    float mean_pathlen;      // Mean path length of detected photons
    float mean_time;         // Mean time of flight
};

/**
 * @brief Mesh file header (binary format).
 * 
 * Format:
 *   Header (48 bytes)
 *   Nodes   (N × 3 × float32)
 *   Elems   (M × 4 × int32)
 *   Tissue  (M × int32)
 *   Neighbors (M × 4 × int32)
 */
struct MeshHeader {
    uint32_t magic;      // 0x42434D4D "MMCM"
    uint32_t version;    // File format version
    uint32_t num_nodes;  // Number of vertices
    uint32_t num_elems;  // Number of tetrahedra
    float bbox_min[3];   // Bounding box min (mm)
    float bbox_max[3];   // Bounding box max (mm)
    uint32_t reserved[2];
};

// Validation
static_assert(sizeof(MeshHeader) == 48, "MeshHeader must be 48 bytes");
static_assert(sizeof(Tetrahedron) == 32, "Tetrahedron should be 32 bytes for cache alignment");

// Helper functions
__host__ __device__ inline float3 make_float3(float x, float y, float z) {
    float3 v; v.x = x; v.y = y; v.z = z; return v;
}

__host__ __device__ inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

__host__ __device__ inline float3 operator/(const float3& a, float s) {
    return make_float3(a.x / s, a.y / s, a.z / s);
}

__host__ __device__ inline float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float length(const float3& v) {
    return sqrtf(dot(v, v));
}

__host__ __device__ inline float3 normalize(const float3& v) {
    float len = length(v);
    return (len > EPS) ? v / len : make_float3(0, 0, 1);
}

__host__ __device__ inline float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
