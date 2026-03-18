#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Tissue types
// ---------------------------------------------------------------------------
enum TissueType : uint8_t {
    TISSUE_AIR       = 0,
    TISSUE_SCALP     = 1,
    TISSUE_SKULL     = 2,
    TISSUE_CSF       = 3,
    TISSUE_GRAY      = 4,
    TISSUE_WHITE     = 5,
    TISSUE_AMYGDALA  = 6,
    NUM_TISSUE_TYPES = 7
};

// ---------------------------------------------------------------------------
// Time-domain parameters
// ---------------------------------------------------------------------------
#define TPSF_BINS       512       // bins in TPSF histogram
#define TPSF_BIN_PS     10.0f     // ps per bin -> 5.12 ns total
#define NUM_TIME_GATES  10        // number of temporal gates (fine resolution in late gates)
#define C_VACUUM_MM_PS  0.2998f   // speed of light in vacuum [mm/ps]

// ---------------------------------------------------------------------------
// Photon path recording
// ---------------------------------------------------------------------------
#define MAX_PATH_STEPS  2048      // max steps recorded per photon path
#define PATHS_PER_DET   64        // paths to record per detector
#define MAX_RECORDED_PATHS (PATHS_PER_DET * 128)  // global cap

// ---------------------------------------------------------------------------
// Optical properties for a single tissue at one wavelength
// ---------------------------------------------------------------------------
struct OpticalProps {
    float mu_a;   // absorption coefficient  [1/mm]
    float mu_s;   // scattering coefficient  [1/mm]
    float g;      // anisotropy factor
    float n;      // refractive index
};

// ---------------------------------------------------------------------------
// Simulation configuration
// ---------------------------------------------------------------------------
struct SimConfig {
    // Grid
    int   nx, ny, nz;          // voxels per axis
    float dx;                  // voxel size [mm] (isotropic)

    // Photon budget
    uint64_t num_photons;

    // Source position [mm] (on scalp surface)
    float src_x, src_y, src_z;

    // Source direction (unit vector, typically pointing inward)
    float src_dx, src_dy, src_dz;

    // Beam radius [mm] (half of beam diameter; 0 = pencil beam)
    float beam_radius;

    // Wavelength index (0 = 760 nm, 1 = 850 nm)
    int wavelength_idx;

    // Optical properties per tissue type (set for current wavelength)
    OpticalProps tissue[NUM_TISSUE_TYPES];

    // Weight threshold for Russian roulette
    float weight_threshold;
    int   roulette_m;          // 1/m survival chance
};

// ---------------------------------------------------------------------------
// Per-photon state
// ---------------------------------------------------------------------------
struct __align__(16) PhotonState {
    float x, y, z;            // position [mm]
    float dx, dy, dz;         // direction cosines
    float weight;              // current photon weight
    int   alive;               // 0 = terminated
};

// ---------------------------------------------------------------------------
// Detector ring on scalp surface
// ---------------------------------------------------------------------------
struct Detector {
    float x, y, z;            // center position [mm]
    float radius;             // acceptance radius [mm]
    int   id;
    float nx, ny, nz;         // surface normal (outward from head) [unit vector]
    float n_critical;         // critical angle cosine (n_air / n_tissue)
};

// ---------------------------------------------------------------------------
// Per-detector accumulation (one per detector per wavelength)
// ---------------------------------------------------------------------------
struct DetectorResult {
    double total_weight;           // sum of exiting photon weights
    double total_pathlength;       // weighted total pathlength [mm]
    double partial_pathlength[NUM_TISSUE_TYPES]; // per-tissue pathlength
    uint64_t num_detected;         // photon count
};
