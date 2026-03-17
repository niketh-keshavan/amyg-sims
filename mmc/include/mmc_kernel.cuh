#pragma once
#include "types.cuh"
#include "mmc_mesh.cuh"

// MMC simulation config (extends base SimConfig with mesh-specific fields)
struct MMCConfig {
    uint64_t num_photons;

    // Source on scalp surface
    float src_x, src_y, src_z;
    float src_dx, src_dy, src_dz;
    float beam_radius;

    int wavelength_idx;

    // Optical properties per tissue
    OpticalProps tissue[NUM_TISSUE_TYPES];

    // Russian roulette parameters
    float weight_threshold;
    int   roulette_m;
};

// Launch the MMC simulation for one wavelength.
// Same output signature as voxel MC for compatibility with analysis pipeline.
void launch_mmc_simulation(
    const MMCDeviceData& dev,
    const MMCConfig& config,
    const Detector* h_dets,
    int n_dets,
    DetectorResult* h_results,
    double* h_tpsf,              // [n_dets * TPSF_BINS]
    double* h_gated_weight,      // [n_dets * NUM_TIME_GATES]
    double* h_gated_partial_pl,  // [n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES]
    uint64_t* h_gated_count,     // [n_dets * NUM_TIME_GATES]
    float* h_path_pos,           // [MAX_RECORDED_PATHS * MAX_PATH_STEPS * 3]
    int*   h_path_det,           // [MAX_RECORDED_PATHS]
    int*   h_path_len,           // [MAX_RECORDED_PATHS]
    int*   h_num_paths           // [1]
);
