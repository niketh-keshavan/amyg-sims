#pragma once
#include "types.cuh"

// Launch the Monte Carlo simulation for one wavelength.
// CW outputs:  results array (n_dets elements)
// TD outputs:  TPSF histograms, time-gated weights/pathlengths/counts
// Path outputs: recorded photon trajectories for visualization
void launch_mc_simulation(
    const uint8_t* h_volume,
    const SimConfig& config,
    const Detector* h_dets,
    int n_dets,
    DetectorResult* h_results,
    float* d_fluence,
    double* h_tpsf,              // [n_dets * TPSF_BINS]
    double* h_gated_weight,      // [n_dets * NUM_TIME_GATES]
    double* h_gated_partial_pl,  // [n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES]
    uint64_t* h_gated_count,     // [n_dets * NUM_TIME_GATES]
    float* h_path_pos,           // [MAX_RECORDED_PATHS * MAX_PATH_STEPS * 3]
    int*   h_path_det,           // [MAX_RECORDED_PATHS] detector id per path
    int*   h_path_len,           // [MAX_RECORDED_PATHS] steps per path
    int*   h_num_paths           // [1] total paths recorded
);
