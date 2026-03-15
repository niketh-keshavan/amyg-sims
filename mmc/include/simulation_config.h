/**
 * @file simulation_config.h
 * @brief Configuration management for MMC simulations.
 */

#pragma once

#include <string>
#include <cstdint>

namespace mmc {

/**
 * @brief Simulation run configuration.
 */
struct SimulationConfig {
    // Input/Output
    std::string mesh_path = "data/mni152_head.mmcmesh";
    std::string output_dir = "results_mmc";
    
    // Photon counts
    uint64_t num_photons_760nm = 100000000;   // 100M photons at 760nm
    uint64_t num_photons_850nm = 100000000;   // 100M photons at 850nm
    
    // Physics parameters
    int max_bounces = 100000;      // Maximum scattering events per photon
    float min_weight = 1e-4f;      // Russian roulette threshold
    float roulette_chance = 0.1f;  // Survival probability in Russian roulette
    
    // Detection
    float tpsf_max_time_ps = 5000.0f;  // Maximum TPSF time (5 ns)
    int record_photon_paths = 1;       // Enable path recording for sensitivity
    uint64_t max_path_records = 10000000;  // Max records per detector
    
    // Time gating (for time-domain simulations)
    int enable_time_gating = 0;
    float gate_start_ps = 0.0f;
    float gate_end_ps = 5000.0f;
    
    // Source
    float source_radius_mm = 2.0f;
    
    // GPU
    int gpu_device = 0;
    int threads_per_block = 256;
    
    // Progress
    int print_interval = 10;  // Seconds between progress updates
    
    /**
     * @brief Load configuration from JSON file.
     */
    bool load_from_file(const std::string& path);
    
    /**
     * @brief Save configuration to JSON file.
     */
    bool save_to_file(const std::string& path) const;
    
    /**
     * @brief Parse command line arguments.
     */
    bool parse_args(int argc, char** argv);
    
    /**
     * @brief Print configuration.
     */
    void print() const;
    
    /**
     * @brief Validate configuration.
     */
    bool validate() const;
};

/**
 * @brief Print usage information.
 */
void print_usage(const char* program_name);

} // namespace mmc
