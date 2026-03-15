/**
 * @file detector_array.h
 * @brief 22-channel fNIRS detector array for temporal lobe measurements.
 * 
 * Arranged on the right temporal scalp targeting the amygdala at MNI coordinates
 * approximately (+24, -2, -20) mm.
 */

#pragma once

#include "types.h"
#include <vector>

namespace mmc {

// MNI152 coordinates for right amygdala centroid
constexpr float AMYG_RIGHT_MNI[3] = {24.0f, -2.0f, -20.0f};
constexpr float AMYG_LEFT_MNI[3]  = {-24.0f, -2.0f, -20.0f};

// Default detector aperture radius (mm)
constexpr float DEFAULT_DETECTOR_RADIUS = 3.0f;
constexpr float DEFAULT_SOURCE_RADIUS = 2.0f;

/**
 * @brief Channel configuration types.
 */
enum ChannelType {
    CH_SHORT,       // Short separation (8 mm) for scalp regression
    CH_PRIMARY,     // Primary direction channels (15-45 mm SDS)
    CH_OFF_AXIS_30, // Off-axis at +/-30 degrees
    CH_OFF_AXIS_60  // Off-axis at +/-60 degrees
};

/**
 * @brief Single channel configuration.
 */
struct ChannelConfig {
    float sds;              // Source-detector separation (mm)
    float angle_deg;        // Angle from primary direction
    ChannelType type;
    const char* name;
};

/**
 * @brief 22-channel array layout.
 * 
 * Based on high-density fNIRS array for amygdala measurements.
 * Source positioned on temporal scalp, detectors arranged radially.
 */
class DetectorArray {
public:
    /**
     * @brief Create detector array for temporal lobe measurement.
     * 
     * @param source_mni Source position in MNI coordinates
     * @param primary_direction Primary measurement direction (toward amygdala)
     * @param use_right_amyg Target right amygdala (true) or left (false)
     */
    DetectorArray(const float* source_mni = nullptr,
                  const float* primary_direction = nullptr,
                  bool use_right_amyg = true);
    
    /**
     * @brief Get number of detectors.
     */
    int get_num_detectors() const { return static_cast<int>(detectors_.size()); }
    
    /**
     * @brief Get detector configurations.
     */
    const std::vector<Detector>& get_detectors() const { return detectors_; }
    
    /**
     * @brief Get detectors as GPU array.
     * 
     * @return Device pointer (caller must cudaFree)
     */
    Detector* upload_to_gpu() const;
    
    /**
     * @brief Print array configuration.
     */
    void print_config() const;
    
    /**
     * @brief Get source position.
     */
    float3 get_source_pos() const { return source_pos_; }
    
    /**
     * @brief Get target (amygdala) position.
     */
    float3 get_target_pos() const { return target_pos_; }

private:
    void build_array();
    float3 project_to_scalp(const float3& pos);
    
    std::vector<Detector> detectors_;
    float3 source_pos_;
    float3 target_pos_;
    float3 primary_dir_;
    bool use_right_amyg_;
};

/**
 * @brief Default temporal array configuration.
 * 
 * Source on right temporal scalp, detectors arranged to sample amygdala depth.
 */
extern const ChannelConfig DEFAULT_CHANNELS[];
constexpr int NUM_DEFAULT_CHANNELS = 22;

} // namespace mmc
