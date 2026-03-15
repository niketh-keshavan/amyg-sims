/**
 * @file detector_array.cpp
 * @brief 22-channel fNIRS detector array implementation.
 */

#include "detector_array.h"
#include <cstdio>
#include <cmath>

namespace mmc {

// Default channel configuration for temporal amygdala measurement
const ChannelConfig DEFAULT_CHANNELS[NUM_DEFAULT_CHANNELS] = {
    // Short separation channels (8 mm) - for scalp regression
    {8.0f,   0.0f,   CH_SHORT,       "SS_0"},
    {8.0f,  90.0f,   CH_SHORT,       "SS_90"},
    {8.0f,  180.0f,  CH_SHORT,       "SS_180"},
    {8.0f,  270.0f,  CH_SHORT,       "SS_270"},
    
    // Primary direction channels (toward amygdala)
    {15.0f,  0.0f,   CH_PRIMARY,     "SDS15"},
    {25.0f,  0.0f,   CH_PRIMARY,     "SDS25"},
    {35.0f,  0.0f,   CH_PRIMARY,     "SDS35"},
    {45.0f,  0.0f,   CH_PRIMARY,     "SDS45"},
    
    // Off-axis 30 degrees
    {15.0f,  30.0f,  CH_OFF_AXIS_30, "SDS15_+30"},
    {25.0f,  30.0f,  CH_OFF_AXIS_30, "SDS25_+30"},
    {35.0f,  30.0f,  CH_OFF_AXIS_30, "SDS35_+30"},
    {45.0f,  30.0f,  CH_OFF_AXIS_30, "SDS45_+30"},
    {15.0f, -30.0f,  CH_OFF_AXIS_30, "SDS15_-30"},
    {25.0f, -30.0f,  CH_OFF_AXIS_30, "SDS25_-30"},
    {35.0f, -30.0f,  CH_OFF_AXIS_30, "SDS35_-30"},
    {45.0f, -30.0f,  CH_OFF_AXIS_30, "SDS45_-30"},
    
    // Off-axis 60 degrees
    {15.0f,  60.0f,  CH_OFF_AXIS_60, "SDS15_+60"},
    {25.0f,  60.0f,  CH_OFF_AXIS_60, "SDS25_+60"},
    {35.0f,  60.0f,  CH_OFF_AXIS_60, "SDS35_+60"},
    {15.0f, -60.0f,  CH_OFF_AXIS_60, "SDS15_-60"},
    {25.0f, -60.0f,  CH_OFF_AXIS_60, "SDS25_-60"},
    {35.0f, -60.0f,  CH_OFF_AXIS_60, "SDS35_-60"},
};

DetectorArray::DetectorArray(const float* source_mni, 
                              const float* primary_direction,
                              bool use_right_amyg) 
    : use_right_amyg_(use_right_amyg) {
    
    // Set default source position (right temporal scalp)
    if (source_mni) {
        source_pos_ = make_float3(source_mni[0], source_mni[1], source_mni[2]);
    } else {
        // Default: superior temporal gyrus region
        source_pos_ = make_float3(55.0f, 10.0f, -15.0f);
    }
    
    // Set target (amygdala)
    const float* amyg = use_right_amyg ? AMYG_RIGHT_MNI : AMYG_LEFT_MNI;
    target_pos_ = make_float3(amyg[0], amyg[1], amyg[2]);
    
    // Compute primary direction (source to amygdala)
    if (primary_direction) {
        primary_dir_ = make_float3(primary_direction[0], 
                                    primary_direction[1], 
                                    primary_direction[2]);
    } else {
        primary_dir_ = normalize(target_pos_ - source_pos_);
    }
    
    build_array();
}

void DetectorArray::build_array() {
    detectors_.clear();
    
    // Build orthonormal basis with primary_dir as "forward"
    float3 forward = primary_dir_;
    float3 up = make_float3(0, 0, 1);
    
    // Make sure forward and up aren't parallel
    if (fabsf(dot(forward, up)) > 0.99f) {
        up = make_float3(0, 1, 0);
    }
    
    float3 right = normalize(cross(forward, up));
    up = normalize(cross(right, forward));
    
    // Create detectors
    for (int i = 0; i < NUM_DEFAULT_CHANNELS; i++) {
        const auto& ch = DEFAULT_CHANNELS[i];
        
        // Convert polar to cartesian offset
        float angle_rad = ch.angle_deg * 3.14159265f / 180.0f;
        float cos_a = cosf(angle_rad);
        float sin_a = sinf(angle_rad);
        
        // Direction: primarily forward, with lateral component based on angle
        float3 offset = forward * cos_a * ch.sds + right * sin_a * ch.sds;
        
        Detector det;
        det.pos = source_pos_ + offset;
        det.radius = DEFAULT_DETECTOR_RADIUS;
        det.source_pos = source_pos_;
        det.source_radius = DEFAULT_SOURCE_RADIUS;
        det.sds = ch.sds;
        det.gate_start = 0.0f;
        det.gate_end = 5000.0f;
        det.use_gate = 0;
        
        detectors_.push_back(det);
    }
}

float3 DetectorArray::project_to_scalp(const float3& pos) {
    // In a full implementation, this would find the actual scalp surface
    // For now, return position as-is assuming it's already on/near surface
    return pos;
}

Detector* DetectorArray::upload_to_gpu() const {
    if (detectors_.empty()) return nullptr;
    
    Detector* device_det = nullptr;
    size_t size = detectors_.size() * sizeof(Detector);
    cudaMalloc(&device_det, size);
    cudaMemcpy(device_det, detectors_.data(), size, cudaMemcpyHostToDevice);
    
    return device_det;
}

void DetectorArray::print_config() const {
    printf("\nDetector Array Configuration:\n");
    printf("  Source position: (%.1f, %.1f, %.1f) mm\n", 
           source_pos_.x, source_pos_.y, source_pos_.z);
    printf("  Target (amygdala): (%.1f, %.1f, %.1f) mm\n",
           target_pos_.x, target_pos_.y, target_pos_.z);
    printf("  Primary direction: (%.3f, %.3f, %.3f)\n",
           primary_dir_.x, primary_dir_.y, primary_dir_.z);
    printf("  Number of channels: %zu\n\n", detectors_.size());
    
    printf("  Channel    SDS(mm)  Angle(deg)  Position (mm)\n");
    printf("  ---------------------------------------------------\n");
    
    for (int i = 0; i < NUM_DEFAULT_CHANNELS && i < static_cast<int>(detectors_.size()); i++) {
        const auto& det = detectors_[i];
        const auto& ch = DEFAULT_CHANNELS[i];
        printf("  %-10s %7.1f  %10.1f  (%.1f, %.1f, %.1f)\n",
               ch.name, ch.sds, ch.angle_deg,
               det.pos.x, det.pos.y, det.pos.z);
    }
    printf("\n");
}

} // namespace mmc
