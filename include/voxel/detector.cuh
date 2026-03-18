#pragma once
#include "types.cuh"
#include <vector>

// ---------------------------------------------------------------------------
// Detector placement
// ---------------------------------------------------------------------------
// Creates detectors on the scalp surface at various source-detector
// separations (SDS). For deep-tissue fNIRS targeting the amygdala,
// we use long SDS (30-50 mm) plus short-separation regression channels.
//
// Detectors are placed along the line from the source toward the amygdala
// projection on the scalp, at specified separations.
// ---------------------------------------------------------------------------

struct DetectorLayout {
    float src_x, src_y, src_z;          // source position [mm]
    std::vector<float> separations_mm;  // source-detector separations
    std::vector<float> angles_deg;      // angle from primary direction per detector
    float det_radius;                   // acceptance radius [mm]

    // Primary direction on scalp surface for detector placement (unit vector)
    float dir_x, dir_y, dir_z;
};

// Build default detector layout for amygdala targeting
// head_params used to place source/detectors on scalp surface
DetectorLayout default_detector_layout();

// Generate detector structs from layout
std::vector<Detector> build_detectors(const DetectorLayout& layout);
