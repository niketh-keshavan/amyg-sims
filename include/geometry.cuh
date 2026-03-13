#pragma once
#include "types.cuh"
#include <vector>

// ---------------------------------------------------------------------------
// Non-uniform skull head model
// ---------------------------------------------------------------------------
// Builds a voxelised head model with concentric ellipsoidal shells and
// a non-uniform skull thickness that is thin (~2.5mm) at the temporal bone
// and thicker (~7mm) at the vertex/frontal/occipital regions.
//
// Skull thickness model:
//   The skull inner surface is parameterized so that temporal regions
//   (lateral, inferior) have thinner skull, matching anatomical data:
//   - Temporal squamous bone: 2-3 mm (Li et al. 2015, Lynnerup 2005)
//   - Frontal/parietal vertex: 6-8 mm
//   - Occipital: 7-10 mm
//
// This is critical for amygdala fNIRS since photons enter through
// the thin temporal bone above the ear.
// ---------------------------------------------------------------------------

struct HeadModelParams {
    // Grid
    int   nx, ny, nz;
    float dx;              // mm per voxel

    // Ellipsoid semi-axes [mm] (outer boundary of each layer)
    float scalp_a, scalp_b, scalp_c;
    float skull_a, skull_b, skull_c;  // skull OUTER boundary

    // Skull inner boundary is computed per-voxel (non-uniform thickness)
    // These define the MINIMUM skull inner surface (thinnest = temporal)
    float skull_inner_min_a, skull_inner_min_b, skull_inner_min_c;
    // These define the MAXIMUM skull inner surface (thickest = vertex)
    float skull_inner_max_a, skull_inner_max_b, skull_inner_max_c;

    // CSF, GM, WM boundaries (concentric from skull inner)
    float csf_a,   csf_b,   csf_c;
    float gm_a,    gm_b,    gm_c;
    float wm_a,    wm_b,    wm_c;

    // Amygdala: modeled as small ellipsoids
    float amyg_l_cx, amyg_l_cy, amyg_l_cz;
    float amyg_l_a,  amyg_l_b,  amyg_l_c;

    float amyg_r_cx, amyg_r_cy, amyg_r_cz;
    float amyg_r_a,  amyg_r_b,  amyg_r_c;
};

// Build the default adult head model (literature-based dimensions)
HeadModelParams default_head_model();

// Voxelise the model
std::vector<uint8_t> build_head_volume(const HeadModelParams& p);
