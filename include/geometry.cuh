#pragma once
#include "types.cuh"
#include <vector>

// ---------------------------------------------------------------------------
// Layered spherical head model
// ---------------------------------------------------------------------------
// Builds a voxelised head model with concentric ellipsoidal shells
// (realistic adult head proportions: ~156 ML x 190 AP x 170 SI mm)
// for scalp, skull, CSF, gray matter, white matter, plus a pair of
// embedded amygdala volumes (left & right).
//
// Returns: tissue label volume (nx * ny * nz) as a flat vector.
// ---------------------------------------------------------------------------

struct HeadModelParams {
    // Grid
    int   nx, ny, nz;
    float dx;              // mm per voxel

    // Ellipsoid semi-axes [mm] (outer boundary of each layer)
    // Order: scalp(outer), skull_outer, skull_inner(=CSF_outer),
    //        CSF_inner(=GM_outer), GM_inner(=WM_outer)
    float scalp_a, scalp_b, scalp_c;
    float skull_a, skull_b, skull_c;
    float csf_a,   csf_b,   csf_c;
    float gm_a,    gm_b,    gm_c;
    float wm_a,    wm_b,    wm_c;

    // Amygdala: modeled as small ellipsoids
    // Left amygdala center (mm, relative to head center)
    float amyg_l_cx, amyg_l_cy, amyg_l_cz;
    float amyg_l_a,  amyg_l_b,  amyg_l_c;  // semi-axes [mm]

    // Right amygdala
    float amyg_r_cx, amyg_r_cy, amyg_r_cz;
    float amyg_r_a,  amyg_r_b,  amyg_r_c;
};

// Build the default adult head model (literature-based dimensions)
HeadModelParams default_head_model();

// Voxelise the model
std::vector<uint8_t> build_head_volume(const HeadModelParams& p);

// Get optical properties for all tissues at a given wavelength index
// wavelength_idx: 0 = 760 nm, 1 = 850 nm
void get_optical_properties(int wavelength_idx, OpticalProps props[NUM_TISSUE_TYPES]);
