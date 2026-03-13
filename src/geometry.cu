#include "geometry.cuh"
#include <cmath>
#include <cstdio>
#include <chrono>

// ---------------------------------------------------------------------------
// Default adult head model with non-uniform skull thickness
// ---------------------------------------------------------------------------
// References:
//   Okada & Delpy (2003) - Near-infrared light propagation in an adult head
//   Li et al. (2015) - Skull thickness measurements from CT
//   Lynnerup (2005) - Cranial thickness in relation to age, sex, and body build
//   Strangman et al. (2014) - Scalp and skull influence on NIRS signal
//   Amunts et al. (2005) - Amygdala morphometry (MNI coordinates)
//
// Skull thickness anatomy:
//   Temporal squamous: 2-3 mm (thinnest, above ear)
//   Frontal bone: 6-8 mm
//   Parietal vertex: 6-7 mm
//   Occipital: 7-10 mm
//
// Model: skull inner surface semi-axes interpolate between "thin" (temporal)
// and "thick" (vertex) based on voxel position. The blending weight uses
// |x|/scalp_a (laterality) and z (inferior = temporal).
// ---------------------------------------------------------------------------

HeadModelParams default_head_model() {
    HeadModelParams p{};

    // Grid: 400x400x400 at 0.5mm = 200x200x200 mm volume
    p.nx = 400; p.ny = 400; p.nz = 400;
    p.dx = 0.5f;

    // Scalp outer surface (adult head: ~156 ML x 190 AP x 170 SI mm)
    p.scalp_a = 78.0f; p.scalp_b = 95.0f; p.scalp_c = 85.0f;

    // Skull outer (scalp thickness ~4mm uniform)
    p.skull_a = 74.0f; p.skull_b = 91.0f; p.skull_c = 81.0f;

    // Skull inner surface: non-uniform thickness
    // At TEMPORAL bone (thin, ~2.5mm): skull inner = skull outer - 2.5mm
    p.skull_inner_min_a = 71.5f;  // 74 - 2.5
    p.skull_inner_min_b = 88.5f;  // 91 - 2.5
    p.skull_inner_min_c = 78.5f;  // 81 - 2.5

    // At VERTEX/frontal/occipital (thick, ~7mm): skull inner = skull outer - 7mm
    p.skull_inner_max_a = 67.0f;  // 74 - 7
    p.skull_inner_max_b = 84.0f;  // 91 - 7
    p.skull_inner_max_c = 74.0f;  // 81 - 7

    // CSF outer = skull inner at thickest (vertex) for consistency
    // The non-uniform skull carves into what would be CSF space at temporal
    p.csf_a = 67.0f; p.csf_b = 84.0f; p.csf_c = 74.0f;

    // Gray matter outer / CSF inner (CSF ~1.5mm)
    p.gm_a = 65.5f; p.gm_b = 82.5f; p.gm_c = 72.5f;

    // White matter outer / GM inner (cortical GM ~3.5mm)
    p.wm_a = 62.0f; p.wm_b = 79.0f; p.wm_c = 69.0f;

    // Amygdala (MNI-based, Amunts et al. 2005)
    // Left amygdala
    p.amyg_l_cx = -24.0f;
    p.amyg_l_cy =  -2.0f;
    p.amyg_l_cz = -18.0f;
    p.amyg_l_a  =   5.0f;
    p.amyg_l_b  =   9.0f;
    p.amyg_l_c  =   6.0f;

    // Right amygdala (mirror)
    p.amyg_r_cx =  24.0f;
    p.amyg_r_cy =  -2.0f;
    p.amyg_r_cz = -18.0f;
    p.amyg_r_a  =   5.0f;
    p.amyg_r_b  =   9.0f;
    p.amyg_r_c  =   6.0f;

    return p;
}

// ---------------------------------------------------------------------------
// Skull thickness blending weight
// ---------------------------------------------------------------------------
// Returns 0.0 at temporal bone (thin), 1.0 at vertex (thick)
// Based on anatomical position:
//   - High |x|/a (lateral) AND low z (inferior) => temporal => thin
//   - Low |x|/a (medial) OR high z (superior) => vertex => thick
//
// We use a smooth blend: w = 1 - lateral_factor * inferior_factor
// where lateral_factor peaks at |x|=scalp_a (pure lateral)
// and inferior_factor peaks at z << 0 (inferior)
// ---------------------------------------------------------------------------
static inline float skull_thickness_weight(float x, float y, float z,
                                            float scalp_a, float scalp_c) {
    // Laterality: how far lateral is this point? (0=midline, 1=fully lateral)
    float lat = fabsf(x) / scalp_a;
    lat = fminf(lat, 1.0f);
    // Smooth step for laterality (temporal region starts at ~60% lateral)
    float lat_factor = 0.0f;
    if (lat > 0.4f) {
        float t = (lat - 0.4f) / 0.4f;  // 0 at 40% lateral, 1 at 80%
        t = fminf(t, 1.0f);
        lat_factor = t * t * (3.0f - 2.0f * t);  // smoothstep
    }

    // Inferiority: temporal bone is inferior (z < 0)
    // z/scalp_c: -1 = bottom, 0 = equator, +1 = top
    float z_norm = z / scalp_c;
    float inf_factor = 0.0f;
    if (z_norm < 0.3f) {
        float t = (0.3f - z_norm) / 0.8f;  // peaks at z_norm = -0.5
        t = fminf(t, 1.0f);
        inf_factor = t * t * (3.0f - 2.0f * t);
    }

    // Temporal thinning = lateral AND inferior
    float temporal = lat_factor * inf_factor;

    // Weight: 0 = thin (temporal), 1 = thick (vertex)
    return 1.0f - temporal;
}

// ---------------------------------------------------------------------------
// Build the voxelised volume with non-uniform skull
// ---------------------------------------------------------------------------
std::vector<uint8_t> build_head_volume(const HeadModelParams& p) {
    size_t total = (size_t)p.nx * p.ny * p.nz;
    std::vector<uint8_t> vol(total, TISSUE_AIR);

    float cx = p.nx * p.dx * 0.5f;
    float cy = p.ny * p.dx * 0.5f;
    float cz = p.nz * p.dx * 0.5f;

    printf("Building head volume: %dx%dx%d voxels (%.1f mm resolution)\n",
           p.nx, p.ny, p.nz, p.dx);
    printf("Volume center: (%.1f, %.1f, %.1f) mm\n", cx, cy, cz);
    printf("Non-uniform skull: temporal ~%.1fmm, vertex ~%.1fmm\n",
           p.skull_a - p.skull_inner_min_a,
           p.skull_a - p.skull_inner_max_a);

    auto build_start = std::chrono::high_resolution_clock::now();

    for (int iz = 0; iz < p.nz; iz++) {
        float z = (iz + 0.5f) * p.dx - cz;

        if (p.nz >= 100 && iz % (p.nz / 50) == 0) {
            auto now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - build_start).count();
            double pct = 100.0 * iz / p.nz;
            int bar_width = 30;
            int filled = (int)(bar_width * pct / 100.0);
            printf("\r  Building volume: [");
            for (int i = 0; i < bar_width; i++)
                printf("%s", i < filled ? "\xe2\x96\x88" : "\xe2\x96\x91");
            printf("] %5.1f%%", pct);
            if (iz > 0) {
                double eta = elapsed * (p.nz - iz) / iz;
                if (eta > 60.0)
                    printf("  ETA %dm%02ds", (int)(eta / 60), (int)eta % 60);
                else
                    printf("  ETA %.0fs", eta);
            }
            printf("    ");
            fflush(stdout);
        }

        for (int iy = 0; iy < p.ny; iy++) {
            float y = (iy + 0.5f) * p.dx - cy;
            for (int ix = 0; ix < p.nx; ix++) {
                float x = (ix + 0.5f) * p.dx - cx;

                size_t idx = ix + (size_t)iy * p.nx + (size_t)iz * p.nx * p.ny;

                // Check amygdala first (embedded in white/gray matter)
                float al = ((x - p.amyg_l_cx) / p.amyg_l_a) * ((x - p.amyg_l_cx) / p.amyg_l_a)
                         + ((y - p.amyg_l_cy) / p.amyg_l_b) * ((y - p.amyg_l_cy) / p.amyg_l_b)
                         + ((z - p.amyg_l_cz) / p.amyg_l_c) * ((z - p.amyg_l_cz) / p.amyg_l_c);

                float ar = ((x - p.amyg_r_cx) / p.amyg_r_a) * ((x - p.amyg_r_cx) / p.amyg_r_a)
                         + ((y - p.amyg_r_cy) / p.amyg_r_b) * ((y - p.amyg_r_cy) / p.amyg_r_b)
                         + ((z - p.amyg_r_cz) / p.amyg_r_c) * ((z - p.amyg_r_cz) / p.amyg_r_c);

                if (al <= 1.0f || ar <= 1.0f) {
                    vol[idx] = TISSUE_AMYGDALA;
                    continue;
                }

                // White matter
                float e_wm = (x/p.wm_a)*(x/p.wm_a) + (y/p.wm_b)*(y/p.wm_b) + (z/p.wm_c)*(z/p.wm_c);
                if (e_wm <= 1.0f) { vol[idx] = TISSUE_WHITE; continue; }

                // Gray matter
                float e_gm = (x/p.gm_a)*(x/p.gm_a) + (y/p.gm_b)*(y/p.gm_b) + (z/p.gm_c)*(z/p.gm_c);
                if (e_gm <= 1.0f) { vol[idx] = TISSUE_GRAY; continue; }

                // CSF
                float e_csf = (x/p.csf_a)*(x/p.csf_a) + (y/p.csf_b)*(y/p.csf_b) + (z/p.csf_c)*(z/p.csf_c);
                if (e_csf <= 1.0f) { vol[idx] = TISSUE_CSF; continue; }

                // --- Non-uniform skull ---
                // Blend skull inner surface between thin (temporal) and thick (vertex)
                float w = skull_thickness_weight(x, y, z, p.scalp_a, p.scalp_c);

                // Interpolated skull inner semi-axes at this position
                float si_a = p.skull_inner_min_a * (1.0f - w) + p.skull_inner_max_a * w;
                float si_b = p.skull_inner_min_b * (1.0f - w) + p.skull_inner_max_b * w;
                float si_c = p.skull_inner_min_c * (1.0f - w) + p.skull_inner_max_c * w;

                // Is this point inside the skull inner surface?
                float e_si = (x/si_a)*(x/si_a) + (y/si_b)*(y/si_b) + (z/si_c)*(z/si_c);
                if (e_si <= 1.0f) {
                    // Inside skull inner = CSF (between non-uniform skull and fixed CSF)
                    vol[idx] = TISSUE_CSF;
                    continue;
                }

                // Skull outer
                float e_sk = (x/p.skull_a)*(x/p.skull_a) + (y/p.skull_b)*(y/p.skull_b) + (z/p.skull_c)*(z/p.skull_c);
                if (e_sk <= 1.0f) { vol[idx] = TISSUE_SKULL; continue; }

                // Scalp
                float e_sc = (x/p.scalp_a)*(x/p.scalp_a) + (y/p.scalp_b)*(y/p.scalp_b) + (z/p.scalp_c)*(z/p.scalp_c);
                if (e_sc <= 1.0f) { vol[idx] = TISSUE_SCALP; continue; }
            }
        }
    }

    {
        auto build_end = std::chrono::high_resolution_clock::now();
        double build_sec = std::chrono::duration<double>(build_end - build_start).count();
        printf("\r  Building volume: [");
        for (int i = 0; i < 30; i++) printf("\xe2\x96\x88");
        printf("] 100.0%%  Done in %.1fs          \n", build_sec);
    }

    // Count voxels per tissue type
    size_t counts[NUM_TISSUE_TYPES] = {};
    for (size_t i = 0; i < total; i++) counts[vol[i]]++;

    const char* names[] = {"Air", "Scalp", "Skull", "CSF", "Gray Matter",
                           "White Matter", "Amygdala"};
    printf("Tissue voxel counts:\n");
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        printf("  %-14s: %zu voxels (%.1f mm^3)\n",
               names[t], counts[t], counts[t] * p.dx * p.dx * p.dx);
    }

    // Print skull thickness at key locations
    printf("Skull thickness verification:\n");
    // Check right temporal (x=+74, y=0, z=-10)
    float test_pts[][3] = {
        { 74.0f,  0.0f, -10.0f},  // right temporal
        {-74.0f,  0.0f, -10.0f},  // left temporal
        {  0.0f,  0.0f,  81.0f},  // vertex
        {  0.0f, 91.0f,   0.0f},  // frontal
        {  0.0f,-91.0f,   0.0f},  // occipital
    };
    const char* loc_names[] = {"R temporal", "L temporal", "Vertex", "Frontal", "Occipital"};
    for (int i = 0; i < 5; i++) {
        float tx = test_pts[i][0], ty = test_pts[i][1], tz = test_pts[i][2];
        float w = skull_thickness_weight(tx, ty, tz, p.scalp_a, p.scalp_c);
        float si_a = p.skull_inner_min_a * (1.0f - w) + p.skull_inner_max_a * w;
        float thickness_a = p.skull_a - si_a;
        printf("  %-12s: w=%.2f, thickness~%.1f mm\n", loc_names[i], w, thickness_a);
    }

    return vol;
}
