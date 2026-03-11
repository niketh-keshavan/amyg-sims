#include "geometry.cuh"
#include <cmath>
#include <cstdio>
#include <chrono>

// ---------------------------------------------------------------------------
// Default adult head model parameters (literature-based)
// ---------------------------------------------------------------------------
// References:
//   - Okada & Delpy (2003) - Near-infrared light propagation in an adult head model
//   - Strangman et al. (2014) - Scalp and skull influence on NIRS signal
//   - Amunts et al. (2005) - Amygdala morphometry
//
// Head center is at the grid center.
// Coordinate system: X = left-right, Y = anterior-posterior, Z = inferior-superior
// ---------------------------------------------------------------------------

HeadModelParams default_head_model() {
    HeadModelParams p{};

    // Grid: 1000x1000x1000 at 0.1 mm resolution = 100x100x100 mm volume
    // 0.1mm voxel size for high-resolution photon path tracking
    p.nx = 1000; p.ny = 1000; p.nz = 1000;
    p.dx = 0.1f;  // mm

    // Head center is at grid center: (50, 50, 50) mm
    // Ellipsoidal layers (semi-axes in mm)
    // These represent half-widths along each axis

    // Scalp outer surface
    p.scalp_a = 48.0f; p.scalp_b = 48.0f; p.scalp_c = 48.0f;

    // Skull outer (scalp thickness ~3-5 mm)
    p.skull_a = 44.0f; p.skull_b = 44.0f; p.skull_c = 44.0f;

    // CSF outer / skull inner (skull thickness ~6-7 mm)
    p.csf_a = 38.0f; p.csf_b = 38.0f; p.csf_c = 38.0f;

    // Gray matter outer / CSF inner (CSF ~1-2 mm)
    p.gm_a = 36.5f; p.gm_b = 36.5f; p.gm_c = 36.5f;

    // White matter outer / GM inner (cortical GM ~3-4 mm)
    p.wm_a = 33.0f; p.wm_b = 33.0f; p.wm_c = 33.0f;

    // Amygdala: ~15-20 mm long, ~10 mm wide, ~12 mm tall
    // Located in medial temporal lobe, roughly:
    //   ~25 mm lateral, ~5 mm anterior, ~15 mm inferior to head center
    // (In our coordinate system, from center at 50,50,50 mm)

    // Left amygdala
    p.amyg_l_cx = -24.0f;  // left of center (negative X)
    p.amyg_l_cy =  5.0f;   // slightly anterior
    p.amyg_l_cz = -15.0f;  // inferior
    p.amyg_l_a  =  5.0f;   // semi-axis X (medial-lateral)
    p.amyg_l_b  =  9.0f;   // semi-axis Y (anterior-posterior)
    p.amyg_l_c  =  6.0f;   // semi-axis Z (inferior-superior)

    // Right amygdala (mirror)
    p.amyg_r_cx =  24.0f;
    p.amyg_r_cy =  5.0f;
    p.amyg_r_cz = -15.0f;
    p.amyg_r_a  =  5.0f;
    p.amyg_r_b  =  9.0f;
    p.amyg_r_c  =  6.0f;

    return p;
}

// ---------------------------------------------------------------------------
// Build the voxelised volume
// ---------------------------------------------------------------------------
std::vector<uint8_t> build_head_volume(const HeadModelParams& p) {
    size_t total = (size_t)p.nx * p.ny * p.nz;
    std::vector<uint8_t> vol(total, TISSUE_AIR);

    float cx = p.nx * p.dx * 0.5f;  // center in mm
    float cy = p.ny * p.dx * 0.5f;
    float cz = p.nz * p.dx * 0.5f;

    printf("Building head volume: %dx%dx%d voxels (%.1f mm resolution)\n",
           p.nx, p.ny, p.nz, p.dx);
    printf("Volume center: (%.1f, %.1f, %.1f) mm\n", cx, cy, cz);

    auto build_start = std::chrono::high_resolution_clock::now();

    for (int iz = 0; iz < p.nz; iz++) {
        float z = (iz + 0.5f) * p.dx - cz;  // relative to center

        // Progress bar every 2% of slices
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

                // Check ellipsoidal shells (inside-out: assign deepest layer first)
                // Ellipsoid test: (x/a)^2 + (y/b)^2 + (z/c)^2 <= 1

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

                // Concentric shells
                float e_wm = (x/p.wm_a)*(x/p.wm_a) + (y/p.wm_b)*(y/p.wm_b) + (z/p.wm_c)*(z/p.wm_c);
                if (e_wm <= 1.0f) { vol[idx] = TISSUE_WHITE; continue; }

                float e_gm = (x/p.gm_a)*(x/p.gm_a) + (y/p.gm_b)*(y/p.gm_b) + (z/p.gm_c)*(z/p.gm_c);
                if (e_gm <= 1.0f) { vol[idx] = TISSUE_GRAY; continue; }

                float e_csf = (x/p.csf_a)*(x/p.csf_a) + (y/p.csf_b)*(y/p.csf_b) + (z/p.csf_c)*(z/p.csf_c);
                if (e_csf <= 1.0f) { vol[idx] = TISSUE_CSF; continue; }

                float e_sk = (x/p.skull_a)*(x/p.skull_a) + (y/p.skull_b)*(y/p.skull_b) + (z/p.skull_c)*(z/p.skull_c);
                if (e_sk <= 1.0f) { vol[idx] = TISSUE_SKULL; continue; }

                float e_sc = (x/p.scalp_a)*(x/p.scalp_a) + (y/p.scalp_b)*(y/p.scalp_b) + (z/p.scalp_c)*(z/p.scalp_c);
                if (e_sc <= 1.0f) { vol[idx] = TISSUE_SCALP; continue; }

                // Outside head = air (already default)
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

    return vol;
}

// ---------------------------------------------------------------------------
// Optical properties at 760 nm and 850 nm
// ---------------------------------------------------------------------------
// Values from literature (Strangman et al. 2003, Okada & Delpy 2003,
// Jacques 2013, Sassaroli & Bhatt 2020)
//
// mu_a: absorption coefficient [1/mm]
// mu_s: reduced scattering coefficient [1/mm] (we store mu_s' and set g)
//       mu_s = mu_s' / (1 - g)
// g:    anisotropy factor
// n:    refractive index
// ---------------------------------------------------------------------------
void get_optical_properties(int wavelength_idx, OpticalProps props[NUM_TISSUE_TYPES]) {
    if (wavelength_idx == 0) {
        // 760 nm
        props[TISSUE_AIR]      = {0.0f,    0.0f,    1.0f, 1.000f};
        props[TISSUE_SCALP]    = {0.0191f, 10.7f,   0.9f, 1.37f};  // mu_s'=1.07
        props[TISSUE_SKULL]    = {0.0136f, 12.5f,   0.9f, 1.56f};  // mu_s'=1.25
        props[TISSUE_CSF]      = {0.0026f,  0.1f,   0.9f, 1.33f};  // nearly transparent
        props[TISSUE_GRAY]     = {0.0186f, 11.0f,   0.9f, 1.37f};  // mu_s'=1.10
        props[TISSUE_WHITE]    = {0.0167f, 13.8f,   0.9f, 1.37f};  // mu_s'=1.38
        props[TISSUE_AMYGDALA] = {0.0200f, 11.0f,   0.9f, 1.37f};  // similar to GM, slightly higher mu_a
    } else {
        // 850 nm
        props[TISSUE_AIR]      = {0.0f,    0.0f,    1.0f, 1.000f};
        props[TISSUE_SCALP]    = {0.0170f,  9.4f,   0.9f, 1.37f};
        props[TISSUE_SKULL]    = {0.0116f, 10.9f,   0.9f, 1.56f};
        props[TISSUE_CSF]      = {0.0026f,  0.1f,   0.9f, 1.33f};
        props[TISSUE_GRAY]     = {0.0192f,  9.6f,   0.9f, 1.37f};
        props[TISSUE_WHITE]    = {0.0145f, 12.1f,   0.9f, 1.37f};
        props[TISSUE_AMYGDALA] = {0.0210f,  9.6f,   0.9f, 1.37f};
    }
}
