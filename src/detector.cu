#include "detector.cuh"
#include "geometry.cuh"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// TD-gated optimized detector layout for amygdala fNIRS
// ---------------------------------------------------------------------------
// Strategy: source on temporal scalp (T4), detectors optimized for TD-gated
// measurement with 2-minute integration time.
//
// Key findings from 10B-photon sim:
//   - SDS 25 mm: best dual-wavelength MBLL (3.5s for 1µM HbO @ 1s)
//   - SDS 30-40 mm: highest single-channel TD SNR at 850nm
//   - Late gates (2-5 ns) carry all amygdala sensitivity
//   - Angles 0° and ±30° best for amygdala
//
// Layout:
//   - Short-separation (8 mm): 2 regression channels (0°, 180°)
//   - Primary (0°): SDS 15, 20, 22, 25, 28, 30, 33, 35, 40 mm (dense 20-35)
//   - ±30° offset: SDS 20, 25, 30, 35 mm
//   - ±60° offset: SDS 25, 35 mm
// Total: 23 detectors
// ---------------------------------------------------------------------------

DetectorLayout default_detector_layout() {
    DetectorLayout layout;

    // Source position: on the right temporal scalp surface
    layout.src_x = 75.0f;
    layout.src_y =  5.0f;
    layout.src_z = -10.0f;

    // Primary direction: inferior-posterior (toward amygdala projection)
    layout.dir_x =  0.0f;
    layout.dir_y = -0.3f;
    layout.dir_z = -0.954f;

    float mag = sqrtf(layout.dir_x * layout.dir_x +
                      layout.dir_y * layout.dir_y +
                      layout.dir_z * layout.dir_z);
    layout.dir_x /= mag;
    layout.dir_y /= mag;
    layout.dir_z /= mag;

    // Short-separation regression channels
    float ss_angles[] = {0.0f, 180.0f};
    for (int i = 0; i < 2; i++) {
        layout.separations_mm.push_back(8.0f);
        layout.angles_deg.push_back(ss_angles[i]);
    }

    // Primary direction (0 deg) — dense sampling in 20-35mm sweet spot
    float primary_sds[] = {15.0f, 20.0f, 22.0f, 25.0f, 28.0f, 30.0f, 33.0f, 35.0f, 40.0f};
    for (int i = 0; i < 9; i++) {
        layout.separations_mm.push_back(primary_sds[i]);
        layout.angles_deg.push_back(0.0f);
    }

    // +30 deg direction
    float off30_sds[] = {20.0f, 25.0f, 30.0f, 35.0f};
    for (int i = 0; i < 4; i++) {
        layout.separations_mm.push_back(off30_sds[i]);
        layout.angles_deg.push_back(30.0f);
    }

    // -30 deg direction
    for (int i = 0; i < 4; i++) {
        layout.separations_mm.push_back(off30_sds[i]);
        layout.angles_deg.push_back(-30.0f);
    }

    // +60 deg direction
    float off60_sds[] = {25.0f, 35.0f};
    for (int i = 0; i < 2; i++) {
        layout.separations_mm.push_back(off60_sds[i]);
        layout.angles_deg.push_back(60.0f);
    }

    // -60 deg direction
    for (int i = 0; i < 2; i++) {
        layout.separations_mm.push_back(off60_sds[i]);
        layout.angles_deg.push_back(-60.0f);
    }

    layout.det_radius = 2.0f;

    return layout;
}

// ---------------------------------------------------------------------------
// Build detector structs from layout
// ---------------------------------------------------------------------------
std::vector<Detector> build_detectors(const DetectorLayout& layout) {
    std::vector<Detector> dets;

    HeadModelParams hm = default_head_model();
    float cx = hm.nx * hm.dx * 0.5f;
    float cy = hm.ny * hm.dx * 0.5f;
    float cz = hm.nz * hm.dx * 0.5f;

    // Compute surface normal at source (radial direction for spherical head)
    float nmag = sqrtf(layout.src_x * layout.src_x +
                       layout.src_y * layout.src_y +
                       layout.src_z * layout.src_z);
    float snx = layout.src_x / nmag;
    float sny = layout.src_y / nmag;
    float snz = layout.src_z / nmag;

    // Primary tangent (the provided direction)
    float t1x = layout.dir_x;
    float t1y = layout.dir_y;
    float t1z = layout.dir_z;

    // Second tangent: surface_normal x primary_dir
    float t2x = sny * t1z - snz * t1y;
    float t2y = snz * t1x - snx * t1z;
    float t2z = snx * t1y - sny * t1x;
    float t2mag = sqrtf(t2x * t2x + t2y * t2y + t2z * t2z);
    if (t2mag > 1e-6f) {
        t2x /= t2mag; t2y /= t2mag; t2z /= t2mag;
    }

    printf("Detector configuration (high-density, %d detectors):\n",
           (int)layout.separations_mm.size());
    printf("  Source (raw): (%.1f, %.1f, %.1f) mm\n",
           layout.src_x, layout.src_y, layout.src_z);
    printf("  Primary dir: (%.3f, %.3f, %.3f)\n",
           layout.dir_x, layout.dir_y, layout.dir_z);
    printf("  Tangent2:    (%.3f, %.3f, %.3f)\n", t2x, t2y, t2z);

    for (size_t i = 0; i < layout.separations_mm.size(); i++) {
        float sds = layout.separations_mm[i];
        float angle_rad = layout.angles_deg[i] * 3.14159265358979f / 180.0f;

        // Rotate primary direction by angle around surface normal
        float dx = t1x * cosf(angle_rad) + t2x * sinf(angle_rad);
        float dy = t1y * cosf(angle_rad) + t2y * sinf(angle_rad);
        float dz = t1z * cosf(angle_rad) + t2z * sinf(angle_rad);

        // Place detector at SDS distance from source
        float det_cx = layout.src_x + dx * sds;
        float det_cy = layout.src_y + dy * sds;
        float det_cz = layout.src_z + dz * sds;

        // Project onto scalp surface (just inside)
        float ex = det_cx / hm.scalp_a;
        float ey = det_cy / hm.scalp_b;
        float ez = det_cz / hm.scalp_c;
        float e_mag = sqrtf(ex * ex + ey * ey + ez * ez);
        if (e_mag > 0.01f) {
            float scale = 0.99f / e_mag;
            det_cx *= scale;
            det_cy *= scale;
            det_cz *= scale;
        }

        // Convert to voxel-space coordinates
        Detector det;
        det.x = det_cx + cx;
        det.y = det_cy + cy;
        det.z = det_cz + cz;
        det.radius = layout.det_radius;
        det.id = (int)i;

        printf("  Det %2d: SDS=%4.0f mm  angle=%+4.0f deg -> voxel(%.1f, %.1f, %.1f) mm\n",
               det.id, sds, layout.angles_deg[i], det.x, det.y, det.z);

        dets.push_back(det);
    }

    return dets;
}
