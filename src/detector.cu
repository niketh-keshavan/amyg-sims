#include "detector.cuh"
#include "geometry.cuh"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Optimized detector layout for amygdala fNIRS
// ---------------------------------------------------------------------------
// Source positioned directly over the right amygdala projection on the
// temporal bone — the thinnest skull region (~2.5mm).
//
// Right amygdala center: (+24, -2, -18) mm from head center
// Temporal scalp surface above it: ~(+76, -2, -18) mm
//
// Source is projected radially outward from amygdala center onto scalp.
// This minimizes the photon path through skull to reach the amygdala.
//
// Detectors: 4mm radius SiPM arrays (e.g., Hamamatsu S14160-3050HS
// 3x3mm active area, or tiled 6x6mm MPPC arrays giving ~3-4mm effective
// radius in circular approximation).
//
// Layout optimized for TD-gated measurement:
//   - Short-separation (8mm): 2 regression channels
//   - Primary (0 deg): SDS 15-40mm, dense in 20-35mm sweet spot
//   - +/-30 deg offset: SDS 20, 25, 30, 35mm
//   - +/-60 deg offset: SDS 25, 35mm
// ---------------------------------------------------------------------------

DetectorLayout default_detector_layout() {
    DetectorLayout layout;

    // Source position: project from right amygdala center radially onto scalp
    // Amygdala at (+24, -2, -18), scalp ellipsoid (78, 95, 85)
    // Direction from center to amygdala = (24, -2, -18), normalize and scale to scalp
    float ax = 24.0f, ay = -2.0f, az = -18.0f;
    // For ellipsoidal projection: find t such that (t*ax/78)^2 + (t*ay/95)^2 + (t*az/85)^2 = 1
    float ea = ax / 78.0f, eb = ay / 95.0f, ec = az / 85.0f;
    float t_scalp = 1.0f / sqrtf(ea * ea + eb * eb + ec * ec);

    layout.src_x = t_scalp * ax;   // ~76mm lateral
    layout.src_y = t_scalp * ay;   // ~-6mm posterior
    layout.src_z = t_scalp * az;   // ~-57mm inferior

    printf("Source placement: projecting from amygdala (%.0f,%.0f,%.0f) onto scalp\n",
           ax, ay, az);
    printf("  Source on scalp: (%.1f, %.1f, %.1f) mm\n",
           layout.src_x, layout.src_y, layout.src_z);

    // Primary direction: inferior-posterior along temporal surface
    // (roughly toward where more amygdala sensitivity is expected)
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

    // Primary direction (0 deg) - dense sampling in 20-35mm sweet spot
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

    // 4mm radius SiPM arrays
    layout.det_radius = 4.0f;

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

    // Surface normal at source (radial for ellipsoid)
    float nmag = sqrtf(layout.src_x * layout.src_x +
                       layout.src_y * layout.src_y +
                       layout.src_z * layout.src_z);
    float snx = layout.src_x / nmag;
    float sny = layout.src_y / nmag;
    float snz = layout.src_z / nmag;

    // Primary tangent
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

    printf("Detector configuration (%d detectors, %.0fmm radius SiPM):\n",
           (int)layout.separations_mm.size(), layout.det_radius);
    printf("  Source (raw): (%.1f, %.1f, %.1f) mm\n",
           layout.src_x, layout.src_y, layout.src_z);
    printf("  Primary dir: (%.3f, %.3f, %.3f)\n",
           layout.dir_x, layout.dir_y, layout.dir_z);

    for (size_t i = 0; i < layout.separations_mm.size(); i++) {
        float sds = layout.separations_mm[i];
        float angle_rad = layout.angles_deg[i] * 3.14159265358979f / 180.0f;

        float dx = t1x * cosf(angle_rad) + t2x * sinf(angle_rad);
        float dy = t1y * cosf(angle_rad) + t2y * sinf(angle_rad);
        float dz = t1z * cosf(angle_rad) + t2z * sinf(angle_rad);

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

        Detector det;
        det.x = det_cx + cx;
        det.y = det_cy + cy;
        det.z = det_cz + cz;
        det.radius = layout.det_radius;
        det.id = (int)i;
        
        // Surface normal: radial direction from head center (outward from head)
        float rx = det_cx, ry = det_cy, rz = det_cz;
        float r_mag = sqrtf(rx*rx + ry*ry + rz*rz);
        if (r_mag > 1e-6f) {
            det.nx = rx / r_mag;
            det.ny = ry / r_mag;
            det.nz = rz / r_mag;
        } else {
            det.nx = det.ny = 0.0f; det.nz = 1.0f;
        }
        
        // Critical angle: photons with dot(dir, normal) < cos_critical are rejected
        // 69 degrees acceptance cone: cos(69°) ≈ 0.358
        // This allows photons within 69° of surface normal to be detected
        det.n_critical = 0.358f;  // cos(69°)

        printf("  Det %2d: SDS=%4.0f mm  angle=%+4.0f deg -> (%.1f, %.1f, %.1f) mm  "
               "normal=(%.2f, %.2f, %.2f)  cos_crit=%.3f\n",
               det.id, sds, layout.angles_deg[i], det.x, det.y, det.z,
               det.nx, det.ny, det.nz, det.n_critical);

        dets.push_back(det);
    }

    return dets;
}
