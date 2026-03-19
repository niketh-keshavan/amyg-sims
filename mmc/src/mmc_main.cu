#include "types.cuh"
#include "optical_properties.cuh"
#include "mmc_mesh.cuh"
#include "mmc_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Save binary file
// ---------------------------------------------------------------------------
static void save_binary(const char* filename, const void* data, size_t bytes) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename); return; }
    fwrite(data, 1, bytes, f);
    fclose(f);
    printf("Saved %s (%zu bytes)\n", filename, bytes);
}

// ---------------------------------------------------------------------------
// Save JSON results (same format as voxel MC for analysis pipeline compat)
// ---------------------------------------------------------------------------
static void save_results_json(
    const char* filename,
    const DetectorResult* results,
    int n_dets,
    const std::vector<float>& separations,
    const std::vector<float>& angles,
    float wavelength_nm,
    uint64_t num_photons,
    const double* gated_weight,
    const double* gated_partial_pl,
    const uint64_t* gated_count
) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    fprintf(f, "{\n");
    fprintf(f, "  \"wavelength_nm\": %.0f,\n", wavelength_nm);
    fprintf(f, "  \"num_photons\": %llu,\n", (unsigned long long)num_photons);
    fprintf(f, "  \"geometry_model\": \"mmc_mni152\",\n");
    fprintf(f, "  \"scattering_model\": \"mie_power_law\",\n");
    fprintf(f, "  \"tpsf_bins\": %d,\n", TPSF_BINS);
    fprintf(f, "  \"tpsf_bin_ps\": %.1f,\n", (double)TPSF_BIN_PS);
    fprintf(f, "  \"time_gate_edges_ps\": [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],\n");
    fprintf(f, "  \"detectors\": [\n");

    const char* tissue_names[] = {
        "air", "scalp", "skull", "csf", "gray_matter", "white_matter", "amygdala"
    };
    const float gate_lo[] = {0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000};
    const float gate_hi[] = {500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 100000};

    for (int d = 0; d < n_dets; d++) {
        fprintf(f, "    {\n");
        fprintf(f, "      \"id\": %d,\n", d);
        fprintf(f, "      \"sds_mm\": %.1f,\n", separations[d]);
        fprintf(f, "      \"angle_deg\": %.1f,\n",
                d < (int)angles.size() ? angles[d] : 0.0f);
        fprintf(f, "      \"detected_photons\": %llu,\n",
                (unsigned long long)results[d].num_detected);
        fprintf(f, "      \"total_weight\": %.10e,\n", results[d].total_weight);
        fprintf(f, "      \"mean_pathlength_mm\": %.4f,\n",
                results[d].num_detected > 0
                    ? results[d].total_pathlength / results[d].total_weight
                    : 0.0);
        fprintf(f, "      \"partial_pathlength_mm\": {\n");
        for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
            double ppl = results[d].total_weight > 0
                ? results[d].partial_pathlength[t] / results[d].total_weight
                : 0.0;
            fprintf(f, "        \"%s\": %.6f%s\n",
                    tissue_names[t], ppl,
                    (t < NUM_TISSUE_TYPES - 1) ? "," : "");
        }
        fprintf(f, "      },\n");

        fprintf(f, "      \"time_gates\": [\n");
        for (int g = 0; g < NUM_TIME_GATES; g++) {
            double gw = gated_weight[d * NUM_TIME_GATES + g];
            uint64_t gc = gated_count[d * NUM_TIME_GATES + g];
            fprintf(f, "        {\n");
            fprintf(f, "          \"gate\": %d,\n", g);
            fprintf(f, "          \"range_ps\": [%.0f, %.0f],\n", gate_lo[g], gate_hi[g]);
            fprintf(f, "          \"detected_photons\": %llu,\n", (unsigned long long)gc);
            fprintf(f, "          \"weight\": %.10e,\n", gw);
            fprintf(f, "          \"partial_pathlength_mm\": {\n");
            for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
                double ppl_g = gw > 0
                    ? gated_partial_pl[(d * NUM_TIME_GATES + g) * NUM_TISSUE_TYPES + t] / gw
                    : 0.0;
                fprintf(f, "            \"%s\": %.6f%s\n",
                        tissue_names[t], ppl_g,
                        (t < NUM_TISSUE_TYPES - 1) ? "," : "");
            }
            fprintf(f, "          }\n");
            fprintf(f, "        }%s\n", (g < NUM_TIME_GATES - 1) ? "," : "");
        }
        fprintf(f, "      ]\n");
        fprintf(f, "    }%s\n", (d < n_dets - 1) ? "," : "");
    }

    fprintf(f, "  ]\n}\n");
    fclose(f);
    printf("Saved %s\n", filename);
}

// ---------------------------------------------------------------------------
// Parse wavelength list
// ---------------------------------------------------------------------------
static std::vector<float> parse_wavelengths(const char* str) {
    std::vector<float> wls;
    const char* p = str;
    while (*p) {
        char* end;
        float val = strtof(p, &end);
        if (end == p) break;
        if (val >= 650.0f && val <= 950.0f) {
            wls.push_back(val);
        } else {
            fprintf(stderr, "WARNING: wavelength %.0f nm out of range [650-950], skipping\n", val);
        }
        p = end;
        if (*p == ',') p++;
    }
    return wls;
}

// ---------------------------------------------------------------------------
// Find source position: nearest scalp surface point to amygdala centroid
// ---------------------------------------------------------------------------
static void find_source_on_scalp(const HostMesh& mesh,
                                  float& src_x, float& src_y, float& src_z,
                                  float& src_dx, float& src_dy, float& src_dz) {
    // Find RIGHT amygdala centroid (positive x hemisphere)
    // Matches voxel MC convention: right amygdala at ~(+24, -2, -18) in MNI space
    float amyg_cx = 0, amyg_cy = 0, amyg_cz = 0;
    int amyg_count = 0;
    for (int e = 0; e < mesh.num_elements; e++) {
        if (mesh.tissue[e] == TISSUE_AMYGDALA) {
            for (int v = 0; v < 4; v++) {
                int vi = mesh.elements[e * 4 + v];
                float nx = mesh.nodes[vi * 3 + 0];
                if (nx > 0.0f) {
                    amyg_cx += nx;
                    amyg_cy += mesh.nodes[vi * 3 + 1];
                    amyg_cz += mesh.nodes[vi * 3 + 2];
                    amyg_count++;
                }
            }
        }
    }

    if (amyg_count == 0) {
        fprintf(stderr, "WARNING: no amygdala elements in mesh, using MNI atlas coordinates\n");
        amyg_cx = 24.0f;
        amyg_cy = -2.0f;
        amyg_cz = -20.0f;
    } else {
        amyg_cx /= amyg_count;
        amyg_cy /= amyg_count;
        amyg_cz /= amyg_count;
    }

    printf("  Right amygdala centroid: (%.1f, %.1f, %.1f) mm [%d vertices]\n",
           amyg_cx, amyg_cy, amyg_cz, amyg_count);

    // Find the scalp surface point closest to the amygdala centroid.
    // This minimizes source-to-target distance (~40-45mm to FT8 area)
    // vs the old ray-projection method which landed at T8 (~52mm away).
    //
    // Strategy: collect all external scalp boundary face centroids, then find
    // the one with minimum Euclidean distance to the amygdala centroid.

    // Collect external scalp face centroids + normals
    struct ScalpFace { float x, y, z, nx, ny, nz; };
    std::vector<ScalpFace> ext_faces;
    static const int FV[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};

    for (int e = 0; e < mesh.num_elements; e++) {
        if (mesh.tissue[e] != TISSUE_SCALP) continue;
        for (int f = 0; f < 4; f++) {
            if (mesh.neighbors[e * 4 + f] != -1) continue;
            ScalpFace sf;
            sf.x = sf.y = sf.z = 0;
            for (int i = 0; i < 3; i++) {
                int vi = mesh.elements[e * 4 + FV[f][i]];
                sf.x += mesh.nodes[vi * 3 + 0];
                sf.y += mesh.nodes[vi * 3 + 1];
                sf.z += mesh.nodes[vi * 3 + 2];
            }
            sf.x /= 3; sf.y /= 3; sf.z /= 3;
            int va = mesh.elements[e * 4 + FV[f][0]];
            int vb = mesh.elements[e * 4 + FV[f][1]];
            int vc = mesh.elements[e * 4 + FV[f][2]];
            float e1x = mesh.nodes[vb*3+0]-mesh.nodes[va*3+0];
            float e1y = mesh.nodes[vb*3+1]-mesh.nodes[va*3+1];
            float e1z = mesh.nodes[vb*3+2]-mesh.nodes[va*3+2];
            float e2x = mesh.nodes[vc*3+0]-mesh.nodes[va*3+0];
            float e2y = mesh.nodes[vc*3+1]-mesh.nodes[va*3+1];
            float e2z = mesh.nodes[vc*3+2]-mesh.nodes[va*3+2];
            sf.nx = e1y*e2z - e1z*e2y;
            sf.ny = e1z*e2x - e1x*e2z;
            sf.nz = e1x*e2y - e1y*e2x;
            float nm = sqrtf(sf.nx*sf.nx + sf.ny*sf.ny + sf.nz*sf.nz);
            if (nm > 0) { sf.nx /= nm; sf.ny /= nm; sf.nz /= nm; }
            ext_faces.push_back(sf);
        }
    }
    printf("  Found %d external scalp boundary faces\n", (int)ext_faces.size());

    // Find scalp point ~40-50mm from amygdala on lateral surface.
    // The closest scalp point is ~60mm (inferior), so we search for points
    // in the 35-55mm range on the lateral temporal surface for better SNR.
    const float TARGET_MIN = 35.0f, TARGET_MAX = 55.0f, TARGET_OPT = 45.0f;
    
    float best_score = -1e30f;
    int best_idx = -1;
    float best_dist = 0;
    
    for (int j = 0; j < (int)ext_faces.size(); j++) {
        float dx = ext_faces[j].x - amyg_cx;
        float dy = ext_faces[j].y - amyg_cy;
        float dz = ext_faces[j].z - amyg_cz;
        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
        
        // Only consider points in target range
        if (dist < TARGET_MIN || dist > TARGET_MAX) continue;
        
        // Ensure normal points outward
        float nx = ext_faces[j].nx, ny = ext_faces[j].ny, nz = ext_faces[j].nz;
        float dot_out = nx * ext_faces[j].x + ny * ext_faces[j].y + nz * ext_faces[j].z;
        if (dot_out < 0) { nx = -nx; ny = -ny; nz = -nz; }
        
        // Direction to amygdala
        float to_amyg_x = -dx/dist, to_amyg_y = -dy/dist, to_amyg_z = -dz/dist;
        
        // Score: alignment + lateralness + height + distance_to_optimal
        float alignment = -(nx*to_amyg_x + ny*to_amyg_y + nz*to_amyg_z);
        float lateralness = (ext_faces[j].x - amyg_cx) * 0.02f;
        float height = -(ext_faces[j].z < amyg_cz - 30.0f ? 
                        (amyg_cz - 30.0f - ext_faces[j].z) * 0.02f : 0);
        float dist_score = -fabsf(dist - TARGET_OPT) * 0.05f;
        
        float score = alignment + lateralness + height + dist_score;
        if (score > best_score) {
            best_score = score; best_idx = j; best_dist = dist;
        }
    }
    
    // Fallback to closest point if no point in range
    if (best_idx < 0) {
        printf("  WARNING: no point in 35-55mm range, using closest\n");
        float best_d2 = 1e30f;
        for (int j = 0; j < (int)ext_faces.size(); j++) {
            float dx = ext_faces[j].x - amyg_cx;
            float dy = ext_faces[j].y - amyg_cy;
            float dz = ext_faces[j].z - amyg_cz;
            float d2 = dx*dx + dy*dy + dz*dz;
            if (d2 < best_d2) { best_d2 = d2; best_idx = j; best_dist = sqrtf(d2); }
        }
    }
    
    if (best_idx < 0) {
        fprintf(stderr, "ERROR: no suitable scalp point found!\n");
        src_x = amyg_cx; src_y = amyg_cy; src_z = amyg_cz;
        src_dx = 0; src_dy = 0; src_dz = 1;
        return;
    }

    float surf_x = ext_faces[best_idx].x;
    float surf_y = ext_faces[best_idx].y;
    float surf_z = ext_faces[best_idx].z;
    float surf_nx = ext_faces[best_idx].nx;
    float surf_ny = ext_faces[best_idx].ny;
    float surf_nz = ext_faces[best_idx].nz;

    // Ensure normal points outward (away from head center)
    float dot_out = surf_nx * surf_x + surf_ny * surf_y + surf_nz * surf_z;
    if (dot_out < 0) { surf_nx = -surf_nx; surf_ny = -surf_ny; surf_nz = -surf_nz; }

    printf("  Scalp surface point: (%.1f, %.1f, %.1f) mm\n", surf_x, surf_y, surf_z);
    printf("  Surface normal: (%.3f, %.3f, %.3f)\n", surf_nx, surf_ny, surf_nz);

    // Place source 0.5mm inward from the actual scalp surface
    src_x = surf_x - surf_nx * 0.5f;
    src_y = surf_y - surf_ny * 0.5f;
    src_z = surf_z - surf_nz * 0.5f;

    // Direction: inward (opposite to surface normal)
    src_dx = -surf_nx;
    src_dy = -surf_ny;
    src_dz = -surf_nz;

    printf("  Source position: (%.1f, %.1f, %.1f) mm\n", src_x, src_y, src_z);
    printf("  Source direction: (%.3f, %.3f, %.3f)\n", src_dx, src_dy, src_dz);
    printf("  Distance to amygdala: %.1f mm\n", best_dist);
}

// ---------------------------------------------------------------------------
// Snap a point to the nearest external scalp surface face centroid.
// Returns the face centroid position and outward normal.
// ---------------------------------------------------------------------------
static void snap_to_scalp_surface(
    const HostMesh& mesh,
    float in_x, float in_y, float in_z,
    float& out_x, float& out_y, float& out_z,
    float& out_nx, float& out_ny, float& out_nz)
{
    static const int FV[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
    float best_d2 = 1e30f;

    for (int e = 0; e < mesh.num_elements; e++) {
        if (mesh.tissue[e] != TISSUE_SCALP) continue;
        for (int f = 0; f < 4; f++) {
            if (mesh.neighbors[e * 4 + f] != -1) continue;
            float fx = 0, fy = 0, fz = 0;
            for (int i = 0; i < 3; i++) {
                int vi = mesh.elements[e * 4 + FV[f][i]];
                fx += mesh.nodes[vi * 3 + 0];
                fy += mesh.nodes[vi * 3 + 1];
                fz += mesh.nodes[vi * 3 + 2];
            }
            fx /= 3; fy /= 3; fz /= 3;
            float ddx = fx - in_x, ddy = fy - in_y, ddz = fz - in_z;
            float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
            if (d2 < best_d2) {
                best_d2 = d2;
                out_x = fx; out_y = fy; out_z = fz;
                int va = mesh.elements[e * 4 + FV[f][0]];
                int vb = mesh.elements[e * 4 + FV[f][1]];
                int vc = mesh.elements[e * 4 + FV[f][2]];
                float e1x = mesh.nodes[vb*3+0]-mesh.nodes[va*3+0];
                float e1y = mesh.nodes[vb*3+1]-mesh.nodes[va*3+1];
                float e1z = mesh.nodes[vb*3+2]-mesh.nodes[va*3+2];
                float e2x = mesh.nodes[vc*3+0]-mesh.nodes[va*3+0];
                float e2y = mesh.nodes[vc*3+1]-mesh.nodes[va*3+1];
                float e2z = mesh.nodes[vc*3+2]-mesh.nodes[va*3+2];
                out_nx = e1y*e2z - e1z*e2y;
                out_ny = e1z*e2x - e1x*e2z;
                out_nz = e1x*e2y - e1y*e2x;
                float nm = sqrtf(out_nx*out_nx + out_ny*out_ny + out_nz*out_nz);
                if (nm > 0) { out_nx /= nm; out_ny /= nm; out_nz /= nm; }
            }
        }
    }

    // Ensure normal points outward
    float dot_out = out_nx * out_x + out_ny * out_y + out_nz * out_z;
    if (dot_out < 0) { out_nx = -out_nx; out_ny = -out_ny; out_nz = -out_nz; }
}

// ---------------------------------------------------------------------------
// Build detectors on scalp surface for MMC mesh
// Places detectors at various source-detector separations along the scalp
// ---------------------------------------------------------------------------
static std::vector<Detector> build_mmc_detectors(
    const HostMesh& mesh,
    float src_x, float src_y, float src_z,
    float src_dx, float src_dy, float src_dz,
    std::vector<float>& out_separations,
    std::vector<float>& out_angles)
{
    // Target SDS values: short-SDS for SSR, mid-range sweet spot, extended range
    float target_sds[] = {
        8,  8,                          // SSR reference (0°, 180°)
        12, 12,                         // SSR reference (±60°)
        20, 22, 25, 28, 30, 33, 35, 40, // axial sweet spot
        20, 25, 30, 35,                 // +30° ring
        20, 25, 30, 35,                 // -30° ring
        25, 35, 25, 35,                 // ±60° ring
        45, 45, 50, 50                  // extended sweet spot (±45°)
    };
    float target_angles[] = {
        0, 180,                         // SSR reference
        60, -60,                        // SSR reference
        0, 0, 0, 0, 0, 0, 0, 0,        // axial
        30, 30, 30, 30,                 // +30°
        -30, -30, -30, -30,             // -30°
        60, 60, -60, -60,              // ±60°
        45, -45, 45, -45                // extended ±45°
    };
    int n_targets = sizeof(target_sds) / sizeof(target_sds[0]);

    // Build a tangent frame on the scalp at the source
    float tx, ty, tz, bx, by, bz;
    if (fabsf(src_dz) < 0.9f) {
        tx = src_dy; ty = -src_dx; tz = 0.0f;
    } else {
        tx = 0.0f; ty = src_dz; tz = -src_dy;
    }
    float tmag = sqrtf(tx*tx + ty*ty + tz*tz);
    tx /= tmag; ty /= tmag; tz /= tmag;
    bx = src_dy*tz - src_dz*ty;
    by = src_dz*tx - src_dx*tz;
    bz = src_dx*ty - src_dy*tx;

    // Collect all boundary scalp face centroids for nearest-surface-point queries
    struct ScalpPoint { float x, y, z, nx, ny, nz; };
    std::vector<ScalpPoint> scalp_pts;

    for (int e = 0; e < mesh.num_elements; e++) {
        if (mesh.tissue[e] != TISSUE_SCALP) continue;
        for (int f = 0; f < 4; f++) {
            if (mesh.neighbors[e * 4 + f] != -1) continue;
            static const int FV[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
            ScalpPoint sp;
            sp.x = sp.y = sp.z = 0;
            for (int i = 0; i < 3; i++) {
                int vi = mesh.elements[e * 4 + FV[f][i]];
                sp.x += mesh.nodes[vi * 3 + 0];
                sp.y += mesh.nodes[vi * 3 + 1];
                sp.z += mesh.nodes[vi * 3 + 2];
            }
            sp.x /= 3; sp.y /= 3; sp.z /= 3;

            int va = mesh.elements[e * 4 + FV[f][0]];
            int vb = mesh.elements[e * 4 + FV[f][1]];
            int vc = mesh.elements[e * 4 + FV[f][2]];
            float e1x = mesh.nodes[vb*3+0]-mesh.nodes[va*3+0];
            float e1y = mesh.nodes[vb*3+1]-mesh.nodes[va*3+1];
            float e1z = mesh.nodes[vb*3+2]-mesh.nodes[va*3+2];
            float e2x = mesh.nodes[vc*3+0]-mesh.nodes[va*3+0];
            float e2y = mesh.nodes[vc*3+1]-mesh.nodes[va*3+1];
            float e2z = mesh.nodes[vc*3+2]-mesh.nodes[va*3+2];
            sp.nx = e1y*e2z - e1z*e2y;
            sp.ny = e1z*e2x - e1x*e2z;
            sp.nz = e1x*e2y - e1y*e2x;
            float nm = sqrtf(sp.nx*sp.nx + sp.ny*sp.ny + sp.nz*sp.nz);
            if (nm > 0) { sp.nx /= nm; sp.ny /= nm; sp.nz /= nm; }
            scalp_pts.push_back(sp);
        }
    }

    std::vector<Detector> dets;
    out_separations.clear();
    out_angles.clear();

    float det_radius = 1.69f; // 3x3mm SiPM

    for (int i = 0; i < n_targets; i++) {
        float sds = target_sds[i];
        float angle_deg = target_angles[i];
        float angle_rad = angle_deg * 3.14159265f / 180.0f;

        // Direction on scalp surface = cos(angle)*tangent + sin(angle)*bitangent
        float dx_scalp = cosf(angle_rad) * tx + sinf(angle_rad) * bx;
        float dy_scalp = cosf(angle_rad) * ty + sinf(angle_rad) * by;
        float dz_scalp = cosf(angle_rad) * tz + sinf(angle_rad) * bz;

        // Iterative SDS placement: step along tangent, snap to surface, correct.
        // For short SDS (<=15mm), tangent stepping overshoots on curved scalp,
        // so we use binary search. For long SDS, multiplicative correction works.
        int best_idx = -1;
        float actual_sds = 0;

        auto snap_to_scalp = [&](float sc, int& out_idx, float& out_sds) {
            float target_x = src_x + dx_scalp * sc;
            float target_y = src_y + dy_scalp * sc;
            float target_z = src_z + dz_scalp * sc;
            float best_d2 = 1e30f;
            out_idx = -1;
            for (int j = 0; j < (int)scalp_pts.size(); j++) {
                float ddx = scalp_pts[j].x - target_x;
                float ddy = scalp_pts[j].y - target_y;
                float ddz = scalp_pts[j].z - target_z;
                float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
                if (d2 < best_d2) { best_d2 = d2; out_idx = j; }
            }
            if (out_idx >= 0) {
                out_sds = sqrtf(
                    (scalp_pts[out_idx].x-src_x)*(scalp_pts[out_idx].x-src_x) +
                    (scalp_pts[out_idx].y-src_y)*(scalp_pts[out_idx].y-src_y) +
                    (scalp_pts[out_idx].z-src_z)*(scalp_pts[out_idx].z-src_z));
            }
        };

        if (sds <= 15.0f) {
            // Binary search for short SDS to avoid overshoot divergence
            float lo = 0.0f, hi = sds;
            // First expand hi until actual_sds >= sds
            snap_to_scalp(hi, best_idx, actual_sds);
            while (best_idx >= 0 && actual_sds < sds && hi < sds * 4.0f) {
                hi *= 1.5f;
                snap_to_scalp(hi, best_idx, actual_sds);
            }
            for (int iter = 0; iter < 20; iter++) {
                float mid = (lo + hi) * 0.5f;
                snap_to_scalp(mid, best_idx, actual_sds);
                if (best_idx < 0) break;
                if (actual_sds < sds) lo = mid;
                else hi = mid;
                if (fabsf(actual_sds - sds) < 0.5f) break;
            }
        } else {
            // Multiplicative correction for long SDS (converges well)
            float scale = sds;
            for (int iter = 0; iter < 15; iter++) {
                snap_to_scalp(scale, best_idx, actual_sds);
                if (best_idx < 0) break;
                if (fabsf(actual_sds - sds) < 0.5f) break;
                if (actual_sds > 0.1f) scale *= sds / actual_sds;
            }
        }

        if (best_idx < 0) continue;

        Detector det;
        det.x = scalp_pts[best_idx].x;
        det.y = scalp_pts[best_idx].y;
        det.z = scalp_pts[best_idx].z;
        det.radius = det_radius;
        det.id = (int)dets.size();
        det.nx = scalp_pts[best_idx].nx;
        det.ny = scalp_pts[best_idx].ny;
        det.nz = scalp_pts[best_idx].nz;

        // Ensure normal points outward (away from head center)
        float dot_out = det.nx*det.x + det.ny*det.y + det.nz*det.z;
        if (dot_out < 0) {
            det.nx = -det.nx; det.ny = -det.ny; det.nz = -det.nz;
        }

        det.n_critical = 1.0f / 1.37f; // n_air / n_scalp

        dets.push_back(det);
        out_separations.push_back(actual_sds);
        out_angles.push_back(angle_deg);

        printf("    Det %2d: target=%5.0fmm actual=%5.1fmm angle=%+4.0f pos=(%.1f,%.1f,%.1f)\n",
               det.id, sds, actual_sds, angle_deg, det.x, det.y, det.z);
    }

    printf("  Built %d detectors on scalp surface\n", (int)dets.size());
    return dets;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("=============================================================\n");
    printf("  MMC fNIRS - Mesh-Based Monte Carlo with MNI152 Head Model\n");
    printf("  Tetrahedral transport | Moller-Trumbore intersection\n");
    printf("  Mie scattering model | Fresnel/Snell boundary physics\n");
    printf("=============================================================\n\n");

    // Parse arguments
    uint64_t num_photons = 100000000ULL;
    std::string mesh_path = "mni152_head.mmcmesh";
    std::string output_dir = "data_mmc";
    std::vector<float> wavelengths;
    int source_index = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--photons") == 0 && i + 1 < argc) {
            num_photons = strtoull(argv[i + 1], nullptr, 10);
            i++;
        } else if (strcmp(argv[i], "--mesh") == 0 && i + 1 < argc) {
            mesh_path = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[i + 1];
            i++;
        } else if (strcmp(argv[i], "--wavelengths") == 0 && i + 1 < argc) {
            wavelengths = parse_wavelengths(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "--source-index") == 0 && i + 1 < argc) {
            source_index = atoi(argv[i + 1]);
            if (source_index < 0 || source_index > 3) {
                fprintf(stderr, "ERROR: --source-index must be 0-3\n");
                return 1;
            }
            i++;
        }
    }

    if (wavelengths.empty()) {
        wavelengths.push_back(730.0f);
        wavelengths.push_back(850.0f);
    }

    const char* source_names[] = {
        "FT8 area (optimal)", "20mm anterior", "20mm posterior", "15mm superior"
    };
    printf("Configuration:\n");
    printf("  Mesh: %s\n", mesh_path.c_str());
    printf("  Photons per wavelength: %llu\n", (unsigned long long)num_photons);
    printf("  Output directory: %s\n", output_dir.c_str());
    printf("  Source index: %d (%s)\n", source_index, source_names[source_index]);
    printf("  Wavelengths:");
    for (float wl : wavelengths) printf(" %.0fnm", wl);
    printf("\n\n");

    // CUDA device info
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    printf("CUDA Device: %s\n", devProp.name);
    printf("  Compute: %d.%d  SMs: %d  Memory: %.0f MB\n\n",
           devProp.major, devProp.minor,
           devProp.multiProcessorCount,
           devProp.totalGlobalMem / (1024.0 * 1024.0));

    // Load mesh
    printf("--- Loading mesh ---\n");
    HostMesh mesh = load_mmcmesh(mesh_path.c_str());

    // Precompute face geometry
    printf("\n--- Precomputing face geometry ---\n");
    std::vector<float> face_normals, face_d;
    precompute_face_geometry(mesh, face_normals, face_d);
    
    // Precompute face pair lookup table (Fix B)
    std::vector<int> face_pair;
    precompute_face_pair(mesh, face_pair);

    // Build grid accelerator
    printf("\n--- Building spatial accelerator ---\n");
    std::vector<int> grid_offsets, grid_counts, grid_tets;
    float cell_size[3];
    build_grid_accelerator(mesh, grid_offsets, grid_counts, grid_tets, cell_size);

    // Upload to GPU
    printf("\n--- Uploading to GPU ---\n");
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("  GPU memory: %.1f MB free / %.1f MB total\n",
           free_mem / (1024.0*1024.0), total_mem / (1024.0*1024.0));

    MMCDeviceData dev_data = upload_mesh_to_gpu(
        mesh, face_normals, face_d, face_pair,
        grid_offsets, grid_counts, grid_tets, cell_size);

    // Find source position on scalp
    printf("\n--- Source placement (index %d: %s) ---\n",
           source_index, source_names[source_index]);
    float src_x, src_y, src_z, src_dx, src_dy, src_dz;
    find_source_on_scalp(mesh, src_x, src_y, src_z, src_dx, src_dy, src_dz);

    // For source indices 1-3, offset from the base optimal source and snap to scalp.
    // MNI approximate directions at the FT8 area (~right temporal):
    //   anterior = roughly +Y direction
    //   posterior = roughly -Y direction
    //   superior = roughly +Z direction
    // We offset from the scalp surface point, then snap back to the nearest
    // scalp face.
    if (source_index > 0) {
        // Get the base scalp surface point (before 0.5mm inset)
        float base_x = src_x + (-src_dx) * 0.5f;
        float base_y = src_y + (-src_dy) * 0.5f;
        float base_z = src_z + (-src_dz) * 0.5f;

        float offset_x = 0, offset_y = 0, offset_z = 0;
        if (source_index == 1) { offset_y = 20.0f; }        // anterior (+Y in MNI)
        else if (source_index == 2) { offset_y = -20.0f; }   // posterior (-Y in MNI)
        else if (source_index == 3) { offset_z = 15.0f; }    // superior (+Z in MNI)

        float target_x = base_x + offset_x;
        float target_y = base_y + offset_y;
        float target_z = base_z + offset_z;

        float snap_x, snap_y, snap_z, snap_nx, snap_ny, snap_nz;
        snap_to_scalp_surface(mesh, target_x, target_y, target_z,
                              snap_x, snap_y, snap_z, snap_nx, snap_ny, snap_nz);

        // Place source 0.5mm inward from scalp surface
        src_x = snap_x - snap_nx * 0.5f;
        src_y = snap_y - snap_ny * 0.5f;
        src_z = snap_z - snap_nz * 0.5f;
        src_dx = -snap_nx;
        src_dy = -snap_ny;
        src_dz = -snap_nz;

        float dist_base = sqrtf((snap_x-base_x)*(snap_x-base_x) +
                                (snap_y-base_y)*(snap_y-base_y) +
                                (snap_z-base_z)*(snap_z-base_z));
        printf("  Offset source %d: (%.1f, %.1f, %.1f) mm, %.1fmm from base\n",
               source_index, src_x, src_y, src_z, dist_base);
        printf("  Source direction: (%.3f, %.3f, %.3f)\n", src_dx, src_dy, src_dz);
    }

    // Build detectors
    printf("\n--- Detector placement ---\n");
    std::vector<float> separations, angles;
    std::vector<Detector> detectors = build_mmc_detectors(
        mesh, src_x, src_y, src_z, src_dx, src_dy, src_dz,
        separations, angles);
    int n_dets = (int)detectors.size();

    // Create output directory
    {
        std::error_code ec;
        fs::create_directories(output_dir, ec);
        if (ec) {
            fprintf(stderr, "Warning: could not create directory %s: %s\n",
                    output_dir.c_str(), ec.message().c_str());
        }
    }

    // Run simulation for each wavelength
    for (int wl = 0; wl < (int)wavelengths.size(); wl++) {
        float wavelength_nm = wavelengths[wl];
        char wl_tag[32];
        snprintf(wl_tag, sizeof(wl_tag), "%.0fnm", wavelength_nm);

        printf("\n=========================================\n");
        printf("  Wavelength: %s\n", wl_tag);
        printf("=========================================\n");

        // Configure
        MMCConfig config{};
        config.num_photons = num_photons;
        config.src_x = src_x;
        config.src_y = src_y;
        config.src_z = src_z;
        config.src_dx = src_dx;
        config.src_dy = src_dy;
        config.src_dz = src_dz;
        config.beam_radius = 7.5f;
        config.wavelength_idx = wl;
        config.weight_threshold = 1e-4f;
        config.roulette_m = 10;

        compute_optical_properties(wavelength_nm, config.tissue);
        print_optical_properties(wavelength_nm, config.tissue);

        // Allocate result buffers
        std::vector<DetectorResult> results(n_dets);
        memset(results.data(), 0, n_dets * sizeof(DetectorResult));

        std::vector<double> tpsf((size_t)n_dets * TPSF_BINS, 0.0);
        std::vector<double> gated_weight((size_t)n_dets * NUM_TIME_GATES, 0.0);
        std::vector<double> gated_partial_pl(
            (size_t)n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES, 0.0);
        std::vector<uint64_t> gated_count((size_t)n_dets * NUM_TIME_GATES, 0);

        size_t path_pos_count = (size_t)MAX_RECORDED_PATHS * MAX_PATH_STEPS * 3;
        std::vector<float> path_pos(path_pos_count, 0.0f);
        std::vector<int> path_det(MAX_RECORDED_PATHS, 0);
        std::vector<int> path_len(MAX_RECORDED_PATHS, 0);
        int num_paths = 0;

        // Launch
        auto t0 = std::chrono::high_resolution_clock::now();

        launch_mmc_simulation(
            dev_data, config,
            detectors.data(), n_dets,
            results.data(),
            tpsf.data(), gated_weight.data(),
            gated_partial_pl.data(), gated_count.data(),
            path_pos.data(), path_det.data(),
            path_len.data(), &num_paths
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        printf("\nSimulation completed in %.2f seconds\n", elapsed);
        printf("Throughput: %.2f M photons/sec\n\n", num_photons / elapsed / 1e6);

        // Print results
        printf("Detector Results (CW):\n");
        printf("%-5s %-6s %-7s %-13s %-13s %-10s %-10s\n",
               "Det", "SDS", "Angle", "Detected", "Weight", "MeanPL", "AmygPL");
        for (int d = 0; d < n_dets; d++) {
            double mean_pl = results[d].total_weight > 0
                ? results[d].total_pathlength / results[d].total_weight : 0;
            double amyg_pl = results[d].total_weight > 0
                ? results[d].partial_pathlength[TISSUE_AMYGDALA] / results[d].total_weight : 0;

            printf("%-5d %-6.0f %+5.0f   %-13llu %-13.4e %-10.2f %-10.4f\n",
                   d, separations[d], angles[d],
                   (unsigned long long)results[d].num_detected,
                   results[d].total_weight,
                   mean_pl, amyg_pl);
        }

        // Save outputs
        char fname[256];

        snprintf(fname, sizeof(fname), "%s/tpsf_%s.bin",
                 output_dir.c_str(), wl_tag);
        save_binary(fname, tpsf.data(), (size_t)n_dets * TPSF_BINS * sizeof(double));

        printf("Recorded photon paths: %d\n", num_paths);
        if (num_paths > 0) {
            snprintf(fname, sizeof(fname), "%s/paths_meta_%s.bin",
                     output_dir.c_str(), wl_tag);
            std::vector<int> meta(num_paths * 2);
            for (int i = 0; i < num_paths; i++) {
                meta[i * 2 + 0] = path_det[i];
                meta[i * 2 + 1] = path_len[i];
            }
            save_binary(fname, meta.data(), num_paths * 2 * sizeof(int));

            snprintf(fname, sizeof(fname), "%s/paths_pos_%s.bin",
                     output_dir.c_str(), wl_tag);
            save_binary(fname, path_pos.data(),
                        (size_t)num_paths * MAX_PATH_STEPS * 3 * sizeof(float));
        }

        snprintf(fname, sizeof(fname), "%s/results_%s.json",
                 output_dir.c_str(), wl_tag);
        save_results_json(fname, results.data(), n_dets,
                         separations, angles,
                         wavelength_nm, num_photons,
                         gated_weight.data(), gated_partial_pl.data(),
                         gated_count.data());
    }

    // Save mesh metadata
    char fname[256];
    snprintf(fname, sizeof(fname), "%s/mesh_meta.json", output_dir.c_str());
    FILE* f = fopen(fname, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"geometry_model\": \"mmc_mni152\",\n");
        fprintf(f, "  \"mesh_file\": \"%s\",\n", mesh_path.c_str());
        fprintf(f, "  \"num_nodes\": %d,\n", mesh.num_nodes);
        fprintf(f, "  \"num_elements\": %d,\n", mesh.num_elements);
        fprintf(f, "  \"bbox_min\": [%.2f, %.2f, %.2f],\n",
                mesh.bbox_min[0], mesh.bbox_min[1], mesh.bbox_min[2]);
        fprintf(f, "  \"bbox_max\": [%.2f, %.2f, %.2f],\n",
                mesh.bbox_max[0], mesh.bbox_max[1], mesh.bbox_max[2]);
        fprintf(f, "  \"scattering_model\": \"mie_power_law\",\n");
        fprintf(f, "  \"detector_radius_mm\": 1.69,\n");
        fprintf(f, "  \"source_index\": %d,\n", source_index);
        fprintf(f, "  \"source_label\": \"%s\",\n", source_names[source_index]);
        fprintf(f, "  \"source_position_mm\": [%.2f, %.2f, %.2f],\n",
                src_x, src_y, src_z);
        fprintf(f, "  \"wavelengths_nm\": [");
        for (int i = 0; i < (int)wavelengths.size(); i++) {
            if (i > 0) fprintf(f, ", ");
            fprintf(f, "%.0f", wavelengths[i]);
        }
        fprintf(f, "],\n");
        fprintf(f, "  \"tissue_labels\": {\n");
        fprintf(f, "    \"0\": \"air\", \"1\": \"scalp\", \"2\": \"skull\",\n");
        fprintf(f, "    \"3\": \"csf\", \"4\": \"gray_matter\",\n");
        fprintf(f, "    \"5\": \"white_matter\", \"6\": \"amygdala\"\n");
        fprintf(f, "  }\n}\n");
        fclose(f);
    }

    free_mesh_gpu(dev_data);

    printf("\n=== MMC simulation complete. Outputs in %s/ ===\n", output_dir.c_str());
    return 0;
}
