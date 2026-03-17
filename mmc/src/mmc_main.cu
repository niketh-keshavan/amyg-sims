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
// Find source position: project from amygdala center to nearest scalp node
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

    // Pure lateral projection along +x from right amygdala onto temporal scalp.
    // The temporal bone directly overlying the amygdala is at the same y,z
    // but further lateral — this is the standard fNIRS optode placement.
    // Previous approaches (radial, ellipsoidal) all landed on the inferior
    // surface because the amygdala z=-20 is below any head center estimate.
    float proj_dx = 1.0f;
    float proj_dy = 0.0f;
    float proj_dz = 0.0f;

    printf("  Projection: pure lateral +x from (%.1f, %.1f, %.1f)\n",
           amyg_cx, amyg_cy, amyg_cz);

    // Find scalp boundary face closest to the lateral ray from amygdala.
    // Constrain to faces with z > amygdala_z - 20mm to prevent inferior surface.
    static const int FV[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};
    float z_min_constraint = amyg_cz - 20.0f;
    float best_score = 1e30f;
    float best_x = 0, best_y = 0, best_z = 0;
    float best_nx = 0, best_ny = 0, best_nz = 0;

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
            fx /= 3.0f; fy /= 3.0f; fz /= 3.0f;

            if (fz < z_min_constraint) continue;

            float vx = fx - amyg_cx;
            float vy = fy - amyg_cy;
            float vz = fz - amyg_cz;

            float t = vx * proj_dx + vy * proj_dy + vz * proj_dz;
            if (t <= 0.0f) continue;

            float perp_x = vx - t * proj_dx;
            float perp_y = vy - t * proj_dy;
            float perp_z = vz - t * proj_dz;
            float perp_dist2 = perp_x*perp_x + perp_y*perp_y + perp_z*perp_z;

            if (perp_dist2 < best_score) {
                best_score = perp_dist2;
                best_x = fx; best_y = fy; best_z = fz;

                int va = mesh.elements[e * 4 + FV[f][0]];
                int vb = mesh.elements[e * 4 + FV[f][1]];
                int vc = mesh.elements[e * 4 + FV[f][2]];
                float e1x = mesh.nodes[vb*3+0]-mesh.nodes[va*3+0];
                float e1y = mesh.nodes[vb*3+1]-mesh.nodes[va*3+1];
                float e1z = mesh.nodes[vb*3+2]-mesh.nodes[va*3+2];
                float e2x = mesh.nodes[vc*3+0]-mesh.nodes[va*3+0];
                float e2y = mesh.nodes[vc*3+1]-mesh.nodes[va*3+1];
                float e2z = mesh.nodes[vc*3+2]-mesh.nodes[va*3+2];
                best_nx = e1y*e2z - e1z*e2y;
                best_ny = e1z*e2x - e1x*e2z;
                best_nz = e1x*e2y - e1y*e2x;
            }
        }
    }

    // Normalize face normal
    float nmag = sqrtf(best_nx*best_nx + best_ny*best_ny + best_nz*best_nz);
    if (nmag > 0) { best_nx /= nmag; best_ny /= nmag; best_nz /= nmag; }

    // Ensure normal points outward (away from amygdala)
    float to_amyg_x = amyg_cx - best_x;
    float to_amyg_y = amyg_cy - best_y;
    float to_amyg_z = amyg_cz - best_z;
    if (best_nx*to_amyg_x + best_ny*to_amyg_y + best_nz*to_amyg_z > 0) {
        best_nx = -best_nx; best_ny = -best_ny; best_nz = -best_nz;
    }

    // Place source just inside the scalp (0.5mm inward along -outward_normal)
    src_x = best_x - best_nx * 0.5f;
    src_y = best_y - best_ny * 0.5f;
    src_z = best_z - best_nz * 0.5f;

    // Direction: inward (toward amygdala)
    src_dx = -best_nx;
    src_dy = -best_ny;
    src_dz = -best_nz;

    float dist_to_amyg = sqrtf((best_x-amyg_cx)*(best_x-amyg_cx) +
                               (best_y-amyg_cy)*(best_y-amyg_cy) +
                               (best_z-amyg_cz)*(best_z-amyg_cz));
    printf("  Source position: (%.1f, %.1f, %.1f) mm\n", src_x, src_y, src_z);
    printf("  Source direction: (%.3f, %.3f, %.3f)\n", src_dx, src_dy, src_dz);
    printf("  Distance to amygdala: %.1f mm\n", dist_to_amyg);
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
    // Target SDS values matching voxel MC detector layout (src/detector.cu:62-100)
    float target_sds[] = { 8, 8, 15, 20, 22, 25, 28, 30, 33, 35, 40,
                           20, 25, 30, 35, 20, 25, 30, 35, 25, 35, 25, 35 };
    float target_angles[] = { 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                              30, 30, 30, 30, -30, -30, -30, -30, 60, 60, -60, -60 };
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

        // Target point on scalp at given SDS and angle
        // Direction on scalp surface = cos(angle)*tangent + sin(angle)*bitangent
        float dx_scalp = cosf(angle_rad) * tx + sinf(angle_rad) * bx;
        float dy_scalp = cosf(angle_rad) * ty + sinf(angle_rad) * by;
        float dz_scalp = cosf(angle_rad) * tz + sinf(angle_rad) * bz;

        float target_x = src_x + dx_scalp * sds;
        float target_y = src_y + dy_scalp * sds;
        float target_z = src_z + dz_scalp * sds;

        // Find nearest scalp surface point
        float best_d2 = 1e30f;
        int best_idx = -1;
        for (int j = 0; j < (int)scalp_pts.size(); j++) {
            float ddx = scalp_pts[j].x - target_x;
            float ddy = scalp_pts[j].y - target_y;
            float ddz = scalp_pts[j].z - target_z;
            float d2 = ddx*ddx + ddy*ddy + ddz*ddz;
            if (d2 < best_d2) { best_d2 = d2; best_idx = j; }
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

        // Ensure normal points outward (away from mesh interior)
        float to_src_x = src_x - det.x;
        float to_src_y = src_y - det.y;
        float to_src_z = src_z - det.z;
        float dot_src = det.nx*to_src_x + det.ny*to_src_y + det.nz*to_src_z;
        // source is inside, so if dot > 0, normal points inward → flip
        if (dot_src > 0) {
            det.nx = -det.nx; det.ny = -det.ny; det.nz = -det.nz;
        }

        det.n_critical = 1.0f / 1.37f; // n_air / n_scalp

        dets.push_back(det);

        // Actual SDS on surface
        float actual_sds = sqrtf(
            (det.x-src_x)*(det.x-src_x) +
            (det.y-src_y)*(det.y-src_y) +
            (det.z-src_z)*(det.z-src_z));
        out_separations.push_back(actual_sds);
        out_angles.push_back(angle_deg);
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
        }
    }

    if (wavelengths.empty()) {
        wavelengths.push_back(730.0f);
        wavelengths.push_back(850.0f);
    }

    printf("Configuration:\n");
    printf("  Mesh: %s\n", mesh_path.c_str());
    printf("  Photons per wavelength: %llu\n", (unsigned long long)num_photons);
    printf("  Output directory: %s\n", output_dir.c_str());
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
        mesh, face_normals, face_d,
        grid_offsets, grid_counts, grid_tets, cell_size);

    // Find source position on scalp
    printf("\n--- Source placement ---\n");
    float src_x, src_y, src_z, src_dx, src_dy, src_dz;
    find_source_on_scalp(mesh, src_x, src_y, src_z, src_dx, src_dy, src_dz);

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
