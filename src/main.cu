#include "types.cuh"
#include "geometry.cuh"
#include "detector.cuh"
#include "mc_kernel.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

// ---------------------------------------------------------------------------
// Save binary outputs for Python post-processing
// ---------------------------------------------------------------------------
static void save_binary(const char* filename, const void* data, size_t bytes) {
    FILE* f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", filename); return; }
    fwrite(data, 1, bytes, f);
    fclose(f);
    printf("Saved %s (%zu bytes)\n", filename, bytes);
}

static void save_results_json(
    const char* filename,
    const DetectorResult* results,
    int n_dets,
    const std::vector<float>& separations,
    const std::vector<float>& angles,
    int wavelength_idx,
    const SimConfig& config,
    const double* gated_weight,
    const double* gated_partial_pl,
    const uint64_t* gated_count
) {
    FILE* f = fopen(filename, "w");
    if (!f) return;

    const char* wl_str = (wavelength_idx == 0) ? "760" : "850";

    fprintf(f, "{\n");
    fprintf(f, "  \"wavelength_nm\": %s,\n", wl_str);
    fprintf(f, "  \"num_photons\": %llu,\n", (unsigned long long)config.num_photons);
    fprintf(f, "  \"voxel_size_mm\": %.2f,\n", config.dx);
    fprintf(f, "  \"grid_size\": [%d, %d, %d],\n", config.nx, config.ny, config.nz);
    fprintf(f, "  \"laser_power_W\": 0.1,\n");
    fprintf(f, "  \"tpsf_bins\": %d,\n", TPSF_BINS);
    fprintf(f, "  \"tpsf_bin_ps\": %.1f,\n", (double)TPSF_BIN_PS);
    fprintf(f, "  \"time_gate_edges_ps\": [0, 500, 1000, 1500, 2500, 4000],\n");
    fprintf(f, "  \"detectors\": [\n");

    const char* tissue_names[] = {
        "air", "scalp", "skull", "csf", "gray_matter", "white_matter", "amygdala"
    };
    const float gate_lo[] = {0, 500, 1000, 1500, 2500, 4000};
    const float gate_hi[] = {500, 1000, 1500, 2500, 4000, 100000};

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

        // Time-gated data
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
// Main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("=============================================================\n");
    printf("  fNIRS Monte Carlo - TD/FD Amygdala Oxygenation\n");
    printf("  Wavelengths: 760 nm & 850 nm | 100 mW pulsed laser\n");
    printf("  0.1 mm voxels | Photon path recording\n");
    printf("  High-density array | TPSF + Time-gated outputs\n");
    printf("=============================================================\n\n");

    // --- Parse arguments ---
    uint64_t num_photons = 100000000ULL;  // 100M default
    std::string output_dir = "data";

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--photons") == 0 && i + 1 < argc) {
            num_photons = strtoull(argv[i + 1], nullptr, 10);
            i++;
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[i + 1];
            i++;
        }
    }

    printf("Configuration:\n");
    printf("  Photons per wavelength: %llu\n", (unsigned long long)num_photons);
    printf("  Output directory: %s\n", output_dir.c_str());
    printf("  Laser power: 100 mW (pulsed, eye/skin-safe)\n");
    printf("  Voxel size: 0.1 mm\n");
    printf("  TPSF bins: %d x %.0f ps = %.1f ns\n\n",
           TPSF_BINS, (double)TPSF_BIN_PS,
           TPSF_BINS * (double)TPSF_BIN_PS / 1000.0);

    // --- CUDA device info ---
    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, dev);
    printf("CUDA Device: %s\n", devProp.name);
    printf("  Compute capability: %d.%d\n", devProp.major, devProp.minor);
    printf("  SMs: %d, Global mem: %.0f MB\n\n",
           devProp.multiProcessorCount,
           devProp.totalGlobalMem / (1024.0 * 1024.0));

    cudaError_t initErr = cudaGetLastError();
    if (initErr != cudaSuccess) {
        fprintf(stderr, "CUDA init error: %s\n", cudaGetErrorString(initErr));
        return 1;
    }

    // --- Build head model ---
    HeadModelParams hm = default_head_model();
    std::vector<uint8_t> volume = build_head_volume(hm);

    // --- Build detectors ---
    DetectorLayout det_layout = default_detector_layout();
    std::vector<Detector> detectors = build_detectors(det_layout);
    int n_dets = (int)detectors.size();

    // --- Allocate fluence volume on GPU ---
    size_t vol_size = (size_t)hm.nx * hm.ny * hm.nz;
    float* d_fluence;
    cudaMalloc(&d_fluence, vol_size * sizeof(float));

    // --- Run simulation for each wavelength ---
    const char* wl_names[] = {"760nm", "850nm"};

    for (int wl = 0; wl < 2; wl++) {
        printf("\n=========================================\n");
        printf("  Wavelength: %s\n", wl_names[wl]);
        printf("=========================================\n");

        // Configure simulation
        SimConfig config{};
        config.nx = hm.nx;
        config.ny = hm.ny;
        config.nz = hm.nz;
        config.dx = hm.dx;
        config.num_photons = num_photons;

        float center_x = hm.nx * hm.dx * 0.5f;
        float center_y = hm.ny * hm.dx * 0.5f;
        float center_z = hm.nz * hm.dx * 0.5f;

        // Project source onto scalp ellipsoid (0.98 = just inside surface)
        float src_cx = det_layout.src_x;
        float src_cy = det_layout.src_y;
        float src_cz = det_layout.src_z;
        float e_src = (src_cx / hm.scalp_a) * (src_cx / hm.scalp_a)
                    + (src_cy / hm.scalp_b) * (src_cy / hm.scalp_b)
                    + (src_cz / hm.scalp_c) * (src_cz / hm.scalp_c);
        float src_scale = 0.98f / sqrtf(e_src);
        src_cx *= src_scale;
        src_cy *= src_scale;
        src_cz *= src_scale;

        config.src_x = src_cx + center_x;
        config.src_y = src_cy + center_y;
        config.src_z = src_cz + center_z;

        // Verify source is inside the scalp
        float e_verify = (src_cx / hm.scalp_a) * (src_cx / hm.scalp_a)
                       + (src_cy / hm.scalp_b) * (src_cy / hm.scalp_b)
                       + (src_cz / hm.scalp_c) * (src_cz / hm.scalp_c);
        int src_ix = (int)(config.src_x / hm.dx);
        int src_iy = (int)(config.src_y / hm.dx);
        int src_iz = (int)(config.src_z / hm.dx);
        int src_tissue = -1;
        if (src_ix >= 0 && src_ix < hm.nx &&
            src_iy >= 0 && src_iy < hm.ny &&
            src_iz >= 0 && src_iz < hm.nz) {
            src_tissue = volume[src_ix + src_iy * hm.nx
                                + src_iz * hm.nx * hm.ny];
        }
        const char* tlbl[] = {"AIR","SCALP","SKULL","CSF","GRAY","WHITE","AMYGDALA"};
        printf("  Source (centered): (%.2f, %.2f, %.2f) mm  [ellipsoid_r=%.4f]\n",
               src_cx, src_cy, src_cz, e_verify);
        printf("  Source (voxel):    (%.2f, %.2f, %.2f) mm  -> tissue=%s\n",
               config.src_x, config.src_y, config.src_z,
               (src_tissue >= 0 && src_tissue < 7) ? tlbl[src_tissue] : "OUT_OF_BOUNDS");

        // Source direction: inward (toward volume center)
        float sx = center_x - config.src_x;
        float sy = center_y - config.src_y;
        float sz = center_z - config.src_z;
        float smag = sqrtf(sx*sx + sy*sy + sz*sz);
        config.src_dx = sx / smag;
        config.src_dy = sy / smag;
        config.src_dz = sz / smag;

        config.wavelength_idx = wl;
        get_optical_properties(wl, config.tissue);

        config.weight_threshold = 1e-4f;
        config.roulette_m = 10;

        // CW results
        std::vector<DetectorResult> results(n_dets);
        memset(results.data(), 0, n_dets * sizeof(DetectorResult));

        // TD results
        std::vector<double> tpsf((size_t)n_dets * TPSF_BINS, 0.0);
        std::vector<double> gated_weight((size_t)n_dets * NUM_TIME_GATES, 0.0);
        std::vector<double> gated_partial_pl(
            (size_t)n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES, 0.0);
        std::vector<uint64_t> gated_count((size_t)n_dets * NUM_TIME_GATES, 0);

        // Path recording buffers
        size_t path_pos_count = (size_t)MAX_RECORDED_PATHS * MAX_PATH_STEPS * 3;
        std::vector<float> path_pos(path_pos_count, 0.0f);
        std::vector<int> path_det(MAX_RECORDED_PATHS, 0);
        std::vector<int> path_len(MAX_RECORDED_PATHS, 0);
        int num_paths = 0;

        // Launch
        auto t0 = std::chrono::high_resolution_clock::now();

        launch_mc_simulation(
            volume.data(), config,
            detectors.data(), n_dets,
            results.data(), d_fluence,
            tpsf.data(), gated_weight.data(),
            gated_partial_pl.data(), gated_count.data(),
            path_pos.data(), path_det.data(),
            path_len.data(), &num_paths
        );

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();

        printf("\nSimulation completed in %.2f seconds\n", elapsed);
        printf("Throughput: %.2f M photons/sec\n\n",
               num_photons / elapsed / 1e6);

        // --- Print results ---
        printf("Detector Results (CW):\n");
        printf("%-5s %-6s %-7s %-13s %-13s %-10s %-10s\n",
               "Det", "SDS", "Angle", "Detected", "Weight", "MeanPL", "AmygPL");
        for (int d = 0; d < n_dets; d++) {
            double mean_pl = results[d].total_weight > 0
                ? results[d].total_pathlength / results[d].total_weight : 0;
            double amyg_pl = results[d].total_weight > 0
                ? results[d].partial_pathlength[TISSUE_AMYGDALA] / results[d].total_weight : 0;
            float angle = d < (int)det_layout.angles_deg.size()
                ? det_layout.angles_deg[d] : 0.0f;

            printf("%-5d %-6.0f %+5.0f   %-13llu %-13.4e %-10.2f %-10.4f\n",
                   d, det_layout.separations_mm[d], angle,
                   (unsigned long long)results[d].num_detected,
                   results[d].total_weight,
                   mean_pl, amyg_pl);
        }

        // --- Save outputs ---
        char fname[256];

        // Fluence: only save for small grids (skip for 0.1mm = 1000^3)
        if (vol_size <= 64000000ULL) {
            std::vector<float> h_fluence(vol_size);
            cudaMemcpy(h_fluence.data(), d_fluence, vol_size * sizeof(float),
                       cudaMemcpyDeviceToHost);

            double fluence_sum = 0.0;
            for (size_t i = 0; i < vol_size; i++) fluence_sum += h_fluence[i];
            printf("Fluence sanity check: total deposited = %.6e\n", fluence_sum);
            if (fluence_sum == 0.0) {
                fprintf(stderr, "WARNING: Zero fluence - kernel may not have executed!\n");
            }

            snprintf(fname, sizeof(fname), "%s/fluence_%s.bin",
                     output_dir.c_str(), wl_names[wl]);
            save_binary(fname, h_fluence.data(), vol_size * sizeof(float));
        } else {
            // For large grids, just check a diagnostic sample
            float sample[1];
            size_t center = vol_size / 2;
            cudaMemcpy(sample, d_fluence + center, sizeof(float), cudaMemcpyDeviceToHost);
            printf("Fluence sanity check (center voxel): %.6e\n", (double)sample[0]);
            printf("(Skipping full fluence save for large grid: %zu voxels)\n", vol_size);
        }

        // TPSF binary
        snprintf(fname, sizeof(fname), "%s/tpsf_%s.bin",
                 output_dir.c_str(), wl_names[wl]);
        save_binary(fname, tpsf.data(), (size_t)n_dets * TPSF_BINS * sizeof(double));

        // Photon paths
        printf("Recorded photon paths: %d\n", num_paths);
        if (num_paths > 0) {
            // Save path metadata: det_id and path_length per path
            snprintf(fname, sizeof(fname), "%s/paths_meta_%s.bin",
                     output_dir.c_str(), wl_names[wl]);
            // Interleave det_id and length: [det0, len0, det1, len1, ...]
            std::vector<int> meta(num_paths * 2);
            for (int i = 0; i < num_paths; i++) {
                meta[i * 2 + 0] = path_det[i];
                meta[i * 2 + 1] = path_len[i];
            }
            save_binary(fname, meta.data(), num_paths * 2 * sizeof(int));

            // Save path positions: [x,y,z] * MAX_PATH_STEPS * num_paths
            snprintf(fname, sizeof(fname), "%s/paths_pos_%s.bin",
                     output_dir.c_str(), wl_names[wl]);
            save_binary(fname, path_pos.data(),
                        (size_t)num_paths * MAX_PATH_STEPS * 3 * sizeof(float));
        }

        // JSON results (with TD data)
        snprintf(fname, sizeof(fname), "%s/results_%s.json",
                 output_dir.c_str(), wl_names[wl]);
        save_results_json(fname, results.data(), n_dets,
                         det_layout.separations_mm, det_layout.angles_deg,
                         wl, config,
                         gated_weight.data(), gated_partial_pl.data(),
                         gated_count.data());
    }

    // --- Save volume for visualization (skip for large grids) ---
    char fname[256];
    if (vol_size <= 64000000ULL) {
        snprintf(fname, sizeof(fname), "%s/volume.bin", output_dir.c_str());
        save_binary(fname, volume.data(), vol_size);
    } else {
        printf("(Skipping volume.bin save for large grid: %zu voxels = %zu MB)\n",
               vol_size, vol_size / (1024 * 1024));
    }

    // Save volume metadata
    snprintf(fname, sizeof(fname), "%s/volume_meta.json", output_dir.c_str());
    FILE* f = fopen(fname, "w");
    if (f) {
        fprintf(f, "{\n");
        fprintf(f, "  \"nx\": %d, \"ny\": %d, \"nz\": %d,\n", hm.nx, hm.ny, hm.nz);
        fprintf(f, "  \"dx\": %.2f,\n", hm.dx);
        fprintf(f, "  \"tissue_labels\": {\n");
        fprintf(f, "    \"0\": \"air\", \"1\": \"scalp\", \"2\": \"skull\",\n");
        fprintf(f, "    \"3\": \"csf\", \"4\": \"gray_matter\",\n");
        fprintf(f, "    \"5\": \"white_matter\", \"6\": \"amygdala\"\n");
        fprintf(f, "  }\n}\n");
        fclose(f);
    }

    cudaFree(d_fluence);

    printf("\n=== Simulation complete. Outputs in %s/ ===\n", output_dir.c_str());
    return 0;
}
