/**
 * @file main.cpp
 * @brief Main entry point for MNI152-based Mesh Monte Carlo simulation.
 * 
 * This simulation uses a tetrahedral mesh derived from the MNI152 atlas
 * to model photon transport through realistic head anatomy for fNIRS
 * sensitivity analysis to the amygdala.
 */

#include "types.h"
#include "mesh_loader.h"
#include "mmc_kernel.h"
#include "optical_props.h"
#include "detector_array.h"
#include "simulation_config.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <vector>
#include <string>

using namespace mmc;

// CUDA error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/**
 * @brief Run simulation at a single wavelength.
 */
void run_wavelength(
    const SimulationConfig& config,
    GPUMesh& gpu_mesh,
    const DetectorArray& detectors,
    int wavelength_nm,
    uint64_t num_photons
) {
    printf("\n========================================\n");
    printf("  Running simulation at %d nm\n", wavelength_nm);
    printf("  Photons: %llu\n", num_photons);
    printf("========================================\n\n");
    
    // Initialize optical properties
    printf("Initializing optical properties...\n");
    OpticalProps* d_props = init_optical_properties_gpu(wavelength_nm);
    print_optical_properties(wavelength_nm);
    
    // Upload detectors
    printf("Uploading detector array...\n");
    int num_dets = detectors.get_num_detectors();
    Detector* d_detectors = detectors.upload_to_gpu();
    printf("  %d detectors\n", num_dets);
    
    // Allocate output arrays
    printf("Allocating output arrays...\n");
    PathRecord* d_records = nullptr;
    uint64_t* d_record_count = nullptr;
    float* d_tpsf = nullptr;
    SimulationStats* d_stats = nullptr;
    
    uint64_t max_records = config.max_path_records;
    CUDA_CHECK(cudaMalloc(&d_records, max_records * sizeof(PathRecord)));
    CUDA_CHECK(cudaMalloc(&d_record_count, sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&d_tpsf, num_dets * TPSF_BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_stats, sizeof(SimulationStats)));
    
    CUDA_CHECK(cudaMemset(d_record_count, 0, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_tpsf, 0, num_dets * TPSF_BINS * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(SimulationStats)));
    
    // Initialize RNG
    printf("Initializing random number generators...\n");
    int threads_per_block = config.threads_per_block;
    int num_blocks = 256;  // Number of blocks
    int num_threads = num_blocks * threads_per_block;
    
    curandState* d_rng_states = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rng_states, num_threads * sizeof(curandState)));
    
    init_rng_kernel<<<num_blocks, threads_per_block>>>(
        d_rng_states, 
        12345ULL + wavelength_nm,  // Seed based on wavelength
        num_threads
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Launch kernel
    printf("\nLaunching photons...\n");
    float3 source_pos = detectors.get_source_pos();
    
    uint64_t photons_per_thread = (num_photons + num_threads - 1) / num_threads;
    
    mmc_kernel<<<num_blocks, threads_per_block>>>(
        gpu_mesh,
        d_props,
        d_detectors,
        num_dets,
        d_records,
        d_record_count,
        d_tpsf,
        d_stats,
        photons_per_thread,
        source_pos,
        config.source_radius_mm,
        wavelength_nm,
        d_rng_states
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("Kernel complete.\n");
    
    // Download results
    SimulationStats stats;
    uint64_t record_count;
    std::vector<float> tpsf(num_dets * TPSF_BINS);
    std::vector<PathRecord> records;
    
    CUDA_CHECK(cudaMemcpy(&stats, d_stats, sizeof(SimulationStats), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&record_count, d_record_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(tpsf.data(), d_tpsf, num_dets * TPSF_BINS * sizeof(float), cudaMemcpyDeviceToHost));
    
    record_count = std::min(record_count, max_records);
    if (record_count > 0) {
        records.resize(record_count);
        CUDA_CHECK(cudaMemcpy(records.data(), d_records, 
                             record_count * sizeof(PathRecord), cudaMemcpyDeviceToHost));
    }
    
    // Print statistics
    printf("\nSimulation Statistics:\n");
    printf("  Photons launched: %llu\n", stats.launched);
    printf("  Detected: %llu (%.4f%%)\n", stats.detected, 
           100.0 * stats.detected / (double)stats.launched);
    printf("  Escaped: %llu\n", stats.escaped);
    printf("  Absorbed: %llu\n", stats.absorbed);
    printf("  Path records: %llu\n", record_count);
    
    // Count amygdala hits
    uint64_t amyg_hits = 0;
    float total_amyg_path = 0.0f;
    for (const auto& rec : records) {
        if (rec.hit_amyg) {
            amyg_hits++;
            total_amyg_path += rec.amyg_pathlen;
        }
    }
    
    if (!records.empty()) {
        printf("  Photons through amygdala: %llu (%.4f%% of detected)\n",
               amyg_hits, 100.0 * amyg_hits / (double)records.size());
        if (amyg_hits > 0) {
            printf("  Mean amygdala path length: %.4f mm\n", 
                   total_amyg_path / amyg_hits);
        }
    }
    
    // Save results
    char filename[256];
    
    // Save TPSF
    snprintf(filename, sizeof(filename), "%s/tpsf_%dmm.bin", 
             config.output_dir.c_str(), wavelength_nm);
    FILE* fp = fopen(filename, "wb");
    if (fp) {
        fwrite(&num_dets, sizeof(int), 1, fp);
        fwrite(&TPSF_BINS, sizeof(int), 1, fp);
        fwrite(tpsf.data(), sizeof(float), num_dets * TPSF_BINS, fp);
        fclose(fp);
        printf("\nSaved TPSF to: %s\n", filename);
    }
    
    // Save path records
    if (!records.empty()) {
        snprintf(filename, sizeof(filename), "%s/paths_%dmm.bin",
                 config.output_dir.c_str(), wavelength_nm);
        fp = fopen(filename, "wb");
        if (fp) {
            fwrite(&record_count, sizeof(uint64_t), 1, fp);
            fwrite(records.data(), sizeof(PathRecord), record_count, fp);
            fclose(fp);
            printf("Saved path records to: %s\n", filename);
        }
    }
    
    // Cleanup
    cudaFree(d_props);
    cudaFree(d_detectors);
    cudaFree(d_records);
    cudaFree(d_record_count);
    cudaFree(d_tpsf);
    cudaFree(d_stats);
    cudaFree(d_rng_states);
}

int main(int argc, char** argv) {
    printf("\n");
    printf("=================================================\n");
    printf("  Mesh-based Monte Carlo fNIRS Simulation\n");
    printf("  MNI152-derived head model for amygdala sensitivity\n");
    printf("=================================================\n");
    
    // Parse configuration
    SimulationConfig config;
    if (!config.parse_args(argc, argv)) {
        return 0;  // Help was printed
    }
    
    if (!config.validate()) {
        return 1;
    }
    
    config.print();
    
    // Initialize GPU
    printf("\nInitializing GPU (device %d)...\n", config.gpu_device);
    CUDA_CHECK(cudaSetDevice(config.gpu_device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, config.gpu_device));
    printf("  Device: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f GB\n", prop.totalGlobalMem / 1e9);
    
    // Create output directory
    #ifdef _WIN32
    system(("mkdir " + config.output_dir + " 2>nul").c_str());
    #else
    system(("mkdir -p " + config.output_dir).c_str());
    #endif
    
    // Load mesh
    GPUMesh gpu_mesh;
    if (!init_mesh(config.mesh_path, gpu_mesh)) {
        fprintf(stderr, "Failed to load mesh. Run generate_mni152_mesh.py first.\n");
        return 1;
    }
    
    // Create detector array
    DetectorArray detectors;
    detectors.print_config();
    
    // Run simulations
    if (config.num_photons_760nm > 0) {
        run_wavelength(config, gpu_mesh, detectors, 760, config.num_photons_760nm);
    }
    
    if (config.num_photons_850nm > 0) {
        run_wavelength(config, gpu_mesh, detectors, 850, config.num_photons_850nm);
    }
    
    // Cleanup
    gpu_mesh.free();
    
    printf("\n========================================\n");
    printf("  Simulation complete\n");
    printf("========================================\n\n");
    
    return 0;
}
