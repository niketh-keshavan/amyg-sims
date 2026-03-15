/**
 * @file simulation_config.cpp
 * @brief Configuration management implementation.
 */

#include "simulation_config.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace mmc {

bool SimulationConfig::load_from_file(const std::string& path) {
    // Simple key-value parsing for now
    // In production, use a proper JSON library
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) return false;
    
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        char key[128], value[128];
        if (sscanf(line, "%127s = %127s", key, value) == 2) {
            if (strcmp(key, "mesh_path") == 0) mesh_path = value;
            else if (strcmp(key, "output_dir") == 0) output_dir = value;
            else if (strcmp(key, "num_photons_760nm") == 0) num_photons_760nm = strtoull(value, nullptr, 10);
            else if (strcmp(key, "num_photons_850nm") == 0) num_photons_850nm = strtoull(value, nullptr, 10);
            else if (strcmp(key, "max_bounces") == 0) max_bounces = atoi(value);
            else if (strcmp(key, "gpu_device") == 0) gpu_device = atoi(value);
        }
    }
    
    fclose(fp);
    return true;
}

bool SimulationConfig::save_to_file(const std::string& path) const {
    FILE* fp = fopen(path.c_str(), "w");
    if (!fp) return false;
    
    fprintf(fp, "# MMC Simulation Configuration\n");
    fprintf(fp, "mesh_path = %s\n", mesh_path.c_str());
    fprintf(fp, "output_dir = %s\n", output_dir.c_str());
    fprintf(fp, "num_photons_760nm = %llu\n", num_photons_760nm);
    fprintf(fp, "num_photons_850nm = %llu\n", num_photons_850nm);
    fprintf(fp, "max_bounces = %d\n", max_bounces);
    fprintf(fp, "min_weight = %f\n", min_weight);
    fprintf(fp, "gpu_device = %d\n", gpu_device);
    
    fclose(fp);
    return true;
}

bool SimulationConfig::parse_args(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mesh") == 0 && i + 1 < argc) {
            mesh_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (strcmp(argv[i], "--photons") == 0 && i + 1 < argc) {
            num_photons_760nm = strtoull(argv[++i], nullptr, 10);
            num_photons_850nm = num_photons_760nm;
        } else if (strcmp(argv[i], "--photons-760") == 0 && i + 1 < argc) {
            num_photons_760nm = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--photons-850") == 0 && i + 1 < argc) {
            num_photons_850nm = strtoull(argv[++i], nullptr, 10);
        } else if (strcmp(argv[i], "--max-bounces") == 0 && i + 1 < argc) {
            max_bounces = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpu") == 0 && i + 1 < argc) {
            gpu_device = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            load_from_file(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

void SimulationConfig::print() const {
    printf("\nSimulation Configuration:\n");
    printf("  Mesh file: %s\n", mesh_path.c_str());
    printf("  Output directory: %s\n", output_dir.c_str());
    printf("  Photons (760 nm): %llu\n", num_photons_760nm);
    printf("  Photons (850 nm): %llu\n", num_photons_850nm);
    printf("  Max bounces: %d\n", max_bounces);
    printf("  Min weight: %e\n", min_weight);
    printf("  GPU device: %d\n", gpu_device);
    printf("  Threads per block: %d\n", threads_per_block);
}

bool SimulationConfig::validate() const {
    if (mesh_path.empty()) {
        fprintf(stderr, "Error: Mesh path not specified\n");
        return false;
    }
    if (num_photons_760nm == 0 && num_photons_850nm == 0) {
        fprintf(stderr, "Error: No photons to simulate\n");
        return false;
    }
    return true;
}

void print_usage(const char* program_name) {
    printf("\nMesh-based Monte Carlo fNIRS Simulation (MNI152)\n");
    printf("Usage: %s [options]\n\n", program_name);
    printf("Options:\n");
    printf("  --mesh <path>         Path to .mmcmesh file (default: data/mni152_head.mmcmesh)\n");
    printf("  --output <dir>        Output directory (default: results_mmc)\n");
    printf("  --photons <N>         Photons per wavelength (default: 100M)\n");
    printf("  --photons-760 <N>     Photons at 760 nm\n");
    printf("  --photons-850 <N>     Photons at 850 nm\n");
    printf("  --max-bounces <N>     Maximum scattering events (default: 100000)\n");
    printf("  --gpu <id>            GPU device ID (default: 0)\n");
    printf("  --config <path>       Load configuration from file\n");
    printf("  --help, -h            Show this help\n");
    printf("\n");
}

} // namespace mmc
