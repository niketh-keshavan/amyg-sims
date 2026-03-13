#pragma once
// ---------------------------------------------------------------------------
// Host-side MMC mesh types (shared between mmc_geometry.cu and mmc_main.cu)
// ---------------------------------------------------------------------------
#include "mmc_types.cuh"
#include <vector>
#include <string>

struct MMCMeshHost {
    uint32_t num_nodes;
    uint32_t num_elems;
    float    bbox_min[3];
    float    bbox_max[3];
    std::vector<float>   nodes;      // [num_nodes * 3]
    std::vector<int32_t> elems;      // [num_elems  * 4]
    std::vector<int32_t> tissue;     // [num_elems]
    std::vector<int32_t> neighbors;  // [num_elems  * 4]
};

struct GridData {
    int   dims[3];
    float cell_size[3];
    std::vector<int32_t> start;
    std::vector<int32_t> count;
    std::vector<int32_t> elems;
};

// Defined in mmc_geometry.cu
MMCMeshHost   load_mmc_mesh(const std::string& path);
GridData      build_spatial_grid(const MMCMeshHost& mesh);
MMCMeshDevice upload_mmc_mesh(const MMCMeshHost& mesh, const GridData& grid);
void          free_mmc_mesh_gpu();
int           find_source_element(const MMCMeshHost& mesh, const float pos[3]);
