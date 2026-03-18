#pragma once
#include "types.cuh"
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

// Magic number and version for .mmcmesh binary format
#define MMC_MESH_MAGIC   0x42434D4D
#define MMC_MESH_VERSION 1

// Uniform grid accelerator resolution (cells per axis)
#define GRID_RES 64
#define GRID_CELLS (GRID_RES * GRID_RES * GRID_RES)

// ---------------------------------------------------------------------------
// Mesh geometry on GPU (flat arrays, no pointers-of-pointers)
// ---------------------------------------------------------------------------
struct MeshData {
    float* nodes;          // [N*3]   vertex positions (x,y,z) in mm
    int*   elements;       // [M*4]   vertex indices per tet (v0,v1,v2,v3)
    int*   tissue;         // [M]     TissueType per element
    int*   neighbors;      // [M*4]   neighbor tet per face (-1 = boundary)
                           //         face k is opposite vertex k

    // Precomputed per-face geometry (M*4 faces)
    float* face_normals;   // [M*4*3] outward unit normal per face
    float* face_d;         // [M*4]   plane constant: dot(normal, vertex_on_face)
    int*   face_pair;      // [M*4]   entry face in neighbor: face_pair[e*4+f] = face idx in neighbor

    int num_nodes;
    int num_elements;

    float bbox_min[3];
    float bbox_max[3];
};

// ---------------------------------------------------------------------------
// Uniform grid accelerator for point-in-tet queries
// ---------------------------------------------------------------------------
struct GridAccel {
    int*   offsets;        // [GRID_CELLS] start index into tets[] for each cell
    int*   counts;         // [GRID_CELLS] number of tets overlapping each cell
    int*   tets;           // [total_entries] packed tet indices
    int    total_entries;

    float  cell_size[3];   // size of each grid cell in mm
    float  bbox_min[3];    // grid origin (same as mesh bbox_min)
};

// ---------------------------------------------------------------------------
// Combined device data handle
// ---------------------------------------------------------------------------
struct MMCDeviceData {
    MeshData mesh;
    GridAccel grid;
};

// ---------------------------------------------------------------------------
// Host-side mesh container
// ---------------------------------------------------------------------------
struct HostMesh {
    std::vector<float> nodes;       // [N*3]
    std::vector<int>   elements;    // [M*4]
    std::vector<int>   tissue;      // [M]
    std::vector<int>   neighbors;   // [M*4]
    int num_nodes;
    int num_elements;
    float bbox_min[3];
    float bbox_max[3];
};

// Load .mmcmesh binary file into host memory
HostMesh load_mmcmesh(const char* path);

// Precompute face normals and plane constants on host
// Returns face_normals [M*4*3] and face_d [M*4]
void precompute_face_geometry(const HostMesh& mesh,
                              std::vector<float>& face_normals,
                              std::vector<float>& face_d);

// Precompute entry-face lookup table for fast neighbor traversal
// face_pair[e*4+f] = face index in neighbor that points back to e
void precompute_face_pair(const HostMesh& mesh, std::vector<int>& face_pair);

// Build uniform grid accelerator on host
void build_grid_accelerator(const HostMesh& mesh,
                            std::vector<int>& grid_offsets,
                            std::vector<int>& grid_counts,
                            std::vector<int>& grid_tets,
                            float cell_size[3]);

// Upload everything to GPU, returns device handles
MMCDeviceData upload_mesh_to_gpu(const HostMesh& mesh,
                                 const std::vector<float>& face_normals,
                                 const std::vector<float>& face_d,
                                 const std::vector<int>& face_pair,
                                 const std::vector<int>& grid_offsets,
                                 const std::vector<int>& grid_counts,
                                 const std::vector<int>& grid_tets,
                                 const float cell_size[3]);

// Free all GPU allocations
void free_mesh_gpu(MMCDeviceData& dev);
