#pragma once
#include "types.cuh"
#include <cstdint>

// ---------------------------------------------------------------------------
// Mesh-Based Monte Carlo (MMC) — Data Structures
// ---------------------------------------------------------------------------
// Photon transport through a tetrahedral mesh of the MNI152 head.
//
// Binary mesh file layout (*.mmcmesh):
//   Header  (48 bytes): magic, version, num_nodes, num_elems, bbox[6], pad[2]
//   Nodes   (N×3 float32): x, y, z [mm] in MNI152 space
//   Elems   (M×4 int32):  v0,v1,v2,v3 node indices (0-based)
//   Tissue  (M×1 int32):  TissueType per element
//   Neighbors (M×4 int32): neighbor element for each of the 4 faces (-1=boundary)
//
// Face convention: face k is opposite vertex k.
//   Face 0: (v1,v2,v3)   Face 1: (v0,v2,v3)
//   Face 2: (v0,v1,v3)   Face 3: (v0,v1,v2)
// ---------------------------------------------------------------------------

#define MMC_MESH_MAGIC    0x42434D4Du   // 'MMCB' little-endian
#define MMC_MESH_VERSION  1u

// Safety cap on boundary crossings per free-path step
#define MMC_MAX_CROSSINGS  1024

// Spatial grid resolution for element lookup
#define MMC_GRID_DIM  64

// ---------------------------------------------------------------------------
// Binary file header
// ---------------------------------------------------------------------------
struct MMCMeshHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t num_nodes;
    uint32_t num_elems;
    float    bbox_min[3];   // mm, MNI152 space
    float    bbox_max[3];   // mm
    uint32_t pad[2];        // reserved
};
static_assert(sizeof(MMCMeshHeader) == 48, "header size mismatch");

// ---------------------------------------------------------------------------
// Device-side mesh (all pointers into GPU memory)
// ---------------------------------------------------------------------------
struct MMCMeshDevice {
    const float*   nodes;       // [num_nodes * 3]  x,y,z interleaved [mm]
    const int32_t* elems;       // [num_elems  * 4]  v0..v3 node indices
    const int32_t* tissue;      // [num_elems]       TissueType per element
    const int32_t* neighbors;   // [num_elems  * 4]  neighbor elem IDs (-1=boundary)

    // Uniform spatial grid for point→element lookup
    const int32_t* grid_start;  // [GRID_CELLS] first index into grid_elems
    const int32_t* grid_count;  // [GRID_CELLS] number of entries
    const int32_t* grid_elems;  // [total_entries] element IDs

    uint32_t num_nodes;
    uint32_t num_elems;

    float bbox_min[3];
    float bbox_max[3];
    float cell_size[3];         // mm per cell axis
    int   grid_dims[3];         // cells per axis (≤ MMC_GRID_DIM)
};

// ---------------------------------------------------------------------------
// MMC-specific simulation configuration (placed in constant memory)
// ---------------------------------------------------------------------------
struct MMCConfig {
    // Photon budget
    uint64_t num_photons;

    // Source
    float src_pos[3];      // center [mm, MNI152]
    float src_dir[3];      // inward unit direction
    float beam_radius;     // disk radius [mm]
    int   src_elem;        // starting element (-1 → find at runtime)

    // Termination
    float weight_threshold;
    int   roulette_m;

    // Optical properties per tissue (current wavelength)
    OpticalProps tissue_props[NUM_TISSUE_TYPES];

    // TPSF
    int   tpsf_bins;
    float tpsf_bin_ps;     // ps per bin

    // Time gates
    float gate_start_ps[NUM_TIME_GATES];
    float gate_end_ps[NUM_TIME_GATES];
    int   num_gates;
};
