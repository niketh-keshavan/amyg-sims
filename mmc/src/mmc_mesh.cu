#include "mmc_mesh.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

// ---------------------------------------------------------------------------
// Load .mmcmesh binary
// ---------------------------------------------------------------------------
HostMesh load_mmcmesh(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open mesh file: %s\n", path);
        exit(1);
    }

    // Header: magic(4) version(4) num_nodes(4) num_elems(4) bbox(6*4) pad(2*4) = 48 bytes
    uint32_t magic, version;
    uint32_t num_nodes, num_elems;
    float bbox[6];
    uint32_t pad[2];

    fread(&magic,     4, 1, f);
    fread(&version,   4, 1, f);
    fread(&num_nodes, 4, 1, f);
    fread(&num_elems, 4, 1, f);
    fread(bbox,       4, 6, f);
    fread(pad,        4, 2, f);

    if (magic != MMC_MESH_MAGIC) {
        fprintf(stderr, "ERROR: invalid mesh magic 0x%08X (expected 0x%08X)\n",
                magic, MMC_MESH_MAGIC);
        fclose(f);
        exit(1);
    }

    printf("Loading mesh: %s\n", path);
    printf("  Version: %u\n", version);
    printf("  Nodes: %u  Elements: %u\n", num_nodes, num_elems);
    printf("  BBox: (%.1f,%.1f,%.1f) -> (%.1f,%.1f,%.1f) mm\n",
           bbox[0], bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]);

    HostMesh mesh;
    mesh.num_nodes = (int)num_nodes;
    mesh.num_elements = (int)num_elems;
    for (int i = 0; i < 3; i++) {
        mesh.bbox_min[i] = bbox[i];
        mesh.bbox_max[i] = bbox[i + 3];
    }

    // Nodes: N*3 float32
    mesh.nodes.resize(num_nodes * 3);
    fread(mesh.nodes.data(), sizeof(float), num_nodes * 3, f);

    // Elements: M*4 int32
    mesh.elements.resize(num_elems * 4);
    fread(mesh.elements.data(), sizeof(int), num_elems * 4, f);

    // Tissue: M int32
    mesh.tissue.resize(num_elems);
    fread(mesh.tissue.data(), sizeof(int), num_elems, f);

    // Neighbors: M*4 int32
    mesh.neighbors.resize(num_elems * 4);
    fread(mesh.neighbors.data(), sizeof(int), num_elems * 4, f);

    fclose(f);

    // Print tissue distribution
    int tissue_counts[NUM_TISSUE_TYPES] = {};
    const char* tissue_names[] = {"air","scalp","skull","csf","gray","white","amygdala"};
    for (int i = 0; i < (int)num_elems; i++) {
        int t = mesh.tissue[i];
        if (t >= 0 && t < NUM_TISSUE_TYPES) tissue_counts[t]++;
    }
    printf("  Tissue distribution:\n");
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        if (tissue_counts[t] > 0)
            printf("    %s: %d tets\n", tissue_names[t], tissue_counts[t]);
    }

    return mesh;
}

// ---------------------------------------------------------------------------
// Precompute face normals and plane constants
// ---------------------------------------------------------------------------
// Face convention: face k is opposite vertex k
//   face 0: (v1, v2, v3)
//   face 1: (v0, v2, v3)
//   face 2: (v0, v1, v3)
//   face 3: (v0, v1, v2)

static const int FACE_VERTS[4][3] = {
    {1, 2, 3},  // face 0
    {0, 2, 3},  // face 1
    {0, 1, 3},  // face 2
    {0, 1, 2},  // face 3
};

void precompute_face_geometry(const HostMesh& mesh,
                              std::vector<float>& face_normals,
                              std::vector<float>& face_d) {
    int M = mesh.num_elements;
    face_normals.resize(M * 4 * 3);
    face_d.resize(M * 4);

    for (int e = 0; e < M; e++) {
        int v[4];
        float p[4][3];
        for (int i = 0; i < 4; i++) {
            v[i] = mesh.elements[e * 4 + i];
            p[i][0] = mesh.nodes[v[i] * 3 + 0];
            p[i][1] = mesh.nodes[v[i] * 3 + 1];
            p[i][2] = mesh.nodes[v[i] * 3 + 2];
        }

        for (int f = 0; f < 4; f++) {
            int a = FACE_VERTS[f][0];
            int b = FACE_VERTS[f][1];
            int c = FACE_VERTS[f][2];

            // Two edges of the face triangle
            float e1[3] = { p[b][0]-p[a][0], p[b][1]-p[a][1], p[b][2]-p[a][2] };
            float e2[3] = { p[c][0]-p[a][0], p[c][1]-p[a][1], p[c][2]-p[a][2] };

            // Normal via cross product
            float nx = e1[1]*e2[2] - e1[2]*e2[1];
            float ny = e1[2]*e2[0] - e1[0]*e2[2];
            float nz = e1[0]*e2[1] - e1[1]*e2[0];

            float len = sqrtf(nx*nx + ny*ny + nz*nz);
            if (len > 1e-12f) {
                nx /= len; ny /= len; nz /= len;
            }

            // Orient outward: normal should point away from the opposite vertex (vertex f)
            float to_opp[3] = { p[f][0]-p[a][0], p[f][1]-p[a][1], p[f][2]-p[a][2] };
            float dot = nx*to_opp[0] + ny*to_opp[1] + nz*to_opp[2];
            if (dot > 0.0f) {
                nx = -nx; ny = -ny; nz = -nz;
            }

            int idx = e * 4 + f;
            face_normals[idx * 3 + 0] = nx;
            face_normals[idx * 3 + 1] = ny;
            face_normals[idx * 3 + 2] = nz;

            // Plane constant: d = dot(normal, vertex_on_face)
            face_d[idx] = nx*p[a][0] + ny*p[a][1] + nz*p[a][2];
        }
    }

    printf("  Precomputed %d face normals/planes\n", M * 4);
}

// ---------------------------------------------------------------------------
// Precompute entry-face lookup table for fast neighbor entry-face lookup
// face_pair[e*4+f] = the face index in neighbors[e*4+f] that maps back to e
// ---------------------------------------------------------------------------
void precompute_face_pair(const HostMesh& mesh, std::vector<int>& face_pair) {
    int M = mesh.num_elements;
    face_pair.resize(M * 4);
    
    for (int e = 0; e < M; e++) {
        for (int f = 0; f < 4; f++) {
            int neighbor = mesh.neighbors[e * 4 + f];
            if (neighbor >= 0) {
                // Find which face of the neighbor points back to e
                int entry_face = -1;
                for (int nf = 0; nf < 4; nf++) {
                    if (mesh.neighbors[neighbor * 4 + nf] == e) {
                        entry_face = nf;
                        break;
                    }
                }
                face_pair[e * 4 + f] = entry_face;
            } else {
                face_pair[e * 4 + f] = -1;  // boundary face
            }
        }
    }
    
    printf("  Precomputed %d face pairs\n", M * 4);
}

// ---------------------------------------------------------------------------
// Uniform grid accelerator for point-in-tet queries
// ---------------------------------------------------------------------------
void build_grid_accelerator(const HostMesh& mesh,
                            std::vector<int>& grid_offsets,
                            std::vector<int>& grid_counts,
                            std::vector<int>& grid_tets,
                            float cell_size[3]) {
    // Compute cell size with small padding to avoid edge cases
    float pad = 0.1f;
    for (int d = 0; d < 3; d++) {
        cell_size[d] = (mesh.bbox_max[d] - mesh.bbox_min[d] + 2*pad) / GRID_RES;
    }

    printf("  Building uniform grid accelerator (%d^3 cells)...\n", GRID_RES);
    printf("    Cell size: %.2f x %.2f x %.2f mm\n",
           cell_size[0], cell_size[1], cell_size[2]);

    // Count tets per cell
    grid_counts.assign(GRID_CELLS, 0);

    // For each tet, find its AABB, map to grid cells
    std::vector<std::vector<int>> cell_lists(GRID_CELLS);

    float origin[3] = {
        mesh.bbox_min[0] - pad,
        mesh.bbox_min[1] - pad,
        mesh.bbox_min[2] - pad
    };

    for (int e = 0; e < mesh.num_elements; e++) {
        // Compute tet AABB
        float tmin[3] = { 1e30f, 1e30f, 1e30f };
        float tmax[3] = {-1e30f,-1e30f,-1e30f };
        for (int i = 0; i < 4; i++) {
            int vi = mesh.elements[e * 4 + i];
            for (int d = 0; d < 3; d++) {
                float val = mesh.nodes[vi * 3 + d];
                tmin[d] = std::min(tmin[d], val);
                tmax[d] = std::max(tmax[d], val);
            }
        }

        // Map AABB to grid cell range
        int cmin[3], cmax[3];
        for (int d = 0; d < 3; d++) {
            cmin[d] = std::max(0, (int)floorf((tmin[d] - origin[d]) / cell_size[d]));
            cmax[d] = std::min(GRID_RES - 1, (int)floorf((tmax[d] - origin[d]) / cell_size[d]));
        }

        // Insert tet into all overlapping cells
        for (int iz = cmin[2]; iz <= cmax[2]; iz++)
        for (int iy = cmin[1]; iy <= cmax[1]; iy++)
        for (int ix = cmin[0]; ix <= cmax[0]; ix++) {
            int cell = iz * GRID_RES * GRID_RES + iy * GRID_RES + ix;
            cell_lists[cell].push_back(e);
        }
    }

    // Flatten to packed arrays
    int total = 0;
    grid_offsets.resize(GRID_CELLS);
    grid_counts.resize(GRID_CELLS);
    for (int i = 0; i < GRID_CELLS; i++) {
        grid_offsets[i] = total;
        grid_counts[i] = (int)cell_lists[i].size();
        total += grid_counts[i];
    }

    grid_tets.resize(total);
    for (int i = 0; i < GRID_CELLS; i++) {
        for (int j = 0; j < (int)cell_lists[i].size(); j++) {
            grid_tets[grid_offsets[i] + j] = cell_lists[i][j];
        }
    }

    // Stats
    int max_count = 0, nonempty = 0;
    for (int i = 0; i < GRID_CELLS; i++) {
        if (grid_counts[i] > 0) nonempty++;
        max_count = std::max(max_count, grid_counts[i]);
    }
    printf("    Total entries: %d  Non-empty cells: %d/%d  Max tets/cell: %d\n",
           total, nonempty, GRID_CELLS, max_count);
}

// ---------------------------------------------------------------------------
// Upload to GPU
// ---------------------------------------------------------------------------
static void* gpu_alloc_copy(const void* src, size_t bytes) {
    void* d_ptr = nullptr;
    cudaMalloc(&d_ptr, bytes);
    cudaMemcpy(d_ptr, src, bytes, cudaMemcpyHostToDevice);
    return d_ptr;
}

MMCDeviceData upload_mesh_to_gpu(const HostMesh& mesh,
                                 const std::vector<float>& face_normals,
                                 const std::vector<float>& face_d,
                                 const std::vector<int>& face_pair,
                                 const std::vector<int>& grid_offsets,
                                 const std::vector<int>& grid_counts,
                                 const std::vector<int>& grid_tets,
                                 const float cell_size[3]) {
    MMCDeviceData dev;
    memset(&dev, 0, sizeof(dev));

    int N = mesh.num_nodes;
    int M = mesh.num_elements;

    // Mesh geometry
    dev.mesh.nodes        = (float*)gpu_alloc_copy(mesh.nodes.data(),     N*3*sizeof(float));
    dev.mesh.elements     = (int*)  gpu_alloc_copy(mesh.elements.data(),  M*4*sizeof(int));
    dev.mesh.tissue       = (int*)  gpu_alloc_copy(mesh.tissue.data(),    M*sizeof(int));
    dev.mesh.neighbors    = (int*)  gpu_alloc_copy(mesh.neighbors.data(), M*4*sizeof(int));
    dev.mesh.face_normals = (float*)gpu_alloc_copy(face_normals.data(),   M*4*3*sizeof(float));
    dev.mesh.face_d       = (float*)gpu_alloc_copy(face_d.data(),         M*4*sizeof(float));
    dev.mesh.face_pair    = (int*)  gpu_alloc_copy(face_pair.data(),      M*4*sizeof(int));
    dev.mesh.num_nodes    = N;
    dev.mesh.num_elements = M;
    memcpy(dev.mesh.bbox_min, mesh.bbox_min, 3*sizeof(float));
    memcpy(dev.mesh.bbox_max, mesh.bbox_max, 3*sizeof(float));

    // Grid accelerator
    int total = (int)grid_tets.size();
    dev.grid.offsets       = (int*)gpu_alloc_copy(grid_offsets.data(), GRID_CELLS*sizeof(int));
    dev.grid.counts        = (int*)gpu_alloc_copy(grid_counts.data(), GRID_CELLS*sizeof(int));
    dev.grid.tets          = (int*)gpu_alloc_copy(grid_tets.data(),   total*sizeof(int));
    dev.grid.total_entries = total;
    memcpy(dev.grid.cell_size, cell_size, 3*sizeof(float));
    // Grid origin = bbox_min - pad (same pad=0.1 as build)
    for (int d = 0; d < 3; d++)
        dev.grid.bbox_min[d] = mesh.bbox_min[d] - 0.1f;

    size_t total_bytes = (size_t)N*3*4 + (size_t)M*4*4 + (size_t)M*4 + (size_t)M*4*4
                       + (size_t)M*4*3*4 + (size_t)M*4*4 + (size_t)M*4*4
                       + GRID_CELLS*4*2 + (size_t)total*4;
    printf("  Uploaded to GPU: %.1f MB\n", total_bytes / 1e6);

    return dev;
}

// ---------------------------------------------------------------------------
// Free GPU allocations
// ---------------------------------------------------------------------------
void free_mesh_gpu(MMCDeviceData& dev) {
    cudaFree(dev.mesh.nodes);
    cudaFree(dev.mesh.elements);
    cudaFree(dev.mesh.tissue);
    cudaFree(dev.mesh.neighbors);
    cudaFree(dev.mesh.face_normals);
    cudaFree(dev.mesh.face_d);
    cudaFree(dev.mesh.face_pair);
    cudaFree(dev.grid.offsets);
    cudaFree(dev.grid.counts);
    cudaFree(dev.grid.tets);
    memset(&dev, 0, sizeof(dev));
}
