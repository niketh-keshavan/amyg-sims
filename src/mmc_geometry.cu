// ---------------------------------------------------------------------------
// MMC Geometry — mesh loading, spatial grid, and GPU upload
// ---------------------------------------------------------------------------

#include "mmc_geometry.h"
#include "mmc_kernel.cuh"

#include <cstring>
#include <cmath>
#include <cstdio>
#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA error check macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) do {                                        \
    cudaError_t _e = (call);                                         \
    if (_e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                __FILE__, __LINE__, cudaGetErrorString(_e));         \
        throw std::runtime_error(cudaGetErrorString(_e));            \
    }                                                                \
} while(0)

// ---------------------------------------------------------------------------
// Load mesh from binary file
// ---------------------------------------------------------------------------
MMCMeshHost load_mmc_mesh(const std::string& path)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("Cannot open mesh file: " + path);

    MMCMeshHeader hdr;
    fread(&hdr, sizeof(hdr), 1, f);
    if (hdr.magic != MMC_MESH_MAGIC)
        throw std::runtime_error("Bad MMC mesh magic in: " + path);
    if (hdr.version != MMC_MESH_VERSION)
        throw std::runtime_error("Unsupported MMC mesh version");

    MMCMeshHost host;
    host.num_nodes = hdr.num_nodes;
    host.num_elems = hdr.num_elems;
    memcpy(host.bbox_min, hdr.bbox_min, sizeof(hdr.bbox_min));
    memcpy(host.bbox_max, hdr.bbox_max, sizeof(hdr.bbox_max));

    printf("Loading MMC mesh: %u nodes, %u elements\n",
           host.num_nodes, host.num_elems);

    host.nodes.resize(host.num_nodes * 3);
    host.elems.resize(host.num_elems * 4);
    host.tissue.resize(host.num_elems);
    host.neighbors.resize(host.num_elems * 4);

    fread(host.nodes.data(),     sizeof(float),   host.num_nodes * 3,   f);
    fread(host.elems.data(),     sizeof(int32_t), host.num_elems * 4,   f);
    fread(host.tissue.data(),    sizeof(int32_t), host.num_elems,       f);
    fread(host.neighbors.data(), sizeof(int32_t), host.num_elems * 4,   f);

    fclose(f);
    return host;
}

// ---------------------------------------------------------------------------
// Build uniform spatial grid for element lookup
// ---------------------------------------------------------------------------
// Each element is registered in every grid cell that contains at least one
// of its four vertex nodes.  This is conservative (no false negatives) and
// sufficient for the point-in-tet refinement step.
// ---------------------------------------------------------------------------

static int cell_idx(const int dims[3], int ix, int iy, int iz)
{
    return ix + dims[0] * (iy + dims[1] * iz);
}

GridData build_spatial_grid(const MMCMeshHost& mesh)
{
    // Choose grid resolution based on mesh density
    int gd = MMC_GRID_DIM;
    int total_cells = gd * gd * gd;

    float dx = (mesh.bbox_max[0] - mesh.bbox_min[0]) / gd;
    float dy = (mesh.bbox_max[1] - mesh.bbox_min[1]) / gd;
    float dz = (mesh.bbox_max[2] - mesh.bbox_min[2]) / gd;

    int dims[3] = {gd, gd, gd};

    // Count pass: how many elements touch each cell
    std::vector<int32_t> count(total_cells, 0);

    auto node_cell = [&](int nidx, int axis) -> int {
        float v;
        if (axis == 0) v = mesh.nodes[3*nidx + 0];
        else if (axis == 1) v = mesh.nodes[3*nidx + 1];
        else                 v = mesh.nodes[3*nidx + 2];
        float cell_sz = (axis==0)?dx:(axis==1)?dy:dz;
        float mn = mesh.bbox_min[axis];
        int c = (int)((v - mn) / cell_sz);
        return std::max(0, std::min(gd-1, c));
    };

    // Track which cells each element occupies (via its vertices)
    // Use a small set per element: vertices span at most a few cells
    for (uint32_t e = 0; e < mesh.num_elems; e++) {
        int imin=gd,imax=-1, jmin=gd,jmax=-1, kmin=gd,kmax=-1;
        for (int v = 0; v < 4; v++) {
            int nid = mesh.elems[e*4 + v];
            int ci  = node_cell(nid, 0);
            int cj  = node_cell(nid, 1);
            int ck  = node_cell(nid, 2);
            imin=std::min(imin,ci); imax=std::max(imax,ci);
            jmin=std::min(jmin,cj); jmax=std::max(jmax,cj);
            kmin=std::min(kmin,ck); kmax=std::max(kmax,ck);
        }
        for (int i=imin;i<=imax;i++)
        for (int j=jmin;j<=jmax;j++)
        for (int k=kmin;k<=kmax;k++)
            count[cell_idx(dims,i,j,k)]++;
    }

    // Prefix sum to get start offsets
    std::vector<int32_t> start(total_cells, 0);
    int32_t total_entries = 0;
    for (int c = 0; c < total_cells; c++) {
        start[c] = total_entries;
        total_entries += count[c];
    }

    // Fill pass
    std::vector<int32_t> elems_list(total_entries);
    std::vector<int32_t> fill(total_cells, 0);

    for (uint32_t e = 0; e < mesh.num_elems; e++) {
        int imin=gd,imax=-1, jmin=gd,jmax=-1, kmin=gd,kmax=-1;
        for (int v = 0; v < 4; v++) {
            int nid = mesh.elems[e*4 + v];
            int ci  = node_cell(nid, 0);
            int cj  = node_cell(nid, 1);
            int ck  = node_cell(nid, 2);
            imin=std::min(imin,ci); imax=std::max(imax,ci);
            jmin=std::min(jmin,cj); jmax=std::max(jmax,cj);
            kmin=std::min(kmin,ck); kmax=std::max(kmax,ck);
        }
        for (int i=imin;i<=imax;i++)
        for (int j=jmin;j<=jmax;j++)
        for (int k=kmin;k<=kmax;k++) {
            int c = cell_idx(dims,i,j,k);
            elems_list[start[c] + fill[c]] = (int32_t)e;
            fill[c]++;
        }
    }

    printf("Spatial grid: %dx%dx%d, %d total cell-element pairs\n",
           gd, gd, gd, total_entries);

    GridData grid;
    grid.dims[0]=gd; grid.dims[1]=gd; grid.dims[2]=gd;
    grid.cell_size[0]=dx; grid.cell_size[1]=dy; grid.cell_size[2]=dz;
    grid.start = std::move(start);
    grid.count = std::move(count);
    grid.elems = std::move(elems_list);
    return grid;
}

// ---------------------------------------------------------------------------
// GPU device pointers (owned by this module)
// ---------------------------------------------------------------------------
static float*   d_nodes     = nullptr;
static int32_t* d_elems     = nullptr;
static int32_t* d_tissue    = nullptr;
static int32_t* d_neighbors = nullptr;
static int32_t* d_grid_start = nullptr;
static int32_t* d_grid_count = nullptr;
static int32_t* d_grid_elems = nullptr;

// ---------------------------------------------------------------------------
// Upload mesh to GPU, return populated MMCMeshDevice
// ---------------------------------------------------------------------------
MMCMeshDevice upload_mmc_mesh(const MMCMeshHost& mesh, const GridData& grid)
{
    size_t nodes_bytes     = mesh.num_nodes * 3 * sizeof(float);
    size_t elems_bytes     = mesh.num_elems * 4 * sizeof(int32_t);
    size_t tissue_bytes    = mesh.num_elems     * sizeof(int32_t);
    size_t neighbor_bytes  = mesh.num_elems * 4 * sizeof(int32_t);
    int    total_cells     = grid.dims[0]*grid.dims[1]*grid.dims[2];
    size_t grid_start_bytes = total_cells * sizeof(int32_t);
    size_t grid_count_bytes = total_cells * sizeof(int32_t);
    size_t grid_elems_bytes = grid.elems.size() * sizeof(int32_t);

    CUDA_CHECK(cudaMalloc(&d_nodes,      nodes_bytes));
    CUDA_CHECK(cudaMalloc(&d_elems,      elems_bytes));
    CUDA_CHECK(cudaMalloc(&d_tissue,     tissue_bytes));
    CUDA_CHECK(cudaMalloc(&d_neighbors,  neighbor_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid_start, grid_start_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid_count, grid_count_bytes));
    CUDA_CHECK(cudaMalloc(&d_grid_elems, grid_elems_bytes));

    CUDA_CHECK(cudaMemcpy(d_nodes,      mesh.nodes.data(),      nodes_bytes,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_elems,      mesh.elems.data(),      elems_bytes,      cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tissue,     mesh.tissue.data(),     tissue_bytes,     cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_neighbors,  mesh.neighbors.data(),  neighbor_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_start, grid.start.data(),      grid_start_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_count, grid.count.data(),      grid_count_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_elems, grid.elems.data(),      grid_elems_bytes, cudaMemcpyHostToDevice));

    MMCMeshDevice dev;
    dev.nodes     = d_nodes;
    dev.elems     = d_elems;
    dev.tissue    = d_tissue;
    dev.neighbors = d_neighbors;
    dev.grid_start = d_grid_start;
    dev.grid_count = d_grid_count;
    dev.grid_elems = d_grid_elems;
    dev.num_nodes  = mesh.num_nodes;
    dev.num_elems  = mesh.num_elems;
    memcpy(dev.bbox_min, mesh.bbox_min, 3*sizeof(float));
    memcpy(dev.bbox_max, mesh.bbox_max, 3*sizeof(float));
    dev.cell_size[0] = grid.cell_size[0];
    dev.cell_size[1] = grid.cell_size[1];
    dev.cell_size[2] = grid.cell_size[2];
    dev.grid_dims[0] = grid.dims[0];
    dev.grid_dims[1] = grid.dims[1];
    dev.grid_dims[2] = grid.dims[2];
    return dev;
}

void free_mmc_mesh_gpu()
{
    cudaFree(d_nodes); cudaFree(d_elems); cudaFree(d_tissue);
    cudaFree(d_neighbors); cudaFree(d_grid_start);
    cudaFree(d_grid_count); cudaFree(d_grid_elems);
    d_nodes=nullptr; d_elems=nullptr; d_tissue=nullptr;
    d_neighbors=nullptr; d_grid_start=nullptr;
    d_grid_count=nullptr; d_grid_elems=nullptr;
}

// ---------------------------------------------------------------------------
// Find source element — closest scalp-tissue element to a given point
// Projects the anatomical source point onto the mesh and returns the
// element ID just inside the scalp surface.
// ---------------------------------------------------------------------------
int find_source_element(const MMCMeshHost& mesh, const float pos[3])
{
    float best_dist = 1e30f;
    int   best_elem = -1;

    for (uint32_t e = 0; e < mesh.num_elems; e++) {
        int t = mesh.tissue[e];
        if (t != TISSUE_SCALP) continue;

        // Centroid
        float cx=0, cy=0, cz=0;
        for (int v=0; v<4; v++) {
            int nid = mesh.elems[e*4+v];
            cx += mesh.nodes[3*nid+0];
            cy += mesh.nodes[3*nid+1];
            cz += mesh.nodes[3*nid+2];
        }
        cx *= 0.25f; cy *= 0.25f; cz *= 0.25f;

        float dx = cx-pos[0], dy = cy-pos[1], dz = cz-pos[2];
        float d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < best_dist) {
            best_dist = d2;
            best_elem = e;
        }
    }
    return best_elem;
}

