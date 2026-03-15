/**
 * @file mesh_loader.cpp
 * @brief MNI152 mesh loading implementation.
 */

#include "mesh_loader.h"
#include <cstdio>
#include <cstring>

namespace mmc {

bool Mesh::load_from_file(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open mesh file: %s\n", path.c_str());
        return false;
    }
    
    // Read header
    MeshHeader header;
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fprintf(stderr, "Error: Cannot read mesh header\n");
        fclose(fp);
        return false;
    }
    
    // Validate magic number
    if (header.magic != 0x42434D4D) {
        fprintf(stderr, "Error: Invalid mesh file magic number: 0x%08X\n", header.magic);
        fclose(fp);
        return false;
    }
    
    printf("Loading mesh version %u:\n", header.version);
    printf("  Nodes: %u, Elements: %u\n", header.num_nodes, header.num_elems);
    
    // Allocate storage
    nodes.resize(header.num_nodes);
    elems.resize(header.num_elems);
    
    // Read nodes
    std::vector<float> node_data(header.num_nodes * 3);
    if (fread(node_data.data(), sizeof(float), node_data.size(), fp) != node_data.size()) {
        fprintf(stderr, "Error: Cannot read node data\n");
        fclose(fp);
        return false;
    }
    
    for (uint32_t i = 0; i < header.num_nodes; i++) {
        nodes[i].x = node_data[i * 3 + 0];
        nodes[i].y = node_data[i * 3 + 1];
        nodes[i].z = node_data[i * 3 + 2];
    }
    
    // Read elements
    std::vector<int32_t> elem_data(header.num_elems * 4);
    if (fread(elem_data.data(), sizeof(int32_t), elem_data.size(), fp) != elem_data.size()) {
        fprintf(stderr, "Error: Cannot read element data\n");
        fclose(fp);
        return false;
    }
    
    for (uint32_t i = 0; i < header.num_elems; i++) {
        elems[i].v[0] = elem_data[i * 4 + 0];
        elems[i].v[1] = elem_data[i * 4 + 1];
        elems[i].v[2] = elem_data[i * 4 + 2];
        elems[i].v[3] = elem_data[i * 4 + 3];
    }
    
    // Read tissue labels
    std::vector<int32_t> tissue_data(header.num_elems);
    if (fread(tissue_data.data(), sizeof(int32_t), header.num_elems, fp) != header.num_elems) {
        fprintf(stderr, "Error: Cannot read tissue labels\n");
        fclose(fp);
        return false;
    }
    
    for (uint32_t i = 0; i < header.num_elems; i++) {
        elems[i].tissue = tissue_data[i];
    }
    
    // Read neighbor connectivity
    std::vector<int32_t> neighbor_data(header.num_elems * 4);
    if (fread(neighbor_data.data(), sizeof(int32_t), neighbor_data.size(), fp) != neighbor_data.size()) {
        fprintf(stderr, "Error: Cannot read neighbor data\n");
        fclose(fp);
        return false;
    }
    
    for (uint32_t i = 0; i < header.num_elems; i++) {
        elems[i].neighbor[0] = neighbor_data[i * 4 + 0];
        elems[i].neighbor[1] = neighbor_data[i * 4 + 1];
        elems[i].neighbor[2] = neighbor_data[i * 4 + 2];
        elems[i].neighbor[3] = neighbor_data[i * 4 + 3];
    }
    
    fclose(fp);
    
    // Set bounding box
    bbox_min = make_float3(header.bbox_min[0], header.bbox_min[1], header.bbox_min[2]);
    bbox_max = make_float3(header.bbox_max[0], header.bbox_max[1], header.bbox_max[2]);
    
    printf("  Bounding box: (%.1f, %.1f, %.1f) to (%.1f, %.1f, %.1f)\n",
           bbox_min.x, bbox_min.y, bbox_min.z,
           bbox_max.x, bbox_max.y, bbox_max.z);
    
    return true;
}

void Mesh::print_stats() const {
    printf("Mesh Statistics:\n");
    printf("  Nodes: %zu\n", nodes.size());
    printf("  Elements: %zu\n", elems.size());
    
    // Count tissues
    int tissue_counts[NUM_TISSUES] = {0};
    for (const auto& elem : elems) {
        if (elem.tissue >= 0 && elem.tissue < NUM_TISSUES) {
            tissue_counts[elem.tissue]++;
        }
    }
    
    const char* tissue_names[] = {"Air", "Scalp", "Skull", "CSF", "Gray", "White", "Amygdala"};
    for (int i = 0; i < NUM_TISSUES; i++) {
        if (tissue_counts[i] > 0) {
            printf("  %s: %d tets (%.1f%%)\n", 
                   tissue_names[i], tissue_counts[i],
                   100.0f * tissue_counts[i] / elems.size());
        }
    }
    
    // Compute volume statistics
    double total_vol = 0.0;
    double amyg_vol = 0.0;
    for (size_t i = 0; i < elems.size(); i++) {
        float vol = compute_volume(static_cast<int>(i));
        total_vol += vol;
        if (elems[i].tissue == TISSUE_AMYGDALA) {
            amyg_vol += vol;
        }
    }
    
    printf("  Total volume: %.1f cm³\n", total_vol / 1000.0f);
    printf("  Amygdala volume: %.2f cm³\n", amyg_vol / 1000.0f);
}

float Mesh::compute_volume(int elem_idx) const {
    const auto& elem = elems[elem_idx];
    float3 v0 = nodes[elem.v[0]];
    float3 v1 = nodes[elem.v[1]];
    float3 v2 = nodes[elem.v[2]];
    float3 v3 = nodes[elem.v[3]];
    
    float3 a = v1 - v0;
    float3 b = v2 - v0;
    float3 c = v3 - v0;
    
    return fabsf(dot(cross(a, b), c)) / 6.0f;
}

float3 Mesh::compute_centroid(int elem_idx) const {
    const auto& elem = elems[elem_idx];
    float3 v0 = nodes[elem.v[0]];
    float3 v1 = nodes[elem.v[1]];
    float3 v2 = nodes[elem.v[2]];
    float3 v3 = nodes[elem.v[3]];
    
    return make_float3(
        (v0.x + v1.x + v2.x + v3.x) * 0.25f,
        (v0.y + v1.y + v2.y + v3.y) * 0.25f,
        (v0.z + v1.z + v2.z + v3.z) * 0.25f
    );
}

void Mesh::compute_bounding_box() {
    if (nodes.empty()) return;
    
    bbox_min = nodes[0];
    bbox_max = nodes[0];
    
    for (const auto& n : nodes) {
        bbox_min.x = fminf(bbox_min.x, n.x);
        bbox_min.y = fminf(bbox_min.y, n.y);
        bbox_min.z = fminf(bbox_min.z, n.z);
        bbox_max.x = fmaxf(bbox_max.x, n.x);
        bbox_max.y = fmaxf(bbox_max.y, n.y);
        bbox_max.z = fmaxf(bbox_max.z, n.z);
    }
}

void GPUMesh::allocate(const Mesh& cpu_mesh) {
    num_nodes = static_cast<int32_t>(cpu_mesh.nodes.size());
    num_elems = static_cast<int32_t>(cpu_mesh.elems.size());
    bbox_min = cpu_mesh.bbox_min;
    bbox_max = cpu_mesh.bbox_max;
    
    cudaMalloc(&nodes, num_nodes * sizeof(float3));
    cudaMalloc(&elems, num_elems * sizeof(Tetrahedron));
}

void GPUMesh::free() {
    if (nodes) cudaFree(nodes);
    if (elems) cudaFree(elems);
    nodes = nullptr;
    elems = nullptr;
    num_nodes = 0;
    num_elems = 0;
}

void GPUMesh::copy_from_cpu(const Mesh& cpu_mesh) {
    cudaMemcpy(nodes, cpu_mesh.nodes.data(), 
               num_nodes * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(elems, cpu_mesh.elems.data(),
               num_elems * sizeof(Tetrahedron), cudaMemcpyHostToDevice);
}

bool init_mesh(const std::string& mesh_path, GPUMesh& gpu_mesh) {
    Mesh cpu_mesh;
    
    printf("Loading mesh from: %s\n", mesh_path.c_str());
    if (!cpu_mesh.load_from_file(mesh_path)) {
        return false;
    }
    
    cpu_mesh.print_stats();
    
    printf("Uploading to GPU...\n");
    gpu_mesh.allocate(cpu_mesh);
    gpu_mesh.copy_from_cpu(cpu_mesh);
    
    printf("Mesh loaded successfully.\n");
    return true;
}

} // namespace mmc
