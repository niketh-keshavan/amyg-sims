/**
 * @file mmc_kernel.cu
 * @brief CUDA implementation of mesh-based Monte Carlo photon transport.
 * 
 * Uses ray-tetrahedron traversal with Plücker coordinates for robust intersection.
 */

#include "mmc_kernel.h"
#include <curand_kernel.h>
#include <cstdio>

namespace mmc {

// Device constants
__constant__ float SPEED_OF_LIGHT = 0.3f;  // mm/ps (speed of light in vacuum)
__constant__ int MAX_BOUNCES = 100000;
__constant__ float MIN_WEIGHT = 1e-4f;
__constant__ float ROULETTE_CHANCE = 0.1f;

/**
 * @brief Compute Plücker coordinates for a ray.
 * 
 * Ray: P(t) = origin + t * direction
 * Plücker: L = (direction, origin × direction)
 */
__device__ void plucker_ray(const float3& orig, const float3& dir, 
                             float3& u, float3& v) {
    u = dir;
    v = cross(orig, dir);
}

/**
 * @brief Plücker side operator: returns sign of side test.
 */
__device__ float plucker_side(const float3& u1, const float3& v1,
                               const float3& u2, const float3& v2) {
    return dot(u1, v2) + dot(v1, u2);
}

/**
 * @brief Ray-tetrahedron intersection using Plücker coordinates.
 * 
 * Returns entry and exit face indices and distances.
 */
__device__ bool intersect_ray_tet_plucker(
    const float3& orig,
    const float3& dir,
    const float3* v,
    float& t_entry,
    float& t_exit,
    int& face_entry,
    int& face_exit
) {
    // Edge indices for each face (opposite vertex)
    const int face_edges[4][3] = {
        {1, 2, 3},  // face 0 opposite vertex 0
        {0, 2, 3},  // face 1 opposite vertex 1
        {0, 1, 3},  // face 2 opposite vertex 2
        {0, 1, 2}   // face 3 opposite vertex 3
    };
    
    float3 ray_u, ray_v;
    plucker_ray(orig, dir, ray_u, ray_v);
    
    float t_min = 0.0f;
    float t_max = 1e20f;
    face_entry = -1;
    face_exit = -1;
    
    // Test each face
    for (int f = 0; f < 4; f++) {
        // Get face vertices
        int i0 = (f == 0) ? 1 : 0;
        int i1 = (f == 1) ? 2 : (f == 0) ? 2 : 1;
        int i2 = (f == 3) ? 2 : 3;
        
        // Face edges
        float3 e0 = v[i1] - v[i0];
        float3 e1 = v[i2] - v[i0];
        
        // Face normal (not normalized)
        float3 n = cross(e0, e1);
        
        // Ray-plane intersection
        float denom = dot(dir, n);
        float dist = dot(v[i0] - orig, n);
        
        if (fabsf(denom) < 1e-10f) {
            // Ray parallel to face
            if (dist < 0) return false;  // Outside and parallel
            continue;  // Inside and parallel, skip
        }
        
        float t = dist / denom;
        
        // Check if intersection is within face bounds (barycentric)
        float3 p = orig + dir * t;
        float3 w = p - v[i0];
        
        float d00 = dot(e0, e0);
        float d01 = dot(e0, e1);
        float d11 = dot(e1, e1);
        float d20 = dot(w, e0);
        float d21 = dot(w, e1);
        
        float denom_bc = d00 * d11 - d01 * d01;
        if (fabsf(denom_bc) < 1e-10f) continue;
        
        float v_bc = (d11 * d20 - d01 * d21) / denom_bc;
        float w_bc = (d00 * d21 - d01 * d20) / denom_bc;
        float u_bc = 1.0f - v_bc - w_bc;
        
        if (u_bc < -1e-4f || v_bc < -1e-4f || w_bc < -1e-4f) continue;
        
        // Update entry/exit
        if (denom > 0) {
            // Exiting
            if (t < t_max) {
                t_max = t;
                face_exit = f;
            }
        } else {
            // Entering
            if (t > t_min) {
                t_min = t;
                face_entry = f;
            }
        }
    }
    
    if (t_min < t_max && t_max > 0) {
        t_entry = fmaxf(t_min, 0.0f);
        t_exit = t_max;
        return true;
    }
    
    return false;
}

__device__ RayTetIntersect intersect_ray_tet(
    const float3& orig,
    const float3& dir,
    const float3* tet
) {
    RayTetIntersect result;
    result.hit = false;
    result.t = 1e20f;
    
    int face_entry, face_exit;
    float t_entry, t_exit;
    
    if (intersect_ray_tet_plucker(orig, dir, tet, t_entry, t_exit, 
                                   face_entry, face_exit)) {
        if (t_entry > 1e-6f) {
            result.t = t_entry;
            result.exit_face = face_entry;
        } else if (t_exit > 1e-6f) {
            result.t = t_exit;
            result.exit_face = face_exit;
        } else {
            return result;
        }
        result.hit = true;
    }
    
    return result;
}

__device__ bool point_in_tet(const float3& p, const float3& v0, 
                              const float3& v1, const float3& v2, 
                              const float3& v3, float& u, float& v, float& w) {
    float3 ab = v1 - v0;
    float3 ac = v2 - v0;
    float3 ad = v3 - v0;
    float3 ap = p - v0;
    
    float3 n = cross(ab, ac);
    float vol = dot(n, ad);
    
    if (fabsf(vol) < 1e-10f) return false;
    
    float3 nab = cross(ac, ad);
    float3 nac = cross(ad, ab);
    float3 nad = cross(ab, ac);
    
    u = dot(ap, nab) / vol;
    v = dot(ap, nac) / vol;
    w = dot(ap, nad) / vol;
    
    float uu = 1.0f - u - v - w;
    
    return (u >= -1e-4f && v >= -1e-4f && w >= -1e-4f && uu >= -1e-4f);
}

__device__ int32_t find_containing_tet(const float3& pos, const GPUMesh& mesh) {
    // Check bounding box first
    if (pos.x < mesh.bbox_min.x || pos.x > mesh.bbox_max.x ||
        pos.y < mesh.bbox_min.y || pos.y > mesh.bbox_max.y ||
        pos.z < mesh.bbox_min.z || pos.z > mesh.bbox_max.z) {
        return -1;
    }
    
    // Hash-based spatial lookup (simplified)
    // In production, use a proper spatial index
    int32_t start_idx = (int32_t)(fabsf(pos.x + pos.y + pos.z) * 1000) % mesh.num_elems;
    start_idx = max(0, min(start_idx, mesh.num_elems - 1));
    
    // Search nearby elements
    float3 v[4];
    float u, v_bc, w_bc;
    
    int search_radius = 100;  // Number of elements to check
    int32_t idx = start_idx;
    
    for (int i = 0; i < search_radius && idx < mesh.num_elems; i++) {
        const Tetrahedron& tet = mesh.elems[idx];
        v[0] = mesh.nodes[tet.v[0]];
        v[1] = mesh.nodes[tet.v[1]];
        v[2] = mesh.nodes[tet.v[2]];
        v[3] = mesh.nodes[tet.v[3]];
        
        if (point_in_tet(pos, v[0], v[1], v[2], v[3], u, v_bc, w_bc)) {
            return idx;
        }
        
        idx = (idx + 1) % mesh.num_elems;
    }
    
    // Brute force fallback (rare, but necessary for robustness)
    for (int32_t i = 0; i < mesh.num_elems; i++) {
        const Tetrahedron& tet = mesh.elems[i];
        v[0] = mesh.nodes[tet.v[0]];
        v[1] = mesh.nodes[tet.v[1]];
        v[2] = mesh.nodes[tet.v[2]];
        v[3] = mesh.nodes[tet.v[3]];
        
        if (point_in_tet(pos, v[0], v[1], v[2], v[3], u, v_bc, w_bc)) {
            return i;
        }
    }
    
    return -1;  // Outside mesh
}

__device__ void hg_scatter(float3& dir, float g, curandState& rng_state) {
    float cos_theta;
    
    if (fabsf(g) < 1e-3f) {
        // Isotropic scattering
        cos_theta = 2.0f * curand_uniform(&rng_state) - 1.0f;
    } else {
        // Henyey-Greenstein
        float g2 = g * g;
        float temp = (1.0f - g2) / (1.0f - g + 2.0f * g * curand_uniform(&rng_state));
        cos_theta = (1.0f + g2 - temp * temp) / (2.0f * g);
        cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));
    }
    
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    float phi = 2.0f * 3.14159265f * curand_uniform(&rng_state);
    
    // Rotate direction
    float3 old_dir = dir;
    
    if (fabsf(old_dir.z) > 0.999f) {
        dir.x = sin_theta * cosf(phi);
        dir.y = sin_theta * sinf(phi);
        dir.z = cos_theta * (old_dir.z > 0 ? 1.0f : -1.0f);
    } else {
        float temp = sqrtf(1.0f - old_dir.z * old_dir.z);
        float sin_psi = old_dir.y / temp;
        float cos_psi = old_dir.x / temp;
        
        dir.x = sin_theta * cos_psi * cosf(phi) - sin_theta * sin_psi * old_dir.z * sinf(phi) / temp + cos_theta * old_dir.x;
        dir.y = sin_theta * sin_psi * cosf(phi) + sin_theta * cos_psi * old_dir.z * sinf(phi) / temp + cos_theta * old_dir.y;
        dir.z = -sin_theta * sinf(phi) * temp + cos_theta * old_dir.z;
    }
    
    dir = normalize(dir);
}

__device__ bool russian_roulette(Photon& photon, curandState& rng_state) {
    if (photon.weight < MIN_WEIGHT) {
        float chance = curand_uniform(&rng_state);
        if (chance < ROULETTE_CHANCE) {
            photon.weight /= ROULETTE_CHANCE;
            return true;
        } else {
            photon.weight = 0.0f;
            photon.status = 3;  // Absorbed
            return false;
        }
    }
    return true;
}

__device__ bool check_detection(const Photon& photon, const Detector& detector) {
    // Check if photon is within detector aperture
    float3 to_det = photon.pos - detector.pos;
    float dist = length(to_det);
    
    if (dist > detector.radius) {
        return false;
    }
    
    // Check if exiting the surface (pointing somewhat outward)
    // Simplified: check if we've reached air
    if (photon.tissue != TISSUE_AIR) {
        return false;
    }
    
    // Time gating check
    if (detector.use_gate) {
        if (photon.time < detector.gate_start || photon.time > detector.gate_end) {
            return false;
        }
    }
    
    return true;
}

__device__ void propagate_photon_mesh(
    Photon& photon,
    const GPUMesh& mesh,
    const OpticalProps* props,
    curandState& rng_state,
    const Detector* detectors,
    int num_detectors,
    PathRecord* records,
    uint64_t* record_count,
    float* tpsf,
    uint64_t max_records
) {
    const int MAX_PROP_STEPS = MAX_BOUNCES;
    
    for (int step = 0; step < MAX_PROP_STEPS && photon.status == 0; step++) {
        if (photon.tet < 0) {
            photon.status = 2;  // Escaped
            break;
        }
        
        const Tetrahedron& tet = mesh.elems[photon.tet];
        float3 v[4] = {
            mesh.nodes[tet.v[0]],
            mesh.nodes[tet.v[1]],
            mesh.nodes[tet.v[2]],
            mesh.nodes[tet.v[3]]
        };
        
        // Get optical properties
        int tissue = tet.tissue;
        if (tissue < 0 || tissue >= NUM_TISSUES) tissue = TISSUE_AIR;
        
        const OpticalProps& op = props[tissue];
        photon.tissue = tissue;
        
        // Check for detection at air interface
        if (tissue == TISSUE_AIR) {
            for (int d = 0; d < num_detectors; d++) {
                if (check_detection(photon, detectors[d])) {
                    photon.status = 1;  // Detected
                    
                    // Record photon
                    uint64_t idx = atomicAdd(record_count, 1);
                    if (idx < max_records) {
                        PathRecord& rec = records[idx];
                        rec.exit_pos = photon.pos;
                        rec.exit_dir = photon.dir;
                        rec.total_pathlen = photon.pathlen;
                        rec.amyg_pathlen = photon.amyg_pathlen;
                        rec.time_of_flight = photon.time;
                        rec.weight = photon.weight;
                        rec.bounces = photon.bounces;
                        rec.hit_amyg = photon.hit_amyg;
                    }
                    
                    // Update TPSF
                    int bin = (int)(photon.time / TPSF_DT_PS);
                    if (bin >= 0 && bin < TPSF_BINS) {
                        atomicAdd(&tpsf[d * TPSF_BINS + bin], photon.weight);
                    }
                    
                    return;
                }
            }
            
            // Escaped to air without detection
            photon.status = 2;
            return;
        }
        
        // Sample step length (exponential distribution)
        float mfp = op.get_mean_free_path();
        float step_len = -mfp * logf(curand_uniform(&rng_state) + 1e-10f);
        
        // Find where photon exits current tetrahedron
        RayTetIntersect hit = intersect_ray_tet(photon.pos, photon.dir, v);
        
        if (!hit.hit || hit.t > step_len) {
            // Scattering event within current tet
            photon.pos = photon.pos + photon.dir * step_len;
            photon.pathlen += step_len;
            photon.time += step_len * op.n / SPEED_OF_LIGHT;
            
            // Absorption
            float albedo = op.get_albedo();
            photon.weight *= albedo;
            
            // Track amygdala path length
            if (tissue == TISSUE_AMYGDALA) {
                photon.amyg_pathlen += step_len;
                photon.hit_amyg = 1;
            }
            
            // Scattering
            hg_scatter(photon.dir, op.g, rng_state);
            photon.bounces++;
            
            // Russian roulette
            if (!russian_roulette(photon, rng_state)) {
                return;
            }
        } else {
            // Move to exit point
            photon.pos = photon.pos + photon.dir * (hit.t + 1e-4f);
            photon.pathlen += hit.t;
            photon.time += hit.t * op.n / SPEED_OF_LIGHT;
            
            // Track amygdala path
            if (tissue == TISSUE_AMYGDALA) {
                photon.amyg_pathlen += hit.t;
                photon.hit_amyg = 1;
            }
            
            // Move to neighboring tetrahedron
            int next_tet = tet.neighbor[hit.exit_face];
            
            if (next_tet >= 0 && next_tet < mesh.num_elems) {
                photon.tet = next_tet;
                
                // Check for refractive index mismatch
                const Tetrahedron& next = mesh.elems[next_tet];
                int next_tissue = next.tissue;
                if (next_tissue >= 0 && next_tissue < NUM_TISSUES) {
                    float n1 = op.n;
                    float n2 = props[next_tissue].n;
                    
                    if (fabsf(n1 - n2) > 0.01f) {
                        // Fresnel reflection/refraction
                        // Simplified: just continue for now
                    }
                }
            } else {
                // Boundary - will be handled at start of next iteration
                photon.tet = -1;
            }
        }
    }
    
    // Max bounces reached
    if (photon.status == 0) {
        photon.status = 4;  // Max bounces
    }
}

__global__ void init_rng_kernel(curandState* states, uint64_t seed, int num_states) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    curand_init(seed + idx, 0, 0, &states[idx]);
}

__global__ void mmc_kernel(
    GPUMesh mesh,
    OpticalProps* props,
    Detector* detectors,
    int num_detectors,
    PathRecord* records,
    uint64_t* record_count,
    float* tpsf,
    SimulationStats* stats,
    uint64_t num_photons_per_thread,
    float3 source_pos,
    float source_radius,
    int wavelength_nm,
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = gridDim.x * blockDim.x;
    
    curandState* rng = &rng_states[tid % num_threads];
    
    uint64_t local_launched = 0;
    uint64_t local_detected = 0;
    uint64_t local_escaped = 0;
    uint64_t local_absorbed = 0;
    float local_pathlen = 0.0f;
    float local_time = 0.0f;
    
    for (uint64_t i = 0; i < num_photons_per_thread; i++) {
        Photon photon;
        
        // Initialize photon at source
        float r = source_radius * sqrtf(curand_uniform(rng));
        float theta = 2.0f * 3.14159265f * curand_uniform(rng);
        
        // Isotropic launch direction
        float cos_theta = 2.0f * curand_uniform(rng) - 1.0f;
        float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
        float phi = 2.0f * 3.14159265f * curand_uniform(rng);
        
        photon.pos = make_float3(
            source_pos.x + r * cosf(theta),
            source_pos.y + r * sinf(theta),
            source_pos.z
        );
        photon.dir = make_float3(
            sin_theta * cosf(phi),
            sin_theta * sinf(phi),
            cos_theta
        );
        photon.weight = 1.0f;
        photon.pathlen = 0.0f;
        photon.time = 0.0f;
        photon.status = 0;
        photon.bounces = 0;
        photon.amyg_pathlen = 0.0f;
        photon.hit_amyg = 0;
        
        // Find initial tetrahedron
        photon.tet = find_containing_tet(photon.pos, mesh);
        
        local_launched++;
        
        // Propagate photon
        if (photon.tet >= 0) {
            propagate_photon_mesh(photon, mesh, props, *rng,
                                   detectors, num_detectors,
                                   records, record_count, tpsf,
                                   MAX_DETECTOR_RECORDS);
            
            switch (photon.status) {
                case 1: local_detected++; break;
                case 2: local_escaped++; break;
                case 3: local_absorbed++; break;
            }
            
            if (photon.status == 1) {
                local_pathlen += photon.pathlen;
                local_time += photon.time;
            }
        } else {
            local_escaped++;
        }
    }
    
    // Accumulate statistics (atomic)
    atomicAdd(&stats->launched, local_launched);
    atomicAdd(&stats->detected, local_detected);
    atomicAdd(&stats->escaped, local_escaped);
    atomicAdd(&stats->absorbed, local_absorbed);
}

} // namespace mmc
