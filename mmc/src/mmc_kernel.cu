#include "mmc_kernel.cuh"
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Portable double atomicAdd
// ---------------------------------------------------------------------------
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 600
__device__ double atomicAddDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#else
__device__ __forceinline__ double atomicAddDouble(double* address, double val) {
    return atomicAdd(address, val);
}
#endif

// ---------------------------------------------------------------------------
// Device constants
// ---------------------------------------------------------------------------
__constant__ MMCConfig   d_mmc_config;
__constant__ Detector    d_mmc_dets[128];
__constant__ int         d_mmc_n_dets;

// Path recording counters
__device__ int d_mmc_path_count[128];
__device__ int d_mmc_total_paths;

// ---------------------------------------------------------------------------
// Henyey-Greenstein scattering: sample new direction
// ---------------------------------------------------------------------------
__device__ void hg_scatter(float g, float* dx, float* dy, float* dz,
                           curandState* rng) {
    float xi1 = curand_uniform(rng);
    float xi2 = curand_uniform(rng);

    float cos_theta;
    if (fabsf(g) < 1e-6f) {
        cos_theta = 2.0f * xi1 - 1.0f;
    } else {
        float tmp = (1.0f - g * g) / (1.0f - g + 2.0f * g * xi1);
        cos_theta = (1.0f + g * g - tmp * tmp) / (2.0f * g);
    }

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2.0f * 3.14159265358979f * xi2;
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    float ux = *dx, uy = *dy, uz = *dz;

    if (fabsf(uz) > 0.99999f) {
        *dx = sin_theta * cos_phi;
        *dy = sin_theta * sin_phi;
        *dz = copysignf(cos_theta, uz);
    } else {
        float inv_sin = rsqrtf(1.0f - uz * uz);
        *dx = sin_theta * (ux * uz * cos_phi - uy * sin_phi) * inv_sin + ux * cos_theta;
        *dy = sin_theta * (uy * uz * cos_phi + ux * sin_phi) * inv_sin + uy * cos_theta;
        *dz = -sin_theta * cos_phi / inv_sin + uz * cos_theta;
    }

    float norm = rsqrtf((*dx)*(*dx) + (*dy)*(*dy) + (*dz)*(*dz));
    *dx *= norm; *dy *= norm; *dz *= norm;
}

// ---------------------------------------------------------------------------
// Fresnel reflection coefficient
// ---------------------------------------------------------------------------
__device__ float fresnel_reflect(float n_in, float n_out, float cos_i) {
    float sin_i2 = 1.0f - cos_i * cos_i;
    float ratio = n_in / n_out;
    float sin_t2 = ratio * ratio * sin_i2;

    if (sin_t2 >= 1.0f) return 1.0f; // TIR

    float cos_t = sqrtf(1.0f - sin_t2);
    float Rs = (n_in * cos_i - n_out * cos_t) / (n_in * cos_i + n_out * cos_t);
    float Rp = (n_out * cos_i - n_in * cos_t) / (n_out * cos_i + n_in * cos_t);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ---------------------------------------------------------------------------
// Refract direction via Snell's law
// Given incident direction d, outward normal n, and refractive index ratio
// Returns refracted direction. Assumes cos_i > 0 (photon entering face).
// ---------------------------------------------------------------------------
__device__ void snell_refract(float* dx, float* dy, float* dz,
                              float nx, float ny, float nz,
                              float n_ratio, float cos_i) {
    float sin_t2 = n_ratio * n_ratio * (1.0f - cos_i * cos_i);
    float cos_t = sqrtf(1.0f - sin_t2);

    // d_refracted = n_ratio * d + (n_ratio * cos_i - cos_t) * n
    // Here n points outward; if photon is entering (cos_i > 0 means d·(-n) > 0),
    // we need the inward normal
    float factor = n_ratio * cos_i - cos_t;
    *dx = n_ratio * (*dx) - factor * nx;
    *dy = n_ratio * (*dy) - factor * ny;
    *dz = n_ratio * (*dz) - factor * nz;

    float norm = rsqrtf((*dx)*(*dx) + (*dy)*(*dy) + (*dz)*(*dz));
    *dx *= norm; *dy *= norm; *dz *= norm;
}

// ---------------------------------------------------------------------------
// Check detector acceptance (same as voxel MC)
// ---------------------------------------------------------------------------
__device__ int check_detectors_mmc(float x, float y, float z,
                                   float ddx, float ddy, float ddz) {
    for (int i = 0; i < d_mmc_n_dets; i++) {
        float diffx = x - d_mmc_dets[i].x;
        float diffy = y - d_mmc_dets[i].y;
        float diffz = z - d_mmc_dets[i].z;
        float dist2 = diffx*diffx + diffy*diffy + diffz*diffz;
        float r = d_mmc_dets[i].radius;
        if (dist2 > r * r) continue;

        float dot_product = ddx * d_mmc_dets[i].nx + ddy * d_mmc_dets[i].ny + ddz * d_mmc_dets[i].nz;
        if (dot_product < d_mmc_dets[i].n_critical) continue;

        return i;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Time gate assignment
// ---------------------------------------------------------------------------
__device__ __forceinline__ int get_time_gate(float tof_ps) {
    if (tof_ps < 500.0f)  return 0;
    if (tof_ps < 1000.0f) return 1;
    if (tof_ps < 1500.0f) return 2;
    if (tof_ps < 2000.0f) return 3;
    if (tof_ps < 2500.0f) return 4;
    if (tof_ps < 3000.0f) return 5;
    if (tof_ps < 3500.0f) return 6;
    if (tof_ps < 4000.0f) return 7;
    if (tof_ps < 5000.0f) return 8;
    return 9;
}

// ---------------------------------------------------------------------------
// Record a detected photon (CW + TD)
// ---------------------------------------------------------------------------
__device__ void record_detection_mmc(
    int det_id, float weight, float total_pl, float* ppl,
    double* det_weight, double* det_pathlength, double* det_partial_pl,
    unsigned long long* det_count,
    double* det_tpsf, double* det_gated_weight,
    double* det_gated_partial_pl, unsigned long long* det_gated_count)
{
    atomicAddDouble(&det_weight[det_id], (double)weight);
    atomicAddDouble(&det_pathlength[det_id], (double)(weight * total_pl));
    atomicAdd(&det_count[det_id], 1ULL);
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        atomicAddDouble(&det_partial_pl[det_id * NUM_TISSUE_TYPES + t],
                  (double)(weight * ppl[t]));
    }

    // Time of flight from optical pathlength
    float opl = 0.0f;
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        opl += ppl[t] * d_mmc_config.tissue[t].n;
    }
    float tof_ps = opl / C_VACUUM_MM_PS;

    // TPSF histogram
    int tbin = __float2int_rd(tof_ps / TPSF_BIN_PS);
    if (tbin >= 0 && tbin < TPSF_BINS) {
        atomicAddDouble(&det_tpsf[det_id * TPSF_BINS + tbin], (double)weight);
    }

    // Time-gated accumulation
    int gate = get_time_gate(tof_ps);
    if (gate >= 0 && gate < NUM_TIME_GATES) {
        atomicAddDouble(&det_gated_weight[det_id * NUM_TIME_GATES + gate], (double)weight);
        atomicAdd(&det_gated_count[det_id * NUM_TIME_GATES + gate], 1ULL);
        for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
            atomicAddDouble(&det_gated_partial_pl[(det_id * NUM_TIME_GATES + gate) * NUM_TISSUE_TYPES + t],
                      (double)(weight * ppl[t]));
        }
    }
}

// ---------------------------------------------------------------------------
// Point-in-tet test using barycentric coordinates (sign of determinants)
// Returns true if point (px,py,pz) is inside tet with vertices v0..v3
// ---------------------------------------------------------------------------
__device__ bool point_in_tet(float px, float py, float pz,
                             const float* __restrict__ nodes,
                             const int* __restrict__ elem) {
    // Load 4 vertices
    float v0x = nodes[elem[0]*3+0], v0y = nodes[elem[0]*3+1], v0z = nodes[elem[0]*3+2];
    float v1x = nodes[elem[1]*3+0], v1y = nodes[elem[1]*3+1], v1z = nodes[elem[1]*3+2];
    float v2x = nodes[elem[2]*3+0], v2y = nodes[elem[2]*3+1], v2z = nodes[elem[2]*3+2];
    float v3x = nodes[elem[3]*3+0], v3y = nodes[elem[3]*3+1], v3z = nodes[elem[3]*3+2];

    // Compute the determinant sign for the tet orientation
    float d0x = v1x-v0x, d0y = v1y-v0y, d0z = v1z-v0z;
    float d1x = v2x-v0x, d1y = v2y-v0y, d1z = v2z-v0z;
    float d2x = v3x-v0x, d2y = v3y-v0y, d2z = v3z-v0z;

    float det_tet = d0x*(d1y*d2z - d1z*d2y)
                  - d0y*(d1x*d2z - d1z*d2x)
                  + d0z*(d1x*d2y - d1y*d2x);

    float sign = (det_tet >= 0.0f) ? 1.0f : -1.0f;

    // For the point to be inside, all 4 sub-determinants must have the same sign
    // Sub-det 0: replace v0 with p
    float p0x = v1x-px, p0y = v1y-py, p0z = v1z-pz;
    float p1x = v2x-px, p1y = v2y-py, p1z = v2z-pz;
    float p2x = v3x-px, p2y = v3y-py, p2z = v3z-pz;
    float s0 = p0x*(p1y*p2z - p1z*p2y) - p0y*(p1x*p2z - p1z*p2x) + p0z*(p1x*p2y - p1y*p2x);
    if (s0 * sign < -1e-6f) return false;

    // Sub-det 1: replace v1 with p
    p0x = v0x-px; p0y = v0y-py; p0z = v0z-pz;
    // p1 = v2-p (already computed)
    // p2 = v3-p (already computed)
    float s1 = p0x*(p1y*p2z - p1z*p2y) - p0y*(p1x*p2z - p1z*p2x) + p0z*(p1x*p2y - p1y*p2x);
    if (s1 * sign > 1e-6f) return false;  // opposite sign convention for odd permutation

    // Sub-det 2: replace v2 with p
    // p0 = v0-p (already computed)
    p1x = v1x-px; p1y = v1y-py; p1z = v1z-pz;
    // p2 = v3-p (already computed)
    float s2 = p0x*(p1y*p2z - p1z*p2y) - p0y*(p1x*p2z - p1z*p2x) + p0z*(p1x*p2y - p1y*p2x);
    if (s2 * sign < -1e-6f) return false;

    // Sub-det 3: replace v3 with p
    // p0 = v0-p (already computed)
    // p1 = v1-p (already computed)
    p2x = v2x-px; p2y = v2y-py; p2z = v2z-pz;
    float s3 = p0x*(p1y*p2z - p1z*p2y) - p0y*(p1x*p2z - p1z*p2x) + p0z*(p1x*p2y - p1y*p2x);
    if (s3 * sign > 1e-6f) return false;

    return true;
}

// ---------------------------------------------------------------------------
// Find enclosing tetrahedron via uniform grid lookup
// ---------------------------------------------------------------------------
__device__ int find_enclosing_tet(float px, float py, float pz,
                                  const float* __restrict__ nodes,
                                  const int* __restrict__ elements,
                                  const int* __restrict__ grid_offsets,
                                  const int* __restrict__ grid_counts,
                                  const int* __restrict__ grid_tets,
                                  const float* grid_bbox_min,
                                  const float* grid_cell_size) {
    int ix = __float2int_rd((px - grid_bbox_min[0]) / grid_cell_size[0]);
    int iy = __float2int_rd((py - grid_bbox_min[1]) / grid_cell_size[1]);
    int iz = __float2int_rd((pz - grid_bbox_min[2]) / grid_cell_size[2]);

    if (ix < 0 || ix >= GRID_RES || iy < 0 || iy >= GRID_RES || iz < 0 || iz >= GRID_RES)
        return -1;

    int cell = iz * GRID_RES * GRID_RES + iy * GRID_RES + ix;
    int offset = grid_offsets[cell];
    int count  = grid_counts[cell];

    for (int i = 0; i < count; i++) {
        int tet_id = grid_tets[offset + i];
        if (point_in_tet(px, py, pz, nodes, &elements[tet_id * 4]))
            return tet_id;
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Ray-tetrahedron exit: find which face the ray exits through
// Uses precomputed face plane equations (normal + plane constant).
// For a convex tet with ray starting inside, the exit face is the one
// with the smallest positive ray-plane intersection distance.
// ---------------------------------------------------------------------------
__device__ float ray_tet_exit(float px, float py, float pz,
                              float dx, float dy, float dz,
                              int tet_id, int entry_face,
                              const float* __restrict__ face_normals,
                              const float* __restrict__ face_d,
                              int* exit_face) {
    float min_t = 1e30f;
    *exit_face = -1;

    int base = tet_id * 4;

    for (int f = 0; f < 4; f++) {
        if (f == entry_face) continue;

        int idx = base + f;
        float nx = face_normals[idx * 3 + 0];
        float ny = face_normals[idx * 3 + 1];
        float nz = face_normals[idx * 3 + 2];

        // dot(normal, dir): positive means ray heading outward through this face
        float denom = nx * dx + ny * dy + nz * dz;
        if (denom < 1e-12f) continue;

        // distance from point to face plane
        float numer = face_d[idx] - (nx * px + ny * py + nz * pz);
        float t = numer / denom;

        if (t > 1e-8f && t < min_t) {
            min_t = t;
            *exit_face = f;
        }
    }

    return min_t;
}

// ---------------------------------------------------------------------------
// Main MMC kernel: one thread simulates multiple photons
// ---------------------------------------------------------------------------
__global__ void mmc_kernel(
    // Mesh data (device pointers)
    const float* __restrict__ nodes,
    const int*   __restrict__ elements,
    const int*   __restrict__ tissue,
    const int*   __restrict__ neighbors,
    const float* __restrict__ face_normals,
    const float* __restrict__ face_d,
    const int*   __restrict__ face_pair,
    int num_elements,
    // Grid accelerator
    const int*   __restrict__ grid_offsets,
    const int*   __restrict__ grid_counts,
    const int*   __restrict__ grid_tets,
    float grid_bbox_min_x, float grid_bbox_min_y, float grid_bbox_min_z,
    float grid_cell_size_x, float grid_cell_size_y, float grid_cell_size_z,
    // Detector accumulators
    double*              __restrict__ det_weight,
    double*              __restrict__ det_pathlength,
    double*              __restrict__ det_partial_pl,
    unsigned long long*  __restrict__ det_count,
    double*              __restrict__ det_tpsf,
    double*              __restrict__ det_gated_weight,
    double*              __restrict__ det_gated_partial_pl,
    unsigned long long*  __restrict__ det_gated_count,
    // Path recording
    float*               __restrict__ path_pos,
    int*                 __restrict__ path_det,
    int*                 __restrict__ path_len,
    // Photon budget
    uint64_t photons_per_thread,
    uint64_t seed_offset
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    curandState rng;
    curand_init(12345, seed_offset + tid, 0, &rng);

    float grid_bmin[3] = { grid_bbox_min_x, grid_bbox_min_y, grid_bbox_min_z };
    float grid_cs[3]   = { grid_cell_size_x, grid_cell_size_y, grid_cell_size_z };

    for (uint64_t p = 0; p < photons_per_thread; p++) {
        // --- Launch photon ---
        float px = d_mmc_config.src_x;
        float py = d_mmc_config.src_y;
        float pz = d_mmc_config.src_z;
        float ddx = d_mmc_config.src_dx;
        float ddy = d_mmc_config.src_dy;
        float ddz = d_mmc_config.src_dz;

        // Beam spread (uniform disk sampling)
        if (d_mmc_config.beam_radius > 0.0f) {
            float r = d_mmc_config.beam_radius * sqrtf(curand_uniform(&rng));
            float theta = 2.0f * 3.14159265f * curand_uniform(&rng);
            float tx, ty, tz, bx, by, bz;
            if (fabsf(ddz) < 0.9f) {
                tx = ddy; ty = -ddx; tz = 0.0f;
            } else {
                tx = 0.0f; ty = ddz; tz = -ddy;
            }
            float tmag = rsqrtf(tx*tx + ty*ty + tz*tz);
            tx *= tmag; ty *= tmag; tz *= tmag;
            bx = ddy*tz - ddz*ty;
            by = ddz*tx - ddx*tz;
            bz = ddx*ty - ddy*tx;
            px += r * (cosf(theta)*tx + sinf(theta)*bx);
            py += r * (cosf(theta)*ty + sinf(theta)*by);
            pz += r * (cosf(theta)*tz + sinf(theta)*bz);
        }

        float weight = 1.0f;
        float ppl[NUM_TISSUE_TYPES];
        for (int t = 0; t < NUM_TISSUE_TYPES; t++) ppl[t] = 0.0f;
        float total_pl = 0.0f;

        // Find initial enclosing tetrahedron
        int current_tet = find_enclosing_tet(px, py, pz,
            nodes, elements, grid_offsets, grid_counts, grid_tets,
            grid_bmin, grid_cs);

        if (current_tet < 0) continue; // source not inside mesh

        int entry_face = -1; // no entry face for first tet

        // Path recording
        int path_slot = -1;
        int path_step = 0;
        bool should_record = false;
        if (curand_uniform(&rng) < 2e-5f) {
            int slot = atomicAdd(&d_mmc_total_paths, 1);
            if (slot < MAX_RECORDED_PATHS) {
                path_slot = slot;
                should_record = true;
                int base = slot * MAX_PATH_STEPS * 3;
                path_pos[base + 0] = px;
                path_pos[base + 1] = py;
                path_pos[base + 2] = pz;
                path_step = 1;
            }
        }

        // --- Transport loop ---
        for (int step = 0; step < 500000; step++) {
            int ttype = tissue[current_tet];
            OpticalProps op = d_mmc_config.tissue[ttype];
            float mu_t = op.mu_a + op.mu_s;

            // Sample scattering distance
            float s_remain;
            if (mu_t < 1e-10f) {
                // Nearly transparent (e.g., CSF): use large step
                s_remain = 100.0f;
            } else {
                s_remain = -logf(curand_uniform(&rng) + 1e-30f) / mu_t;
            }

            // Traverse through tetrahedra until scattering event
            bool scattered = false;
            int boundary_crossings = 0;
            const int MAX_BOUNDARY_CROSSINGS = 10000;  // Safety limit
            while (s_remain > 1e-8f && boundary_crossings < MAX_BOUNDARY_CROSSINGS) {
                // Find exit face and distance
                int exit_face;
                float d_exit = ray_tet_exit(px, py, pz, ddx, ddy, ddz,
                                            current_tet, entry_face,
                                            face_normals, face_d, &exit_face);

                if (exit_face < 0) {
                    // Degenerate tet or numerical issue — terminate photon
                    s_remain = 0.0f;
                    weight = 0.0f;
                    break;
                }

                if (s_remain < d_exit) {
                    // Scatter within this element
                    px += ddx * s_remain;
                    py += ddy * s_remain;
                    pz += ddz * s_remain;
                    ppl[ttype] += s_remain;
                    total_pl += s_remain;

                    // Absorb weight
                    if (mu_t > 1e-10f) {
                        float absorbed = weight * (op.mu_a / mu_t);
                        weight -= absorbed;
                    }

                    // Scatter
                    hg_scatter(op.g, &ddx, &ddy, &ddz, &rng);
                    entry_face = -1; // after scattering, no entry face
                    scattered = true;
                    s_remain = 0.0f;

                    // Record path
                    if (should_record && path_step < MAX_PATH_STEPS) {
                        int base = path_slot * MAX_PATH_STEPS * 3;
                        path_pos[base + path_step*3 + 0] = px;
                        path_pos[base + path_step*3 + 1] = py;
                        path_pos[base + path_step*3 + 2] = pz;
                        path_step++;
                    }
                    break;
                }

                // Move to face boundary
                px += ddx * d_exit;
                py += ddy * d_exit;
                pz += ddz * d_exit;
                ppl[ttype] += d_exit;
                total_pl += d_exit;
                boundary_crossings++;

                // Absorb proportional weight for this sub-step
                if (mu_t > 1e-10f) {
                    float frac = d_exit / (d_exit + s_remain);
                    float absorbed = weight * (op.mu_a / mu_t) * frac;
                    weight -= absorbed;
                }

                s_remain -= d_exit;

                // Get neighbor across exit face
                int neighbor = neighbors[current_tet * 4 + exit_face];

                if (neighbor < 0) {
                    // External boundary — photon exits mesh
                    // Get face outward normal for detector check
                    int nidx = current_tet * 4 + exit_face;
                    float fnx = face_normals[nidx*3+0];
                    float fny = face_normals[nidx*3+1];
                    float fnz = face_normals[nidx*3+2];

                    float n_in = op.n;
                    float n_out = 1.0f; // air

                    float cos_i = -(ddx*fnx + ddy*fny + ddz*fnz);
                    if (cos_i < 0.0f) cos_i = -cos_i;

                    float R = fresnel_reflect(n_in, n_out, cos_i);
                    if (curand_uniform(&rng) < R) {
                        // Reflect back into current tet
                        float d_dot_n = ddx*fnx + ddy*fny + ddz*fnz;
                        ddx -= 2.0f * d_dot_n * fnx;
                        ddy -= 2.0f * d_dot_n * fny;
                        ddz -= 2.0f * d_dot_n * fnz;
                        float norm = rsqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                        ddx *= norm; ddy *= norm; ddz *= norm;

                        // Nudge back inside
                        px -= fnx * 1e-4f;
                        py -= fny * 1e-4f;
                        pz -= fnz * 1e-4f;

                        entry_face = exit_face;
                        continue; // continue traversal with remaining s
                    }

                    // Transmitted — check detectors
                    int det_id = check_detectors_mmc(px, py, pz, ddx, ddy, ddz);
                    if (det_id >= 0) {
                        record_detection_mmc(det_id, weight, total_pl, ppl,
                            det_weight, det_pathlength, det_partial_pl, det_count,
                            det_tpsf, det_gated_weight, det_gated_partial_pl, det_gated_count);
                        if (should_record && path_step > 1) {
                            int det_cnt = atomicAdd(&d_mmc_path_count[det_id], 1);
                            if (det_cnt < PATHS_PER_DET) {
                                path_det[path_slot] = det_id;
                                path_len[path_slot] = path_step;
                            } else {
                                path_len[path_slot] = 0;
                            }
                        }
                    } else if (should_record) {
                        path_len[path_slot] = 0;
                    }
                    weight = 0.0f;
                    s_remain = 0.0f;
                    break;
                }

                // Internal boundary — check refractive index mismatch
                int next_ttype = tissue[neighbor];
                float n_in = op.n;
                float n_out = d_mmc_config.tissue[next_ttype].n;

                if (fabsf(n_in - n_out) > 1e-5f) {
                    int nidx = current_tet * 4 + exit_face;
                    float fnx = face_normals[nidx*3+0];
                    float fny = face_normals[nidx*3+1];
                    float fnz = face_normals[nidx*3+2];

                    float cos_i = -(ddx*fnx + ddy*fny + ddz*fnz);
                    if (cos_i < 0.0f) cos_i = -cos_i;

                    float R = fresnel_reflect(n_in, n_out, cos_i);
                    if (curand_uniform(&rng) < R) {
                        // Reflect
                        float d_dot_n = ddx*fnx + ddy*fny + ddz*fnz;
                        ddx -= 2.0f * d_dot_n * fnx;
                        ddy -= 2.0f * d_dot_n * fny;
                        ddz -= 2.0f * d_dot_n * fnz;
                        float norm = rsqrtf(ddx*ddx + ddy*ddy + ddz*ddz);
                        ddx *= norm; ddy *= norm; ddz *= norm;

                        px -= fnx * 1e-4f;
                        py -= fny * 1e-4f;
                        pz -= fnz * 1e-4f;

                        entry_face = exit_face;
                        continue;
                    }

                    // Refract (Snell's law)
                    int nidx2 = current_tet * 4 + exit_face;
                    float rnx = face_normals[nidx2*3+0];
                    float rny = face_normals[nidx2*3+1];
                    float rnz = face_normals[nidx2*3+2];
                    snell_refract(&ddx, &ddy, &ddz, rnx, rny, rnz, n_in/n_out, cos_i);

                    // Adjust remaining scattering distance for new medium
                    float new_mu_t = d_mmc_config.tissue[next_ttype].mu_a +
                                     d_mmc_config.tissue[next_ttype].mu_s;
                    if (new_mu_t > 1e-10f && mu_t > 1e-10f) {
                        s_remain *= mu_t / new_mu_t;
                    }
                }

                // Cross into neighbor
                // The entry face in the neighbor is the face shared with current_tet
                // Use precomputed face_pair for O(1) lookup (Fix B)
                int new_entry_face = face_pair[current_tet * 4 + exit_face];
                
                // Safety check: new_entry_face should be 0-3 for valid interior faces
                if (new_entry_face < 0 || new_entry_face > 3) {
                    // Invalid entry face - this shouldn't happen for interior faces
                    // Terminate photon to avoid infinite loop
                    weight = 0.0f;
                    break;
                }

                current_tet = neighbor;
                entry_face = new_entry_face;
                ttype = tissue[current_tet];
                op = d_mmc_config.tissue[ttype];
                mu_t = op.mu_a + op.mu_s;

                // Nudge slightly into new tet to avoid re-triggering same face
                px += ddx * 1e-5f;
                py += ddy * 1e-5f;
                pz += ddz * 1e-5f;
            }

            if (weight <= 0.0f) break;

            // Russian roulette
            if (weight < d_mmc_config.weight_threshold) {
                if (curand_uniform(&rng) < (1.0f / (float)d_mmc_config.roulette_m)) {
                    weight *= (float)d_mmc_config.roulette_m;
                } else {
                    if (should_record) path_len[path_slot] = 0;
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void launch_mmc_simulation(
    const MMCDeviceData& dev,
    const MMCConfig& config,
    const Detector* h_dets,
    int n_dets,
    DetectorResult* h_results,
    double* h_tpsf,
    double* h_gated_weight,
    double* h_gated_partial_pl,
    uint64_t* h_gated_count,
    float* h_path_pos,
    int*   h_path_det,
    int*   h_path_len,
    int*   h_num_paths
) {
    // Upload config to constant memory
    cudaMemcpyToSymbol(d_mmc_config, &config, sizeof(MMCConfig));
    cudaMemcpyToSymbol(d_mmc_dets, h_dets, n_dets * sizeof(Detector));
    cudaMemcpyToSymbol(d_mmc_n_dets, &n_dets, sizeof(int));

    // Allocate CW detector accumulators
    double*              d_det_weight;
    double*              d_det_pathlength;
    double*              d_det_partial_pl;
    unsigned long long*  d_det_count;

    cudaMalloc(&d_det_weight,     n_dets * sizeof(double));
    cudaMalloc(&d_det_pathlength, n_dets * sizeof(double));
    cudaMalloc(&d_det_partial_pl, n_dets * NUM_TISSUE_TYPES * sizeof(double));
    cudaMalloc(&d_det_count,      n_dets * sizeof(unsigned long long));

    cudaMemset(d_det_weight,     0, n_dets * sizeof(double));
    cudaMemset(d_det_pathlength, 0, n_dets * sizeof(double));
    cudaMemset(d_det_partial_pl, 0, n_dets * NUM_TISSUE_TYPES * sizeof(double));
    cudaMemset(d_det_count,      0, n_dets * sizeof(unsigned long long));

    // Allocate TD accumulators
    double*              d_det_tpsf;
    double*              d_det_gated_weight;
    double*              d_det_gated_partial_pl;
    unsigned long long*  d_det_gated_count;

    cudaMalloc(&d_det_tpsf,             n_dets * TPSF_BINS * sizeof(double));
    cudaMalloc(&d_det_gated_weight,     n_dets * NUM_TIME_GATES * sizeof(double));
    cudaMalloc(&d_det_gated_partial_pl, n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES * sizeof(double));
    cudaMalloc(&d_det_gated_count,      n_dets * NUM_TIME_GATES * sizeof(unsigned long long));

    cudaMemset(d_det_tpsf,             0, n_dets * TPSF_BINS * sizeof(double));
    cudaMemset(d_det_gated_weight,     0, n_dets * NUM_TIME_GATES * sizeof(double));
    cudaMemset(d_det_gated_partial_pl, 0, n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES * sizeof(double));
    cudaMemset(d_det_gated_count,      0, n_dets * NUM_TIME_GATES * sizeof(unsigned long long));

    // Allocate path recording buffers
    float*  d_path_pos;
    int*    d_path_det;
    int*    d_path_len;
    size_t path_pos_bytes = (size_t)MAX_RECORDED_PATHS * MAX_PATH_STEPS * 3 * sizeof(float);
    cudaMalloc(&d_path_pos, path_pos_bytes);
    cudaMalloc(&d_path_det, MAX_RECORDED_PATHS * sizeof(int));
    cudaMalloc(&d_path_len, MAX_RECORDED_PATHS * sizeof(int));
    cudaMemset(d_path_pos, 0, path_pos_bytes);
    cudaMemset(d_path_det, 0, MAX_RECORDED_PATHS * sizeof(int));
    cudaMemset(d_path_len, 0, MAX_RECORDED_PATHS * sizeof(int));

    // Zero path counters
    {
        int zero = 0;
        int zeros[128] = {};
        cudaMemcpyToSymbol(d_mmc_total_paths, &zero, sizeof(int));
        cudaMemcpyToSymbol(d_mmc_path_count, zeros, sizeof(zeros));
    }

    // Launch configuration
    int block_size = 256;
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    int total_threads = (max_threads / block_size) * block_size;
    int num_blocks = total_threads / block_size;
    if (num_blocks > 16384) num_blocks = 16384;

    uint64_t threads_total = (uint64_t)num_blocks * block_size;

    int num_batches = 20;
    uint64_t photons_per_batch = (config.num_photons + num_batches - 1) / num_batches;
    uint64_t photons_per_thread_per_batch =
        (photons_per_batch + threads_total - 1) / threads_total;
    uint64_t actual_per_batch = photons_per_thread_per_batch * threads_total;

    printf("  GPU: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);
    printf("  Launching %d blocks x %d threads = %llu threads\n",
           num_blocks, block_size, (unsigned long long)threads_total);
    printf("  Batches: %d x %llu photons/batch\n",
           num_batches, (unsigned long long)actual_per_batch);

    auto launch_start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; batch++) {
        uint64_t seed_off = (uint64_t)batch * threads_total;

        mmc_kernel<<<num_blocks, block_size>>>(
            dev.mesh.nodes, dev.mesh.elements, dev.mesh.tissue,
            dev.mesh.neighbors, dev.mesh.face_normals, dev.mesh.face_d, dev.mesh.face_pair,
            dev.mesh.num_elements,
            dev.grid.offsets, dev.grid.counts, dev.grid.tets,
            dev.grid.bbox_min[0], dev.grid.bbox_min[1], dev.grid.bbox_min[2],
            dev.grid.cell_size[0], dev.grid.cell_size[1], dev.grid.cell_size[2],
            d_det_weight, d_det_pathlength, d_det_partial_pl, d_det_count,
            d_det_tpsf, d_det_gated_weight, d_det_gated_partial_pl, d_det_gated_count,
            d_path_pos, d_path_det, d_path_len,
            photons_per_thread_per_batch, seed_off
        );

        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "\nCUDA error at batch %d: %s\n",
                    batch, cudaGetErrorString(err));
            break;
        }

        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - launch_start).count();
        uint64_t photons_done = (uint64_t)(batch + 1) * actual_per_batch;
        double pct = 100.0 * (batch + 1) / num_batches;
        double rate = photons_done / elapsed / 1e6;
        double eta = elapsed * (num_batches - batch - 1) / (batch + 1);

        int bar_width = 30;
        int filled = (int)(bar_width * pct / 100.0);
        printf("\r  [");
        for (int i = 0; i < bar_width; i++)
            printf("%s", i < filled ? "\xe2\x96\x88" : "\xe2\x96\x91");
        printf("] %5.1f%%  %.1f Mph/s", pct, rate);
        if (eta > 60.0)
            printf("  ETA %dm%02ds", (int)(eta / 60), (int)eta % 60);
        else
            printf("  ETA %ds", (int)eta);
        printf("    ");
        fflush(stdout);
    }
    printf("\n");

    // Copy CW results back
    std::vector<double>   h_det_weight(n_dets);
    std::vector<double>   h_det_pathlength(n_dets);
    std::vector<double>   h_det_partial_pl(n_dets * NUM_TISSUE_TYPES);
    std::vector<unsigned long long> h_det_count(n_dets);

    cudaMemcpy(h_det_weight.data(),     d_det_weight,     n_dets * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_det_pathlength.data(), d_det_pathlength, n_dets * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_det_partial_pl.data(), d_det_partial_pl, n_dets * NUM_TISSUE_TYPES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_det_count.data(),      d_det_count,      n_dets * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    for (int d = 0; d < n_dets; d++) {
        h_results[d].total_weight     = h_det_weight[d];
        h_results[d].total_pathlength = h_det_pathlength[d];
        h_results[d].num_detected     = h_det_count[d];
        for (int t = 0; t < NUM_TISSUE_TYPES; t++)
            h_results[d].partial_pathlength[t] = h_det_partial_pl[d * NUM_TISSUE_TYPES + t];
    }

    // Copy TD results back
    cudaMemcpy(h_tpsf, d_det_tpsf,
               n_dets * TPSF_BINS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_weight, d_det_gated_weight,
               n_dets * NUM_TIME_GATES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_partial_pl, d_det_gated_partial_pl,
               n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_count, d_det_gated_count,
               n_dets * NUM_TIME_GATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Copy path recording results
    int total_paths_recorded = 0;
    cudaMemcpyFromSymbol(&total_paths_recorded, d_mmc_total_paths, sizeof(int));
    if (total_paths_recorded > MAX_RECORDED_PATHS)
        total_paths_recorded = MAX_RECORDED_PATHS;
    *h_num_paths = total_paths_recorded;

    if (total_paths_recorded > 0) {
        cudaMemcpy(h_path_det, d_path_det,
                   total_paths_recorded * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_path_len, d_path_len,
                   total_paths_recorded * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_path_pos, d_path_pos,
                   (size_t)total_paths_recorded * MAX_PATH_STEPS * 3 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    }

    // Cleanup
    cudaFree(d_det_weight);
    cudaFree(d_det_pathlength);
    cudaFree(d_det_partial_pl);
    cudaFree(d_det_count);
    cudaFree(d_det_tpsf);
    cudaFree(d_det_gated_weight);
    cudaFree(d_det_gated_partial_pl);
    cudaFree(d_det_gated_count);
    cudaFree(d_path_pos);
    cudaFree(d_path_det);
    cudaFree(d_path_len);
}
