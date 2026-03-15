#include "mc_kernel.cuh"
#include <curand_kernel.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstring>

// ---------------------------------------------------------------------------
// Portable double atomicAdd (CAS fallback for archs without native support)
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
// Device constants (copied once per launch)
// ---------------------------------------------------------------------------
__constant__ SimConfig d_config;
__constant__ Detector  d_dets[128];   // max 128 detectors
__constant__ int       d_n_dets;

// 3D texture for cached volume access (spatial locality)
__constant__ cudaTextureObject_t d_vol_tex;  // set at kernel launch

// Path recording: per-detector counters
__device__ int d_path_count[128];          // how many paths recorded per detector
__device__ int d_total_paths;              // global path counter

// ---------------------------------------------------------------------------
// Helper: voxel index from position
// ---------------------------------------------------------------------------
__device__ __forceinline__
int pos_to_voxel(float x, float y, float z) {
    int ix = __float2int_rd(x / d_config.dx);
    int iy = __float2int_rd(y / d_config.dx);
    int iz = __float2int_rd(z / d_config.dx);
    if (ix < 0 || ix >= d_config.nx ||
        iy < 0 || iy >= d_config.ny ||
        iz < 0 || iz >= d_config.nz)
        return -1;
    return ix + iy * d_config.nx + iz * d_config.nx * d_config.ny;
}

// ---------------------------------------------------------------------------
// Cached volume lookup via 3D texture (exploits spatial locality)
// ---------------------------------------------------------------------------
__device__ __forceinline__
uint8_t get_tissue_cached(float x, float y, float z) {
    // Normalize to [0,1] texture coordinates
    float u = x / (d_config.nx * d_config.dx);
    float v = y / (d_config.ny * d_config.dx);
    float w = z / (d_config.nz * d_config.dx);
    
    // Texture automatically handles boundary clamping and caches spatial locality
    return (uint8_t)tex3D<uint8_t>(d_vol_tex, u, v, w);
}

// Fallback: direct global memory access (for non-textured builds)
__device__ __forceinline__
uint8_t get_tissue(const uint8_t* volume, float x, float y, float z) {
    int idx = pos_to_voxel(x, y, z);
    if (idx < 0) return TISSUE_AIR;
    return volume[idx];
}

// ---------------------------------------------------------------------------
// Henyey-Greenstein scattering: sample new direction
// ---------------------------------------------------------------------------
__device__ void henyey_greenstein(float g, float* dx, float* dy, float* dz,
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
    *dx *= norm;
    *dy *= norm;
    *dz *= norm;
}

// ---------------------------------------------------------------------------
// Fresnel reflection at tissue boundary
// ---------------------------------------------------------------------------
__device__ float fresnel_reflect(float n_in, float n_out, float cos_i) {
    float sin_i2 = 1.0f - cos_i * cos_i;
    float ratio = n_in / n_out;
    float sin_t2 = ratio * ratio * sin_i2;

    if (sin_t2 >= 1.0f) return 1.0f; // total internal reflection

    float cos_t = sqrtf(1.0f - sin_t2);
    float Rs = ((n_in * cos_i - n_out * cos_t) / (n_in * cos_i + n_out * cos_t));
    float Rp = ((n_out * cos_i - n_in * cos_t) / (n_out * cos_i + n_in * cos_t));
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ---------------------------------------------------------------------------
// Check if photon exits at a detector
// Rejects photons that exceed the critical angle (dot product < cos_critical)
// ---------------------------------------------------------------------------
__device__ int check_detectors(float x, float y, float z, float ddx, float ddy, float ddz) {
    for (int i = 0; i < d_n_dets; i++) {
        // Check spatial acceptance (within detector radius)
        float diffx = x - d_dets[i].x;
        float diffy = y - d_dets[i].y;
        float diffz = z - d_dets[i].z;
        float dist2 = diffx * diffx + diffy * diffy + diffz * diffz;
        float r = d_dets[i].radius;
        if (dist2 > r * r)
            continue;  // outside detector area
        
        // Check angular acceptance: photon direction vs detector surface normal
        // Photon must be coming from inside the head (aligned with normal)
        float dot_product = ddx * d_dets[i].nx + ddy * d_dets[i].ny + ddz * d_dets[i].nz;
        if (dot_product < d_dets[i].n_critical)
            continue;  // rejected: exceeds critical angle (grazing exit)
        
        return i;  // accepted
    }
    return -1;
}

// ---------------------------------------------------------------------------
// Time gate assignment
// ---------------------------------------------------------------------------
__device__ __forceinline__ int get_time_gate(float tof_ps) {
    // Fine resolution in 1.5-5 ns range where amygdala sensitivity lives
    // Gates: [0-500, 500-1000, 1000-1500, 1500-2000, 2000-2500,
    //         2500-3000, 3000-3500, 3500-4000, 4000-5000, 5000+] ps
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
// Record a detected photon (CW + TD accumulation)
// ---------------------------------------------------------------------------
__device__ void record_detection(
    int det_id, float weight, float total_pl, float* ppl,
    double* det_weight, double* det_pathlength, double* det_partial_pl,
    unsigned long long* det_count,
    double* det_tpsf, double* det_gated_weight,
    double* det_gated_partial_pl, unsigned long long* det_gated_count)
{
    // CW accumulation
    atomicAddDouble(&det_weight[det_id], (double)weight);
    atomicAddDouble(&det_pathlength[det_id], (double)(weight * total_pl));
    atomicAdd(&det_count[det_id], 1ULL);
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        atomicAddDouble(&det_partial_pl[det_id * NUM_TISSUE_TYPES + t],
                  (double)(weight * ppl[t]));
    }

    // Compute time of flight from optical pathlength
    float opl = 0.0f;
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        opl += ppl[t] * d_config.tissue[t].n;
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
// Main Monte Carlo kernel: one thread per photon
// ---------------------------------------------------------------------------
__global__ void mc_kernel(
    const uint8_t* __restrict__ volume,
    float*         __restrict__ fluence,
    double*        __restrict__ det_weight,
    double*        __restrict__ det_pathlength,
    double*        __restrict__ det_partial_pl,
    unsigned long long* __restrict__ det_count,
    double*        __restrict__ det_tpsf,
    double*        __restrict__ det_gated_weight,
    double*        __restrict__ det_gated_partial_pl,
    unsigned long long* __restrict__ det_gated_count,
    float*         __restrict__ path_pos,
    int*           __restrict__ path_det,
    int*           __restrict__ path_len,
    uint64_t       photons_per_thread,
    uint64_t       seed_offset
) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;

    curandState rng;
    curand_init(seed_offset + tid, 0, 0, &rng);

    for (uint64_t p = 0; p < photons_per_thread; p++) {
        // --- Launch photon with beam spread ---
        float px = d_config.src_x;
        float py = d_config.src_y;
        float pz = d_config.src_z;
        float ddx = d_config.src_dx;
        float ddy = d_config.src_dy;
        float ddz = d_config.src_dz;

        // Uniform disk sampling perpendicular to source direction
        if (d_config.beam_radius > 0.0f) {
            float r = d_config.beam_radius * sqrtf(curand_uniform(&rng));
            float theta = 2.0f * 3.14159265f * curand_uniform(&rng);
            // Build tangent vectors to source direction
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
            float offset_x = r * (cosf(theta) * tx + sinf(theta) * bx);
            float offset_y = r * (cosf(theta) * ty + sinf(theta) * by);
            float offset_z = r * (cosf(theta) * tz + sinf(theta) * bz);
            px += offset_x;
            py += offset_y;
            pz += offset_z;
        }

        float weight = 1.0f;

        float ppl[NUM_TISSUE_TYPES];
        for (int t = 0; t < NUM_TISSUE_TYPES; t++) ppl[t] = 0.0f;
        float total_pl = 0.0f;

        // Path recording: for the rare photon we want to record,
        // pre-allocate a global memory slot and write directly.
        int path_slot = -1;
        int path_step = 0;
        bool should_record = false;
        // ~1 in 50000 photons is a candidate for recording
        if (curand_uniform(&rng) < 2e-5f) {
            int slot = atomicAdd(&d_total_paths, 1);
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

        // --- Propagation loop ---
        for (int step = 0; step < 500000; step++) {
            uint8_t tissue = get_tissue(volume, px, py, pz);

            if (tissue == TISSUE_AIR) {
                int det_id = check_detectors(px, py, pz, ddx, ddy, ddz);
                if (det_id >= 0) {
                    record_detection(det_id, weight, total_pl, ppl,
                        det_weight, det_pathlength, det_partial_pl, det_count,
                        det_tpsf, det_gated_weight, det_gated_partial_pl, det_gated_count);
                    // Finalize recorded path
                    if (should_record && path_step > 1) {
                        int det_cnt = atomicAdd(&d_path_count[det_id], 1);
                        if (det_cnt < PATHS_PER_DET) {
                            path_det[path_slot] = det_id;
                            path_len[path_slot] = path_step;
                        } else {
                            // Too many for this detector, invalidate
                            path_len[path_slot] = 0;
                        }
                    }
                } else if (should_record) {
                    // Not detected, invalidate path
                    path_len[path_slot] = 0;
                }
                break;
            }

            OpticalProps op = d_config.tissue[tissue];
            float mu_t = op.mu_a + op.mu_s;

            if (mu_t < 1e-10f) {
                float big_step = d_config.dx;
                px += ddx * big_step;
                py += ddy * big_step;
                pz += ddz * big_step;
                ppl[tissue] += big_step;
                total_pl += big_step;
                continue;
            }

            // Sample step size
            float s = -logf(curand_uniform(&rng) + 1e-30f) / mu_t;

            // Move photon
            px += ddx * s;
            py += ddy * s;
            pz += ddz * s;
            ppl[tissue] += s;
            total_pl += s;

            // Record path position (write directly to global memory)
            if (should_record && path_step < MAX_PATH_STEPS) {
                int base = path_slot * MAX_PATH_STEPS * 3;
                path_pos[base + path_step * 3 + 0] = px;
                path_pos[base + path_step * 3 + 1] = py;
                path_pos[base + path_step * 3 + 2] = pz;
                path_step++;
            }

            // Check boundary after move
            uint8_t new_tissue = get_tissue(volume, px, py, pz);
            if (new_tissue == TISSUE_AIR) {
                // Fresnel reflection at tissue-air boundary
                float n_in  = d_config.tissue[tissue].n;
                float n_out = 1.0f;
                float cos_i = fmaxf(fabsf(ddx), fmaxf(fabsf(ddy), fabsf(ddz)));
                float R = fresnel_reflect(n_in, n_out, cos_i);
                if (curand_uniform(&rng) < R) {
                    px -= ddx * s;
                    py -= ddy * s;
                    pz -= ddz * s;
                    ppl[tissue] -= s;
                    total_pl -= s;
                    float ax = fabsf(ddx), ay = fabsf(ddy), az = fabsf(ddz);
                    if (ax >= ay && ax >= az) ddx = -ddx;
                    else if (ay >= ax && ay >= az) ddy = -ddy;
                    else ddz = -ddz;
                    continue;
                }
                // Photon transmitted: check detectors
                int det_id = check_detectors(px, py, pz, ddx, ddy, ddz);
                if (det_id >= 0) {
                    record_detection(det_id, weight, total_pl, ppl,
                        det_weight, det_pathlength, det_partial_pl, det_count,
                        det_tpsf, det_gated_weight, det_gated_partial_pl, det_gated_count);
                    // Finalize recorded path
                    if (should_record && path_step > 1) {
                        int det_cnt = atomicAdd(&d_path_count[det_id], 1);
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
                break;
            }

            // Handle refractive index mismatch at boundaries
            if (new_tissue != tissue) {
                float n_in  = d_config.tissue[tissue].n;
                float n_out = d_config.tissue[new_tissue].n;
                if (fabsf(n_in - n_out) > 1e-5f) {
                    float cos_i = fmaxf(fabsf(ddx), fmaxf(fabsf(ddy), fabsf(ddz)));
                    float R = fresnel_reflect(n_in, n_out, cos_i);
                    if (curand_uniform(&rng) < R) {
                        px -= ddx * s;
                        py -= ddy * s;
                        pz -= ddz * s;
                        ppl[tissue] -= s;
                        total_pl -= s;

                        float ax = fabsf(ddx), ay = fabsf(ddy), az = fabsf(ddz);
                        if (ax >= ay && ax >= az) ddx = -ddx;
                        else if (ay >= ax && ay >= az) ddy = -ddy;
                        else ddz = -ddz;
                        continue;
                    }
                }
            }

            // Deposit weight (absorption)
            float absorbed = weight * (op.mu_a / mu_t);
            weight -= absorbed;

            // Record fluence
            int vidx = pos_to_voxel(px, py, pz);
            if (vidx >= 0) {
                atomicAdd(&fluence[vidx], absorbed);
            }

            // Scatter
            henyey_greenstein(op.g, &ddx, &ddy, &ddz, &rng);

            // Russian roulette
            if (weight < d_config.weight_threshold) {
                if (curand_uniform(&rng) < (1.0f / (float)d_config.roulette_m)) {
                    weight *= (float)d_config.roulette_m;
                } else {
                    break;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host launcher
// ---------------------------------------------------------------------------
void launch_mc_simulation(
    const uint8_t* h_volume,
    const SimConfig& config,
    const Detector* h_dets,
    int n_dets,
    DetectorResult* h_results,
    float* d_fluence,
    double* h_tpsf,
    double* h_gated_weight,
    double* h_gated_partial_pl,
    uint64_t* h_gated_count,
    float* h_path_pos,
    int*   h_path_det,
    int*   h_path_len,
    int*   h_num_paths
) {
    int nx = config.nx, ny = config.ny, nz = config.nz;
    size_t vol_size = (size_t)nx * ny * nz;

    // --- Upload volume ---
    uint8_t* d_volume;
    cudaMalloc(&d_volume, vol_size);
    cudaMemcpy(d_volume, h_volume, vol_size, cudaMemcpyHostToDevice);

    // --- Upload config to constant memory ---
    cudaMemcpyToSymbol(d_config, &config, sizeof(SimConfig));
    cudaMemcpyToSymbol(d_dets, h_dets, n_dets * sizeof(Detector));
    cudaMemcpyToSymbol(d_n_dets, &n_dets, sizeof(int));

    // --- Zero fluence ---
    cudaMemset(d_fluence, 0, vol_size * sizeof(float));

    // --- Allocate CW detector accumulators ---
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

    // --- Allocate TD accumulators ---
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

    // --- Allocate path recording buffers ---
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

    // Zero device-side path counters
    {
        int zero = 0;
        int zeros[128] = {};
        cudaMemcpyToSymbol(d_total_paths, &zero, sizeof(int));
        cudaMemcpyToSymbol(d_path_count, zeros, sizeof(zeros));
    }

    // --- Launch configuration ---
    int block_size = 256;
    int num_blocks;

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int max_threads = prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor;
    int total_threads = (max_threads / block_size) * block_size;
    num_blocks = total_threads / block_size;

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
    printf("  Total photons: %llu\n",
           (unsigned long long)(actual_per_batch * num_batches));

    // --- Launch kernel in batches with progress bar ---
    auto launch_start = std::chrono::high_resolution_clock::now();

    for (int batch = 0; batch < num_batches; batch++) {
        uint64_t seed_offset = (uint64_t)batch * threads_total;

        mc_kernel<<<num_blocks, block_size>>>(
            d_volume, d_fluence,
            d_det_weight, d_det_pathlength, d_det_partial_pl, d_det_count,
            d_det_tpsf, d_det_gated_weight, d_det_gated_partial_pl, d_det_gated_count,
            d_path_pos, d_path_det, d_path_len,
            photons_per_thread_per_batch, seed_offset
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

    // --- Copy CW results back ---
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
        for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
            h_results[d].partial_pathlength[t] = h_det_partial_pl[d * NUM_TISSUE_TYPES + t];
        }
    }

    // --- Copy TD results back ---
    cudaMemcpy(h_tpsf, d_det_tpsf,
               n_dets * TPSF_BINS * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_weight, d_det_gated_weight,
               n_dets * NUM_TIME_GATES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_partial_pl, d_det_gated_partial_pl,
               n_dets * NUM_TIME_GATES * NUM_TISSUE_TYPES * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gated_count, d_det_gated_count,
               n_dets * NUM_TIME_GATES * sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // --- Copy path recording results back ---
    int total_paths_recorded = 0;
    cudaMemcpyFromSymbol(&total_paths_recorded, d_total_paths, sizeof(int));
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

    // --- Cleanup ---
    cudaFree(d_volume);
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
