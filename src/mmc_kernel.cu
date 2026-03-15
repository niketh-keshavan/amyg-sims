// ---------------------------------------------------------------------------
// Mesh-Based Monte Carlo — CUDA photon transport kernel
// ---------------------------------------------------------------------------
// Algorithm: Fang & Boas (2009), "Tetrahedral mesh-based Monte Carlo method"
// Each thread propagates one photon through a tetrahedral head mesh.
//
// Photon state:
//   - position (float3, mm, MNI152 space)
//   - direction (float3, unit vector)
//   - weight (float, Beer-Lambert attenuation)
//   - current element ID (int)
//   - per-tissue partial optical pathlength (float[7])
//
// Free-path sampling uses optical depth tracking across tissue boundaries:
//   tau_scatter = -log(xi)   [sampled once per scatter event]
//   tau_so_far  += mu_t * d  [accumulated as photon traverses each segment]
//   Scattering occurs when tau_so_far >= tau_scatter.
// This correctly handles heterogeneous tissue without rescaling.
// ---------------------------------------------------------------------------

#include "mmc_kernel.cuh"
#include "mmc_types.cuh"
#include "types.cuh"
#include <curand_kernel.h>
#include <math_constants.h>

// ---- Math helpers ----------------------------------------------------------

__device__ __forceinline__ float3 operator+(float3 a, float3 b) { return make_float3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ __forceinline__ float3 operator-(float3 a, float3 b) { return make_float3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ __forceinline__ float3 operator*(float s, float3 a)  { return make_float3(s*a.x, s*a.y, s*a.z); }
__device__ __forceinline__ float3 operator*(float3 a, float s)  { return make_float3(s*a.x, s*a.y, s*a.z); }
__device__ __forceinline__ float3 operator-(float3 a)           { return make_float3(-a.x,-a.y,-a.z); }
__device__ __forceinline__ float  dot(float3 a, float3 b)       { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ __forceinline__ float3 cross(float3 a, float3 b)     { return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x); }
__device__ __forceinline__ float  len2(float3 a)                { return dot(a,a); }
__device__ __forceinline__ float3 normalize3(float3 a)          { return a * rsqrtf(dot(a,a)); }

// ---- Mesh accessors --------------------------------------------------------

__device__ __forceinline__ float3 get_node(const float* nodes, int idx)
{
    int b = idx * 3;
    return make_float3(nodes[b], nodes[b+1], nodes[b+2]);
}

__device__ __forceinline__ void get_elem_verts(
    const float* nodes, const int32_t* elems, int eid,
    float3& v0, float3& v1, float3& v2, float3& v3)
{
    int b = eid * 4;
    v0 = get_node(nodes, elems[b+0]);
    v1 = get_node(nodes, elems[b+1]);
    v2 = get_node(nodes, elems[b+2]);
    v3 = get_node(nodes, elems[b+3]);
}

// ---- Ray-triangle intersection (Möller-Trumbore) ---------------------------
// Returns positive t if ray hits triangle, else -1.
// Epsilon tolerances allow photon at face to find the opposite face.

__device__ float ray_tri_intersect(float3 pos, float3 dir,
                                    float3 va, float3 vb, float3 vc)
{
    const float EPS = 1e-9f;
    float3 e1 = vb - va;
    float3 e2 = vc - va;
    float3 h  = cross(dir, e2);
    float  a  = dot(e1, h);
    if (fabsf(a) < EPS) return -1.0f;

    float  f  = 1.0f / a;
    float3 s  = pos - va;
    float  u  = f * dot(s, h);
    if (u < -1e-5f || u > 1.0f + 1e-5f) return -1.0f;

    float3 q  = cross(s, e1);
    float  v  = f * dot(dir, q);
    if (v < -1e-5f || u + v > 1.0f + 1e-5f) return -1.0f;

    float  t  = f * dot(e2, q);
    return (t > 1e-7f) ? t : -1.0f;   // exclude hits behind/at origin
}

// ---- Find exit face from current tetrahedron --------------------------------
// Returns distance to exit (mm) and sets exit_face (0-3).
// Face k is opposite vertex k:  0→(v1,v2,v3), 1→(v0,v2,v3),
//                               2→(v0,v1,v3), 3→(v0,v1,v2)

__device__ float ray_tet_exit(float3 pos, float3 dir,
                               float3 v0, float3 v1, float3 v2, float3 v3,
                               int* exit_face)
{
    float min_t = 1e30f;
    *exit_face = -1;
    float t;

    t = ray_tri_intersect(pos, dir, v1, v2, v3);
    if (t > 0 && t < min_t) { min_t = t; *exit_face = 0; }

    t = ray_tri_intersect(pos, dir, v0, v2, v3);
    if (t > 0 && t < min_t) { min_t = t; *exit_face = 1; }

    t = ray_tri_intersect(pos, dir, v0, v1, v3);
    if (t > 0 && t < min_t) { min_t = t; *exit_face = 2; }

    t = ray_tri_intersect(pos, dir, v0, v1, v2);
    if (t > 0 && t < min_t) { min_t = t; *exit_face = 3; }

    return min_t;
}

// ---- Outward face normal (points away from opposite vertex) ----------------

__device__ float3 face_outward_normal(float3 v0, float3 v1, float3 v2, float3 v3,
                                       int face)
{
    float3 fa, fb, fc, opp;
    switch (face) {
        case 0: fa=v1; fb=v2; fc=v3; opp=v0; break;
        case 1: fa=v0; fb=v2; fc=v3; opp=v1; break;
        case 2: fa=v0; fb=v1; fc=v3; opp=v2; break;
        default:fa=v0; fb=v1; fc=v2; opp=v3; break;
    }
    float3 n = normalize3(cross(fb - fa, fc - fa));
    // Ensure n points away from opposite vertex (outward)
    if (dot(n, fa - opp) < 0.0f) n = -n;
    return n;
}

// ---- Fresnel reflectance ----------------------------------------------------

__device__ float fresnel_R(float cos_i, float n1, float n2)
{
    float sin_i2 = fmaxf(0.0f, 1.0f - cos_i*cos_i);
    float eta    = n1 / n2;
    float sin_t2 = eta * eta * sin_i2;
    if (sin_t2 >= 1.0f) return 1.0f;                  // total internal reflection
    float cos_t  = sqrtf(1.0f - sin_t2);
    float rs = (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t);
    float rp = (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t);
    return 0.5f * (rs*rs + rp*rp);
}

// ---- Refracted ray direction (Snell's law) ----------------------------------
// n points from medium 1 → medium 2 (same side as incident ray direction d).

__device__ float3 refract_dir(float3 d, float3 n, float n1, float n2)
{
    float eta   = n1 / n2;
    float cos_i = dot(d, n);                           // > 0: d and n aligned
    float k     = 1.0f - eta*eta*(1.0f - cos_i*cos_i);
    if (k < 0.0f) return d - 2.0f*cos_i*n;            // TIR fallback
    return normalize3(eta*d + (eta*cos_i - sqrtf(k))*n);
}

__device__ float3 reflect_dir(float3 d, float3 n)
{
    return d - 2.0f * dot(d, n) * n;
}

// ---- Henyey-Greenstein scattering ------------------------------------------

__device__ void scatter_hg(float3& dir, float g, curandState* rng)
{
    float cos_theta;
    float u = curand_uniform(rng);
    if (fabsf(g) < 1e-4f) {
        cos_theta = 2.0f * u - 1.0f;
    } else {
        float tmp = (1.0f - g*g) / (1.0f - g + 2.0f*g*u);
        cos_theta  = (1.0f + g*g - tmp*tmp) / (2.0f*g);
    }
    cos_theta = fmaxf(-1.0f, fminf(1.0f, cos_theta));

    float sin_theta = sqrtf(fmaxf(0.0f, 1.0f - cos_theta*cos_theta));
    float phi       = 2.0f * CUDART_PI_F * curand_uniform(rng);
    float cos_phi   = cosf(phi);
    float sin_phi   = sinf(phi);

    // Build orthonormal basis around current direction
    float3 d = dir;
    float3 t, s;
    if (fabsf(d.x) < 0.9f) {
        float3 ax = make_float3(1.f, 0.f, 0.f);
        t = normalize3(ax - dot(ax,d)*d);
    } else {
        float3 ay = make_float3(0.f, 1.f, 0.f);
        t = normalize3(ay - dot(ay,d)*d);
    }
    s = cross(d, t);

    dir = normalize3(cos_theta*d + sin_theta*(cos_phi*t + sin_phi*s));
}

// ---- Point-in-tetrahedron test ---------------------------------------------
// Uses signed-volume (determinant) test for each sub-tet.

__device__ bool point_in_tet(float3 p, float3 v0, float3 v1, float3 v2, float3 v3)
{
    // scalar triple product sign of full tet
    float3 e1 = v1-v0, e2 = v2-v0, e3 = v3-v0;
    float  dt = dot(cross(e1,e2),e3);
    const float eps = 1e-5f * fabsf(dt);

    // Sub-tet volumes share sign with dt iff p is inside
    float d0 = dot(cross(v2-v1, v3-v1), p-v1);
    float d1 = dot(cross(v2-v0, v3-v0), v1-p) * -1.f;  // flip one operand
    // Rewrite as: sub-tet (p,v1,v2,v3) sign vs full tet sign
    // Simpler: barycentric via solving 3×3 system
    // Use: p = v0 + l1*(v1-v0) + l2*(v2-v0) + l3*(v3-v0), all lk in [0,1], sum<=1

    float denom = dot(e1, cross(e2, e3));
    if (fabsf(denom) < 1e-12f) return false;

    float3 dp = p - v0;
    float  l1 = dot(dp, cross(e2, e3)) / denom;
    if (l1 < -1e-5f || l1 > 1.0f + 1e-5f) return false;

    float  l2 = dot(e1, cross(dp, e3)) / denom;
    if (l2 < -1e-5f || l2 > 1.0f + 1e-5f) return false;

    float  l3 = dot(e1, cross(e2, dp)) / denom;
    if (l3 < -1e-5f || l3 > 1.0f + 1e-5f) return false;

    return (l1 + l2 + l3 <= 1.0f + 1e-5f);
    (void)d0; (void)d1; (void)dt; (void)eps;
}

// ---- Spatial grid helpers --------------------------------------------------

__device__ int grid_cell_idx(const MMCMeshDevice& mesh, float3 p)
{
    int ix = (int)((p.x - mesh.bbox_min[0]) / mesh.cell_size[0]);
    int iy = (int)((p.y - mesh.bbox_min[1]) / mesh.cell_size[1]);
    int iz = (int)((p.z - mesh.bbox_min[2]) / mesh.cell_size[2]);
    ix = max(0, min(mesh.grid_dims[0]-1, ix));
    iy = max(0, min(mesh.grid_dims[1]-1, iy));
    iz = max(0, min(mesh.grid_dims[2]-1, iz));
    return ix + mesh.grid_dims[0] * (iy + mesh.grid_dims[1] * iz);
}

__device__ int mmc_find_element(float3 pos, const MMCMeshDevice& mesh)
{
    int cell = grid_cell_idx(mesh, pos);
    int start = mesh.grid_start[cell];
    int count = mesh.grid_count[cell];

    for (int i = 0; i < count; i++) {
        int eid = mesh.grid_elems[start + i];
        float3 v0, v1, v2, v3;
        get_elem_verts(mesh.nodes, mesh.elems, eid, v0, v1, v2, v3);
        if (point_in_tet(pos, v0, v1, v2, v3))
            return eid;
    }
    return -1;  // not found
}

// ---- Record a detected photon ----------------------------------------------

__device__ void record_detection(
    int det_id, float weight,
    const float tissue_opl[NUM_TISSUE_TYPES],   // partial optical pathlengths [mm]
    float total_opl,                             // total optical pathlength [mm]
    const MMCConfig& config,
    double* det_weight, double* det_pathlength,
    unsigned long long* det_count,
    float* tpsf,
    double* gated_weight, double* gated_pathlength)
{
    // Physical TOF in ps: OPL / (c_vacuum) — n already baked into OPL
    float tof_ps = total_opl / C_VACUUM_MM_PS;

    atomicAdd(&det_weight[det_id], (double)weight);
    atomicAdd(&det_count[det_id],  1ULL);

    for (int t = 0; t < NUM_TISSUE_TYPES; t++)
        atomicAdd(&det_pathlength[det_id*NUM_TISSUE_TYPES + t], (double)tissue_opl[t]);

    // TPSF histogram
    int bin = (int)(tof_ps / config.tpsf_bin_ps);
    if (bin >= 0 && bin < config.tpsf_bins)
        atomicAdd(&tpsf[det_id * config.tpsf_bins + bin], weight);

    // Time-gated accumulation
    for (int g = 0; g < config.num_gates; g++) {
        if (tof_ps >= config.gate_start_ps[g] && tof_ps < config.gate_end_ps[g]) {
            atomicAdd(&gated_weight[det_id*NUM_TIME_GATES + g], (double)weight);
            int base = (det_id*NUM_TIME_GATES + g)*NUM_TISSUE_TYPES;
            for (int t = 0; t < NUM_TISSUE_TYPES; t++)
                atomicAdd(&gated_pathlength[base + t], (double)tissue_opl[t]);
        }
    }
}

// ===========================================================================
//  RNG initializer
// ===========================================================================

__global__ void mmc_init_rng(curandState* states, unsigned long long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n)
        curand_init(seed, tid, 0, &states[tid]);
}

// ===========================================================================
//  Main MMC photon transport kernel
// ===========================================================================

__global__ void mmc_photon_kernel(
    MMCMeshDevice       mesh,
    MMCConfig           config,
    const Detector*     detectors,
    int                 num_detectors,
    double*             det_weight,
    double*             det_pathlength,
    unsigned long long* det_count,
    float*              tpsf,
    double*             gated_weight,
    double*             gated_pathlength,
    curandState*        rng_states)
{
    int tid            = blockIdx.x * blockDim.x + threadIdx.x;
    long long n_threads = (long long)gridDim.x * blockDim.x;

    curandState rng = rng_states[tid];

    // Precompute source basis vectors (disk sampling)
    float3 src_dir = make_float3(config.src_dir[0], config.src_dir[1], config.src_dir[2]);
    float3 e1, e2;
    if (fabsf(src_dir.x) < 0.9f) {
        e1 = normalize3(cross(make_float3(1.f,0.f,0.f), src_dir));
    } else {
        e1 = normalize3(cross(make_float3(0.f,1.f,0.f), src_dir));
    }
    e2 = cross(src_dir, e1);

    for (long long ph = (long long)tid; ph < (long long)config.num_photons; ph += n_threads)
    {
        // ---- Launch photon --------------------------------------------------
        float r   = config.beam_radius * sqrtf(curand_uniform(&rng));
        float phi = 2.0f * CUDART_PI_F * curand_uniform(&rng);
        float cx  = r * cosf(phi);
        float cy  = r * sinf(phi);

        float3 pos = make_float3(
            config.src_pos[0] + cx*e1.x + cy*e2.x,
            config.src_pos[1] + cx*e1.y + cy*e2.y,
            config.src_pos[2] + cx*e1.z + cy*e2.z);
        float3 dir    = src_dir;
        float  weight = 1.0f;

        // Locate starting element
        int elem_id = config.src_elem;
        if (elem_id < 0)
            elem_id = mmc_find_element(pos, mesh);
        if (elem_id < 0) continue;   // launch point outside mesh — skip

        float tissue_opl[NUM_TISSUE_TYPES] = {};
        float total_opl = 0.0f;

        // ---- Propagation loop ----------------------------------------------
        while (weight > config.weight_threshold) {

            int cur_tissue = mesh.tissue[elem_id];

            // Photon exited to air (external boundary)
            if (cur_tissue == TISSUE_AIR || elem_id < 0) {
                // Check all detectors
                for (int d = 0; d < num_detectors; d++) {
                    float3 dp   = make_float3(detectors[d].x, detectors[d].y, detectors[d].z);
                    float3 diff = pos - dp;
                    if (len2(diff) > detectors[d].radius * detectors[d].radius)
                        continue;  // outside detector area
                    
                    // Check angular acceptance: photon direction vs detector surface normal
                    // Photon must be coming from inside the head (aligned with normal)
                    float3 det_n = make_float3(detectors[d].nx, detectors[d].ny, detectors[d].nz);
                    float cos_theta = dot(dir, det_n);
                    if (cos_theta < detectors[d].n_critical)
                        continue;  // rejected: exceeds critical angle (grazing exit)
                    
                    record_detection(d, weight, tissue_opl, total_opl,
                                     config,
                                     det_weight, det_pathlength, det_count,
                                     tpsf, gated_weight, gated_pathlength);
                    break;   // count each photon once
                }
                break;   // terminate photon
            }

            // Sample optical depth to next scatter event
            float tau_scatter = -logf(curand_uniform(&rng));
            float tau_accum   = 0.0f;

            // ---- Inner traversal loop: cross tet boundaries until scatter --
            bool scattered   = false;
            int  crossings   = MMC_MAX_CROSSINGS;

            while (!scattered && crossings-- > 0) {

                cur_tissue = mesh.tissue[elem_id];
                if (cur_tissue == TISSUE_AIR || elem_id < 0) break;

                const OpticalProps& op = config.tissue_props[cur_tissue];
                float mu_t = op.mu_a + op.mu_s;
                if (mu_t < 1e-10f) { elem_id = -1; break; }

                // Vertices of current element
                float3 v0, v1, v2, v3;
                get_elem_verts(mesh.nodes, mesh.elems, elem_id, v0, v1, v2, v3);

                // Find exit face
                int   exit_face;
                float d_exit = ray_tet_exit(pos, dir, v0, v1, v2, v3, &exit_face);

                if (exit_face < 0 || d_exit > 1e10f) {
                    // Degenerate element — skip ahead slightly
                    pos = pos + 1e-4f * dir;
                    elem_id = mmc_find_element(pos, mesh);
                    break;
                }

                float tau_exit     = mu_t * d_exit;
                float tau_remain   = tau_scatter - tau_accum;

                if (tau_remain <= tau_exit) {
                    // Scatter occurs within this element
                    float d_scat = tau_remain / mu_t;
                    weight *= expf(-op.mu_a * d_scat);
                    pos     = pos + d_scat * dir;
                    tissue_opl[cur_tissue] += d_scat * op.n;
                    total_opl              += d_scat * op.n;
                    scattered = true;

                } else {
                    // Move to exit face
                    weight *= expf(-op.mu_a * d_exit);
                    pos     = pos + d_exit * dir;
                    tissue_opl[cur_tissue] += d_exit * op.n;
                    total_opl              += d_exit * op.n;
                    tau_accum += tau_exit;

                    int neighbor = mesh.neighbors[elem_id * 4 + exit_face];

                    if (neighbor < 0) {
                        // External boundary — Fresnel at tissue→air interface
                        float3 fn    = face_outward_normal(v0,v1,v2,v3, exit_face);
                        float  cos_i = fmaxf(0.0f, dot(dir, fn));
                        float  R     = fresnel_R(cos_i, op.n, 1.0f);

                        if (curand_uniform(&rng) < R) {
                            // Specular reflection back into tissue
                            dir = reflect_dir(dir, fn);
                            pos = pos + (-2e-5f * fn);   // nudge inside
                            // tau_accum unchanged, continue inner loop
                        } else {
                            // Transmitted to air — exit head
                            elem_id = -1;
                            break;
                        }

                    } else {
                        // Internal tissue boundary
                        int   next_tissue = mesh.tissue[neighbor];
                        float n_in  = op.n;
                        float n_out = config.tissue_props[next_tissue].n;

                        if (fabsf(n_in - n_out) > 1e-4f) {
                            float3 fn    = face_outward_normal(v0,v1,v2,v3, exit_face);
                            float  cos_i = fmaxf(0.0f, dot(dir, fn));
                            float  R     = fresnel_R(cos_i, n_in, n_out);

                            if (curand_uniform(&rng) < R) {
                                // Reflect — stay in current element
                                dir = reflect_dir(dir, fn);
                                pos = pos + (-2e-5f * fn);
                                // tau_accum unchanged, continue
                            } else {
                                // Refract into neighbor
                                dir     = refract_dir(dir, fn, n_in, n_out);
                                elem_id = neighbor;
                                // Rescale remaining optical depth for new mu_t
                                // tau_remain is still tau_scatter - tau_accum;
                                // next iteration uses new mu_t automatically
                            }
                        } else {
                            elem_id = neighbor;
                        }
                    }
                }
            }   // end inner traversal loop

            if (elem_id < 0 || cur_tissue == TISSUE_AIR) continue;  // outer loop checks

            if (!scattered) break;   // safety: shouldn't happen

            // ---- Henyey-Greenstein scatter ----------------------------------
            scatter_hg(dir, config.tissue_props[mesh.tissue[elem_id]].g, &rng);

            // ---- Russian roulette -------------------------------------------
            if (weight < config.weight_threshold) {
                if (curand_uniform(&rng) < 1.0f / (float)config.roulette_m)
                    weight *= (float)config.roulette_m;
                else
                    break;
            }
        }   // end outer propagation loop
    }   // end photon batch loop

    rng_states[tid] = rng;
}
