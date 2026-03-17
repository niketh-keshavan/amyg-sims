## Physics Core Focus
Your primary focus is implementing a physically accurate Mesh-based Monte Carlo (MMC) solver for tissue optics. Code must rigorously adhere to:
* **Photon Transport**: Radiative transfer equation (RTE) approximations in complex biological media.
* **Scattering & Absorption**: Henyey-Greenstein phase functions for anisotropic scattering; Beer-Lambert weight drops ($\Delta W = W \cdot \mu_a / \mu_t$).
* **Boundary Physics**: Accurate Möller–Trumbore ray-triangle intersections. You must correctly calculate Fresnel reflections, Snell's law refraction, and Total Internal Reflection (TIR) across refractive index ($n$) mismatches at tetrahedral boundaries.
* **Conservation**: Strict energy conservation checks and Russian Roulette for unbiased photon termination.

## Agent Workflow Directives

**1. PLAN MODE FOR COMPLEX TASKS**
* You **must** output a step-by-step technical plan before writing any code for complex physics logic (e.g., BVH traversal algorithms, shared memory reductions for photon fluence, cross-element boundary logic). 
* Wait for my approval on the mathematical and structural approach before implementing.

**2. PROMPT FOR HIGH-COST EXECUTION**
* Be highly aware of computational costs. 
* **STOP AND PROMPT ME** before executing long-running tasks, such as generating massive high-resolution MNI152 meshes, compiling enormous CUDA kernels from scratch, or running >10^7 photon simulations for validation. 

**3. DELEGATE SIMPLE TASKS**
* Do not waste your context window on trivialities. 
* Explicitly delegate simple, repetitive, or boilerplate tasks (e.g., writing basic shell scripts, simple Python plot wrappers, renaming variables, or file moving) back to me. This includes basic fixes when given errors. Just say: *"Please have another agent handle [Task X]."*

## CUDA/Performance Rules
* Optimize strictly for memory coalescing and minimizing warp divergence during the photon transport loop.
* Use flat arrays for mesh and BVH structures to ensure fast GPU device memory access.