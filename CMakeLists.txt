cmake_minimum_required(VERSION 3.18)
project(amyg_fnirs_mc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Auto-detect GPU architecture, fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 120)
endif()

find_package(CUDAToolkit REQUIRED)

# curand may be an imported target or a plain library depending on toolkit version
if(NOT TARGET CUDA::curand)
    find_library(CURAND_LIB curand HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()

set(CURAND_LINK $<IF:$<TARGET_EXISTS:CUDA::curand>,CUDA::curand,${CURAND_LIB}>)

# ── Voxel-based MC (original) ──────────────────────────────────────────────
add_executable(mc_fnirs
    src/voxel/main.cu
    src/voxel/mc_kernel.cu
    src/voxel/geometry.cu
    src/voxel/detector.cu
    src/voxel/optical_properties.cu
)
target_include_directories(mc_fnirs PRIVATE include/voxel)
if(TARGET CUDA::curand)
    target_link_libraries(mc_fnirs PRIVATE CUDA::curand)
elseif(CURAND_LIB)
    target_link_libraries(mc_fnirs PRIVATE ${CURAND_LIB})
else()
    message(FATAL_ERROR "curand not found. Install libcurand-dev or set CUDAToolkit_LIBRARY_DIR.")
endif()
target_compile_options(mc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# ── 3D Model Exporter ─────────────────────────────────────────────────────
add_executable(export_3d_model
    src/voxel/export_3d_model.cu
)
target_include_directories(export_3d_model PRIVATE include/voxel)
target_link_libraries(export_3d_model PRIVATE CUDA::cudart)
target_compile_options(export_3d_model PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)

# ── Mesh-Based MC (MMC) with MNI152 head model ────────────────────────────
# This is built separately from the mmc/ subdirectory
# See mmc/README.md for build instructions
option(BUILD_MMC "Build the Mesh-based Monte Carlo (MMC) simulation" OFF)

if(BUILD_MMC)
    message(STATUS "Building MMC from mmc/ subdirectory")
    add_subdirectory(mmc)
endif()
