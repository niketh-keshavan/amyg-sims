cmake_minimum_required(VERSION 3.18)
project(amyg_fnirs_mc LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Auto-detect GPU architecture, fall back to common ones
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 120)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(mc_fnirs
    src/main.cu
    src/mc_kernel.cu
    src/geometry.cu
    src/detector.cu
)

target_include_directories(mc_fnirs PRIVATE include)
target_link_libraries(mc_fnirs PRIVATE CUDA::curand)

# Optimization flags
target_compile_options(mc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
