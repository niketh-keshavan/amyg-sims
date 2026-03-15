cmake_minimum_required(VERSION 3.18)
project(mmc_fnirs LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Auto-detect GPU architecture
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89 90 120)
endif()

find_package(CUDAToolkit REQUIRED)

# curand
if(NOT TARGET CUDA::curand)
    find_library(CURAND_LIB curand 
        HINTS ${CUDAToolkit_LIBRARY_DIR} ${CUDA_TOOLKIT_ROOT_DIR}/lib64)
endif()
set(CURAND_LINK $<IF:$<TARGET_EXISTS:CUDA::curand>,CUDA::curand,${CURAND_LIB}>)

# Source files
set(MMC_SOURCES
    src/main.cpp
    src/mesh_loader.cpp
    src/optical_props.cpp
    src/detector_array.cpp
    src/simulation_config.cpp
    src/mmc_kernel.cu
)

# Main executable
add_executable(mmc_fnirs ${MMC_SOURCES})

target_include_directories(mmc_fnirs PRIVATE include)

target_link_libraries(mmc_fnirs PRIVATE 
    $<IF:$<TARGET_EXISTS:CUDA::curand>,CUDA::curand,${CURAND_LIB}>
)

target_compile_options(mmc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3 -lineinfo>
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall>
)

# Set output directory
set_target_properties(mmc_fnirs PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Print configuration
message(STATUS "MMC Configuration:")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  Source files: ${MMC_SOURCES}")
