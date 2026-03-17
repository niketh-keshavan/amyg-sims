cmake_minimum_required(VERSION 3.18)

# MMC (Mesh-Based Monte Carlo) with MNI152 head model
project(mmc_fnirs LANGUAGES CXX CUDA)

# Find CUDA
find_package(CUDAToolkit REQUIRED)
set(CURAND_LINK CUDA::curand)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source paths
set(ROOT_SRC ${CMAKE_SOURCE_DIR}/src)
set(ROOT_INC ${CMAKE_SOURCE_DIR}/include)

add_executable(mmc_fnirs
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmc_main.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmc_mesh.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mmc_kernel.cu
    ${ROOT_SRC}/optical_properties.cu
    ${ROOT_SRC}/detector.cu
)

target_include_directories(mmc_fnirs PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ROOT_INC}
)

target_link_libraries(mmc_fnirs PRIVATE ${CURAND_LINK})

# Set CUDA architectures (supports RTX 3090 = sm_86)
set_target_properties(mmc_fnirs PROPERTIES
    CUDA_ARCHITECTURES "75;80;86"
)

target_compile_options(mmc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
