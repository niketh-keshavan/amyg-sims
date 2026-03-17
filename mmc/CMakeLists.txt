# MMC (Mesh-Based Monte Carlo) with MNI152 head model
add_executable(mmc_fnirs
    src/mmc_main.cu
    src/mmc_mesh.cu
    src/mmc_kernel.cu
    ${PROJECT_SOURCE_DIR}/src/optical_properties.cu
    ${PROJECT_SOURCE_DIR}/src/detector.cu
)

target_include_directories(mmc_fnirs PRIVATE
    include
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mmc_fnirs PRIVATE ${CURAND_LINK})

target_compile_options(mmc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
