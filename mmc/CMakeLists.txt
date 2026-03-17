# MMC (Mesh-Based Monte Carlo) with MNI152 head model
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

target_compile_options(mmc_fnirs PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
)
