/**
 * @file optical_props.h
 * @brief Optical properties for MNI152 head tissues.
 * 
 * Properties from literature:
 * - Okada & Delpy (2003) Applied Optics
 * - Jacques (2013) Physics in Medicine and Biology
 * - Strangman et al. (2014) NeuroImage
 */

#pragma once

#include "types.h"

namespace mmc {

/**
 * @brief Get optical properties for all tissues at specified wavelength.
 * 
 * @param wavelength_nm Wavelength (760 or 850 nm)
 * @param props Output array (size NUM_TISSUES)
 */
void get_optical_properties(int wavelength_nm, OpticalProps* props);

/**
 * @brief Initialize optical properties on GPU.
 * 
 * @param wavelength_nm Wavelength
 * @return Device pointer to properties array
 */
OpticalProps* init_optical_properties_gpu(int wavelength_nm);

/**
 * @brief Print optical properties table.
 */
void print_optical_properties(int wavelength_nm);

// Specific wavelength data
namespace data {

// At 760 nm
constexpr OpticalProps PROPS_760[NUM_TISSUES] = {
    // AIR - not used for photon transport
    {0.0f,    0.0f,     0.0f,  1.0f,  0.0f},
    // SCALP
    {0.019f,  1.2f,     0.9f,  1.4f,  0.12f},
    // SKULL
    {0.012f,  1.0f,     0.9f,  1.55f, 0.10f},
    // CSF
    {0.004f,  0.01f,    0.0f,  1.35f, 0.01f},
    // GRAY MATTER
    {0.018f,  0.9f,     0.89f, 1.36f, 0.099f},
    // WHITE MATTER
    {0.017f,  1.2f,     0.84f, 1.36f, 0.192f},
    // AMYGDALA (treated as gray matter with slight differences)
    {0.020f,  0.95f,    0.89f, 1.36f, 0.1045f},
};

// At 850 nm
constexpr OpticalProps PROPS_850[NUM_TISSUES] = {
    // AIR
    {0.0f,    0.0f,     0.0f,  1.0f,  0.0f},
    // SCALP
    {0.023f,  0.95f,    0.9f,  1.4f,  0.095f},
    // SKULL
    {0.016f,  0.8f,     0.9f,  1.55f, 0.08f},
    // CSF
    {0.004f,  0.01f,    0.0f,  1.35f, 0.01f},
    // GRAY MATTER
    {0.027f,  0.75f,    0.89f, 1.36f, 0.0825f},
    // WHITE MATTER
    {0.025f,  1.0f,     0.84f, 1.36f, 0.16f},
    // AMYGDALA
    {0.029f,  0.78f,    0.89f, 1.36f, 0.0858f},
};

} // namespace data

} // namespace mmc
