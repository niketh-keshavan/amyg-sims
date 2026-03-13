#pragma once
#include "types.cuh"

// ---------------------------------------------------------------------------
// Wavelength-dependent optical properties via Mie scattering power law
// and chromophore-based absorption
// ---------------------------------------------------------------------------
// References:
//   Jacques (2013) "Optical properties of biological tissues: a review"
//   Prahl (1999) tabulated extinction coefficients
//   van der Zee (1992) tissue optical properties
//   Bevilacqua et al. (1999) skull optical properties
//
// Scattering:  mu_s'(lambda) = a_mie * (lambda/500nm)^(-b_mie)   [1/mm]
//              mu_s = mu_s' / (1 - g)
//
// Absorption:  mu_a(lambda) = sum( eps_i(lambda) * c_i )
//   Chromophores: HbO2, HHb, water, lipid
//
// Anisotropy:  g(lambda) ~ g_500 * (lambda/500)^(-g_power)
// ---------------------------------------------------------------------------

#define MAX_WAVELENGTHS 8

// Tissue scattering parameters (Mie power law)
struct TissueScatterParams {
    float a_mie;       // mu_s' at 500nm [1/mm]
    float b_mie;       // scattering power exponent
    float g_500;       // anisotropy at 500nm
    float g_power;     // anisotropy wavelength exponent
    float n;           // refractive index

    // Chromophore concentrations for absorption
    float blood_volume_fraction;
    float StO2;
    float water_fraction;
    float lipid_fraction;
    float baseline_mu_a;  // residual absorption [1/mm]
};

// Compute optical properties for all tissues at a given wavelength
void compute_optical_properties(float wavelength_nm, OpticalProps props[NUM_TISSUE_TYPES]);

// Get tissue scatter params for inspection
TissueScatterParams get_tissue_scatter_params(int tissue_type);

// Print optical properties table
void print_optical_properties(float wavelength_nm, const OpticalProps props[NUM_TISSUE_TYPES]);
