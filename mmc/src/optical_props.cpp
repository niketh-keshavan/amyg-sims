/**
 * @file optical_props.cpp
 * @brief Optical properties implementation.
 */

#include "optical_props.h"
#include <cstdio>

namespace mmc {

void get_optical_properties(int wavelength_nm, OpticalProps* props) {
    const OpticalProps* src = nullptr;
    
    switch (wavelength_nm) {
        case 760:
            src = data::PROPS_760;
            break;
        case 850:
            src = data::PROPS_850;
            break;
        default:
            fprintf(stderr, "Warning: Unknown wavelength %d nm, using 760 nm\n", wavelength_nm);
            src = data::PROPS_760;
            break;
    }
    
    for (int i = 0; i < NUM_TISSUES; i++) {
        props[i] = src[i];
    }
}

OpticalProps* init_optical_properties_gpu(int wavelength_nm) {
    OpticalProps host_props[NUM_TISSUES];
    get_optical_properties(wavelength_nm, host_props);
    
    OpticalProps* device_props = nullptr;
    cudaMalloc(&device_props, NUM_TISSUES * sizeof(OpticalProps));
    cudaMemcpy(device_props, host_props, NUM_TISSUES * sizeof(OpticalProps), 
               cudaMemcpyHostToDevice);
    
    return device_props;
}

void print_optical_properties(int wavelength_nm) {
    OpticalProps props[NUM_TISSUES];
    get_optical_properties(wavelength_nm, props);
    
    const char* names[] = {"Air", "Scalp", "Skull", "CSF", "Gray", "White", "Amygdala"};
    
    printf("\nOptical Properties at %d nm:\n", wavelength_nm);
    printf("%-12s %10s %10s %10s %10s %10s\n", 
           "Tissue", "mu_a", "mu_s", "g", "n", "mu_s'");
    printf("%-12s %10s %10s %10s %10s %10s\n", 
           "", "(1/mm)", "(1/mm)", "", "", "(1/mm)");
    
    for (int i = 0; i < NUM_TISSUES; i++) {
        printf("%-12s %10.4f %10.4f %10.4f %10.4f %10.4f\n",
               names[i],
               props[i].mu_a,
               props[i].mu_s,
               props[i].g,
               props[i].n,
               props[i].mu_s_prime);
    }
    printf("\n");
}

} // namespace mmc
