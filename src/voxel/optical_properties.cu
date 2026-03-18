#include "optical_properties.cuh"
#include <cmath>
#include <cstdio>

// ---------------------------------------------------------------------------
// Chromophore extinction coefficient table
// ---------------------------------------------------------------------------
// HbO2 and HHb: from Prahl/OMLC (Scott Prahl compilation of Gratzer data)
// Units converted: original cm^-1 M^-1 -> mM^-1 mm^-1 (divide by 10000)
// Water: from Hale & Querry (1973), Kou et al. (1993)
// Lipid: from van Veen et al. (2005)
//
// Table covers 650-950nm in 10nm steps (fNIRS window)
// ---------------------------------------------------------------------------

struct ChromophoreRow {
    float wavelength_nm;
    float eps_HbO2;    // mM^-1 mm^-1
    float eps_HHb;     // mM^-1 mm^-1
    float mu_a_water;  // 1/mm (pure water)
    float mu_a_lipid;  // 1/mm
};

// 31 rows: 650nm to 950nm in 10nm steps
static const ChromophoreRow CHROMOPHORE_TABLE[] = {
    // nm    eps_HbO2   eps_HHb    mu_a_water  mu_a_lipid
    {650,   0.0382,    0.3346,    0.000326,   0.0010},
    {660,   0.0334,    0.3229,    0.000340,   0.0009},
    {670,   0.0303,    0.2960,    0.000355,   0.0009},
    {680,   0.0272,    0.2637,    0.000372,   0.0008},
    {690,   0.0240,    0.2266,    0.000396,   0.0008},
    {700,   0.0214,    0.1946,    0.000423,   0.0008},
    {710,   0.0196,    0.1695,    0.000454,   0.0009},
    {720,   0.0186,    0.1500,    0.000505,   0.0010},
    {730,   0.0170,    0.1308,    0.000564,   0.0011},
    {740,   0.0154,    0.1132,    0.000619,   0.0012},
    {750,   0.0140,    0.0988,    0.000710,   0.0013},
    {760,   0.0130,    0.0874,    0.000812,   0.0014},
    {770,   0.0124,    0.0780,    0.000940,   0.0015},
    {780,   0.0122,    0.0712,    0.001070,   0.0016},
    {790,   0.0126,    0.0659,    0.001230,   0.0015},
    {800,   0.0136,    0.0618,    0.001440,   0.0014},  // isosbestic
    {810,   0.0150,    0.0588,    0.001670,   0.0013},
    {820,   0.0170,    0.0568,    0.001920,   0.0012},
    {830,   0.0194,    0.0557,    0.002220,   0.0012},
    {840,   0.0226,    0.0554,    0.002530,   0.0013},
    {850,   0.0260,    0.0553,    0.002880,   0.0014},
    {860,   0.0290,    0.0555,    0.003220,   0.0016},
    {870,   0.0316,    0.0558,    0.003560,   0.0018},
    {880,   0.0338,    0.0564,    0.003920,   0.0021},
    {890,   0.0356,    0.0568,    0.004280,   0.0025},
    {900,   0.0366,    0.0570,    0.004620,   0.0030},
    {910,   0.0376,    0.0572,    0.005070,   0.0036},
    {920,   0.0392,    0.0575,    0.006070,   0.0043},
    {930,   0.0414,    0.0580,    0.008200,   0.0050},
    {940,   0.0440,    0.0586,    0.012000,   0.0058},
    {950,   0.0468,    0.0594,    0.018000,   0.0066},
};
static const int NUM_CHROMOPHORE_ROWS = sizeof(CHROMOPHORE_TABLE) / sizeof(CHROMOPHORE_TABLE[0]);

// ---------------------------------------------------------------------------
// Interpolate chromophore data at arbitrary wavelength
// ---------------------------------------------------------------------------
static ChromophoreRow interpolate_chromophores(float wl) {
    // Clamp to table range
    if (wl <= CHROMOPHORE_TABLE[0].wavelength_nm)
        return CHROMOPHORE_TABLE[0];
    if (wl >= CHROMOPHORE_TABLE[NUM_CHROMOPHORE_ROWS - 1].wavelength_nm)
        return CHROMOPHORE_TABLE[NUM_CHROMOPHORE_ROWS - 1];

    // Find bracketing rows
    int lo = 0;
    for (int i = 0; i < NUM_CHROMOPHORE_ROWS - 1; i++) {
        if (CHROMOPHORE_TABLE[i + 1].wavelength_nm >= wl) {
            lo = i;
            break;
        }
    }
    int hi = lo + 1;

    float t = (wl - CHROMOPHORE_TABLE[lo].wavelength_nm) /
              (CHROMOPHORE_TABLE[hi].wavelength_nm - CHROMOPHORE_TABLE[lo].wavelength_nm);

    ChromophoreRow r;
    r.wavelength_nm = wl;
    r.eps_HbO2   = CHROMOPHORE_TABLE[lo].eps_HbO2   * (1 - t) + CHROMOPHORE_TABLE[hi].eps_HbO2   * t;
    r.eps_HHb    = CHROMOPHORE_TABLE[lo].eps_HHb    * (1 - t) + CHROMOPHORE_TABLE[hi].eps_HHb    * t;
    r.mu_a_water = CHROMOPHORE_TABLE[lo].mu_a_water * (1 - t) + CHROMOPHORE_TABLE[hi].mu_a_water * t;
    r.mu_a_lipid = CHROMOPHORE_TABLE[lo].mu_a_lipid * (1 - t) + CHROMOPHORE_TABLE[hi].mu_a_lipid * t;
    return r;
}

// ---------------------------------------------------------------------------
// Tissue-specific scattering and composition parameters
// ---------------------------------------------------------------------------
// Jacques (2013), Bevilacqua et al. (1999), Okada & Delpy (2003),
// Strangman et al. (2014), Gebhart et al. (2006)
//
// a_mie: reduced scattering at 500nm [1/mm]
// b_mie: Mie scattering power (wavelength exponent)
// g_500: anisotropy at 500nm
// g_power: wavelength dependence of g (typically ~0.05-0.15)
// Blood parameters: volume fraction and StO2 are baseline resting values
// ---------------------------------------------------------------------------

static TissueScatterParams TISSUE_PARAMS[NUM_TISSUE_TYPES] = {
    // AIR: no scattering
    {0.0f, 0.0f, 1.0f, 0.0f, 1.000f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f},

    // SCALP: dermis-like tissue, moderate blood content
    // Jacques (2013): a'~1.7mm^-1 at 500nm for skin, b~1.12
    // Blood volume ~3%, StO2 ~0.75, water ~0.65
    {1.70f, 1.12f, 0.90f, 0.05f, 1.37f,
     0.03f, 0.75f, 0.65f, 0.10f, 0.002f},

    // SKULL (compact bone): high scattering, lower blood
    // Bevilacqua (1999): a'~2.0mm^-1 at 500nm, b~0.65 (lower power = flatter)
    // Firbank (1993): skull mu_s' ~ 1.3mm^-1 at 800nm
    // Blood volume ~1%, water ~0.12 (bone is drier)
    {2.00f, 0.65f, 0.93f, 0.03f, 1.56f,
     0.01f, 0.70f, 0.12f, 0.20f, 0.005f},

    // CSF: nearly transparent, very low scattering
    // g=0.9 retained per user specification (accepted value for MC sims)
    // Negligible blood, ~99% water
    {0.01f, 1.0f, 0.90f, 0.0f, 1.33f,
     0.0f, 0.0f, 0.99f, 0.0f, 0.0004f},

    // GRAY MATTER: high scattering, high blood content
    // Jacques (2013): a'~2.4mm^-1 at 500nm, b~1.61
    // Okada & Delpy (2003): CBV ~4%, StO2 ~0.70, water ~0.80
    {2.40f, 1.61f, 0.90f, 0.05f, 1.37f,
     0.04f, 0.70f, 0.80f, 0.05f, 0.001f},

    // WHITE MATTER: highest scattering (myelinated axons)
    // Jacques (2013): a'~4.0mm^-1 at 500nm, b~1.15
    // CBV ~2%, StO2 ~0.70, water ~0.72
    {4.00f, 1.15f, 0.90f, 0.05f, 1.37f,
     0.02f, 0.70f, 0.72f, 0.10f, 0.001f},

    // AMYGDALA: gray matter subtype, slightly different vasculature
    // Similar to GM but slightly higher blood volume (deep gray nuclei)
    // CBV ~5%, StO2 ~0.68, water ~0.82
    {2.40f, 1.61f, 0.90f, 0.05f, 1.37f,
     0.05f, 0.68f, 0.82f, 0.05f, 0.001f},
};

TissueScatterParams get_tissue_scatter_params(int tissue_type) {
    if (tissue_type < 0 || tissue_type >= NUM_TISSUE_TYPES)
        return TISSUE_PARAMS[0];
    return TISSUE_PARAMS[tissue_type];
}

// ---------------------------------------------------------------------------
// Compute optical properties at arbitrary wavelength
// ---------------------------------------------------------------------------
void compute_optical_properties(float wavelength_nm, OpticalProps props[NUM_TISSUE_TYPES]) {
    ChromophoreRow chrom = interpolate_chromophores(wavelength_nm);

    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        const TissueScatterParams& tp = TISSUE_PARAMS[t];

        if (t == TISSUE_AIR) {
            props[t] = {0.0f, 0.0f, 1.0f, 1.0f};
            continue;
        }

        // --- Reduced scattering: Mie power law ---
        // mu_s'(lambda) = a_mie * (lambda/500)^(-b_mie)
        float lambda_ratio = wavelength_nm / 500.0f;
        float mu_sp = tp.a_mie * powf(lambda_ratio, -tp.b_mie);

        // --- Anisotropy: weak wavelength dependence ---
        float g = tp.g_500 * powf(lambda_ratio, -tp.g_power);
        if (g > 0.99f) g = 0.99f;
        if (g < 0.0f) g = 0.0f;

        // --- Total scattering coefficient ---
        // mu_s = mu_s' / (1 - g)
        float mu_s = mu_sp / (1.0f - g);

        // --- Absorption: chromophore-based ---
        // Blood absorption: weighted sum of HbO2 and HHb
        // Typical hemoglobin concentration in blood: ~2.3 mM (150 g/L)
        float cHb_blood = 2.3f;  // mM total hemoglobin in whole blood
        float cHbO2 = tp.blood_volume_fraction * cHb_blood * tp.StO2;
        float cHHb  = tp.blood_volume_fraction * cHb_blood * (1.0f - tp.StO2);

        float mu_a = cHbO2 * chrom.eps_HbO2
                   + cHHb  * chrom.eps_HHb
                   + tp.water_fraction * chrom.mu_a_water
                   + tp.lipid_fraction * chrom.mu_a_lipid
                   + tp.baseline_mu_a;

        props[t].mu_a = mu_a;
        props[t].mu_s = mu_s;
        props[t].g    = g;
        props[t].n    = tp.n;
    }
}

// ---------------------------------------------------------------------------
// Print optical properties for debugging
// ---------------------------------------------------------------------------
void print_optical_properties(float wavelength_nm, const OpticalProps props[NUM_TISSUE_TYPES]) {
    const char* names[] = {"Air", "Scalp", "Skull", "CSF", "Gray Matter",
                           "White Matter", "Amygdala"};
    printf("  Optical properties at %.0f nm:\n", wavelength_nm);
    printf("  %-14s  %8s  %8s  %8s  %5s  %5s\n",
           "Tissue", "mu_a", "mu_s", "mu_s'", "g", "n");
    printf("  %-14s  %8s  %8s  %8s  %5s  %5s\n",
           "", "[1/mm]", "[1/mm]", "[1/mm]", "", "");
    for (int t = 0; t < NUM_TISSUE_TYPES; t++) {
        float mu_sp = props[t].mu_s * (1.0f - props[t].g);
        printf("  %-14s  %8.4f  %8.2f  %8.4f  %5.3f  %5.3f\n",
               names[t], props[t].mu_a, props[t].mu_s, mu_sp,
               props[t].g, props[t].n);
    }
}
