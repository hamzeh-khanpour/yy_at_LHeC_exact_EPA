# The `integrated_sleptons_cross_section` function computes the integrated sleptons production cross-section 
# for a given threshold energy `W0` and electron/proton beam energies `eEbeam` and `pEbeam`.
#   - It defines an integrand combining sleptons cross-section and the interpolated S_yy.
#   - If S_yy is zero at a given W, it skips that W value to avoid unnecessary computations.
#   - The result is integrated over W from W0 up to the maximum value set by `sqrt(s_cms)`.
#   - Integration warnings are caught, ensuring stable results.
#   - This function returns the integrated cross-section result in picobarns (pb).
# Hamzeh Khanpour, Laurent Forthomme, and Krzysztof Piotrzkowski -- November 2024


import numpy as np
import math
import scipy.integrate as integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


# Load photon luminosity data from the text file
#data = np.loadtxt('Elastic_Photon_Luminosity_Spectrum_q2emax_100000_q2pmax_10_using_vegas_FCC_he.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_10_q2emax_100000_q2pmax_10_using_vegas_FCC_he.txt', comments='#')
#data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_50_q2emax_100000_q2pmax_1000_using_vegas_FCC_he.txt', comments='#')
data = np.loadtxt('Inelastic_Photon_Luminosity_Spectrum_MNmax_300_q2emax_100000_q2pmax_100000_using_vegas_FCC_he.txt', comments='#')

W_data = data[:, 0]
S_yy_data = data[:, 1]


# Create an interpolation function for S_yy
S_yy_interp = interp1d(W_data, S_yy_data, kind='linear', bounds_error=False, fill_value=0.0)


# sleptons Production Cross-Section Calculation at Given W
def cs_sleptons_w_condition(wvalue):
    re = 2.8179403262e-15 * 137.0 / 128.0
    me = 0.510998950e-3
    msleptons = 100.0
    hbarc2 =  0.389
    alpha2 = (1.0/137.0)*(1.0/137.0)

    # Element-wise calculation of beta using np.where
    beta = np.sqrt(np.where(1.0 - 4.0 * msleptons * msleptons / wvalue**2.0 >= 0.0, 1.0 - 4.0 * msleptons * msleptons / wvalue**2.0, np.nan))

    # Element-wise calculation of cs using np.where
    cs = np.where(wvalue > msleptons, (2.0 * np.pi * alpha2 * hbarc2 ) / wvalue**2.0 * (beta) * \
             (2.0 - beta**2.0 - (1.0 - beta**4.0)/(2.0 * beta) * np.log( (1.0 + beta)/(1.0 - beta)) ), 0.0) * 1e9

    return cs



# Integrated sleptons Production Cross-Section from W_0 to sqrt(s_cms)
def integrated_sleptons_cross_section(W0, eEbeam, pEbeam):
    s_cms = 4.0 * eEbeam * pEbeam  # Center-of-mass energy squared

    def integrand(W):
        # Get the sleptons cross-section and S_yy value
        sleptons_cross_section = cs_sleptons_w_condition(W)
        S_yy_value = S_yy_interp(W)

        # Skip integration if S_yy is zero to avoid contributing zero values
        if S_yy_value == 0.0:
            #print(f"Skipping W={W} due to S_yy=0")
            return 0.0

        return sleptons_cross_section * S_yy_value     # to calculates the integrated sleptons cross-section

    try:
        result, _ = integrate.quad(integrand, W0, np.sqrt(s_cms), epsrel=1e-4)
    except integrate.IntegrationWarning:
        print(f"Warning: Integration for sleptons production cross-section did not converge for W_0={W0}")
        result = 0.0
    except Exception as e:
        print(f"Error during integration for sleptons production cross-section: {e}")
        result = 0.0
    return result



################################################################################


# Parameters
eEbeam = 60.0  # Electron beam energy in GeV
pEbeam = 50000.0  # Proton beam energy in GeV
mslepton = 100.0  # slepton mass in GeV 
W0_value = 2*mslepton  # GeV

# Calculate Integrated mu-mu Production Cross-Section at W_0 = 10 GeV
integrated_cross_section_value = integrated_sleptons_cross_section(W0_value, eEbeam, pEbeam)
print(f"Integrated inelastic mslepton Production Cross-Section at W_0 = {W0_value} GeV: {integrated_cross_section_value:.6e} pb")

    
################################################################################


