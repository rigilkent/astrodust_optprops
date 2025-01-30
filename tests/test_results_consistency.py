import pytest
import numpy as np
import optprops as opt
from pathlib import Path
from scipy.io import readsav

tests_dir = Path(__file__).parent

@pytest.mark.parametrize("star_name, lstar, tstar, mstar, spec_file, idl_outputs_file", [
    ('BBStar', 0.70, 5000, 0.90, None, 'optprops_test_blackbody.sav'),
    ('Vega',   47.2, None, 2.15, tests_dir / 'vega_spectrum.txt', 'optprops_test_vega.sav')
])
def test_consistency_with_legacy_idl(star_name, lstar, tstar, mstar, spec_file, idl_outputs_file):
    """
    Tests the consistency of the optical properties calculations against legacy IDL outputs.
    This includes verifying temperatures, scattering efficiencies, and spectral radiance values
    for different dust particle diameters and star configurations, both for actual spectra (Vega)
    and blackbody approximations.

    Parameters:
        lstar (float): Luminosity of the star in solar luminosities.
        tstar (float): Effective temperature of the star in Kelvin.
        mstar (float): Mass of the star in solar masses.
        spec_file (str or None): Filename of the spectral file, if applicable.
        idl_outputs_file (str): Filename of the IDL outputs saved in a .sav format for comparison.

    The function tests various optical properties like temperatures, Qabs, Qpr, Qsca, beta factors,
    and spectral radiance by comparing the computed values with the expected results from IDL.
    """
    # Load the appropriate expected outputs
    idl_outputs = readsav(tests_dir / idl_outputs_file)
    expected_temps = idl_outputs.temps.astype('float64').T      # n_diams x n_dists
    expected_Qabs = idl_outputs.Qabs.astype('float64')          # n_wavs x n_diams
    expected_Qpr = idl_outputs.Qpr.astype('float64')            # n_wavs x n_diams
    expected_Qsca = idl_outputs.Qsca.astype('float64')          # n_wavs x n_diams
    expected_betas = idl_outputs.betas.astype('float64')        # n_diams
    expected_diam_bl = idl_outputs.diam_bl.astype('float64')
    expected_bnus = idl_outputs.bnus.astype('float64')
    expected_bnus = np.transpose(expected_bnus, (2,1,0))        # n_wavs x n_diams x n_dists

    # Setup test parameters
    dists = np.array([1, 10, 100])
    wavs = np.array([.2, 2, 20, 200])
    diams = np.array([.1, 1, 10, 100, 1000])
    matrl = opt.Material(qsil=.5, qice=.5, mpor=.5)
    star = opt.Star(name=star_name, lum_suns=lstar, mass_suns=mstar, temp=tstar, spectrum_file=spec_file)
        
    # Call the functions from core module to get outputs
    temps = opt.calc_temperatures(diams, dists, matrl=matrl, star=star)
    Qabs = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl) for d in diams])
    Qpr = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl, qout=2) for d in diams])
    Qsca = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl, qout=3) for d in diams])
    betas = opt.calc_beta_factors(diams, matrl=matrl, star=star)
    diam_bl = opt.calc_blowout_diameters(matrl=matrl, star=star, beta_blow=0.5)
    bnus = opt.calc_spectral_radiances_for_all_temps(wavs, temps)

    # Assertions to compare Python outputs with IDL outputs
    np.testing.assert_allclose(temps, expected_temps, rtol=0.002)
    np.testing.assert_allclose(Qabs, expected_Qabs, rtol=0.001)
    np.testing.assert_allclose(Qpr, expected_Qpr, rtol=0.001)
    np.testing.assert_allclose(Qsca, expected_Qsca, rtol=0.01, atol=1e-10)
    np.testing.assert_allclose(betas, expected_betas, rtol=0.003)
    np.testing.assert_allclose(diam_bl, expected_diam_bl, rtol=0.001)
    np.testing.assert_allclose(bnus, expected_bnus, rtol=0.02) # Needs a bit higher tolerance because
    # even rel. dev. of temp of 0.001 can cause ~% changes in spect. radiance at wavs shortward of peak.

# Run file as script to execute all test defined in this file
if __name__ == '__main__':
    pytest.main([tests_dir / 'test_results_consistency.py', '--verbose'])



