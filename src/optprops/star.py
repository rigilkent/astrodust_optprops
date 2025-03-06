import numpy as np
import astropy.units as u
import astropy.constants as const
from optprops.optics_core import calculate_spectral_flux_density_bb
import matplotlib.pyplot as plt

class Star:
    """
    This class represents a stellar object with properties such as temperature, luminosity,
    mass, and spectral characteristics. It can be initialized either as a blackbody with 
    a specified temperature or using a custom spectrum from a file.
    """
    def __init__(self, name='star', temp=None, lum_suns=1, mass_suns=1, spectrum_file=None, verbose=True):
        """Initialize a Star object.

        Args:
            name (str, optional): Name of the star. Defaults to 'Sun'.
            temp (float, optional): Temperature of the star in Kelvin if using blackbody approximation.
                Defaults to None.
            lum_suns (float, optional): Luminosity in solar units. Defaults to 1.
            mass_suns (float, optional): Mass in solar units. Defaults to 1.
            spectrum_file (str, optional): Path to file containing custom stellar spectrum.
                Defaults to None.
            verbose (bool, optional): Whether to print status messages. Defaults to True.

        Raises:
            ValueError: If neither temperature nor spectrum file is provided.

        Note:
            Either temp or spectrum_file must be provided to initialize the star object.
        """
        self.name = name
        self.lum_suns = lum_suns
        self.mass_suns = mass_suns
        self.spectrum_file = spectrum_file
        self.verbose = verbose

        if name.lower()=='sun' and spectrum_file is None:
            self.temp = 5770
            self.is_blackbody = True
        elif temp is None and spectrum_file is None:
            raise ValueError("Either temp or spectrum_file must be supplied.")
        elif temp is not None and spectrum_file is not None:
            raise ValueError("Only one of temp or spectrum_file should be supplied.")
        elif temp is None:
            self.is_blackbody = False
            self.import_star_spectrum(spectrum_file)
        else:
            self.is_blackbody = True
            self.temp = temp
        

    def import_star_spectrum(self, spectrum_file):
        """Imports and processes a stellar spectrum from a file.
        This method reads a spectrum file containing wavelength and flux density data,
        converts the units, calculates various spectral quantities, and determines the
        star's temperature from its bolometric luminosity.
        Args:
            spectrum_file (str): Path to the spectrum file. File should contain two columns:
                wavelength (µm) and spectral flux density (W/m^2/µm), with 2 header rows.
        Returns:
            None
        Attributes Modified:
            wavs (ndarray): Wavelengths in µm
            flux_lam (ndarray): Flux densities in W/m^2/µm
            flux_nu (ndarray): Flux densities in MJy
            log_wavs (ndarray): Log10 of wavelengths
            log_flux_lam (ndarray): Log10 of flux densities
            log_flux_nu (ndarray): Log10 of flux densities in frequency units
            temp (float): Star temperature in Kelvin derived from bolometric luminosity
        """
        
        # Read in the spectrum, i.e. the spectral flux density (F_lambda) in W/m^2/µm of the star.
        data = np.loadtxt(spectrum_file, skiprows=2)
        wavs = data[:, 0]  # Wavelength in µm
        flux_lam = data[:, 1]  # Flux density in W/m^2/µm

        # Add units
        wavs *= u.um
        flux_lam *= u.W / u.m**2 / u.um

        flux_nu = (flux_lam * wavs**2 / const.c).to(u.jansky)
        flux_nu *= 1e-6  # This makes it MJy, to be strictly the same as IDL. Variable is unused.

        self.wavs = wavs.value
        self.flux_lam = flux_lam.value
        self.flux_nu = flux_nu.value
        self.log_wavs = np.log10(wavs.value)
        self.log_flux_lam = np.log10(flux_lam.value)
        self.log_flux_nu = np.log10(flux_nu.value)

        # Calculate star temperature from its bolometric luminosity
        tstar = (np.trapezoid(flux_lam, wavs) / const.sigma_sb) ** 0.25 
        # Could use simpson here for high accuracy (1 promill different)
        # but using trapz for comparison with legacy IDL
        self.temp = tstar.value
        if self.verbose:
            print(f'Star temperature derived from spectrum: T_eff = {self.temp:.1f} K')


    def plot_spectrum(self, ax=None, show_bb_fit=True):
        """Plot stellar spectrum and optionally its blackbody fit.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            show_bb_fit (bool, optional): Whether to plot blackbody fit. Defaults to True.

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
            
        Note:
            If star is initialized with temperature (blackbody), only shows the blackbody curve.
            If initialized with spectrum file, shows actual spectrum and optionally BB fit.
        """
        if ax is None:
            _, ax = plt.subplots()
            
        if self.is_blackbody:
            wavs = np.logspace(-1,3,100)
            bb_flux = calculate_spectral_flux_density_bb(wavs=wavs, temp=self.temp)
            ax.loglog(wavs, bb_flux, '--', color='k', linewidth=.5)
        else:
            ax.loglog(self.wavs, self.flux_lam, label='Spectrum', zorder=3)
            if show_bb_fit:
                bb_flux = calculate_spectral_flux_density_bb(self.wavs, self.temp)
                ax.loglog(self.wavs, bb_flux, '--', color='k', linewidth=.5, 
                    label=r'Blackbody with $T_\mathrm{eff}=$' + f'{self.temp:.0f} K')
                ax.legend()

        ax.set_xlabel('Wavelength (µm)')
        ax.set_ylabel('Flux (W/m²/µm)')
        
        return ax