import numpy as np
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from astrodust_optprops.optics_core import calculate_spectral_flux_density_bb

class Star:
    """
    This class represents a stellar object with properties such as temperature, luminosity,
    mass, and spectral characteristics. It can be initialized in three ways:
    
    1. **Blackbody approximation**: Specify a temperature
    2. **File-based spectrum**: Provide a path to a spectrum file
    3. **Array-based spectrum**: Provide wavelength and flux arrays with astropy units
    
    All initialization methods result in a unified interface for accessing stellar spectra
    via the `get_spectral_flux_density()` method.
    """
    def __init__(self, name='star', temp=None, lum_suns=1, mass_suns=1, spectrum_file=None, 
                 spectrum_wavs=None, spectrum_flux=None, verbose=True):
        """Initialize a Star object.

        Args:
            name (str, optional): Name of the star. Defaults to 'star'.
            temp (float, optional): Temperature of the star in Kelvin if using blackbody approximation.
                Defaults to None.
            lum_suns (float, optional): Luminosity in solar units. Defaults to 1.
            mass_suns (float, optional): Mass in solar units. Defaults to 1.
            spectrum_file (str, optional): Path to file containing custom stellar spectrum.
                Defaults to None.
            spectrum_wavs (array-like with units, optional): Wavelengths for custom spectrum.
                Should have astropy units of length (e.g., u.um, u.nm, u.angstrom).
            spectrum_flux (array-like with units, optional): Spectral flux density for custom spectrum.
                Should have astropy units of flux density (e.g., u.W/u.m**2/u.um, u.erg/u.s/u.cm**2/u.angstrom).
            verbose (bool, optional): Whether to print status messages. Defaults to True.

        Raises:
            ValueError: If conflicting or insufficient parameters are provided.

        Note:
            Exactly one of the following must be provided:
            - temp (for blackbody)
            - spectrum_file (for file-based spectrum)
            - spectrum_wavs AND spectrum_flux (for array-based spectrum)
        """
        self.name = name
        self.lum_suns = lum_suns
        self.mass_suns = mass_suns
        self.spectrum_file = spectrum_file
        self.verbose = verbose

        # Count how many spectrum sources are provided
        spectrum_sources = sum([
            temp is not None,
            spectrum_file is not None,
            spectrum_wavs is not None and spectrum_flux is not None
        ])

        # Special case for Sun
        if name.lower() == 'sun' and spectrum_sources == 0:
            self.temp = 5770
            self.is_blackbody = True
        elif spectrum_sources == 0:
            raise ValueError("One of the following must be provided: temp, spectrum_file, or (spectrum_wavs AND spectrum_flux)")
        elif spectrum_sources > 1:
            raise ValueError("Only one spectrum source should be provided: temp, spectrum_file, or (spectrum_wavs AND spectrum_flux)")
        elif temp is not None:
            self.is_blackbody = True
            self.temp = temp
        elif spectrum_file is not None:
            self.is_blackbody = False
            self.import_star_spectrum(spectrum_file)
        elif spectrum_wavs is not None and spectrum_flux is not None:
            self.is_blackbody = False
            self.import_star_spectrum_arrays(spectrum_wavs, spectrum_flux)
        else:
            raise ValueError("Invalid spectrum specification")
        

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

    def import_star_spectrum_arrays(self, spectrum_wavs, spectrum_flux):
        """Imports and processes a stellar spectrum from wavelength and flux arrays.
        
        Args:
            spectrum_wavs (array-like with units): Wavelengths with astropy units.
            spectrum_flux (array-like with units): Spectral flux density with astropy units.
                Can be either:
                - Per wavelength: W/m²/µm, erg/s/cm²/Å, etc.
                - Per frequency: Jy, W/m²/Hz, etc.
        """
        
        # Convert inputs to astropy quantities if they aren't already
        if not hasattr(spectrum_wavs, 'unit'):
            raise ValueError("spectrum_wavs must have astropy units (e.g., u.um, u.nm)")
        if not hasattr(spectrum_flux, 'unit'):
            raise ValueError("spectrum_flux must have astropy units")
            
        # Convert to numpy arrays with units
        wavs = u.Quantity(spectrum_wavs)
        flux = u.Quantity(spectrum_flux)
        
        # Validate wavelength units
        try:
            wavs_um = wavs.to(u.um)
        except u.UnitConversionError:
            raise ValueError("spectrum_wavs must have units of length (e.g., u.um, u.nm, u.angstrom)")
        
        # Try to convert flux - first attempt per-wavelength, then per-frequency
        try:
            # Try per-wavelength units first
            flux_lam = flux.to(u.W / u.m**2 / u.um)
            if self.verbose:
                print(f"   Using per-wavelength flux units: {flux.unit}")
        except u.UnitConversionError:
            try:
                # Try per-frequency units (like Jy)
                flux_nu = flux.to(u.Jy)
                # Convert F_ν to F_λ using: F_λ = F_ν * c / λ²
                flux_lam = (flux_nu * const.c / wavs_um**2).to(u.W / u.m**2 / u.um)
                if self.verbose:
                    print(f"   Converting from per-frequency units: {flux.unit} → W/m²/µm")
            except u.UnitConversionError:
                raise ValueError(
                    "spectrum_flux must have units of spectral flux density.\n"
                    "Acceptable units include:\n"
                    "- Per wavelength: u.W/u.m**2/u.um, u.erg/u.s/u.cm**2/u.angstrom\n" 
                    "- Per frequency: u.Jy, u.W/u.m**2/u.Hz"
                )
    
        # Ensure arrays are sorted by wavelength
        sort_idx = np.argsort(wavs.value)
        wavs = wavs[sort_idx]
        flux_lam = flux_lam[sort_idx]
        
        # Calculate flux in frequency units
        flux_nu = (flux_lam * wavs**2 / const.c).to(u.jansky)
        flux_nu *= 1e-6  # Convert to MJy for consistency with file import

        # Store the processed spectrum data
        self.wavs = wavs.value
        self.flux_lam = flux_lam.value
        self.flux_nu = flux_nu.value
        self.log_wavs = np.log10(wavs.value)
        self.log_flux_lam = np.log10(flux_lam.value)
        self.log_flux_nu = np.log10(flux_nu.value)

        # Calculate star temperature from its bolometric luminosity
        tstar = (np.trapezoid(flux_lam, wavs) / const.sigma_sb) ** 0.25
        self.temp = tstar.value
        if self.verbose:
            print(f'Star temperature derived from spectrum arrays: T_eff = {self.temp:.1f} K')


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
            wavs = np.logspace(-1, 3, 100)
            flux = self.get_spectral_flux_density(wavs)
            ax.loglog(wavs, flux, '--', color='k', linewidth=.5)
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

    def get_spectral_flux_density(self, wavs):
        """Get spectral flux density at specified wavelengths.
        
        This method provides a unified interface for getting stellar spectral flux density
        regardless of whether the star is modeled as a blackbody or uses a custom spectrum.
        
        Args:
            wavs (float or array-like): Wavelength(s) in µm at which to get flux density
            
        Returns:
            float or np.ndarray: Spectral flux density in W/m²/µm
        """
        wavs = np.atleast_1d(wavs)
        
        if self.is_blackbody:
            flux = calculate_spectral_flux_density_bb(wavs, self.temp)
        else:
            # Interpolate custom spectrum in log space for better accuracy
            logwavs = np.log10(wavs)
            log_flux = np.interp(logwavs, self.log_wavs, self.log_flux_lam)
            log_flux[np.isinf(log_flux)] = -50.0  # Handle extrapolation beyond data range
            flux = 10.0**log_flux
            
        # Return scalar if input was scalar
        if np.isscalar(wavs) or len(wavs) == 1:
            return flux.item() if hasattr(flux, 'item') else flux
        return flux