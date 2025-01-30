import numpy as np
import pickle
import astropy.units as u
import astropy.constants as const
import importlib.resources as pkg_resources
from pathlib import Path
from optprops.core import calc_spectral_flux_density_bb

with pkg_resources.files('optprops') / 'material_data' as matrl_data_path:
    matrl_data_path = Path(matrl_data_path)

class Star:
    """
    This class represents a stellar object with properties such as temperature, luminosity,
    mass, and spectral characteristics. It can be initialized either as a blackbody with 
    a specified temperature or using a custom spectrum from a file.
    """
    def __init__(self, name='Sun', temp=None, lum_suns=1, mass_suns=1, spectrum_file=None, verbose=True):
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

        if temp is None and spectrum_file is None:
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
            spect_wavs (ndarray): Wavelengths in µm
            spect_flss (ndarray): Flux densities in W/m^2/µm
            spect_fnuss (ndarray): Flux densities in MJy
            spect_logwavs (ndarray): Log10 of wavelengths
            spect_logflss (ndarray): Log10 of flux densities
            spect_logfnuss (ndarray): Log10 of flux densities in frequency units
            temp (float): Star temperature in Kelvin derived from bolometric luminosity
        """
        
        # Read in the spectrum, i.e. the spectral flux density (F_lambda) in W/m^2/µm of the star.
        data = np.loadtxt(spectrum_file, skiprows=2)
        wavs = data[:, 0]  # Wavelength in µm
        flss = data[:, 1]  # Flux density in W/m^2/µm

        # Add units
        wavs *= u.um
        flss *= u.W / u.m**2 / u.um

        fnuss = (flss * wavs**2 / const.c).to(u.jansky)
        fnuss *= 1e-6 # This makes it MJy, to be strictly the same as IDL, however varialbe is unused.

        self.spect_wavs = wavs.value
        self.spect_flss = flss.value
        self.spect_fnuss = fnuss.value
        self.spect_logwavs = np.log10(wavs.value)
        self.spect_logflss = np.log10(flss.value)
        self.spect_logfnuss = np.log10(fnuss.value)

        # Calculate star temperature from its bolometric luminosity
        tstar = (np.trapezoid(flss, wavs) / const.sigma_sb) ** 0.25 
        # Could use simpson here for high accuracy (1 promill different)
        # but using trapz for comparison with legacy IDL
        self.temp = tstar.value
        if self.verbose:
            print(f'Star temperature derived from spectrum = {self.temp:.1f} K')


    def plot_spectrum(self, file_name=None):
        """Plots the spectrum of the star and that of a black body fit.
        Args:
            file_name (str, optional): Path where to save the plot. If None, saves to same 
                location as spectrum file but with .png extension.
        Returns:
            None: The method saves the plot to a file but does not return anything.
        """
        fig = plt.figure()
        plt.loglog(self.spect_wavs, self.spect_flss, label='from file')
        plt.loglog(self.spect_wavs, calc_spectral_flux_density_bb(self.spect_wavs, self.temp), label='BB fit')
        plt.xlabel('Wavelength (µm)')
        plt.ylabel('Flux (W/m²/µm)')
        plt.title(f'{self.name} Spectrum')
        plt.legend()

        if file_name is None:
            file_name_base = Path(self.spectrum_file)
            file_name = str(path.with_suffix('.png'))
        fig.savefig(file_name, bbox_inches='tight', pad_inches=0.01)

class Material:
    """
    Represents a composite material with dielectric properties.
    This class handles the calculation of effective dielectric constants for composite materials
    consisting of silicates, organics, water ice, and vacuum (porosity). It supports both
    Maxwell-Garnett and Bruggeman effective medium theories for computing the bulk properties.
    """
    def __init__(self, qsil=1.0/3.0, qice=0.0, mpor=0.0, cryst=False, emt='Maxwell-Garnett'):
        """
        Initialize a Material object.

        Args:
            qsil (float): Volume fraction of silicate in the core-mantle grain material, 
                          i.e., Vsi/(Vsi+Vor)
            qice (float): Volume fraction of water ice in the voids of the matrix of core-mantle grains,
                          i.e., Vh2o/(Vh2o+Vvac)
            mpor (float): Matrix porosity; volume fraction of voids in the matrix of core-mantle grains,
                          i.e., (Vh2o+Vvac)/(Vsi+Vor+Vh2o+Vvac)
            cryst (bool): Boolean indicating if the silicate is crystalline. 
                          Default is False (silicate is amorphous).
        """
        self.qsil = qsil
        self.qice = qice
        self.mpor = mpor
        self.cryst = cryst
        self.emt = emt

        # Add absolute volume fractions
        self.qsil_tot = qsil * (1 - mpor)
        self.qorg_tot = (1 - qsil) * (1 - mpor)
        self.qice_tot = qice * mpor
        self.poro_tot = (1 - qice) * mpor

        self.calculate_composite_material_props()

    def calculate_composite_material_props(self):
        """
        Calculates composite material properties using either Maxwell-Garnett or Bruggeman EMT.
        
        Returns:
            None. Updates the following instance attributes:
                eps (complex): Effective dielectric function of composite
                wavs (float): Wavelength grid points
                density (float): Bulk density of composite
                info (str): String describing material composition
        Raises:
            ValueError: If invalid EMT type is specified (must be 'Maxwell-Garnett'/'mg' or 'Bruggeman'/'bg')
        """
        if self.cryst:
            mat_pure_sil = self.get_pure_material_props('lg_crsil.dat')
            sil_state = '(cryst)'
        else:
            mat_pure_sil = self.get_pure_material_props('lg_amsil.dat')
            sil_state = '(amorph)'

        mat_pure_org = self.get_pure_material_props('lg_org.dat')
        mat_pure_ice = self.get_pure_material_props('lg_ice.dat')
        mat_pure_vac = {'matname': 'Vacuum', 'eps': 1.0, 'wavs': 0.0, 'density': 0.0}

        wavs = mat_pure_sil['wavs']

        if self.emt.lower() in ['maxwell-garnett', 'mg']:
            # Calculate the effective medium properties using Maxwell-Garnett theory. Order matters.
            # First, compute the effective properties of the core-mantle grains, making up the matrix.
            mat_refract = self.calculate_eff_medium_maxwell_garnett(
                mat_matrix=mat_pure_org, mat_incl=mat_pure_sil, q_incl=self.qsil, wavs=wavs)
            if self.qice == 0.0:
                mat_refract_in_ice = mat_refract
            else:
                # Second, compute the effective properties of the grains in the ice matrix.
                vol_frac_grains_in_ice = (1.0 - self.mpor) / (1.0 - self.mpor * (1.0 - self.qice))
                mat_refract_in_ice = self.calculate_eff_medium_maxwell_garnett(
                    mat_matrix=mat_pure_ice, mat_incl=mat_refract, q_incl=vol_frac_grains_in_ice, wavs=wavs)
            # Third, compute the effective properties of the grains-ice mixture in a 'vacuum-matrix'.
            vol_frac_solids_in_total = 1.0 - self.poro_tot
            mat_composite = self.calculate_eff_medium_maxwell_garnett(
                mat_matrix=mat_pure_vac, mat_incl=mat_refract_in_ice, q_incl=vol_frac_solids_in_total, wavs=wavs)
        elif self.emt.lower() in ['bruggeman', 'bg']:
            # Calculate the effective medium properties using Bruggeman theory.
            materials = [mat_pure_org, mat_pure_sil, mat_pure_ice, mat_pure_vac]
            fractions = [self.qorg_tot, self.qsil_tot, self.qice_tot, self.poro_tot]
            mat_composite = self.calculate_eff_medium_bruggeman(materials, fractions, wavs)
        else:
            raise ValueError("Invalid EMT specified. Choose either 'Maxwell-Garnett' or 'Bruggeman'.")

        
        self.eps = mat_composite['eps']
        self.wavs = mat_composite['wavs']
        self.density = mat_composite['density']
        self.info = f"Composite: qsil={self.qsil:.3f} {sil_state}, qice={self.qice:.3f}, mpor={self.mpor:.3f}"

    def get_pure_material_props(self, matrl_file):
        """
        Reads the dielectric constants vs wavelength for a material and its density from a file.

        Args:
            matrl_file (str): Path of file containing material data.

        Returns:
            dict: Dictionary containing material name, dielectric constants, wavelengths, and density.
        """
        
        with open(matrl_data_path / matrl_file, 'r') as file:
            matname = file.readline().strip()
            density = float(file.readline().strip())
            _ = file.readline().strip()
            n_eps = int(file.readline().strip())
            _ = file.readline().strip()
            wavs, eps_real, eps_imag = np.loadtxt(file, max_rows=n_eps, unpack=True)

        if not np.all(np.diff(wavs) > 0):
            wavs = wavs[::-1]
            eps_real = eps_real[::-1]
            eps_imag = eps_imag[::-1]

        eps = eps_real + 1j * eps_imag

        return {'matname': matname, 'eps': eps, 'wavs': wavs, 'density': density}

    def calculate_eff_medium_bruggeman(self, materials, fractions, wavs):
        """
        Calculates the effective dielectric constant using the Bruggeman rule.

        Args:
            materials (list of dict): List of dictionaries containing material properties.
            fractions (list of float): List of volume fractions for each material.
            wavs (numpy.ndarray): Wavelengths at which to calculate the composite material's dielectric constants.

        Returns:
            dict: Dictionary containing composite material name, dielectric constants, wavelengths, and density.
        """
        def bruggeman_eq(eps_eff, eps_list, f_list):
            return np.sum([f * (eps - eps_eff) / (eps + 2 * eps_eff) for eps, f in zip(eps_list, f_list)], axis=0)

        eps_list = [material['eps'] for material in materials]
        density_list = [material['density'] for material in materials]

        # Initial guess for eps_eff
        eps_eff_initial = np.mean(eps_list, axis=0)

        # Solve for eps_eff using fsolve
        eps_eff = fsolve(bruggeman_eq, eps_eff_initial, args=(eps_list, fractions))

        # Calculate the effective density
        density_eff = np.sum([f * density for f, density in zip(fractions, density_list)])

        matnamecomp = "Bruggeman Composite"

        return {'matname': matnamecomp, 'eps': eps_eff, 'wavs': wavs, 'density': density_eff}

    def calculate_eff_medium_maxwell_garnett(self, mat_matrix, mat_incl, q_incl, wavs=None):
        """
        Calculates the effective dielectric constants for a composite material
        composed of a matrix with inclusions using Maxwell-Garnett theory.

        Args:
            mat_matrix (dict): Dictionary containing properties of the matrix material:
                - 'wavs': wavelengths array (µm)
                - 'eps': dielectric constants array
                - 'density': density of material (g/cm^3)
                - 'matname': name of material
            mat_incl (dict): Dictionary containing properties of the inclusion material (same keys as mat_matrix).
            q_incl (float): Volume fraction of inclusions (between 0 and 1)
            wavs (array-like, optional): Target wavelength array for interpolation (µm).
                If None, uses the wavelengths from mat_incl. Defaults to None.
        Returns:
            dict: Dictionary containing the effective medium properties (same keys as mat_matrix).

        Notes:
            The Maxwell-Garnett formula assumes spherical inclusions and is valid for
            small volume fractions of inclusions.
        """
        if wavs is None:
            wavs = mat_incl['wavs']

        eps_incl = self.interpolate_eps(wavs_to=wavs, wavs_from=mat_incl['wavs'], eps=mat_incl['eps'])
        eps_matrix = self.interpolate_eps(wavs_to=wavs, wavs_from=mat_matrix['wavs'], eps=mat_matrix['eps'])

        x = q_incl * (eps_incl - eps_matrix) / (eps_incl + 2.0 * eps_matrix)
        epscomp = eps_matrix * (1 + 2.0 * x) / (1.0 - x)
        densitycomp = q_incl * mat_incl['density'] + (1.0 - q_incl) * mat_matrix['density']
        matnamecomp = f"({mat_incl['matname']}{q_incl:.3f}{mat_matrix['matname']})"

        return {'matname': matnamecomp, 'eps': epscomp, 'wavs': wavs, 'density': densitycomp}

    def interpolate_eps(self, wavs_to, wavs_from=None, eps=None):
        """
        Interpolates the complex dielectric function (epsilon) from the class's wavelengths to another set.

        Args:
            wavs_to (numpy.ndarray): Target set of wavelengths to interpolate to.
            wavs_from (numpy.ndarray, optional): Source set of wavelengths to interpolate from. Defaults to the class's wavelengths.
            eps (numpy.ndarray, optional): Source complex dielectric function values. Defaults to the class's eps.

        Returns:
            numpy.ndarray: Interpolated complex dielectric function values corresponding to `wavs_to`.
            If the source wavelengths and `wavs_to` are the same, returns the original `eps`.
        """
        if wavs_from is None:
            wavs_from = self.wavs
        if eps is None:
            eps = self.eps

        if np.array_equal(wavs_from, wavs_to) or np.isscalar(wavs_from):
            return eps

        if not np.all(np.diff(wavs_from) > 0):
            # raise Warning("Wavelengths are not in ascending order. This must not happen!")
            wavs_from = wavs_from[::-1]
            eps = eps[::-1]
        
        real_part = np.interp(wavs_to, wavs_from, eps.real)
        imag_part = np.interp(wavs_to, wavs_from, eps.imag)
        return real_part + 1j * imag_part

class OpticalModel:
    """
    Class to hold input class objects (Star and Material) as well as computed resulting optical properties.
    Used for saving of results.
    
    Attributes:
        star (Star): The star object containing star properties.
        matrl (Material): The material object containing material properties.
        temps (numpy.ndarray): Temperatures for each dust particle diameter and distance.
        Qabs (numpy.ndarray): Absorption efficiency for each dust particle diameter and wavelength.
        Qpr (numpy.ndarray): Radiation pressure efficiency for each dust particle diameter and wavelength.
        Qsca (numpy.ndarray): Scattering efficiency for each dust particle diameter and wavelength.
        beta (numpy.ndarray): Radiation pressure efficiency for each dust particle diameter.
        diam_blow (numpy.ndarray): Blowout diameters for each dust particle diameter.
        bnus (numpy.ndarray): Spectral radiance for each dust particle diameter, distance, and wavelength. 
    """
    def __init__(self, star, matrl, diams=None, dists=None, wavs=None, temps=None,
            Qabs=None, Qpr=None, Qsca=None, betas=None, diam_blow=None, bnus=None):
        self.star = star
        self.matrl = matrl
        self.diams = diams
        self.dists = dists
        self.wavs = wavs
        self.temps = temps
        self.Qabs = Qabs
        self.Qpr = Qpr
        self.Qsca = Qsca
        self.betas = betas
        self.diam_blow = diam_blow
        self.bnus = bnus
    
    def save(self, file_name):
        """
        Save the OpticalModel object to a file using pickle.
        
        Args:
            file_name (str): The name of the file to save the object to.
        """
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
        
    @staticmethod
    def load(file_name):
        """
        Load an OpticalModel object from a file using pickle.
        
        Args:
            file_name (str): The name of the file to load the object from.
        
        Returns:
            OpticalModel: The loaded OpticalModel object.
        """
        with open(file_name, 'rb') as file:
            return pickle.load(file)