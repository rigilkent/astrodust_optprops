# particles.py
import numpy as np
import warnings
import astropy.constants as const
import astropy.units as u
import astrodust_optprops.optics_core as core
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import LogNorm

class Particles:
    def __init__(self, diams, wavs, matrl, dists=None, 
                 show_progress=True, precompute_Qs=True,
                 suppress_mie_resonance=False, size_averaging_window=2):
        """Initialize Particles object.
        
        Args:
            diams (array-like): Particle diameters in µm
            wavs (array-like): Wavelengths in µm
            matrl (Material): Material object containing dust properties
            dists (array-like, optional): Distances from star in au. Required for temperature calculations.
            show_progress (bool, optional): Whether to show progress bars for computations. Defaults to True.
            precompute_Qs (bool, optional): Whether to precompute scattering coefficients. Defaults to True.
            suppress_mie_resonance (bool, optional): Whether to suppress Mie resonances by averaging over nearby sizes. Defaults to False.
            size_averaging_window (float, optional): Determines the diameter averaging window in Mie resonances suppression. Defaults to 1.
            
        Raises:
            ValueError: If any of the required parameters is None
        """
        if any(x is None for x in [diams, wavs, matrl]):
            raise ValueError("diams, wavs, and matrl must be provided")
            
        if hasattr(diams, 'unit'):
            diams = diams.to('um').value
        else:
            print('Variable \'diams\' has no astropy.unit, assuming µm')
        self.diams = np.array(diams)
        self.wavs = np.array(wavs)
        self.matrl = matrl
        self.dists = np.array(dists) if dists is not None else None
        self.cross_areas = np.pi * (self.diams/2)**2 * u.um**2
        self.volumes = (4/3) * np.pi * (self.diams/2)**3 * u.um**3
        self.masses = (self.volumes * (self.matrl.density * u.g / u.cm**3)).to(u.g)

        self.n_wavs = len(wavs)
        self.n_diams = len(diams)
        self.n_dists = len(dists) if dists is not None else None
        
        self.temps = None
        self.Qabs = None
        self.Qpr = None
        self.Qsca = None
        self.betas = None
        self.bnus = None
        self.diams_blow = None

        self.show_progress = show_progress
        self.suppress_mie_resonance = suppress_mie_resonance
        self.size_averaging_window = size_averaging_window
        self.precompute_Qs = precompute_Qs

        if self.suppress_mie_resonance and not self.precompute_Qs:
            warnings.warn("Mie resonance suppression requires precomputed Q coefficients. "
                          "Setting precompute_Qs=True.", UserWarning)
            self.precompute_Qs = True

        # Optional: precomputation of coefficients (faster for if num(dists) high; needed for resoannce suppression)
        if self.precompute_Qs:
            # Create broad wavelength grid for precomputation
            self.precomputed_wavs = np.logspace(-1, 4, 1500) # >= 1500 for test to succeed
            self._precompute_coefficients()

    def _compute_coefficients_for_diameters(self, diameters):
        """Helper method to compute raw Q coefficients for a set of diameters."""
        Qabs = np.zeros((len(diameters), len(self.precomputed_wavs)))
        Qpr = np.zeros_like(Qabs)
        Qsca = np.zeros_like(Qabs)
        
        for i, diam in enumerate(tqdm(
            diameters,
            desc="Precomputing Q coefficients",
            disable=not self.show_progress,
            smoothing=.7,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'
        )):
            Qs = core.calculate_scatt_efficiency_coeffs(self.precomputed_wavs, diam, self.matrl)
            Qabs[i] = Qs['abs']
            Qpr[i] = Qs['pr']
            Qsca[i] = Qs['sca']
        
        return Qabs, Qpr, Qsca

    def _precompute_coefficients(self):
        """Initialize lookup tables for scattering coefficients.
        
        If suppression is enabled, computes coefficients on a fine diameter grid and averages 
        within log-normal symmetric windows around requested diameters to reduce Mie resonances.
        Otherwise, computes coefficients directly at requested diameters.
        """
        self.precomputed_Qabs = {}
        self.precomputed_Qpr = {}
        self.precomputed_Qsca = {}
        
        if self.suppress_mie_resonance:
            # Create a fine logarithmic grid of diameters
            min_diam = self.diams.min() / self.size_averaging_window
            max_diam = self.diams.max() * self.size_averaging_window
            points_per_decade = 150                 # No benefit above 150
            n_decades = np.log10(max_diam) - np.log10(min_diam)
            n_fine = int(points_per_decade * n_decades)
            fine_diams = np.logspace(np.log10(min_diam), np.log10(max_diam), n_fine)
            
            # Compute coefficients on fine grid
            fine_Qabs, fine_Qpr, fine_Qsca = self._compute_coefficients_for_diameters(fine_diams)
            
            # For each requested diameter, average over nearby grid points
            for diam in self.diams:
                # Define diameter window for averaging
                diam_min = diam / self.size_averaging_window
                diam_max = diam * self.size_averaging_window
                mask = (fine_diams >= diam_min) & (fine_diams <= diam_max)
                
                # Average coefficients within window
                # Use log-averaging (geometric mean) for coefficients to accurately represent 
                # multiplicative variations in particle sizes and prevent bias toward larger sizes, 
                # as confirmed by tests matching single-size coefficients in non-resonant regime.
                self.precomputed_Qabs[diam] = 10**np.mean(np.log10(fine_Qabs[mask]), axis=0)
                self.precomputed_Qpr[diam] = 10**np.mean(np.log10(fine_Qpr[mask]), axis=0)
                self.precomputed_Qsca[diam] = 10**np.mean(np.log10(fine_Qsca[mask]), axis=0)

        else:
            # If not suppressing, just compute at requested diameters
            Qabs, Qpr, Qsca = self._compute_coefficients_for_diameters(self.diams)
            for i, diam in enumerate(self.diams):
                self.precomputed_Qabs[diam] = Qabs[i]
                self.precomputed_Qpr[diam] = Qpr[i]
                self.precomputed_Qsca[diam] = Qsca[i]

    def get_Q_interpolator(self, diam, Q_type='abs'):
        """Returns a function that interpolates Q coefficients at given wavelengths.
        
        Args:
            diam (float): Particle diameter
            Q_type (str): Type of coefficient ('abs', 'pr', or 'sca')
        
        Returns:
            callable: Function that takes wavelength array and returns interpolated Q values
        
        Note:
            Uses logarithmic interpolation to handle large variations in Q values
            while ensuring they remain positive.
        """
        if not self.precompute_Qs:
            return None
            
        Q_dict = {
            'abs': self.precomputed_Qabs,
            'pr': self.precomputed_Qpr,
            'sca': self.precomputed_Qsca
        }
        
        Q_type = Q_type.lower()
        if diam in self.diams:
            # Direct lookup if diameter exists
            Q_values = Q_dict[Q_type][diam]
        else:
            # Find nearest diameters and interpolate Q values
            idx = np.searchsorted(self.diams, diam)
            if idx == 0:
                # Extrapolate below smallest diameter
                d1, d2 = self.diams[0], self.diams[1]
            elif idx == len(self.diams):
                # Extrapolate above largest diameter
                d1, d2 = self.diams[-2], self.diams[-1]
            else:
                # Interpolate between two diameters
                d1, d2 = self.diams[idx-1], self.diams[idx]
                
            # Log-space interpolation weight
            log_d = np.log10(diam)
            log_d1 = np.log10(d1)
            log_d2 = np.log10(d2)
            w = (log_d - log_d1) / (log_d2 - log_d1)
            
            # Get Q values and handle zeros/negatives before taking log
            Q1 = np.maximum(Q_dict[Q_type][d1], 1e-30)
            Q2 = np.maximum(Q_dict[Q_type][d2], 1e-30)
            
            # Interpolate in log space
            log_Q = (1 - w) * np.log10(Q1) + w * np.log10(Q2)
            Q_values = 10.0**log_Q

        def interpolator(wavs):
            # Log interpolation in wavelength dimension
            return 10.0**np.interp(
                np.log10(wavs),
                np.log10(self.precomputed_wavs),
                np.log10(np.maximum(Q_values, 1e-30)),
                left=-30, right=-30  # Will become 1e-30 after 10**
            )
            
        return interpolator

    # @profile
    def calculate_temperatures(self, star):
        """Calculate equilibrium temperatures for particles
        
        Args:
            star (Star): Star object containing stellar properties
            
        Returns:
            numpy.ndarray: Array of temperatures for each particle diameter and distance
        """
        if self.dists is None:
            raise ValueError("Distances (dists) must be set to calculate temperatures")
            
        n_diams = len(self.diams)
        n_dists = len(self.dists)
        self.temps = np.zeros((n_dists, n_diams))

        # Pre-calculate stellar spectrum quantities
        star_logwavs = core.get_logwav_integration_grid(star.temp)    
        star_flux = core.calculate_spectral_flux_density(star_logwavs, star=star)
        total_star_flux = core.integrate_log_spectrum(star_flux, star_logwavs)
        
        # Calculate blackbody temperatures at each distance as initial guesses
        temps_bb = np.array([core.calculate_blackbody_temp(star=star, dist=dist) 
                           for dist in self.dists])

        # Calculate absorption efficiency averaged over stellar spectrum
        stellar_qabs = np.zeros(n_diams)
        for i, diam in enumerate(self.diams):
            Q_interpolator = self.get_Q_interpolator(diam, 'abs') if self.precompute_Qs else None
            absorbed_flux = core.calculate_spectral_flux_density(
                star_logwavs, star=star, Q_type='abs', 
                diam=diam, matrl=self.matrl,
                Q_interpolator=Q_interpolator
            )
            total_absorbed_flux = core.integrate_log_spectrum(absorbed_flux, star_logwavs)
            stellar_qabs[i] = total_absorbed_flux / total_star_flux

        # Calculate temperatures for each distance and diameter
        distance_iter = tqdm(
            enumerate(self.dists),
            total=len(self.dists),
            desc="Calculating dust temperatures",
            disable=not self.show_progress,
            smoothing=0.1,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}]'
        )
        
        for dist_idx, dist in distance_iter:
            for diam_idx, diam in enumerate(self.diams):
                # Use temperature of previous particle diameter at same distance as initial guess
                temp_guess = temps_bb[dist_idx] if diam_idx == 0 else self.temps[dist_idx, diam_idx-1]

                Q_interpolator = self.get_Q_interpolator(diam, 'abs') if self.precompute_Qs else None
                temp_equil = self._calculate_equilibrium_temp(
                    diam=diam, dist=dist, temp_bb=temps_bb[dist_idx],
                    stellar_qabs=stellar_qabs[diam_idx], star=star,
                    initial_guess=temp_guess, Q_interpolator=Q_interpolator
                )
                self.temps[dist_idx, diam_idx] = temp_equil
        
        return self.temps

    def _calculate_equilibrium_temp(self, diam, dist, temp_bb, stellar_qabs, star,
                                  initial_guess=100, max_iterations=100, tolerance=0.001, 
                                  Q_interpolator=None):
        """Calculate equilibrium temperature for a single particle.
        
        Args:
            diam (float): Particle diameter in µm
            dist (float): Distance from star in au
            temp_bb (float): Blackbody temperature at this distance
            stellar_qabs (float): Absorption efficiency averaged over stellar spectrum
            star (Star): Star object
            initial_guess (float): Initial temperature guess. Defaults to 100 K.
            max_iterations (int, optional): Maximum number of iterations. Defaults to 100.
            tolerance (float, optional): Convergence tolerance. Defaults to 0.001.
            Q_interpolator (callable, optional): Function that interpolates Qabs at given wavelengths.
            
        Returns:
            float: Equilibrium temperature in Kelvin
        """
        curr_temp = initial_guess
        prev_temp = curr_temp
        prev_diff = None
        damping = 1

        for iter_count in range(max_iterations):
            # Calculate blackbody flux at current temperature
            bb_flux = const.sigma_sb.value * curr_temp**4
            curr_logwavs = core.get_logwav_integration_grid(curr_temp)
            
            # Calculate absorption averaged over emission spectrum
            emitted_flux = core.calculate_spectral_flux_density(
                curr_logwavs, temp=curr_temp, Q_type='abs',
                diam=diam, matrl=self.matrl, Q_interpolator=Q_interpolator
            )
            total_emitted_flux = core.integrate_log_spectrum(emitted_flux, curr_logwavs)
            curr_qabs = total_emitted_flux / bb_flux

            # Calculate temperature difference
            target_temp = temp_bb * np.sqrt(np.sqrt(stellar_qabs / curr_qabs))
            temp_diff = curr_temp - target_temp

            # Update temperature
            if prev_diff is None:
                # First iteration - use simple step
                new_temp = curr_temp - temp_diff
            elif (temp_diff * prev_diff) < 0 or abs(temp_diff - prev_diff) < 0.1:
                # Sign flip occurred or denominator is < 0.1 K - use simple damped step to avoid oscillation
                damping *= 0.75
                new_temp = curr_temp - damping * temp_diff
            else:
                # Otherwise - use secant method
                new_temp = curr_temp - temp_diff * (curr_temp - prev_temp) / (temp_diff - prev_diff)
            
            if abs(new_temp / curr_temp - 1) < tolerance:
                return new_temp
            
            prev_temp = curr_temp
            prev_diff = temp_diff
            curr_temp = max(new_temp, 1.0)  # Ensure temperature is positive

        raise RuntimeError(f"Temperature calculation did not converge for diameter={diam}, distance={dist}")

    def calculate_scattering_properties(self):
        """Calculate Qabs, Qpr, and Qsca for the particles"""
        self.Qabs = np.zeros((len(self.diams), len(self.wavs)))
        self.Qpr = np.zeros_like(self.Qabs)
        self.Qsca = np.zeros_like(self.Qabs)

        for i, diam in enumerate(self.diams):
            if self.precompute_Qs:
                # Use precomputed values via interpolation
                self.Qabs[i] = self.get_Q_interpolator(diam, 'abs')(self.wavs)
                self.Qpr[i] = self.get_Q_interpolator(diam, 'pr')(self.wavs)
                self.Qsca[i] = self.get_Q_interpolator(diam, 'sca')(self.wavs)
            else:
                # Calculate directly
                Qs = core.calculate_scatt_efficiency_coeffs(self.wavs, diam, matrl=self.matrl)
                self.Qabs[i] = Qs['abs']
                self.Qpr[i] = Qs['pr']
                self.Qsca[i] = Qs['sca']
        
        return self.Qabs, self.Qpr, self.Qsca

    def calculate_beta_factors(self, star, diams=None, n_step=400):
        """Calculate beta factors (ratio of radiation to gravitational force)
        
        Args:
            star (Star): Star object containing stellar properties
            diams (float or array-like, optional): Specific diameter(s) to calculate betas for.
                If None, uses self.diams and stores results in self.betas.
            n_step (int, optional): Number of integration steps. Defaults to 400.
            
        Returns:
            numpy.ndarray: Beta factors for each particle diameter. If input diams was scalar,
                          returns a scalar result.
        """
        # Handle scalar input
        input_was_scalar = np.isscalar(diams)
        diams_to_use = np.atleast_1d(diams) if diams is not None else self.diams
        
        logwavs = core.get_logwav_integration_grid(star.temp, n_step=n_step)
        
        # Calculate total stellar flux
        star_flux = core.calculate_spectral_flux_density(logwavs, star=star)
        star_flux_tot = core.integrate_log_spectrum(flux=star_flux, logwavs=logwavs)

        # Calculate Qpr averaged over stellar spectrum
        qpr_star_flux_tot = np.zeros_like(diams_to_use)
        for i, diam in enumerate(diams_to_use):
            Q_interpolator = self.get_Q_interpolator(diam, 'pr') if self.precompute_Qs else None
            qpr_star_flux = core.calculate_spectral_flux_density(
                logwavs, star=star, Q_type='pr',
                diam=diam, matrl=self.matrl, 
                Q_interpolator=Q_interpolator
            )
            qpr_star_flux_tot[i] = core.integrate_log_spectrum(flux=qpr_star_flux, logwavs=logwavs)
        
        effective_qpr = qpr_star_flux_tot / star_flux_tot

        # Calculate beta factors
        constant = 0.7652   # = (Lsun/(4*pi*c*G*Msun))*1e3 in (g/cm^3)µm, where
                            # Lsun = 3.826e26 W, Msun = 1.989e30 kg, 
                            # c = 2.997924580e8 m/s, G = 6.67259e-11 m^3/kg/s^2
        betafacts = constant * (1.5 / (self.matrl.density * diams_to_use)) * \
                    (star.lum_suns / star.mass_suns)
        betas = betafacts * effective_qpr
        
        # Only store results if using self.diams
        if diams is None:
            self.betas = betas

        # Return scalar if input was scalar
        if input_was_scalar:
            return betas[0]
            
        return betas

    def calculate_blowout_diameters(self, star, beta_blow=0.5):
        """
        Calculate the diameter range of grains that are blown out by radiation pressure.
        
        This method determines the size range of dust particles that will be ejected from 
        the system due to radiation pressure overcoming gravitational force. It uses an
        iterative process to find the particle diameters where beta = beta_blow.

        Args:
            star (Star): Star object containing stellar properties
            beta_blow (float, optional): Critical beta value for blowout. Defaults to 0.5.
                For circular orbits, particles are blown out when beta > 0.5.
                For highly eccentric orbits, blowout occurs at beta > ~1.0.

        Returns:
            numpy.ndarray: Array containing [lower_diameter, upper_diameter] in µm,
                          or [None, None] if no particles are blown out.
                          lower_diameter = 0 means blowout occurs down to smallest sizes.

        Note:
            Beta is the ratio of radiation pressure force to gravitational force.
            For most stars, there is a range of particle sizes that get blown out,
            typically with a lower and upper diameter limit.
        """
        # Define diameter range for testing
        n_test_diams = 30
        test_diams = np.logspace(-2, 3, n_test_diams)  # 0.01 to 1000 µm
        
        # Get initial rough estimate of beta values at the test diameters
        test_betas = self.calculate_beta_factors(star, diams=test_diams, n_step=20)
        k = np.where(test_betas > beta_blow)[0]
        
        if len(k) == 0:
            return (None, None)  # No particles blown out
            
        # Find upper diameter limit
        # Calculate accurate beta values for the diameter limits around the threshold
        kup = k.max()
        if kup < n_test_diams - 1:
            test_diams_near = test_diams[kup:kup+2]
        else:
            test_diams_near = test_diams[-2:]       # If all beta>beta_blow, use last two values

        test_betas_near = self.calculate_beta_factors(star, diams=test_diams_near)
        diam_up = 10**np.interp(np.log10(beta_blow), 
                                np.log10(test_betas_near[::-1]), 
                                np.log10(test_diams_near[::-1]))
        beta_up = self.calculate_beta_factors(diams=diam_up, star=star)
        if not np.isclose(beta_up, beta_blow, rtol=0.03):
            warnings.warn(f"""Error or inaccuracy in calculating upper size limit.
                Requested beta: {beta_blow}. Actual beta at the upper size limit: {beta_up}.""", RuntimeWarning)
            
        # Find lower diameter limit
        if k.min() > 0:
            test_diams_near = test_diams[k.min()-1:k.min()+1]
            test_betas_near = self.calculate_beta_factors(star, diams=test_diams_near)
            diam_low = 10**np.interp(np.log10(beta_blow),
                                    np.log10(test_betas_near),
                                    np.log10(test_diams_near))
            beta_low = self.calculate_beta_factors(diams=diam_low, star=star)
            if not np.isclose(beta_low, beta_blow, rtol=0.03):
                warnings.warn(f"""Error or inaccuracy in calculating lower size limit.
                    Requested beta: {beta_blow}. Actual beta at the lower size limit: {beta_low}""", RuntimeWarning)
        else:
            diam_low = 0
            
        # Create result tuple
        diams_blow = (diam_low, diam_up)
        
        # Only store in instance if using default beta_blow
        if beta_blow == 0.5:
            self.diams_blow = diams_blow
            
        return diams_blow

    def calculate_all(self, star):
        """Calculate all optical properties
        
        Args:
            star (Star): Star object containing stellar properties
        """
        if self.dists is None:
            raise ValueError("Distances (dists) must be set to compute all properties")
            
        self.calculate_scattering_properties()
        self.calculate_temperatures(star)
        self.calculate_beta_factors(star)
        self.calculate_blowout_diameters(star)
        self.bnus = self.calculate_spectral_radiance_bb(self.wavs, self.temps)

    @staticmethod
    def calculate_spectral_radiance_bb(wavs, temps, domain='freq'):
        """Simple wrapper for core.calculate_spectral_radiance_bb.
        
        This static method provides direct access to the optprops.optics_core function 
        which automatically handles input array shapes.
        
        Args:
            wavs (array-like): Wavelengths in µm
            temps (array-like): Temperature(s) in K. Can have any shape.
            domain (str, optional): 'freq' for frequency domain (Jy/sr) or 'wav' for wavelength
                domain (W/m^2/sr/µm). Defaults to 'freq'.
                
        Returns:
            numpy.ndarray: Spectral radiance in specified domain units
        """
        return core.calculate_spectral_radiance_bb(wavs, temps, domain=domain)

    def interpolate_temperatures(self, target_distances):
        """Interpolates temperatures for particles based on radial distances.

        Args:
            target_distances: Contains the astocentric distances (in au) where temperatures
                              should be interpolated to. Must be a 1D array of distances.

        Sets:
            self.temps with interpolated temperature array of shape (rbin.num, self.n_diams).

        Note:
            Original temperature data comes from self._optprops_prtl which must have
              'dists' (distances in au) and 'temps' attributes
        """
        if self.temps is None:
            raise ValueError("Temperatures have not been calculated. Run calculate_temperatures() first.")
        
        log_target_temps = np.zeros((len(target_distances), self.n_diams))
        for iD in range(self.n_diams):          # interpolate in log space
            log_target_temps[:, iD] = np.interp(np.log10(target_distances), np.log10(self.dists), np.log10(self.temps[:, iD]))
        target_temps = 10**log_target_temps     # return to lin space
        return target_temps

    def plot_Qabs(self, ax=None, diams=None, as_contour=False, n_contour_levels=100, add_contour_lines=[.1, 1]):
        """Plot absorption efficiency (Qabs) as function of wavelength.

        Can create either line plots for specific particle diameters or a 2D contour plot
        showing Qabs for all particle sizes and wavelengths.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            diams (array-like, optional): Diameters to plot in line plot mode. If None and
                as_contour=False, uses [3, 10, 30, 100, 300, 1000] µm. Ignored if as_contour=True.
            as_contour (bool, optional): If True, creates contour plot. If False, creates line
                plots. Defaults to False.
            n_contour_levels (int, optional): Number of contour levels in 2D plot. Only used
                if as_contour=True. Defaults to 100.
            add_contour_lines (list, optional): Values at which to add contour lines in 2D
                plot. Only used if as_contour=True. Set to None to disable. Defaults to [0.1, 1].

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
            
        if as_contour==True:
            if diams is not None:
                warnings.warn("plot_Qabs(): Input 'diams' is ignored when as_contour=True", UserWarning)
            contour = ax.contourf(self.wavs, self.diams, self.Qabs,
                                 levels=np.logspace(-2, np.log10(2), n_contour_levels),
                                 norm=LogNorm(vmin=1e-2, vmax=2),
                                 extend='min')
            
            cbar = plt.colorbar(contour, ax=ax, label=r'$Q_\mathrm{abs}$', pad=.01)
            cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=11)
            cbar.set_ticks([.01, .1, 1])
            cbar.ax.minorticks_on()
            if add_contour_lines is not None:
                contour_lines = ax.contour(self.wavs, self.diams, self.Qabs, levels=add_contour_lines, 
                       colors='k', linewidths=.3, linestyles='dashed')
                cbar.add_lines(contour_lines)
            ax.set_ylabel(r'Particle diameter (µm)')
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            ax.hlines(1, xmin=0, xmax=1e4, linestyle='--', linewidth=.8, color='black', alpha=.5)
            if diams is None:
                diams = np.logspace(.5,3,6)
            diams = diams[np.logical_and(diams >= self.diams.min(), diams <= self.diams.max())]
            # Create color progression using a colormap
            colors = plt.cm.winter(np.linspace(0, 1, len(diams)))
            for diam, color in zip(diams, colors):
                idx = np.argmin(np.abs(self.diams - diam))
                ax.loglog(self.wavs, self.Qabs[idx, :], label=f'{self.diams[idx]:.0f} µm', color=color, zorder=self.n_diams-idx)
            ax.legend(title="Particle diameters:", loc='lower left', framealpha=1)
            ax.set_ylabel(r'$Q_\mathrm{abs}$', fontsize=12)
            ax.set_ylim(1e-2, 2)
            ax.set_xlim(self.wavs.min(), self.wavs.max())

        ax.set_xlabel('Wavelength (µm)')

        return ax

    def plot_beta(self, ax=None, ylog=False, marker='None'):
        """Plot beta factors as a function of particle size.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        
        if self.betas is None:
            raise ValueError("""Beta factors have not been calculated.
                                Run calculate_beta_factors() first.""")

        ax.plot(self.diams, self.betas, marker=marker, linestyle='-', zorder=1)
        ax.axhline(y=1, color='grey', linestyle='--', linewidth=0.8, zorder=0)

        ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel('Particle diameter (µm)')
        ax.set_ylabel(r'$\beta$')
        

        return ax

    def plot_temp(self, ax=None, n_contour_levels=100, add_contour_lines=[110], add_blowout=True):
        """Plot equilibrium temperatures as a function of distance and particle size.

        Can create either line plots for specific distances or a 2D contour plot
        showing temperatures for all particle sizes and distances.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            n_contour_levels (int, optional): Number of contour levels in 2D plot. Only used
                if as_contour=True. Defaults to 100.
            add_contour_lines (list, optional): Values at which to add contour lines in 2D
                plot. Only used if as_contour=True. Set to None to disable. Defaults to [100, 200, 300].

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if self.temps is None:
            raise ValueError("Temperatures have not been calculated. Run calculate_temperatures() first.")

        if ax is None:
            _, ax = plt.subplots()


        clim = [self.temps.min(), self.temps.max()]

        contour = ax.contourf(self.dists, self.diams, self.temps.T, cmap='afmhot',
                                levels=np.linspace(clim[0], clim[1], n_contour_levels))

        cbar = plt.colorbar(contour, ax=ax, label='Temperature (K)', pad=.01)
        cbar.set_ticks([tick for tick in np.linspace(0, 10e3, 51) if clim[0] <= tick <= clim[1]])
        cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=11)
        if add_contour_lines is not None:
            contour_lines = ax.contour(self.dists, self.diams, self.temps.T, levels=add_contour_lines, 
                                        colors='cyan', linewidths=1)
            ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.0f K', inline_spacing=-17)
            cbar.add_lines(contour_lines)

        if add_blowout:
            if self.diams_blow is not None:
                if self.diams_blow[0] > 0:
                    ax.axhline(y=self.diams_blow[0], color='grey', linestyle='--', linewidth=0.8, zorder=1)
                ax.axhline(y=self.diams_blow[1], color='grey', linestyle='--', linewidth=0.8, zorder=1)
                ax.text(self.dists.max()/2, self.diams_blow[1], r'$\beta=0.5$', 
                        ha='right', va='bottom', color='grey', fontsize=10)
            else:
                warnings.warn("plot_temp(): Blowout diameters have not been calculated.", UserWarning)

        ax.set_xlabel('Distance (au)')
        ax.set_ylabel('Particle diameter (µm)')
        ax.set_xscale('log')
        ax.set_yscale('log')

        return ax

    def plot_Q(self, ax=None, diams=None, as_contour=False, n_contour_levels=100, add_contour_lines=[.1, 1], Q_type='abs'):
        """Plot optical efficiency (Qabs, Qsca, or Qpr) as a function of wavelength.

        Can create either line plots for specific particle diameters or a 2D contour plot
        showing the selected Q property for all particle sizes and wavelengths.

        Args:
            ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, creates new figure.
            diams (array-like, optional): Diameters to plot in line plot mode. If None and
                as_contour=False, uses [3, 10, 30, 100, 300, 1000] µm. Ignored if as_contour=True.
            as_contour (bool, optional): If True, creates contour plot. If False, creates line
                plots. Defaults to False.
            n_contour_levels (int, optional): Number of contour levels in 2D plot. Only used
                if as_contour=True. Defaults to 100.
            add_contour_lines (list, optional): Values at which to add contour lines in 2D
                plot. Only used if as_contour=True. Set to None to disable. Defaults to [0.1, 1].
            Q_type (str, optional): Type of optical efficiency to plot ('abs', 'sca', or 'pr').
                Defaults to 'abs'.

        Returns:
            matplotlib.axes.Axes: The axes containing the plot.
        """
        if Q_type not in ['abs', 'sca', 'pr']:
            raise ValueError("Invalid Q_type. Must be one of 'abs', 'sca', or 'pr'.")

        Q_data = getattr(self, f"Q{Q_type}")

        if ax is None:
            _, ax = plt.subplots()

        if as_contour:
            if diams is not None:
                warnings.warn("plot_Q(): Input 'diams' is ignored when as_contour=True", UserWarning)
            contour = ax.contourf(self.wavs, self.diams, Q_data,
                                  levels=np.logspace(-2, np.log10(2), n_contour_levels),
                                  norm=LogNorm(vmin=1e-2, vmax=2),
                                  extend='min')

            cbar = plt.colorbar(contour, ax=ax, label=rf'$Q_\mathrm{{{Q_type}}}$', pad=.01)
            cbar.ax.set_ylabel(cbar.ax.get_ylabel(), fontsize=11)
            cbar.set_ticks([.01, .1, 1])
            cbar.ax.minorticks_on()
            if add_contour_lines is not None:
                contour_lines = ax.contour(self.wavs, self.diams, Q_data, levels=add_contour_lines,
                                           colors='k', linewidths=.3, linestyles='dashed')
                cbar.add_lines(contour_lines)
            ax.set_ylabel(r'Particle diameter (µm)')
            ax.set_yscale('log')
            ax.set_xscale('log')
        else:
            ax.hlines(1, xmin=0, xmax=1e4, linestyle='--', linewidth=.8, color='black', alpha=.5)
            if diams is None:
                diams = np.logspace(.5, 3, 6)
            diams = diams[np.logical_and(diams >= self.diams.min(), diams <= self.diams.max())]
            # Create color progression using a colormap
            colors = plt.cm.winter(np.linspace(0, 1, len(diams)))
            for diam, color in zip(diams, colors):
                idx = np.argmin(np.abs(self.diams - diam))
                ax.loglog(self.wavs, Q_data[idx, :], label=f'{self.diams[idx]:.0f} µm', color=color, zorder=self.n_diams - idx)
            ax.legend(title="Particle diameters:", loc='lower left', framealpha=1)
            ax.set_ylabel(rf'$Q_\mathrm{{{Q_type}}}$', fontsize=12)
            ax.set_ylim(1e-2, max(np.ceil(Q_data.max()), 2))
            ax.set_xlim(self.wavs.min(), self.wavs.max())

        ax.set_xlabel('Wavelength (µm)')

        return ax
