import numpy as np
from numba import jit


def get_logwav_integration_grid(temperature, n_step=400):
    """
    Returns an array of log10(λ) over the wavelength range a blackbody spectrum 
    of a given temperature should be integrated.

    Args:
        temperature (float): The temperature of the blackbody in Kelvin.
        n_step (int, optional): The number of steps in the wavelength range. Defaults to 400.
    Returns:
        np.ndarray: An array of log10(λ) values over the specified wavelength range.
    """
    if temperature <= 0:
        raise ValueError("Temperature must be greater than 0 K")
    wav_1 = 289.8 / temperature
    wav_2 = 2898000.0 / temperature
    logwav_1 = np.log10(wav_1)
    logwav_2 = np.log10(wav_2)
    logwavs = np.linspace(logwav_1, logwav_2, n_step)

    return logwavs

def calculate_spectral_flux_density(logwavs, temp=None, star=None, diam=None, 
                                    Q_interpolator=None, Q_type=None, matrl=None):
    """
    Calculate Q*π*B_λ, the efficiency-weighted spectral flux density.

    Product includes:
    - Q: Scattering/absorption/pr efficiency
    - π*B_λ: Blackbody spectral flux density (=F_λ) 

    Args:
        logwavs (float or array-like): Logarithm (base 10) of the wavelength(s).
        temp (float, optional): Temperature of the black body. Either 'temp' or 'star' must be provided.
        star (Star, optional): Star object containing the star's properties. Either 'temp' or 'star' must be provided.
        diam (float, optional): Diameter of the particle.
        Q_interpolator (callable, optional): Function that takes wavelength array and returns Q coefficients.
                                           If provided, used instead of calculating coefficients directly.
        Q_type (str or None, optional): Type of coefficient. One of:
            None: returns unmodified flux (Q=1.0)
            'abs': returns absorption-weighted flux
            'pr': returns radiation pressure-weighted flux
            'sca': returns scattering-weighted flux
        matrl (Material, optional): Material object containing the material properties. Required if Q_type != 'abs'.

    Returns:
        np.ndarray: The calculated Q*pi*B_λ values.

    Note:
        The factor "pi" converts spectral radiance, B_λ (W/m^2/µm/sr), into spec. flux density, F_λ (W/m^2/µm).
    """

    if temp is None and star is None:
        raise ValueError("Either 'temp' or 'star' must be provided.")
    if Q_interpolator is None and matrl is None and Q_type is not None:
        raise ValueError("If 'Q_interpolator' is not provided, 'matrl' AND 'Q_type' must be provided.")

    wavs = 10.0**logwavs
    
    if temp is not None:
        flux = calculate_spectral_flux_density_bb(wavs, temp)
    elif star.is_blackbody:
        flux = calculate_spectral_flux_density_bb(wavs, star.temp)
    else:
        # Use stellar spectrum
        log_flux = np.interp(logwavs, star.log_wavs, star.log_flux_lam)
        log_flux[np.isinf(log_flux)] = -50.0
        flux = 10.0**log_flux        
        
    if Q_interpolator is not None:              # Use interpolator if available
        return flux * Q_interpolator(wavs)
    elif Q_type is not None:                    # Otherwise calculate directly
        return flux * calculate_scatt_efficiency_coeffs(wavs, diam, matrl=matrl)[Q_type]
    else:
        return flux

def integrate_log_spectrum(flux, logwavs):
    """Integrate a spectrum over log wavelength space.
    
    Args:
        logwavs (array-like): Log10 of wavelengths
        flux (array-like): Spectral flux density at each wavelength
        
    Returns:
        float: Integrated flux
        
    Note:
        When integrating F_λ over d(log λ), need to multiply by λ*ln(10)
        because d(log λ)/dλ = 1/(λ*ln(10)).    
    """
    wavs = 10.0**logwavs
    return np.trapezoid(flux * wavs * np.log(10.0), x=logwavs)

def calculate_spectral_flux_density_bb(wavs, temp):
    """
    Calculate the spectral flux density, Fλ=π*Bλ, of a black body at a given temperature.

    This function computes the spectral radiance (B_λ) and converts it to 
    spectral flux density (F_λ) by multiplying with π, which effectively integrates
    the radiance over a hemisphere (assuming isotropic and Lambertian emission).

    Args:
        wavs (float or array_like): Wavelength(s) in µm at which the spectral flux density is calculated.
        temp (float): Temperature of the black body in Kelvin.

    Returns:
        float or ndarray: The spectral flux density in (W/m^2/µm).
    """
    return calculate_spectral_radiance_bb(wavs, temp) * np.pi

def calc_spectral_flux_density_bb(wavs, temp):
    """
    Wrapper needed for old optmod files. (Runs v40s)
    """
    return calculate_spectral_flux_density_bb(wavs, temp)

def calculate_spectral_radiance_bb(wavs, temp, domain='wavelength'):
    """
    Calculate the spectral radiance (Bλ or B_nu) in W/m^2/µm/sr of a black body 
    for a given temperature and one or more wavelengths using Planck's law.

    Planck's law describes the spectral density of electromagnetic radiation 
    emitted by a black body in thermal equilibrium at a certain temperature. 
    This function returns the spectral radiance either in the 
    wavelength (W/m^2/µm/sr) or frequency (Jy/sr) domain.
    
    Args:
        wavs (float or np.ndarray): Wavelength(s) in µm or frequency in Hz.
        temp (float or np.ndarray): Temperature(s) in K. Can have any shape.
        domain (str, optional): Domain of output ('wavelength' or 'frequency'). Defaults to 'wavelength'.
    
    Returns:
        float or np.ndarray: Spectral radiance in W/m²/µm/sr or Jy/sr.
    """
    temp = np.asarray(temp)[..., None]  # Add wavelength dimension to any shape input
    
    if domain[0:3] == 'wav':
        # Calculate spectral radiance in W/m²/µm/sr
        k1 = 1.1910439e8        # = 2hc^2 in (W/m²)µm^4
        fact1 = k1 / wavs**5
    elif domain[0:4] == 'freq':
        # Calculate spectral radiance in Jy/sr
        k1 = 3.9728949e+19      # = 2hc in Jy*µm^3
        fact1 = k1 / wavs**3
    else:
        raise ValueError("Invalid domain. Choose 'wavelength' or 'frequency'.")
    
    k2 = 14387.69               # = hc/k in µm*K
    fact2 = k2 / (wavs * temp)
    fact2 = np.clip(fact2, a_min=None, a_max=88) # Clip to avoid overflow in exp(fact2)

    return fact1 / (np.exp(fact2) - 1)

def calculate_blackbody_temp(star, dist):
    """
    Calculate temperature of a black body at given distance from star.
    Uses approximation T = 278.3 * sqrt(sqrt(L_star)) / sqrt(r)
    where L_star is in solar luminosities and r in AU.
    
    Args:
        star (Star): Star object containing luminosity in solar units
        dist (float): Distance from star in AU
    
    Returns:
        float: Blackbody temperature in Kelvin
    """
    return 278.3 * np.sqrt(np.sqrt(star.lum_suns)) / np.sqrt(dist)

def calculate_scatt_efficiency_coeffs(wavs, diam, matrl):
    """
    Calculates the scattering coefficients for particles of different sizes
    at given wavelengths and for a given dielectric function.
    Uses Mie theory, Rayleigh-Gans theory, or Geometric Optics, 
    depending on the conditions.

    Args:
        wavs (np.ndarray): Array of wavelengths (in µm) at which to calculate the scattering.
        diam (float): Diameter of the particle.
        matrl (Material): Object containing the dielectric properties and density of the grains.

    Returns:
        dict: Dictionary containing all coefficients with keys:
            'abs': absorption efficiency
            'pr': radiation pressure efficiency
            'sca': scattering efficiency
    """
    if not np.all(np.diff(wavs) > 0):
            raise ValueError("Wavelengths are not in ascending order. This must not happen.")

    # Interpolate the dielectric constants at the desired "wavs" values
    # Create masks for different wavelength ranges
    in_range = (wavs >= matrl.wavs[0]) & (wavs <= matrl.wavs[-1])
    too_long = wavs > matrl.wavs[-1]
    too_short = wavs < matrl.wavs[0]

    dielec = np.zeros_like(wavs, dtype=complex)
    dielec[in_range] = matrl.interpolate_eps(wavs_to=wavs[in_range])
    # wav beyond largest matrl.wavs --> eps.real stays constant & eps.imag falls off as 1/wav)
    dielec[too_long] = matrl.eps[-1].real + 1j * (matrl.eps[-1].imag * matrl.wavs[-1] / wavs[too_long])
    # wav below smallest matrl.wavs --> assume eps does not change much from closest tabulation
    dielec[too_short] = matrl.eps[0]

    # Get all Qs for this particle
    qabs = np.ones_like(wavs)
    qpr = np.ones_like(wavs)
    qsca = np.ones_like(wavs)
    x = np.pi * diam * matrl.refmed / wavs
    m = np.sqrt(dielec)
    mx = np.abs(m) * x 
    mx1 = np.abs(m - 1) * x
    
    # Define regions for different scattering regimes
    mie_region = mx <= 1000.0
    rayleigh_gans_region = (mx > 1000.0) & (mx1 <= 0.001)
    geometric_region = (mx > 1000.0) & (mx1 > 0.001) & (wavs < matrl.wavs[-1])

    # Apply Mie theory, where wavelengths aren't small compared to particle size
    if np.any(mie_region):
        k = np.where(mie_region)[0]
        for i in k:
            qabs[i], qpr[i], qsca[i] = calculate_coeffs_mie_theory(x[i], m[i])
    
    # Apply Rayleigh-Gans theory, where wavelengths are small compared to size and m~1
    if np.any(rayleigh_gans_region):
        k = np.where(rayleigh_gans_region)[0]
        qabs[k], qpr[k], qsca[k] = calculate_coeffs_rayleigh_gans(x[k], m[k])
    
    # Apply Geometric Optics, where wavelengths are small compared to size and not m~1
    if np.any(geometric_region):
        k = np.where(geometric_region)[0]
        qabs[k], qpr[k], qsca[k] = calculate_coeffs_geom_optics(np.real(m[k]))

    # Store all coefficients in a dictionary
    return {
        'abs': qabs,
        'pr': qpr,
        'sca': qsca
    }

@jit(nopython=True)
def calculate_coeffs_mie_theory(x, m, n_ang=1):
    """
    Calculate optical coefficients using Mie theory (following Bohren & Huffman, 1998).
    Returns all three coefficients (Qabs, Qpr, Qsca).
    
    Args:
        x (float): Size parameter, defined as π * diameter * refractive_index_medium / wavelength.
        m (complex): Relative refractive index of the material, defined as sqrt(epsilon).
        n_ang (int, optional): Number of angles at which to calculate the scattering intensities. Defaults to 1.
    Returns:
        tuple: The requested optical coefficients (Qabs, Qpr, Qsca).
    """
    
    # Number of terms needed in expansion for Mie coeffs (see BH p477)
    n_terms = round(2 + x + 4 * x**0.33333333)
    
    # The logarithmic derivatives of y=mx are calculated by downward recurence, 
    # using nmx = (n_terms>abs(mx))+15 terms (see BH p478)
    y = x * m
    nmx = max(n_terms, round(abs(y))) + 15
    d = np.zeros(nmx, dtype=np.complex128)
    
    for i in range(nmx - 2, 0, -1):
        n = i + 1
        temp = n / y
        d[i - 1] = temp - 1.0 / (d[i] + temp)
    
    # Some functions used in the Mie calculation, including the logarithmic derivatives, 
    # dmnx = D_n/m + n/x, and mdnx = m*D_n + n/x (see BH p127)
    ns = np.arange(1, n_terms + 1)
    nx = ns / x
    dmnx = d[:n_terms] / m + nx
    mdnx = m * d[:n_terms] + nx
    # As well as other functions including just n
    ns2 = ns * ns
    fn1 = (2 * ns + 1) / (ns2 + ns)
    fn2 = (ns2 - 1) / ns
    nsdouble = (2 * ns - 1)
    
    # Consider the angles at which to calculate the scattering intensities
    if n_ang == 1:
        theta = np.array([0.0])
        amu = np.array([1.0])
        n_ang_tot = 2
    else:
        theta = np.linspace(0, 0.5 * np.pi, n_ang)
        amu = np.cos(theta)
        n_ang_tot = 2 * n_ang - 1
    
    # Note that here pi = pi_{n=1}, pi1 = pi_{n=0}
    pi1 = np.zeros(n_ang)
    pi = pi1 + 1
    pi0 = np.zeros_like(pi1)
    s1 = np.zeros(n_ang_tot, dtype=np.complex128)
    s2 = s1.copy()
    
    # The Riccati-Bessel functions are calculated by upward recurrence starting with: 
    # psi_{-1} = cos(x), psi_0 = sin(x); chi_{-1} = -sin(x), chi_0 = cos(x); xi = psi-i*chi (see BH p478).
    # Note that at each "n": psi, psi1, and psi0 are the values of "psi" 
    # for n,n-1,n-2 (and for chi, xi, pi, an, and bn)
    psi0 = np.cos(x)
    psi1 = np.sin(x)
    chi0 = -np.sin(x)
    chi1 = np.cos(x)
    # xi0 = psi0 - 1j * chi0    # not needed.
    xi1 = psi1 - 1j * chi1
    
    # Set some other initial values
    qsca = 0.0
    asymf = 0.0
    p = -1.0

    an1 = np.complex128(0 + 0j)
    bn1 = np.complex128(0 + 0j)
    
    # Series calculation
    for i, n in enumerate(ns):
        # Calculate the new Ricatti-Bessel functions from the previous two (see BH p. 478)
        psi = nsdouble[i] * psi1 / x - psi0
        chi = nsdouble[i] * chi1 / x - chi0
        xi = psi - 1j * chi

        # Calculate An and Bn from the relations (see BH p. 127):
        an = (dmnx[i] * psi - psi1) / (dmnx[i] * xi - xi1)
        bn = (mdnx[i] * psi - psi1) / (mdnx[i] * xi - xi1)
        
        # Augment the sums for Qsca and ASMYF = g = GSCA = <cos(theta)> (see VH p. 128):
        qsca += (2 * n + 1) * (abs(an)**2 + abs(bn)**2)
        asymf += fn1[i] * (an * np.conj(bn)).real

        # Calculate "pi" and "tau" from previous two using relations (BH)
        if n > 1:
            asymf += fn2[i] * ((an * np.conj(an1)).real + (bn * np.conj(bn1)).real)
        
        if n > 1:
            pi =  (nsdouble[i] * amu * pi1 - n * pi0) / (n - 1)
        tau = n * amu * pi - (n + 1) * pi1
        
        # Calculate the scattering intensity pattern at the desired angles <=90 (see VH p. 125):
        s1[:n_ang] += fn1[i] * (an * pi + bn * tau)
        s2[:n_ang] += fn1[i] * (bn * pi + an * tau)

        # Calculate the scattering intensity pattern at angles >90
        p = -p
        s1[n_ang:] += fn1[i] * p * (an * pi[::-1] - bn * tau[::-1])
        s2[n_ang:] += fn1[i] * p * (bn * pi[::-1] - an * tau[::-1])

        # Store previous values of An,Bn,psi,chi,xi,pi for use in next step
        an1 = an
        bn1 = bn
        psi0, psi1 = psi1, psi
        chi0, chi1 = chi1, chi
        xi1 = xi
        pi0, pi1 = pi1, pi

    # Calculate the optical coefficients (see VH page 128)
    x2 = x * x
    asymf = 2 * asymf / qsca                # Asymmetry factor
    qsca = 2 * qsca / x2
    qext = 4 * s1[0].real / x2              # Extinction efficiency
    qabs = qext - qsca
    qpr = qext - qsca * asymf
    # qback = abs(s1[-1])**2 / (x2 * np.pi)   # Backscattering efficiency

    return qabs, qpr, qsca

def calculate_coeffs_rayleigh_gans(x, m):
    """Calculate coefficients (Qabs, Qpr, Qsca) using Rayleigh-Gans theory (BH158 and LD93).

    Args:
        x (float): Size parameter of the particle.
        m (complex): Complex refractive index of the particle.
    Returns:
        tuple: The requested optical coefficients (Qabs, Qpr, Qsca).
    """

    x2 = x * x
    qabs = 8.0 * np.imag(m) * x / 3.0
    qsca = 32.0 * np.abs(m - 1)**2 * x2 * x2 / (27.0 + 16.0 * x2)
    qpr = qabs + qsca / (1 + 0.3 * x2)
    return qabs, qpr, qsca

def calculate_coeffs_geom_optics(refreal, n_step=1000):
    """Calculate coefficients (Qabs, Qpr, Qsca) using geometric optics.
    
    Args:
        refreal (np.ndarray): Real component(s) of the refractive index.
        n_step (int, optional): Number of integration steps. Defaults to 1000.

    Returns:
        tuple: The requested optical coefficients (Qabs, Qpr, Qsca).
    """

    x = np.linspace(0, 1, n_step)
    sint = np.sqrt(1.0 - x)
    cos2t = 2 * x - 1

    n_wav = len(refreal)
    qabs = np.zeros(n_wav)
    qpr = np.zeros(n_wav)
    
    for i in range(n_wav):
        sintp = np.sqrt(np.maximum(0.0, 1.0 - x / refreal[i]**2))
        tempir1 = sint + refreal[i] * sintp
        dw1 = np.ones(n_step)
        k1 = tempir1 != 0
        dw1[k1] = ((sint[k1] - refreal[i] * sintp[k1]) / tempir1[k1])**2

        tempir2 = refreal[i] * sint + sintp
        dw2 = np.ones(n_step)
        k2 = tempir2 != 0
        dw2[k2] = ((refreal[i] * sint[k2] - sintp[k2]) / tempir2[k2])**2
        
        # Calculate qabs using dw1 and dw2 directly
        # For the integration, add a fudge factor (0.6) which is then subtracted
        int1_abs = dw1
        int2_abs = dw2
        qabs[i] = 1.0 - 0.5 * (np.trapezoid(int1_abs + int2_abs + 0.6, x=x) - 0.6)

        # Calculate qpr using dw1 and dw2 multiplied by cos2t
        int1_pr = dw1 * cos2t
        int2_pr = dw2 * cos2t
        qpr[i] = 1.0 - 0.5 * (np.trapezoid(int1_pr + int2_pr + 0.6, x=x) - 0.6)

    # In the geometric optics regime, Q_ext = 2, due to diffraction, 
    # see "extinction paradox" in Bohren & Huffman, 1983.
    qext = 2
    qsca = qext - qabs

    return qabs, qpr, qsca
