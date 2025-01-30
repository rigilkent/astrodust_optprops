import numpy as np
import optprops as opt
from pathlib import Path

# --------------------------- Setup test parameters -----------------------------
# Create star object with Fomalhaut parameters
star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92, 
    spectrum_file=Path(__file__).parent / 'fomalhaut_spectrum.txt')

# Create material object example volume fractions of silicate, ice, and voids (see optprops.Material)
matrl = opt.Material(qsil=.5, qice=.5, mpor=.5)

dists = np.array([1, 10, 100])
wavs  = np.array([.2, 2, 20, 200])
diams = np.array([1, 10, 100, 1000])

# --------------------------- Calculate optical properties -----------------------
temps = opt.calc_temperatures(diams, dists, matrl=matrl, star=star) # n_diams x n_dists

Qabs = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl) for d in diams]) # n_diams x n_wavs
Qpr = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl, qout=2) for d in diams])
Qsca = np.array([opt.calc_scatt_efficiency_coeffs(wavs, d, matrl=matrl, qout=3) for d in diams])

betas = opt.calc_beta_factors(diams, matrl=matrl, star=star)
diam_blow = opt.calc_blowout_diameters(matrl=matrl, star=star, beta_blow=0.5)
                                            
# --------------------------- Print the results ----------------------------------
print("Dust temperatures:\n", temps)

print("betas:", betas)

print("diam_blow:", diam_blow)
