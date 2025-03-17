from pathlib import Path
import numpy as np
import optprops as opt
import time

script_dir = Path(__file__).parent

# --------------------------- Setup test parameters -----------------------------
# Create star object with Fomalhaut parameters
star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92,
    spectrum_file=script_dir / 'fomalhaut_spectrum.txt')

# Create material object example volume fractions of silicate, ice, and voids
matrl = opt.Material(qsil=.4, qice=0.0, mpor=.0, emt='maxwell-garnett')

wavs = np.logspace(0, 4, 300)
diams = np.logspace(.5, 3.5, 25)
dists = np.logspace(1, 3, 20)

# --------------------------- Calculate optical properties -----------------------
# Create Particles instance and calculate all properties
prtl = opt.Particles(diams=diams, wavs=wavs, dists=dists, 
                     matrl=matrl, suppress_mie_resonance=True)
prtl.calculate_all(star)

# --------------------------- Save complete state --------------------------------
model = opt.OpticalModel(star=star, prtl=prtl)
model.save(script_dir / 'fomalhaut_results.pkl')



# ----------------------------- Later, load and use: -------------------------------
# loaded_model = opt.OpticalModel.load(script_dir / 'fomalhaut_results.pkl')
# prtl = loaded_model.prtl
# star = loaded_model.star

# Make a few plots:
ax0 = star.plot_spectrum()    
ax0.figure.savefig(script_dir / 'star_spectrum.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
ax1 = prtl.plot_Qabs()
ax1.figure.savefig(script_dir / 'Qabs.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
ax2 = prtl.plot_beta()
ax2.figure.savefig(script_dir / 'beta.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
ax3 = prtl.plot_temp()
ax3.figure.savefig(script_dir / 'temp.png', dpi=300, bbox_inches='tight', pad_inches=0.01)

