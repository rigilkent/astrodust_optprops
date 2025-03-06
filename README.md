# optprops
A tool to compute the optical properties of circumstellar dust particles 
as well as their resulting equilibrium temperatures and thermal emission.


Optprops first finds the optical constants of a composite material made up of 
silicate, refractory carbonaceous material, water ice, and vacuum
using effective medium theory.
It then applies Mie theory, Rayleigh-Gans theory, or geometric optics
in the respective wavelength regimes to determine the particles' 
absorption coefficients and temperatures.

Optical constants of the composite may be computed using the Maxwell-Garnett rule,
treating dust as aggregates of core-mantle grains, following
[Li & Greenberg (1997)](https://ui.adsabs.harvard.edu/abs/1997A%26A...323..566L/abstract),
or using the Bruggeman rule.
This package is largely a reimplementation of the approach used by
[Wyatt & Dent (2002)](https://doi.org/10.1046/j.1365-8711.2002.05533.x).


## Usage

See the `examples` directory for more detailed usage examples.

The package provides several key classes for modeling the optical properties:

```python
import optprops as opt

# Create a star object
star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92, temp=8500)

# Create a material object
matrl = opt.Material(...)

# Define wavelengths, particles diameters, and distances to the star
wavs = [...]
diams = [...]
dists = [...]

# Create a Particles instance and calculate all properties
prtl = opt.Particles(diams=diams, wavs=wavs, matrl=matrl, dists=dists)
prtl.calculate_all(star)

print('Particle Qabs:\n', prtl.Qabs)
print('Particle temperatures:\n', prtl.temps)
```

## Installation

```bash
git clone https://github.com/rigilkent/optprops.git
cd optprops
pip install .
```