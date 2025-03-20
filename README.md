# astrodust_optprops
A tool to compute the optical properties of circumstellar dust particles 
as well as their resulting equilibrium temperatures and thermal emission.


```optprops``` first finds the optical constants of a composite material made up of 
silicate, refractory carbonaceous material, water ice, and vacuum
using effective medium theory.
It applies Mie theory, Rayleigh-Gans theory, or geometric optics
in the respective wavelength regimes to determine the particles' 
optical coefficients.
These can then be used to compute the distance- and size-dependent equilibrium temperatures
of particles around a given star, as well as their β-factors (the ratio of radiation pressure force to gravity), or their blowout sizes.

Optical constants of the composite may be computed using the Maxwell-Garnett rule,
treating dust as aggregates of core-mantle grains, following
[Li & Greenberg (1997)](https://ui.adsabs.harvard.edu/abs/1997A%26A...323..566L/abstract),
or using the Bruggeman rule.
```optprops``` can also suppress Mie resonance ripples in the optical coefficients by averaging over nearby particle sizes.

## Usage

See the `examples` directory for more detailed usage examples.

The package provides several key classes for modeling the optical properties:

```python
import astrodust_optprops as opt

# Create a star object
star = opt.Star(name='Fomalhaut', lum_suns=16.6, mass_suns=1.92, temp=8500)

# Create a material object
matrl = opt.Material(...)

# Define wavelengths, particles diameters, and their distances to the star
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
git clone https://github.com/rigilkent/astrodust_optprops.git
cd astrodust_optprops
pip install .
```