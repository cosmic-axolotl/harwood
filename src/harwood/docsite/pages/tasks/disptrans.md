Puts spectra's dispersion in other units -- nanometers, microns, hertz,
electron volts, inverse centimeters, a velocity from a line -- and takes air
wavelengths to vacuum or back.

```console
$ harwood disptrans in o "km/s 4100 angstroms"
o: Dispersion transformed to km/s.
$ harwood disptrans in o angstroms --air air2vac
o: Dispersion transformed to angstroms in vacuum with
  t = 15. C, p = 760. mmHg, f = 4. mmHg.
$ harwood disptrans in o "inverse centimeters" --linearize
```

The units are IRAF's: `angstroms`, `nanometers`, `microns`, `millimeters`,
`centimeters`, `meters`, `hertz` to `gigahertz`, `m/s` and `km/s` with a
reference wavelength and its units (`km/s 6563 angstroms`, relativistic),
`ev`, `kev`, `mev` and `z`, each perhaps `log` or `inverse`, shortened as far
as it stays one word, and the abbreviations of IRAF's `units.dat` (`nm`, `Hz`,
`wn` for inverse centimeters). Units IRAF does not know stop the task, as
IRAF's: `inverse cm` is not one, its `cm` being an abbreviation only of the
whole.

Each pixel's coordinate is converted, and with `--air` scaled by Edlén's
index of refraction for `--t`, `--p` and `--f`. A straight line is fitted to
the new coordinates, or a cubic spline of as many pieces as it takes to be
within `--error` pixels of every one, and the dispersion is that function, as
IRAF's world coordinates hold it. `--linearize` resamples the fluxes onto an
even grid in the new units, not conserving flux.

The spectrum's header says the new units (`APUNITS`), but the other tasks
read the wavelength column in Angstroms: disptrans is for looking at a
spectrum, or handing it on, in other units. Ten sessions IRAF 2.18 ran say the
same, give each pixel IRAF's coordinate to a part in 10¹⁴ (10¹² in air and
vacuum, through the index of refraction), and its flux to the bit, resampled
within a unit of the last place -- IRAF's scales kept in real as IRAF keeps
them, so a nanometer is 4010 Å times 0.1000000015.
