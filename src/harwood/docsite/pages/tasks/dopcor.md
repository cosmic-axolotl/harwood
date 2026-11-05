Takes a spectrum to the rest frame of a redshift or a velocity: its wavelengths
over 1+z, and, if asked, its fluxes by a power of 1+z.

```console
$ harwood dopcor ngc4486.fits ngc4486.rest.fits --redshift 0.00428 --verbose
ngc4486.rest: Doppler correction - apertures=all, redshift= 0.00428, flux factor=0.
$ harwood dopcor star.fits --redshift -VHELIO --isvelocity
```

`--redshift` is a number, or the name of a header keyword holding one;
`--isvelocity` makes it a velocity in km/s, turned into a redshift
relativistically, sqrt((1+v/c)/(1-v/c)) - 1. A minus before a keyword undoes
it -- a velocity by its sign, a redshift to 1/(1+z) - 1 -- so `-VHELIO` takes
out a heliocentric correction the header records. A minus before a number only
makes it negative.

**The correction replaces the last one.** Each aperture keeps the z it has been
corrected by (`APZ` in its header), as IRAF keeps it in the spectrum's world
coordinates, and a second dopcor puts its own z in that one's place: the
wavelengths are always the dispersion's over 1+z. `--add` puts the two
together instead, (1+z)(1+zold) - 1, as for a heliocentric correction and then
a galaxy's own velocity.

`--flux` multiplies every band of the spectrum -- flux, error, sky, raw -- by
(1+z)^`--factor`, 3 by default (a flux per unit wavelength); `--no-dispersion`
leaves the wavelengths alone. `--apertures 1,3` corrects some apertures and
copies the rest as they are. The header gets a `DOPCOR01` card (`DOPCOR02`
after it, and on) with the z, the factor when the fluxes were corrected, and
the apertures, as IRAF writes it.

Sixteen chains of runs IRAF 2.18 made -- numbers, velocities, keywords with and
without the minus, the flux by two factors, some apertures, a correction added
and one replaced, two cards, an impossible redshift and velocity, a keyword
the header lacks -- say, write and correct the same: the wavelengths to a part
in 10⁹ of IRAF's world coordinates, the fluxes to the bit. Where IRAF writes
the spectrum in place when no output is given, harwood writes it to the run's
directory; give the input's own name as the output to correct it in place.
IRAF's one-dimensional images, with no apertures, are harwood's spectra of one
aperture, and their verbose line says `apertures=` as a multispec's does.
