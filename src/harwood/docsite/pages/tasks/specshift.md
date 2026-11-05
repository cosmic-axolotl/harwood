Moves the wavelength zero point of spectra: adds a number to their dispersion,
in place, as IRAF's specshift does.

```console
$ harwood specshift galaxy.fits 0.42 --verbose
  Aperture 1:  4009.999999999999 --> 4010.420000000001
  Aperture 2:  4020. --> 4020.419999999999
  Aperture 3:  4029.999999999999 --> 4030.42
$ harwood specshift star.fits --apertures 2 -- -1.5
```

The zero point is the weakest part of a dispersion solution: the arc taken an
hour before, flexure since, and every wavelength is off by the same amount. A
sky line measured where it should not be says by how much, and the shift (a
negative one after `--`) puts it back. `--apertures` shifts some apertures only.

The shift is in the dispersion's own coordinates, as IRAF's is: in a spectrum
[dopcor](task:dopcor) has corrected, the wavelengths move by shift/(1+z), and
each aperture's z stays what it was.

As in IRAF, `--verbose` says each aperture's start before and after and never
the line `shift = ...` its code means to print first: that line waits for a
count of apertures the loop never makes. The starts are doubles printed with
IRAF's `%g`, whose arithmetic prints 4010 as 4009.999999999999; harwood prints
them the same, and four sessions IRAF 2.18 ran, one after a dopcor, say the
same to the character, the wavelengths held to IRAF's world coordinates.
