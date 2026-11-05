Moves the wavelength zero point of spectra: adds a number to their dispersion,
in place, as IRAF's specshift does.

```console
$ harwood specshift 'night1/*.w.fits' --shift 0.42 --verbose
  Aperture 1:  4010. --> 4010.42
  Aperture 2:  4020. --> 4020.42
$ harwood specshift star.fits --shift -1.5 --apertures 2
```

The zero point is the weakest part of a dispersion solution: the arc taken an
hour before, flexure since, and every wavelength is off by the same amount. A
sky line measured where it should not be says by how much, and `--shift` puts
it back. `--apertures` shifts some apertures only.

The shift is in the dispersion's own coordinates, as IRAF's is: in a spectrum
[dopcor](task:dopcor) has corrected, the wavelengths move by shift/(1+z), and
each aperture's z stays what it was.

As in IRAF, `--verbose` says each aperture's start before and after and never
the line `shift = ...` its code means to print first: that line waits for a
count of apertures the loop never makes. The starts are doubles printed with
`%g` in full, and their last figures are those of IRAF's world-coordinate
arithmetic; four sessions IRAF 2.18 ran, one after a dopcor, are held to them
to a part in 10¹², and the wavelengths to IRAF's world coordinates the same.
